#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal_buffer_utils.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace py = pybind11;

namespace {

enum {
    INTERPOLATION_NEAREST = 0,
    INTERPOLATION_LINEAR = 1,
    INTERPOLATION_AREA = 2,
};

struct FitCropToCanvasParams {
    int input_width;
    int input_height;
    int channels;
    int source_x;
    int source_y;
    int source_width;
    int source_height;
    int canvas_width;
    int canvas_height;
    int draw_width;
    int draw_height;
    int offset_x;
    int offset_y;
    int interpolation;
};

struct TransformToCanvasParams {
    int input_width;
    int input_height;
    int channels;
    int canvas_width;
    int canvas_height;
    int border_mode;
    float inverse_matrix[9];
};

struct TransformCropToCanvasParams {
    int input_width;
    int input_height;
    int channels;
    int transform_width;
    int transform_height;
    int canvas_width;
    int canvas_height;
    int source_x;
    int source_y;
    int source_width;
    int source_height;
    int draw_width;
    int draw_height;
    int offset_x;
    int offset_y;
    int interpolation;
    int border_mode;
    int lens_enabled;
    float lens_k1;
    int mesh_enabled;
    int mesh_grid_width;
    int mesh_grid_height;
    int use_mesh_lut;
    int use_mesh_dense;
    float inverse_matrix[9];
};

// area/nearest の mesh bicubic を LUT 化するためのエントリ（整数キャンバス座標ごと）。
// MSL 側の MeshCubicLut (int4 + float4) と 32 バイトレイアウトを一致させる。
struct MeshCubicLutEntry {
    int32_t idx[4];
    float w[4];
};

constexpr const char* kMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct FitCropToCanvasParams {
    int input_width;
    int input_height;
    int channels;
    int source_x;
    int source_y;
    int source_width;
    int source_height;
    int canvas_width;
    int canvas_height;
    int draw_width;
    int draw_height;
    int offset_x;
    int offset_y;
    int interpolation;
};

struct TransformToCanvasParams {
    int input_width;
    int input_height;
    int channels;
    int canvas_width;
    int canvas_height;
    int border_mode;
    float inverse_matrix[9];
};

struct TransformCropToCanvasParams {
    int input_width;
    int input_height;
    int channels;
    int transform_width;
    int transform_height;
    int canvas_width;
    int canvas_height;
    int source_x;
    int source_y;
    int source_width;
    int source_height;
    int draw_width;
    int draw_height;
    int offset_x;
    int offset_y;
    int interpolation;
    int border_mode;
    int lens_enabled;
    float lens_k1;
    int mesh_enabled;
    int mesh_grid_width;
    int mesh_grid_height;
    int use_mesh_lut;
    int use_mesh_dense;
    float inverse_matrix[9];
};

struct MeshCubicLut {
    int4 idx;
    float4 w;
};

static inline float read_channel(
    const device float* input,
    constant FitCropToCanvasParams& p,
    int x,
    int y,
    int ch
) {
    x = clamp(x, 0, p.input_width - 1);
    y = clamp(y, 0, p.input_height - 1);
    return input[(y * p.input_width + x) * p.channels + ch];
}

static inline float sample_nearest(
    const device float* input,
    constant FitCropToCanvasParams& p,
    int dx,
    int dy,
    int ch
) {
    int sx = p.source_x + min(int(floor(float(dx) * float(p.source_width) / float(p.draw_width))), p.source_width - 1);
    int sy = p.source_y + min(int(floor(float(dy) * float(p.source_height) / float(p.draw_height))), p.source_height - 1);
    return read_channel(input, p, sx, sy, ch);
}

static inline float sample_linear(
    const device float* input,
    constant FitCropToCanvasParams& p,
    int dx,
    int dy,
    int ch
) {
    float sx = (float(dx) + 0.5f) * float(p.source_width) / float(p.draw_width) - 0.5f;
    float sy = (float(dy) + 0.5f) * float(p.source_height) / float(p.draw_height) - 0.5f;
    sx += float(p.source_x);
    sy += float(p.source_y);

    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float ax = sx - float(x0);
    float ay = sy - float(y0);

    float v00 = read_channel(input, p, x0, y0, ch);
    float v10 = read_channel(input, p, x1, y0, ch);
    float v01 = read_channel(input, p, x0, y1, ch);
    float v11 = read_channel(input, p, x1, y1, ch);
    float top = mix(v00, v10, ax);
    float bottom = mix(v01, v11, ax);
    return mix(top, bottom, ay);
}

static inline float sample_area(
    const device float* input,
    constant FitCropToCanvasParams& p,
    int dx,
    int dy,
    int ch
) {
    float x0 = float(p.source_x) + float(dx) * float(p.source_width) / float(p.draw_width);
    float x1 = float(p.source_x) + float(dx + 1) * float(p.source_width) / float(p.draw_width);
    float y0 = float(p.source_y) + float(dy) * float(p.source_height) / float(p.draw_height);
    float y1 = float(p.source_y) + float(dy + 1) * float(p.source_height) / float(p.draw_height);

    if (x1 <= x0 || y1 <= y0) {
        return 0.0f;
    }

    int ix0 = int(floor(x0));
    int ix1 = int(ceil(x1));
    int iy0 = int(floor(y0));
    int iy1 = int(ceil(y1));
    float accum = 0.0f;
    float weight_sum = 0.0f;

    for (int yy = iy0; yy < iy1; ++yy) {
        float wy = max(0.0f, min(y1, float(yy + 1)) - max(y0, float(yy)));
        if (wy <= 0.0f) {
            continue;
        }
        for (int xx = ix0; xx < ix1; ++xx) {
            float wx = max(0.0f, min(x1, float(xx + 1)) - max(x0, float(xx)));
            float weight = wx * wy;
            if (weight <= 0.0f) {
                continue;
            }
            accum += read_channel(input, p, xx, yy, ch) * weight;
            weight_sum += weight;
        }
    }

    return weight_sum > 0.0f ? accum / weight_sum : 0.0f;
}

static inline float3 read_rgb_clamped(
    const device float* input,
    constant FitCropToCanvasParams& p,
    int x,
    int y
) {
    x = clamp(x, 0, p.input_width - 1);
    y = clamp(y, 0, p.input_height - 1);
    int base = (y * p.input_width + x) * 3;
    return float3(input[base + 0], input[base + 1], input[base + 2]);
}

// RGB(3ch) 用の float3 特化版。1ch はスカラー経路のまま。
static inline float3 sample_nearest_rgb(
    const device float* input,
    constant FitCropToCanvasParams& p,
    int dx,
    int dy
) {
    int sx = p.source_x + min(int(floor(float(dx) * float(p.source_width) / float(p.draw_width))), p.source_width - 1);
    int sy = p.source_y + min(int(floor(float(dy) * float(p.source_height) / float(p.draw_height))), p.source_height - 1);
    return read_rgb_clamped(input, p, sx, sy);
}

static inline float3 sample_linear_rgb(
    const device float* input,
    constant FitCropToCanvasParams& p,
    int dx,
    int dy
) {
    float sx = (float(dx) + 0.5f) * float(p.source_width) / float(p.draw_width) - 0.5f;
    float sy = (float(dy) + 0.5f) * float(p.source_height) / float(p.draw_height) - 0.5f;
    sx += float(p.source_x);
    sy += float(p.source_y);

    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float ax = sx - float(x0);
    float ay = sy - float(y0);

    float3 v00 = read_rgb_clamped(input, p, x0, y0);
    float3 v10 = read_rgb_clamped(input, p, x1, y0);
    float3 v01 = read_rgb_clamped(input, p, x0, y1);
    float3 v11 = read_rgb_clamped(input, p, x1, y1);
    float3 top = mix(v00, v10, ax);
    float3 bottom = mix(v01, v11, ax);
    return mix(top, bottom, ay);
}

static inline float3 sample_area_rgb(
    const device float* input,
    constant FitCropToCanvasParams& p,
    int dx,
    int dy
) {
    float x0 = float(p.source_x) + float(dx) * float(p.source_width) / float(p.draw_width);
    float x1 = float(p.source_x) + float(dx + 1) * float(p.source_width) / float(p.draw_width);
    float y0 = float(p.source_y) + float(dy) * float(p.source_height) / float(p.draw_height);
    float y1 = float(p.source_y) + float(dy + 1) * float(p.source_height) / float(p.draw_height);

    if (x1 <= x0 || y1 <= y0) {
        return float3(0.0f);
    }

    int ix0 = int(floor(x0));
    int ix1 = int(ceil(x1));
    int iy0 = int(floor(y0));
    int iy1 = int(ceil(y1));
    float3 accum = float3(0.0f);
    float weight_sum = 0.0f;

    for (int yy = iy0; yy < iy1; ++yy) {
        float wy = max(0.0f, min(y1, float(yy + 1)) - max(y0, float(yy)));
        if (wy <= 0.0f) {
            continue;
        }
        for (int xx = ix0; xx < ix1; ++xx) {
            float wx = max(0.0f, min(x1, float(xx + 1)) - max(x0, float(xx)));
            float weight = wx * wy;
            if (weight <= 0.0f) {
                continue;
            }
            accum += read_rgb_clamped(input, p, xx, yy) * weight;
            weight_sum += weight;
        }
    }

    return weight_sum > 0.0f ? accum / weight_sum : float3(0.0f);
}

kernel void fit_crop_to_canvas_kernel(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant FitCropToCanvasParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.canvas_width || y >= p.canvas_height) {
        return;
    }

    int out_base = (y * p.canvas_width + x) * p.channels;
    int dx = x - p.offset_x;
    int dy = y - p.offset_y;
    bool inside = dx >= 0 && dy >= 0 && dx < p.draw_width && dy < p.draw_height;

    if (p.channels == 3) {
        float3 value = float3(0.0f);
        if (inside) {
            if (p.interpolation == 0) {
                value = sample_nearest_rgb(input, p, dx, dy);
            } else if (p.interpolation == 1) {
                value = sample_linear_rgb(input, p, dx, dy);
            } else {
                value = sample_area_rgb(input, p, dx, dy);
            }
        }
        output[out_base + 0] = value.x;
        output[out_base + 1] = value.y;
        output[out_base + 2] = value.z;
        return;
    }

    for (int ch = 0; ch < p.channels; ++ch) {
        float value = 0.0f;
        if (inside) {
            if (p.interpolation == 0) {
                value = sample_nearest(input, p, dx, dy, ch);
            } else if (p.interpolation == 1) {
                value = sample_linear(input, p, dx, dy, ch);
            } else {
                value = sample_area(input, p, dx, dy, ch);
            }
        }
        output[out_base + ch] = value;
    }
}

// BORDER_REFLECT（端画素複製あり）の定数時間版。周期 2*length に折り畳み、
// 前半はそのまま・後半は鏡像。旧 while ループと同一の写像（遠距離座標でも O(1)）。
static inline int reflect_coord(int p, int length) {
    if (length <= 1) {
        return 0;
    }
    int period = 2 * length;
    int m = ((p % period) + period) % period;
    return m < length ? m : period - 1 - m;
}

// BORDER_REFLECT_101 相当（端画素複製なし、周期 2*(length-1)）の定数時間版。
static inline float reflect_coord_float(float p, int length) {
    if (length <= 1) {
        return 0.0f;
    }
    float upper = float(length - 1);
    float period = 2.0f * upper;
    float m = fmod(p, period);
    if (m < 0.0f) {
        m += period;
    }
    return m <= upper ? m : period - m;
}

static inline float3 read_transform_rgb(
    const device float* input,
    constant TransformToCanvasParams& p,
    int x,
    int y
) {
    if (p.border_mode == 1) {
        x = reflect_coord(x, p.input_width);
        y = reflect_coord(y, p.input_height);
    } else if (x < 0 || y < 0 || x >= p.input_width || y >= p.input_height) {
        return float3(0.0f);
    }
    int base = (y * p.input_width + x) * 3;
    return float3(input[base + 0], input[base + 1], input[base + 2]);
}

// pybind 側で RGB(3ch) 必須のため float3 でベクトル化（旧スカラー per-ch ループの3倍のメモリ命令を削減）。
static inline float3 sample_transform_linear_rgb(
    const device float* input,
    constant TransformToCanvasParams& p,
    float sx,
    float sy
) {
    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float ax = sx - float(x0);
    float ay = sy - float(y0);

    float3 v00 = read_transform_rgb(input, p, x0, y0);
    float3 v10 = read_transform_rgb(input, p, x1, y0);
    float3 v01 = read_transform_rgb(input, p, x0, y1);
    float3 v11 = read_transform_rgb(input, p, x1, y1);
    float3 top = mix(v00, v10, ax);
    float3 bottom = mix(v01, v11, ax);
    return mix(top, bottom, ay);
}

kernel void transform_to_canvas_kernel(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant TransformToCanvasParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.canvas_width || y >= p.canvas_height) {
        return;
    }

    float fx = float(x);
    float fy = float(y);
    float denom = p.inverse_matrix[6] * fx + p.inverse_matrix[7] * fy + p.inverse_matrix[8];
    int out_base = (y * p.canvas_width + x) * 3;
    if (fabs(denom) < 1.0e-12f) {
        output[out_base + 0] = 0.0f;
        output[out_base + 1] = 0.0f;
        output[out_base + 2] = 0.0f;
        return;
    }

    float sx = (p.inverse_matrix[0] * fx + p.inverse_matrix[1] * fy + p.inverse_matrix[2]) / denom;
    float sy = (p.inverse_matrix[3] * fx + p.inverse_matrix[4] * fy + p.inverse_matrix[5]) / denom;
    float3 value = sample_transform_linear_rgb(input, p, sx, sy);
    output[out_base + 0] = value.x;
    output[out_base + 1] = value.y;
    output[out_base + 2] = value.z;
}

static inline float read_transform_crop_channel(
    const device float* input,
    constant TransformCropToCanvasParams& p,
    int x,
    int y,
    int ch
) {
    if (p.border_mode == 1) {
        x = reflect_coord(x, p.input_width);
        y = reflect_coord(y, p.input_height);
    } else if (x < 0 || y < 0 || x >= p.input_width || y >= p.input_height) {
        return 0.0f;
    }
    return input[(y * p.input_width + x) * p.channels + ch];
}

static inline float3 read_transform_crop_rgb(
    const device float* input,
    constant TransformCropToCanvasParams& p,
    int x,
    int y
) {
    if (p.border_mode == 1) {
        x = reflect_coord(x, p.input_width);
        y = reflect_coord(y, p.input_height);
    } else if (x < 0 || y < 0 || x >= p.input_width || y >= p.input_height) {
        return float3(0.0f);
    }
    int base = (y * p.input_width + x) * 3;
    return float3(input[base + 0], input[base + 1], input[base + 2]);
}

static inline float3 read_transform_crop_rgb_constant(
    const device float* input,
    constant TransformCropToCanvasParams& p,
    int x,
    int y
) {
    if (x < 0 || y < 0 || x >= p.input_width || y >= p.input_height) {
        return float3(0.0f);
    }
    int base = (y * p.input_width + x) * 3;
    return float3(input[base + 0], input[base + 1], input[base + 2]);
}

static inline float sample_transform_crop_linear(
    const device float* input,
    constant TransformCropToCanvasParams& p,
    float sx,
    float sy,
    int ch
) {
    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float ax = sx - float(x0);
    float ay = sy - float(y0);

    float v00 = read_transform_crop_channel(input, p, x0, y0, ch);
    float v10 = read_transform_crop_channel(input, p, x1, y0, ch);
    float v01 = read_transform_crop_channel(input, p, x0, y1, ch);
    float v11 = read_transform_crop_channel(input, p, x1, y1, ch);
    float top = mix(v00, v10, ax);
    float bottom = mix(v01, v11, ax);
    return mix(top, bottom, ay);
}

static inline float3 sample_transform_crop_linear_rgb(
    const device float* input,
    constant TransformCropToCanvasParams& p,
    float sx,
    float sy
) {
    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float ax = sx - float(x0);
    float ay = sy - float(y0);

    float3 v00 = read_transform_crop_rgb(input, p, x0, y0);
    float3 v10 = read_transform_crop_rgb(input, p, x1, y0);
    float3 v01 = read_transform_crop_rgb(input, p, x0, y1);
    float3 v11 = read_transform_crop_rgb(input, p, x1, y1);
    float3 top = mix(v00, v10, ax);
    float3 bottom = mix(v01, v11, ax);
    return mix(top, bottom, ay);
}

static inline float3 sample_transform_crop_linear_rgb_constant(
    const device float* input,
    constant TransformCropToCanvasParams& p,
    float sx,
    float sy
) {
    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float ax = sx - float(x0);
    float ay = sy - float(y0);

    float3 v00 = read_transform_crop_rgb_constant(input, p, x0, y0);
    float3 v10 = read_transform_crop_rgb_constant(input, p, x1, y0);
    float3 v01 = read_transform_crop_rgb_constant(input, p, x0, y1);
    float3 v11 = read_transform_crop_rgb_constant(input, p, x1, y1);
    float3 top = mix(v00, v10, ax);
    float3 bottom = mix(v01, v11, ax);
    return mix(top, bottom, ay);
}

static inline float2 apply_lens_distortion_source(
    constant TransformCropToCanvasParams& p,
    float sx,
    float sy
) {
    float center_x = float(p.input_width) * 0.5f;
    float center_y = float(p.input_height) * 0.5f;
    float max_radius = sqrt(center_x * center_x + center_y * center_y);
    if (max_radius <= 0.0f) {
        return float2(sx, sy);
    }

    float dx = (sx - center_x) / max_radius;
    float dy = (sy - center_y) / max_radius;
    float r2 = dx * dx + dy * dy;
    float distortion = 1.0f + p.lens_k1 * r2;
    return float2(
        center_x + dx * distortion * max_radius,
        center_y + dy * distortion * max_radius
    );
}

static inline float cubic_weight(float x) {
    constexpr float a = -0.75f;
    x = fabs(x);
    if (x <= 1.0f) {
        return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
    }
    if (x < 2.0f) {
        return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
    }
    return 0.0f;
}

// mesh_map は絶対座標ではなくキャンバス座標からの「変位」を保持する。
// 絶対座標 (= ほぼ線形ランプ) を bicubic 拡大すると Keys(a=-0.75) が 1 次多項式を
// 再現できないため、変形していない領域まで coarse セル周期で波打つ。変位なら
// 0 の領域は厳密に 0 のまま。呼び出し側が自分のキャンバス座標へ足す。
static inline float sample_mesh_map_cubic(
    const device float* mesh_map,
    constant TransformCropToCanvasParams& p,
    float tx,
    float ty
) {
    if (p.mesh_grid_width <= 1 || p.mesh_grid_height <= 1) {
        return 0.0f;
    }

    float gx = (tx + 0.5f) * float(p.mesh_grid_width) / float(p.transform_width) - 0.5f;
    float gy = (ty + 0.5f) * float(p.mesh_grid_height) / float(p.transform_height) - 0.5f;

    int ix = int(floor(gx));
    int iy = int(floor(gy));
    float accum = 0.0f;
    float weight_sum = 0.0f;
    for (int yy = -1; yy <= 2; ++yy) {
        int sy = clamp(iy + yy, 0, p.mesh_grid_height - 1);
        float wy = cubic_weight(gy - float(iy + yy));
        for (int xx = -1; xx <= 2; ++xx) {
            int sx = clamp(ix + xx, 0, p.mesh_grid_width - 1);
            float wx = cubic_weight(gx - float(ix + xx));
            float weight = wx * wy;
            accum += mesh_map[sy * p.mesh_grid_width + sx] * weight;
            weight_sum += weight;
        }
    }
    int fallback_x = clamp(ix, 0, p.mesh_grid_width - 1);
    int fallback_y = clamp(iy, 0, p.mesh_grid_height - 1);
    return weight_sum != 0.0f ? accum / weight_sum : mesh_map[fallback_y * p.mesh_grid_width + fallback_x];
}

// mesh 変位適用後の transform 座標から入力画像をサンプルする共通部
// （行列逆変換 + レンズ + bilinear）。
static inline float3 sample_transform_crop_project(
    const device float* input,
    constant TransformCropToCanvasParams& p,
    float tx,
    float ty
) {
    float denom = p.inverse_matrix[6] * tx + p.inverse_matrix[7] * ty + p.inverse_matrix[8];
    if (fabs(denom) < 1.0e-12f) {
        return float3(0.0f);
    }

    float sx = (p.inverse_matrix[0] * tx + p.inverse_matrix[1] * ty + p.inverse_matrix[2]) / denom;
    float sy = (p.inverse_matrix[3] * tx + p.inverse_matrix[4] * ty + p.inverse_matrix[5]) / denom;
    if (p.lens_enabled) {
        if (p.border_mode == 1) {
            sx = reflect_coord_float(sx, p.input_width);
            sy = reflect_coord_float(sy, p.input_height);
        } else if (sx < 0.0f || sy < 0.0f || sx > float(p.input_width - 1) || sy > float(p.input_height - 1)) {
            return float3(0.0f);
        }
        float2 lens_source = apply_lens_distortion_source(p, sx, sy);
        return sample_transform_crop_linear_rgb_constant(input, p, lens_source.x, lens_source.y);
    }
    return sample_transform_crop_linear_rgb(input, p, sx, sy);
}

static inline float3 sample_transform_crop_at_canvas_point(
    const device float* input,
    constant TransformCropToCanvasParams& p,
    const device float* mesh_map_x,
    const device float* mesh_map_y,
    float tx,
    float ty
) {
    if (p.mesh_enabled) {
        float disp_tx = sample_mesh_map_cubic(mesh_map_x, p, tx, ty);
        float disp_ty = sample_mesh_map_cubic(mesh_map_y, p, tx, ty);
        tx += disp_tx;
        ty += disp_ty;
    }
    return sample_transform_crop_project(input, p, tx, ty);
}

// LUT 化した bicubic mesh 参照。整数キャンバス座標では sample_mesh_map_cubic と
// 同一式（重み・clamp 済みインデックスを C++ 側で事前計算）になるため、
// area/nearest 経路の出力は従来と float 丸め誤差内で一致する。
// x/y 両マップで重みを共有し、旧実装の 2×(16 cubic_weight + 除算) をタップ毎に削減。
// 返り値は sample_mesh_map_cubic と同じく「変位」。
static inline float2 mesh_lut_lookup(
    const device float* mesh_map_x,
    const device float* mesh_map_y,
    constant TransformCropToCanvasParams& p,
    MeshCubicLut ex,
    MeshCubicLut ey
) {
    float accum_x = 0.0f;
    float accum_y = 0.0f;
    float weight_sum = 0.0f;
    for (int j = 0; j < 4; ++j) {
        int row = ey.idx[j] * p.mesh_grid_width;
        float wy = ey.w[j];
        for (int i = 0; i < 4; ++i) {
            float weight = ex.w[i] * wy;
            int off = row + ex.idx[i];
            accum_x += mesh_map_x[off] * weight;
            accum_y += mesh_map_y[off] * weight;
            weight_sum += weight;
        }
    }
    if (weight_sum != 0.0f) {
        return float2(accum_x / weight_sum, accum_y / weight_sum);
    }
    // 旧実装の fallback (clamp(ix) = k=0 のインデックス) と同じ
    int off = ey.idx.y * p.mesh_grid_width + ex.idx.y;
    return float2(mesh_map_x[off], mesh_map_y[off]);
}

static inline float3 sample_transform_crop_at_lut_point(
    const device float* input,
    constant TransformCropToCanvasParams& p,
    const device float* mesh_map_x,
    const device float* mesh_map_y,
    const device MeshCubicLut* mesh_lut_x,
    const device MeshCubicLut* mesh_lut_y,
    int txi,
    int tyi
) {
    float tx = float(txi);
    float ty = float(tyi);
    if (p.mesh_enabled) {
        float2 disp = mesh_lut_lookup(mesh_map_x, mesh_map_y, p, mesh_lut_x[txi], mesh_lut_y[tyi]);
        tx += disp.x;
        ty += disp.y;
    }
    return sample_transform_crop_project(input, p, tx, ty);
}

static inline float3 sample_transform_crop_area_rgb(
    const device float* input,
    constant TransformCropToCanvasParams& p,
    const device float* mesh_map_x,
    const device float* mesh_map_y,
    const device MeshCubicLut* mesh_lut_x,
    const device MeshCubicLut* mesh_lut_y,
    const device float2* mesh_dense,
    int dx,
    int dy
) {
    float tx0 = float(p.source_x) + float(dx) * float(p.source_width) / float(p.draw_width);
    float tx1 = float(p.source_x) + float(dx + 1) * float(p.source_width) / float(p.draw_width);
    float ty0 = float(p.source_y) + float(dy) * float(p.source_height) / float(p.draw_height);
    float ty1 = float(p.source_y) + float(dy + 1) * float(p.source_height) / float(p.draw_height);
    if (tx1 <= tx0 || ty1 <= ty0) {
        return float3(0.0f);
    }

    int ix0 = int(floor(tx0));
    int ix1 = int(ceil(tx1));
    int iy0 = int(floor(ty0));
    int iy1 = int(ceil(ty1));
    float3 accum = float3(0.0f);
    float weight_sum = 0.0f;
    // タップは整数キャンバス座標のみ → mesh は dense(事前 bake) / LUT 経路が使える。
    bool use_dense = p.mesh_enabled && p.use_mesh_dense;
    bool use_lut = p.mesh_enabled && p.use_mesh_lut;

    for (int yy = iy0; yy < iy1; ++yy) {
        float wy = max(0.0f, min(ty1, float(yy + 1)) - max(ty0, float(yy)));
        if (wy <= 0.0f) {
            continue;
        }
        // y 側 LUT エントリは行内で不変なのでホイスト。
        MeshCubicLut ey = { int4(0), float4(0.0f) };
        if (use_lut && !use_dense) {
            ey = mesh_lut_y[yy];
        }
        int dense_row = (yy - p.source_y) * p.source_width - p.source_x;
        for (int xx = ix0; xx < ix1; ++xx) {
            float wx = max(0.0f, min(tx1, float(xx + 1)) - max(tx0, float(xx)));
            float weight = wx * wy;
            if (weight <= 0.0f) {
                continue;
            }
            float3 value;
            if (use_dense) {
                // タップ毎の 32 ロード + LUT 参照を bake 済み float2 1 ロードに置換。
                float2 mapped = mesh_dense[dense_row + xx];
                value = sample_transform_crop_project(input, p, mapped.x, mapped.y);
            } else if (use_lut) {
                float2 disp = mesh_lut_lookup(mesh_map_x, mesh_map_y, p, mesh_lut_x[xx], ey);
                value = sample_transform_crop_project(input, p, float(xx) + disp.x, float(yy) + disp.y);
            } else {
                value = sample_transform_crop_at_canvas_point(input, p, mesh_map_x, mesh_map_y, float(xx), float(yy));
            }
            accum += value * weight;
            weight_sum += weight;
        }
    }

    return weight_sum > 0.0f ? accum / weight_sum : float3(0.0f);
}

// area/nearest 用の mesh 変位密マップ bake。整数キャンバス座標で mesh_lut_lookup と
// 同一式を評価するため、メインカーネルの LUT 経路とビット一致する。
// 対象は source_rect 領域のみ（area/nearest のタップは必ずこの範囲内）。
kernel void bake_mesh_dense_kernel(
    const device float* mesh_map_x [[buffer(0)]],
    const device float* mesh_map_y [[buffer(1)]],
    const device MeshCubicLut* mesh_lut_x [[buffer(2)]],
    const device MeshCubicLut* mesh_lut_y [[buffer(3)]],
    device float2* dense [[buffer(4)]],
    constant TransformCropToCanvasParams& p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int rx = int(gid.x);
    int ry = int(gid.y);
    if (rx >= p.source_width || ry >= p.source_height) {
        return;
    }
    int tx = p.source_x + rx;
    int ty = p.source_y + ry;
    // dense には変位ではなく絶対座標を bake する（消費側はそのまま project する）。
    dense[ry * p.source_width + rx] =
        float2(float(tx), float(ty))
        + mesh_lut_lookup(mesh_map_x, mesh_map_y, p, mesh_lut_x[tx], mesh_lut_y[ty]);
}

kernel void transform_crop_to_canvas_kernel(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant TransformCropToCanvasParams& p [[buffer(2)]],
    const device float* mesh_map_x [[buffer(3)]],
    const device float* mesh_map_y [[buffer(4)]],
    const device MeshCubicLut* mesh_lut_x [[buffer(5)]],
    const device MeshCubicLut* mesh_lut_y [[buffer(6)]],
    const device float2* mesh_dense [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= p.canvas_width || y >= p.canvas_height) {
        return;
    }

    int out_base = (y * p.canvas_width + x) * p.channels;
    int dx = x - p.offset_x;
    int dy = y - p.offset_y;
    bool inside = dx >= 0 && dy >= 0 && dx < p.draw_width && dy < p.draw_height;
    if (!inside) {
        for (int ch = 0; ch < p.channels; ++ch) {
            output[out_base + ch] = 0.0f;
        }
        return;
    }

    float3 value;
    if (p.interpolation == 2) {
        value = sample_transform_crop_area_rgb(input, p, mesh_map_x, mesh_map_y, mesh_lut_x, mesh_lut_y, mesh_dense, dx, dy);
        output[out_base + 0] = value.x;
        output[out_base + 1] = value.y;
        output[out_base + 2] = value.z;
        return;
    }

    if (p.interpolation == 0) {
        // nearest もタップは整数キャンバス座標 → dense / LUT 経路。
        int txi = p.source_x + min(int(floor(float(dx) * float(p.source_width) / float(p.draw_width))), p.source_width - 1);
        int tyi = p.source_y + min(int(floor(float(dy) * float(p.source_height) / float(p.draw_height))), p.source_height - 1);
        if (p.mesh_enabled && p.use_mesh_dense) {
            float2 mapped = mesh_dense[(tyi - p.source_y) * p.source_width + (txi - p.source_x)];
            value = sample_transform_crop_project(input, p, mapped.x, mapped.y);
        } else if (p.mesh_enabled && p.use_mesh_lut) {
            value = sample_transform_crop_at_lut_point(input, p, mesh_map_x, mesh_map_y, mesh_lut_x, mesh_lut_y, txi, tyi);
        } else {
            value = sample_transform_crop_at_canvas_point(input, p, mesh_map_x, mesh_map_y, float(txi), float(tyi));
        }
        output[out_base + 0] = value.x;
        output[out_base + 1] = value.y;
        output[out_base + 2] = value.z;
        return;
    }

    // linear はタップが小数座標のため従来の bicubic 直接評価を維持
    //（LUT を bilinear 参照すると reference と一致しなくなる）。
    float tx = float(p.source_x) + (float(dx) + 0.5f) * float(p.source_width) / float(p.draw_width) - 0.5f;
    float ty = float(p.source_y) + (float(dy) + 0.5f) * float(p.source_height) / float(p.draw_height) - 0.5f;
    value = sample_transform_crop_at_canvas_point(input, p, mesh_map_x, mesh_map_y, tx, ty);
    output[out_base + 0] = value.x;
    output[out_base + 1] = value.y;
    output[out_base + 2] = value.z;
}
)METAL";

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) {
        throw std::runtime_error("Metal function not found");
    }
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        std::string message = error ? [[error localizedDescription] UTF8String] : "unknown Metal pipeline error";
        throw std::runtime_error(message);
    }
    return pipeline;
}

struct MetalPipelines {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
    id<MTLComputePipelineState> fit_crop_to_canvas;
    id<MTLComputePipelineState> transform_to_canvas;
    id<MTLComputePipelineState> transform_crop_to_canvas;
    id<MTLComputePipelineState> bake_mesh_dense;
    // mesh 変位密マップのフレーム間キャッシュ（GPU 常駐、CPU から触らない）。
    // キーは coarse map バイト列 + grid/transform 寸法 + region の FNV-1a。
    // ズーム/パンや crop 枠調整では mesh が不変なため bake を再利用できる。
    std::mutex mesh_dense_mutex;
    id<MTLBuffer> mesh_dense_buffer;
    uint64_t mesh_dense_key;
    size_t mesh_dense_bytes;
};

MetalPipelines& metal_pipelines() {
    static MetalPipelines state{};
    static std::once_flag once;
    static std::string init_error;

    std::call_once(once, []() {
        @autoreleasepool {
            state.device = MTLCreateSystemDefaultDevice();
            if (!state.device) {
                init_error = "Metal device is unavailable";
                return;
            }

            NSError* error = nil;
            NSString* source = [NSString stringWithUTF8String:kMetalSource];
            state.library = [state.device newLibraryWithSource:source options:nil error:&error];
            if (!state.library) {
                init_error = error ? [[error localizedDescription] UTF8String] : "unknown Metal library error";
                return;
            }

            state.queue = [state.device newCommandQueue];
            if (!state.queue) {
                init_error = "Metal command queue is unavailable";
                return;
            }

            try {
                state.fit_crop_to_canvas = make_pipeline(state.device, state.library, @"fit_crop_to_canvas_kernel");
                state.transform_to_canvas = make_pipeline(state.device, state.library, @"transform_to_canvas_kernel");
                state.transform_crop_to_canvas = make_pipeline(state.device, state.library, @"transform_crop_to_canvas_kernel");
                state.bake_mesh_dense = make_pipeline(state.device, state.library, @"bake_mesh_dense_kernel");
            } catch (const std::exception& exc) {
                init_error = exc.what();
            }
        }
    });

    if (!init_error.empty()) {
        throw std::runtime_error(init_error);
    }
    return state;
}

void dispatch_2d(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    NSUInteger width,
    NSUInteger height
) {
    [encoder setComputePipelineState:pipeline];
    NSUInteger tw = std::max<NSUInteger>(1, pipeline.threadExecutionWidth);
    NSUInteger th = std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup / tw);
    if (th > 16) {
        th = 16;
    }
    MTLSize threads_per_group = MTLSizeMake(tw, th, 1);
    MTLSize grid = MTLSizeMake(width, height, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads_per_group];
}

int interpolation_code(const std::string& interpolation) {
    if (interpolation == "nearest") {
        return INTERPOLATION_NEAREST;
    }
    if (interpolation == "linear") {
        return INTERPOLATION_LINEAR;
    }
    if (interpolation == "area") {
        return INTERPOLATION_AREA;
    }
    throw std::invalid_argument("Metal image transform supports nearest, linear, and area interpolation");
}

bool invert_3x3(const double m[9], float out[9]) {
    double det =
        m[0] * (m[4] * m[8] - m[5] * m[7]) -
        m[1] * (m[3] * m[8] - m[5] * m[6]) +
        m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (std::abs(det) < 1.0e-12) {
        return false;
    }
    double inv_det = 1.0 / det;
    double inv[9] = {
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    };
    for (int i = 0; i < 9; ++i) {
        out[i] = static_cast<float>(inv[i]);
    }
    return true;
}

// mesh 変位密マップ bake の有効化フラグ。既定 off（メモリ使用と挙動変化を
// 段階導入するため）。PLATYPUS_IMAGE_TRANSFORM_MESH_DENSE=1 で有効。
bool mesh_dense_enabled() {
    const char* value = std::getenv("PLATYPUS_IMAGE_TRANSFORM_MESH_DENSE");
    if (!value) {
        return false;
    }
    std::string v(value);
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return std::tolower(c); });
    return v == "1" || v == "true" || v == "on" || v == "yes";
}

// 密マップの上限バイト数。超える region では LUT 経路へフォールバック。
size_t mesh_dense_budget_bytes() {
    const char* value = std::getenv("PLATYPUS_IMAGE_TRANSFORM_MESH_DENSE_BUDGET_MB");
    long mb = value ? std::strtol(value, nullptr, 10) : 256;
    if (mb <= 0) {
        mb = 256;
    }
    return static_cast<size_t>(mb) * 1024 * 1024;
}

uint64_t fnv1a_hash(const void* data, size_t length, uint64_t hash = 1469598103934665603ULL) {
    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < length; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

// MSL 側 cubic_weight (Keys, a=-0.75) と同一式のホスト版。
// LUT に事前計算する重みがカーネル内評価と一致するよう式の形を揃える。
float cubic_weight_host(float x) {
    constexpr float a = -0.75f;
    x = std::fabs(x);
    if (x <= 1.0f) {
        return (a + 2.0f) * x * x * x - (a + 3.0f) * x * x + 1.0f;
    }
    if (x < 2.0f) {
        return a * x * x * x - 5.0f * a * x * x + 8.0f * a * x - 4.0f * a;
    }
    return 0.0f;
}

// 整数キャンバス座標ごとの bicubic mesh 参照 LUT（clamp 済みインデックス + 重み）。
// gx = (t+0.5)*grid/extent - 0.5 は sample_mesh_map_cubic と同一。
std::vector<MeshCubicLutEntry> build_mesh_cubic_lut(int extent, int grid, int transform_extent) {
    std::vector<MeshCubicLutEntry> lut(static_cast<size_t>(std::max(1, extent)));
    for (int t = 0; t < extent; ++t) {
        float g = (static_cast<float>(t) + 0.5f) * static_cast<float>(grid) / static_cast<float>(transform_extent) - 0.5f;
        int i0 = static_cast<int>(std::floor(g));
        for (int k = -1; k <= 2; ++k) {
            lut[static_cast<size_t>(t)].idx[k + 1] = std::min(std::max(i0 + k, 0), grid - 1);
            lut[static_cast<size_t>(t)].w[k + 1] = cubic_weight_host(g - static_cast<float>(i0 + k));
        }
    }
    return lut;
}

std::vector<double> matrix_to_3x3(const py::object& matrix) {
    py::array_t<double, py::array::c_style | py::array::forcecast> arr = py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(matrix);
    py::buffer_info info = arr.request();
    if (info.ndim != 2) {
        throw std::invalid_argument("matrix must be 2D");
    }
    const double* data = static_cast<const double*>(info.ptr);
    if (info.shape[0] == 2 && info.shape[1] == 3) {
        return {
            data[0], data[1], data[2],
            data[3], data[4], data[5],
            0.0, 0.0, 1.0,
        };
    }
    if (info.shape[0] == 3 && info.shape[1] == 3) {
        return {
            data[0], data[1], data[2],
            data[3], data[4], data[5],
            data[6], data[7], data[8],
        };
    }
    throw std::invalid_argument("matrix must be 2x3 or 3x3");
}

}  // namespace

py::array_t<float> fit_crop_to_canvas(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    const py::object& source_rect,
    int canvas_width,
    int canvas_height,
    int draw_width,
    int draw_height,
    int offset_x,
    int offset_y,
    const std::string& interpolation
) {
    py::buffer_info in = image.request();
    if (in.ndim != 3 || (in.shape[2] != 1 && in.shape[2] != 3)) {
        throw std::invalid_argument("image must be a 3D float32 array with 1 or 3 channels");
    }

    py::sequence rect = py::cast<py::sequence>(source_rect);
    if (rect.size() < 4) {
        throw std::invalid_argument("source_rect must have four values");
    }
    int source_x = py::cast<int>(rect[0]);
    int source_y = py::cast<int>(rect[1]);
    int source_width = std::max(1, py::cast<int>(rect[2]));
    int source_height = std::max(1, py::cast<int>(rect[3]));

    const int input_width = static_cast<int>(in.shape[1]);
    const int input_height = static_cast<int>(in.shape[0]);
    const int channels = static_cast<int>(in.shape[2]);
    canvas_width = std::max(1, canvas_width);
    canvas_height = std::max(1, canvas_height);
    draw_width = std::max(1, draw_width);
    draw_height = std::max(1, draw_height);

    if (source_x < 0 || source_y < 0 || source_x + source_width > input_width || source_y + source_height > input_height) {
        throw std::invalid_argument("source_rect must be inside image bounds");
    }
    if (offset_x < 0 || offset_y < 0 || offset_x + draw_width > canvas_width || offset_y + draw_height > canvas_height) {
        throw std::invalid_argument("draw rectangle must fit inside canvas");
    }

    std::vector<py::ssize_t> shape{
        static_cast<py::ssize_t>(canvas_height),
        static_cast<py::ssize_t>(canvas_width),
        static_cast<py::ssize_t>(channels),
    };
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    @autoreleasepool {
        MetalPipelines& pipelines = metal_pipelines();

        const size_t input_bytes = static_cast<size_t>(input_width) * static_cast<size_t>(input_height) * static_cast<size_t>(channels) * sizeof(float);
        const size_t output_bytes = static_cast<size_t>(canvas_width) * static_cast<size_t>(canvas_height) * static_cast<size_t>(channels) * sizeof(float);

        BufferBinding input_buffer = make_buffer_for_input(pipelines.device, in.ptr, input_bytes);
        BufferBinding output_buffer = make_buffer_for_output(pipelines.device, out.ptr, output_bytes);

        FitCropToCanvasParams params{
            input_width,
            input_height,
            channels,
            source_x,
            source_y,
            source_width,
            source_height,
            canvas_width,
            canvas_height,
            draw_width,
            draw_height,
            offset_x,
            offset_y,
            interpolation_code(interpolation),
        };

        id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setBuffer:input_buffer.buffer offset:input_buffer.offset atIndex:0];
        [encoder setBuffer:output_buffer.buffer offset:output_buffer.offset atIndex:1];
        // params は 4KB 未満なので毎呼び出しの MTLBuffer 確保を避けて setBytes で渡す。
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_2d(encoder, pipelines.fit_crop_to_canvas, canvas_width, canvas_height);
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer error]) {
            throw std::runtime_error([[[command_buffer error] localizedDescription] UTF8String]);
        }

        if (!output_buffer.no_copy) {
            std::memcpy(out.ptr, [output_buffer.buffer contents], output_bytes);
        }
    }

    return result;
}

py::array_t<float> transform_to_canvas(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    const py::object& matrix,
    int canvas_width,
    int canvas_height,
    const std::string& transform_type,
    const std::string& interpolation,
    const std::string& border_mode
) {
    if (interpolation != "linear") {
        throw std::invalid_argument("Metal transform_to_canvas supports linear interpolation only");
    }
    if (border_mode != "constant" && border_mode != "reflect") {
        throw std::invalid_argument("Metal transform_to_canvas supports constant and reflect borders only");
    }

    py::buffer_info in = image.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must be a 3D RGB float32 array");
    }

    const int input_width = static_cast<int>(in.shape[1]);
    const int input_height = static_cast<int>(in.shape[0]);
    const int channels = static_cast<int>(in.shape[2]);
    canvas_width = std::max(1, canvas_width);
    canvas_height = std::max(1, canvas_height);

    std::vector<double> matrix3 = matrix_to_3x3(matrix);
    float inverse_matrix[9];
    if (!invert_3x3(matrix3.data(), inverse_matrix)) {
        throw std::invalid_argument("matrix is singular");
    }

    std::vector<py::ssize_t> shape{
        static_cast<py::ssize_t>(canvas_height),
        static_cast<py::ssize_t>(canvas_width),
        static_cast<py::ssize_t>(channels),
    };
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    @autoreleasepool {
        MetalPipelines& pipelines = metal_pipelines();

        const size_t input_bytes = static_cast<size_t>(input_width) * static_cast<size_t>(input_height) * static_cast<size_t>(channels) * sizeof(float);
        const size_t output_bytes = static_cast<size_t>(canvas_width) * static_cast<size_t>(canvas_height) * static_cast<size_t>(channels) * sizeof(float);

        BufferBinding input_buffer = make_buffer_for_input(pipelines.device, in.ptr, input_bytes);
        BufferBinding output_buffer = make_buffer_for_output(pipelines.device, out.ptr, output_bytes);

        TransformToCanvasParams params{};
        params.input_width = input_width;
        params.input_height = input_height;
        params.channels = channels;
        params.canvas_width = canvas_width;
        params.canvas_height = canvas_height;
        params.border_mode = border_mode == "reflect" ? 1 : 0;
        for (int i = 0; i < 9; ++i) {
            params.inverse_matrix[i] = inverse_matrix[i];
        }

        id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setBuffer:input_buffer.buffer offset:input_buffer.offset atIndex:0];
        [encoder setBuffer:output_buffer.buffer offset:output_buffer.offset atIndex:1];
        // params は 4KB 未満なので毎呼び出しの MTLBuffer 確保を避けて setBytes で渡す。
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_2d(encoder, pipelines.transform_to_canvas, canvas_width, canvas_height);
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer error]) {
            throw std::runtime_error([[[command_buffer error] localizedDescription] UTF8String]);
        }

        if (!output_buffer.no_copy) {
            std::memcpy(out.ptr, [output_buffer.buffer contents], output_bytes);
        }
    }

    return result;
}

py::array_t<float> transform_crop_to_canvas(
    py::array_t<float, py::array::c_style | py::array::forcecast> image,
    const py::object& matrix,
    const py::object& source_rect,
    int transform_width,
    int transform_height,
    int canvas_width,
    int canvas_height,
    int draw_width,
    int draw_height,
    int offset_x,
    int offset_y,
    const std::string& transform_type,
    const std::string& interpolation,
    const std::string& border_mode,
    float lens_strength,
    float lens_scale,
    const py::object& mesh_map_x_obj,
    const py::object& mesh_map_y_obj
) {
    if (interpolation != "nearest" && interpolation != "linear" && interpolation != "area") {
        throw std::invalid_argument("Metal transform_crop_to_canvas supports nearest, linear, and area interpolation");
    }
    if (border_mode != "constant" && border_mode != "reflect") {
        throw std::invalid_argument("Metal transform_crop_to_canvas supports constant and reflect borders only");
    }
    if (std::abs(lens_scale - 1.0f) > 0.01f) {
        throw std::invalid_argument("Metal transform_crop_to_canvas supports lens_scale=1.0 only");
    }

    py::buffer_info in = image.request();
    if (in.ndim != 3 || in.shape[2] != 3) {
        throw std::invalid_argument("image must be a 3D RGB float32 array");
    }

    py::sequence rect = py::cast<py::sequence>(source_rect);
    if (rect.size() < 4) {
        throw std::invalid_argument("source_rect must have four values");
    }
    int source_x = py::cast<int>(rect[0]);
    int source_y = py::cast<int>(rect[1]);
    int source_width = std::max(1, py::cast<int>(rect[2]));
    int source_height = std::max(1, py::cast<int>(rect[3]));

    const int input_width = static_cast<int>(in.shape[1]);
    const int input_height = static_cast<int>(in.shape[0]);
    const int channels = static_cast<int>(in.shape[2]);
    canvas_width = std::max(1, canvas_width);
    canvas_height = std::max(1, canvas_height);
    draw_width = std::max(1, draw_width);
    draw_height = std::max(1, draw_height);

    if (source_x < 0 || source_y < 0 || source_x + source_width > transform_width || source_y + source_height > transform_height) {
        throw std::invalid_argument("source_rect must be inside transformed canvas bounds");
    }
    if (offset_x < 0 || offset_y < 0 || offset_x + draw_width > canvas_width || offset_y + draw_height > canvas_height) {
        throw std::invalid_argument("draw rectangle must fit inside canvas");
    }

    std::vector<double> matrix3 = matrix_to_3x3(matrix);
    float inverse_matrix[9];
    if (!invert_3x3(matrix3.data(), inverse_matrix)) {
        throw std::invalid_argument("matrix is singular");
    }

    bool mesh_enabled = !mesh_map_x_obj.is_none() && !mesh_map_y_obj.is_none();
    py::array_t<float, py::array::c_style | py::array::forcecast> mesh_map_x_arr;
    py::array_t<float, py::array::c_style | py::array::forcecast> mesh_map_y_arr;
    std::unique_ptr<py::buffer_info> mesh_map_x_info;
    std::unique_ptr<py::buffer_info> mesh_map_y_info;
    int mesh_grid_width = 1;
    int mesh_grid_height = 1;
    std::vector<float> dummy_mesh_map(1, 0.0f);

    if (mesh_enabled) {
        mesh_map_x_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(mesh_map_x_obj);
        mesh_map_y_arr = py::cast<py::array_t<float, py::array::c_style | py::array::forcecast>>(mesh_map_y_obj);
        mesh_map_x_info = std::make_unique<py::buffer_info>(mesh_map_x_arr.request());
        mesh_map_y_info = std::make_unique<py::buffer_info>(mesh_map_y_arr.request());
        if (mesh_map_x_info->ndim != 2 || mesh_map_y_info->ndim != 2) {
            throw std::invalid_argument("mesh maps must be 2D float32 arrays");
        }
        if (mesh_map_x_info->shape[0] != mesh_map_y_info->shape[0] || mesh_map_x_info->shape[1] != mesh_map_y_info->shape[1]) {
            throw std::invalid_argument("mesh_map_x and mesh_map_y must have the same shape");
        }
        mesh_grid_height = static_cast<int>(mesh_map_x_info->shape[0]);
        mesh_grid_width = static_cast<int>(mesh_map_x_info->shape[1]);
        if (mesh_grid_width < 2 || mesh_grid_height < 2) {
            mesh_enabled = false;
            mesh_grid_width = 1;
            mesh_grid_height = 1;
        }
    }

    std::vector<py::ssize_t> shape{
        static_cast<py::ssize_t>(canvas_height),
        static_cast<py::ssize_t>(canvas_width),
        static_cast<py::ssize_t>(channels),
    };
    py::array_t<float> result(shape);
    py::buffer_info out = result.request();

    @autoreleasepool {
        MetalPipelines& pipelines = metal_pipelines();

        const size_t input_bytes = static_cast<size_t>(input_width) * static_cast<size_t>(input_height) * static_cast<size_t>(channels) * sizeof(float);
        const size_t output_bytes = static_cast<size_t>(canvas_width) * static_cast<size_t>(canvas_height) * static_cast<size_t>(channels) * sizeof(float);

        BufferBinding input_buffer = make_buffer_for_input(pipelines.device, in.ptr, input_bytes);
        BufferBinding output_buffer = make_buffer_for_output(pipelines.device, out.ptr, output_bytes);
        const void* mesh_x_ptr = mesh_enabled ? mesh_map_x_info->ptr : static_cast<const void*>(dummy_mesh_map.data());
        const void* mesh_y_ptr = mesh_enabled ? mesh_map_y_info->ptr : static_cast<const void*>(dummy_mesh_map.data());
        const size_t mesh_bytes = static_cast<size_t>(mesh_grid_width) * static_cast<size_t>(mesh_grid_height) * sizeof(float);
        BufferBinding mesh_x_buffer = make_buffer_for_input(pipelines.device, mesh_x_ptr, mesh_bytes);
        BufferBinding mesh_y_buffer = make_buffer_for_input(pipelines.device, mesh_y_ptr, mesh_bytes);

        TransformCropToCanvasParams params{};
        params.input_width = input_width;
        params.input_height = input_height;
        params.channels = channels;
        params.transform_width = transform_width;
        params.transform_height = transform_height;
        params.canvas_width = canvas_width;
        params.canvas_height = canvas_height;
        params.source_x = source_x;
        params.source_y = source_y;
        params.source_width = source_width;
        params.source_height = source_height;
        params.draw_width = draw_width;
        params.draw_height = draw_height;
        params.offset_x = offset_x;
        params.offset_y = offset_y;
        params.interpolation = interpolation_code(interpolation);
        params.border_mode = border_mode == "reflect" ? 1 : 0;
        params.lens_enabled = std::abs(lens_strength) > 1.0e-6f ? 1 : 0;
        params.lens_k1 = lens_strength / 200.0f;
        params.mesh_enabled = mesh_enabled ? 1 : 0;
        params.mesh_grid_width = mesh_grid_width;
        params.mesh_grid_height = mesh_grid_height;
        for (int i = 0; i < 9; ++i) {
            params.inverse_matrix[i] = inverse_matrix[i];
        }

        // area/nearest のタップは整数キャンバス座標なので、bicubic mesh 参照の
        // インデックス・重みを軸ごとに事前計算してカーネルの 2×16 cubic_weight
        // 評価を LUT 読みに置き換える（linear は小数タップのため対象外）。
        // 8192px でも軸あたり 256KB 程度、生成コストは数十 µs。
        std::vector<MeshCubicLutEntry> mesh_lut_x_host(1);
        std::vector<MeshCubicLutEntry> mesh_lut_y_host(1);
        bool use_mesh_lut = mesh_enabled && params.interpolation != INTERPOLATION_LINEAR;
        if (use_mesh_lut) {
            mesh_lut_x_host = build_mesh_cubic_lut(transform_width, mesh_grid_width, transform_width);
            mesh_lut_y_host = build_mesh_cubic_lut(transform_height, mesh_grid_height, transform_height);
        }
        params.use_mesh_lut = use_mesh_lut ? 1 : 0;
        BufferBinding mesh_lut_x_buffer = make_buffer_for_input(
            pipelines.device, mesh_lut_x_host.data(), mesh_lut_x_host.size() * sizeof(MeshCubicLutEntry));
        BufferBinding mesh_lut_y_buffer = make_buffer_for_input(
            pipelines.device, mesh_lut_y_host.data(), mesh_lut_y_host.size() * sizeof(MeshCubicLutEntry));

        // 密マップ bake（env ゲート・バジェット制限付き）。coarse map と region が
        // 前回と同じなら GPU 常駐バッファを再利用して bake をスキップする
        // （ズーム/パン等で mesh 不変のまま fused が再実行されるケースが対象）。
        // キャッシュバッファは commit〜完了まで差し替えられないよう mutex を保持する。
        std::unique_lock<std::mutex> dense_lock(pipelines.mesh_dense_mutex, std::defer_lock);
        bool use_mesh_dense = false;
        bool dense_needs_bake = false;
        id<MTLBuffer> dense_buffer = nil;
        if (use_mesh_lut && mesh_dense_enabled()) {
            const size_t dense_bytes =
                static_cast<size_t>(source_width) * static_cast<size_t>(source_height) * sizeof(float) * 2;
            if (dense_bytes <= mesh_dense_budget_bytes()) {
                uint64_t key = fnv1a_hash(mesh_map_x_info->ptr, mesh_bytes);
                key = fnv1a_hash(mesh_map_y_info->ptr, mesh_bytes, key);
                const int32_t key_dims[8] = {
                    mesh_grid_width, mesh_grid_height, transform_width, transform_height,
                    source_x, source_y, source_width, source_height,
                };
                key = fnv1a_hash(key_dims, sizeof(key_dims), key);

                dense_lock.lock();
                if (pipelines.mesh_dense_buffer != nil && pipelines.mesh_dense_key == key) {
                    dense_buffer = pipelines.mesh_dense_buffer;
                } else {
                    if (pipelines.mesh_dense_buffer == nil || pipelines.mesh_dense_bytes != dense_bytes) {
                        pipelines.mesh_dense_buffer =
                            [pipelines.device newBufferWithLength:dense_bytes options:MTLResourceStorageModePrivate];
                        pipelines.mesh_dense_bytes = dense_bytes;
                    }
                    if (pipelines.mesh_dense_buffer != nil) {
                        pipelines.mesh_dense_key = key;
                        dense_buffer = pipelines.mesh_dense_buffer;
                        dense_needs_bake = true;
                    }
                }
                use_mesh_dense = dense_buffer != nil;
                if (!use_mesh_dense) {
                    dense_lock.unlock();
                }
            }
        }
        params.use_mesh_dense = use_mesh_dense ? 1 : 0;

        id<MTLCommandBuffer> command_buffer = [pipelines.queue commandBuffer];
        if (dense_needs_bake) {
            id<MTLComputeCommandEncoder> bake_encoder = [command_buffer computeCommandEncoder];
            [bake_encoder setBuffer:mesh_x_buffer.buffer offset:mesh_x_buffer.offset atIndex:0];
            [bake_encoder setBuffer:mesh_y_buffer.buffer offset:mesh_y_buffer.offset atIndex:1];
            [bake_encoder setBuffer:mesh_lut_x_buffer.buffer offset:mesh_lut_x_buffer.offset atIndex:2];
            [bake_encoder setBuffer:mesh_lut_y_buffer.buffer offset:mesh_lut_y_buffer.offset atIndex:3];
            [bake_encoder setBuffer:dense_buffer offset:0 atIndex:4];
            [bake_encoder setBytes:&params length:sizeof(params) atIndex:5];
            dispatch_2d(bake_encoder, pipelines.bake_mesh_dense, source_width, source_height);
            [bake_encoder endEncoding];
        }
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        [encoder setBuffer:input_buffer.buffer offset:input_buffer.offset atIndex:0];
        [encoder setBuffer:output_buffer.buffer offset:output_buffer.offset atIndex:1];
        // params は 4KB 未満なので毎呼び出しの MTLBuffer 確保を避けて setBytes で渡す。
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        [encoder setBuffer:mesh_x_buffer.buffer offset:mesh_x_buffer.offset atIndex:3];
        [encoder setBuffer:mesh_y_buffer.buffer offset:mesh_y_buffer.offset atIndex:4];
        [encoder setBuffer:mesh_lut_x_buffer.buffer offset:mesh_lut_x_buffer.offset atIndex:5];
        [encoder setBuffer:mesh_lut_y_buffer.buffer offset:mesh_lut_y_buffer.offset atIndex:6];
        // 未使用時は既存バッファを流用（use_mesh_dense=0 のときカーネルは参照しない）。
        [encoder setBuffer:(use_mesh_dense ? dense_buffer : mesh_x_buffer.buffer)
                    offset:(use_mesh_dense ? 0 : mesh_x_buffer.offset)
                   atIndex:7];
        dispatch_2d(encoder, pipelines.transform_crop_to_canvas, canvas_width, canvas_height);
        [encoder endEncoding];

        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if ([command_buffer error]) {
            throw std::runtime_error([[[command_buffer error] localizedDescription] UTF8String]);
        }

        if (!output_buffer.no_copy) {
            std::memcpy(out.ptr, [output_buffer.buffer contents], output_bytes);
        }
    }

    return result;
}

PYBIND11_MODULE(_image_transform_metal, m) {
    m.doc() = "Metal image transform backend";
    m.def("metal_available", []() {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            return device != nil;
        }
    });
    m.def(
        "fit_crop_to_canvas",
        &fit_crop_to_canvas,
        py::arg("image"),
        py::arg("source_rect"),
        py::arg("canvas_width"),
        py::arg("canvas_height"),
        py::arg("draw_width"),
        py::arg("draw_height"),
        py::arg("offset_x") = 0,
        py::arg("offset_y") = 0,
        py::arg("interpolation") = "area"
    );
    m.def(
        "transform_to_canvas",
        &transform_to_canvas,
        py::arg("image"),
        py::arg("matrix"),
        py::arg("canvas_width"),
        py::arg("canvas_height"),
        py::arg("transform_type") = "affine",
        py::arg("interpolation") = "linear",
        py::arg("border_mode") = "reflect"
    );
    m.def(
        "transform_crop_to_canvas",
        &transform_crop_to_canvas,
        py::arg("image"),
        py::arg("matrix"),
        py::arg("source_rect"),
        py::arg("transform_width"),
        py::arg("transform_height"),
        py::arg("canvas_width"),
        py::arg("canvas_height"),
        py::arg("draw_width"),
        py::arg("draw_height"),
        py::arg("offset_x") = 0,
        py::arg("offset_y") = 0,
        py::arg("transform_type") = "affine",
        py::arg("interpolation") = "linear",
        py::arg("border_mode") = "reflect",
        py::arg("lens_strength") = 0.0f,
        py::arg("lens_scale") = 1.0f,
        py::arg("mesh_map_x") = py::none(),
        py::arg("mesh_map_y") = py::none()
    );
}
