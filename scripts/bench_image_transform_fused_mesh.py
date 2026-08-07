import argparse
import os
import pathlib
import sys
import time

import cv2
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=4096)
    parser.add_argument("--height", type=int, default=2560)
    parser.add_argument("--canvas-width", type=int, default=1600)
    parser.add_argument("--canvas-height", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--grid-step", type=int, default=64)
    # settle フレーム(area)が最重量なのに linear 固定だったため選択可能にする。
    parser.add_argument("--interpolation", choices=["linear", "area", "nearest"], default="linear")
    args = parser.parse_args()

    os.environ["PLATYPUS_IMAGE_TRANSFORM_BACKEND"] = "metal"

    import effects  # noqa: F401
    import params
    from cores import core
    from cores.distortion_correction import calculate_mesh_mls_coarse_map
    from effect_backends import image_transform_adapter

    status = image_transform_adapter.backend_status()
    if not status.native:
        raise SystemExit(f"Metal backend unavailable: {status.detail}")

    rng = np.random.default_rng(123)
    image = rng.random((args.height, args.width, 3), dtype=np.float32)
    transform_matrix, size, transform_type = core.combined_rotation_canvas_matrix(image.shape, 0.0, 0, None)
    mesh_size = (4, 4)
    control_points = {
        (1, 1): (0.035, -0.02),
        (2, 2): (-0.045, 0.035),
        (3, 2): (0.025, 0.02),
    }
    param = {
        "original_img_size": (args.width, args.height),
        "rotation": 0.0,
        "rotation2": 0.0,
        "flip_mode": 0,
        "matrix": np.eye(3),
        "disp_info": (0, 0, size, size, 1.0),
    }
    tcg_info = params.param_to_tcg_info(param)

    start = time.perf_counter()
    mesh_map_x, mesh_map_y = calculate_mesh_mls_coarse_map(
        size,
        size,
        mesh_size,
        control_points,
        tcg_info=tcg_info,
        grid_step=args.grid_step,
    )
    map_ms = (time.perf_counter() - start) * 1000.0

    source_rect = (0, 0, size, size)
    draw = min(args.canvas_width, args.canvas_height)
    offset_x = (args.canvas_width - draw) // 2
    offset_y = (args.canvas_height - draw) // 2
    fused_kwargs = dict(
        matrix=transform_matrix,
        source_rect=source_rect,
        transform_width=size,
        transform_height=size,
        canvas_width=args.canvas_width,
        canvas_height=args.canvas_height,
        draw_width=draw,
        draw_height=draw,
        offset_x=offset_x,
        offset_y=offset_y,
        transform_type=transform_type,
        interpolation=args.interpolation,
        border_mode="constant",
        mesh_map_x=mesh_map_x,
        mesh_map_y=mesh_map_y,
    )

    for _ in range(args.warmup):
        image_transform_adapter.transform_crop_to_canvas(image, **fused_kwargs)

    fused_times = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        image_transform_adapter.transform_crop_to_canvas(image, **fused_kwargs)
        fused_times.append((time.perf_counter() - start) * 1000.0)

    two_pass_times = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        transformed = image_transform_adapter.transform_to_canvas(
            image,
            transform_matrix,
            size,
            size,
            transform_type=transform_type,
            interpolation="linear",
            border_mode="constant",
        )
        # mesh_map_* は変位なので拡大後に identity を足し戻す（fused 側と同じ扱い）。
        full_map_x = cv2.resize(mesh_map_x, (size, size), interpolation=cv2.INTER_CUBIC)
        full_map_y = cv2.resize(mesh_map_y, (size, size), interpolation=cv2.INTER_CUBIC)
        full_map_x += np.arange(size, dtype=np.float32)[None, :]
        full_map_y += np.arange(size, dtype=np.float32)[:, None]
        meshed = cv2.remap(
            transformed,
            full_map_x,
            full_map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        image_transform_adapter.fit_crop_to_canvas(
            meshed,
            source_rect,
            args.canvas_width,
            args.canvas_height,
            draw,
            draw,
            offset_x,
            offset_y,
            args.interpolation,
        )
        two_pass_times.append((time.perf_counter() - start) * 1000.0)

    fused_avg = float(np.mean(fused_times))
    two_pass_avg = float(np.mean(two_pass_times))
    print(f"mesh_map_ms={map_ms:.3f}")
    print(f"fused_mesh_ms={fused_avg:.3f}")
    print(f"two_pass_mesh_ms={two_pass_avg:.3f}")
    print(f"render_speedup={two_pass_avg / fused_avg:.2f}x")


if __name__ == "__main__":
    main()
