import cv2
import numpy as np
import math


def create_ghost(
    image: np.ndarray,  # RGB float32 (0.0-1.0)
    light_source_coords: list[tuple[int, int]],
    global_intensity: float = 0.4,  # 全体的なゴーストの強度
    vintage_amount: float = 0.0,  # 0=虹色 / 1=オールドレンズ風(淡い色被り+ベーリングフレア)。連続ブレンド
    vintage_hue: float = 0.0,  # ビンテージ色の色相回転(度)。vintage_amount が大きい時の色味を変える
    base_radius: int = 80,
    num_components: int = 1,  # ゴーストの構成要素（同心リング）の数
    blur_sigma: float = 3.0,  # 各コンポーネントのぼかし
    chromatic_aberration_strength: float = 0.6,  # 色収差(虹色)の幅。小さいほど自然
    ghost_ring_thickness: float = 0.35,  # ゴースト環の相対的な厚み (小さいほど細い)
    # --- レンズ特性 ---
    lens_center: tuple[int, int] = None,  # レンズの中心座標 (デフォルトは画像中心)
    max_eccentricity: float = 0.0,  # 楕円度 (0=真円, 0.95=細長い)
    offset_ratio: float = 2.0,  # ゴースト位置(光源と中心を結ぶ線上): 0=光源 / 1=中心 / 2=反対側(鏡像)
    rotation_angle: float = 0.0,  # 楕円の回転角度 (度, 内部用)
    perspective_distortion: float = 0.0,  # 遠近によるリングの歪み
    source_fade: float = 0.6,  # 光源側の縁の薄さ (0=均一 / 1=光源側を最大まで薄く)。形は不変
    spherical_aberration_strength: float = 0.0,  # 球面収差によるにじみ
    # --- ギザギザ(輪郭を角度ノイズで変形) ---
    spike_strength: float = 0.3,  # ギザギザの振幅 (0=滑らかな円 / 大=ウニ状)
    spike_density: int = 600,  # ギザギザの歯数 (角度方向の周波数)。0=トゲ無し
    spike_randomness: float = 0.6,  # トゲの不規則さ (0=均一に外向き / 1=始点(内側含む)と長さをバラバラに)
    ghost_tail_strength: float = 0.0,  # ゴーストの尾の強さ (内部用)
    random_seed: int = None,  # 乱数シード
) -> np.ndarray:
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / 255.0
    elif image.dtype != np.float32:
        image = image.astype(np.float32)

    img_height, img_width = image.shape[:2]
    ghosted_image = image.copy()

    rng = np.random.default_rng(random_seed)

    if lens_center is None:
        lens_center = (img_width // 2, img_height // 2)

    # 虹色(色収差リング)の定義。実写に馴染むよう既定で低彩度(輝度方向へ40%寄せ)にする。
    _rainbow_pure = np.array([
        [1.0, 0.0, 0.0],    # 赤
        [1.0, 0.5, 0.0],    # 橙
        [1.0, 1.0, 0.0],    # 黄
        [0.0, 1.0, 0.0],    # 緑
        [0.0, 0.0, 1.0],    # 青
        [0.5, 0.0, 1.0],    # 藍
        [1.0, 0.0, 1.0],    # 紫
    ], dtype=np.float32)
    _rainbow_lum = _rainbow_pure.mean(axis=1, keepdims=True)
    rainbow_colors_rgb = (_rainbow_pure * 0.6 + _rainbow_lum * 0.4).astype(np.float32)

    # ビンテージ(無コートレンズ)風の色被り。vintage_hue(度)で色相を回転して色味を変えられる。
    def _hue_rot(rgb, deg):
        if deg % 360.0 == 0.0:
            return rgb
        import colorsys
        h, l, s = colorsys.rgb_to_hls(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        h = (h + deg / 360.0) % 1.0
        return colorsys.hls_to_rgb(h, l, s)

    vintage_core_rgb = np.array(_hue_rot((1.0, 0.85, 0.65), vintage_hue), dtype=np.float32).reshape(1, 1, 3)          # 暖色アンバー芯
    vintage_inner_fringe_rgb = np.array(_hue_rot((1.0, 0.55, 0.85), vintage_hue), dtype=np.float32).reshape(1, 1, 3)  # 内側: マゼンタ
    vintage_outer_fringe_rgb = np.array(_hue_rot((0.65, 1.0, 0.75), vintage_hue), dtype=np.float32).reshape(1, 1, 3)  # 外側: グリーン

    for sx, sy in light_source_coords:
        # ゴーストのオフセット。X/Y 共通の offset_ratio を使うので、ゴーストは必ず
        # 「光源とレンズ中心を結ぶ線上」を動く(負の値も許容)。
        base_offset_x = (lens_center[0] - sx) * offset_ratio
        base_offset_y = (lens_center[1] - sy) * offset_ratio

        current_light_source_ghost = np.zeros_like(image)
        ux0, uy0, ux1, uy1 = img_width, img_height, 0, 0  # 全コンポーネントの合併ROI(合成範囲)

        for i in range(num_components):
            component_radius = base_radius * (0.8 + i * 0.1)

            component_offset_x = base_offset_x * (1.0 + i * 0.2)
            component_offset_y = base_offset_y * (1.0 + i * 0.2)
            ghost_center_x = sx + component_offset_x
            ghost_center_y = sy + component_offset_y

            current_eccentricity = float(np.clip(max_eccentricity, 0.0, 0.95))
            major_axis = component_radius
            minor_axis = max(component_radius * (1.0 - current_eccentricity), 1.0)

            theta_rad = math.radians(rotation_angle)
            cos_theta = math.cos(theta_rad)
            sin_theta = math.sin(theta_rad)

            # --- ROI(バウンディングボックス)算出 ---
            # ゴースト(リング本体+トゲ)は ghost_center まわり、楕円長軸 major_axis の
            # (1+reach) 倍までしか及ばない。全画面ではなくこの局所領域だけ計算する(高速化)。
            # 全画面と数値的に等価(領域外は 0、blur マージン込みで境界も一致)。
            reach = ghost_ring_thickness * (1.0 + 4.0 * spike_strength)  # トゲ外向き到達(外側span)
            comp_blur = blur_sigma * (1.0 + i * 0.5)
            roi_half = major_axis * (1.0 + reach) * 1.15 + comp_blur * 3.0 + 8.0
            x0 = int(max(0, math.floor(ghost_center_x - roi_half)))
            x1 = int(min(img_width, math.ceil(ghost_center_x + roi_half)))
            y0 = int(max(0, math.floor(ghost_center_y - roi_half)))
            y1 = int(min(img_height, math.ceil(ghost_center_y + roi_half)))
            if x1 <= x0 or y1 <= y0:
                continue
            ux0, uy0, ux1, uy1 = min(ux0, x0), min(uy0, y0), max(ux1, x1), max(uy1, y1)
            x_coords, y_coords = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))

            dx_map = x_coords - ghost_center_x
            dy_map = y_coords - ghost_center_y

            if ghost_tail_strength > 0:
                dist_from_source = np.sqrt((x_coords - sx) ** 2 + (y_coords - sy) ** 2)
                tail_factor_map = np.clip((dist_from_source / (base_radius * 2.0)) * ghost_tail_strength, 0.0, 1.0)
                nz = dist_from_source > 0
                tox = np.zeros_like(dx_map); toy = np.zeros_like(dy_map)
                tox[nz] = -(x_coords[nz] - sx) / dist_from_source[nz] * component_radius * 0.2 * tail_factor_map[nz]
                toy[nz] = -(y_coords[nz] - sy) / dist_from_source[nz] * component_radius * 0.2 * tail_factor_map[nz]
                dx_map = dx_map + tox
                dy_map = dy_map + toy

            rotated_dx_map = dx_map * cos_theta + dy_map * sin_theta
            rotated_dy_map = -dx_map * sin_theta + dy_map * cos_theta

            dist_norm_map = np.sqrt((rotated_dx_map / major_axis) ** 2 + (rotated_dy_map / minor_axis) ** 2)

            # 遠近歪み
            if perspective_distortion > 0:
                pf = 1.0 - (dist_norm_map - 1.0) * perspective_distortion * np.sign(dist_norm_map - 1.0)
                dist_norm_map = dist_norm_map * np.clip(pf, 0.5, 1.5)

            # 球面収差(リング半径を僅かにランダムシフト)
            if spherical_aberration_strength > 0:
                dist_norm_map = dist_norm_map + spherical_aberration_strength * (dist_norm_map - 1.0) * rng.uniform(-0.5, 0.5)

            ring_distance = np.abs(dist_norm_map - 1.0)
            # リング幅は ghost_ring_thickness で厳密に決める(chromatic で広げない)。以前は
            # *(0.5+0.5*chromatic) で chromatic を上げると幅が広がり、外側に紫の大輪が出ていた。
            # chromatic は下の色相分布(虹色の出方)だけを制御する。
            thickness_scale = np.maximum(0.0, 1.0 - ring_distance / max(ghost_ring_thickness, 1e-4))

            if not np.any(thickness_scale > 0):
                continue

            ring_out = dist_norm_map - 1.0  # 0=リング境界 / +=外側 / -=内側
            # reach はROI算出時に計算済み(トゲ外向き到達=外側span)。
            # 外側ソフト化(馴染ませ): 外側1/3あたりから外へ向けて徐々に透明にし、くっきりした外縁を消す。
            # 内側(ring_out<2/3*thickness)は不変。リング本体はしっかり、トゲは弱め(先端を残す)に掛ける。
            _fade_start = ghost_ring_thickness * (2.0 / 3.0)
            _t_out = np.clip((ring_out - _fade_start) / max(reach - _fade_start, 1e-4), 0.0, 1.0)
            outer_soft = 1.0 - 0.8 * _t_out        # リング本体
            outer_soft_fur = 1.0 - 0.3 * _t_out    # トゲ(ソフト化を弱める)

            # 色(色収差リング)
            normalized_distance_from_center = (dist_norm_map - 1.0) / ghost_ring_thickness
            color_progress = normalized_distance_from_center * chromatic_aberration_strength * 0.5 + 0.5
            color_idx_map = np.clip(color_progress * (len(rainbow_colors_rgb) - 1), 0, len(rainbow_colors_rgb) - 1)
            idx_floor = np.clip(color_idx_map.astype(int), 0, len(rainbow_colors_rgb) - 1)
            idx_ceil = (idx_floor + 1) % len(rainbow_colors_rgb)
            frac_c = (color_idx_map - idx_floor)[:, :, np.newaxis]
            color_map = rainbow_colors_rgb[idx_floor] * (1 - frac_c) + rainbow_colors_rgb[idx_ceil] * frac_c

            # ビンテージ色ブレンド
            if vintage_amount > 0.0:
                va = float(np.clip(vintage_amount, 0.0, 1.0))
                radial_sign = np.clip(normalized_distance_from_center, -1.0, 1.0)
                fringe_color = np.where(radial_sign[:, :, np.newaxis] < 0, vintage_inner_fringe_rgb, vintage_outer_fringe_rgb)
                fringe_w = np.clip(np.abs(radial_sign) * 0.5 * chromatic_aberration_strength, 0.0, 1.0)[:, :, np.newaxis]
                vintage_color_map = vintage_core_rgb * (1.0 - fringe_w) + fringe_color * fringe_w
                color_map = color_map * (1.0 - va) + vintage_color_map * va
                desat = va * 0.4
                color_lum = np.mean(color_map, axis=2, keepdims=True)
                color_map = color_map * (1.0 - desat) + color_lum * desat

            # 環の外縁フェード
            edge_fade_width = ghost_ring_thickness * 0.3
            edge_fade_factor = np.where(
                ring_distance < ghost_ring_thickness - edge_fade_width,
                1.0,
                np.maximum(0.0, (ghost_ring_thickness - ring_distance) / edge_fade_width),
            )

            # 方向性の減衰(クレッセント): 光源側を薄く、反対側を濃く。source_fade は「濃さ」だけ線形に
            # 制御し、勾配の形・リングの大きさは変えない。
            gx = ghost_center_x - sx
            gy = ghost_center_y - sy
            gmag = math.hypot(gx, gy)
            if gmag > 1e-6:
                ux = gx / gmag
                uy = gy / gmag
                proj = ((x_coords - ghost_center_x) * ux + (y_coords - ghost_center_y) * uy) / max(major_axis, 1.0)
                crescent = np.clip(0.5 + 0.5 * proj, 0.0, 1.0)  # 光源側=0 / 反対側=1
                sf = float(np.clip(source_fade, 0.0, 1.0))
                decay_factor_map = 1.0 - sf * (1.0 - crescent)
            else:
                decay_factor_map = np.ones_like(dx_map)

            component_intensity_decay = math.exp(-i * 0.2)
            alpha_map = np.clip(
                thickness_scale * edge_fade_factor * decay_factor_map * outer_soft * global_intensity * component_intensity_decay,
                0.0, 1.0,
            )

            component_ghost_layer = color_map * alpha_map[:, :, np.newaxis]

            # --- ギザギザ: リング全周から放射状に伸びる線状ノイズ(fur) ---
            # 角度方向の細いランダム線(放射状) × リング外縁からの radial フェード。全角度に分布する
            # ので「全体にかかる」。pix_ang は ghost_center まわりで平行移動不変 → offset を動かしても
            # トゲはリングと一緒に移動する。seed 固定で再現。
            if spike_strength > 0 and spike_density > 0:
                n_streaks = max(4, int(spike_density))
                sv = rng.random(n_streaks)   # 各ウェッジの濃さ
                lf = rng.random(n_streaks)   # 各ウェッジの長さ係数
                so = rng.random(n_streaks)   # 各ウェッジの始点係数
                r = float(np.clip(spike_randomness, 0.0, 1.0))
                pix_ang = np.arctan2(dy_map, dx_map)
                idx = (((pix_ang + math.pi) / (2.0 * math.pi)) * n_streaks).astype(int) % n_streaks
                # ring_out / reach は上で計算済み(外側ソフト化と共有)。
                # spike_randomness(r): 「均一なサンバースト(r=0)」→「不規則なトゲ(r=1)」。
                # 濃さ・長さ・内側食い込みの“ばらつき量”を r で増やす。濃さのばらつきは blur が
                # かかっても残り角度方向の明暗差として見えるので、r を上げると明確に乱れる。
                inten = (1.0 - r) + r * np.clip((sv[idx] - 0.15) / 0.85, 0.0, 1.0) ** 1.3   # 濃さのばらつき
                tip = reach * (1.0 - r + r * (0.15 + 1.6 * lf[idx]))                          # 外向き長さのばらつき
                back = ghost_ring_thickness * (1.0 + r) + reach * 0.9 * so[idx] * r           # 内側: 内縁まで+rで更に内へ
                out_part = np.clip(1.0 - ring_out / np.maximum(tip, 1e-4), 0.0, 1.0)   # ring_out>=0: 外へ
                in_part = np.clip(1.0 + ring_out / np.maximum(back, 1e-4), 0.0, 1.0)   # ring_out<0: 内へ
                radial = np.where(ring_out >= 0.0, out_part, in_part)
                fur_alpha = np.clip(
                    spike_strength * inten * radial * decay_factor_map * outer_soft_fur * global_intensity * component_intensity_decay,
                    0.0, 1.0,
                )
                # トゲは中間色(暖色寄りの白)。color_map だと外側が虹色の端=紫になり「外側の紫」を助長する。
                fur_color = np.array([1.0, 0.95, 0.88], dtype=np.float32).reshape(1, 1, 3)
                component_ghost_layer = component_ghost_layer + fur_color * fur_alpha[:, :, np.newaxis]

            if comp_blur > 0:
                ksize = max(3, int(comp_blur * 2) + 1)
                if ksize % 2 == 0:
                    ksize += 1
                component_ghost_layer = cv2.GaussianBlur(component_ghost_layer, (ksize, ksize), comp_blur)

            current_light_source_ghost[y0:y1, x0:x1] += component_ghost_layer

        if ux1 <= ux0 or uy1 <= uy0:
            continue  # この光源は寄与なし

        # 合成は合併ROI内だけで行う(領域外は寄与ゼロ)。全画面のsum/clip/blendを避ける。
        sub = np.clip(current_light_source_ghost[uy0:uy1, ux0:ux1], 0.0, 1.0)
        alpha_channel = np.clip(np.sum(sub, axis=2) * global_intensity, 0.0, 1.0)[:, :, np.newaxis]
        dst = ghosted_image[uy0:uy1, ux0:ux1]
        ghosted_image[uy0:uy1, ux0:ux1] = dst * (1.0 - alpha_channel) + sub * alpha_channel

    # ベーリングフレア / 霞 (vintage_amount)
    if vintage_amount > 0.0:
        va = float(np.clip(vintage_amount, 0.0, 1.0))
        lum = np.mean(ghosted_image, axis=2).astype(np.float32)
        sigma = max(1.0, base_radius * 0.6)
        # 大 sigma の全画面ガウスは重い。十分に滑らかなので縮小→blur→拡大で近似(高速化)。
        ds = int(np.clip(sigma / 4.0, 1, 8))
        if ds > 1:
            small = cv2.resize(lum, (max(1, img_width // ds), max(1, img_height // ds)), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma / ds)
            glow = cv2.resize(small, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
        else:
            glow = cv2.GaussianBlur(lum, (0, 0), sigmaX=sigma)
        glow = glow[:, :, np.newaxis]
        flare_color = np.array(_hue_rot((1.0, 0.92, 0.82), vintage_hue), dtype=np.float32).reshape(1, 1, 3)
        ghosted_image = ghosted_image + glow * flare_color * (va * global_intensity * 0.5)
        black_lift = va * global_intensity * 0.04
        ghosted_image = ghosted_image * (1.0 - black_lift) + black_lift

    return np.clip(ghosted_image, 0.0, 1.0)


# ============================================================================
# プリセット（単一情報源）
# ============================================================================
# main.py のプリセット選択と __main__ の両方が参照する。各キーは create_ghost() の引数名と一致。
# light_source_coords / lens_center / random_seed は実行時メタ情報(座標は 700x500 基準)。
GHOST_PRESETS = {
    # 実写によくある大きく薄い暖色リング(画面いっぱいの弧)。光源位置によらず中央に出る(offset 1.0)。
    "Big Soft Ring": {
        "global_intensity": 0.4, "vintage_amount": 0.8, "base_radius": 430, "num_components": 1,
        "blur_sigma": 5.0, "chromatic_aberration_strength": 0.3, "ghost_ring_thickness": 0.07,
        "max_eccentricity": 0.0, "offset_ratio": 1.0, "perspective_distortion": 0.0,
        "source_fade": 0.3, "spherical_aberration_strength": 0.0, "spike_strength": 0.0, "spike_density": 600, "spike_randomness": 0.6,
        "light_source_coords": [(560, 110)], "lens_center": (350, 250), "random_seed": 45,
    },
    # 大きな細い円弧で縁が虹色(スペクトル)。光源と反対側に弧が出る(offset 1.8)。
    "Rainbow Arc": {
        "global_intensity": 0.7, "vintage_amount": 0.1, "base_radius": 340, "num_components": 1,
        "blur_sigma": 1.2, "chromatic_aberration_strength": 2.2, "ghost_ring_thickness": 0.03,
        "max_eccentricity": 0.1, "offset_ratio": 1.8, "perspective_distortion": 0.0,
        "source_fade": 0.4, "spherical_aberration_strength": 0.0, "spike_strength": 0.0, "spike_density": 600, "spike_randomness": 0.6,
        "light_source_coords": [(560, 110)], "lens_center": (350, 250), "random_seed": 45,
    },
    # 暖色の小さな玉(二次ゴースト)。chromatic=0 + vintage=1 で虹色を消し暖色一色の塗り円に。
    "Warm Orb": {
        "global_intensity": 0.5, "vintage_amount": 1.0, "base_radius": 45, "num_components": 1,
        "blur_sigma": 5.0, "chromatic_aberration_strength": 0.0, "ghost_ring_thickness": 1.0,
        "max_eccentricity": 0.0, "offset_ratio": 1.5, "perspective_distortion": 0.0,
        "source_fade": 0.2, "spherical_aberration_strength": 0.0, "spike_strength": 0.0, "spike_density": 600, "spike_randomness": 0.6,
        "light_source_coords": [(540, 130)], "lens_center": (350, 250), "random_seed": 45,
    },
    # トゲのウニ状ゴースト(中サイズ・二次)。
    "Urchin": {
        "global_intensity": 0.7, "vintage_amount": 0.0, "base_radius": 120, "num_components": 2,
        "blur_sigma": 1.5, "chromatic_aberration_strength": 0.8, "ghost_ring_thickness": 0.3,
        "max_eccentricity": 0.0, "offset_ratio": 2.0, "perspective_distortion": 0.0,
        "source_fade": 0.5, "spherical_aberration_strength": 0.0, "spike_strength": 0.9, "spike_density": 600, "spike_randomness": 0.6,
        "light_source_coords": [(540, 120)], "lens_center": (350, 250), "random_seed": 104,
    },
    # オールドレンズ風の暖色(アンバー)二重ゴースト+ベーリング。
    "Vintage Amber": {
        "global_intensity": 0.5, "vintage_amount": 1.0, "base_radius": 150, "num_components": 2,
        "blur_sigma": 5.0, "chromatic_aberration_strength": 0.8, "ghost_ring_thickness": 0.4,
        "max_eccentricity": 0.2, "offset_ratio": 1.8, "perspective_distortion": 0.0,
        "source_fade": 0.5, "spherical_aberration_strength": 0.05, "spike_strength": 0.2, "spike_density": 600, "spike_randomness": 0.6,
        "light_source_coords": [(520, 160)], "lens_center": (350, 250), "random_seed": 111,
    },
}
