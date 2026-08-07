"""
マスク Mesh warp の core ロジック。Kivy 側 (widgets/mask_editor2.py CompositMask) と
Headless 側 (cores/mask2/headless_masks.py HeadlessCompositMask) の両方から共用する。

TPS で composit ラスタを変形する。座標系は TCG 正規化値 [-0.5, +0.5]。
画像 mesh と同じ原画像 px スケールで TPS を学習し、最後に composit (texture-px) 用に
map をスケール変換して cv2.remap する。
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict

import numpy as np
import cv2
from scipy.interpolate import Rbf

import effects
from cores.distortion_correction.warp_correction import (
    outer_ring_pins_tcg, build_rbf_kwargs, coarse_axis_coords, _mls_affine_map,
)

_WARP_MAP_CACHE_MAX = 32
_warp_map_cache = OrderedDict()


def _cache_array_key(arr, decimals=6):
    return np.round(np.asarray(arr, dtype=np.float64), decimals=decimals).tobytes()


def _get_cached_maps(key):
    item = _warp_map_cache.get(key)
    if item is None:
        return None
    _warp_map_cache.move_to_end(key)
    return item


def _put_cached_maps(key, maps):
    _warp_map_cache[key] = maps
    _warp_map_cache.move_to_end(key)
    while len(_warp_map_cache) > _WARP_MAP_CACHE_MAX:
        _warp_map_cache.popitem(last=False)


def normalize_mesh_cps(raw_cps):
    """シリアライズ時に文字列キー (\"r,c\") になっている可能性のある制御点辞書を
    (r, c) tuple キーに正規化する。空 dict / 変形ゼロ要素は除外。"""
    if not raw_cps:
        return {}
    out = {}
    for k, v in raw_cps.items():
        if isinstance(k, str):
            try:
                parts = k.strip("()").split(",")
                key = (int(parts[0]), int(parts[1]))
            except Exception:
                continue
        else:
            try:
                key = (int(k[0]), int(k[1]))
            except Exception:
                continue
        try:
            ox, oy = float(v[0]), float(v[1])
        except Exception:
            continue
        if abs(ox) > 1e-6 or abs(oy) > 1e-6:
            out[key] = (ox, oy)
    return out


def mesh_cps_hash_key(raw_cps):
    """制御点 dict を hash 可能な tuple に正規化する (キャッシュキー用)。
    キー型 (str / tuple / list) を吸収し、ソート済 tuple を返す。"""
    if not raw_cps:
        return ()
    items = []
    for k, v in raw_cps.items():
        if isinstance(k, str):
            try:
                parts = k.strip("()").split(",")
                kt = (int(parts[0]), int(parts[1]))
            except Exception:
                continue
        else:
            try:
                kt = (int(k[0]), int(k[1]))
            except Exception:
                continue
        try:
            vt = (float(v[0]), float(v[1]))
        except Exception:
            continue
        items.append((kt, vt))
    items.sort()
    return tuple(items)


def warp_mask_tps(composit, mesh_size, cps, orig_img_size,
                  tcg_to_texture_fn=None, tcg_to_full_fn=None,
                  output_shape=None, source_origin_tex=(0.0, 0.0),
                  bounds_only=False):
    """マスクラスタ (composit, texture 空間) を mesh CP で変形する。

    **座標系の分解 (拡大表示・射影補正で破綻しないために重要):**
      tcg_to_texture = D_disp ∘ F
        F      = center_rotate(c) + imax   … 画像 geometry (回転 + matrix=射影を含む)。
                 画像 mesh (warp_mesh / tcg_to_ref_image) が変形する空間そのもの。
        D_disp = (F - disp01) * scale + offset … crop/zoom 部。**アフィン (等方スケール+平行移動)**。

    変形場 (MLS/RBF) は **F 空間で構築** する:
      - 全 CP + 外周 pin が F 空間に存在 → 拡大表示でも安定 (texture 空間で直接組むと
        可視外 CP が抜けて map が折り返し、変形部マスクが 0/1 に潰れる)。
      - F は画像 mesh と同一座標 → 連動コピーが射影/回転込みで厳密一致。
    composit は texture 空間なので、アフィンな D_disp で共役する:
        texture_map(t) = D_disp( MLS_F( D_disp^-1(t) ) )
    D_disp はアフィンなので 3 点で厳密同定でき、Jacobian 符号も保たれ fold しない。
    射影成分は F 内に閉じるので、射影補正があっても位置がズームでズレない。

    Args:
        tcg_to_texture_fn: callable(cx_px,cy_px)->(tex_x,tex_y)  画像中心px→texture px (D_disp∘F)
        tcg_to_full_fn:    callable(cx_px,cy_px)->(full_x,full_y) 画像中心px→フル画像px (F)
            両方与えられたとき F 空間 + アフィン共役。無いと正方 letterbox にフォールバック。
    """
    src_h, src_w = composit.shape[:2]
    if output_shape is None:
        h, w = src_h, src_w
    else:
        h, w = int(output_shape[0]), int(output_shape[1])
    _t_start = time.perf_counter()
    rows, cols = mesh_size
    if not orig_img_size or max(orig_img_size) <= 0:
        return None if bounds_only else composit
    orig_w, orig_h = float(orig_img_size[0]), float(orig_img_size[1])
    imax = max(orig_w, orig_h) / 2.0
    image_dim = max(orig_w, orig_h)

    _dbg = _os_getenv_debug()
    if _dbg and not bounds_only:
        logging.warning(
            "[MASK_WARP] composit.shape=%s output=%s src_origin=%s orig=%s imax=%s mesh_size=%s n_cps=%d cps=%s has_fn=%s",
            composit.shape, (h, w), source_origin_tex, orig_img_size, imax, mesh_size, len(cps),
            dict(list(cps.items())[:3]), (tcg_to_texture_fn is not None and tcg_to_full_fn is not None),
        )

    # D_disp (F 空間 -> texture 空間, アフィン) を 3 点で同定する。
    # F が射影でも D_disp はアフィンなので、F(c_i)->texture(c_i) の対応 3 組で厳密。
    Adisp = None
    if tcg_to_texture_fn is not None and tcg_to_full_fn is not None:
        try:
            cs = [(0.0, 0.0), (float(orig_w), 0.0), (0.0, float(orig_h))]  # 非退化な3点
            Fp = np.array([tcg_to_full_fn(cx, cy) for cx, cy in cs], dtype=np.float64)     # (3,2)
            Tp = np.array([tcg_to_texture_fn(cx, cy) for cx, cy in cs], dtype=np.float64)  # (3,2)
            Fmat = np.column_stack([Fp, np.ones(3)])      # (3,3): [Fx,Fy,1]
            Finv = np.linalg.inv(Fmat)
            ax = Finv @ Tp[:, 0]      # T_x = ax·[Fx,Fy,1]
            ay = Finv @ Tp[:, 1]
            Adisp = np.array([[ax[0], ax[1]], [ay[0], ay[1]]], dtype=np.float64)  # 2x2
            bdisp = np.array([ax[2], ay[2]], dtype=np.float64)                    # 2
            Adisp_inv = np.linalg.inv(Adisp)
        except Exception:
            Adisp = None  # 同定失敗 → フォールバック

    if Adisp is not None:
        # CP/pin は F 空間 (= 画像 mesh と同一) に配置 → 安定 & 連動コピー一致
        def _to_model(norm_x, norm_y):
            return tcg_to_full_fn(norm_x * orig_w, norm_y * orig_h)
        def _grid_to_model(gx, gy):
            # texture grid -> F 空間 (D_disp^-1, アフィン逆)
            dx = gx - bdisp[0]; dy = gy - bdisp[1]
            return (Adisp_inv[0, 0] * dx + Adisp_inv[0, 1] * dy,
                    Adisp_inv[1, 0] * dx + Adisp_inv[1, 1] * dy)
        def _model_to_tex(mx, my):
            # F 空間 source -> texture (D_disp, アフィン)
            return (Adisp[0, 0] * mx + Adisp[0, 1] * my + bdisp[0],
                    Adisp[1, 0] * mx + Adisp[1, 1] * my + bdisp[1])
    else:
        # フォールバック: disp_info 無し・正方 letterbox (= 旧挙動, stand-alone test 用)。
        sx_scale = (w - 1) / (image_dim - 1) if image_dim > 1 else 1.0
        sy_scale = (h - 1) / (image_dim - 1) if image_dim > 1 else 1.0
        def _to_model(norm_x, norm_y):
            return ((norm_x * orig_w + imax) * sx_scale,
                    (norm_y * orig_h + imax) * sy_scale)
        def _grid_to_model(gx, gy):
            return gx, gy            # grid は既に model(=texture) 空間
        def _model_to_tex(mx, my):
            return mx, my

    src_pts = []
    dst_pts = []
    for r in range(rows + 1):
        for c in range(cols + 1):
            tx_norm = -0.5 + c / cols
            ty_norm = -0.5 + r / rows
            sx, sy = _to_model(tx_norm, ty_norm)
            off_x_norm, off_y_norm = cps.get((r, c), (0.0, 0.0))
            dx, dy = _to_model(tx_norm + off_x_norm, ty_norm + off_y_norm)
            src_pts.append((sx, sy))
            dst_pts.append((dx, dy))

    # 外周 pin (CP grid 外側を固定して外挿ドリフトを抑止)。model 空間へ同じ変換で写す。
    for tcg_x, tcg_y in outer_ring_pins_tcg():
        px, py = _to_model(tcg_x, tcg_y)
        src_pts.append((px, py))
        dst_pts.append((px, py))

    if all(s == d for s, d in zip(src_pts, dst_pts)):
        if bounds_only:
            return (0.0, 0.0, float(w - 1), float(h - 1))
        return composit

    src_arr = np.array(src_pts, dtype=np.float64)
    dst_arr = np.array(dst_pts, dtype=np.float64)

    # coarse grid は画像 mesh と同じ「フル画像正方形」基準で作る。
    # crop 内だけのローカル grid にすると cv2.resize の補間位相が画像 warp とズレ、
    # 拡大表示時に mask だけ 10px 前後ズレることがある。フル基準 coarse map は
    # CP/mesh が同じなら viewport に依存しないのでキャッシュし、viewport ごとには
    # その coarse map を crop 座標でサンプルする。
    grid_step = 32
    rbf_kw = build_rbf_kwargs(orig_img_size, mesh_size)
    fn = rbf_kw.get('function', 'thin_plate')
    coarse_dim = int(round(image_dim))
    # サンプル位置は coarse map を読む側 (cv2.remap / cv2.resize の half-pixel 規約)
    # と厳密に一致させる。画像 mesh 側と同じ coarse_axis_coords を使う。
    if Adisp is not None:
        grid_w = max(2, int((coarse_dim + grid_step - 1) // grid_step))
        grid_h = max(2, int((coarse_dim + grid_step - 1) // grid_step))
        coarse_x_coords = coarse_axis_coords(coarse_dim, grid_w)
        coarse_y_coords = coarse_axis_coords(coarse_dim, grid_h)
        cnx, cny = np.meshgrid(coarse_x_coords, coarse_y_coords)
    else:
        grid_w = max(2, int((w + grid_step - 1) // grid_step))
        grid_h = max(2, int((h + grid_step - 1) // grid_step))
        coarse_x_coords = coarse_axis_coords(w, grid_w)
        coarse_y_coords = coarse_axis_coords(h, grid_h)
        tex_gx, tex_gy = np.meshgrid(coarse_x_coords, coarse_y_coords)
        cnx, cny = _grid_to_model(tex_gx, tex_gy)   # model 空間の query 点

    coarse_cache_key = (
        "coarse_model_disp",
        (coarse_dim, coarse_dim) if Adisp is not None else (w, h),
        tuple(mesh_size),
        tuple(orig_img_size),
        grid_step,
        fn,
        tuple(sorted((k, v) for k, v in rbf_kw.items())),
        _cache_array_key(src_arr),
        _cache_array_key(dst_arr),
    )
    final_cache_key = (
        "final_texture",
        (w, h),
        coarse_cache_key,
        _cache_array_key(Adisp if Adisp is not None else np.eye(2)),
        _cache_array_key(bdisp if Adisp is not None else np.zeros(2)),
    )
    cached_maps = _get_cached_maps(final_cache_key)
    if cached_maps is not None:
        map_x, map_y = cached_maps
        cache_hit = True
    else:
        cache_hit = False
        # coarse map は絶対座標ではなく model 空間の **変位** で持つ。絶対座標を
        # bicubic (Keys a=-0.75) で拡大すると 1 次多項式が再現されず、ほぼ identity な
        # map 全体に coarse セル周期の波が乗り、変形していない領域まで歪む
        # (画像 mesh 側 upsample_coarse_mesh_map と同じ理由)。
        cached_coarse = _get_cached_maps(coarse_cache_key)
        if cached_coarse is not None:
            dx_model, dy_model = cached_coarse
        else:
            if fn == 'mls':
                # MLS affine (フル画像 px の安定空間で解く → fold しない)
                sx_model, sy_model = _mls_affine_map(src_arr, dst_arr, cnx, cny)
            else:
                try:
                    rbf_x = Rbf(dst_arr[:, 0], dst_arr[:, 1], src_arr[:, 0], **rbf_kw)
                    rbf_y = Rbf(dst_arr[:, 0], dst_arr[:, 1], src_arr[:, 1], **rbf_kw)
                except Exception:
                    logging.exception("mask RBF train failed; skipping warp")
                    return composit
                sx_model = rbf_x(cnx.ravel(), cny.ravel()).reshape(grid_h, grid_w)
                sy_model = rbf_y(cnx.ravel(), cny.ravel()).reshape(grid_h, grid_w)
            dx_model = (sx_model - cnx).astype(np.float32)
            dy_model = (sy_model - cny).astype(np.float32)
            _put_cached_maps(coarse_cache_key, (dx_model, dy_model))

        if Adisp is not None:
            out_x, out_y = np.meshgrid(
                np.arange(w, dtype=np.float32),
                np.arange(h, dtype=np.float32),
            )
            full_x, full_y = _grid_to_model(out_x, out_y)
            # cv2.resize(coarse, (coarse_dim, coarse_dim), INTER_CUBIC) と同じ
            # half-pixel 変換で coarse map を直接サンプルする。
            sample_x = ((full_x + 0.5) * (grid_w / float(coarse_dim)) - 0.5).astype(np.float32)
            sample_y = ((full_y + 0.5) * (grid_h / float(coarse_dim)) - 0.5).astype(np.float32)
            src_model_x = cv2.remap(
                dx_model, sample_x, sample_y,
                cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
            ) + full_x
            src_model_y = cv2.remap(
                dy_model, sample_x, sample_y,
                cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
            ) + full_y
            map_x, map_y = _model_to_tex(src_model_x, src_model_y)
        else:
            # フォールバックでは model == texture 空間 (_grid_to_model / _model_to_tex は
            # 恒等) なので、変位をそのまま texture 解像度へ拡大して座標を足し戻す。
            map_x = cv2.resize(dx_model, (w, h), interpolation=cv2.INTER_CUBIC)
            map_y = cv2.resize(dy_model, (w, h), interpolation=cv2.INTER_CUBIC)
            map_x = map_x + np.arange(w, dtype=np.float32)[None, :]
            map_y = map_y + np.arange(h, dtype=np.float32)[:, None]
        map_x = np.asarray(map_x, dtype=np.float32)
        map_y = np.asarray(map_y, dtype=np.float32)
        _put_cached_maps(final_cache_key, (map_x, map_y))

    if _dbg and not bounds_only:
        cy, cx = h // 2, w // 2
        logging.warning(
            "[MASK_WARP] composit=%dx%d image_dim=%.0f grid=%dx%d step=%d cache=%s "
            "map_x range=[%.2f, %.2f] map_y range=[%.2f, %.2f] center_shift=(%.2f, %.2f)",
            w, h, image_dim, grid_w, grid_h, grid_step, "hit" if cache_hit else "miss",
            float(map_x.min()), float(map_x.max()),
            float(map_y.min()), float(map_y.max()),
            float(map_x[cy, cx]) - cx, float(map_y[cy, cx]) - cy,
        )

    if bounds_only:
        return (
            float(np.nanmin(map_x)),
            float(np.nanmin(map_y)),
            float(np.nanmax(map_x)),
            float(np.nanmax(map_y)),
        )

    # BORDER_REPLICATE: composit 端の外を参照する画素は端値を複製する。
    # BORDER_CONSTANT=0 だと export 時に画像端へ「透明な歪んだ跡」(0 を引っ張った
    # streak) が出るため、端のマスク値を延長して streak を消す。
    result = cv2.remap(
        composit,
        map_x - float(source_origin_tex[0]),
        map_y - float(source_origin_tex[1]),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if _dbg:
        logging.warning(
            "[MASK_WARP] done composit=%dx%d grid=%dx%d cache=%s elapsed_ms=%.1f",
            w, h, grid_w, grid_h, "hit" if cache_hit else "miss",
            (time.perf_counter() - _t_start) * 1000.0,
        )
    return result


def _os_getenv_debug():
    import os as _os
    return _os.getenv("PLATYPUS_DEBUG_MESH_WARP", "0").strip().lower() in ("1", "true", "yes", "on")


def apply_mask_mesh_warp(composit, effects_param, orig_img_size,
                         tcg_to_texture_fn=None, tcg_to_full_fn=None,
                         output_shape=None, source_origin_tex=(0.0, 0.0)):
    """Composit ラスタに mask Mesh warp を適用する高レベル API。
    mask_mesh_control_points が空なら no-op で同じ配列を返す。
    tcg_to_texture_fn / tcg_to_full_fn は editor/ctx の tcg_to_texture / tcg_to_full_image
    を渡す (前者=disp_info込み texture px, 後者=フル画像px = MLS構築空間)。
    例外発生時は warp 前の composit を返して合成自体を守る。"""
    raw_cps = effects.Mask2Effect.get_param(effects_param, 'mask_mesh_control_points')
    cps = normalize_mesh_cps(raw_cps)
    if not cps:
        return composit
    raw_size = effects.Mask2Effect.get_param(effects_param, 'mask_mesh_size')
    try:
        mesh_size = (int(raw_size[0]), int(raw_size[1]))
    except Exception:
        mesh_size = (4, 4)
    try:
        return warp_mask_tps(composit, mesh_size, cps, orig_img_size,
                             tcg_to_texture_fn, tcg_to_full_fn,
                             output_shape=output_shape,
                             source_origin_tex=source_origin_tex)
    except Exception:
        logging.exception("mask mesh warp failed; returning unwarped composit")
        return composit


def mask_mesh_source_bounds(effects_param, orig_img_size,
                            tcg_to_texture_fn=None, tcg_to_full_fn=None,
                            output_shape=None):
    """output texture を warp するときに参照される source texture 範囲を返す。

    戻り値は (min_x, min_y, max_x, max_y) in texture px。CP が空なら None。
    CompositMask 側で子マスクを描く source viewport を必要最小限に広げるために使う。
    """
    raw_cps = effects.Mask2Effect.get_param(effects_param, 'mask_mesh_control_points')
    cps = normalize_mesh_cps(raw_cps)
    if not cps or output_shape is None:
        return None
    raw_size = effects.Mask2Effect.get_param(effects_param, 'mask_mesh_size')
    try:
        mesh_size = (int(raw_size[0]), int(raw_size[1]))
    except Exception:
        mesh_size = (4, 4)
    try:
        dummy = np.zeros((1, 1), dtype=np.float32)
        return warp_mask_tps(
            dummy, mesh_size, cps, orig_img_size,
            tcg_to_texture_fn, tcg_to_full_fn,
            output_shape=output_shape,
            bounds_only=True,
        )
    except Exception:
        logging.exception("mask mesh source bounds failed")
        return None
