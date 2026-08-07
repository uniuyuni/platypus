"""
ガイド線変形補正API

メッシュワープ、ラインガイド補正、ポイントベース補正
"""

import numpy as np
import cv2
import logging
from typing import Dict, Tuple, List, Optional
from scipy.interpolate import Rbf

import params


# scipy.interpolate.Rbf で epsilon を意味のあるパラメータとして取る basis 関数。
# それ以外 (thin_plate, linear, cubic, quintic) は epsilon を無視する。
_RBF_EPSILON_BASIS = ('multiquadric', 'inverse')


def _mls_affine_map(src_px, dst_px, coarse_x_mesh, coarse_y_mesh, alpha=2.0):
    """Moving Least Squares (affine) で map_x / map_y を計算する。
    Schaefer et al. 2006 "Image Deformation Using Moving Least Squares" の affine 版。

    各 query 点 v について、制御点を距離で重み付けした **局所 affine 変換** を解く:
      w_i      = 1 / |dst_i - v|^(2*alpha)          (近い CP ほど強い)
      p* , q*  = w 加重平均 (それぞれ dst, src の重心)
      M        = argmin Σ w_i |(dst_i - p*) M - (src_i - q*)|^2   (2x2 affine)
      map(v)   = (v - p*) M + q*

    Gaussian IDW (Nadaraya-Watson) と違い、これは **補間** である:
      - CP 上 (v = dst_i) では w_i → ∞ で厳密に src_i を返す → CP付近が正しく歪む
      - グローバル affine 項を持たない (各点が独立に局所 affine) → TPS のような
        「全体内側シフト」が構造的に発生しない。CP 群から離れた領域は最寄り CP 群の
        局所 affine = ほぼ identity に収束 → 端が動かない

    Args:
        src_px / dst_px: shape (N, 2) image-px の src(元位置)/dst(変形後)座標
        coarse_x_mesh / coarse_y_mesh: shape (grid_h, grid_w)
        alpha: 重みの鋭さ (大きいほど局所的)。Schaefer の標準は 1〜2。
    Returns:
        (map_x_coarse, map_y_coarse): shape (grid_h, grid_w), dtype float64
    """
    P = np.asarray(dst_px, dtype=np.float64)   # 変形後 = query 空間
    Q = np.asarray(src_px, dtype=np.float64)   # 元位置 = sample 元 (map の値)
    grid_h, grid_w = coarse_x_mesh.shape
    G = np.stack([coarse_x_mesh.ravel(), coarse_y_mesh.ravel()], axis=1)  # (M, 2)

    # (M, N) 重み。CP に一致する点は d2≈0 → 巨大 weight → 厳密通過 (補間性)。
    diff = G[:, None, :] - P[None, :, :]           # (M, N, 2)
    d2 = np.maximum((diff * diff).sum(axis=2), 1e-8)  # (M, N)
    w = 1.0 / np.power(d2, alpha)                  # (M, N)
    wsum = w.sum(axis=1)                           # (M,)

    pstar = (w[:, :, None] * P[None]).sum(axis=1) / wsum[:, None]  # (M, 2)
    qstar = (w[:, :, None] * Q[None]).sum(axis=1) / wsum[:, None]  # (M, 2)

    Phat = P[None] - pstar[:, None, :]             # (M, N, 2)
    Qhat = Q[None] - qstar[:, None, :]             # (M, N, 2)

    # A = Σ w Phat^T Phat, B = Σ w Phat^T Qhat   (各 grid 点ごと 2x2)
    A = np.einsum('mn,mni,mnj->mij', w, Phat, Phat)  # (M, 2, 2)
    B = np.einsum('mn,mni,mnj->mij', w, Phat, Qhat)  # (M, 2, 2)

    # ridge 正則化 (1 CP が支配して A が特異化するケースを安定化)。
    # その場合 (v - p*) ≈ 0 なので map ≈ q* となり ridge の影響は無視できる。
    diagA = A[:, 0, 0] + A[:, 1, 1]                 # (M,)
    ridge = (1e-8 * np.maximum(diagA, 1e-12))[:, None, None] * np.eye(2)[None]
    M_mat = np.linalg.solve(A + ridge, B)           # (M, 2, 2)

    mapped = np.einsum('mi,mij->mj', (G - pstar), M_mat) + qstar  # (M, 2)
    map_x = mapped[:, 0].reshape(grid_h, grid_w)
    map_y = mapped[:, 1].reshape(grid_h, grid_w)
    return map_x, map_y


def coarse_axis_coords(length, count):
    """coarse map の 1 軸ぶんのサンプル位置を返す。

    coarse map を full 解像度へ拡大するのは cv2.resize (CPU) と Metal shader だが、
    どちらも「配列 index j は画素中心 (j+0.5)*length/count - 0.5 にある」という
    half-pixel 規約で読む。生成側が linspace(0, length-1, count) でサンプルすると
    規約がずれ、map 全体に線形の誤差 (grid_step=32 で ±16px、64 で ±33px) が乗って
    画像がわずかに拡大・平行移動する。読み手と同じ位置でサンプルする。
    """
    return (np.arange(int(count), dtype=np.float64) + 0.5) * (float(length) / int(count)) - 0.5


def upsample_coarse_mesh_map(disp_x_coarse, disp_y_coarse, width, height):
    """coarse な **変位** マップを full 解像度の絶対座標 map へ拡大する。

    絶対座標の map をそのまま cv2.resize すると全体が波打つ。cv2 の INTER_CUBIC
    (Keys, a=-0.75) は 1 次多項式を再現できず (重み和は 1 だが 1 次モーメントが
    ずれる)、ほぼ identity = 線形ランプである map に coarse セル周期・振幅
    ±0.047*grid_step px の波を注入するため。変形していない領域の直線まで
    波打つのはこれが原因。

    変位 (map - 座標) を拡大して identity を足し戻せば、変位 0 の領域は厳密に 0 の
    ままになり、残る誤差も変位の勾配に比例するだけになる。
    """
    dx = cv2.resize(np.asarray(disp_x_coarse, dtype=np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(np.asarray(disp_y_coarse, dtype=np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    dx += np.arange(width, dtype=np.float32)[None, :]
    dy += np.arange(height, dtype=np.float32)[:, None]
    return dx, dy


def build_rbf_kwargs(orig_img_size, mesh_size):
    """config.json の mesh_rbf_function に従って scipy.interpolate.Rbf の引数を作る。
    画像 mesh / マスク mesh 両方から呼ばれて同じ kwargs を返すので、両者は常に
    同一の RBF / 同一の epsilon で trained → 連動コピー前提の数値一致を維持する。

    scipy.interpolate.Rbf の epsilon は **multiplier** で、各 basis 関数で以下のように
    使われる (epsilon が大きいほど influence 範囲が狭まる):
      - gaussian: exp(-(epsilon*r)^2) → 影響半径 ≒ 1/epsilon (1/e で decay)
      - multiquadric: sqrt(1 + (epsilon*r)^2)
      - inverse: 1 / sqrt(1 + (epsilon*r)^2)

    「影響半径 ≒ CP 間隔」に設定 → 隣接 CP まで influence が届く、2 CP 先で ≒ 0。
    epsilon = 1 / CP間隔 = 1 / (max(orig)/max(mesh_size)) = max(mesh_size) / max(orig)。
    """
    try:
        import config
        fn = config.get_config('mesh_rbf_function')
    except Exception:
        fn = None
    if not fn:
        fn = 'thin_plate'
    kwargs = {'function': fn, 'smooth': 0}
    if fn in _RBF_EPSILON_BASIS:
        try:
            ms_max = max(int(mesh_size[0]), int(mesh_size[1]))
            orig_max = float(max(orig_img_size)) if orig_img_size else 1.0
            if ms_max > 0 and orig_max > 0:
                # 影響半径 = CP 間隔 (= orig_max / ms_max) になる epsilon
                cp_spacing = orig_max / ms_max
                kwargs['epsilon'] = 1.0 / cp_spacing
        except Exception:
            pass

    # PLATYPUS_DEBUG_MESH_WARP=1 で何の RBF が実際に使われているか確認できるよう log。
    import os as _os, logging as _logging
    if _os.getenv("PLATYPUS_DEBUG_MESH_WARP", "0").strip().lower() in ("1", "true", "yes", "on"):
        _logging.warning(
            "[RBF_KW] function=%s smooth=%s epsilon=%s orig=%s mesh_size=%s",
            kwargs.get('function'), kwargs.get('smooth'), kwargs.get('epsilon'),
            orig_img_size, mesh_size,
        )
    return kwargs


def warp_mesh_with_mapper(
    image: np.ndarray,
    mesh_size: Tuple[int, int],
    control_points: Dict[Tuple[int, int], Tuple[float, float]],
    tcg_to_pixel_fn,
    interpolation: str = 'bicubic',
    border_value=(0, 0, 0),
    extra_pin_points_tcg: Optional[List[Tuple[float, float]]] = None,
    orig_img_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    メッシュワープ補正（TPS + Coarse Grid Approximation）。
    coord-mapper 注入式で、image 側 (tcg_to_ref_image) と mask 側 (ctx.tcg_to_texture)
    のどちらでも利用できる。

    Args:
        image: numpy.ndarray、dtype=float32、shape=(H, W, 3) または (H, W)
        mesh_size: tuple、(rows, cols)
        control_points: dict
            キー: (row_index, col_index)
            値: (offset_x, offset_y)（TCG座標系のオフセット）
        tcg_to_pixel_fn: callable (tcg_x, tcg_y) -> (px, py)
            TCG 正規化座標を image 上のピクセル座標へ変換する関数
        interpolation: str、'bilinear' | 'bicubic'
        border_value: cv2.remap の borderValue。マスクなど単一チャンネルは 0 を渡す。
        extra_pin_points_tcg: 追加の固定点 (src == dst で offset=0 とする) のリスト。
            TCG 正規化座標で指定する。CP grid (TCG ±0.5) よりも外側に置くと
            画像端での TPS 外挿が抑止され、warp が局所的になる。

    Returns:
        補正後画像
    """
    # 入力検証
    _validate_image(image)

    rows, cols = mesh_size
    if not (3 <= rows <= 10 and 3 <= cols <= 10):
        # メッシュサイズは柔軟に対応するため警告に留めるか、一旦そのまま
        pass

    height, width = image.shape[:2]
    image_shape = (height, width)

    # 1. 制御点の収集 (TCG座標系)
    # Source: 元のグリッド (Regular Grid)
    # Dest: 変形後のグリッド (Deformed Grid) = Source + Offset
    # 我々は Dest(x,y) -> Source(x,y) のマッピングを求めたい

    base_coords = get_mesh_coordinates(image_shape, mesh_size) # (rows+1, cols+1, 2)

    src_points = [] # Regular
    dst_points = [] # Deformed

    for r in range(rows + 1):
        for c in range(cols + 1):
            bx, by = base_coords[r, c]
            off_x, off_y = control_points.get((r, c), (0.0, 0.0))

            src_points.append((bx, by))
            dst_points.append((bx + off_x, by + off_y))

    # 追加 pin: src == dst で TPS に渡す。CP grid 外側の領域を「ここは動かさない」
    # と TPS に教えることで、外挿による全体ドリフトを抑止する。
    if extra_pin_points_tcg:
        for px, py in extra_pin_points_tcg:
            src_points.append((px, py))
            dst_points.append((px, py))

    # 変形がない場合は早期リターン
    if all(src == dst for src, dst in zip(src_points, dst_points)):
        return image

    # 2. 座標変換 (TCG -> Image Pixel)
    src_px_list = [tcg_to_pixel_fn(px, py) for px, py in src_points]
    dst_px_list = [tcg_to_pixel_fn(px, py) for px, py in dst_points]

    src_px = np.array(src_px_list)
    dst_px = np.array(dst_px_list)

    # 3. RBF / IDW で map_x, map_y を計算 (config.json で切替)。
    # 4. Coarse Grid サンプリング
    grid_step = 32
    grid_h = (height + grid_step - 1) // grid_step
    grid_w = (width + grid_step - 1) // grid_step
    coarse_x_coords = coarse_axis_coords(width, grid_w)
    coarse_y_coords = coarse_axis_coords(height, grid_h)
    coarse_x_mesh, coarse_y_mesh = np.meshgrid(coarse_x_coords, coarse_y_coords)

    rbf_kw = build_rbf_kwargs(orig_img_size, mesh_size)
    fn = rbf_kw.get('function', 'thin_plate')

    if fn == 'mls':
        # Moving Least Squares (affine)。補間性 + 局所性を両立し、TPS の affine 遠方項
        # による「全体内側シフト」を解消する (default)。
        map_x_coarse, map_y_coarse = _mls_affine_map(
            src_px, dst_px, coarse_x_mesh, coarse_y_mesh,
        )
    else:
        # scipy.interpolate.Rbf (TPS / multiquadric / inverse / linear / cubic / quintic)
        try:
            rbf_x = Rbf(dst_px[:, 0], dst_px[:, 1], src_px[:, 0], **rbf_kw)
            rbf_y = Rbf(dst_px[:, 0], dst_px[:, 1], src_px[:, 1], **rbf_kw)
        except Exception:
            logging.exception("RBF Fitting Error")
            return image
        flat_cx = coarse_x_mesh.ravel()
        flat_cy = coarse_y_mesh.ravel()
        map_x_coarse = rbf_x(flat_cx, flat_cy).reshape(grid_h, grid_w)
        map_y_coarse = rbf_y(flat_cx, flat_cy).reshape(grid_h, grid_w)

    # 6. マップのアップスケーリング（変位を拡大して identity を足し戻す）
    map_x, map_y = upsample_coarse_mesh_map(
        map_x_coarse - coarse_x_mesh,
        map_y_coarse - coarse_y_mesh,
        width,
        height,
    )

    # [MESH_DEBUG] map の統計 (warp 量を測る、マスク mesh の同等 log と比較するため)
    import os as _os, logging as _logging
    if _os.getenv("PLATYPUS_DEBUG_MESH_WARP", "0").strip().lower() in ("1", "true", "yes", "on"):
        cy, cx = height // 2, width // 2
        shift_x = float(map_x[cy, cx]) - cx
        shift_y = float(map_y[cy, cx]) - cy
        _logging.warning(
            "[IMG_WARP] image.shape=%s n_cps=%d "
            "map_x range=[%.2f, %.2f] map_y range=[%.2f, %.2f] "
            "center_shift=(%.2f, %.2f)",
            image.shape, len(control_points),
            float(map_x.min()), float(map_x.max()),
            float(map_y.min()), float(map_y.max()),
            shift_x, shift_y,
        )

    # 7. リマッピング
    if interpolation == 'bilinear':
        interp_flag = cv2.INTER_LINEAR
    else:  # 'bicubic'
        interp_flag = cv2.INTER_CUBIC

    corrected = cv2.remap(
        image,
        map_x,
        map_y,
        interp_flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return corrected


def warp_mesh(
    image: np.ndarray,
    mesh_size: Tuple[int, int],
    control_points: Dict[Tuple[int, int], Tuple[float, float]],
    tcg_info: Dict = None,
    interpolation: str = 'bicubic',
    border_value=(0, 0, 0),
) -> np.ndarray:
    """
    メッシュワープ補正（画像 + マスク 共用 API）。warp_mesh_with_mapper のラッパで、
    tcg_to_ref_image を mapper として使用する。マスク (H, W) もそのまま渡せる。

    Args:
        image: numpy.ndarray、dtype=float32、shape=(H, W, 3) または (H, W)
        mesh_size: tuple、(rows, cols)
        control_points: dict
            キー: (row_index, col_index)
            値: (offset_x, offset_y)（TCG座標系のオフセット）
        tcg_info: 座標変換情報 (from params.param_to_tcg_info)
        interpolation: str、'bilinear' | 'bicubic'
        border_value: cv2.remap の borderValue (画像は (0,0,0)、マスクは 0)

    Returns:
        補正後画像
    """
    def _mapper(px, py):
        return params.tcg_to_ref_image(px, py, image, tcg_info)

    orig = tcg_info['original_img_size'] if (tcg_info is not None and 'original_img_size' in tcg_info) else None
    return warp_mesh_with_mapper(
        image,
        mesh_size,
        control_points,
        tcg_to_pixel_fn=_mapper,
        interpolation=interpolation,
        border_value=border_value,
        extra_pin_points_tcg=outer_ring_pins_tcg(),
        orig_img_size=orig,
    )


def calculate_mesh_mls_coarse_map(
    width: int,
    height: int,
    mesh_size: Tuple[int, int],
    control_points: Dict[Tuple[int, int], Tuple[float, float]],
    tcg_info: Dict = None,
    grid_step: int = 32,
    extra_pin_points_tcg: Optional[List[Tuple[float, float]]] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """MLS mesh warp用のcoarse mapを計算する。

    `warp_mesh()` のMLS branchと同じ src/dst 制御点を使い、画像本体をremapせずに
    coarse mapだけを返す。GPU fused previewではこのmapをsampleして中間画像生成を避ける。

    返すのは絶対座標ではなく **変位** (map - キャンバス座標)。消費側 (Metal shader /
    image_transform_reference) は bicubic で拡大した値を自身のキャンバス座標へ足す。
    絶対座標を bicubic 拡大すると画像全体が波打つため（`upsample_coarse_mesh_map`
    の説明を参照）。
    """
    rows, cols = mesh_size
    image_shape = (height, width)
    base_coords = get_mesh_coordinates(image_shape, mesh_size)

    src_points = []
    dst_points = []
    for r in range(rows + 1):
        for c in range(cols + 1):
            bx, by = base_coords[r, c]
            off_x, off_y = control_points.get((r, c), (0.0, 0.0))
            src_points.append((bx, by))
            dst_points.append((bx + off_x, by + off_y))

    if extra_pin_points_tcg is None:
        extra_pin_points_tcg = outer_ring_pins_tcg()
    if extra_pin_points_tcg:
        for px, py in extra_pin_points_tcg:
            src_points.append((px, py))
            dst_points.append((px, py))

    if all(src == dst for src, dst in zip(src_points, dst_points)):
        return None

    class DummyImage:
        def __init__(self, w, h):
            self.shape = (h, w, 3)

    dummy_img = DummyImage(width, height)

    def _mapper(px, py):
        return params.tcg_to_ref_image(px, py, dummy_img, tcg_info)

    src_px = np.array([_mapper(px, py) for px, py in src_points])
    dst_px = np.array([_mapper(px, py) for px, py in dst_points])

    grid_step = max(1, int(grid_step))
    grid_h = (height + grid_step - 1) // grid_step
    grid_w = (width + grid_step - 1) // grid_step
    coarse_x_coords = coarse_axis_coords(width, grid_w)
    coarse_y_coords = coarse_axis_coords(height, grid_h)
    coarse_x_mesh, coarse_y_mesh = np.meshgrid(coarse_x_coords, coarse_y_coords)
    map_x, map_y = _mls_affine_map(src_px, dst_px, coarse_x_mesh, coarse_y_mesh)
    disp_x = (map_x - coarse_x_mesh).astype(np.float32)
    disp_y = (map_y - coarse_y_mesh).astype(np.float32)
    return disp_x, disp_y


# デフォルト margin=0.15 用の事前計算済 pin 座標 (hot path で list allocation を避けるため)。
_OUTER_RING_PINS_DEFAULT = (
    (-0.65, -0.65), (0.0, -0.65), (0.65, -0.65),
    (-0.65,  0.0),                 (0.65,  0.0),
    (-0.65,  0.65), (0.0,  0.65), (0.65,  0.65),
)


def outer_ring_pins_tcg(margin: float = 0.15):
    """CP grid (TCG ±0.5) の外側に offset=0 で固定する pin の TCG 座標 8 点を返す。
    TPS の affine 成分による「動かしていない CP も外周方向に引っ張られる」現象を
    抑制するためのバウンダリ制約。画像 mesh / マスク mesh 共用。

    Args:
        margin: CP grid 外側へどれだけ離すか (TCG 単位)。大きいほど抑制力が強いが、
            画像領域から離れすぎると TPS 行列が ill-conditioned になる。0.15 は経験値。
    """
    if margin == 0.15:
        return _OUTER_RING_PINS_DEFAULT
    r = 0.5 + margin
    return (
        (-r, -r), (0.0, -r), (+r, -r),
        (-r,  0.0),           (+r,  0.0),
        (-r, +r), (0.0, +r), (+r, +r),
    )

def correct_with_lines(
    image: np.ndarray,
    reference_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    tcg_info: Dict = None,
    interpolation: str = 'bicubic',
    homography: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    参照線を使用して画像の透視歪みを補正する関数
    
    Lightroomのガイド付き変形(Guided Upright)機能を実装。
    ユーザーが指定した線分が垂直または水平になるようにホモグラフィ変換を適用。
    
    Parameters
    ----------
    image : np.ndarray
        入力画像 (H, W, C) または (H, W)
    reference_lines : List[Tuple[Tuple[float, float], Tuple[float, float]]]
        補正に使用する参照線のリスト。各線は ((x1, y1), (x2, y2)) の形式。
        座標は画像中心を原点とした正規化座標系 (Y軸下向きが正)。
        **これらの座標は歪んだ画像上での線の位置を示す**
        各線は補正後に垂直または水平になる必要がある。
        - 線が0本: エラー
        - 線が1本: 入力画像をそのまま返す（H=None）
        - 線が2-4本: 正常に補正
        - 線が5本以上: 最後の4本を使用
    tcg_info : Dict, optional
        座標変換情報。Noneの場合は恒等変換を仮定。
    interpolation : str, optional
        補間方法: 'nearest', 'bilinear', 'bicubic', 'lanczos'
        デフォルトは 'bicubic'
        
    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        (補正された画像, ホモグラフィ行列)
        - 補正された画像: 透視補正後の画像
        - ホモグラフィ行列 (3x3): 補正に使用した変換行列
          Noneの場合は変換なし（線が1本の場合）
          
          この行列Hは以下のように使用できます:
          - 順変換（歪んだ→補正後）: corrected_pt = H @ distorted_pt
          - 逆変換（補正後→歪んだ）: distorted_pt = H_inv @ corrected_pt
            ここで H_inv = np.linalg.inv(H)
        
    Notes
    -----
    - 線が1本だけの場合は入力画像をそのまま返す
    - 線が5本以上の場合は最後の4本のみを使用
    - 各線は垂直線または水平線として扱われる
    - 線の向きは自動的に判定される
    - ホモグラフィ行列を用いた透視変換を適用
    
    Examples
    --------
    >>> # 建物の縦の線を補正
    >>> lines = [
    ...     ((-0.3, -0.5), (-0.3, 0.5)),  # 左の縦線
    ...     ((0.3, -0.5), (0.3, 0.5))     # 右の縦線
    ... ]
    >>> corrected, H = correct_with_lines(image, lines)
    >>> 
    >>> # 逆変換（補正後の座標→元の座標）
    >>> if H is not None:
    ...     H_inv = np.linalg.inv(H)
    ...     # 補正後の点(100, 200)を元の座標に変換
    ...     pt = np.array([100, 200, 1])
    ...     original_pt = H_inv @ pt
    ...     original_pt = original_pt[:2] / original_pt[2]
    """
    
    h, w = image.shape[:2]

    if homography is not None:
        # 呼び出し側で計算・調整済み (減衰クランプ等) のホモグラフィをそのまま使う。
        H = np.asarray(homography, dtype=np.float64)
    else:
        # エッジケース: 線が0本
        if len(reference_lines) == 0:
            return image, None

        # エッジケース: 線が1本の場合は入力をそのまま返す
        if len(reference_lines) == 1:
            return image, None

        # エッジケース: 線が5本以上の場合は最後の4本のみを使用
        if len(reference_lines) > 4:
            reference_lines = reference_lines[-4:]

        H = calculate_lines_homography(reference_lines, w, h, tcg_info)

    if H is None:
        # 計算できなかった場合（線が不足など）は元の画像を返す
        # correct_with_linesの仕様上、線が1本の場合はNoneを返す
        return image, None
    
    # 補間方法の選択
    interp_flags = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'lanczos': cv2.INTER_LANCZOS4
    }
    
    if interpolation not in interp_flags:
        raise ValueError(f"未対応の補間方法: {interpolation}")
    
    # ホモグラフィ変換を適用
    corrected = cv2.warpPerspective(
        image, H, (w, h),
        flags=interp_flags[interpolation],
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return corrected, H

def calculate_lines_homography(
    reference_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    width: int,
    height: int,
    tcg_info: Dict = None
) -> Optional[np.ndarray]:
    """
    参照線からホモグラフィ行列を計算する
    
    Returns
    -------
    Optional[np.ndarray]
        ホモグラフィ行列 (3x3) または None
    """
    # エッジケース: 線が0本
    if len(reference_lines) == 0:
        return None
    
    # エッジケース: 線が1本の場合は計算不可（仕様）
    if len(reference_lines) == 1:
        return None
    
    # エッジケース: 線が5本以上の場合は最後の4本のみを使用
    if len(reference_lines) > 4:
        reference_lines = reference_lines[-4:]
    
    # 正規化座標からピクセル座標への変換
    # これらは歪んだ画像上での実際の線の位置
    # Note: tcg_to_ref_image requires an 'image' argument for shape property usually,
    # but params.tcg_to_ref_image signature is (cx, cy, ref_img, tcg_info).
    # It uses ref_img.shape. We need to mock it or update tcg_to_ref_image to check shape from tcg_info?
    # Actually params.tcg_to_ref_image uses ref_img.shape if apply_disp_info is True or for sizing.
    # In GeometryEffect context, we might not have the image object.
    # But here we pass width/height.
    # We should create a dummy object with shape property or modify tcg_to_ref_image?
    # Converting to pixel coordinates:
    # It basically maps normalized to pixel based on tcg_info['original_img_size'] and matrix.
    # Let's check params.py again.
    
    class DummyImage:
        def __init__(self, w, h):
            self.shape = (h, w, 3)
            
    dummy_img = DummyImage(width, height)

    pixel_lines = []
    for line in reference_lines:
        p1_norm, p2_norm = line
        p1_pixel = params.tcg_to_ref_image(p1_norm[0], p1_norm[1], dummy_img, tcg_info)
        p2_pixel = params.tcg_to_ref_image(p2_norm[0], p2_norm[1], dummy_img, tcg_info)
        pixel_lines.append((p1_pixel, p2_pixel))
    
    # 各線が垂直か水平かを判定し、対応点を生成
    src_points, dst_points = _generate_correspondence_points_improved(pixel_lines, width, height)
    
    if len(src_points) < 4:
        # raise ValueError("ホモグラフィ計算に十分な対応点が得られませんでした")
        return None
    
    # ホモグラフィ行列を計算
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    # 点が4つの場合は直接計算、それ以上ならRANSAC
    if len(src_points) == 4:
        H = cv2.getPerspectiveTransform(src_points, dst_points)
    else:
        H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    if H is None:
        # raise ValueError("ホモグラフィ行列の計算に失敗しました")
        return None
        
    return H

def _classify_line_orientation(p1: Tuple[float, float], p2: Tuple[float, float]) -> str:
    """
    線分が垂直線か水平線かを判定
    
    Parameters
    ----------
    p1, p2 : Tuple[float, float]
        線分の端点
        
    Returns
    -------
    str
        'vertical' または 'horizontal'
    """
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    
    # 傾きの大きい方で判定
    if dy > dx:
        return 'vertical'
    else:
        return 'horizontal'


def _generate_correspondence_points_improved(
    pixel_lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    width: int,
    height: int
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    参照線から対応点を生成（改善版）
    
    各線が補正後に垂直または水平になるように対応点を設定する
    
    重要: src_points は歪んだ画像上の実際の点
          dst_points は補正後にあるべき理想的な点
    
    改善点:
    - 垂直線と水平線を分けて処理
    - 線の長さを保持
    - 画像の中心を基準に補正
    - 垂直線と水平線を混在させた時の安定性向上
    
    Parameters
    ----------
    pixel_lines : List[Tuple[Tuple[float, float], Tuple[float, float]]]
        ピクセル座標での参照線（歪んだ画像上の位置）
    width, height : int
        画像のサイズ
        
    Returns
    -------
    Tuple[List, List]
        (元の点(歪んでいる), 目標点(垂直/水平に補正された)) のタプル
    """
    src_points = []
    dst_points = []
    
    # 垂直線と水平線を分類
    vertical_lines = []
    horizontal_lines = []
    
    for p1, p2 in pixel_lines:
        orientation = _classify_line_orientation(p1, p2)
        if orientation == 'vertical':
            vertical_lines.append((p1, p2))
        else:
            horizontal_lines.append((p1, p2))
    
    # 垂直線の処理
    for p1, p2 in vertical_lines:
        # 元の点（歪んだ画像上の実際の位置）
        src_points.append(p1)
        src_points.append(p2)
        
        # 垂直線として補正: X座標を2点の平均に揃える
        # Y座標はそのまま維持
        x_avg = (p1[0] + p2[0]) / 2.0
        dst_points.append((x_avg, p1[1]))
        dst_points.append((x_avg, p2[1]))
    
    # 水平線の処理
    for p1, p2 in horizontal_lines:
        # 元の点（歪んだ画像上の実際の位置）
        src_points.append(p1)
        src_points.append(p2)
        
        # 水平線として補正: Y座標を2点の平均に揃える
        # X座標はそのまま維持
        y_avg = (p1[1] + p2[1]) / 2.0
        dst_points.append((p1[0], y_avg))
        dst_points.append((p2[0], y_avg))
    
    # 垂直線と水平線を混在させた場合の追加チェック
    # 対応点が極端に偏らないようにする
    if len(vertical_lines) > 0 and len(horizontal_lines) > 0:
        # 垂直線と水平線の比率をチェック
        v_ratio = len(vertical_lines) / len(pixel_lines)
        h_ratio = len(horizontal_lines) / len(pixel_lines)
        
        # バランスが悪い場合は警告（実装上は問題なく動作）
        if v_ratio < 0.3 or h_ratio < 0.3:
            # 片方が極端に少ない場合でも処理は続行
            pass
    
    return src_points, dst_points


def get_mesh_coordinates(
    image_shape: Tuple[int, int],
    mesh_size: Tuple[int, int]
) -> np.ndarray:
    """
    メッシュ座標を取得
    
    Args:
        image_shape: tuple、(H, W)
        mesh_size: tuple、(rows, cols)
    
    Returns:
        numpy.ndarray、shape=(rows+1, cols+1, 2)、TCG座標系の交点座標（正規化済み）
    """
    height, width = image_shape
    rows, cols = mesh_size
    
    # TCG座標系でメッシュを生成（正規化座標）
    # X: -0.5 〜 +0.5
    # Y: -0.5 〜 +0.5
    
    x_coords = np.linspace(-0.5, 0.5, cols + 1)
    y_coords = np.linspace(-0.5, 0.5, rows + 1)  # 上から下
    
    mesh_coords = np.zeros((rows + 1, cols + 1, 2), dtype=np.float32)
    
    for row in range(rows + 1):
        for col in range(cols + 1):
            mesh_coords[row, col, 0] = x_coords[col]
            mesh_coords[row, col, 1] = y_coords[row]
    
    return mesh_coords


def _validate_image(image: np.ndarray):
    """画像の検証"""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be numpy.ndarray, got {type(image)}")

    if image.dtype != np.float32:
        raise TypeError(f"image must be float32, got {image.dtype}")

    # (H, W, 3) または (H, W) (マスク用) のいずれかを許容
    if not (
        (image.ndim == 3 and image.shape[2] == 3)
        or image.ndim == 2
    ):
        raise TypeError(f"image must have shape (H, W, 3) or (H, W), got {image.shape}")
