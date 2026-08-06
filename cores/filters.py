
import numpy as np
import cv2
from numba import njit, prange

from threads import lock_numba

# 大きい sigma のガウシアンは分離フィルタでも O(N*sigma) で重い。
# 「縮小 → 小 sigma でブラー → 拡大」で近似すると O(N) 近くに落ちる。
# ボケ用途では見た目の差はごく僅か（平均誤差 < 1e-3 程度）。
_FAST_BLUR_SIGMA_THRESHOLD = 8.0


def _fast_isotropic_blur(image, sigma):
    """等方ガウシアンの高速近似（sigma が大きいときだけ縮小→拡大）。"""
    sigma = float(sigma)
    if sigma <= 0.0:
        return image
    if sigma <= _FAST_BLUR_SIGMA_THRESHOLD:
        return cv2.GaussianBlur(image, (0, 0), sigma)
    f = min(8.0, sigma / 4.0)
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(round(w / f))), max(1, int(round(h / f)))),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _fast_horizontal_blur(image, ksize):
    """横方向のみのガウシアン（ksize 指定）の高速近似。横だけ縮小して戻す。"""
    ksize = int(ksize) | 1
    if ksize <= 1:
        return image
    # cv2 が ksize から導く実効 sigma。
    sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    if sigma <= _FAST_BLUR_SIGMA_THRESHOLD:
        return cv2.GaussianBlur(image, (ksize, 1), 0)
    f = min(8.0, sigma / 4.0)
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(round(w / f))), h), interpolation=cv2.INTER_AREA)
    sk = int(2 * round(3.0 * (sigma / f)) + 1)
    small = cv2.GaussianBlur(small, (sk, 1), 0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def lensblur_filter(image, radius):
    # カーネルを生成
    kernel_size = int(2 * radius + 1)
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    
    # カーネルに円を描く
    radius = int(radius)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    kernel /= np.sum(kernel)

    # レンズブラーを適用
    blurred_image = cv2.filter2D(np.array(image), -1, kernel)
    return blurred_image

# apply_lensblur は effect_backends.lens_blur_adapter（Metal 優先 / CPU reference フォールバック）へ移植。
# CPU reference 実装は effect_backends/lens_blur_reference.py を参照。

def scratch_effect(image, scratch_intensity=1.0, shift_parcent=1.0, resolution_scale=1.0):
    """
    モザイク効果に特化した引っ掻きフィルター（高速化版）
    画像をより判別困難にする

    resolution_scale: 入力がプレビュー縮小/フルのどの解像度かを表す係数
        （= core.calc_resolution_scale）。傷サイズを画像比で一定に保ち、
        preview と export で見た目（傷の粗さ・カバレッジ）を揃えるために使う。
    """
    h, w = image.shape[:2]
    result = image.copy()

    # 引っ掻き効果を段階的に適用
    num_passes = 3
    rscale = max(float(resolution_scale), 1e-3)

    # 旧実装は傷を1本ずつ Python ループで描いて preview でも 200ms 超と激重だった。
    # 「セル単位のランダム変位マップ」を作って 1 回の cv2.remap で済ませることで、
    # 同じ質感（横スジ）のままループを撤廃して高速化する（preview ~4-5 倍速）。
    base_x, base_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )

    for pass_num in range(num_passes):
        # セル（傷）の大きさは画像長辺比を一定にする（rscale 追従）。
        # これで grain（傷の粒）が preview/export で揃い、拡大時と同じ細かさになる。
        scratch_size = max(1, int(5 * (pass_num + 1) * scratch_intensity * rscale))
        # 変位させるセルの割合（旧実装の被覆率 0.25*(p+1) を踏襲）。
        coverage = min(1.0, 0.25 * (pass_num + 1) * scratch_intensity)
        gh = max(1, h // scratch_size)
        gw = max(1, w // scratch_size)

        # セルごとに ±size のランダム変位。coverage の割合だけ動かし、残りは据え置き。
        dx = np.random.randint(-scratch_size, scratch_size + 1, (gh, gw)).astype(np.float32)
        dy = np.random.randint(-scratch_size, scratch_size + 1, (gh, gw)).astype(np.float32)
        keep = (np.random.random((gh, gw)) < coverage).astype(np.float32)
        dx *= keep
        dy *= keep

        # セル解像度の変位マップを INTER_NEAREST で実画素へ拡大（セル内は一定変位＝ブロックずれ）。
        map_x = base_x + cv2.resize(dx, (w, h), interpolation=cv2.INTER_NEAREST)
        map_y = base_y + cv2.resize(dy, (w, h), interpolation=cv2.INTER_NEAREST)
        result = cv2.remap(
            result, map_x, map_y,
            interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT,
        )


    # ガウシアンブラーのカーネルサイズを調整（奇数にする必要がある）
    kernel_size = int(555 * shift_parcent)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # scratch の処理時間の大半はこの巨大な横ブラー。縮小→拡大で近似して高速化。
    result = _fast_horizontal_blur(result, kernel_size)

    return result


def mosaic_effect(image, block_size=16):
    """
    モザイク効果を適用する関数
    [params]
    image: (H,W,3) float32形式のRGB画像（0.0-1.0）
    block_size: モザイクのブロックサイズ（ピクセル）
    """
    h, w = image.shape[:2]

    block_size = int(block_size)
    if block_size <= 0:
        return image.copy()

    # Python 二重ループ（ブロック小で激重）を resize で置換。
    # INTER_AREA で縮小＝ブロック平均、INTER_NEAREST で拡大＝平均色の敷き詰め。
    nw = max(1, round(w / block_size))
    nh = max(1, round(h / block_size))
    small = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    
def frosted_glass_effect(image, blur_radius=10, noise_scale=0.01):
    """
    フロストガラス効果を適用する関数
    [params]
    image: (H,W,3) float32形式のRGB画像（0.0-1.0）
    blur_radius: ぼかし強度
    noise_scale: ノイズの強度（0.0-0.1）
    """
    h, w = image.shape[:2]
    
    # ガウシアンブラーの最適化
    kernel_size = int(4 * blur_radius) | 1  # 奇数保証
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 
                             sigmaX=blur_radius, 
                             sigmaY=blur_radius,
                             borderType=cv2.BORDER_REPLICATE)
    
    # ノイズ生成（-1.0〜1.0の範囲）
    noise_x = (np.random.rand(h,w) * 2 - 1) * noise_scale * w
    noise_y = (np.random.rand(h,w) * 2 - 1) * noise_scale * h
    
    # 座標マップ生成
    x_map, y_map = np.meshgrid(np.arange(w), np.arange(h))
    x_map = (x_map + noise_x).astype(np.float32)
    y_map = (y_map + noise_y).astype(np.float32)
    
    # リマップ処理
    result = cv2.remap(blurred, x_map, y_map,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)
    
    return result



if __name__ == '__main__':
    # 入力画像の読み込み（0.0-1.0のfloat32に変換）
    input_img = cv2.imread("test_input.jpg").astype(np.float32) / 255.0
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    # 各効果の適用
    scratch_img = scratch_effect(input_img, scratch_intensity=1.0, shift_parcent=1.5)
    mosaic_img = mosaic_effect(input_img, block_size=80)
    frosted_img = frosted_glass_effect(input_img, blur_radius=10, noise_scale=0.01)

    # 結果の保存
    scratch_img = cv2.cvtColor(scratch_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test_scratch.jpg", (scratch_img*255).astype(np.uint8))
    mosaic_img = cv2.cvtColor(mosaic_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test_mosaic.jpg", (mosaic_img*255).astype(np.uint8))
    frosted_img = cv2.cvtColor(frosted_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test_frosted.jpg", (frosted_img*255).astype(np.uint8))


@lock_numba
@njit('f4[:,:](f4[:,:], i4, i4)', parallel=True, fastmath=True, cache=True)
def fast_median_filter(img, kernel_size=3, num_bins=1024):
    """
    量子化とヒストグラムベースの高速メディアンフィルタ
    float32画像を高速処理可能
    
    Parameters:
        img (np.ndarray): 入力画像 (float32)
        kernel_size (int): カーネルサイズ (奇数)
        num_bins (int): 量子化ビン数 (速度/精度のトレードオフ)
    
    Returns:
        np.ndarray: フィルタリング後の画像 (float32)
    """
    h, w = img.shape
    pad = kernel_size // 2
    median_index = (kernel_size * kernel_size) // 2
    
    # 画像の最小値/最大値を計算
    min_val = np.min(img)
    max_val = np.max(img)
    scale = (num_bins - 1) / (max_val - min_val + 1e-7)
    
    # 量子化画像の作成
    quantized = ((img - min_val) * scale).astype(np.float32)
    
    # パディング追加 (reflectモード)
    padded = np.zeros((h + 2*pad, w + 2*pad), dtype=np.float32)
    padded[pad:-pad, pad:-pad] = quantized
    for i in prange(pad):
        padded[i, pad:-pad] = quantized[pad-i-1]  # 上端
        padded[-(i+1), pad:-pad] = quantized[-(pad-i)]  # 下端
        padded[pad:-pad, i] = quantized[:, pad-i-1]  # 左端
        padded[pad:-pad, -(i+1)] = quantized[:, -(pad-i)]  # 右端
    
    # 出力画像初期化
    result = np.zeros((h, w), dtype=np.float32)
    
    # メイン処理 (行単位で並列化。y方向は各行が独立した hist を持つため prange で安全)
    for y in prange(h):
        hist = np.zeros(num_bins, dtype=np.uint16)
        # 初期ヒストグラム構築（同一 hist への += の積み上げなので range で逐次実行）
        for ky in range(kernel_size):
            for kx in range(kernel_size):
                val = padded[y + ky, kx]
                hist[int(val)] += 1

        # 行方向にスライディング（hist を左右にずらしながら更新するため x は逐次依存。range で実行）
        for x in range(w):
            # 中央値計算
            cumsum = 0
            for b in range(num_bins):
                cumsum += hist[b]
                if cumsum > median_index:
                    result[y, x] = min_val + b / scale
                    break

            # ヒストグラム更新 (左カラム削除/右カラム追加、同じ hist を書き換えるため range で実行)
            if x < w - 1:
                for ky in range(kernel_size):
                    # 左カラム削除
                    left_val = padded[y + ky, x]
                    hist[int(left_val)] -= 1
                    # 右カラム追加
                    right_val = padded[y + ky, x + kernel_size]
                    hist[int(right_val)] += 1
                    
    return result

@lock_numba
@njit('f4[:,:,:](f4[:,:,:], f4[:,:,:], f4, f4)', parallel=True, fastmath=True, cache=True)
def _orton_blend_kernel(base, blurred, opacity, intensity):
    """orton_effect のスクリーン合成〜最終ブレンドを1パスに融合した版。

    clip/screen_layer/where/multiply/addWeighted×2 を画素ごとに直接計算することで、
    numpyの中間配列の確保・読み書き（フル解像度で計6回分）を避ける。数式自体は元実装と同一。
    """
    h, w, c = base.shape
    out = np.empty((h, w, c), dtype=np.float32)
    inv_opacity = np.float32(1.0) - opacity
    inv_intensity = np.float32(1.0) - intensity
    for i in prange(h):
        for j in range(w):
            for k in range(c):
                b = base[i, j, k]
                if b > 1.0:
                    screen = b
                else:
                    bc = b if b > 0.0 else np.float32(0.0)
                    one_minus = np.float32(1.0) - bc
                    screen = np.float32(1.0) - one_minus * one_minus
                mult = screen * blurred[i, j, k]
                result = screen * inv_opacity + mult * opacity
                out[i, j, k] = b * inv_intensity + result * intensity
    return out

def orton_effect(image, blur_radius=30, opacity=0.75, intensity=0.5):
    """
    オートン効果を適用する関数（方法B: 最上位レイヤーが乗算+ぼかし）
    
    Parameters:
    -----------
    image : np.ndarray
        入力画像 (H, W, 3) のfloat32 RGB画像 (値域: 0.0-1.0)
    blur_radius : float
        ガウスぼかしの半径（標準偏差）。デフォルトは30
        大きいほど柔らかい効果
    opacity : float
        最上位レイヤー（乗算+ぼかし）の不透明度 (0.0-1.0)
        デフォルトは0.75。大きいほど効果が強い
    intensity : float
        効果の強さ (0.0-1.0)
        デフォルトは0.5。大きいほど効果が強い
    
    Returns:
    --------
    result : np.ndarray
        オートン効果を適用した画像 (H, W, 3) のfloat32 RGB画像
    """
    # ぼかし画像の作成（各チャンネル独立にぼかし）
    # 大きい blur_radius（最大 sigma~100）の等方ブラーが律速。縮小→拡大で近似して高速化。
    blurred = _fast_isotropic_blur(image, blur_radius)

    # スクリーンレイヤー（1-(1-base)^2、HDRはbaseそのまま）〜乗算〜opacity/intensityブレンドまでを
    # 1パスのnumbaカーネルで計算（中間配列の確保・読み書きを避けるための融合。数式は従来と同一）。
    base = np.ascontiguousarray(image, dtype=np.float32)
    blurred = np.ascontiguousarray(blurred, dtype=np.float32)
    return _orton_blend_kernel(base, blurred, np.float32(opacity), np.float32(intensity))
