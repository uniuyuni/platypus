import logging

import numpy as np
import cv2
from numba import njit, prange
import time

from threads import lock_numba

# 同ファイルの rgb_to_hlc_gain / hlc_gain_to_rgb と異なり @lock_numba を意図的に外している。
# threads.lock_numba はコンパイル時だけでなく呼び出しの実行区間全体を numba_lock
# （プロセス全体で共有の非再入 Lock）で直列化する実装のため、付けるとコンパイル時保護に
# とどまらず、アプリ内の他の @lock_numba 対象すべてと実行時の排他が発生するようになる
# （現状の唯一の呼び出し元 dehaze_reference._estimate_depth_map は他の lock_numba 区間の
# 内側からは呼ばれていないため付けてもデッドロックはしないが、実行時の並行性という
# 挙動そのものを変えてしまうため、低優先度クリーンアップの範囲としては見送った）。
@njit(parallel=True)
def rgb2hls(img):
    """
    Convert RGB image to HLS + Gain (HDR safe normalization).
    
    Returns 4 channels: (H, L, S, Gain)
    H: 0-360
    L: 0.0-1.0 (Normalized brightness)
    S: 0.0-1.0 (Saturation)
    Gain: >= 1.0 (HDR multiplier)
    
    Logic:
    Gain = max(1.0, max(R, G, B))
    RGB_norm = RGB / Gain
    L = max(RGB_norm) (Value)
    S = delta / max(RGB_norm)
    """
    rows, cols, _ = img.shape
    out = np.empty((rows, cols, 4), dtype=img.dtype)
    
    for i in prange(rows):
        for j in range(cols):
            r = img[i, j, 0]
            g = img[i, j, 1]
            b = img[i, j, 2]
            
            # Clamp negatives for stability
            r_c = 0.0 if r < 0.0 else r
            g_c = 0.0 if g < 0.0 else g
            b_c = 0.0 if b < 0.0 else b
            
            max_val = max(r_c, g_c, b_c)
            
            # Calculate Gain
            gain = 1.0
            if max_val > 1.0:
                gain = max_val
            
            # Normalize RGB
            # If gain > 1, max_val becomes 1.0 after normalization
            # If gain = 1, max_val is <= 1.0
            
            # We don't need to actually divide r,g,b if we compute L, S smart.
            # L_norm = max_val / gain
            
            l = max_val / gain
            
            # S = (Max - Min) / Max
            # Since S is ratio, S(RGB) == S(RGB_norm).
            # So we can calculate S from original values.
            
            min_val = min(r_c, g_c, b_c)
            delta = max_val - min_val
            
            # Saturation Damping (epsilon)
            # HDR時にepsilonをスケールする案を検討したが、結局rawベースの計算のまま据え置いた。
            damp_epsilon = 0.005

            if max_val <= 1e-9:
                s = 0.0
                h = 0.0
            else:
                s = delta / (max_val + damp_epsilon)
                
                if delta < 1e-7:
                    h = 0.0
                elif max_val == r_c:
                    h = (g_c - b_c) / delta
                    if g_c < b_c:
                        h += 6.0
                elif max_val == g_c:
                    h = (b_c - r_c) / delta + 2.0
                else:
                    h = (r_c - g_c) / delta + 4.0
                h /= 6.0
            
            out[i, j, 0] = h * 360.0
            out[i, j, 1] = l
            out[i, j, 2] = s
            out[i, j, 3] = gain
            
    return out


# Rec.709の輝度係数（色差計算用）
KR = 0.2126
KG = 0.7152
KB = 0.0722

# 彩度ガンマ外のソフトクリップ閾値（対 C_max 比）。
# 彩度ブースト等で Chroma が gamut を超えると、逆変換後のRGBが負/過大になり、
# 後段のチャンネル単位クリップで色相がずれる。これを防ぐため、色相(H)と輝度(L)を保ったまま
# Chroma だけを C_max 以下に滑らかに圧縮する。この値を 1.0 にするとハードクリップ相当（既存色
# 不変だが折れ点）、小さくするほど手前から緩やかに丸める（ごく高彩度の既存色がわずかに低下）。
GAMUT_SOFTCLIP_KNEE = 0.95

# C_max を求めるときの「正規化チャンネルの上限」。
# rgb_to_hlc_gain は RGB を gain=max(R,G,B) で割るため、正規化空間では必ずどれかの
# チャンネルが 1.0 ちょうどになる = 全ピクセルが常に gamut 境界上にある。よってここを
# 1.0 にすると C == C_max となり、ソフトクリップ（ratio→∞ でも上限 1.0）に阻まれて
# 彩度のプラス方向が一切効かなくなるうえ、無操作の往復でも彩度が約1.7%失われる。
# HDR ヘッドルームを持つパイプラインなので、正規化チャンネルが 1.0 を超えること自体は
# 許容し（= 元の最大チャンネルより明るくなってよい）、C = C_norm * 1.5 のスケールに
# 合わせて 1.5 を上限とする。負値側の制約（-L/k）は色相ズレを防ぐため従来どおり維持する。
GAMUT_NORM_CEILING = 1.5

@lock_numba
@njit("f4[:,:,:](f4[:,:,:])", parallel=True, fastmath=True)
def rgb_to_hlc_gain(rgb):
    """
    RGB(HDR) -> HLC+Gain変換 (線形YCbCr風、Gain = max(R,G,B))
    
    Parameters:
    -----------
    rgb : ndarray, shape (H, W, 3), dtype=float32
        HDR RGB画像
    
    Returns:
    --------
    hlcg : ndarray, shape (H, W, 4), dtype=float32
        H: 色相 [0, 360)
        L: 輝度 [0, 1] - 線形加重平均
        C: 彩度 [0, 1] - 線形
        G: Gain - max(R,G,B)
    """
    H_img, W = rgb.shape[0], rgb.shape[1]
    hlcg = np.empty((H_img, W, 4), dtype=np.float32)
    
    for i in prange(H_img):  # 外側(i)はprangeで並列化。内側(j)はprangeにするとなぜかクラッシュするためrangeを使用
        for j in range(W):
            r, g, b = rgb[i, j, 0], rgb[i, j, 1], rgb[i, j, 2]
            
            # Gain = max(R,G,B) に変更
            max_val = max(r, max(g, b))
            gain = max_val
            hlcg[i, j, 3] = gain
            
            if gain < 1e-10:
                hlcg[i, j, 0] = 0.0
                hlcg[i, j, 1] = 0.0
                hlcg[i, j, 2] = 0.0
                continue
            
            # 正規化
            r_norm = r / gain
            g_norm = g / gain
            b_norm = b / gain
            
            # 輝度（正規化空間）
            Y_norm = KR * r_norm + KG * g_norm + KB * b_norm
            L = Y_norm
            hlcg[i, j, 1] = L
            
            # 色差成分（線形）
            Cb = b_norm - Y_norm
            Cr = r_norm - Y_norm
            
            # 彩度（極座標）
            C = (Cb * Cb + Cr * Cr) ** 0.5
            
            # 理論上の最大Chroma
            # 正規化空間で max(R,G,B) = 1.0 の時
            # 最大の色差は約 sqrt((1-Y)^2 + (1-Y)^2) = sqrt(2) * (1-Y)
            # 最悪ケース: Y=0の時、C_max ≈ sqrt(2) ≈ 1.414
            C_max = 1.5  # 安全マージン
            C_norm = C / C_max
            C_norm = min(1.0, C_norm)
            hlcg[i, j, 2] = C_norm
            
            # 色相
            if C < 1e-8:
                H = 0.0
            else:
                H = np.arctan2(Cr, Cb) * 180.0 / np.pi
                if H < 0.0:
                    H += 360.0
            
            hlcg[i, j, 0] = H
    
    return hlcg

@lock_numba
@njit("f4[:,:,:](f4[:,:,:])", parallel=True, fastmath=True)
def hlc_gain_to_rgb(hlcg):
    """
    HLC+Gain -> RGB(HDR)変換
    
    Parameters:
    -----------
    hlcg : ndarray, shape (H, W, 4), dtype=float32
        H, L, C, G
    
    Returns:
    --------
    rgb : ndarray, shape (H, W, 3), dtype=float32
    """
    H_img, W = hlcg.shape[0], hlcg.shape[1]
    rgb = np.empty((H_img, W, 3), dtype=np.float32)
    
    for i in prange(H_img):
        for j in range(W):
            H = hlcg[i, j, 0]
            L = hlcg[i, j, 1]
            C_norm = hlcg[i, j, 2]
            gain = hlcg[i, j, 3]
            
            # Y (正規化空間)
            Y_norm = L

            # Chromaを復元
            C = C_norm * 1.5

            # 極座標 → 色差成分の係数
            H_rad = H * np.pi / 180.0
            cosH = np.cos(H_rad)
            sinH = np.sin(H_rad)

            # 色相(H)・輝度(L)を保ったまま正規化RGB(L + C*k)が[0, GAMUT_NORM_CEILING]に収まる
            # 最大Chroma C_max を求め、それを超える分だけ滑らかに圧縮する
            # （負RGB→チャンネル単位クリップによる色相シフトを防ぐ）。
            #   r_norm = L + C*sinH, b_norm = L + C*cosH, g_norm = L + C*kg
            kr = sinH
            kb = cosH
            kg = -(KR / KG) * sinH - (KB / KG) * cosH
            C_max = 1.0e30
            if kr > 1e-6:
                lim = (GAMUT_NORM_CEILING - L) / kr
                if lim < C_max: C_max = lim
            elif kr < -1e-6:
                lim = -L / kr
                if lim < C_max: C_max = lim
            if kb > 1e-6:
                lim = (GAMUT_NORM_CEILING - L) / kb
                if lim < C_max: C_max = lim
            elif kb < -1e-6:
                lim = -L / kb
                if lim < C_max: C_max = lim
            if kg > 1e-6:
                lim = (GAMUT_NORM_CEILING - L) / kg
                if lim < C_max: C_max = lim
            elif kg < -1e-6:
                lim = -L / kg
                if lim < C_max: C_max = lim
            if C_max < 0.0:
                C_max = 0.0

            if C_max <= 1e-8:
                C = 0.0
            elif C > GAMUT_SOFTCLIP_KNEE * C_max:
                ratio = C / C_max
                t = (ratio - GAMUT_SOFTCLIP_KNEE) / (1.0 - GAMUT_SOFTCLIP_KNEE)
                ratio = GAMUT_SOFTCLIP_KNEE + (1.0 - GAMUT_SOFTCLIP_KNEE) * (t / (1.0 + t))
                C = ratio * C_max

            Cb = C * cosH
            Cr = C * sinH

            # Y, Cb, Cr -> R, G, B
            g_norm = Y_norm - (KR / KG) * Cr - (KB / KG) * Cb
            r_norm = Cr + Y_norm
            b_norm = Cb + Y_norm

            # Gainを適用
            rgb[i, j, 0] = r_norm * gain
            rgb[i, j, 1] = g_norm * gain
            rgb[i, j, 2] = b_norm * gain
    
    return rgb


def test_basic_colors_detailed():
    """8色の詳細テスト"""
    
    colors = [
        ('赤 (Red)',         1.0, 0.0, 0.0),
        ('オレンジ (Orange)', 1.0, 0.5, 0.0),
        ('黄色 (Yellow)',    1.0, 1.0, 0.0),
        ('緑 (Green)',       0.0, 1.0, 0.0),
        ('シアン (Cyan)',    0.0, 1.0, 1.0),
        ('青 (Blue)',        0.0, 0.0, 1.0),
        ('紫 (Violet)',      0.5, 0.0, 1.0),
        ('マゼンタ (Magenta)', 1.0, 0.0, 1.0),
    ]
    
    logging.info("="*80)
    logging.info("Pure Colors Test with Gain = max(R,G,B)")
    logging.info("="*80)
    logging.info(f"{'Color':<20} {'RGB Input':<15} {'H':<10} {'L':<8} {'C':<8} {'Gain':<8} {'RGB Restored':<15} {'Error'}")
    logging.info("-"*80)
    
    for name, r, g, b in colors:
        # 変換
        H, L, C, G = rgb_to_hlc_gain_single(r, g, b)
        
        # 逆変換
        r_out, g_out, b_out = hlc_gain_to_rgb_single(H, L, C, G)
        
        # エラー計算
        error = max(abs(r - r_out), abs(g - g_out), abs(b - b_out))
        
        rgb_in = f"({r:.1f},{g:.1f},{b:.1f})"
        rgb_out_str = f"({r_out:.3f},{g_out:.3f},{b_out:.3f})"
        
        logging.info(f"{name:<20} {rgb_in:<15} {H:>7.2f}° {L:>7.3f} {C:>7.3f} {G:>7.3f} {rgb_out_str:<15} {error:.2e}")
    
    logging.info("\n" + "="*80)
    logging.info("Hue Ranges (based on midpoints)")
    logging.info("="*80)
    
    hues = []
    for name, r, g, b in colors:
        H, _, _, _ = rgb_to_hlc_gain_single(r, g, b)
        hues.append((name, H))
    
    for i in range(len(hues)):
        name, curr_h = hues[i]
        prev_h = hues[i - 1][1]
        next_h = hues[(i + 1) % len(hues)][1]
        
        # 前の色との中間点
        if curr_h < prev_h:
            start = (prev_h + curr_h + 360) / 2
            if start >= 360:
                start -= 360
        else:
            start = (prev_h + curr_h) / 2
        
        # 次の色との中間点
        if next_h < curr_h:
            end = (curr_h + next_h + 360) / 2
            if end >= 360:
                end -= 360
        else:
            end = (curr_h + next_h) / 2
        
        if start > end:
            logging.info(f"{name:<20} {start:>7.2f}° ~ 360° / 0° ~ {end:>7.2f}°")
        else:
            logging.info(f"{name:<20} {start:>7.2f}° ~ {end:>7.2f}°")


def rgb_to_hlc_gain_single(r, g, b):
    """単一ピクセル変換"""
    max_val = max(r, max(g, b))
    gain = max_val
    
    if gain < 1e-10:
        return 0.0, 0.0, 0.0, 0.0
    
    r_norm = r / gain
    g_norm = g / gain
    b_norm = b / gain
    
    Y_norm = KR * r_norm + KG * g_norm + KB * b_norm
    L = Y_norm
    
    Cb = b_norm - Y_norm
    Cr = r_norm - Y_norm
    
    C = (Cb * Cb + Cr * Cr) ** 0.5
    C_norm = min(1.0, C / 1.5)
    
    if C < 1e-8:
        H = 0.0
    else:
        H = np.arctan2(Cr, Cb) * 180.0 / np.pi
        if H < 0.0:
            H += 360.0
    
    return H, L, C_norm, gain


def hlc_gain_to_rgb_single(H, L, C_norm, gain):
    """単一ピクセル逆変換"""
    Y_norm = L
    C = C_norm * 1.5
    
    H_rad = H * np.pi / 180.0
    Cb = C * np.cos(H_rad)
    Cr = C * np.sin(H_rad)
    
    g_norm = Y_norm - (KR / KG) * Cr - (KB / KG) * Cb
    r_norm = Cr + Y_norm
    b_norm = Cb + Y_norm
    
    return r_norm * gain, g_norm * gain, b_norm * gain

def linear_rgb_to_ycbcr(img_rgb):
    """
    Linear RGB画像を Linear YCbCr に変換します。
    ガンマ補正を含まない単純な行列演算のため、HDR値(1.0超)も正確に保持されます。

    Args:
        img_rgb (numpy.ndarray): Linear RGB画像 (H, W, 3), float32

    Returns:
        numpy.ndarray: Linear YCbCr画像 (H, W, 3), float32
        - Channel 0: Y  (輝度)
        - Channel 1: Cb (青 - 輝度)
        - Channel 2: Cr (赤 - 輝度)
    """
    r, g, b = cv2.split(img_rgb)

    # BT.709 係数
    # Y = 0.2126 R + 0.7152 G + 0.0722 B
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    # Cb = (B - Y) / 1.8556
    cb = (b - y) / 1.8556
    
    # Cr = (R - Y) / 1.5748
    cr = (r - y) / 1.5748

    return cv2.merge((y, cb, cr))


def linear_ycbcr_to_rgb(img_ycbcr):
    """
    Linear YCbCr画像を Linear RGB に戻します。
    可逆変換であり、数値誤差を除き、元のRGB値が復元されます。

    Args:
        img_ycbcr (numpy.ndarray): Linear YCbCr画像 (H, W, 3), float32

    Returns:
        numpy.ndarray: Linear RGB画像 (H, W, 3), float32
    """
    y, cb, cr = cv2.split(img_ycbcr)

    # 逆変換係数 (BT.709 Derived)
    # R = Y + 1.5748 Cr
    # B = Y + 1.8556 Cb
    # G = Y - 0.1873 Cb - 0.4681 Cr

    r = y + 1.5748 * cr
    b = y + 1.8556 * cb
    g = y - 0.1873 * cb - 0.4681 * cr

    return cv2.merge((r, g, b))


def test_saturation_linearity():
    """彩度の線形性テスト"""
    logging.info("\n" + "="*80)
    logging.info("Saturation Linearity Test")
    logging.info("="*80)
    logging.info("Testing: Red (1,0,0) with varying saturation")
    logging.info("-"*80)
    logging.info(f"{'Original C':<15} {'RGB Output':<25} {'Restored C':<15} {'Error'}")
    logging.info("-"*80)
    
    r, g, b = 1.0, 0.0, 0.0
    H, L, C_orig, G = rgb_to_hlc_gain_single(r, g, b)
    
    for c_factor in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]:
        C_test = min(1.0, C_orig * c_factor)
        
        r_out, g_out, b_out = hlc_gain_to_rgb_single(H, L, C_test, G)
        
        # 再変換して確認
        H_back, L_back, C_back, G_back = rgb_to_hlc_gain_single(r_out, g_out, b_out)
        
        error = abs(C_test - C_back)
        rgb_str = f"({r_out:.3f},{g_out:.3f},{b_out:.3f})"
        
        logging.info(f"{C_test:<15.3f} {rgb_str:<25} {C_back:<15.3f} {error:.2e}")


def test_gradient_quality():
    """グラデーションの品質テスト"""
    logging.info("\n" + "="*80)
    logging.info("Gradient Quality Test")
    logging.info("="*80)
    
    # 微妙なグラデーション
    h, w = 1024, 256
    gradient = np.zeros((h, w, 3), dtype=np.float32)
    for i in range(h):
        val = 0.5 + 0.1 * (i / h)
        gradient[i, :] = val
    
    logging.info("Input: %sx%s gradient from 0.5 to 0.6", h, w)
    
    # 変換
    start = time.time()
    hlcg = rgb_to_hlc_gain(gradient)
    t1 = time.time() - start
    
    start = time.time()
    restored = hlc_gain_to_rgb(hlcg)
    t2 = time.time() - start
    
    # エラー
    error = np.abs(gradient - restored)
    
    logging.info("Timing: Forward %.2fms, Backward %.2fms", t1*1000, t2*1000)
    logging.info("Max error: %.2e", error.max())
    logging.info("Mean error: %.2e", error.mean())
    
    # グラデーションの滑らかさ
    center_col = gradient[:, w//2, 0]
    restored_col = restored[:, w//2, 0]
    
    diff_orig = np.diff(center_col)
    diff_rest = np.diff(restored_col)
    
    logging.info("\nGradient smoothness:")
    logging.info("  Original diff std: %.2e", diff_orig.std())
    logging.info("  Restored diff std: %.2e", diff_rest.std())
    logging.info("  Ratio: %.6f", diff_rest.std() / diff_orig.std())
    
    # HLCの連続性
    H_col = hlcg[:, w//2, 0]
    L_col = hlcg[:, w//2, 1]
    C_col = hlcg[:, w//2, 2]
    
    logging.info("\nHLC continuity:")
    logging.info("  H diff std: %.2e", np.diff(H_col).std())
    logging.info("  L diff std: %.2e", np.diff(L_col).std())
    logging.info("  C diff std: %.2e", np.diff(C_col).std())


if __name__ == "__main__":
    # 基本色のテスト
    test_basic_colors_detailed()
    
    # 彩度の線形性テスト
    test_saturation_linearity()
    
    # グラデーション品質テスト
    test_gradient_quality()
    
    logging.info("\n" + "="*80)
    logging.info("✓ All tests completed")
    logging.info("="*80)
