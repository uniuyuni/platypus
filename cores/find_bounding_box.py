import logging
import numpy as np
import cv2
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # noop decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from threads import lock_numba

def find_bounding_box(image, threshold=None, margin_ratio=None, aspect_ratio=None, verbose=False):
    """
    OpenCVの輪郭抽出を用いて、画像内の有効領域（黒くない部分）の内接最大矩形を検出する。
    有効領域の内側に完全に収まる最大の矩形（Largest Inscribed Rectangle）を返す。
    
    Parameters:
    -----------
    image : numpy.ndarray
        RGB画像 (H, W, 3) の float32 配列 (値の範囲: 0.0-1.0)
    threshold : float or None
        黒とみなす閾値（0.0-1.0）。Noneの場合は0.001を使用
    margin_ratio : float or None
        安全マージンの比率（0.0-1.0）。Noneの場合は0.0を使用
        ※内側検出の場合、マージンは「さらに内側へ」作用する
    aspect_ratio : float or None
        矩形の縦横比（幅/高さ）。Noneの場合は任意の縦横比で最大を探索
        例: 16/9 ≈ 1.778 (16:9), 4/3 ≈ 1.333 (4:3), 1.0 (正方形)
    verbose : bool
        詳細情報を表示するかどうか (デフォルト: False)
    
    Returns:
    --------
    tuple : (x1, y1, x2, y2) 元画像サイズでのバウンディングボックスの座標
    """
    # パラメータを自動設定
    if threshold is None:
        # 暗部も有効画素とするため低めの閾値
        # 0.01 = 約 2.5/255、周辺減光やJPEGアーティファクトを考慮
        threshold = 0.001
    
    # 内接矩形探索ではマージン不要（アルゴリズムが内側を保証するため）
    if margin_ratio is None:
        margin_ratio = 0.0
    
    if verbose:
        logging.info("[検出設定] threshold=%.3f, margin_ratio=%.3f", threshold, margin_ratio)

    
    h, w = image.shape[:2]
    
    # 1. 画像の前処理（グレースケール）
    if len(image.shape) == 3:
        # RGBA対応: アルファチャンネルが含まれると平均値が下がる可能性があるため無視する
        if image.shape[2] == 4:
            gray = np.mean(image[:, :, :3], axis=2)
        else:
            gray = np.mean(image, axis=2)
    else:
        gray = image
        
    # 値域チェックと閾値調整
    img_max = np.max(gray)
    
    eff_threshold = threshold
        
    if verbose:
        logging.info(
            "[検出設定] MaxVal=%.2f, Threshold=%.3f (Original=%s), Margin=%.3f",
            img_max,
            eff_threshold,
            threshold,
            margin_ratio,
        )

    binary = ((gray > eff_threshold) * 255).astype(np.uint8)
    
    # ノイズ除去
    # ユーザー要件: "外周が苦手" / "最大が取れない" / "鋭角な角が消える"
    # -> OPEN(縮小)は行わず、CLOSE(穴埋め)のみ行う。
    # 髭ノイズ(外に出ているゴミ)は contours フィルタリング(面積)で弾く。
    
    # カーネルサイズ（固定値）
    k_dim = 3
    # 奇数にする
    if k_dim % 2 == 0: k_dim += 1

    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dim, k_dim))
    
    # 穴埋めのみ (iterations=2 で十分)
    binary_filled = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 2. 輪郭抽出と「クリーンなマスク」の作成 (Full Resolution)
    # ここで最大輪郭を抽出し、中身を塗りつぶしたマスクを作ることで、
    # ・内部ノイズ（黒点）による矩形分断を防ぐ
    # ・外部ノイズ（背景ゴミ）を無視する
    contours, _ = cv2.findContours(binary_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        if verbose:
            logging.warning("[警告] 有効領域なし")
        return (0, 0, w - 1, h - 1)

    # 最大面積の輪郭を採用
    # 画面全体の 0.1% 以下のゴミは無視
    min_area = w * h * 0.001
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not valid_contours:
        main_contour = max(contours, key=cv2.contourArea)
    else:
        main_contour = max(valid_contours, key=cv2.contourArea)
        
    # 外接矩形を取得（バックアップ用）
    bx, by, bw, bh = cv2.boundingRect(main_contour)
    
    # クリーンなマスク (Full Res)
    clean_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(clean_mask, [main_contour], -1, 255, -1)
    
    # 3. 内接最大矩形の探索
    if aspect_ratio is not None:
        # 縦横比指定の場合
        x1, y1, x2, y2 = _find_largest_inscribed_rectangle_with_aspect(clean_mask, aspect_ratio)
    else:
        # 任意の縦横比の場合
        x1, y1, x2, y2 = _find_largest_inscribed_rectangle(clean_mask)
    
    # マージン適用（Smart Margin）
    margin_tol = max(w, h) * 0.01  # 1%未満の隙間なら埋める (吸着)
    
    if x1 < margin_tol: x1 = 0
    if y1 < margin_tol: y1 = 0
    if (w - 1 - x2) < margin_tol: x2 = w - 1
    if (h - 1 - y2) < margin_tol: y2 = h - 1
    
    # マージン適用（必要な場合のみ。基本0）
    if margin_ratio > 0:
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        mx = int(bbox_w * margin_ratio)
        my = int(bbox_h * margin_ratio)
        x1 += mx
        y1 += my
        x2 -= mx
        y2 -= my
        
    # 最終チェック（反転防止）
    if x2 <= x1 or y2 <= y1:
        return (0, 0, w-1, h-1)
        
    if verbose:
        logging.info("[最終結果] 内接矩形座標: x1=%s, y1=%s, x2=%s, y2=%s", x1, y1, x2, y2)
    
    return (x1, y1, x2, y2)


def find_largest_inscribed_rectangle_in_mask(mask, aspect_ratio=None, threshold=0.999, verbose=False):
    """
    変形後の有効画素マスク内に完全に収まる最大矩形を返す。

    `find_bounding_box` と違い、輪郭の塗りつぶしや穴埋めを行わないため、
    Geometry 変形で生じた黒い余白を矩形内へ含めない用途に使う。
    """
    if mask.ndim == 3:
        mask = np.min(mask[:, :, :3], axis=2)

    valid_mask = (mask >= threshold).astype(np.uint8) * 255
    h, w = valid_mask.shape[:2]

    if np.max(valid_mask) == 0:
        if verbose:
            logging.warning("[警告] 有効マスク領域なし")
        return (0, 0, 0, 0)

    if aspect_ratio is not None:
        x1, y1, x2, y2 = _find_largest_inscribed_rectangle_with_aspect(valid_mask, aspect_ratio)
    else:
        x1, y1, x2, y2 = _find_largest_inscribed_rectangle(valid_mask)

    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(x1, min(int(x2), w - 1))
    y2 = max(y1, min(int(y2), h - 1))

    if verbose:
        logging.info("[有効マスク最大矩形] x1=%s, y1=%s, x2=%s, y2=%s", x1, y1, x2, y2)

    return (x1, y1, x2, y2)

@lock_numba
@jit(nopython=True, cache=True)
def _largest_rectangle_in_histogram_jit(histogram):
    """
    ヒストグラムにおける最大矩形を求める（Numba JIT最適化版）
    
    Parameters:
    -----------
    histogram : numpy array
        各列の高さ
    
    Returns:
    --------
    tuple : (max_area, left, right, height)
    """
    stack = []
    max_area = 0
    max_left = 0
    max_right = 0
    max_height = 0
    index = 0
    n = len(histogram)
    
    while index < n:
        if len(stack) == 0 or histogram[index] >= histogram[stack[-1]]:
            stack.append(index)
            index += 1
        else:
            top = stack.pop()
            width = index if len(stack) == 0 else index - stack[-1] - 1
            area = histogram[top] * width
            
            if area > max_area:
                max_area = area
                max_height = histogram[top]
                max_right = index - 1
                max_left = 0 if len(stack) == 0 else stack[-1] + 1
    
    while len(stack) > 0:
        top = stack.pop()
        width = index if len(stack) == 0 else index - stack[-1] - 1
        area = histogram[top] * width
        
        if area > max_area:
            max_area = area
            max_height = histogram[top]
            max_right = index - 1
            max_left = 0 if len(stack) == 0 else stack[-1] + 1
    
    return (max_area, max_left, max_right, max_height)


def _largest_rectangle_in_histogram(histogram):
    """
    ヒストグラムにおける最大矩形を求める
    
    Parameters:
    -----------
    histogram : list or array
        各列の高さ
    
    Returns:
    --------
    tuple : (max_area, left, right, height)
    """
    if HAS_NUMBA:
        return _largest_rectangle_in_histogram_jit(np.asarray(histogram, dtype=np.int32))
    
    # Numbaがない場合のフォールバック
    stack = []
    max_area = 0
    max_rect = (0, 0, 0, 0)
    index = 0
    
    while index < len(histogram):
        if not stack or histogram[index] >= histogram[stack[-1]]:
            stack.append(index)
            index += 1
        else:
            top = stack.pop()
            width = index if not stack else index - stack[-1] - 1
            area = histogram[top] * width
            
            if area > max_area:
                max_area = area
                height = histogram[top]
                right = index - 1
                left = 0 if not stack else stack[-1] + 1
                max_rect = (max_area, left, right, height)
    
    while stack:
        top = stack.pop()
        width = index if not stack else index - stack[-1] - 1
        area = histogram[top] * width
        
        if area > max_area:
            max_area = area
            height = histogram[top]
            right = index - 1
            left = 0 if not stack else stack[-1] + 1
            max_rect = (max_area, left, right, height)
    
    return max_rect


def _find_largest_inscribed_rectangle(mask):
    """
    バイナリマスク内の最大内接矩形を検出（高速化版）
    
    ヒストグラム法を使用して正確な最大内接矩形を計算。
    NumPyベクトル化により高速化。
    
    Parameters:
    -----------
    mask : numpy.ndarray
        バイナリマスク (255が有効領域)
    
    Returns:
    --------
    tuple : (x1, y1, x2, y2) 矩形の座標
    """
    h, w = mask.shape
    binary = (mask > 0).astype(np.int32)
    
    # NumPyベクトル化で高さマップを高速計算
    heights = np.zeros((h, w), dtype=np.int32)
    heights[0] = binary[0]
    
    # ベクトル化: 上方向に連続する1の数を累積
    for i in range(1, h):
        heights[i] = np.where(binary[i] == 1, heights[i-1] + 1, 0)
    
    # 各行でヒストグラムベースの最大矩形を探索
    max_area = 0
    best_rect = None
    
    for i in range(h):
        area, left, right, height = _largest_rectangle_in_histogram(heights[i])
        
        if area > max_area:
            max_area = area
            # 矩形の座標を計算
            y2 = i
            y1 = i - height + 1
            x1 = left
            x2 = right
            best_rect = (x1, y1, x2, y2)
    
    if best_rect is None or max_area == 0:
        # フォールバック: 外接矩形を返す
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(main_contour)
            return (x, y, x + cw - 1, y + ch - 1)
        return (0, 0, w-1, h-1)
    
    return best_rect

@lock_numba
@jit(nopython=True, cache=True)
def _find_aspect_rect_jit(heights, aspect_ratio, step):
    """
    縦横比指定の内接矩形探索（Numba JIT最適化版）
    
    Parameters:
    -----------
    heights : numpy.ndarray (int32)
        高さマップ
    aspect_ratio : float
        幅/高さの比率
    step : int
        行のサンプリング間隔
    
    Returns:
    --------
    tuple : (max_area, x1, y1, x2, y2)
    """
    h, w = heights.shape
    max_area = 0
    best_x1, best_y1, best_x2, best_y2 = 0, 0, 0, 0
    
    # 各行をスキャン（サンプリング）
    # 本関数は @jit(nopython=True) のみで parallel=True が付いていないため、
    # prange は実質 range として実行される。max_area 等の共有スカラを書き換える
    # ボディなので parallel=True を付けると競合するが、そもそも並列化されていない。
    for row_idx in range(0, h, step):
        # 各開始位置をスキャン
        for start in range(w):
            if heights[row_idx, start] == 0:
                continue
            
            min_h = heights[row_idx, start]
            
            # 右に伸ばしながら探索
            max_end = min(start + int(min_h * aspect_ratio * 2), w)
            for end in range(start, max_end):
                if heights[row_idx, end] == 0:
                    break
                
                if heights[row_idx, end] < min_h:
                    min_h = heights[row_idx, end]
                
                width = end - start + 1
                required_height = int(width / aspect_ratio)
                
                # 必要な高さが利用可能な高さを超えたら終了
                if required_height > min_h:
                    break
                
                if required_height > 0:
                    area = width * required_height
                    if area > max_area:
                        max_area = area
                        best_y2 = row_idx
                        best_y1 = row_idx - required_height + 1
                        best_x1 = start
                        best_x2 = end
    
    return (max_area, best_x1, best_y1, best_x2, best_y2)


def _find_largest_inscribed_rectangle_with_aspect(mask, aspect_ratio):
    """
    指定された縦横比で最大内接矩形を探索
    
    Parameters:
    -----------
    mask : numpy.ndarray
        バイナリマスク (0 または 255)
    aspect_ratio : float
        幅/高さの比率（例: 16/9=1.778, 4/3=1.333, 1.0=正方形）
    
    Returns:
    --------
    tuple : (x1, y1, x2, y2) 矩形の座標
    """
    h, w = mask.shape
    binary = (mask > 0).astype(np.int32)
    
    # 高さマップを計算
    heights = np.zeros((h, w), dtype=np.int32)
    heights[0] = binary[0]
    for i in range(1, h):
        heights[i] = np.where(binary[i] == 1, heights[i-1] + 1, 0)
    
    # サンプリング間隔を決定（最大200行をスキャン）
    step = max(1, h // 200)
    
    # Numba JIT版を呼び出し
    if HAS_NUMBA:
        max_area, x1, y1, x2, y2 = _find_aspect_rect_jit(heights, aspect_ratio, step)
    else:
        # Numbaがない場合のフォールバック
        max_area = 0
        best_rect = None
        
        for row_idx in range(0, h, step):
            histogram = heights[row_idx]
            
            for start in range(w):
                if histogram[start] == 0:
                    continue
                
                min_h = histogram[start]
                max_end = min(start + int(min_h * aspect_ratio * 2), w)
                
                for end in range(start, max_end):
                    if histogram[end] == 0:
                        break
                    
                    min_h = min(min_h, histogram[end])
                    width = end - start + 1
                    required_height = int(width / aspect_ratio)
                    
                    if required_height > min_h:
                        break
                    
                    if required_height > 0:
                        area = width * required_height
                        if area > max_area:
                            max_area = area
                            y2 = row_idx
                            y1 = row_idx - required_height + 1
                            x1 = start
                            x2 = end
                            best_rect = (x1, y1, x2, y2)
        
        if best_rect:
            return best_rect
        max_area = 0
        x1, y1, x2, y2 = 0, 0, 0, 0
    
    # 矩形が見つからない場合は外接矩形を返す
    if max_area == 0:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
            return (x, y, x + cw - 1, y + ch - 1)
        return (0, 0, 0, 0)
    
    return (x1, y1, x2, y2)
