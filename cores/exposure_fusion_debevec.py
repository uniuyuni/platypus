import cv2
import numpy as np
import os
import logging


logger = logging.getLogger(__name__)


def _to_merge_input(img):
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = np.clip(img, 0.0, 1.0)
    return np.round(img * 255.0).astype(np.uint8)

def exposure_fusion_debevec(img, out_ldr=False):
    """
    単一画像から linear float32 HDR を生成（スケール正規化版）
    
    Returns:
        hdr_linear (np.ndarray): linear float32, 値域は物理輝度 [0, ∞)
        ldr_linear (np.ndarray): トーンマッピング済み（確認用）
    """
    # 2. 露出ブラケットのシミュレート（sRGB空間）
    def simulate_ev(img, ev):
        return img * (2.0 ** ev)

    images = [
        _to_merge_input(simulate_ev(img, -2.0)),
        _to_merge_input(simulate_ev(img,  0.0)),
        _to_merge_input(simulate_ev(img,  2.0))
    ]
    
    # 3. 【重要】露出時間の「相対比率のみ」を反映（絶対値は1.0に正規化）
    # EV差 ±2 → 露出比 1:4:16 → 中央を1.0に正規化
    exposure_times = np.array([0.25, 1.0, 4.0], dtype=np.float32)

    # 4. DebevecでHDR合成
    merge_debevec = cv2.createMergeDebevec()
    hdr_raw = merge_debevec.process(images, times=exposure_times)

    # 5. 【重要】出力スケールの正規化（中央露出画像の中間輝度を基準に調整）
    # 物理的に正確な絶対輝度は不可能なため、相対的な階調を保持するようスケーリング
    mid_ref = np.percentile(img[..., 1], 50)  # 緑チャネルの中央値（輝度代理）
    hdr_mid = np.percentile(hdr_raw[..., 1], 50)
    if hdr_mid > 1e-6:
        scale = mid_ref / hdr_mid
        hdr_linear = hdr_raw * scale
    else:
        hdr_linear = hdr_raw.copy()
    
    # 露出バイアスで暗く
    #hdr_linear = hdr_linear * (2.0 ** -1.0)

    # 7. 確認用プレビュー：Reinhard トーンマッピング
    if out_ldr == True:
        #tonemap = cv2.createTonemapReinhard(gamma=1.0, intensity=-0.2, light_adapt=0.95, color_adapt=1.0)
        tonemap = cv2.createTonemapMantiuk(gamma=1.0, scale=0.85, saturation=1.1)
        #tonemap = cv2.createTonemapDrago(gamma=1.0, saturation=1.0, bias=0.85)
        ldr_linear = np.clip(tonemap.process(hdr_linear), 0, 1)
    else:
        ldr_linear = None

    hdr_linear = np.nan_to_num(hdr_linear, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    logger.debug("HDR生成完了")
    logger.debug(
        "正規化後値域 HDR: Min=%.4f, Max=%.4f, Mean=%.4f",
        hdr_linear.min(),
        hdr_linear.max(),
        hdr_linear.mean(),
    )
    logger.debug("dtype: %s, shape: %s", hdr_linear.dtype, hdr_linear.shape)
    if ldr_linear is not None:
        ldr_linear = np.nan_to_num(ldr_linear, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        logger.debug(
            "正規化後値域 LDR: Min=%.4f, Max=%.4f, Mean=%.4f",
            ldr_linear.min(),
            ldr_linear.max(),
            ldr_linear.mean(),
        )
        logger.debug("dtype: %s, shape: %s", ldr_linear.dtype, ldr_linear.shape)
    
    return (hdr_linear, ldr_linear)

# 実行例
if __name__ == "__main__":
    img_path = "./tests/hdr_sample.jpg"

    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {img_path}")
    img = img.astype(np.float32) / 255.0
    img = np.power(img, 2.2)

    # 適当な画像を配置して実行してください
    hdr, ldr = exposure_fusion_debevec(img, out_ldr=True)
    hdr = np.power(hdr, 1.0 / 2.2)
    hdr = np.clip(hdr * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite("./tests/hdr_result_hdr.png", hdr)

    ldr = np.power(ldr, 1.0 / 2.2)
    ldr = np.clip(ldr * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite("./tests/hdr_result_ldr.png", ldr)
    
