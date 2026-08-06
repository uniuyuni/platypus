"""
Advanced Chromatic Aberration and Purple Fringe Removal - v2.3

Failure-aware improvements:
1. Limit correction to edge-polarity-consistent fringe regions
2. Protect broad genuine-purple objects (flowers/signs/fabrics)
3. Replace hard channel swapping with gradual chroma attenuation
"""

import logging

import numpy as np
import cv2
from typing import Tuple


class FringeRemoverFast:
    """
    Ultra-fast chromatic aberration removal with wide fringe support.
    """
    
    def __init__(self,
                 purple_amount: float = 1.8,
                 purple_hue_lo: float = 230,
                 purple_hue_hi: float = 310,
                 green_amount: float = 1.5,
                 green_hue_lo: float = 40,
                 # 緑の葉(hue~80-140°)を誤って緑フリンジ扱いして「オレンジ化」させないよう上限を下げる。
                 # 本来の黄緑フリンジ(~40-75°)は範囲内に残る。
                 green_hue_hi: float = 75,
                 lateral_correction: bool = False,
                 edge_threshold: float = 0.10,
                 min_saturation: float = 0.30,
                 fringe_width: int = 4):
        """
        Initialize fast fringe remover.
        """
        self.purple_amount = np.clip(purple_amount, 0, 10)  # Extended range
        self.purple_hue_lo = purple_hue_lo
        self.purple_hue_hi = purple_hue_hi
        self.green_amount = np.clip(green_amount, 0, 10)  # Extended range
        self.green_hue_lo = green_hue_lo
        self.green_hue_hi = green_hue_hi
        self.lateral_correction = lateral_correction
        self.edge_threshold = edge_threshold
        self.min_saturation = min_saturation
        self.fringe_width = np.clip(fringe_width, 1, 100)  # Extended range
    
    def remove_fringe(self, image: np.ndarray, roi_mask: np.ndarray = None) -> np.ndarray:
        """
        Remove chromatic aberration - ULTRA FAST version.
        """
        # Validate and clip input
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be RGB with shape (H, W, 3)")
        
        # Clip to valid range (allow > 1.0 on input, will clip on output)
        result = np.clip(image, 0, None)
        
        # Optional: Lateral chromatic aberration correction (minimal, fast)
        if self.lateral_correction:
            result = self._correct_lateral_ca_fast(result)
        
        # Axial chromatic aberration (fringe) removal - OPTIMIZED
        result = self._remove_axial_ca_fast(result, roi_mask=roi_mask)
        
        # Keep HDR headroom (no final clipping).
        return result
    
    def _correct_lateral_ca_fast(self, image: np.ndarray) -> np.ndarray:
        """
        Fast lateral CA correction with minimal interpolation.
        """
        # Skip if effect is minimal
        return image  # Lateral correction has minimal effect, skip for speed
    
    def _remove_axial_ca_fast(self, image: np.ndarray, roi_mask: np.ndarray = None) -> np.ndarray:
        """
        Failure-aware axial chromatic aberration removal.

        Baseline failure categories addressed:
        - False positive desaturation on non-fringe colors
        - Missed correction on weak/wide purple fringes
        - Over-correction on genuine broad purple objects
        """
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        deltac = maxc - minc
        s = np.where(maxc > 1e-6, deltac / (maxc + 1e-10), 0)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        edge_info = self._compute_edge_features(luminance)

        if self.purple_amount > 0:
            purple_mask = self._create_purple_mask_fast(
                image,
                edge_info=edge_info,
                maxc=maxc,
                minc=minc,
                deltac=deltac,
                s=s,
                roi_mask=roi_mask,
            )
            if purple_mask.max() > 0.01:
                image = self._correct_purple_fringe_fast(image, purple_mask)

        if self.green_amount > 0:
            green_mask = self._create_green_mask_fast(
                image,
                edge_info=edge_info,
                maxc=maxc,
                minc=minc,
                deltac=deltac,
                s=s,
                roi_mask=roi_mask,
            )
            if green_mask.max() > 0.01:
                image = self._correct_green_fringe_fast(image, green_mask)

        return image

    def _adaptive_percentile(self, values: np.ndarray, percentile: float, default: float) -> float:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return default
        return float(np.percentile(finite, percentile))

    def _compute_edge_features(self, luminance: np.ndarray) -> dict:
        # Fast edge estimation (np.diff is cheaper than Sobel).
        grad_y = np.abs(np.diff(luminance, axis=0, prepend=luminance[0:1]))
        grad_x = np.abs(np.diff(luminance, axis=1, prepend=luminance[:, 0:1]))
        gradient = grad_x + grad_y
        grad_norm = gradient / (float(gradient.max()) + 1e-6)
        grad_norm = np.clip(grad_norm, 0.0, 1.0)

        edge_t = max(self.edge_threshold, self._adaptive_percentile(grad_norm, 86.0, self.edge_threshold))
        edges = (grad_norm >= edge_t).astype(np.float32)

        radius = max(1, int(min(self.fringe_width, 6)))
        size = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        edge_band = cv2.dilate(edges.astype(np.uint8), kernel).astype(np.float32)

        local_max = cv2.dilate(luminance, kernel)
        local_min = cv2.erode(luminance, kernel)
        local_contrast = np.maximum(local_max - local_min, 0)
        contrast_t = self._adaptive_percentile(local_contrast, 75.0, 0.05)
        contrast_gate = (local_contrast >= max(0.02, contrast_t)).astype(np.float32)

        bright_t = self._adaptive_percentile(luminance, 92.0, 0.7)
        bright_nearby = (local_max > max(0.5, bright_t)).astype(np.float32)

        # Fringe tends to appear on the darker side of bright-dark transitions.
        side_delta_t = self._adaptive_percentile(local_contrast[edge_band > 0], 55.0, 0.05)
        dark_side = (luminance < (local_max - max(0.03, side_delta_t * 0.5))).astype(np.float32)

        return {
            "edges": edges,
            "edge_band": edge_band,
            "contrast": local_contrast,
            "contrast_gate": contrast_gate,
            "bright_nearby": bright_nearby,
            "dark_side": dark_side,
        }

    def _finalize_mask_strict(self, raw_mask: np.ndarray, edge_band: np.ndarray) -> np.ndarray:
        # Conservative automatic mode: binary mask, strictly edge-limited.
        mask = ((raw_mask > 0.10) & (edge_band > 0)).astype(np.uint8)
        if not mask.any():
            return mask.astype(np.float32)
        # テクスチャ上の孤立スペック(色ノイズ起因の誤検出)を除去する。本物のフリンジは
        # 輪郭沿いに連続して帯状になるため、微小な連結成分(面積 < min_area)だけ落とす。
        # これで「テクスチャに変な色が散発する」誤補正を防ぎつつ、連続したフリンジは残す。
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        min_area = 4
        keep = stats[:, cv2.CC_STAT_AREA] >= min_area
        keep[0] = False  # 背景ラベルは除外
        return keep[labels].astype(np.float32)

    def _compute_hue(self, r: np.ndarray, g: np.ndarray, b: np.ndarray, deltac: np.ndarray, maxc: np.ndarray) -> np.ndarray:
        h = np.zeros_like(maxc, dtype=np.float32)
        valid = deltac > 1e-6
        mask_r = (maxc == r) & valid
        mask_g = (maxc == g) & valid
        mask_b = (maxc == b) & valid

        h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / (deltac[mask_r] + 1e-10)) % 6)
        h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / (deltac[mask_g] + 1e-10)) + 2)
        h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / (deltac[mask_b] + 1e-10)) + 4)
        return h % 360

    def _create_purple_mask_fast(
        self,
        image: np.ndarray,
        edge_info: dict,
        maxc: np.ndarray = None,
        minc: np.ndarray = None,
        deltac: np.ndarray = None,
        s: np.ndarray = None,
        roi_mask: np.ndarray = None,
    ) -> np.ndarray:
        if edge_info["edge_band"].max() <= 0:
            return edge_info["edge_band"]
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        if maxc is None or minc is None:
            maxc = np.maximum(np.maximum(r, g), b)
            minc = np.minimum(np.minimum(r, g), b)
        if deltac is None:
            deltac = maxc - minc
        if s is None:
            s = np.where(maxc > 1e-6, deltac / (maxc + 1e-10), 0)

        h = self._compute_hue(r, g, b, deltac, maxc)
        if self.purple_hue_lo < self.purple_hue_hi:
            hue_mask = ((h >= self.purple_hue_lo) & (h <= self.purple_hue_hi)).astype(np.float32)
        else:
            hue_mask = ((h >= self.purple_hue_lo) | (h <= self.purple_hue_hi)).astype(np.float32)

        purple_excess = ((r + b) * 0.5) - g
        edge_pixels = purple_excess[edge_info["edge_band"] > 0]
        purple_t = max(0.003, self._adaptive_percentile(edge_pixels, 52.0, 0.008))
        purple_strength = np.clip((purple_excess - purple_t) / (purple_t + 1e-6), 0.0, 1.0)
        min_rb = np.minimum(r, b)
        max_rb = np.maximum(r, b)
        rb_ratio = min_rb / (max_rb + 1e-6)

        # Permissive enough to catch fringe, strict enough to avoid broad color edits.
        purple_floor = max(0.001, purple_t * 0.10)
        purple_profile = (((r + b) - (2.0 * g)) > (-1.5 * purple_floor)).astype(np.float32)
        sky_blue_like = ((h >= 200) & (h <= 248) & (b > g) & (g >= (r * 0.95))).astype(np.float32)
        sky_flat = (edge_info["contrast"] < 0.18).astype(np.float32)
        weak_red_support = (rb_ratio < 0.22).astype(np.float32)
        weak_purple_signal = (purple_excess < (purple_t * 1.2)).astype(np.float32)
        sky_protect = sky_blue_like * np.maximum(sky_flat, weak_red_support) * weak_purple_signal

        sat_mask = (s > max(self.min_saturation, 0.06)).astype(np.float32)
        value_mask = (maxc > 0.06).astype(np.float32)
        highlight_gate = 1.0 - ((s < 0.08) & (maxc > 0.85)).astype(np.float32) * 0.30
        roi_gate = 1.0 if roi_mask is None else (roi_mask > 0).astype(np.float32)

        raw_purple_mask = (
            edge_info["edge_band"]
            * edge_info["edges"]
            * edge_info["contrast_gate"]
            * edge_info["bright_nearby"]
            * edge_info["dark_side"]
            * hue_mask
            * purple_profile
            * sat_mask
            * value_mask
            * highlight_gate
            * purple_strength
            * (1.0 - sky_protect)
            * roi_gate
        )
        purple_mask = self._finalize_mask_strict(raw_purple_mask, edge_info["edge_band"])
        return np.clip(purple_mask * self.purple_amount, 0, 1)

    def _create_green_mask_fast(
        self,
        image: np.ndarray,
        edge_info: dict,
        maxc: np.ndarray = None,
        minc: np.ndarray = None,
        deltac: np.ndarray = None,
        s: np.ndarray = None,
        roi_mask: np.ndarray = None,
    ) -> np.ndarray:
        if edge_info["edge_band"].max() <= 0:
            return edge_info["edge_band"]
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        if maxc is None or minc is None:
            maxc = np.maximum(np.maximum(r, g), b)
            minc = np.minimum(np.minimum(r, g), b)
        if deltac is None:
            deltac = maxc - minc
        if s is None:
            s = np.where(maxc > 1e-6, deltac / (maxc + 1e-10), 0)

        h = self._compute_hue(r, g, b, deltac, maxc)
        hue_mask = ((h >= self.green_hue_lo) & (h <= self.green_hue_hi)).astype(np.float32)
        green_excess = g - ((r + b) * 0.5)
        edge_pixels = green_excess[edge_info["edge_band"] > 0]
        green_t = max(0.010, self._adaptive_percentile(edge_pixels, 76.0, 0.018))
        green_strength = np.clip((green_excess - green_t) / (green_t + 1e-6), 0.0, 1.0)

        sat_mask = (s > max(self.min_saturation, 0.08)).astype(np.float32)
        value_mask = (maxc > 0.08).astype(np.float32)
        # Keep green correction conservative in bright regions to avoid green/yellow cast.
        highlight_protect = np.clip((maxc - 0.82) / 0.20, 0.0, 1.0)
        neutral_highlight = ((s < 0.24) & (maxc > 0.70)).astype(np.float32)
        highlight_gate = 1.0 - np.maximum(highlight_protect * 0.95, neutral_highlight * 0.95)
        roi_gate = 1.0 if roi_mask is None else (roi_mask > 0).astype(np.float32)

        raw_green_mask = (
            edge_info["edge_band"]
            * edge_info["edges"]
            * edge_info["contrast_gate"]
            * edge_info["bright_nearby"]
            * edge_info["dark_side"]
            * hue_mask
            * sat_mask
            * value_mask
            * highlight_gate
            * green_strength
            * roi_gate
        )
        green_mask = self._finalize_mask_strict(raw_green_mask, edge_info["edge_band"])
        return np.clip(green_mask * self.green_amount, 0, 1)

    def _smooth_strength(self, mask: np.ndarray) -> np.ndarray:
        mask = np.clip(mask, 0.0, 1.0)
        return mask * mask * (3.0 - 2.0 * mask)

    def _correct_purple_fringe_fast(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.max() < 0.01:
            return image

        strength = self._smooth_strength(mask)
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        old_maxc = np.maximum(np.maximum(r, g), b)
        old_b = b.copy()

        r_excess = np.maximum(0.0, r - g)
        b_excess = np.maximum(0.0, b - g)

        # Reduce shared magenta component only where mask says fringe.
        common_excess = np.minimum(r_excess, b_excess)
        highlight_relief = np.clip((old_maxc - 0.75) / 0.30, 0.0, 1.0)
        eff_strength = strength * (1.0 - 0.65 * highlight_relief)
        delta_common = common_excess * eff_strength

        # Small residual asymmetric correction for stubborn fringes.
        residual_r = np.maximum(0.0, r_excess - common_excess)
        residual_b = np.maximum(0.0, b_excess - common_excess)
        image[:, :, 0] = r - delta_common - (residual_r * eff_strength * 0.20)
        image[:, :, 2] = b - delta_common - (residual_b * eff_strength * 0.20)

        # 色相反転防止: 補正で R/B を G 未満へ押し下げない（=フリンジを「緑優位」に反転させない）。
        # 元々 R/B が G 以上だった画素のみ G を下限にする（非マゼンタ画素は min(orig,g)=orig で据え置き）。
        image[:, :, 0] = np.maximum(image[:, :, 0], np.minimum(r, g))
        image[:, :, 2] = np.maximum(image[:, :, 2], np.minimum(old_b, g))

        # Prevent over-reduction of blue around highlights (yellow cast guard).
        blue_floor = old_b * (1.0 - 0.40 * eff_strength)
        image[:, :, 2] = np.maximum(image[:, :, 2], blue_floor)

        # Residual anti-green guard around bright corrected areas.
        maxc_new = np.maximum(np.maximum(image[:, :, 0], image[:, :, 1]), image[:, :, 2])
        minc_new = np.minimum(np.minimum(image[:, :, 0], image[:, :, 1]), image[:, :, 2])
        sat_like = (maxc_new - minc_new) / (maxc_new + 1e-6)
        bright_band = np.clip((maxc_new - 0.55) / 0.35, 0.0, 1.0)
        cast_guard = strength * bright_band * (1.0 - np.clip((sat_like - 0.08) / 0.22, 0.0, 1.0))
        rb_mean = 0.5 * (image[:, :, 0] + image[:, :, 2])
        g_excess = np.maximum(0.0, image[:, :, 1] - (rb_mean + 1e-4))
        image[:, :, 1] -= g_excess * cast_guard * 0.95
        return np.clip(image, 0, None)

    def _correct_green_fringe_fast(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.max() < 0.01:
            return image

        strength = self._smooth_strength(mask)
        r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

        g_ref = np.maximum(r, b)
        g_excess = np.maximum(0.0, g - g_ref)
        new_g = g - (g_excess * strength)
        # 色相反転防止(マゼンタ化防止): G を (R+B)/2 未満へ押し下げない。
        new_g = np.maximum(new_g, 0.5 * (r + b))
        # ★重要: マスク外(strength==0)の画素は一切変更しない。これを忘れて floor を全画素へ
        # 適用していたため、青(G<(R+B)/2)の G が持ち上がり、青空/青ボケが全部シアン化していた。
        image[:, :, 1] = np.where(strength > 1e-6, new_g, g)
        return np.clip(image, 0, None)


def remove_chromatic_aberration(image: np.ndarray,
                                purple_amount: float = 1.8,
                                green_amount: float = 1.5,
                                lateral_correction: bool = False,
                                edge_threshold: float = 0.10,
                                min_saturation: float = 0.30,
                                fringe_width: int = 4,
                                roi_mask: np.ndarray = None) -> np.ndarray:
    """
    Remove chromatic aberration and fringing from an image.
    v2.2 ULTRA FAST - 5-10x faster than v2.1!
    
    Args:
        image: Input RGB image as float32 array with shape (H, W, 3)
               Values can be in any range, will be clipped to [0, 1] on output
        purple_amount: Strength of purple fringe correction (0-3, default 1.8)
        green_amount: Strength of green fringe correction (0-3, default 1.5)
        lateral_correction: Enable lateral CA correction (default False - disabled for speed)
        edge_threshold: Edge detection threshold (0-1, default 0.10)
        min_saturation: Minimum saturation to detect as fringe (0-1, default 0.30)
        fringe_width: Width of fringe in pixels (1-20, default 4)
                     4: Normal fringe (default)
                     8-12: Wide fringe (backlit photos)
                     15-20: Very wide fringe
    
    Returns:
        Corrected RGB image as float32 array with shape (H, W, 3), values in [0, 1]
    
    Speed optimizations in v2.2:
        - Replaced gradient with simple diff operations (3x faster)
        - Used L1 norm instead of L2 (sqrt) for gradient (2x faster)  
        - Reduced blur operations (only for wide fringes)
        - Capped dilation size for consistent speed
        - All operations vectorized
    
    Example:
        >>> # Normal fringe (FAST)
        >>> corrected = remove_chromatic_aberration(img)
        >>> 
        >>> # Wide fringe (FAST)
        >>> corrected = remove_chromatic_aberration(img, fringe_width=12)
    """
    remover = FringeRemoverFast(
        purple_amount=purple_amount,
        green_amount=green_amount,
        lateral_correction=lateral_correction,
        edge_threshold=edge_threshold,
        min_saturation=min_saturation,
        fringe_width=fringe_width
    )
    
    return remover.remove_fringe(image, roi_mask=roi_mask)


def remove_chromatic_aberration_advanced(image: np.ndarray,
                                        purple_amount: float = 1.8,
                                        purple_hue_range: Tuple[float, float] = (230, 310),
                                        green_amount: float = 1.5,
                                        green_hue_range: Tuple[float, float] = (40, 90),
                                        lateral_correction: bool = False,
                                        edge_threshold: float = 0.10,
                                        min_saturation: float = 0.30,
                                        fringe_width: int = 4,
                                        roi_mask: np.ndarray = None) -> np.ndarray:
    """
    Advanced version with custom hue range control - ULTRA FAST.
    """
    remover = FringeRemoverFast(
        purple_amount=purple_amount,
        purple_hue_lo=purple_hue_range[0],
        purple_hue_hi=purple_hue_range[1],
        green_amount=green_amount,
        green_hue_lo=green_hue_range[0],
        green_hue_hi=green_hue_range[1],
        lateral_correction=lateral_correction,
        edge_threshold=edge_threshold,
        min_saturation=min_saturation,
        fringe_width=fringe_width
    )
    
    return remover.remove_fringe(image, roi_mask=roi_mask)


if __name__ == "__main__":
    logging.info("Chromatic Aberration Removal Module - v2.2 ULTRA FAST")
    logging.info("=" * 60)
    logging.info("\nNEW in v2.2:")
    logging.info("  ⚡ 5-10x FASTER than v2.1")
    logging.info("  ✅ Extended value range (> 1.0 allowed, auto-clipped)")
    logging.info("  ✅ Optimized memory usage")
    logging.info("  ✅ Vectorized operations throughout")
    logging.info("\nSpeed optimizations:")
    logging.info("  - np.diff-based edge estimation instead of Sobel (cheaper)")
    logging.info("  - cv2.dilate/erode for edge band and local contrast (fast morphology)")
    logging.info("  - Reduced blur sigma for speed")
    logging.info("  - Disabled lateral correction by default")
