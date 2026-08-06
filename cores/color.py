# 現役: effects.py / imageset.py / main.py / cores/mask2/hls_mask.py などから import されている

import numpy as np
from functools import partial

_SRGB_UINT8_TO_LINEAR_LUT = None


def sRGB_to_linear_LUT(encoded):
    """uint8 sRGB画像を256段LUTでlinear float32へ変換する。"""
    global _SRGB_UINT8_TO_LINEAR_LUT
    if _SRGB_UINT8_TO_LINEAR_LUT is None:
        values = np.arange(256, dtype=np.float32) / 255.0
        _SRGB_UINT8_TO_LINEAR_LUT = srgb_gamma_decode(values).astype(np.float32)
    return _SRGB_UINT8_TO_LINEAR_LUT[np.asarray(encoded, dtype=np.uint8)]


# RGB Working Space のデータベース
M_RGB_WORKING_SPACES = {
    "sRGB": {
        "rgb_to_xyz": np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ], dtype=np.float32),
        "xyz_to_rgb": np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252]
        ], dtype=np.float32)
    },
    "Adobe RGB": {
        "rgb_to_xyz": np.array([
            [0.5767309, 0.1855540, 0.1881852],
            [0.2973769, 0.6273491, 0.0752741],
            [0.0270343, 0.0706872, 0.9911085]
        ], dtype=np.float32),
        "xyz_to_rgb": np.array([
            [ 2.0413690, -0.5649464, -0.3446944],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0134474, -0.1183897,  1.0154096]
        ], dtype=np.float32)
    },
    "ProPhoto RGB": {
        "rgb_to_xyz": np.array([
            [0.7976749, 0.1351917, 0.0313534],
            [0.2880402, 0.7118741, 0.0000857],
            [0.0000000, 0.0000000, 0.8252100]
        ], dtype=np.float32),
        "xyz_to_rgb": np.array([
            [ 1.3459433, -0.2556075, -0.0511118],
            [-0.5445989,  1.5081673,  0.0205351],
            [ 0.0000000,  0.0000000,  1.2118128]
        ], dtype=np.float32)
    },
    "Wide Gamut RGB": {
        "rgb_to_xyz": np.array([
            [0.7161046, 0.1009296, 0.1471858],
            [0.2581874, 0.7249378, 0.0168748],
            [0.0000000, 0.0517813, 0.7734287]
        ], dtype=np.float32),
        "xyz_to_rgb": np.array([
            [ 1.4628067, -0.1840623, -0.2743606],
            [-0.5217933,  1.4472381,  0.0677227],
            [ 0.0349342, -0.0968930,  1.2884065]
        ], dtype=np.float32)
    },
}

# 名前からマトリクスを取得する関数
def _get_matrix(space_name, direction="rgb_to_xyz"):
    """
    指定された RGB Working Space の変換行列を取得します。
    
    Parameters:
        space_name (str): RGB Working Space の名前 (例: "sRGB", "Adobe RGB")
        direction (str): "rgb_to_xyz" または "xyz_to_rgb"
    
    Returns:
        np.ndarray: 指定された方向の変換行列 (dtype=np.float32)
    """
    if space_name not in M_RGB_WORKING_SPACES:
        raise ValueError(f"Unknown RGB Working Space: {space_name}")
    if direction not in ["rgb_to_xyz", "xyz_to_rgb"]:
        raise ValueError("Direction must be 'rgb_to_xyz' or 'xyz_to_rgb'")
    
    return M_RGB_WORKING_SPACES[space_name][direction]

def rgb_to_xyz(rgb, space_name, gamma=False):

    if gamma == True:
        rgb = rgb_gamma_decode(rgb, space_name)

    xyz = np.dot(rgb, _get_matrix(space_name, "rgb_to_xyz").T)

    return xyz

def xyz_to_rgb(xyz, space_name, gamma=False):

    rgb = np.dot(xyz, _get_matrix(space_name, "xyz_to_rgb").T)
 
    # ガンマ補正
    if gamma == True:
        rgb = rgb_gamma_encode(rgb, space_name)

    return rgb

def _apply_chromatic_adaptation(XYZ, transform_matrix):
    """
    各 XYZ 値に対して、指定された変換行列を適用する関数。
    
    Parameters:
        XYZ (numpy.ndarray): 色データ。形状は (..., 3)。
        transform_matrix (numpy.ndarray): 3x3 の色彩適応変換行列。
        
    Returns:
        numpy.ndarray: 変換後の XYZ 値。形状は input と同じ。
    """
    # XYZ ... (..., 3) の場合、各色値ごとに行列積を計算
    # np.dot 利用の場合、axis=-1 を想定して計算するため reshape せずに済む
    return np.tensordot(XYZ, transform_matrix.T, axes=([-1], [0]))

def d50_to_d65(XYZ):
    """
    D50 白色点から D65 白色点への変換を実施する関数。
    
    Parameters:
        XYZ (numpy.ndarray): D50 白色点でキャリブレーションされた XYZ 値。形状は (..., 3)。
        
    Returns:
        numpy.ndarray: D65 白色点に適用された XYZ 値。形状は input と同じ。
    """
    # Bradford 法 による D50→D65 変換行列
    M_d50_to_d65 = np.array([
        [ 0.9555766, -0.0230393,  0.0631636],
        [-0.0282895,  1.0099416,  0.0210077],
        [ 0.0122982, -0.0204830,  1.3299098]
    ], dtype=np.float32)
    return _apply_chromatic_adaptation(XYZ, M_d50_to_d65)

M_D65_to_D50 = np.array([
    [ 1.0478112,  0.0228866, -0.0501270],
    [ 0.0295424,  0.9904844, -0.0170491],
    [-0.0092345,  0.0150436,  0.7521316]
], dtype=np.float32)

def d65_to_d50(XYZ):
    """
    D65 白色点から D50 白色点への変換を実施する関数。
    
    Parameters:
        XYZ (numpy.ndarray): D65 白色点でキャリブレーションされた XYZ 値。形状は (..., 3)。
        
    Returns:
        numpy.ndarray: D50 白色点に適用された XYZ 値。形状は input と同じ。
    """
    # Bradford 法 による D65→D50 変換行列
    return _apply_chromatic_adaptation(XYZ, M_D65_to_D50)

def srgb_gamma_encode(linear):
    """sRGBのガンマ補正"""
    return np.where(linear <= 0.0031308,
                   linear * 12.92,
                   1.055 * np.power(linear, 1/2.4) - 0.055)

def srgb_gamma_decode(encoded):
    """sRGBの逆ガンマ補正"""
    return np.where(encoded <= 0.04045,
                   encoded / 12.92,
                   np.power((encoded + 0.055) / 1.055, 2.4))

def adobe_rgb_gamma_encode(linear):
    """AdobeRGBのガンマ補正"""
    return np.where(linear <= 0.00304,
                   linear * 12.92,
                   1.055 * np.power(linear, 1/2.4) - 0.055)

def adobe_rgb_gamma_decode(encoded):
    """AdobeRGBの逆ガンマ補正"""
    return np.where(encoded <= 0.03928,
                   encoded / 12.92,
                   np.power((encoded + 0.055) / 1.055, 2.4))

def prophoto_rgb_gamma_encode(linear):
    """ProPhotoRGBのガンマ補正（γ = 1.8）"""
    return np.where(linear < 1/16,
                   linear * 16,
                   np.power(linear, 1/1.8))

def prophoto_rgb_gamma_decode(encoded):
    """ProPhotoRGBの逆ガンマ補正"""
    return np.where(encoded < 1/16,
                   encoded / 16,
                   np.power(encoded, 1.8))

def rgb_gamma_encode(linear, space_name):
    if space_name == "sRGB":
        return srgb_gamma_encode(linear)
    elif space_name == "Adobe RGB":
        return adobe_rgb_gamma_encode(linear)
    elif space_name == "ProPhoto RGB":
        return prophoto_rgb_gamma_encode(linear)
    
    return np.power(linear, 1/2.2)

def rgb_gamma_decode(encoded, space_name):
    if space_name == "sRGB":
        return srgb_gamma_decode(encoded)
    elif space_name == "Adobe RGB":
        return adobe_rgb_gamma_decode(encoded)
    elif space_name == "ProPhoto RGB":
        return prophoto_rgb_gamma_decode(encoded)
    
    return np.power(encoded, 2.2)
