
import os
import logging
import numpy as np
import json
import exiftool
import struct
import tempfile
from effect_backends import colour_functions_adapter as colour_functions
import pyvips
import subprocess
import cv2

import cores.core as core
import define
from enums import ImageFidelity
from imageset import ImageSet
import effects
import pipeline
import params
import effects
import config
from cores.mask2 import Mask2HeadlessPipeline
from utils import rating_io


def _export_cancel_requested(cancel_event):
    return cancel_event is not None and cancel_event.is_set()


class ExportFormatError(RuntimeError):
    pass


def _make_temp_output_path(final_path):
    directory = os.path.dirname(final_path) or "."
    base, ext = os.path.splitext(os.path.basename(final_path))
    fd, tmp_path = tempfile.mkstemp(prefix=f".{base}.", suffix=ext, dir=directory)
    os.close(fd)
    try:
        os.remove(tmp_path)
    except FileNotFoundError:
        # 既に存在しない場合はファイル名だけ確保できていれば十分（ベストエフォート）
        pass
    return tmp_path


def _cleanup_temp_output(path):
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        # 既に削除済みなら何もしなくてよい（ベストエフォートな後始末）
        pass
    except OSError:
        logging.exception("failed to remove temporary export file: %s", path)


_ICC_PROFILE_ALIASES = {
    'sRGB': 'sRGB',
    'sRGB IEC61966-2.1': 'sRGB',
    'Adobe RGB (1998)': 'Adobe RGB (1998)',
    'ProPhoto RGB': 'ProPhoto RGB',
    'ACES2065-1': 'ACES2065-1',
    'ACEScg': 'ACEScg',
    'Display P3': 'Display P3',
    'ITU-R BT.2020': 'Rec.2020',
    'ITU-R BT.709': 'Rec.709',
}

_ICC_PROFILE_CHROMATICITIES = {
    'WideGamut RGB': (
        0.7347, 0.2653,
        0.1152, 0.8264,
        0.1566, 0.0177,
        0.3457, 0.3585,
    ),
    'XYZD65': (
        1.0, 0.0,
        0.0, 1.0,
        0.0, 0.0,
        1.0 / 3.0, 1.0 / 3.0,
    ),
}

_NUMPY_DTYPE_TO_VIPS_FORMAT = {
    np.dtype(np.uint8): 'uchar',
    np.dtype(np.uint16): 'ushort',
    np.dtype(np.float32): 'float',
    np.dtype(np.float64): 'double',
}


def _numpy_to_vips_image(img):
    img = np.ascontiguousarray(img)
    if img.ndim == 2:
        height, width = img.shape
        bands = 1
    else:
        height, width, bands = img.shape

    format = _NUMPY_DTYPE_TO_VIPS_FORMAT[np.dtype(img.dtype)]
    return pyvips.Image.new_from_memory(img.tobytes(), width, height, bands, format)


def get_available_icc_profiles():
    try:
        names = [
            os.path.splitext(name)[0]
            for name in os.listdir('icc')
            if name.lower().endswith('.icc') and os.path.isfile(os.path.join('icc', name))
        ]
    except OSError:
        names = []

    names = sorted(set(names), key=str.casefold)
    if 'sRGB IEC61966-2.1' in names:
        names.remove('sRGB IEC61966-2.1')
        names.insert(0, 'sRGB IEC61966-2.1')
    elif not names:
        names = ['sRGB IEC61966-2.1']
    return names


def _icc_profile_path(icc_profile):
    return os.path.join('icc', f'{icc_profile}.icc')


def _s15fixed16(data):
    return struct.unpack('>i', data)[0] / 65536.0


def _u8fixed8(data):
    return struct.unpack('>H', data)[0] / 256.0


def _icc_tag_table(icc_profile):
    path = _icc_profile_path(icc_profile)
    with open(path, 'rb') as f:
        data = f.read()

    count = struct.unpack('>I', data[128:132])[0]
    tags = {}
    for i in range(count):
        offset = 132 + i * 12
        signature = data[offset:offset + 4].decode('latin1')
        tag_offset, tag_size = struct.unpack('>II', data[offset + 4:offset + 12])
        tags[signature] = data[tag_offset:tag_offset + tag_size]
    return tags


def _icc_xyz_tag_value(tag):
    if tag[:4] != b'XYZ ':
        raise ExportFormatError("ICC XYZ tag has unexpected type")
    return np.array([_s15fixed16(tag[8 + i * 4:12 + i * 4]) for i in range(3)], dtype=np.float64)


def _icc_profile_matrix_and_whitepoint(icc_profile):
    tags = _icc_tag_table(icc_profile)
    try:
        red = _icc_xyz_tag_value(tags['rXYZ'])
        green = _icc_xyz_tag_value(tags['gXYZ'])
        blue = _icc_xyz_tag_value(tags['bXYZ'])
        white = _icc_xyz_tag_value(tags['wtpt'])
    except KeyError as e:
        raise ExportFormatError(f"ICC profile is missing required RGB XYZ tag: {icc_profile}") from e

    matrix = np.column_stack([red, green, blue])
    return matrix, white


def _xy_from_xyz(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    total = float(np.sum(xyz))
    if total == 0:
        return np.array([0.0, 0.0], dtype=np.float64)
    return np.array([xyz[0] / total, xyz[1] / total], dtype=np.float64)


def _icc_curve_encoding(icc_profile):
    tags = _icc_tag_table(icc_profile)
    tag = tags.get('rTRC')
    if not tag:
        return ('linear',)

    tag_type = tag[:4]
    if tag_type == b'curv':
        count = struct.unpack('>I', tag[8:12])[0]
        if count == 0:
            return ('linear',)
        if count == 1:
            gamma = _u8fixed8(tag[12:14])
            if abs(gamma - 1.0) < 1e-6:
                return ('linear',)
            return ('gamma', gamma)
        if icc_profile == 'sRGB IEC61966-2.1':
            return ('srgb',)
        return ('table',)

    if tag_type == b'para':
        function_type = struct.unpack('>H', tag[8:10])[0]
        param_counts = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}
        count = param_counts.get(function_type)
        if count is None:
            return ('linear',)
        params = [_s15fixed16(tag[12 + i * 4:16 + i * 4]) for i in range(count)]
        if function_type == 0 and abs(params[0] - 1.0) < 1e-6:
            return ('linear',)
        return ('para', function_type, tuple(params))

    return ('linear',)


def _encode_profile_transfer(linear, icc_profile):
    linear = np.asarray(linear, dtype=np.float32)
    encoding = _icc_curve_encoding(icc_profile)
    kind = encoding[0]

    if kind == 'linear':
        return linear
    if kind == 'srgb':
        return colour_functions.linear_to_sRGB(linear).astype(np.float32)
    if kind == 'gamma':
        gamma = encoding[1]
        return np.where(linear >= 0, np.power(np.maximum(linear, 0), 1.0 / gamma), linear).astype(np.float32)
    if kind == 'para':
        _, function_type, params = encoding
        y = np.asarray(linear, dtype=np.float32)
        if function_type == 0:
            g, = params
            return np.where(y >= 0, np.power(np.maximum(y, 0), 1.0 / g), y).astype(np.float32)
        if function_type in (3, 4):
            g, a, b, c, d, *rest = params
            e = rest[0] if rest else 0.0
            f = rest[1] if len(rest) > 1 else 0.0
            split_y = c * d + (f if function_type == 4 else 0.0)
            high = (np.power(np.maximum(y - e, 0), 1.0 / g) - b) / a
            low = (y - (f if function_type == 4 else 0.0)) / c if c != 0 else y
            return np.where(y >= split_y, high, low).astype(np.float32)

    return linear


def _resize_export_image(img, resize_str):
    if not resize_str:
        return img

    try:
        scale = 1.0
        h, w = img.shape[:2]

        if resize_str.endswith('%'):
            scale = float(resize_str[:-1]) / 100.0
        else:
            parts = resize_str.lower().split('x')
            if len(parts) != 2:
                raise ValueError

            tx_str, ty_str = parts

            target_w = int(tx_str) if tx_str else None
            target_h = int(ty_str) if ty_str else None

            if target_w and target_h:
                scale = min(target_w / w, target_h / h)
            elif target_w:
                scale = target_w / w
            elif target_h:
                scale = target_h / h

        if scale != 1.0:
            target_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

    except ValueError:
        print(f"Export: Invalid resize string {resize_str}")

    return img


def _sharpen_export_image(img, sharpen):
    if sharpen <= 0:
        return img

    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sharpen), sigmaY=float(sharpen))
    return img + (img - blurred)


def _quantize_export_image(img, format, dithering):
    if format == '.EXR':
        return img.astype(np.float32, copy=False)

    img = np.clip(img, 0, 1).astype(np.float32, copy=False)

    if dithering:
        match format:
            case '.JPG' | '.JPEG' | '.JXL' | '.HEIF' | '.PNG':
                return core.jjn_dither_uint8(img)
            case '.TIF' | '.TIFF':
                return core.jjn_dither_uint16(img)
    else:
        match format:
            case '.JPG' | '.JPEG' | '.JXL' | '.HEIF' | '.PNG':
                return (img * 255).astype(np.uint8)
            case '.TIF' | '.TIFF':
                return (img * 65535).astype(np.uint16)

    return (img * 255).astype(np.uint8)


def _prepare_output_array(img, format, resize_str, sharpen, dithering):
    img = img.astype(np.float32, copy=False)
    img = _resize_export_image(img, resize_str)
    img = _sharpen_export_image(img, sharpen)
    return _quantize_export_image(img, format, dithering)


def _prepare_output_vips_image(img, format, resize_str, sharpen, dithering):
    img = _prepare_output_array(img, format, resize_str, sharpen, dithering)
    return _numpy_to_vips_image(img)


def _export_chromatic_adaptation_transform():
    try:
        return config.get_config('cat')
    except Exception:
        return 'cat16'


def _profile_colourspace_name(icc_profile):
    return _ICC_PROFILE_ALIASES.get(icc_profile, icc_profile)


def _convert_to_profile_linear(img, icc_profile, apply_gamut_mapping=True):
    colourspace_name = _profile_colourspace_name(icc_profile)
    if colourspace_name in colour_functions.RGB_COLOURSPACES:
        out = colour_functions.RGB_to_RGB(
            img,
            'ProPhoto RGB',
            colourspace_name,
            _export_chromatic_adaptation_transform(),
            apply_cctf_encoding=False,
            apply_gamut_mapping=False,
        )
        if apply_gamut_mapping:
            out = colour_functions.apply_RGB_gamut_mapping(out)
        return out.astype(np.float32)

    matrix, _white_xyz = _icc_profile_matrix_and_whitepoint(icc_profile)
    prophoto_xyz_d50 = colour_functions.RGB_to_XYZ(
        img,
        colourspace='ProPhoto RGB',
        illuminant_XYZ=colour_functions.ILLUMINANTS['D50'],
        chromatic_adaptation_transform=_export_chromatic_adaptation_transform(),
    )
    rgb = np.dot(prophoto_xyz_d50.reshape(-1, 3), np.linalg.inv(matrix).T)
    return rgb.reshape(img.shape).astype(np.float32)


def _convert_export_color(img, format, icc_profile):
    # EXR はシーンリニアの交換フォーマット。色域外・負値・HDR をそのまま保持するのが作法なので
    # ガマットマッピングを行わない（受け手アプリが chromaticities を見て表示変換する）。
    # 非 EXR は従来どおりガマット内に収めてから transfer 付与＋[0,1]クリップ。
    is_exr = (format == '.EXR')
    img = _convert_to_profile_linear(img, icc_profile, apply_gamut_mapping=not is_exr)
    if not is_exr:
        img = _encode_profile_transfer(img, icc_profile)
        img = np.clip(img, 0, 1) # ここじゃないとダメ
    return img


def _openexr_chromaticities_for_profile(icc_profile):
    if icc_profile in _ICC_PROFILE_CHROMATICITIES:
        return _ICC_PROFILE_CHROMATICITIES[icc_profile]

    colourspace_name = _profile_colourspace_name(icc_profile)
    colourspace = colour_functions.RGB_COLOURSPACES.get(colourspace_name)
    if colourspace is not None:
        primaries = np.asarray(colourspace.primaries, dtype=np.float32).reshape(3, 2)
        whitepoint = np.asarray(colourspace.whitepoint, dtype=np.float32)
        return tuple(float(v) for v in (*primaries.reshape(-1), *whitepoint))

    matrix, white_xyz = _icc_profile_matrix_and_whitepoint(icc_profile)
    primaries = np.array([_xy_from_xyz(matrix[:, i]) for i in range(3)], dtype=np.float32)
    whitepoint = _xy_from_xyz(white_xyz).astype(np.float32)
    return tuple(float(v) for v in (*primaries.reshape(-1), *whitepoint))


def _openexr_aces_image_container_flag(icc_profile):
    return 1 if icc_profile == 'ACES2065-1' else None


def _write_openexr_file(path, img, chromaticities=None, aces_image_container_flag=None):
    try:
        import OpenEXR
    except ImportError as e:
        raise ExportFormatError(
            "OpenEXR export requires the OpenEXR Python package. "
            "Run setup.sh or install requirements.txt in the pixi environment."
        ) from e

    img = np.ascontiguousarray(img.astype(np.float32, copy=False))
    if img.ndim == 2:
        channels = {"Y": img}
    elif img.ndim == 3 and img.shape[2] == 1:
        channels = {"Y": np.ascontiguousarray(img[:, :, 0])}
    elif img.ndim == 3 and img.shape[2] >= 3:
        channels = {"RGB": np.ascontiguousarray(img[:, :, :3])}
    else:
        raise ExportFormatError(f"Unsupported EXR image shape: {img.shape}")

    header = {
        "compression": OpenEXR.PIZ_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    if chromaticities is not None:
        header["chromaticities"] = chromaticities
    if aces_image_container_flag is not None:
        header["acesImageContainerFlag"] = int(aces_image_container_flag)

    with OpenEXR.File(header, channels) as outfile:
        outfile.write(path)


_SAFE_TAGS = [
    # EXIF（カメラと撮影設定）
    "EXIF:Make",
    "EXIF:Model",
    "EXIF:Software",
    "EXIF:ExposureTime",
    "EXIF:FNumber",
    "EXIF:ApertureValue",
    "EXIF:Aperture",
    "EXIF:ISO",
    "EXIF:ISOSpeedRatings",
    "EXIF:ShutterSpeedValue",
    "EXIF:ExposureProgram",
    "EXIF:ExposureCompensation",
    "EXIF:ExposureBiasValue",
    "EXIF:MeteringMode",
    "EXIF:Flash",
    "EXIF:FlashMode",
    "EXIF:WhiteBalance",
    "EXIF:FocalLength",
    "EXIF:FocalLengthIn35mmFormat",
    "EXIF:DigitalZoomRatio",
    "EXIF:LensModel",
    "EXIF:LensInfo",
    "EXIF:LensMake",
    "EXIF:LensSerialNumber",
    "EXIF:SceneCaptureType",
    "EXIF:Contrast",
    "EXIF:Saturation",
    "EXIF:Sharpness",
    "EXIF:SubjectDistance",
    "EXIF:SubjectDistanceRange",
    "EXIF:BrightnessValue",
    "EXIF:WhiteBalance",
    "EXIF:PictureMode",
    "EXIF:SubjectDistanceRange",
    
    # EXIF（基本情報）
    "EXIF:Artist",
    "EXIF:Copyright",
    "EXIF:ImageDescription",
    "EXIF:UserComment",
    "EXIF:XPTitle",
    "EXIF:XPComment",
    "EXIF:XPAuthor",
    "EXIF:XPKeywords",
    "EXIF:XPSubject",
    "EXIF:DocumentName",
    "EXIF:Orientation",
    #"EXIF:ImageWidth",
    #"EXIF:ImageHeight",
    #"EXIF:XResolution",
    #"EXIF:YResolution",
    #"EXIF:ResolutionUnit",
    
    # EXIF（日時情報）
    "EXIF:DateTimeOriginal",
    "EXIF:CreateDate",
    "EXIF:ModifyDate",
    "EXIF:DateTimeDigitized",
    
    # EXIF（GPS情報）
    "EXIF:GPSLatitude",
    "EXIF:GPSLongitude",
    "EXIF:GPSAltitude",
    "EXIF:GPSTimeStamp",
    "EXIF:GPSDateStamp",
    "EXIF:GPSProcessingMethod",
    "EXIF:GPSImgDirection",
    
    # IPTC（基本情報）
    "IPTC:ObjectName",
    "IPTC:Keywords",
    "IPTC:Caption-Abstract",
    "IPTC:Writer-Editor",
    "IPTC:Headline",
    "IPTC:SpecialInstructions",
    "IPTC:Byline",
    "IPTC:BylineTitle",
    "IPTC:Credit",
    "IPTC:Source",
    "IPTC:CopyrightNotice",
    "IPTC:Contact",
    
    # XMP（基本情報）
    "XMP:Title",
    "XMP:Description",
    "XMP:Creator",
    "XMP:Rights",
    "XMP:Subject",
    "XMP:Label",
    "XMP:Rating",
    "XMP:CreateDate",
    "XMP:ModifyDate"
]

def make_safe_metadata(exif_data, gpssw):
    safe_metadata = {}
    for tag in _SAFE_TAGS:
        group, field = tag.split(':')
        
        # タグが存在する場合のみ追加
        if field in exif_data:
            # gpsswがTrue、またはgpsswがFalseかつGPSタグではない場合に情報を保持する
            if gpssw == True or 'GPS' not in field:
                safe_metadata[field] = exif_data[field]
    return safe_metadata
  
class ExportFile():

    FORMAT = {
        '.JPG': 'JPEG',
        '.TIFF': 'TIFF',
        '.EXR': 'OpenEXR',
        '.JXL': 'JPEG XL',
        '.HEIF': 'HEIF',
        '.PNG': 'PNG',
    }

    def __init__(self, file_path, exif_data, export_rating: int = 0):
        self.file_path = str(file_path)
        self.exif_data = exif_data.copy()
        # メインで viewer / primary と同期した 0～5。スレッド安全のため起動前に退避
        self.export_rating = int(export_rating) if export_rating is not None else 0

        self.ex_path = None
        self.quality = 100
        self.icc_profile = "sRGB"
        self.imgset = None
        self.effects = effects.create_effects()
        self.param = {}
        self.mask_editor2 = None

    def write_to_file(
        self,
        ex_path,
        quality,
        resize_str,
        sharpen,
        icc_profile,
        exifsw,
        gpssw,
        dithering,
        cancel_event=None,
    ):
        if _export_cancel_requested(cancel_event):
            return False

        self.quality = quality
        self.ex_path = ex_path
        self.icc_profile = icc_profile
        self.imgset = ImageSet()
        result = self.imgset.preload(self.file_path, self.exif_data, self.param)
        self.imgset.load(result, self.file_path, self.exif_data, self.param)

        if _export_cancel_requested(cancel_event):
            return False

        params.apply_original_geometry_if_missing(self.param, self.imgset.img)
        if not params.has_original_img_size(self.param):
            logging.error("エクスポート中止: original_img_size を確定できません")
            return False

        self.mask_editor2 = Mask2HeadlessPipeline()

        #self.mask_editor2.set_orientation(self.param.get('rotation', 0), self.param.get('rotation2', 0), self.param.get('flip_mode', 0))
        self.mask_editor2.set_texture_size(self.imgset.img.shape[1], self.imgset.img.shape[0])
        self.mask_editor2.set_primary_param(self.param, params.get_disp_info(self.param))
        self.mask_editor2.set_ref_image(self.imgset.img, self.imgset.img)
        #self.mask_editor2.update()

        params.load_json(self.file_path, self.param, self.mask_editor2, load_heavy=True)
        self.param['_source_file_path'] = self.file_path
        self.param['image_fidelity'] = getattr(self.imgset, 'fidelity', ImageFidelity.FULL).value
        self.param.pop("rating", None)  # 星は param に入れない（er で書き出し）
        er = max(0, min(5, int(self.export_rating)))

        if _export_cancel_requested(cancel_event):
            return False

        img = pipeline.export_pipeline(self.imgset.img, self.effects, self.param, self.mask_editor2)

        if _export_cancel_requested(cancel_event):
            return False

        format = os.path.splitext(self.ex_path)[1].upper()
        img = _convert_export_color(img, format, self.icc_profile)

        if _export_cancel_requested(cancel_event):
            return False

        # Save options
        save_options = {}
        match format:
            case '.JPG' | '.JPEG' | '.HEIF' | '.JXL' | '.PNG':
                save_options['Q'] = self.quality
            case '.TIF' | '.TIFF':
                save_options['compression'] = 'deflate'

        # ディレクトリがなかったら作成
        ex_dir = os.path.dirname(self.ex_path)
        os.makedirs(ex_dir, exist_ok=True)
        tmp_ex_path = _make_temp_output_path(self.ex_path)

        if format == '.EXR':
            try:
                img = _prepare_output_array(img, format, resize_str, sharpen, dithering)
                if _export_cancel_requested(cancel_event):
                    return False
                _write_openexr_file(
                    tmp_ex_path,
                    img,
                    chromaticities=_openexr_chromaticities_for_profile(self.icc_profile),
                    aces_image_container_flag=_openexr_aces_image_container_flag(self.icc_profile),
                )
                if _export_cancel_requested(cancel_event):
                    return False
                os.replace(tmp_ex_path, self.ex_path)
                tmp_ex_path = None
                return True
            finally:
                _cleanup_temp_output(tmp_ex_path)

        vips_image = _prepare_output_vips_image(img, format, resize_str, sharpen, dithering)

        if _export_cancel_requested(cancel_event):
            return False

        # ICCプロファイルファイルを読み込む
        try:
            with open('icc/' + self.icc_profile + '.icc', 'rb') as f:
                icc_data = f.read()
            
            # 画像にICCプロファイルを設定
            vips_image = vips_image.copy()
            vips_image.set_type(pyvips.GValue.blob_type, 'icc-profile-data', icc_data)
            vips_image.set_type(pyvips.GValue.gstr_type, 'icc-profile-description', self.icc_profile)
            vips_image.set_type(pyvips.GValue.gstr_type, 'exif-ifd0-InterColorProfile', self.icc_profile)
            vips_image.set_type(pyvips.GValue.gint_type, 'exif-ifd0-ColorSpace', 0xfffe)

        except Exception as e:
            logging.warning(f"ICC profile {self.icc_profile} load failed: {e}")

        # ファイル書き込み
        if _export_cancel_requested(cancel_event):
            return False

        try:
            vips_image.write_to_file(tmp_ex_path, **save_options)
            del vips_image

            # 元EXIF 等の一括コピー（メタData プリセット）
            if exifsw and format != '.EXR':
                if _export_cancel_requested(cancel_event):
                    return False
                with exiftool.ExifToolHelper(common_args=['-P', '-overwrite_original']) as et:
                    ex_src = dict(self.exif_data) if self.exif_data else {}
                    ex_src.pop("Rating", None)
                    safe_metadata = make_safe_metadata(ex_src, gpssw)
                    safe_metadata["Software"] = define.APPNAME + " " + define.VERSION
                    et.set_tags(tmp_ex_path, tags=safe_metadata)

            # 星はメタ一括の ON/OFF に依存しない。メタ OFF でも exiftool で付与。
            if format != '.EXR' and not _export_cancel_requested(cancel_event) and os.path.isfile(tmp_ex_path):
                try:
                    rating_io.write_exported_file_rating(tmp_ex_path, int(er))
                except Exception:
                    logging.exception("export write rating to output file")

            if _export_cancel_requested(cancel_event):
                return False
            os.replace(tmp_ex_path, self.ex_path)
            tmp_ex_path = None
            return True
        finally:
            _cleanup_temp_output(tmp_ex_path)
