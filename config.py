
import os
import json
import logging
import multiprocessing

from utils import paths

_config = None
_main_widget = None
_preview_texture_size = None

def init_config(widget):
    global _config, _main_widget, _preview_texture_size
    _config = multiprocessing.Manager().dict()
    _main_widget = widget

    _config.update(
    {
        'import_path': os.getcwd(),
        'lut_path': os.getcwd() + "/lut",
        'preview_size': 640,
        'ai_demosaic': False,
        'raw_auto_exposure': True,
        'scale_threshold': 0.5,
        'inpaint_resize_limit': 1024,
        'inpaint_use_realesrgan': True,
        'display_color_gamut': "auto", #"sRGB",
        'gpu_device': "mps",
        'cat': "cat16",
        'base_resolution_scale': [4096, 4096],
        'display_output_dither': False,
        'display_output_downscale': False,
        'debug_nan_inf_check': False,
        # float32 画像を .pmck などへ保存する際の圧縮形式。
        # 1: Zstd + byte shuffle (互換読み込み用)
        # 2: radiance_codec exact lossless (default)
        'image_codec_version': 2,
        # mesh warp (画像 mesh / マスク mesh) の変形手法。
        # 'mls' (default, Moving Least Squares affine: 補間性+局所性、内側シフトなし) |
        # 'thin_plate' (TPS, scipy 仕様で affine 遠方項) |
        # 'multiquadric' | 'inverse' | 'linear' | 'cubic' | 'quintic'
        # config.json に書き換え + アプリ再起動で切替。両者 (画像 / マスク mesh) に
        # 同時適用される (= 連動コピー前提の数値一致を維持)。
        'mesh_rbf_function': 'mls',
    })
    _preview_texture_size = (_config['preview_size'], _config['preview_size'])

    if not os.path.exists(paths.config_path()):
        save_config()


def get_preview_min_size():
    global _config
    if _config is None:
        return 640
    return int(_config['preview_size'])


def get_preview_texture_size():
    global _preview_texture_size
    if _preview_texture_size is None:
        size = get_preview_min_size()
        return (size, size)
    return _preview_texture_size


def get_preview_texture_side():
    width, height = get_preview_texture_size()
    return min(width, height)


def set_preview_texture_size(width, height):
    global _preview_texture_size
    width = max(1, int(round(width)))
    height = max(1, int(round(height)))
    _preview_texture_size = (width, height)

def get_config(key):
    global _config

    if key == 'preview_width':
        return get_preview_texture_size()[0]
    if key == 'preview_height':
        return get_preview_texture_size()[1]

    return _config[key]

def set_config(key, value):
    _config[key] = value
    if key == 'preview_size':
        width, height = get_preview_texture_size()
        min_size = get_preview_min_size()
        set_preview_texture_size(max(width, min_size), max(height, min_size))
    _apply_config(key)
    save_config()

def _apply_config(key):
    global _main_widget, _config
    if key == 'lut_path':
        _main_widget.set_lut_path(_config.get('lut_path', os.getcwd() + "/lut"))
    elif key == 'import_path':
        import_path = _config.get('import_path', os.getcwd())
        _main_widget.ids['viewer'].set_path(import_path)
        if hasattr(_main_widget, "on_import_path_applied"):
            _main_widget.on_import_path_applied(import_path)
    elif key in ['display_output_dither', 'display_output_downscale']:
        _main_widget.texture = None
    elif key == 'preview_size' and _main_widget is not None:
        if hasattr(_main_widget, "sync_preview_widget_min_size"):
            _main_widget.sync_preview_widget_min_size()

def apply_config():
    global _config
    for key in _config:
        _apply_config(key)

def save_config():
    global _config
    file_path = paths.config_path()
    with open(file_path, 'w') as f:
        json.dump(dict(_config), f)

def load_config():
    global _config
    file_path = paths.config_path()
    try:
        with open(file_path, 'r') as f:
            _config.update(json.load(f))
            min_size = get_preview_min_size()
            set_preview_texture_size(min_size, min_size)
            apply_config()
    except FileNotFoundError:
        pass
    except json.JSONDecodeError as e:
        logging.warning(f"設定ファイルの読み込みに失敗しました（不正な JSON）。デフォルト設定で続行します: {file_path} ({e})")
