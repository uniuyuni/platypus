
import os
import sys
#if __name__ == '__main__':
#    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
import cv2
import time
import uuid
import hashlib
import re
from enum import Enum
import copy
import logging
import importlib
import threading
from functools import partial

from kivy.app import App as KVApp
from kivy.config import Config as KVConfig
from kivy.core.window import Window as KVWindow
from kivy.uix.widget import Widget as KVWidget
from kivy.uix.image import Image as KVImage
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.uix.floatlayout import FloatLayout as KVFloatLayout
from kivy.properties import (
    NumericProperty as KVNumericProperty, ObjectProperty as KVObjectProperty, ListProperty as KVListProperty,
    StringProperty as KVStringProperty, BooleanProperty as KVBooleanProperty, Property as KVProperty
)
from kivy.graphics import (
    Color as KVColor, Ellipse as KVEllipse, Line as KVLine, PushMatrix as KVPushMatrix, PopMatrix as KVPopMatrix, Rotate as KVRotate, Translate as KVTranslate,
    Rectangle as KVRectangle, ScissorPush as KVScissorPush, ScissorPop as KVScissorPop,
)
from kivy.graphics.texture import Texture as KVTexture
from kivy.clock import Clock as KVClock
from kivy.uix.label import Label as KVLabel

import cores.core as core
from cores.ai_image_cache import AIImageCache
from cores.mask2 import mask_geometry as mask_geometry_mod
import params
import effects
import threads
import utils.utils as utils
import processing_dialog
from history import LayerCtrl, get_history_ctrl
import macos as device

from cores.mask2 import mask_rasters
from cores.mask2 import edge_refine
from cores.mask2 import extended_params
from cores.mask2 import cache_keys
from cores.mask2 import hls_mask


_DEBUG_MASK_GEOMETRY = os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"}
_DEBUG_MASK_ZOOM_SYNC = os.getenv("PLATYPUS_DEBUG_MASK_ZOOM_SYNC", "0").strip().lower() in {"1", "true", "yes", "on"}
_DEBUG_LIQUIFY = os.getenv("PLATYPUS_DEBUG_LIQUIFY", "0").strip().lower() in {"1", "true", "yes", "on"}


def _mask_geom_debug(message, *args):
    if _DEBUG_MASK_GEOMETRY:
        logging.warning("[MASK_GEOM] " + message, *args)


def _mask_zoom_sync_debug(message, *args):
    if _DEBUG_MASK_ZOOM_SYNC:
        logging.warning("[MASK_ZOOM_SYNC] " + message, *args)


def _liquify_debug(message, *args):
    if _DEBUG_LIQUIFY:
        logging.warning("[LIQUIFY] " + message, *args)


def _mask_geom_id(mask):
    if mask is None:
        return None
    mask_id = getattr(mask, "mask_id", "")
    short_id = str(mask_id)[:8] if mask_id else "no-id"
    return f"{mask.__class__.__name__}:{short_id}@{id(mask):x}"


def _mask_geom_matrix_hash(matrix):
    if matrix is None:
        return None
    try:
        return hash(np.asarray(matrix, dtype=np.float64).tobytes())
    except Exception:
        return None


def _hashable_cache_value(value):
    if isinstance(value, dict):
        return tuple(
            (str(k), _hashable_cache_value(v))
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_cache_value(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mask_geom_param_summary(param):
    if not param:
        return {}
    keys = (
        "switch_mask_geometry",
        "mask_rotation",
        "mask_translation_x",
        "mask_translation_y",
        "mask_scale_x",
        "mask_scale_y",
        "mask_flip_mode",
    )
    return {key: param.get(key) for key in keys if key in param}


def _mask_geom_image_stats(image):
    if image is None:
        return None
    try:
        return {
            "shape": tuple(int(v) for v in image.shape),
            "hash": hash(np.ascontiguousarray(image).tobytes()),
            "min": float(np.nanmin(image)),
            "max": float(np.nanmax(image)),
            "sum": float(np.nansum(image)),
            "nonzero": int(np.count_nonzero(image)),
        }
    except Exception:
        return {"shape": getattr(image, "shape", None)}


def _clip_mask_range(image, allow_over_one=False, allow_under_zero=False):
    min_value = None if allow_under_zero else 0
    max_value = None if allow_over_one else 1
    if min_value is None and max_value is None:
        return image
    return np.clip(image, min_value, max_value)


# mask Mesh の core ヘルパは cores/mask2/mask_mesh.py に集約済み (Kivy / Headless で共用)。
from cores.mask2.mask_mesh import (
    apply_mask_mesh_warp as _apply_mask_mesh_warp_shared,
    mesh_cps_hash_key as _mesh_cps_hash_key,
    mask_mesh_source_bounds as _mask_mesh_source_bounds_shared,
    normalize_mesh_cps as _normalize_mesh_cps,
)


def _linked_primary_mesh_hash_key(mask):
    """linked モードの Composit: 画像 mesh の CP が変わったらキャッシュ invalidate するため、
    画像 mesh の CP データを hash に含める。linked でないか primary_param が取れないなら空。"""
    linked = effects.Mask2Effect.get_param(mask.effects_param, 'mask_mesh_link_to_image')
    if not linked:
        return ()
    editor = getattr(mask, 'editor', None)
    root = getattr(editor, 'root', None) if editor is not None else None
    primary = getattr(root, 'primary_param', None) if root is not None else None
    if not primary:
        return ()
    return (
        tuple(primary.get('mesh_size') or ()),
        _mesh_cps_hash_key(primary.get('control_points')),
    )


def _axis_polyline_with_arrow(origin_win, end_win, arrow_len=12.0, arrow_angle_deg=25.0):
    """軸線 (origin -> end) の先端に矢印を付けた連続 polyline の points 列を返す。
    Kivy の Line は polyline なので、矢印の両羽を `... end -> tipL -> end -> tipR` の
    順で繋ぐと 1 つの Line で軸線+矢印を描画できる。"""
    ox, oy = origin_win
    ex, ey = end_win
    dx, dy = ex - ox, ey - oy
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (ox, oy, ex, ey)
    ux, uy = dx / length, dy / length
    a = math.radians(arrow_angle_deg)
    # 矢印の羽は end から見て「軸の逆向きを ±a 度回転した方向」へ
    cL, sL = math.cos(math.pi - a), math.sin(math.pi - a)
    cR, sR = math.cos(math.pi + a), math.sin(math.pi + a)
    tipL = (ex + arrow_len * (ux * cL - uy * sL),
            ey + arrow_len * (ux * sL + uy * cL))
    tipR = (ex + arrow_len * (ux * cR - uy * sR),
            ey + arrow_len * (ux * sR + uy * cR))
    return (ox, oy, ex, ey, tipL[0], tipL[1], ex, ey, tipR[0], tipR[1])


def _effective_mask_mesh_param(editor, effects_param):
    linked = effects.Mask2Effect.get_param(effects_param, 'mask_mesh_link_to_image')
    if linked:
        root = getattr(editor, 'root', None)
        primary = getattr(root, 'primary_param', None) if root is not None else None
        if primary is not None:
            merged = dict(effects_param)
            merged['mask_mesh_control_points'] = primary.get('control_points', {})
            merged['mask_mesh_size'] = primary.get('mesh_size', [4, 4])
            _mask_zoom_sync_debug(
                "mask_mesh effective linked=True primary_cps=%d local_cps=%d primary_hash=%s local_hash=%s",
                len(_normalize_mesh_cps(primary.get('control_points'))),
                len(_normalize_mesh_cps(effects.Mask2Effect.get_param(effects_param, 'mask_mesh_control_points'))),
                _mesh_cps_hash_key(primary.get('control_points')),
                _mesh_cps_hash_key(effects.Mask2Effect.get_param(effects_param, 'mask_mesh_control_points')),
            )
            return merged
    _mask_zoom_sync_debug(
        "mask_mesh effective linked=False cps=%d hash=%s",
        len(_normalize_mesh_cps(effects.Mask2Effect.get_param(effects_param, 'mask_mesh_control_points'))),
        _mesh_cps_hash_key(effects.Mask2Effect.get_param(effects_param, 'mask_mesh_control_points')),
    )
    return effects_param


def _mask_mesh_source_padding(editor, effects_param, output_shape):
    """現在 viewport 外を warp が参照できるよう、ソースマスク描画に必要な余白を見積もる。"""
    cps = _normalize_mesh_cps(effects.Mask2Effect.get_param(effects_param, 'mask_mesh_control_points'))
    if not cps:
        return 0
    try:
        disp = params.get_disp_info(editor.tcg_info)
        scale = float(disp[4])
        orig_w, orig_h = editor.tcg_info['original_img_size']
        max_move = 0.0
        for ox, oy in cps.values():
            max_move = max(max_move, abs(float(ox)) * float(orig_w), abs(float(oy)) * float(orig_h))
        # MLS の局所変形は CP 移動量より少し外側を参照することがあるため余裕を足す。
        pad = int(math.ceil(max_move * scale + 64.0))
        return max(0, min(pad, max(int(output_shape[0]), int(output_shape[1])) * 2))
    except Exception:
        return 0


def _mask_mesh_source_region(editor, effects_param, output_shape):
    """mask mesh warp が実際に参照する texture 範囲だけを source として描く。"""
    orig = editor.tcg_info.get('original_img_size') if isinstance(getattr(editor, 'tcg_info', None), dict) else None
    bounds = _mask_mesh_source_bounds_shared(
        effects_param,
        orig,
        getattr(editor, 'tcg_to_texture', None),
        getattr(editor, 'tcg_to_full_image', None),
        output_shape=output_shape,
    )
    if bounds is None:
        pad = _mask_mesh_source_padding(editor, effects_param, output_shape)
        if pad <= 0:
            return (int(output_shape[1]), int(output_shape[0])), (0.0, 0.0), 0
        return (
            (int(output_shape[1]) + pad * 2, int(output_shape[0]) + pad * 2),
            (-float(pad), -float(pad)),
            pad,
        )

    min_x, min_y, max_x, max_y = bounds
    out_h, out_w = int(output_shape[0]), int(output_shape[1])
    margin = 4
    origin_x = min(0, int(math.floor(min_x)) - margin)
    origin_y = min(0, int(math.floor(min_y)) - margin)
    end_x = max(out_w, int(math.ceil(max_x)) + margin + 1)
    end_y = max(out_h, int(math.ceil(max_y)) + margin + 1)
    max_expand = max(out_w, out_h) * 2
    origin_x = max(origin_x, -max_expand)
    origin_y = max(origin_y, -max_expand)
    end_x = min(end_x, out_w + max_expand)
    end_y = min(end_y, out_h + max_expand)
    source_w = max(1, end_x - origin_x)
    source_h = max(1, end_y - origin_y)
    pad_equiv = max(-origin_x, -origin_y, end_x - out_w, end_y - out_h, 0)
    return (source_w, source_h), (float(origin_x), float(origin_y)), int(pad_equiv)


def _apply_mask_mesh_warp(composit, editor, effects_param,
                          output_shape=None, source_origin_tex=(0.0, 0.0)):
    """共通ヘルパ cores.mask2.mask_mesh.apply_mask_mesh_warp への薄いラッパ。
    mask_mesh_link_to_image=True の Composit は **画像 mesh の CP を都度参照** する
    (Mask2Effect の mask_mesh_link_to_image 仕様)。False なら自前 CP。"""
    orig = editor.tcg_info.get('original_img_size') if isinstance(getattr(editor, 'tcg_info', None), dict) else None
    # t2t=texture px (disp_info込み), t2f=フル画像px (F=MLS構築空間)。マスク warp の
    # 共役 (射影込みでズーム位置がズレないため) に両方必要。
    t2t = getattr(editor, 'tcg_to_texture', None)
    t2f = getattr(editor, 'tcg_to_full_image', None)
    effective = _effective_mask_mesh_param(editor, effects_param)
    return _apply_mask_mesh_warp_shared(
        composit, effective, orig, t2t, t2f,
        output_shape=output_shape,
        source_origin_tex=source_origin_tex,
    )


class MaskType(str, Enum):
    COMPOSIT = 'composit'
    CIRCULAR = 'circular'
    GRADIENT = 'gradient'
    FULL = 'full'
    FREEDRAW = 'free_draw'
    POLYLINE = 'polyline'
    SEGMENT = 'segment'
    DEPTHMAP = 'depth_map'
    FACE = 'face'
    TARGET_TEXT = 'target_text'

# マスク「コピー」で引き継ぐ「形状決定系」の mask2_* effects_param キーのホワイトリスト。
# 調整・描画系(mask2_blend_mode, mask2_color_dodge/burn, mask2_mix_black/white,
# mask2_skin_smooth_amount 等)や非 mask2 キー(mask_rotation 等のマスク Geometry)は
# コピー対象外(新規デフォルトのまま)。実キー名は effects.py Mask2Effect.get_param_dict()
# の定義に準拠(widgets/mask2_content.py の copy 選択フローから参照される)。
MASK2_SHAPE_PARAM_KEYS = (
    'mask2_invert', 'mask2_allow_over_one', 'mask2_allow_under_zero',
    'mask2_blur', 'mask2_open_space', 'mask2_close_space',
    'mask2_freedraw_brush_size', 'mask2_freedraw_brush_hardness', 'mask2_polyline_fill',
    'mask2_hue_min', 'mask2_hue_max', 'mask2_hue_distance',
    'mask2_lum_min', 'mask2_lum_max', 'mask2_lum_distance',
    'mask2_sat_min', 'mask2_sat_max', 'mask2_sat_distance',
    'mask2_depth_min', 'mask2_depth_max', 'mask2_depth_balance',
    'mask2_edge_refine_mode', 'mask2_edge_refine_radius', 'mask2_edge_refine_strength', 'mask2_edge_refine_bias',
    'mask2_face_face', 'mask2_face_brows', 'mask2_face_eyes', 'mask2_face_nose', 'mask2_face_mouth', 'mask2_face_lips',
)

def _next_copy_name(name):
    """コピー時のマスク名を生成する。末尾の ' copy' / ' copy N' を検出して連番を振り、
    'copy copy ...' の連鎖を防ぐ。
      'Circle'        -> 'Circle copy'
      'Circle copy'   -> 'Circle copy 2'
      'Circle copy 2' -> 'Circle copy 3'
    """
    m = re.match(r'^(.*) copy(?: (\d+))?$', name)
    if m:
        base = m.group(1)
        n = int(m.group(2)) if m.group(2) else 1
        return f"{base} copy {n + 1}"
    return f"{name} copy"

# コントロールポイントのクラス
class ControlPoint(KVWidget):
    HIT_RADIUS_PX = 10.0

    touching = KVBooleanProperty(False)
    is_center = KVBooleanProperty(False)  # 中心のコントロールポイントかどうか
    color = KVListProperty([0, 0, 0])  # デフォルトの色
    ctrl_center = KVListProperty([0, 0])
    type = KVListProperty(['c', 0])

    def __init__(self, editor, **kwargs):
        super().__init__(**kwargs)
        self.editor = editor
        with self.canvas:
            KVPushMatrix()
            self.scissor = self.editor.push_scissor()
            self.translate = KVTranslate()
            #self.rotate = KVRotate(angle=0, origin=(0, 0))            
            self.color_instruction = KVColor(*self.color)
            self.circle = KVEllipse(pos=(-10, -10), size=(20, 20))
            self.editor.pop_scissor()
            KVPopMatrix()
        self.center = (0, 0)
        #self.update_graphics()
        self.bind(center=self.update_graphics, color=self.update_color)

    def collide_point(self, x, y):
        """Hit-test against the visible 20px control-point circle.

        Kivy's default widget hit area is size based (normally 100x100), while
        ControlPoint intentionally does not set widget size because that changes
        the TCG center values. Keep the hit-test in window pixels instead.
        """
        mask = getattr(self, 'parent', None)
        converter = getattr(mask, 'tcg_to_window_for_overlay', None)
        if not callable(converter):
            converter = getattr(self.editor, 'tcg_to_window', None)
        if callable(converter):
            wx, wy = converter(x, y)
            cx, cy = converter(self.center_x, self.center_y)
            dx = wx - cx
            dy = wy - cy
            return dx * dx + dy * dy <= self.HIT_RADIUS_PX * self.HIT_RADIUS_PX
        return super().collide_point(x, y)

    def update_graphics(self, *args):
        mask = getattr(self, 'parent', None)
        converter = getattr(mask, 'tcg_to_window_for_overlay', None)
        if callable(converter):
            cx, cy = converter(self.center_x, self.center_y)
        else:
            cx, cy = self.editor.tcg_to_window(self.center_x, self.center_y)
        self.translate.x = cx
        self.translate.y = cy
        self.editor.set_scissor(self.scissor)
        #self.size = self.editor.window_to_tcg_scale(20, 20) # sizeをセットすると何故かcenterの値がおかしくなるのでコメントアウト

    def update_color(self, *args):
        self.color_instruction.rgb = self.color

    def on_touch_down(self, touch):
        self.touching = True
        return True

    def on_touch_move(self, touch):
        if self.touching:
            mask = getattr(self, 'parent', None)
            converter = getattr(mask, 'window_to_tcg_for_interaction', None)
            if callable(converter):
                cx, cy = converter(*touch.pos)
            else:
                cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.ctrl_center = [cx, cy]
            #self.cnter = (cx, cy)
            return True
        return False

    def on_touch_up(self, touch):
        if self.touching:
            self.touching = False
            return True
        return False

        return False

# マスクのベースクラス
class BaseMask(KVWidget):
    color = KVListProperty([1, 0, 0, 0.5])  # デフォルトの半透明赤色
    selected = KVBooleanProperty(False)
    active = KVBooleanProperty(False)
    name = KVStringProperty("Mask")
    mask_id = KVStringProperty("")

    def __init__(self, editor, **kwargs):
        super().__init__(**kwargs)
        if not self.mask_id:
            self.mask_id = str(uuid.uuid4())
        self.editor = editor  # MaskEditorのインスタンスへの参照
        self.control_points = []  # 標準のPythonリストで管理
        self.bind(active=self.on_active_changed)

        # エフェクトパラメータ保持
        self.effects = self._create_effects()
        self.effects_param = {}
        params.set_image_param_for_mask2(self.effects_param, self.editor.get_image_size())
        params.set_temperature_to_param(self.effects_param, *core.invert_RGB2TempTint((1.0, 1.0, 1.0)))

        self.is_draw_mask = True
        self.image_mask_cache = None
        self.image_mask_cache_hash = None
        self.image_mask_cache_key = None
        self.do_draw_composit_mask = True
        self._initial_touch_started = False
        # CP ドラッグで begin_history_layer_ctrl("Update") を発行したかどうか。
        # begin と end のペアを厳密に保つためのフラグ(作成中は begin しない)。
        self._cp_history_started = False
        self._image_mask_pending_lock = threading.Lock()
        self._image_mask_pending_event = None
        self._image_mask_pending_key = None
        self._image_mask_pending_error = None

    def _create_effects(self):
        root = getattr(self.editor, 'root', None)
        distortion_callback = getattr(root, 'distortion_callback', None)
        light_rays_callback = getattr(root, 'light_rays_callback', None)
        view_param_provider = getattr(self.editor, 'get_effect_view_param', None)
        return effects.create_effects(
            distortion_callback=distortion_callback if callable(distortion_callback) else None,
            light_rays_callback=light_rays_callback if callable(light_rays_callback) else None,
            view_param_provider=view_param_provider if callable(view_param_provider) else None,
        )

    def _get_or_compute_image_mask_cache(self, cache_key, compute_func, label):
        while True:
            with self._image_mask_pending_lock:
                if self.image_mask_cache is not None and self.image_mask_cache_key == cache_key:
                    return self.image_mask_cache
                if self._image_mask_pending_event is None:
                    event = threading.Event()
                    self._image_mask_pending_event = event
                    self._image_mask_pending_key = cache_key
                    self._image_mask_pending_error = None
                    break
                event = self._image_mask_pending_event
                pending_key = self._image_mask_pending_key

            logging.info("%s prediction already running pending_key=%s requested_key=%s; waiting", label, pending_key, cache_key)
            event.wait()
            with self._image_mask_pending_lock:
                pending_error = self._image_mask_pending_error
            if pending_error is not None:
                raise pending_error

        # 共有ストア(AIImageCache)に同一キーの結果が既にあれば推論をスキップして再利用する。
        # DepthMapMask の self.editor.get_ai_depth_map と対称の到達経路。
        store_get = getattr(self.editor, "get_ai_mask_bitmap", None)
        cached = store_get(cache_key) if callable(store_get) else None
        if cached is not None:
            result = cached
        else:
            try:
                result = compute_func()
            except BaseException as exc:
                with self._image_mask_pending_lock:
                    if self._image_mask_pending_event is event:
                        self._image_mask_pending_error = exc
                        self._image_mask_pending_event = None
                        self._image_mask_pending_key = None
                        self.image_mask_cache_key = None
                    event.set()
                raise
            store_put = getattr(self.editor, "put_ai_mask_bitmap", None)
            if callable(store_put):
                store_put(cache_key, result)

        with self._image_mask_pending_lock:
            if self._image_mask_pending_event is event:
                self.image_mask_cache = result
                self.image_mask_cache_key = cache_key
                self._image_mask_pending_error = None
                self._image_mask_pending_event = None
                self._image_mask_pending_key = None
            event.set()
        return result

    def _serialize_image_mask_cache(self, dict):
        """AI マスク(Segment/Face/TargetText)用: インラインを出さず参照キーのみ保存する。
        実体は AIImageCache 共有ストア(image_mask_cache_key で参照)側にある。"""
        if self.image_mask_cache_key is not None:
            dict['image_mask_cache_key'] = self.image_mask_cache_key

    def _deserialize_image_mask_cache(self, dict):
        """image_mask_cache のストア対応 deserialize。
        旧形式(インライン圧縮の image_mask_cache)はデコードしてストアへ移行し、
        新形式(image_mask_cache_key のみ)はストアから引く(無ければ既存の再計算パスに委ねる)。"""
        self.image_mask_cache = None
        self.image_mask_cache_key = None
        legacy_inline = dict.get('image_mask_cache', None)
        key = dict.get('image_mask_cache_key', None)
        if legacy_inline is not None:
            image = utils.convert_image_from_list(legacy_inline)
            if key is None:
                # キー欠損の旧ファイル: 内容の sha1 で代替キーを生成
                key = ['mask2-ai-cache-legacy', hashlib.sha1(image.tobytes()).hexdigest()]
            self.image_mask_cache = image
            self.image_mask_cache_key = key
            store_put = getattr(self.editor, "put_ai_mask_bitmap", None)
            if callable(store_put):
                store_put(key, image)
        elif key is not None:
            self.image_mask_cache_key = key
            store_get = getattr(self.editor, "get_ai_mask_bitmap", None)
            if callable(store_get):
                self.image_mask_cache = store_get(key)
            logging.warning(
                "[DIAG-REINFER] deserialize %s key=%s bitmap_found=%s read_store_id=%s suspend=%s",
                self.__class__.__name__, key, self.image_mask_cache is not None,
                id(getattr(self.editor, "ai_image_cache", None)),
                getattr(self.editor, "_bitmap_sweep_suspend_depth", "n/a"),
            )

    def invalidate_render_cache(self):
        old_hash = self.image_mask_cache_hash
        derived_hash_attrs = (
            'segment_mask_cache_hash',
            'depth_map_mask_cache_hash',
            'faces_mask_cache_hash',
        )
        has_derived_cache = False
        for attr in derived_hash_attrs:
            if hasattr(self, attr):
                setattr(self, attr, None)
                has_derived_cache = True

        if not has_derived_cache:
            self.image_mask_cache_hash = None
            self.image_mask_cache_key = None
        _mask_geom_debug(
            "invalidate_render_cache mask=%s class=%s old_hash=%s derived=%s",
            _mask_geom_id(self),
            self.__class__.__name__,
            old_hash,
            has_derived_cache,
        )

    def clear(self):
        for cp in self.control_points:
            self.remove_widget(cp)
        self.control_points = []
        self.effects_param = params.delete_not_special_param(self.effects_param)
        effects.reeffect_all(self.effects)

    def start(self):
        pass

    def end(self):
        pass

    def release(self):
        # マスク破棄時のリソース解放フック(既定は何もしない)。
        # Window.bind 等マスクの生存期間全体を跨ぐバインドを持つサブクラスはここで unbind する。
        pass

    def in_creation_stroke(self):
        # 作成確定(_create_end_new_mask)を初回ストローク完了まで遅らせたいマスクは
        # True を返す(Polyline のように複数タッチで初期形状を作るマスク用)。
        return False

    def is_composit(self):
        return isinstance(self, CompositMask)

    def follows_mask_geometry(self):
        return True

    def _call_in_mask_geometry_space(self, func, *args, **kwargs):
        if self.follows_mask_geometry() or self.editor is None:
            return func(*args, **kwargs)
        return self.editor._call_with_image_only_matrix(func, *args, **kwargs)

    def window_to_tcg_for_interaction(self, x, y):
        return self._call_in_mask_geometry_space(self.editor.window_to_tcg, x, y)

    def tcg_to_window_for_overlay(self, x, y):
        return self._call_in_mask_geometry_space(self.editor.tcg_to_window, x, y)

    def refresh_control_points_for_overlay(self):
        for cp in getattr(self, 'control_points', []):
            cp.property('center').dispatch(cp)

    def _draw_brush_size(self):
        value = effects.Mask2Effect.get_param(self.effects_param, 'mask2_freedraw_brush_size')
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 300.0
        return max(2.0, min(2000.0, value))

    def _draw_brush_hardness(self):
        value = effects.Mask2Effect.get_param(self.effects_param, 'mask2_freedraw_brush_hardness')
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 100.0
        return max(0.0, min(100.0, value))

    def _set_mask2_slider_value(self, slider_id, value):
        root = getattr(self.editor, 'root', None)
        ids = getattr(root, 'ids', {}) if root is not None else {}
        try:
            slider = ids.get(slider_id)
        except AttributeError:
            slider = None
        if slider is None:
            return
        try:
            slider.set_slider_value(value)
        except AttributeError:
            slider.value = value

    def _set_draw_brush_param(self, key, value, slider_id):
        self.effects_param[key] = value
        if key == 'mask2_freedraw_brush_size':
            self.brush_size = value
        self._set_mask2_slider_value(slider_id, value)

    def _adjust_draw_brush_from_scroll(self, touch):
        if not getattr(touch, 'is_mouse_scrolling', False):
            return False
        if touch.button == 'scrollup':
            direction = -1
        elif touch.button == 'scrolldown':
            direction = 1
        else:
            return False

        modifiers = KVWindow.modifiers or []
        if 'meta' in modifiers or 'ctrl' in modifiers:
            value = max(0.0, min(100.0, self._draw_brush_hardness() + direction * 5.0))
            self._set_draw_brush_param(
                'mask2_freedraw_brush_hardness',
                value,
                'slider_mask2_freedraw_brush_hardness',
            )
        else:
            value = max(2.0, min(2000.0, self._draw_brush_size() + direction * 10.0))
            self._set_draw_brush_param(
                'mask2_freedraw_brush_size',
                value,
                'slider_mask2_freedraw_brush_size',
            )

        updater = getattr(self, 'update_brush_cursor', None)
        if callable(updater):
            updater(touch.pos[0], touch.pos[1])
        return True

    def _touch_in_initial_placement_area(self, touch):
        return bool(self.editor.collide_point(*touch.pos))

    def _begin_initial_touch_if_in_placement_area(self, touch):
        if not self._touch_in_initial_placement_area(touch):
            self._initial_touch_started = False
            return False
        self._initial_touch_started = True
        return True

    def _initial_touch_can_finish(self):
        if not self._initial_touch_started:
            return False
        self._initial_touch_started = False
        return True

    def on_active_changed(self, instance, value):
        if value:
            self.show_all_control_points()
        else:
            self.show_center_control_point_only()

    def show_all_control_points(self, redraw=True):
        self.opacity = 1
        for cp in self.control_points:
            cp.opacity = 1
            if cp.is_center:
                cp.color = [0, 0, 1]  # アクティブなマスクの中心点
            else:
                if cp.type[0] == 'r' or cp.type[0] == 's':
                    cp.color = [1, 1, 0]
                else:
                    cp.color = [1, 1, 1]  # 他のコントロールポイントは白色
        self.is_draw_mask = True
        self.refresh_control_points_for_overlay()
        if redraw:
            self.update_mask()

    def show_center_control_point_only(self, redraw=True):
        self.opacity = 0.2
        for cp in self.control_points:
            if cp.is_center:
                cp.opacity = 2
                cp.color = [1, 0, 0]  # 非アクティブなマスクの中心点は赤色
            else:
                cp.opacity = 0  # 非表示
        self.is_draw_mask = False
        self.refresh_control_points_for_overlay()
        if redraw:
            self.update_mask()

    def show_hidden(self, keep_overlay=False, redraw=True):
        """マスクの CP 表示を隠す (別 Composit 所属マスク用、または Mesh Edit モード時)。
        効果自体は pipeline 側で反映され続ける。
        keep_overlay=True なら is_draw_mask を残して overlay 描画を継続する
        (Mesh Edit モード中の active Composit 用)。"""
        self.opacity = 0
        for cp in self.control_points:
            cp.opacity = 0
        if keep_overlay:
            self.is_draw_mask = True
            self.refresh_control_points_for_overlay()
            if redraw:
                self.update_mask()
        else:
            self.is_draw_mask = False
            self.refresh_control_points_for_overlay()

    def update_visibility_for_active(self, active_mask, mesh_edit_active):
        """active_mask と Composit 関係に応じて表示モードを切り替える。"""
        editor = self.editor
        if editor is None:
            return
        if mesh_edit_active:
            if active_mask is not None and editor._is_in_same_composit(self, active_mask):
                self.show_hidden(keep_overlay=True, redraw=False)
            else:
                self.show_hidden(keep_overlay=False, redraw=False)
            return
        policy = editor.mask_visibility_policy_for(self, active_mask)
        if policy["control_points"] == "all":
            self.show_all_control_points(redraw=False)
        elif policy["control_points"] == "center":
            self.show_center_control_point_only(redraw=False)
        else:
            self.show_hidden(keep_overlay=policy["overlay"], redraw=False)
        self.is_draw_mask = bool(policy["overlay"])

    def is_center_click(self, touch):
        for cp in self.control_points:
            cx, cy = self.window_to_tcg_for_interaction(*touch.pos)
            if cp.collide_point(cx, cy):
                return cp.is_center
        return False

    def on_touch_down(self, touch):
        for cp in self.control_points:
            cx, cy = self.window_to_tcg_for_interaction(*touch.pos)
            if cp.collide_point(cx, cy): #or (self.editor.collide_point(*touch.pos) and isinstance(self, FreeDrawMask)): # フリーだけコントロールポイント関係ない
                if cp.is_center:
                    self.editor.set_active_mask(self)
                    cp.on_touch_down(touch)
                    # マスク作成中は Create 用の begin が保留中のため、Update の begin で
                    # 上書きしない(begin/end ペア崩壊 → undo ずれの原因になる)
                    if self.editor.created_mask is None:
                        get_history_ctrl().begin_history_layer_ctrl(self.editor, "Update", self.editor.get_mask_list().index(self), None)
                        self._cp_history_started = True
                    self.is_draw_mask = True
                    return True

                elif self.active:
                    cp.on_touch_down(touch)
                    if self.editor.created_mask is None:
                        get_history_ctrl().begin_history_layer_ctrl(self.editor, "Update", self.editor.get_mask_list().index(self), None)
                        self._cp_history_started = True
                    self.is_draw_mask = True
                    return True

        return False

    def on_touch_move(self, touch):
        for cp in self.control_points:
            if cp.touching:
                cp.on_touch_move(touch)
                self.is_draw_mask = True
                self.editor.request_mask_render_update(
                    self,
                    reason="control_point_drag",
                    refresh_visibility=False,
                    redraw_overlay=True,
                    redraw_pipeline=True,
                )
                return True
        return False

    def on_touch_up(self, touch):
        for cp in self.control_points:
            if cp.touching:
                cp.on_touch_up(touch)
                # begin したときだけ end する(ペア厳守)
                if self._cp_history_started:
                    get_history_ctrl().end_history_layer_ctrl(self.editor, "Update", self.editor.get_mask_list().index(self))
                    self._cp_history_started = False
                self.editor.request_mask_render_update(
                    self,
                    reason="control_point_touch_up",
                    redraw_overlay=True,
                    redraw_pipeline=True,
                )
                return True
        return False

    def get_name(self):
        return self.name

    def update(self):
        if len(self.control_points) > 0:
            cp_center = self.control_points[0]
            cp_center.property('ctrl_center').dispatch(cp_center)
            self.is_draw_mask = True
            self.update_mask()

    def update_control_points(self):
        pass

    def on_center_control_point_move(self, instance, value):
        dx = instance.ctrl_center[0] - self.center_x
        dy = instance.ctrl_center[1] - self.center_y
        moved = not (math.isclose(dx, 0.0, abs_tol=1e-9) and math.isclose(dy, 0.0, abs_tol=1e-9))
        if moved:
            self.center = (self.center_x + dx, self.center_y + dy)
            for cp in self.control_points:
                #if cp != instance:
                center = (cp.center_x + dx, cp.center_y + dy)
                if cp.center[0] == center[0] and cp.center[1] == center[1]:
                    cp.property('center').dispatch(cp) # 値が同じだとディスパッチされないから
                else:
                    cp.center = center
        else:
            for cp in self.control_points:
                cp.property('center').dispatch(cp)
        self.update_control_points()
        self.update_mask()
        if moved and not getattr(instance, 'touching', False):
            self.editor.request_mask_render_update(
                self,
                reason="center_control_point_move",
                redraw_overlay=False,
                redraw_pipeline=True,
            )
    
    def draw_mask_to_fbo(self, absolute=False):
        if getattr(self.editor, "_suppress_mask_overlay_draw", False):
            return
        if self.active == True or absolute == True:
            if self.follows_mask_geometry():
                mask_image = self.get_mask_image()
            else:
                mask_image = self.editor._call_with_image_only_matrix(self.get_mask_image)
            # イメージを描画してもらう
            self.editor.draw_mask_image(mask_image)

    def _redraw_mask_content(self, reason="mask_content"):
        """Mask pixels changed without going through ControlPoint properties."""
        if self.editor is None:
            return
        self.editor.request_mask_render_update(
            self,
            reason=reason,
            refresh_visibility=False,
            redraw_overlay=True,
            redraw_pipeline=True,
        )

    def _fit_image_mask_to_texture(self, image):
        """original/full-image 空間の mask を現在 texture に配置する。

        Mask Mesh の source 拡張中は disp_info の x/y が負になることがある。
        numpy slice に負値を渡すと画像末尾からの参照になり、AI系マスクが消えるため、
        範囲外は 0 として扱う。
        """
        texture_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        texture_w, texture_h = texture_size
        if image is None or texture_w <= 0 or texture_h <= 0:
            return np.zeros((max(texture_h, 0), max(texture_w, 0)), dtype=np.float32)

        disp_info = params.get_disp_info(self.editor.tcg_info)
        if disp_info is None:
            return np.zeros((texture_h, texture_w), dtype=image.dtype)

        nw, nh, ox, oy = core.crop_size_and_offset_from_texture(texture_w, texture_h, disp_info)
        if nw <= 0 or nh <= 0:
            return np.zeros((texture_h, texture_w), dtype=image.dtype)

        cx, cy, cw, ch, _scale = disp_info
        cx, cy, cw, ch = int(cx), int(cy), int(cw), int(ch)
        if cw <= 0 or ch <= 0:
            return np.zeros((texture_h, texture_w), dtype=image.dtype)

        src_h, src_w = image.shape[:2]
        orig_w, orig_h = self.editor.tcg_info.get('original_img_size', (src_w, src_h))
        maxsize = max(int(orig_w), int(orig_h))
        if (src_w, src_h) == (int(orig_w), int(orig_h)) and (src_w, src_h) != (maxsize, maxsize):
            pad_x = (maxsize - int(orig_w)) / 2.0
            pad_y = (maxsize - int(orig_h)) / 2.0
            cx = float(cx) - pad_x
            cy = float(cy) - pad_y
            _mask_zoom_sync_debug(
                "fit_image_mask_to_texture source_unpadded mask=%s image=%s orig=%s pad=(%.2f,%.2f) disp=%s src_rect=(%.2f,%.2f,%s,%s)",
                _mask_geom_id(self), getattr(image, "shape", None),
                (orig_w, orig_h), pad_x, pad_y, disp_info, cx, cy, cw, ch,
            )

        in_bounds = 0 <= cx and 0 <= cy and cx + cw <= src_w and cy + ch <= src_h
        integer_rect = abs(float(cx) - round(float(cx))) < 1e-6 and abs(float(cy) - round(float(cy))) < 1e-6
        if in_bounds and integer_rect:
            x0 = int(round(cx))
            y0 = int(round(cy))
            content = cv2.resize(image[y0:y0 + ch, x0:x0 + cw], (nw, nh))
        else:
            sx = float(cw) / float(nw)
            sy = float(ch) / float(nh)
            # cv2.resize の half-pixel 規則に寄せた dst->src 変換。
            matrix = np.array([
                [sx, 0.0, float(cx) + sx * 0.5 - 0.5],
                [0.0, sy, float(cy) + sy * 0.5 - 0.5],
            ], dtype=np.float32)
            content = cv2.warpAffine(
                image,
                matrix,
                (nw, nh),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            _mask_zoom_sync_debug(
                "fit_image_mask_to_texture out_of_bounds mask=%s image=%s disp=%s texture=%s content=%s offset=%s",
                _mask_geom_id(self), getattr(image, "shape", None), disp_info,
                texture_size, getattr(content, "shape", None), (ox, oy),
            )

        out = np.zeros((texture_h, texture_w) + image.shape[2:], dtype=content.dtype)
        dst_x0 = max(0, int(ox))
        dst_y0 = max(0, int(oy))
        dst_x1 = min(texture_w, int(ox) + nw)
        dst_y1 = min(texture_h, int(oy) + nh)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return out

        src_x0 = dst_x0 - int(ox)
        src_y0 = dst_y0 - int(oy)
        src_x1 = src_x0 + (dst_x1 - dst_x0)
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        out[dst_y0:dst_y1, dst_x0:dst_x1] = content[src_y0:src_y1, src_x0:src_x1]
        return out

    def _apply_extened_params(
            self,
            image,
            edge_refine_draw_strokes=None,
            edge_refine_enabled=True):
        simg = self._apply_mask_space(image)
        if edge_refine_enabled:
            simg, edge_support = self._apply_edge_refine(simg, edge_refine_draw_strokes=edge_refine_draw_strokes)
        else:
            edge_support = None
        return extended_params.apply_post_edge_params(
            self.editor,
            self.effects_param,
            simg,
            self.center,
            edge_support=edge_support,
        )

    def _edge_refine_fill_grown_region(self):
        return True

    def _edge_refine_seed_from_guide(self):
        return False

    def _get_edge_refine_seed_mask(self, mask_shape, current_mask=None):
        return None

    def get_hash_items(self):
        # tcg_info['matrix'] のバイト列を末尾に含めることで、CompositMask.get_mask_image が
        # render 中に matrix を mask Geom 込みに swap した瞬間にキャッシュが自動 invalidate される。
        matrix_bytes = b''
        try:
            if self.editor is not None and 'matrix' in self.editor.tcg_info:
                lock = getattr(self.editor, '_matrix_lock', None)
                if lock is None:
                    matrix = self.editor.tcg_info['matrix']
                else:
                    with lock:
                        matrix = np.array(self.editor.tcg_info['matrix'], dtype=np.float64, copy=True)
                matrix_bytes = np.asarray(matrix, dtype=np.float64).tobytes()
        except Exception:
            matrix_bytes = b''
        # Mask2 パラメータ部は headless の get_mask_hash_tuple を再利用(二重定義を排除)。
        # タプル連結なので並び・値は従来と完全に同一 -> hash 値も不変。末尾に GUI 固有の
        # linked mesh hash と matrix バイト列(mask Geom swap 検知用)を付与する。
        return extended_params.get_mask_hash_tuple(self.effects_param) + (
            _linked_primary_mesh_hash_key(self),
            matrix_bytes,
        )

    def _apply_mask_space(self, image):
        return extended_params._apply_mask_space(self.editor, self.effects_param, image)

    def _quick_select_switch_enabled(self):
        return extended_params._quick_select_switch_enabled(self.effects_param)

    def _apply_edge_refine(self, image, edge_refine_draw_strokes=None):
        if not self._edge_refine_enabled_for_mask():
            return image, None
        if not self._quick_select_switch_enabled():
            return image, None
        mode = effects.Mask2Effect.get_param(self.effects_param, 'mask2_edge_refine_mode')
        if not edge_refine.is_enabled(mode):
            return image, None
        guide = self._get_edge_refine_guide_image(image.shape[:2])
        guide_point = self._get_edge_refine_guide_point()
        seed_mask = self._get_edge_refine_seed_mask(image.shape[:2], image)
        refined, support = edge_refine.refine_mask_edge_aware(
            guide,
            image,
            guide_point=guide_point,
            mode=mode,
            radius=self._edge_refine_radius_to_texture(
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_edge_refine_radius')),
            strength=effects.Mask2Effect.get_param(self.effects_param, 'mask2_edge_refine_strength'),
            edge_bias=self._edge_refine_edge_bias_to_texture(
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_edge_refine_bias')),
            fill_grown_region=self._edge_refine_fill_grown_region(),
            seed_from_guide=self._edge_refine_seed_from_guide(),
            seed_mask=seed_mask,
            debug_label=self.__class__.__name__,
            support_softness=self._edge_refine_support_softness(),
            selection_strategy=self._edge_refine_selection_strategy(),
            draw_strokes=edge_refine_draw_strokes,
            return_support=True,
        )
        if self._edge_refine_selection_strategy() == edge_refine.STRATEGY_DRAW:
            refined = self._respect_soft_drawing(refined, image)
        return refined, support

    def _respect_soft_drawing(self, refined, drawn):
        # headless 実装に一本化(_respect_soft_drawing_region と同一ロジック)。
        return extended_params._respect_soft_drawing_region(refined, drawn)

    def _edge_refine_enabled_for_mask(self):
        return True

    def _edge_refine_support_softness(self):
        return 0.0

    def _edge_refine_selection_strategy(self):
        return edge_refine.STRATEGY_REFINE

    def _edge_refine_radius_to_texture(self, radius):
        return extended_params._edge_refine_radius_to_texture(self.editor, radius)

    def _edge_refine_edge_bias_to_texture(self, edge_bias):
        return extended_params._edge_refine_edge_bias_to_texture(self.editor, edge_bias)

    def _get_edge_refine_guide_image(self, mask_shape):
        crop = getattr(self.editor, 'crop_image_rgb', None)
        crop_shape = getattr(crop, 'shape', None)
        path = None
        result = None
        if crop is not None and crop_shape[:2] == tuple(mask_shape):
            path = "crop_image_rgb"
            result = crop
        else:
            original = self.editor.get_original_image_rgb()
            if original is not None:
                path = "fitted_original(NO_ROTATION)"
                guide = self._fit_image_mask_to_texture(original)
                if getattr(guide, 'shape', (None, None))[:2] != tuple(mask_shape):
                    guide = cv2.resize(guide, (int(mask_shape[1]), int(mask_shape[0])), interpolation=cv2.INTER_LINEAR)
                result = guide
            else:
                hls = getattr(self.editor, 'crop_image_hls', None)
                if hls is not None:
                    path = "crop_image_hls"
                    result = hls[..., 1]
        self._debug_log_guide_geometry(path, crop_shape, mask_shape)
        return result

    def _debug_log_guide_geometry(self, path, crop_shape, mask_shape):
        if os.environ.get("PLATYPUS_DEBUG_EDGE_REFINE", "").strip().lower() not in {"1", "true", "yes", "on"} \
                and not os.environ.get("QS_DUMP_INPUT"):
            return
        try:
            tcg = self.editor.tcg_info
            disp = params.get_disp_info(tcg)
            logging.info(
                "[QS_GUIDE_GEOM] path=%s crop_shape=%s mask_shape=%s rotation=%.4f rotation2=%.4f flip=%s "
                "disp_info=%s matrix_is_identity=%s",
                path, crop_shape, tuple(mask_shape),
                float(tcg.get('rotation', 0.0)), float(tcg.get('rotation2', 0.0)),
                tcg.get('flip_mode'), disp,
                bool(np.allclose(np.asarray(tcg.get('matrix', np.eye(3)), dtype=np.float64),
                                 np.eye(3), atol=1e-6)) if tcg.get('matrix') is not None else None,
            )
        except Exception:
            logging.exception("[QS_GUIDE_GEOM] logging failed")

    def _get_edge_refine_guide_point(self):
        return extended_params._get_edge_refine_guide_point(self.editor, getattr(self, 'center', None))

    # depth/blur/HLS の純粋な計算は headless 経路(cores.mask2.extended_params)に一本化し、
    # GUI 側はそこへ委譲する(self.editor が ctx、self.center が center_tcg として互換)。
    # これにより Mask2 範囲パラメータの計算ロジックが 2 重定義されなくなる。
    def _apply_depth_mask(self, image):
        return extended_params._apply_depth_mask(self.effects_param, image)

    def _apply_mask_blur(self, image):
        return extended_params._apply_mask_blur(self.effects_param, image)

    def _draw_hue_mask(self, mask):
        return extended_params._draw_hue_mask(self.editor, self.effects_param, mask, self.center)

    def _draw_lum_mask(self, mask):
        return extended_params._draw_lum_mask(self.editor, self.effects_param, mask, self.center)

    def _draw_sat_mask(self, mask):
        return extended_params._draw_sat_mask(self.editor, self.effects_param, mask, self.center)

# マスクの合成マスク
class CompositMask(BaseMask):

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Composit"
        effects.set_composit_mask_noop_defaults(self.effects_param)
        self.mask_list = list()
        self.initializing = False

    def add_mask(self, mask, maskop='Add', index=0):
        # 子マスクの追加
        if mask is None:
            logging.warning("CompositMask.add_mask: mask is None; ignored")
            return
        # 履歴 replay やシリアライズ経由で None 等の不正な op が入ると
        # get_mask_image の合成で落ちるため、挿入時点で正規化する
        if maskop not in ('Add', 'Subtract'):
            logging.error(f"CompositMask.add_mask: invalid maskop {maskop!r}; falling back to 'Add'")
            maskop = 'Add'
        self.mask_list.insert(index, (mask, maskop))
        if self.editor is not None:
            ready = not getattr(mask, 'initializing', False)
            self.editor.request_mask_render_update(
                mask,
                reason="composit.add_mask",
                structure_changed=True,
                redraw_overlay=ready,
                redraw_pipeline=ready,
            )

    def remove_mask(self, mask):
        # 子マスクの削除
        removed = False
        for item in self.mask_list:
            if item[0] is mask:
                mask.clear()
                self.mask_list.remove(item)
                removed = True
                break
        if removed:
            if self.editor is not None:
                self.editor.request_mask_render_update(
                    self,
                    reason="composit.remove_mask",
                    structure_changed=True,
                    redraw_overlay=True,
                    redraw_pipeline=True,
                )

    def get_mask_list(self):
        # 子マスクのリスト
        return self.mask_list

    def get_mask(self, index):
        # 子マスクの取得
        return self.mask_list[index]

    def find_mask_op(self, mask):
        # 登録されている子マスクのタイプを取得
        for cmask, maskop in self.mask_list:
            if cmask is mask:
                return maskop
        return None

    def clear(self):
        # 子マスクのクリア
        for mask, _ in self.mask_list:
            mask.clear()
        self.mask_list.clear()

    def on_touch_down(self, touch):
        """
        # 先にアクティブなマスクのイベントを処理する
        active_mask = self.editor.get_active_mask()
        if active_mask is not None and active_mask is not self:
            if active_mask.on_touch_down(touch):
                return True
        # 子マスクのイベント処理（逆順で、上に描画されたものを先に）
        for mask, _ in reversed(self.mask_list):
            if mask.active or mask.initializing:
                if mask.on_touch_down(touch):
                    return True
        """
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        """
        active_mask = self.editor.get_active_mask()
        if active_mask is not None and active_mask is not self:
            if active_mask.on_touch_move(touch):
                return True
        for mask, _ in reversed(self.mask_list):
            if mask.active or mask.initializing:
                if mask.on_touch_move(touch):
                    return True
        """
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        """
        active_mask = self.editor.get_active_mask()
        if active_mask is not None and active_mask is not self:
            if active_mask.on_touch_up(touch):
                return True
        for mask, _ in reversed(self.mask_list):
            if mask.active or mask.initializing:
                if mask.on_touch_up(touch):
                    return True
        """
        return super().on_touch_up(touch)

    def serialize(self):
        # パラメータの余計なものを削除
        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)
        
        mdict = {
            'type': MaskType.COMPOSIT,
            'name': self.name,
            'effects_param': param,
            'mask_list': list(),
        }
        # 子マスクのシリアライズ
        for mask, maskop in self.mask_list:
            mdict['mask_list'].append((mask.serialize(), maskop))

        return mdict

    def deserialize(self, dict):
        self.name = dict['name']
        self.effects_param.update(dict['effects_param'])
        effects.set_composit_mask_noop_defaults(self.effects_param)
        # 古いファイル互換: mask_mesh_link_to_image が未設定なら、自前 CP の有無で判定
        # (空 → linked、あり → local)。Headless 側と同じ流儀。
        if 'mask_mesh_link_to_image' not in dict.get('effects_param', {}):
            self.effects_param['mask_mesh_link_to_image'] = \
                not bool(dict.get('effects_param', {}).get('mask_mesh_control_points'))
        # 子マスクのデシリアライズ
        index = self.editor.mask_list.index(self)
        for i, mask_info in enumerate(dict['mask_list']):
            index += 1
            new_mask = self.editor._create_mask(mask_info[0]['type'], index)
            new_mask.deserialize(mask_info[0])
            self.add_mask(new_mask, mask_info[1], i)

    def update_mask(self):
        if self.is_draw_mask == True:
            self.draw_mask_to_fbo()

    def _composit_cache_key(self, output_shape, source_size, source_origin_tex):
        child_state = tuple(
            (
                getattr(child, 'mask_id', None),
                maskop,
                getattr(child, 'image_mask_cache_hash', None),
                _hashable_cache_value(getattr(child, 'image_mask_cache_key', None)),
                getattr(child, 'segment_mask_cache_hash', None),
                getattr(child, 'depth_map_mask_cache_hash', None),
                getattr(child, 'faces_mask_cache_hash', None),
            )
            for child, maskop in getattr(self, 'mask_list', [])
        )
        return hash((
            self.get_hash_items(),
            self.editor.get_hash_items(),
            tuple(output_shape),
            tuple(source_size),
            tuple(round(float(v), 6) for v in source_origin_tex),
            child_state,
        ))

    def get_mask_image(self):
        # mask Geometry: この Composit の mask Geom matrix を tcg_info['matrix'] に
        # 一時的に乗せて、子マスクの座標変換に含めるよう差し替え。finally で必ず復元。
        editor = self.editor
        with editor._matrix_lock:
            saved_matrix = editor.tcg_info['matrix']
            saved_texture_size = tuple(editor.texture_size)
            saved_disp = params.get_disp_info(editor.tcg_info)
            base = editor._image_only_matrix
            enabled = False
            matrix_before_hash = _mask_geom_matrix_hash(saved_matrix)
            base_hash = _mask_geom_matrix_hash(base)
            if base is not None:
                enabled = mask_geometry_mod.is_enabled(self.effects_param)
                if enabled:
                    M_mask = mask_geometry_mod.build_matrix_tcg(
                        self.effects_param, editor.tcg_info['original_img_size'])
                    editor.tcg_info['matrix'] = M_mask @ base
                else:
                    editor.tcg_info['matrix'] = base.copy()
            _mask_geom_debug(
                "Composit.get_mask_image start composit=%s enabled=%s before_matrix=%s base_matrix=%s render_matrix=%s params=%s children=%d",
                _mask_geom_id(self),
                enabled,
                matrix_before_hash,
                base_hash,
                _mask_geom_matrix_hash(editor.tcg_info.get('matrix')),
                _mask_geom_param_summary(self.effects_param),
                len(getattr(self, 'mask_list', [])),
            )

            try:
                effective_mesh_param = _effective_mask_mesh_param(editor, self.effects_param)
                output_shape = (int(editor.texture_size[1]), int(editor.texture_size[0]))
                # Mask Mesh は Mask Geom ではなく画像 Geom にだけ追従する仕様。
                # 子マスク自体はこの Composit の Mask Geom 込みで rasterize するが、
                # mesh の変形場は image-only matrix で計算する。
                source_size, source_origin_tex, pad = editor._call_with_image_only_matrix(
                    _mask_mesh_source_region, editor, effective_mesh_param, output_shape)
                source_w, source_h = int(source_size[0]), int(source_size[1])
                expand_source = (source_w, source_h) != (int(output_shape[1]), int(output_shape[0]))
                if expand_source and saved_disp is None:
                    expand_source = False
                    source_origin_tex = (0.0, 0.0)
                if expand_source and saved_disp is not None:
                    _, _, saved_ox, saved_oy = core.crop_size_and_offset_from_texture(
                        int(output_shape[1]), int(output_shape[0]), saved_disp)
                    if saved_ox != 0 or saved_oy != 0:
                        # disp_info cannot express "expand the texture by a few px while
                        # preserving the existing letterbox offset". Reusing it here turns
                        # portrait/landscape crops back into a square viewport and shifts
                        # AI masks by the padding width when leaving Ge.
                        _mask_zoom_sync_debug(
                            "Composit.get_mask_image skip_expanded_source_letterbox composit=%s output=%s source=%s origin=%s saved_disp=%s offset=(%s,%s)",
                            _mask_geom_id(self), output_shape,
                            (source_w, source_h), source_origin_tex,
                            saved_disp, saved_ox, saved_oy,
                        )
                        expand_source = False
                        source_size = (int(output_shape[1]), int(output_shape[0]))
                        source_w, source_h = int(source_size[0]), int(source_size[1])
                        source_origin_tex = (0.0, 0.0)
                if expand_source and saved_disp is not None:
                    scale = float(saved_disp[4]) if saved_disp[4] else 1.0
                    expanded_disp = (
                        saved_disp[0] + source_origin_tex[0] / scale,
                        saved_disp[1] + source_origin_tex[1] / scale,
                        source_w / scale,
                        source_h / scale,
                        scale,
                    )
                    editor.texture_size = (source_w, source_h)
                    params.set_disp_info(editor.tcg_info, expanded_disp)
                    _mask_zoom_sync_debug(
                        "Composit.get_mask_image expanded_source composit=%s output=%s source=%s origin=%s pad=%d saved_disp=%s expanded_disp=%s",
                        _mask_geom_id(self), output_shape,
                        (source_w, source_h), source_origin_tex, pad, saved_disp,
                        params.get_disp_info(editor.tcg_info),
                    )

                cache_key = self._composit_cache_key(output_shape, (source_w, source_h), source_origin_tex)
                if self.image_mask_cache is not None and self.image_mask_cache_hash == cache_key:
                    _mask_zoom_sync_debug(
                        "Composit.get_mask_image cache_hit composit=%s output=%s source=%s origin=%s hash=%s",
                        _mask_geom_id(self), output_shape, (source_w, source_h), source_origin_tex, cache_key,
                    )
                    return self.image_mask_cache

                # 合成マスクの画像作成
                composit = np.zeros((int(editor.texture_size[1]), int(editor.texture_size[0])), dtype=np.float32)
                allow_over_one = False
                allow_under_zero = False

                for mask, maskop in reversed(self.mask_list):
                    if mask.follows_mask_geometry():
                        mimage = mask.get_mask_image()
                    else:
                        mimage = editor._call_with_image_only_matrix(mask.get_mask_image)
                    if getattr(mimage, "shape", None) != composit.shape:
                        logging.warning(
                            "CompositMask.get_mask_image: child mask size mismatch composit=%s child=%s expected=%s got=%s; retrying",
                            _mask_geom_id(self),
                            _mask_geom_id(mask),
                            composit.shape,
                            getattr(mimage, "shape", None),
                        )
                        try:
                            mask.invalidate_render_cache()
                            if mask.follows_mask_geometry():
                                mimage = mask.get_mask_image()
                            else:
                                mimage = editor._call_with_image_only_matrix(mask.get_mask_image)
                        except Exception:
                            logging.exception("CompositMask.get_mask_image: child mask retry failed")
                    if getattr(mimage, "shape", None) != composit.shape:
                        logging.warning(
                            "CompositMask.get_mask_image: child mask size still mismatched composit=%s child=%s expected=%s got=%s; using empty mask for this frame",
                            _mask_geom_id(self),
                            _mask_geom_id(mask),
                            composit.shape,
                            getattr(mimage, "shape", None),
                        )
                        mimage = np.zeros_like(composit)
                    mask_allow_over_one = False
                    mask_allow_under_zero = False
                    match(maskop):
                        case 'Add':
                            composit = _clip_mask_range(composit + mimage, mask_allow_over_one, mask_allow_under_zero)
                        case 'Subtract':
                            composit = _clip_mask_range(composit - mimage, mask_allow_over_one, mask_allow_under_zero)
                        case _:
                            # add_mask で正規化しているため通常は到達しない。
                            # 古い履歴/保存データに不正 op が残っていても描画は止めない。
                            logging.error(f"Unknown mask operation: {maskop}; treating as 'Add'")
                            composit = _clip_mask_range(composit + mimage, mask_allow_over_one, mask_allow_under_zero)

                # mask Mesh warp: 合成済 composit に非線形 TPS 変形を適用 (空なら no-op)。
                # 変形場は image-only matrix で固定し、Mask Geom の回転/scale/flip には
                # 追従させない。
                if expand_source and saved_disp is not None:
                    editor.texture_size = saved_texture_size
                    params.set_disp_info(editor.tcg_info, saved_disp)
                composit = editor._call_with_image_only_matrix(
                    _apply_mask_mesh_warp,
                    composit, editor, effective_mesh_param,
                    output_shape=output_shape,
                    source_origin_tex=source_origin_tex,
                )
                cache_key = self._composit_cache_key(output_shape, (source_w, source_h), source_origin_tex)
                self.image_mask_cache = composit
                self.image_mask_cache_hash = cache_key
                if _DEBUG_MASK_GEOMETRY:
                    _mask_geom_debug(
                        "Composit.get_mask_image done composit=%s stats=%s",
                        _mask_geom_id(self),
                        _mask_geom_image_stats(composit),
                    )
                return composit
            finally:
                editor.texture_size = saved_texture_size
                if saved_disp is not None:
                    params.set_disp_info(editor.tcg_info, saved_disp)
                editor.tcg_info['matrix'] = saved_matrix


# 円形グラデーションマスクのクラス
class CircularGradientMask(BaseMask):
    inner_radius_x = KVNumericProperty(0)
    inner_radius_y = KVNumericProperty(0)
    outer_radius_x = KVNumericProperty(0)
    outer_radius_y = KVNumericProperty(0)
    rotate_rad = KVNumericProperty(0)

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Circle"
        self.initializing = True  # 初期配置中かどうか

        with self.canvas:
            KVPushMatrix()
            self.scissor = self.editor.push_scissor()
            self.translate = KVTranslate(*self.center)
            self.rotate = KVRotate(angle=0, origin=(0, 0))
            KVColor(*self.color)
            self.outer_line = KVLine(ellipse=(0, 0, 0, 0), width=2) # 外側の円
            self.inner_line = KVLine(ellipse=(0, 0, 0, 0), width=2) # 内側の円
            self.editor.pop_scissor()
            KVPopMatrix()

        #self.update_mask()

    def _edge_refine_fill_grown_region(self):
        return False

    def _edge_refine_seed_from_guide(self):
        return True

    def _edge_refine_enabled_for_mask(self):
        return False

    def on_touch_down(self, touch):
        if self.initializing:
            if not self._begin_initial_touch_if_in_placement_area(touch):
                return False
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            self.inner_radius_x = 0
            self.inner_radius_y = 0
            self.outer_radius_x = 0
            self.outer_radius_y = 0
            return True
        else:            
            return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.initializing:
            if not self._initial_touch_started:
                return False
            # mask Geom の非一様 scale が ON のとき、TCG 距離 (window_to_tcg 後の
            # euclidean) を半径にすると render 時に matrix 由来の各軸 scale が
            # 乗って画面上で楕円になってしまう。配置時は画面上で「正円」になるよう、
            # window 空間で半径を取り、各軸 effective scale で逆スケールして TCG rx/ry
            # に格納する (= render 時に matrix が打ち消して画面上で正円)。
            center_win = self.editor.tcg_to_window(self.center_x, self.center_y)
            dx_win = touch.x - center_win[0]
            dy_win = touch.y - center_win[1]
            radius_win = (dx_win ** 2 + dy_win ** 2) ** 0.5
            # window 半径 → 「matrix scale を含まない」TCG 基準半径
            base_r_tcg = self.editor.window_to_tcg_scale(radius_win, 0)[0]
            # tcg_info['matrix'] の linear 部の列ノルム = 各軸の effective scale
            # (rotation も含まれるが、軸単位の総合 scale としてはこれで十分)
            try:
                with self.editor._matrix_lock:
                    M = np.array(self.editor.tcg_info['matrix'], dtype=np.float64, copy=True)
                # numpy 境界で float() 化 (params.apply_matrix と同じ流儀)。
                # M[i,j] は np.float64 で、そのまま Kivy NumericProperty に流すと
                # "invalid format" で弾かれるため、ここで Python float に確定させる。
                sx_eff = float(((M[0, 0]) ** 2 + (M[1, 0]) ** 2) ** 0.5)
                sy_eff = float(((M[0, 1]) ** 2 + (M[1, 1]) ** 2) ** 0.5)
                if sx_eff < 1e-6:
                    sx_eff = 1.0
                if sy_eff < 1e-6:
                    sy_eff = 1.0
            except Exception:
                # 高頻度経路のためログ抑制
                sx_eff = 1.0
                sy_eff = 1.0
            self.outer_radius_x = base_r_tcg / sx_eff
            self.outer_radius_y = base_r_tcg / sy_eff
            self.inner_radius_x = self.outer_radius_x * 0.5
            self.inner_radius_y = self.outer_radius_y * 0.5
            self.update_mask()
            return True
        else:
            return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.initializing:
            if not self._initial_touch_can_finish():
                return False
            self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            return True
        else:
            return super().on_touch_up(touch)

    def create_control_points(self):
        # 8つのコントロールポイントを作成
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        types  = ['x', 'r', 'y', 'r', 'x', 'r', 'y', 'r']
        self.control_points = []
        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        for i, angle in enumerate(angles):
            # 内側のコントロールポイント
            cp_inner = ControlPoint(self.editor)
            cp_inner.type = [types[i], angle]
            cp_inner.center = self.calculate_point(self.inner_radius_x, self.inner_radius_y, angle)
            cp_inner.ctrl_center = cp_inner.center
            cp_inner.bind(ctrl_center=self.on_inner_control_point_move)
            self.control_points.append(cp_inner)
            self.add_widget(cp_inner)

            # 外側のコントロールポイント
            cp_outer = ControlPoint(self.editor)
            cp_outer.type = [types[i], angle]
            cp_outer.center = self.calculate_point(self.outer_radius_x, self.outer_radius_y, angle)
            cp_outer.ctrl_center = cp_outer.center
            cp_outer.bind(ctrl_center=self.on_outer_control_point_move)
            self.control_points.append(cp_outer)
            self.add_widget(cp_outer)

        if not self.active:
            self.show_center_control_point_only()
        else:
            self.show_all_control_points()  # アクティブなら全ポイントの色・表示を更新

    def serialize(self):
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))
        ix, iy = params.norm_param(self.effects_param, (self.inner_radius_x, self.inner_radius_y))
        ox, oy = params.norm_param(self.effects_param, (self.outer_radius_x, self.outer_radius_y))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)
        
        dict = {
            'type': MaskType.CIRCULAR,
            'name': self.name,
            'center': [cx, cy],
            'inner_radius': [ix, iy],
            'outer_radius': [ox, oy],
            'rotate_rad': self.rotate_rad,
            'effects_param': param
        }
        return dict

    def deserialize(self, dict):
        self.initializing = False
        self.name = dict['name']
        cx, cy = dict['center']
        ix, iy = dict['inner_radius']
        ox, oy = dict['outer_radius']
        self.rotate_rad = dict['rotate_rad']
        self.effects_param.update(dict['effects_param'])

        self.center = params.denorm_param(self.effects_param, (cx, cy))
        self.inner_radius_x, self.inner_radius_y = params.denorm_param(self.effects_param, (ix, iy))
        self.outer_radius_x, self.outer_radius_y = params.denorm_param(self.effects_param, (ox, oy))

        self.create_control_points()
        #self.update_mask()
 
    def calculate_point(self, radius_x, radius_y, angle_deg):
        angle_rad = math.radians(angle_deg)
        radius_x = radius_x
        radius_y = radius_y
        dx = radius_x * math.cos(angle_rad)
        dy = radius_y * math.sin(angle_rad)
        new_r_x = dx * math.cos(-self.rotate_rad) - dy * math.sin(-self.rotate_rad)
        new_r_y = dx * math.sin(-self.rotate_rad) + dy * math.cos(-self.rotate_rad)
        return (self.center_x + new_r_x, self.center_y + new_r_y)

    def calculate_rotate(self, radius_x, radius_y, angle_deg, dx, dy):
        angle_rad = math.radians(angle_deg)
        px = radius_x * math.cos(angle_rad)
        py = radius_y * math.sin(angle_rad)
        rotate_rad = -math.atan2(dy, dx)
        new_rad = rotate_rad+math.atan2(py, px)
        return new_rad

    def update_ellipse(self, dx, dy):
        # 回転角の変化に応じて、半径を更新
        new_r_x = dx * math.cos(self.rotate_rad) - dy * math.sin(self.rotate_rad)
        new_r_y = dx * math.sin(self.rotate_rad) + dy * math.cos(self.rotate_rad)
        
        return (abs(new_r_x), abs(new_r_y))


    def on_outer_control_point_move(self, instance, value):
        if self.active:
            dx = instance.ctrl_center[0] - self.center_x
            dy = instance.ctrl_center[1] - self.center_y
            sx = self.inner_radius_x / self.outer_radius_x
            sy = self.inner_radius_y / self.outer_radius_y
            if instance.type[0] == 'x':
                self.outer_radius_x, _ = self.update_ellipse(dx, dy)
                self.inner_radius_x = self.outer_radius_x * sx
                self.outer_radius_x = max(10, max(self.outer_radius_x, self.inner_radius_x))
            elif instance.type[0] == 'y':
                _, self.outer_radius_y = self.update_ellipse(dx, dy)
                self.inner_radius_y = self.outer_radius_y * sy
                self.outer_radius_y = max(10, max(self.outer_radius_y, self.inner_radius_y))
            elif instance.type[0] == 'r':
                self.rotate_rad = self.calculate_rotate(self.outer_radius_x, self.outer_radius_y, instance.type[1], dx, dy)
            self.update_control_points()
            self.update_mask()
            self.editor.start_draw_image()

    def on_inner_control_point_move(self, instance, value):
        if self.active:
            dx = instance.ctrl_center[0] - self.center_x
            dy = instance.ctrl_center[1] - self.center_y
            sx = self.inner_radius_x / self.outer_radius_x
            sy = self.inner_radius_y / self.outer_radius_y
            if instance.type[0] == 'x':
                self.inner_radius_x, _ = self.update_ellipse(dx, dy)
                self.inner_radius_x = max(5, min(self.inner_radius_x, self.outer_radius_x-10))
                #self.inner_radius_y = self.outer_radius_y * sx
            elif instance.type[0] == 'y':
                _, self.inner_radius_y = self.update_ellipse(dx, dy)
                self.inner_radius_y = max(5, min(self.inner_radius_y, self.outer_radius_y-10))
                #self.inner_radius_x = self.outer_radius_x * sy
            elif instance.type[0] == 'r':
                self.rotate_rad = self.calculate_rotate(self.inner_radius_x, self.inner_radius_y, instance.type[1], dx, dy)
            self.update_control_points()
            self.update_mask()
            self.editor.start_draw_image()

    def update_control_points(self):
        # コントロールポイントの位置を更新
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
        cp_center = self.control_points[0]
        cp_center.center = self.center
        index = 1  # 0は中心点
        for angle in angles:
            cp_inner = self.control_points[index]
            cp_inner.center_x, cp_inner.center_y = self.calculate_point(self.inner_radius_x, self.inner_radius_y, angle)
            index += 1
            cp_outer = self.control_points[index]
            cp_outer.center_x, cp_outer.center_y = self.calculate_point(self.outer_radius_x, self.outer_radius_y, angle)
            index += 1

    def _matrix_transformed_ellipse(self, rx_tcg, ry_tcg):
        """Jacobian-at-ellipse-center: 楕円中心位置での center_rotate Jacobian を
        軸ベクトルに線形適用し、SVD で (rx, ry, rotate_rad) を再パラメータ化。

        center_rotate = apply_orientation + R(-(rotation+rotation2)) + apply_matrix。
        apply_matrix の Jacobian を中心 (cx_pre, cy_pre) で評価し、軸方向には同じ線形
        変形を掛ける (位置依存性は中心のみに反映、軸方向では一定)。
        これにより、強 projective (Four Points 等) でも ellipse 形状が弧上で不均一に
        歪まず、中心位置で評価された perspective が ellipse 全体に均一に効く。

        Returns: (new_rx_tcg, new_ry_tcg, new_rotate_rad_for_rasterizer)
        matrix = identity 時、affine 時は既存 / sample-and-fit と同一の結果。
        """
        with self.editor._matrix_lock:
            tcg_info = dict(self.editor.tcg_info)
            tcg_info['matrix'] = np.array(self.editor.tcg_info['matrix'], dtype=np.float64, copy=True)
        theta = self.rotate_rad
        cx, cy = self.center
        c, s = math.cos(theta), math.sin(theta)

        try:
            # apply_orientation + R(-(rotation+rotation2)) で中心を pre-matrix coord に運ぶ
            cx_o, cy_o, rot2 = params.apply_orientation(cx, cy, tcg_info)
            rad = -(tcg_info['rotation'] + rot2)
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            cx_pre = cx_o * cos_r - cy_o * sin_r
            cy_pre = cx_o * sin_r + cy_o * cos_r

            # apply_matrix の解析的 Jacobian at (cx_pre, cy_pre)
            # apply_matrix(x, y) = ((ax+by+e)/w, (cx+dy+f)/w), w = gx+hy+i
            M = np.asarray(tcg_info['matrix'], dtype=np.float64)
            a, b, e = M[0]
            c_m, d, f = M[1]
            g, h, i = M[2]
            denom = g * cx_pre + h * cy_pre + i
            if abs(denom) < 1e-12:
                return rx_tcg, ry_tcg, self.editor.get_rotate_rad(self.rotate_rad)
            num_x = a * cx_pre + b * cy_pre + e
            num_y = c_m * cx_pre + d * cy_pre + f
            d2 = denom * denom
            J_mat = np.array([
                [(a * denom - num_x * g) / d2, (b * denom - num_x * h) / d2],
                [(c_m * denom - num_y * g) / d2, (d * denom - num_y * h) / d2],
            ])

            # 全 Jacobian = J_mat @ R(rad) @ F (= linearized center_rotate at TCG center)
            R_rad = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            flip = tcg_info['flip_mode']
            F = np.array([
                [-1.0 if (flip & 1) else 1.0, 0.0],
                [0.0, -1.0 if (flip & 2) else 1.0],
            ])
            full_jacobian = J_mat @ R_rad @ F
        except Exception:
            # 高頻度経路のためログ抑制
            return rx_tcg, ry_tcg, self.editor.get_rotate_rad(self.rotate_rad)

        # 楕円軸ベクトル (TCG image-coord, Y-down)
        # x 軸方向 (rx 倍): (cos θ, -sin θ) · rx
        # y 軸方向 (ry 倍): (sin θ, cos θ) · ry
        ax_image = np.array([rx_tcg * c, -rx_tcg * s])
        ay_image = np.array([ry_tcg * s, ry_tcg * c])

        ax_post = full_jacobian @ ax_image
        ay_post = full_jacobian @ ay_image

        Mat = np.column_stack([ax_post, ay_post])
        try:
            U, S, _ = np.linalg.svd(Mat)
        except np.linalg.LinAlgError:
            # 高頻度経路のためログ抑制
            return rx_tcg, ry_tcg, self.editor.get_rotate_rad(self.rotate_rad)

        new_rx = float(S[0]) if len(S) > 0 else rx_tcg
        new_ry = float(S[1]) if len(S) > 1 else ry_tcg
        # U は image-coord (Y-down) standard rotation。Kivy/画面 CCW positive 規約に negate
        new_angle_image = math.atan2(float(U[1, 0]), float(U[0, 0]))
        new_rot = -new_angle_image
        return new_rx, new_ry, new_rot

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            logging.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        # matrix 追従の SVD 再パラメータ化 (inner と outer は同じ rotation を共有させる)
        new_inner_rx, new_inner_ry, new_rotate_rad = self._matrix_transformed_ellipse(
            self.inner_radius_x, self.inner_radius_y)
        new_outer_rx, new_outer_ry, _ = self._matrix_transformed_ellipse(
            self.outer_radius_x, self.outer_radius_y)

        with self.canvas:
            self.editor.set_scissor(self.scissor)
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
            self.rotate.angle = math.degrees(new_rotate_rad)
            ix, iy = self.editor.tcg_to_window_scale(new_inner_rx, new_inner_ry)
            self.inner_line.ellipse = (-ix, -iy, ix*2, iy*2)
            ox, oy = self.editor.tcg_to_window_scale(new_outer_rx, new_outer_ry)
            self.outer_line.ellipse = (-ox, -oy, ox*2, oy*2)

        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def get_mask_image(self):
        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_texture(*self.center)
        # matrix 追従の SVD 再パラメータ化 (inner と outer の rotation を共有)
        new_inner_rx, new_inner_ry, new_rotate_rad = self._matrix_transformed_ellipse(
            self.inner_radius_x, self.inner_radius_y)
        new_outer_rx, new_outer_ry, _ = self._matrix_transformed_ellipse(
            self.outer_radius_x, self.outer_radius_y)
        inner_axes = self.editor.tcg_to_image_scale(new_inner_rx, new_inner_ry)
        outer_axes = self.editor.tcg_to_image_scale(new_outer_rx, new_outer_ry)
        rotate_rad = new_rotate_rad
        if effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_settings') == True:
            invert = not effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert')
        else:
            invert = False

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, center, inner_axes, outer_axes, rotate_rad, invert))
        cache_miss = self.image_mask_cache is None or self.image_mask_cache_hash != newhash
        _mask_geom_debug(
            "Circular.get_mask_image mask=%s initializing=%s cache=%s old_hash=%s new_hash=%s matrix=%s center=%s inner=%s outer=%s rotate=%s invert=%s",
            _mask_geom_id(self),
            self.initializing,
            "miss" if cache_miss else "hit",
            self.image_mask_cache_hash,
            newhash,
            _mask_geom_matrix_hash(self.editor.tcg_info.get('matrix')),
            center,
            inner_axes,
            outer_axes,
            rotate_rad,
            invert,
        )
        if cache_miss and self.initializing == False:

            # グラデーションを描画
            gradient_image = self.draw_elliptical_gradient(image_size, center, inner_axes, outer_axes, rotate_rad, invert, 1.5)

            # ルミノシティマスクを作成
            gradient_image = self._apply_extened_params(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash

        result = self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        if _DEBUG_MASK_GEOMETRY:
            _mask_geom_debug(
                "Circular.get_mask_image result mask=%s stats=%s",
                _mask_geom_id(self),
                _mask_geom_image_stats(result),
            )
        return result

    def draw_elliptical_gradient(self, image_size, center, inner_axes, outer_axes, angle_rad, invert=False, smoothness=1):
        return mask_rasters.draw_elliptical_gradient(
            image_size, center, inner_axes, outer_axes, angle_rad, invert, smoothness
        )

# GradientMask クラス
class GradientMask(BaseMask):
    start_point = KVListProperty([0, 0])    # グラデーションの開始点
    end_point = KVListProperty([0, 0])      # グラデーションの終点
    
    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Line"
        self.initializing = True  # 初期配置中かどうか
        self._initial_anchor_set = False

        with self.canvas:
            KVPushMatrix()
            self.scissor = self.editor.push_scissor()
            self.translate = KVTranslate(*self.center)
            self.rotate = KVRotate(angle=0, origin=(0, 0))
            KVColor(*self.color)
            self.start_line = KVLine(points=[], width=2)
            self.center_line = KVLine(points=[], width=2)
            self.end_line = KVLine(points=[], width=2)
            self.editor.pop_scissor()
            KVPopMatrix()

        self.rotate_rad = 0
        #self.update_mask()

    def _edge_refine_fill_grown_region(self):
        return False

    def _edge_refine_seed_from_guide(self):
        return True

    def _edge_refine_enabled_for_mask(self):
        return False
    
    def on_touch_down(self, touch):
        if self.initializing:
            if not self._begin_initial_touch_if_in_placement_area(touch):
                return False
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self._initial_anchor_set = True
            self.center = (cx, cy)
            self.start_point = [cx, cy]
            return True
        else:
            return super().on_touch_down(touch)
    
    def on_touch_move(self, touch):
        if self.initializing:
            if not self._initial_touch_started:
                return False
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.end_point = [cx, cy]
            self.center = [(self.start_point[0] + self.end_point[0]) / 2,
                           (self.start_point[1] + self.end_point[1]) / 2]
            dx = self.end_point[0] - self.start_point[0]
            dy = self.end_point[1] - self.start_point[1]
            self.rotate_rad = math.atan2(dy, dx)
            self.update_mask()
            return True
        else:
            return super().on_touch_move(touch)
    
    def on_touch_up(self, touch):
        if self.initializing:
            if not self._initial_touch_can_finish():
                return False
            self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            return True
        else:
            return super().on_touch_up(touch)
    
    def serialize(self):
        sx, sy = params.norm_param(self.effects_param, (self.start_point[0], self.start_point[1]))
        ex, ey = params.norm_param(self.effects_param, (self.end_point[0], self.end_point[1]))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)
         
        dict = {
            'type': MaskType.GRADIENT,
            'name': self.name,
            'start_point': [sx, sy],
            'end_point': [ex, ey],
            'effects_param': param
        }
        return dict

    def deserialize(self, dict):
        self.initializing = False
        self._initial_anchor_set = True
        self.name = dict['name']
        sx, sy = dict['start_point']
        ex, ey = dict['end_point']
        self.effects_param.update(dict['effects_param'])

        self.start_point = params.denorm_param(self.effects_param, (sx, sy))
        self.end_point = params.denorm_param(self.effects_param, (ex, ey))

        self.center = [(self.start_point[0] + self.end_point[0]) / 2,
                       (self.start_point[1] + self.end_point[1]) / 2]
        
        self.create_control_points()
        #self.update_mask()

    def create_control_points(self):
        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = self.center
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)
    
        # グラデーションの開始点と終点のコントロールポイント
        cp_start = ControlPoint(self.editor)
        cp_start.center = self.start_point
        cp_start.ctrl_center = cp_start.center
        cp_start.type = ['s', 0]
        cp_start.bind(ctrl_center=self.on_control_point_move)
        self.control_points.append(cp_start)
        self.add_widget(cp_start)
    
        cp_end = ControlPoint(self.editor)
        cp_end.center = self.end_point
        cp_end.ctrl_center = cp_end.center
        cp_end.type = ['e', 0]
        cp_end.bind(ctrl_center=self.on_control_point_move)
        self.control_points.append(cp_end)
        self.add_widget(cp_end)
    
        if not self.active:
            self.show_center_control_point_only()
        else:
            self.show_all_control_points()  # アクティブなら全ポイントの色・表示を更新
    
    def calculate_point(self, point, dir):
        r = np.sqrt((point[0]-self.center_x)**2+(point[1]-self.center_y)**2)
        dx = dir * r
        dy = 0.
        new_r_x = dx * np.cos(-self.rotate_rad) + dy * np.sin(-self.rotate_rad)
        new_r_y = dy * np.cos(-self.rotate_rad) - dx * np.sin(-self.rotate_rad)
        return (float(self.center_x + new_r_x), float(self.center_y + new_r_y))

    def on_center_control_point_move(self, instance, value):
        dx = instance.ctrl_center[0] - self.center[0]
        dy = instance.ctrl_center[1] - self.center[1]
        moved = not (math.isclose(dx, 0.0, abs_tol=1e-9) and math.isclose(dy, 0.0, abs_tol=1e-9))
        if moved:
            self.start_point = [self.start_point[0] + dx, self.start_point[1] + dy]
            self.end_point = [self.end_point[0] + dx, self.end_point[1] + dy]
            self.center = [self.center[0] + dx, self.center[1] + dy]
            for cp in self.control_points:
                #if cp != instance:
                center = (cp.center_x + dx, cp.center_y + dy)
                if cp.center[0] == center[0] and cp.center[1] == center[1]:
                    cp.property('center').dispatch(cp) # 値が同じだとディスパッチされないから
                else:
                    cp.center = center
        else:
            for cp in self.control_points:
                cp.property('center').dispatch(cp)
        self.update_control_points()
        self.update_mask()
        if moved:
            self.editor.start_draw_image()
    
    def on_control_point_move(self, instance, value):
        if self.active:
            if instance == self.control_points[1]:
                self.start_point = [instance.ctrl_center[0], instance.ctrl_center[1]]
                dx = self.center_x - self.start_point[0]
                dy = self.center_y - self.start_point[1]
                self.end_point[0] = self.center_x + dx
                self.end_point[1] = self.center_y + dy
            elif instance == self.control_points[2]:
                self.end_point = [instance.ctrl_center[0], instance.ctrl_center[1]]
                dx = self.center_x - self.end_point[0]
                dy = self.center_y - self.end_point[1]
                self.start_point[0] = self.center_x + dx
                self.start_point[1] = self.center_y + dy
            # 再計算
            dx = self.end_point[0] - self.start_point[0]
            dy = self.end_point[1] - self.start_point[1]
            self.rotate_rad = math.atan2(dy, dx)
            self.update_control_points()
            self.update_mask()
            self.editor.start_draw_image()        

    def update_control_points(self):
        # コントロールポイントの位置を更新
        cp_center = self.control_points[0]
        cp_center.center = self.center
        cp_start = self.control_points[1]
        cp_start.center = self.start_point
        cp_end = self.control_points[2]
        cp_end.center = self.end_point
    
    def calculate_line(self, point1, point2, dir):
        p1x, p1y = self.editor.tcg_to_window(*point1)
        p2x, p2y = self.editor.tcg_to_window(*point2)
        r = math.sqrt((p1x-p2x)**2+(p1y-p2y)**2)
        dx = dir * r
        dy = -self.editor.width
        new_dx1 = dx
        new_dy1 = dy
        dx = dir * r
        dy = self.editor.width
        new_dx2 = dx
        new_dy2 = dy
        dx = p1x-p2x
        dy = p1y-p2y
        rad = 0 if dx == 0 else math.atan2(dy, dx)
        return (new_dx1, new_dy1, new_dx2, new_dy2), rad

    def _line_segments_from_window(self, p1_win, p2_win, dir):
        """calculate_line と同じ計算だが window 座標を直接受け取る (= 内部で
        tcg_to_window を再呼び出ししない)。direction-preserving の座標と一緒に使う。"""
        p1x, p1y = p1_win
        p2x, p2y = p2_win
        r = math.sqrt((p1x-p2x)**2+(p1y-p2y)**2)
        dx_a = dir * r
        dy_a = -self.editor.width
        new_dx1 = dx_a
        new_dy1 = dy_a
        dx_b = dir * r
        dy_b = self.editor.width
        new_dx2 = dx_b
        new_dy2 = dy_b
        ddx = p1x - p2x
        ddy = p1y - p2y
        rad = 0 if ddx == 0 else math.atan2(ddy, ddx)
        return (new_dx1, new_dy1, new_dx2, new_dy2), rad

    def _dir_preserving_post(self):
        """Mask Geom 非一様 scale でラインが回転する問題を吸収するため、
        matrix の pure rotation 成分 (R = U @ V^T from SVD of Jacobian-at-center)
        だけを direction に適用、長さは |J @ dir_unit| でスケール。

        Returns (dir_x_post, dir_y_post, length_scale, length_tcg) または None (degenerate)。
        - dir_post = pure rotation を適用した unit direction (TCG image-coord, Y-down)
        - length_scale = TCG-image-pixel 空間での方向沿いスケール
        - length_tcg = 元の TCG 距離 (端点間)
        """
        dx_tcg = self.end_point[0] - self.start_point[0]
        dy_tcg = self.end_point[1] - self.start_point[1]
        length_tcg = math.hypot(dx_tcg, dy_tcg)
        if length_tcg < 1e-9:
            return None

        with self.editor._matrix_lock:
            tcg_info = dict(self.editor.tcg_info)
            tcg_info['matrix'] = np.array(self.editor.tcg_info['matrix'], dtype=np.float64, copy=True)
        cx, cy = self.center
        try:
            cx_o, cy_o, rot2 = params.apply_orientation(cx, cy, tcg_info)
            rad = -(tcg_info['rotation'] + rot2)
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            cx_pre = cx_o * cos_r - cy_o * sin_r
            cy_pre = cx_o * sin_r + cy_o * cos_r

            M = np.asarray(tcg_info['matrix'], dtype=np.float64)
            a, b, e_ = M[0]
            c_m, d, f_ = M[1]
            g, h, i_ = M[2]
            denom = g * cx_pre + h * cy_pre + i_
            if abs(denom) < 1e-12:
                return None
            num_x = a * cx_pre + b * cy_pre + e_
            num_y = c_m * cx_pre + d * cy_pre + f_
            d2 = denom * denom
            J_mat = np.array([
                [(a * denom - num_x * g) / d2, (b * denom - num_x * h) / d2],
                [(c_m * denom - num_y * g) / d2, (d * denom - num_y * h) / d2],
            ])
            R_rad = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            flip = tcg_info['flip_mode']
            F = np.array([
                [-1.0 if (flip & 1) else 1.0, 0.0],
                [0.0, -1.0 if (flip & 2) else 1.0],
            ])
            full_J = J_mat @ R_rad @ F

            U, S, Vt = np.linalg.svd(full_J)
            R_pure = U @ Vt  # 2x2 pure rotation in TCG-image-pixel space (Y-down)

            dir_unit = np.array([dx_tcg / length_tcg, dy_tcg / length_tcg])
            dir_post = R_pure @ dir_unit
            j_dir = full_J @ dir_unit
            length_scale = float(np.linalg.norm(j_dir))
        except Exception:
            return None

        return (float(dir_post[0]), float(dir_post[1]), length_scale, length_tcg)

    def _dir_preserving_endpoints_tex(self):
        """direction-preserving な (center_tex, start_tex, end_tex) を返す。"""
        editor = self.editor
        center_tex = editor.tcg_to_texture(*self.center)
        result = self._dir_preserving_post()
        if result is None:
            return (center_tex,
                    editor.tcg_to_texture(*self.start_point),
                    editor.tcg_to_texture(*self.end_point))
        dir_x, dir_y, length_scale, length_tcg = result
        disp_scale = params.get_disp_info(editor.tcg_info)[4]
        half_len_tex = length_tcg * length_scale * disp_scale / 2.0
        cx_tex, cy_tex = center_tex
        # texture 座標系は TCG-image-pixel と同じ Y-down なので direction はそのまま
        start_tex = (cx_tex - half_len_tex * dir_x, cy_tex - half_len_tex * dir_y)
        end_tex = (cx_tex + half_len_tex * dir_x, cy_tex + half_len_tex * dir_y)
        return (center_tex, start_tex, end_tex)

    def _dir_preserving_endpoints_win(self):
        """direction-preserving な (center_win, start_win, end_win) を返す (overlay 用)。"""
        editor = self.editor
        center_win = editor.tcg_to_window(*self.center)
        result = self._dir_preserving_post()
        if result is None:
            return (center_win,
                    editor.tcg_to_window(*self.start_point),
                    editor.tcg_to_window(*self.end_point))
        dir_x, dir_y, length_scale, length_tcg = result
        disp_scale = params.get_disp_info(editor.tcg_info)[4]
        dpi = device.dpi_scale()
        half_len_win = length_tcg * length_scale * disp_scale * dpi / 2.0
        cx_win, cy_win = center_win
        # window 座標は Y-up なので direction の Y 成分を反転
        dx_win = half_len_win * dir_x
        dy_win = -half_len_win * dir_y
        start_win = (cx_win - dx_win, cy_win - dy_win)
        end_win = (cx_win + dx_win, cy_win + dy_win)
        return (center_win, start_win, end_win)

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            logging.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        if self.initializing and not self._initial_anchor_set:
            self.start_line.points = []
            self.center_line.points = []
            self.end_line.points = []
            return

        with self.canvas:
            self.editor.set_scissor(self.scissor)
            center_win, start_win, end_win = self._dir_preserving_endpoints_win()
            if self.initializing:
                self.translate.x, self.translate.y = start_win
                self.start_line.points, _ = self._line_segments_from_window(start_win, start_win, 0)
                self.center_line.points, _ = self._line_segments_from_window(center_win, start_win, +1)
                self.end_line.points, rad = self._line_segments_from_window(end_win, start_win, +1)
            else:
                self.translate.x, self.translate.y = center_win
                self.start_line.points, rad = self._line_segments_from_window(start_win, center_win, -1)
                self.center_line.points, _ = self._line_segments_from_window(center_win, center_win, 0)
                self.end_line.points, _ = self._line_segments_from_window(end_win, center_win, +1)

            self.rotate.angle = math.degrees(rad)

        # ControlPoint の表示位置を direction-preserving に同期。
        # 既存の bind(center=update_graphics) は raw tcg_to_window を使うため、
        # mask Geom 非一様 scale 等のとき line raster と CP の位置がズレる。
        # ここで translate を直接上書きして line と一致させる。
        # touching=True (= ユーザが drag 中の CP) は finger position を維持するため上書きしない。
        if not self.initializing:
            self._sync_cps_to_dir_preserving(center_win, start_win, end_win)

        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def _sync_cps_to_dir_preserving(self, center_win, start_win, end_win):
        """CP の visual translate を direction-preserving 計算結果で上書き。
        touching 中の CP は user finger を追従させるためスキップ。"""
        cps = self.control_points
        if len(cps) < 3:
            return
        positions = (center_win, start_win, end_win)
        for cp, pos in zip(cps[:3], positions):
            if getattr(cp, 'touching', False):
                continue
            try:
                cp.translate.x = pos[0]
                cp.translate.y = pos[1]
                self.editor.set_scissor(cp.scissor)
            except Exception:
                pass

    def get_mask_image(self):
        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center, start_point, end_point = self._dir_preserving_endpoints_tex()
        if effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_settings') == True:
            if effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert') == True:
                start_point, end_point = end_point, start_point

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, center, start_point, end_point))
        cache_miss = self.image_mask_cache is None or self.image_mask_cache_hash != newhash
        _mask_geom_debug(
            "Gradient.get_mask_image mask=%s initializing=%s cache=%s old_hash=%s new_hash=%s matrix=%s center=%s start=%s end=%s",
            _mask_geom_id(self),
            self.initializing,
            "miss" if cache_miss else "hit",
            self.image_mask_cache_hash,
            newhash,
            _mask_geom_matrix_hash(self.editor.tcg_info.get('matrix')),
            center,
            start_point,
            end_point,
        )
        if cache_miss and self.initializing == False:
            # グラデーションを描画
            gradient_image = self.draw_gradient(image_size, center, start_point, end_point, 1)
            
            # ルミノシティマスクを作成
            gradient_image = self._apply_extened_params(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash

        result = self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)
        if _DEBUG_MASK_GEOMETRY:
            _mask_geom_debug(
                "Gradient.get_mask_image result mask=%s stats=%s",
                _mask_geom_id(self),
                _mask_geom_image_stats(result),
            )
        return result
    
    def draw_gradient(self, image_size, center, start_point, end_point, smoothness=1):
        return mask_rasters.draw_linear_gradient(
            image_size, center, start_point, end_point, smoothness
        )

# 全体マスクのクラス
class FullMask(BaseMask):

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Full"
        self.initializing = True  # 初期配置中かどうか

        self.center = (0, 0)

        with self.canvas:
            KVPushMatrix()
            self.translate = KVTranslate(*self.center)
            KVPopMatrix()

        #self.update_mask()

    def _edge_refine_fill_grown_region(self):
        return True

    def _edge_refine_seed_from_guide(self):
        return True

    def _edge_refine_enabled_for_mask(self):
        return False

    def on_touch_down(self, touch):
        if self.initializing:
            if not self._begin_initial_touch_if_in_placement_area(touch):
                return False
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            return True
        else: 
            return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.initializing:
            if not self._initial_touch_can_finish():
                return False
            self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            return True
        else:
            return super().on_touch_up(touch)

    def create_control_points(self):
        self.control_points = []

        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        if not self.active:
            self.show_center_control_point_only()

    def serialize(self):
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)
        
        dict = {
            'type': MaskType.FULL,
            'name': self.name,
            'center': [cx, cy],
            'effects_param': param
        }
        return dict

    def deserialize(self, dict):
        self.initializing = False
        cx, cy = dict['center']
        self.name = dict['name']
        self.effects_param.update(dict['effects_param'])

        self.center = params.denorm_param(self.effects_param, (cx, cy))

        # 描き直し
        self.create_control_points()
        #self.update_mask()    

    def update_control_points(self):
        cp_center = self.control_points[0]
        cp_center.center = self.center

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            logging.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            cx, cy = self.editor.tcg_to_window(*self.center)
            self.translate.x, self.translate.y = cx, cy
        
        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        center = self.editor.tcg_to_texture(*self.center)

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, center))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
            # 描画
            gradient_image = self.draw_full(image_size, center)

            # ルミノシティマスクを作成
            gradient_image = self._apply_extened_params(gradient_image)

            self.image_mask_cache = gradient_image
            self.image_mask_cache_hash = newhash
        
        return self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    def draw_full(self, image_size, center):
        # 画像の初期化
        image = np.ones((image_size[1], image_size[0]), dtype=np.float32)

        return image

# 自由描画マスクのクラス
class FreeDrawMask(BaseMask):

    class Line:
        def __init__(self, is_erasing=False, size=10, soft=100):
            self.is_erasing = is_erasing
            self.size = size
            self.soft = soft
            self.points = []

        def add_point(self, x, y):
            self.points.append((x, y))

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Draw"
        self.initializing = True

        self.lines = []  # 複数の線を保持
        self.current_line = None
        self.brush_size = self._draw_brush_size()
        self._stroke_history_started = False
        self._drag_base_mask = None

        with self.canvas:
            KVPushMatrix()
            self.scissor = self.editor.push_scissor()
            self.translate = KVTranslate(0, 0)
            self.rotate = KVRotate(angle=0, origin=(0, 0))
            self.brush_color = KVColor((0, 1, 1, 1))
            self.brush_cursor = KVLine(ellipse=(0, 0, self.brush_size, self.brush_size), width=2)
            # 内側ハードゾーン (hardness の視覚化用 2 重目の円)
            self.brush_cursor_inner = KVLine(ellipse=(0, 0, 0, 0), width=2)
            self.editor.pop_scissor()
            KVPopMatrix()

        # マスクの生存期間全体で bind したままにする(start/end で張り直さない)。
        # 選択されていない間も on_mouse_pos は届くが、ハンドラ内で active/created を見て
        # 表示を抑制している。release() で unbind する。
        KVWindow.bind(mouse_pos=self.on_mouse_pos)
        self._mouse_pos_bound = True

    def _edge_refine_fill_grown_region(self):
        return True

    def _get_edge_refine_seed_mask(self, mask_shape, current_mask=None):
        if current_mask is None:
            return None
        seed = edge_refine.make_confident_seed(current_mask)
        return seed if np.any(seed) else None

    def _edge_refine_selection_strategy(self):
        return edge_refine.STRATEGY_DRAW

    def start(self):
        self.brush_color.rgba = (1, 1, 1, 1)

    def end(self):
        self.brush_color.rgba = (0, 0, 0, 0)

    def release(self):
        # マスク削除時に Window の bind を解除する(unbind は二重に呼んでも安全)。
        if getattr(self, '_mouse_pos_bound', False):
            KVWindow.unbind(mouse_pos=self.on_mouse_pos)
            self._mouse_pos_bound = False

    def clear(self):
        self.lines = []
        self.current_line = None
        self._drag_base_mask = None
        super().clear()

    def serialize(self):
        """マスクの状態をシリアライズ"""
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))
        
        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)

        lines = []
        for line in self.lines:
            lines.append({
                'is_erasing': line.is_erasing,
                'size': line.size,
                'soft': line.soft,
                'points': copy.deepcopy(line.points)
            })
        
        dict = {
            'type': MaskType.FREEDRAW,
            'name': self.name,
            'center': [cx, cy],
            'lines': lines,
            'effects_param': param
        }
        return dict

    def deserialize(self, dict):
        self.initializing = False
        self.name = dict['name']
        cx, cy = dict['center']

        lines = []
        for line in dict['lines']:
            lineobj = FreeDrawMask.Line(
                is_erasing=line['is_erasing'],
                size=line['size'],
                soft=line['soft'],
            )
            for point in line['points']:
                lineobj.add_point(*point)
            lines.append(lineobj)
        self.lines = lines

        self.effects_param.update(dict['effects_param'])
        self.center = params.denorm_param(self.effects_param, (cx, cy))

        self.create_control_points()

    def on_mouse_pos(self, window, pos):
        # 自分が active / 作成中でないなら brush_cursor を完全に非表示にする。
        # __init__ で KVWindow.bind(mouse_pos=self.on_mouse_pos) しているので、
        # マスクを選択していない / 別マスクを選択中でも mouse_pos イベントは飛んでくる。
        # 表示制御は brush_color の alpha で行う。
        root = getattr(self.editor, 'root', None)
        is_liquify_active = False
        try:
            check = getattr(root, 'is_liquify_editor_active', None)
            is_liquify_active = bool(check()) if callable(check) else False
        except Exception:
            # 高頻度経路のためログ抑制
            is_liquify_active = False
        if is_liquify_active:
            self.brush_color.rgba = (0, 0, 0, 0)
            return
        is_active = self.editor.get_active_mask() is self
        is_created = self.editor.get_created_mask() is self
        if not (is_active or is_created):
            self.brush_color.rgba = (0, 0, 0, 0)
            return
        self.brush_color.rgba = (1, 1, 1, 1)
        self.update_brush_cursor(pos[0], pos[1])

    def _pan_mode_active(self):
        """スペースキー押下中はパン優先。Draw 描画系の touch を完全に無効化する。"""
        root = getattr(self.editor, 'root', None)
        return bool(getattr(root, 'is_press_space', False))

    def on_touch_down(self, touch):
        if self._pan_mode_active():
            # マスク側はイベントを掴まない。親 (preview_widget) の panning handler に流す。
            return False
        if self.editor.get_active_mask() != self and self.editor.get_created_mask() != self:
            return super().on_touch_down(touch)
        
        if self.editor.is_center_click_anyone(touch, self):
            return False

        if touch.is_mouse_scrolling:
            if self.editor.collide_point(*touch.pos):
                # 描画中または消去中はブラシサイズを変更できない
                if self.current_line is None:
                    return self._adjust_draw_brush_from_scroll(touch)

        was_initializing = self.initializing
        if not self._touch_in_draw_area(touch):
            return super().on_touch_down(touch)

        if was_initializing:
            cx, cy = self.editor.window_to_tcg(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            self.create_control_points()
            self.editor.set_active_mask(self)
            self.initializing = False

        # 右クリックで消去モード、左クリックで描画モード
        is_erasing = (touch.button == 'right')
        if not was_initializing and self.editor.created_mask is None:
            get_history_ctrl().begin_history_layer_ctrl(self.editor, "Update", self.editor.get_mask_list().index(self), None)
            self._stroke_history_started = True

        self._drag_base_mask = self._current_committed_preview_base()

        self.brush_size = self._draw_brush_size()
        hardness = self._draw_brush_hardness()
        self.current_line = FreeDrawMask.Line(is_erasing, self.brush_size, hardness)
        self.current_line.add_point(*self.editor.window_to_tcg(*touch.pos))
        self.editor.set_active_mask(self)
        self.lines.append(self.current_line)

        self._redraw_mask_content("freedraw_touch_down")
        
        # 初期化時はBaseMaskの方を呼び出さない
        if was_initializing:
            return True

        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self._pan_mode_active():
            return False
        if self.current_line is not None:
            self.current_line.add_point(*self.editor.window_to_tcg(*touch.pos))
            self._redraw_mask_content("freedraw_touch_move")
            return True

        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self._pan_mode_active():
            # パンモード抜け遅延（touch_up が先に来る場合）に備え、
            # ストロークの後始末は通常パスでも行う。ここでは描画動作だけ抑止する。
            if self.current_line is not None:
                self.current_line = None
                self._drag_base_mask = None
                self._redraw_mask_content("freedraw_touch_up_pan")
                if self._stroke_history_started:
                    get_history_ctrl().end_history_layer_ctrl(self.editor, "Update", self.editor.get_mask_list().index(self))
                    self._stroke_history_started = False
            return False
        if self.current_line is not None:
            self.current_line = None
            self._drag_base_mask = None
            # マスクを更新
            self._redraw_mask_content("freedraw_touch_up")
            if self._stroke_history_started:
                get_history_ctrl().end_history_layer_ctrl(self.editor, "Update", self.editor.get_mask_list().index(self))
                self._stroke_history_started = False
            return True

        return super().on_touch_up(touch)

    def _touch_in_draw_area(self, touch):
        checker = getattr(self.editor, "window_point_in_image_rect", None)
        if callable(checker):
            return bool(checker(*touch.pos))
        return bool(self.editor.collide_point(*touch.pos))

    def create_control_points(self):
        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

    def update_brush_cursor(self, x, y):
        self.brush_size = self._draw_brush_size()
        brush_size = self.editor.tcg_to_window_scale(self.brush_size, 0)[0]
        self.translate.x, self.translate.y = x - brush_size / 2, y - brush_size / 2
        # update_mask がジオメトリ回転角を self.rotate.angle にセットするが、回転原点が
        # (0,0) のままだと円カーソルの中心 (brush/2, brush/2) が R·c-c だけ移動し、回転方向
        # へズレる。原点をカーソル中心に合わせ、その場回転(円なので見た目不変)にする。
        self.rotate.origin = (brush_size / 2, brush_size / 2)
        self.brush_cursor.ellipse = (0, 0, brush_size, brush_size)
        # 内側ハードゾーンの直径 = 外径 × hardness/100 (freedraw_raster の create_natural_brush
        # と同じ意味で、hardness=100 で同径、=0 で消失)
        try:
            hardness = self._draw_brush_hardness()
        except Exception:
            # 高頻度経路のためログ抑制
            hardness = 100.0
        inner_size = max(0.0, brush_size * (hardness / 100.0))
        inner_off = (brush_size - inner_size) / 2.0
        self.brush_cursor_inner.ellipse = (inner_off, inner_off, inner_size, inner_size)

    def _edge_refine_preview_freeze_enabled(self):
        if not self._quick_select_switch_enabled():
            return False
        mode = effects.Mask2Effect.get_param(self.effects_param, 'mask2_edge_refine_mode')
        return edge_refine.is_enabled(mode)

    def _current_committed_preview_base(self):
        if not self._edge_refine_preview_freeze_enabled():
            return None
        try:
            base = self.get_mask_image()
            return np.asarray(base, dtype=np.float32).copy()
        except Exception:
            logging.exception("FreeDraw Quick Select preview base capture failed")
            return None

    def _preview_current_line_over_base(self, image_size, copy_lines):
        if self.current_line is None or self._drag_base_mask is None:
            return None
        try:
            line_index = self.lines.index(self.current_line)
        except ValueError:
            # 高頻度経路のためログ抑制
            line_index = len(copy_lines) - 1
        if line_index < 0 or line_index >= len(copy_lines):
            return None

        base = np.asarray(self._drag_base_mask, dtype=np.float32)
        expected_shape = (int(image_size[1]), int(image_size[0]))
        if base.shape[:2] != expected_shape:
            return None

        src = copy_lines[line_index]
        preview_line = FreeDrawMask.Line(False, src.size, src.soft)
        for point in src.points:
            preview_line.add_point(*point)
        current = mask_rasters.draw_line_texture(
            image_size,
            [preview_line],
            allow_over_one=False,
            allow_under_zero=False,
        )
        current = self._apply_extened_params(
            current,
            edge_refine_draw_strokes=[preview_line],
            edge_refine_enabled=False,
        )
        if bool(getattr(src, "is_erasing", False)):
            return np.clip(base - current, 0.0, 1.0).astype(np.float32, copy=False)
        return np.maximum(base, current).astype(np.float32, copy=False)

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            return

        self.editor.set_scissor(self.scissor)
        self.rotate.angle = math.degrees(self.editor.get_rotate_rad(0))

        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def get_mask_image(self):
        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        copy_lines = []
        for i, src_line in enumerate(self.lines):
            copy_line = FreeDrawMask.Line(src_line.is_erasing, self.editor.tcg_to_image_scale(src_line.size, 0)[0], src_line.soft)
            for point in src_line.points:
                copy_line.add_point(*self.editor.tcg_to_texture(*point))
            copy_lines.append(copy_line)

        line_hash = tuple(
            (line.is_erasing, line.size, line.soft, tuple(line.points))
            for line in self.lines
        )
        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, line_hash))
        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
            allow_over_one = False
            allow_under_zero = False
            mask = mask_rasters.draw_line_texture(
                image_size,
                copy_lines,
                allow_over_one=allow_over_one,
                allow_under_zero=allow_under_zero,
            )

            # Invert (Gradient/Circular と同じ流儀: raster 後・extended_params 前)
            if effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_settings') == True:
                if effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert') == True:
                    mask = 1.0 - mask

            drag_preview = self._preview_current_line_over_base(image_size, copy_lines)
            if drag_preview is not None:
                mask = drag_preview
            else:
                # ルミナンスとマスクを作成
                full_refined = extended_params.render_freedraw_edge_refine_full_view(
                    self.editor,
                    self.effects_param,
                    self.lines,
                    self.center,
                    mask.shape,
                    debug_label=f"{self.__class__.__name__}Full",
                )
                if full_refined is None:
                    mask = self._apply_extened_params(mask, edge_refine_draw_strokes=copy_lines)
                else:
                    mask = full_refined

            self.image_mask_cache = mask
            self.image_mask_cache_hash = newhash

        return self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)


# 折れ線マスクのクラス
class PolylineMask(BaseMask):
    """頂点をクリックして折れ線を描き、閉じれば塗りつぶせるマスク。

    各 polyline は確定後に頂点 ControlPoint で再編集できる。
    """

    # 始点と現在地が十分近いと判定する画面上のピクセル距離 (TCG 換算は描画時に動的計算)
    _CLOSE_HIT_RADIUS_PX = 16.0

    class Polyline:
        def __init__(self, is_erasing=False, size=10, soft=100,
                     is_closed=False, is_filled=True):
            self.is_erasing = bool(is_erasing)
            self.size = float(size)
            self.soft = float(soft)
            self.is_closed = bool(is_closed)
            self.is_filled = bool(is_filled)
            self.points = []

        def add_point(self, x, y):
            self.points.append((float(x), float(y)))

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Polyline"
        self.initializing = True

        self.polylines = []          # 確定済み polyline のリスト
        self.current_polyline = None # 描画中の polyline
        self.brush_size = self._draw_brush_size()  # 線幅 (TCG)
        self._stroke_history_started = False
        self._last_finish_time = 0.0  # 直近の polyline 確定時刻 (シングルクリック閉じ直後のダブルタップ余波抑制用)

        # コントロールポイント描画状態
        self._vertex_control_points = []  # [(polyline_idx, point_idx, ControlPoint)]

        with self.canvas:
            # スコープ A: ラバーバンドと始点ハイライト (ウィンドウ座標、translate なし)
            KVPushMatrix()
            self._overlay_scissor = self.editor.push_scissor()
            self.preview_color = KVColor((1, 1, 0, 0))   # 初期不可視
            self.preview_line = KVLine(points=[], width=1)
            self.start_color = KVColor((0, 1, 1, 0))
            self.start_indicator = KVLine(ellipse=(0, 0, 0, 0), width=2)
            self.editor.pop_scissor()
            KVPopMatrix()

            # スコープ B: ブラシカーソル (translate でカーソル位置に移動)
            KVPushMatrix()
            self.scissor = self.editor.push_scissor()
            self.translate = KVTranslate(0, 0)
            self.rotate = KVRotate(angle=0, origin=(0, 0))
            self.brush_color = KVColor((0, 1, 1, 0))     # 初期不可視
            self.brush_cursor = KVLine(ellipse=(0, 0, self.brush_size, self.brush_size), width=2)
            # 内側ハードゾーン (hardness の視覚化用 2 重目の円)
            self.brush_cursor_inner = KVLine(ellipse=(0, 0, 0, 0), width=2)
            self.editor.pop_scissor()
            KVPopMatrix()

        # マスクの生存期間全体で bind したままにする(start/end で張り直さない)。
        # FreeDrawMask と同様、選択されていない間の表示抑制はハンドラ側で行う。
        KVWindow.bind(mouse_pos=self.on_mouse_pos)
        self._mouse_pos_bound = True

    def _edge_refine_fill_grown_region(self):
        return True

    def _get_edge_refine_seed_mask(self, mask_shape, current_mask=None):
        if current_mask is None:
            return None
        seed = edge_refine.make_confident_seed(current_mask)
        return seed if np.any(seed) else None

    def _edge_refine_selection_strategy(self):
        return edge_refine.STRATEGY_DRAW

    # ---- ライフサイクル ----
    def start(self):
        self.brush_color.rgba = (1, 1, 1, 1)

    def end(self):
        # 描画中の polyline は終了扱い (開いた折れ線として確定 or 破棄)
        self.commit_in_progress()
        self.brush_color.rgba = (0, 0, 0, 0)
        self.preview_color.rgba = (1, 1, 0, 0)
        self.start_color.rgba = (0, 1, 1, 0)

    def release(self):
        # マスク削除時に Window の bind を解除する(unbind は二重に呼んでも安全)。
        if getattr(self, '_mouse_pos_bound', False):
            KVWindow.unbind(mouse_pos=self.on_mouse_pos)
            self._mouse_pos_bound = False

    def in_creation_stroke(self):
        # 描画中の polyline がある間は作成確定を遅らせる(確定は初回 polyline の
        # コミット後。これで Create オペレーションに初期形状が正しく入る)。
        return self.current_polyline is not None

    def clear(self):
        self.polylines = []
        self.current_polyline = None
        self._clear_vertex_control_points()
        super().clear()

    # ---- シリアライズ ----
    def serialize(self):
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)

        polys = []
        for p in self.polylines:
            polys.append({
                'is_erasing': p.is_erasing,
                'size': p.size,
                'soft': p.soft,
                'is_closed': p.is_closed,
                'is_filled': p.is_filled,
                'points': copy.deepcopy(p.points),
            })

        return {
            'type': MaskType.POLYLINE,
            'name': self.name,
            'center': [cx, cy],
            'polylines': polys,
            'effects_param': param,
        }

    def deserialize(self, dict):
        self.initializing = False
        self.name = dict['name']
        cx, cy = dict['center']

        polys = []
        for p in dict.get('polylines', []):
            polyobj = PolylineMask.Polyline(
                is_erasing=p.get('is_erasing', False),
                size=p.get('size', 10),
                soft=p.get('soft', 100),
                is_closed=p.get('is_closed', False),
                is_filled=p.get('is_filled', True),
            )
            for point in p.get('points', []):
                polyobj.add_point(*point)
            polys.append(polyobj)
        self.polylines = polys

        self.effects_param.update(dict['effects_param'])
        self.center = params.denorm_param(self.effects_param, (cx, cy))

        self.create_control_points()
        # 確定 polyline の頂点 ControlPoint も復元
        self._rebuild_vertex_control_points()

    # ---- マウス入力 ----
    def on_mouse_pos(self, window, pos):
        # 自分が active / 作成中でないなら brush_cursor + preview_line を非表示にする。
        # FreeDrawMask と同等。
        is_active = self.editor.get_active_mask() is self
        is_created = self.editor.get_created_mask() is self
        if not (is_active or is_created):
            self.brush_color.rgba = (0, 0, 0, 0)
            # preview line も消す
            try:
                self.preview_line.points = []
            except Exception:
                # 高頻度経路のためログ抑制
                pass
            return
        self.brush_color.rgba = (1, 1, 1, 1)
        self.update_brush_cursor(pos[0], pos[1])
        self._update_preview_line(pos)

    def _pan_mode_active(self):
        root = getattr(self.editor, 'root', None)
        return bool(getattr(root, 'is_press_space', False))

    def _close_hit_distance_tcg(self):
        """始点との距離判定用しきい値 (TCG 単位)。"""
        return self.editor.window_to_tcg_scale(self._CLOSE_HIT_RADIUS_PX, 0)[0]

    def _is_near_first_point(self, tcg_x, tcg_y):
        if self.current_polyline is None or len(self.current_polyline.points) < 2:
            return False
        sx, sy = self.current_polyline.points[0]
        thr = self._close_hit_distance_tcg()
        return (tcg_x - sx) ** 2 + (tcg_y - sy) ** 2 <= thr * thr

    def consumes_double_tap(self, touch):
        """ダブルタップを polyline 確定として消費するかどうか。
        プレビュー領域内で描画中の場合、または直前のクリックで始点閉じ確定した直後
        (ダブルクリックの 2 打目が置き去りになるケース) は True を返し、
        preview_widget のズーム切替を抑制する。"""
        if self.current_polyline is None:
            return self._recently_finished_polyline()
        try:
            return bool(self.editor.collide_point(*touch.pos))
        except Exception:
            return False

    def on_touch_down(self, touch):
        if self._pan_mode_active():
            return False

        # アクティブマスクが自分でないかつ作成中でもないなら標準 ControlPoint 経路
        if self.editor.get_active_mask() != self and self.editor.get_created_mask() != self:
            return super().on_touch_down(touch)

        # preview_widget (= self.editor) の外側 (パラメータパネル等) のクリックは無視。
        # editor 内なら image 範囲外 (レターボックス部分) でも頂点設定 OK。
        if not self.editor.collide_point(*touch.pos):
            return False

        if self.editor.is_center_click_anyone(touch, self):
            return False

        # スクロールで線幅 (描画中以外のみ)
        if touch.is_mouse_scrolling:
            if self.editor.collide_point(*touch.pos):
                if self.current_polyline is None:
                    return self._adjust_draw_brush_from_scroll(touch)

        # 確定 polyline の頂点 ControlPoint クリックを最優先で処理する。
        # 描画中 (current_polyline is not None) はラバーバンドと衝突するので無効化。
        if self.current_polyline is None and not self.initializing:
            for cp in self._iter_vertex_control_points():
                cx, cy = self.editor.window_to_tcg(*touch.pos)
                if cp.collide_point(cx, cy):
                    return super().on_touch_down(touch)

        tcg_x, tcg_y = self.editor.window_to_tcg(*touch.pos)

        # 初期化時 (最初の左クリックでのみマスク中心を確定)。
        # 右クリックでの初期化は中心だけ残って polyline が始まらないので不可。
        was_initializing = self.initializing
        if was_initializing and touch.button == 'left' and not getattr(touch, 'is_double_tap', False):
            if not self._begin_initial_touch_if_in_placement_area(touch):
                return False
            self.center_x = tcg_x
            self.center_y = tcg_y
            self.create_control_points()
            self.editor.set_active_mask(self)
            self.initializing = False
            self._initial_touch_started = False

        # 右クリック: 描画中のみ直近頂点を取消 (idle 右クリックは何もしない)
        if touch.button == 'right':
            if self.current_polyline is not None:
                if self.current_polyline.points:
                    self.current_polyline.points.pop()
                if len(self.current_polyline.points) <= 0:
                    self.current_polyline = None
                    if self._stroke_history_started:
                        get_history_ctrl().end_history_layer_ctrl(
                            self.editor, "Update", self.editor.get_mask_list().index(self))
                        self._stroke_history_started = False
                self._redraw_mask_content("polyline_undo_point")
                return True
            # idle 右クリックは消費せず親に流す
            return super().on_touch_down(touch)

        # 左クリック (描画/閉じる/ダブルクリック開放確定)
        if touch.button == 'left':
            # ダブルタップ: 直近の頂点はそのまま採用し開放確定 (旧仕様は前段 down の頂点を
            # pop していたが、ユーザーが置いた頂点をキャンセルしないでほしいとの要望)。
            if getattr(touch, 'is_double_tap', False):
                if self.current_polyline is not None:
                    self._finish_current_polyline(is_closed=False)
                    self._redraw_mask_content("polyline_finish_open")
                    if self.initializing:
                        self.initializing = False
                    return True
                # 1 打目 (シングルクリック) で既に始点閉じ確定済みなら、2 打目は
                # 置き去りの余波なので何もせず消費するだけ (ズーム切替に流さない)。
                if self._recently_finished_polyline():
                    return True
                return super().on_touch_down(touch)

            # 描画中で始点付近をクリックしたら閉じて確定
            if self.current_polyline is not None and self._is_near_first_point(tcg_x, tcg_y):
                self._finish_current_polyline(is_closed=True)
                self._redraw_mask_content("polyline_finish_closed")
                if self.initializing:
                    self.initializing = False
                return True

            # 通常の頂点追加
            if self.current_polyline is None:
                self._begin_new_polyline(tcg_x, tcg_y, is_erasing=False)
            else:
                self.current_polyline.add_point(tcg_x, tcg_y)
                self._redraw_mask_content("polyline_add_point")
            if was_initializing:
                return True
            return super().on_touch_down(touch)

        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self._pan_mode_active():
            return False
        # 描画中のラバーバンドは on_mouse_pos で更新するため move は不要
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self._pan_mode_active():
            return False
        return super().on_touch_up(touch)

    # ---- 描画中 polyline 制御 ----
    def _begin_new_polyline(self, tcg_x, tcg_y, is_erasing):
        # マスク作成中は Create 用の begin が保留中のため Update を begin しない
        # (FreeDrawMask の was_initializing ガードと同じ意図。初回ストロークは
        # Create オペレーションのシリアライズに含まれる)
        if not self._stroke_history_started and self.editor.created_mask is None:
            get_history_ctrl().begin_history_layer_ctrl(
                self.editor, "Update", self.editor.get_mask_list().index(self), None)
            self._stroke_history_started = True
        self.brush_size = self._draw_brush_size()
        hardness = self._draw_brush_hardness()
        self.current_polyline = PolylineMask.Polyline(
            is_erasing=is_erasing,
            size=self.brush_size,
            soft=hardness,
            is_closed=False,
            is_filled=True,
        )
        self.current_polyline.add_point(tcg_x, tcg_y)
        self.editor.set_active_mask(self)
        self._redraw_mask_content("polyline_begin")

    def _recently_finished_polyline(self):
        """直前の touch で polyline を確定 (単クリックでの始点閉じ含む) した直後かどうか。
        ダブルクリックの 1 打目で閉じた場合、2 打目は current_polyline が None のまま
        on_image_touch_down に先着されズーム切替と誤認されるため、その余波を抑制する。"""
        elapsed = time.time() - self._last_finish_time
        threshold = KVConfig.getint('postproc', 'double_tap_time') * 0.001
        return 0.0 <= elapsed <= threshold

    def _finish_current_polyline(self, is_closed: bool):
        if self.current_polyline is None:
            return
        self._last_finish_time = time.time()
        # 頂点 2 個未満なら破棄
        if len(self.current_polyline.points) < 2:
            self.current_polyline = None
        else:
            self.current_polyline.is_closed = bool(is_closed)
            # 塗りつぶしの可否は param (mask2_polyline_fill) で動的判定するので
            # is_filled の値はここでは触らない (旧仕様の不可逆的な False 化は廃止)。
            self.polylines.append(self.current_polyline)
            self.current_polyline = None
            # 確定 polyline の頂点 ControlPoint を生成
            self._rebuild_vertex_control_points()
        if self._stroke_history_started:
            get_history_ctrl().end_history_layer_ctrl(
                self.editor, "Update", self.editor.get_mask_list().index(self))
            self._stroke_history_started = False

    def commit_in_progress(self):
        """描画中の polyline があれば「開いた折れ線」として確定する。
        タブ切替やマスク非アクティブ化など、描画コンテキストを抜けるときに呼ぶ。"""
        if self.current_polyline is not None:
            self._finish_current_polyline(is_closed=False)
            self._redraw_mask_content("polyline_commit_in_progress")
        # 残留オーバーレイをリセット
        self.preview_line.points = []
        self.preview_color.rgba = (1, 1, 0, 0)
        self.start_indicator.ellipse = (0, 0, 0, 0)
        self.start_color.rgba = (0, 1, 1, 0)

    def _update_preview_line(self, window_pos):
        """ラバーバンド (直近頂点 → カーソル) と始点強調を更新。"""
        if self.current_polyline is None or len(self.current_polyline.points) == 0:
            self.preview_color.rgba = (1, 1, 0, 0)
            self.start_color.rgba = (0, 1, 1, 0)
            return
        # 直近頂点 → カーソル 線分
        last_tcg = self.current_polyline.points[-1]
        last_wx, last_wy = self.editor.tcg_to_window(*last_tcg)
        self.preview_line.points = [last_wx, last_wy, window_pos[0], window_pos[1]]
        self.preview_color.rgba = (1, 1, 0, 0.8)

        # 始点強調 (頂点が 2 個以上、つまり閉じ判定可能なときのみ表示)
        if len(self.current_polyline.points) >= 2:
            sx, sy = self.editor.tcg_to_window(*self.current_polyline.points[0])
            r_px = self._CLOSE_HIT_RADIUS_PX
            self.start_indicator.ellipse = (sx - r_px, sy - r_px, r_px * 2, r_px * 2)
            tcg_x, tcg_y = self.editor.window_to_tcg(*window_pos)
            if self._is_near_first_point(tcg_x, tcg_y):
                self.start_color.rgba = (0, 1, 0, 1)  # 閉じる予告
            else:
                self.start_color.rgba = (0, 1, 1, 0.6)
        else:
            self.start_color.rgba = (0, 1, 1, 0)

    # ---- ControlPoint ----
    def create_control_points(self):
        # 中心 ControlPoint (FreeDrawMask に倣う)
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

    def _iter_vertex_control_points(self):
        for _, _, cp in self._vertex_control_points:
            yield cp

    def _vertex_cp_set(self):
        return {cp for _, _, cp in self._vertex_control_points}

    def show_all_control_points(self, redraw=True):
        """BaseMask の実装は非中心 CP を全部赤に塗るが、Polyline では頂点 CP は青を維持する。
        redraw=False は refresh_mask_visibility 等の一括更新用(最後にまとめて再描画される)。"""
        self.opacity = 1.0
        vertex_cps = self._vertex_cp_set()
        for cp in self.control_points:
            cp.opacity = 1
            if cp.is_center:
                cp.color = [0, 1, 0]
            elif cp in vertex_cps:
                cp.color = [0.2, 0.6, 1.0]  # 青
            else:
                cp.color = [1, 0, 0]
        self.is_draw_mask = True
        if redraw:
            self.update_mask()

    def show_center_control_point_only(self, redraw=True):
        """非アクティブ時: 中心 CP のみ表示 (頂点 CP は隠す)。"""
        self.opacity = 0.2
        vertex_cps = self._vertex_cp_set()
        for cp in self.control_points:
            if cp.is_center:
                cp.opacity = 2
                cp.color = [1, 0, 0]
            else:
                # 頂点 CP もそれ以外も非表示
                cp.opacity = 0
                if cp in vertex_cps:
                    cp.color = [0.2, 0.6, 1.0]  # 復帰時用に色だけ保持
        self.is_draw_mask = False
        if redraw:
            self.update_mask()

    def _clear_vertex_control_points(self):
        for _, _, cp in self._vertex_control_points:
            self.remove_widget(cp)
            if cp in self.control_points:
                self.control_points.remove(cp)
        self._vertex_control_points = []

    def _rebuild_vertex_control_points(self):
        """確定 polyline の各頂点に ControlPoint を生成する。"""
        self._clear_vertex_control_points()
        for pi, poly in enumerate(self.polylines):
            for vi, (px, py) in enumerate(poly.points):
                cp = ControlPoint(self.editor)
                cp.center = (px, py)
                cp.ctrl_center = cp.center
                cp.is_center = False
                cp.color = [0.2, 0.6, 1.0]  # 青系
                cp.bind(ctrl_center=self._make_vertex_callback(pi, vi))
                self.control_points.append(cp)
                self.add_widget(cp)
                self._vertex_control_points.append((pi, vi, cp))
        # アクティブ/非アクティブの表示状態を最新化
        if self.active:
            self.show_all_control_points()
        else:
            self.show_center_control_point_only()

    def _make_vertex_callback(self, pi, vi):
        def _cb(instance, value):
            try:
                self.polylines[pi].points[vi] = (instance.ctrl_center[0], instance.ctrl_center[1])
            except (IndexError, AttributeError):
                # 高頻度経路のためログ抑制
                return
            # ControlPoint の見た目位置も更新 (ctrl_center だけでは center が動かない)
            new_center = (instance.ctrl_center[0], instance.ctrl_center[1])
            if instance.center[0] == new_center[0] and instance.center[1] == new_center[1]:
                instance.property('center').dispatch(instance)
            else:
                instance.center = new_center
            self._redraw_mask_content("polyline_vertex_move")
        return _cb

    def on_center_control_point_move(self, instance, value):
        # 中心移動: 全 polyline の全頂点を平行移動 (FreeDrawMask 同様の流儀)
        dx = instance.ctrl_center[0] - self.center_x
        dy = instance.ctrl_center[1] - self.center_y
        self.center = (self.center_x + dx, self.center_y + dy)
        # 確定 polyline の頂点を平行移動
        for poly in self.polylines:
            poly.points = [(p[0] + dx, p[1] + dy) for p in poly.points]
        # 描画中 polyline も追従させる
        if self.current_polyline is not None:
            self.current_polyline.points = [(p[0] + dx, p[1] + dy) for p in self.current_polyline.points]
        # 頂点 ControlPoint の center 値も同期 (super の制御点移動は中心のみ動かす)
        for pi, vi, cp in self._vertex_control_points:
            try:
                px, py = self.polylines[pi].points[vi]
                if cp.center[0] == px and cp.center[1] == py:
                    cp.property('center').dispatch(cp)
                else:
                    cp.center = (px, py)
            except (IndexError, AttributeError):
                # 高頻度経路のためログ抑制
                pass
        # 中心 ControlPoint の center も追従
        super_cp_iter = (cp for cp in self.control_points if cp.is_center)
        for cp in super_cp_iter:
            center = (cp.center_x + dx, cp.center_y + dy)
            if cp.center[0] == center[0] and cp.center[1] == center[1]:
                cp.property('center').dispatch(cp)
            else:
                cp.center = center
        self._redraw_mask_content("polyline_center_move")

    # ---- カーソル ----
    def update_brush_cursor(self, x, y):
        self.brush_size = self._draw_brush_size()
        brush_size = self.editor.tcg_to_window_scale(self.brush_size, 0)[0]
        self.translate.x, self.translate.y = x - brush_size / 2, y - brush_size / 2
        # update_mask がジオメトリ回転角を self.rotate.angle にセットするが、回転原点が
        # (0,0) のままだと円カーソルの中心 (brush/2, brush/2) が R·c-c だけ移動し、回転方向
        # へズレる。原点をカーソル中心に合わせ、その場回転(円なので見た目不変)にする。
        self.rotate.origin = (brush_size / 2, brush_size / 2)
        self.brush_cursor.ellipse = (0, 0, brush_size, brush_size)
        # 内側ハードゾーンの直径 = 外径 × hardness/100 (FreeDrawMask と同じ意味)
        try:
            hardness = self._draw_brush_hardness()
        except Exception:
            # 高頻度経路のためログ抑制
            hardness = 100.0
        inner_size = max(0.0, brush_size * (hardness / 100.0))
        inner_off = (brush_size - inner_size) / 2.0
        self.brush_cursor_inner.ellipse = (inner_off, inner_off, inner_size, inner_size)

    # ---- マスク描画 ----
    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            return
        self.editor.set_scissor(self.scissor)
        self.rotate.angle = math.degrees(self.editor.get_rotate_rad(0))

        if self.is_draw_mask:
            if self.do_draw_composit_mask:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def _build_render_polylines(self, image_size):
        """確定 polyline + 描画中 polyline をテクスチャ座標に変換して返す。

        塗りつぶし (is_filled) は param mask2_polyline_fill で動的に決定する
        (チェックボックス ON/OFF で全 polyline 即時切り替え)。
        ただし開いた折れ線 (is_closed=False) は塗りつぶし不可。
        """
        fill_enabled = bool(effects.Mask2Effect.get_param(self.effects_param, 'mask2_polyline_fill'))
        render = list(self.polylines)
        if self.current_polyline is not None and len(self.current_polyline.points) >= 1:
            render.append(self.current_polyline)
        result = []
        for src in render:
            tex_poly = mask_rasters.Polyline(
                is_erasing=src.is_erasing,
                size=self.editor.tcg_to_image_scale(src.size, 0)[0],
                soft=src.soft,
                is_closed=src.is_closed,
                is_filled=fill_enabled and src.is_closed,
            )
            for point in src.points:
                tex_poly.add_point(*self.editor.tcg_to_texture(*point))
            result.append(tex_poly)
        return result

    def get_mask_image(self):
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        copy_polys = self._build_render_polylines(image_size)

        fill_enabled = bool(effects.Mask2Effect.get_param(self.effects_param, 'mask2_polyline_fill'))
        poly_hash = tuple(
            (p.is_erasing, p.size, p.soft, p.is_closed, tuple(p.points))
            for p in (self.polylines + ([self.current_polyline] if self.current_polyline else []))
        )
        # fill_enabled もキャッシュキーに含めて、チェックボックス ON/OFF で再計算させる
        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size, poly_hash, fill_enabled))

        if (self.image_mask_cache is None or self.image_mask_cache_hash != newhash) and self.initializing == False:
            mask = mask_rasters.draw_polyline_texture(
                image_size,
                copy_polys,
                allow_over_one=False,
                allow_under_zero=False,
            )

            # Invert (Gradient/Circular と同じ流儀: raster 後・extended_params 前)
            if effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_settings') == True:
                if effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert') == True:
                    mask = 1.0 - mask

            mask = self._apply_extened_params(mask)
            self.image_mask_cache = mask
            self.image_mask_cache_hash = newhash

        return self.image_mask_cache if self.image_mask_cache is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)


# セグメントマスクのクラス
class SegmentMask(BaseMask):
    corner = KVListProperty([0, 0])

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Segment"
        self.initializing = True  # 初期配置中かどうか

        self.center = (0, 0)
        self.corner = (0, 0)

        self.segment_mask_cache = None
        self.segment_mask_cache_hash = None

        with self.canvas:
            KVPushMatrix()
            self.scissor = self.editor.push_scissor()
            # center位置への移動
            self.translate = KVTranslate(0, 0)
            KVColor(1, 0, 0, 1)
            self.rect_line = KVLine(points=[], close=True, width=max(1.0, 1.5 * device.dpi_scale()))
            self.editor.pop_scissor()
            KVPopMatrix()

        #self.update_mask()

    def follows_mask_geometry(self):
        return False

    def _edge_refine_support_softness(self):
        return 1.0

    def on_touch_down(self, touch):
        if self.initializing:
            if not self._begin_initial_touch_if_in_placement_area(touch):
                return False
            cx, cy = self.window_to_tcg_for_interaction(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            self.corner = [cx, cy]
            #self.update_mask()
            return True
        else: 
            self.is_draw_mask = False
            handled = super().on_touch_down(touch)
            if handled:
                self.editor.draw_mask_image(None)
                self.update_mask()
            return handled

    def on_touch_move(self, touch):
        if self.initializing:
            if not self._initial_touch_started:
                return False
            cx, cy = self.window_to_tcg_for_interaction(*touch.pos)
            self.corner = [cx, cy]
            self.update_mask()
            return True
        else:
            for cp in self.control_points:
                if cp.touching:
                    cp.on_touch_move(touch)
                    self.is_draw_mask = False
                    self.update_mask()
                    return True
            return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.initializing:
            if not self._initial_touch_can_finish():
                return False
            self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            self.get_mask_image() # 即座に計算開始
            #self.update_mask()
            self.update_draw_mask()
            return True
        else:
            for cp in self.control_points:
                if cp.touching:
                    cp.on_touch_up(touch)
                    get_history_ctrl().end_history_layer_ctrl(
                        self.editor, "Update", self.editor.get_mask_list().index(self)
                    )
                    self.is_draw_mask = True
                    self.update_mask()
                    self.editor.request_mask_render_update(
                        self,
                        reason="segment_control_point_touch_up",
                        redraw_overlay=True,
                        redraw_pipeline=True,
                    )
                    return True
            return super().on_touch_up(touch)

    def create_control_points(self):
        # 中心のコントロールポイント（始点）
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        # コーナーのコントロールポイント（終点）
        cp_corner = ControlPoint(self.editor)
        cp_corner.center = self.corner
        cp_corner.ctrl_center = cp_corner.center
        cp_corner.type = ['corner', 0]
        # コーナーもコントロールポイントとして独立して動かせるようにする
        cp_corner.bind(ctrl_center=self.on_corner_control_point_move)
        self.control_points.append(cp_corner)
        self.add_widget(cp_corner)

        if not self.active:
            self.show_center_control_point_only()

    def serialize(self):
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))
        crx, cry = params.norm_param(self.effects_param, (self.corner[0], self.corner[1]))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)
        
        dict = {
            'type': MaskType.SEGMENT,
            'name': self.name,
            'center': [cx, cy],
            'corner': [crx, cry],
            'effects_param': param
        }
        # マスクデータはストア参照のみ保存(実体は AIImageCache 共有ストア側にある)
        self._serialize_image_mask_cache(dict)

        return dict

    def deserialize(self, dict):
        self.initializing = False
        cx, cy = dict['center']
        crx, cry = dict.get('corner', [cx, cy]) # 後方互換性
        self.name = dict['name']
        self.effects_param.update(dict['effects_param'])
        self.center = params.denorm_param(self.effects_param, (cx, cy))
        self.corner = params.denorm_param(self.effects_param, (crx, cry))

        # マスクデータ展開(ストア参照。旧形式は移行)
        self._deserialize_image_mask_cache(dict)

        # 描き直し
        self.create_control_points()
        #self.update_mask()

    def update_control_points(self):
        if len(self.control_points) > 0:
            cp_center = self.control_points[0]
            cp_center.center = self.center
        if len(self.control_points) > 1:
            cp_corner = self.control_points[1]
            cp_corner.center = self.corner

    def on_center_control_point_move(self, instance, value):
        # 始点移動：コーナーは動かさない（ボックスの形が変わる）
        self.center = value
        self.update_control_points()

        #super().on_center_control_point_move(instance, value)
        # update_maskはsuper()の中で呼ばれる

    def on_corner_control_point_move(self, instance, value):
        self.corner = value
        self.update_control_points()
#        instance.center = value
        #self.update_mask()

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            logging.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            self.editor.set_scissor(self.scissor)
            cx, cy = self.center
            crx, cry = self.corner

            wp1 = self.tcg_to_window_for_overlay(cx, cy)
            wp2 = self.tcg_to_window_for_overlay(crx, cy)
            wp3 = self.tcg_to_window_for_overlay(crx, cry)
            wp4 = self.tcg_to_window_for_overlay(cx, cry)

            # BaseMaskの仕組みでTranslateされているが、回転に対応するためTranslateを無効化（0,0）して絶対座標で描く
            self.translate.x, self.translate.y = 0, 0

            self.rect_line.points = [*wp1, *wp2, *wp3, *wp4]

    def update_draw_mask(self):
        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        original_image_size = tuple(self.editor.get_image_size())
        center = self.editor.tcg_to_original_image(*self.center)
        corner = self.editor.tcg_to_original_image(*self.corner)
        if effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_settings') == True:
            invert = effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert')
        else:
            invert = False
        segment_mask = None

        # _draw_segmentを呼び出さなければならない用
        cache_key = cache_keys.segment_cache_key(original_image_size, center, corner, False)
        if (self.image_mask_cache is None or self.image_mask_cache_key != cache_key) and self.initializing == False:
            # [DIAG-REINFER] 再推論判定の診断ログ（原因特定後に削除）
            try:
                _store_get = getattr(self.editor, "get_ai_mask_bitmap", None)
                _in_store = _store_get(cache_key) is not None if callable(_store_get) else None
                _orig = self.editor.get_original_image_rgb()
                logging.warning(
                    "[DIAG-REINFER] segment reinfer-trigger cache_present=%s key_match=%s in_store=%s store_id=%s\n"
                    "  stored_key   =%s\n  recomputed_key=%s\n"
                    "  self.center=%s self.corner=%s original_image_size=%s orig_rgb_shape=%s",
                    self.image_mask_cache is not None,
                    self.image_mask_cache_key == cache_key,
                    _in_store,
                    id(getattr(self.editor, "ai_image_cache", None)),
                    self.image_mask_cache_key, cache_key,
                    self.center, self.corner, original_image_size,
                    (None if _orig is None else getattr(_orig, "shape", None)),
                )
            except Exception:
                logging.exception("[DIAG-REINFER] logging failed")
            # 描画
            cx, cy = center
            crx, cry = corner
            
            # 2点からバウンディングボックスを計算 (XYWH)
            min_x = min(cx, crx)
            min_y = min(cy, cry)
            w = abs(cx - crx)
            h = abs(cy - cry)
            
            # predict_sam3 に渡す box = [x, y, w, h]
            processing_dialog.set_processing_text("Make SegmentMask SAM3...")
            segment_mask = self._get_or_compute_image_mask_cache(
                cache_key,
                lambda: processing_dialog.wait_processing(self._draw_segment, original_image_size, [min_x, min_y, w, h], False),
                "SegmentMask SAM3",
            )
            #segment_mask = self._draw_segment(original_image_size, [min_x, min_y, w, h])

        # その他更新用
        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size))
        if self.image_mask_cache is not None and (self.image_mask_cache is segment_mask or self.segment_mask_cache is None or self.segment_mask_cache_hash != newhash) and self.initializing == False:
            self.segment_mask_cache_hash = newhash

            # SegmentMask用のキャッシュ
            segment_mask = self.image_mask_cache
            if invert:
                segment_mask = 1.0 - segment_mask

            # パラメータに従って画像を変形
            disp_info, rotate_rad, flip, matrix = self.editor.get_hash_items()
            segment_mask = core.rotation(segment_mask, np.rad2deg(rotate_rad), flip, np.array(matrix).reshape(3, 3))
            #segment_mask = core.crop_image_with_disp_info(segment_mask, disp_info)

            segment_mask = self._fit_image_mask_to_texture(segment_mask)

            # ルミノシティマスクを作成
            segment_mask = self._apply_extened_params(segment_mask)

            self.segment_mask_cache = segment_mask

        if segment_mask is None:
            segment_mask = self.segment_mask_cache

        return segment_mask if segment_mask is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    def _draw_segment(self, image_size, bbox, invert):
        from cores.mask2 import inference_runtime as mask2_inference_runtime

        img = self.editor.get_original_image_rgb()
        return mask2_inference_runtime.predict_sam3_bbox(img, bbox, invert)

class DepthMapMask(BaseMask):

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Depth Map"
        self.initializing = True  # 初期配置中かどうか
        self.center = (0, 0)

        self.depth_map_mask_cache = None
        self.depth_map_mask_cache_hash = None

        with self.canvas:
            KVPushMatrix()
            self.translate = KVTranslate(*self.center)
            KVPopMatrix()

        #self.update_mask()

    def follows_mask_geometry(self):
        return False

    def _edge_refine_support_softness(self):
        return 1.0

    def on_touch_down(self, touch):
        if self.initializing:
            if not self._begin_initial_touch_if_in_placement_area(touch):
                return False
            cx, cy = self.window_to_tcg_for_interaction(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            return True
        else: 
            return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.initializing:
            if not self._initial_touch_can_finish():
                return False
            self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            return True
        else:
            return super().on_touch_up(touch)

    def create_control_points(self):
        self.control_points = []

        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        if not self.active:
            self.show_center_control_point_only()

    def serialize(self):
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)

        dict = {
            'type': MaskType.DEPTHMAP,
            'name': self.name,
            'center': [cx, cy],
            'effects_param': param
        }

        return dict

    def deserialize(self, dict):
        self.initializing = False
        cx, cy = dict['center']
        self.name = dict['name']
        self.effects_param.update(dict['effects_param'])
        self.center = params.denorm_param(self.effects_param, (cx, cy))

        # 描き直し
        self.create_control_points()
        #self.update_mask()
     
    def update_control_points(self):
        cp_center = self.control_points[0]
        cp_center.center = self.center

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            logging.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            cx, cy = self.tcg_to_window_for_overlay(*self.center)
            self.translate.x, self.translate.y = cx, cy
        
        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        original_image_size = tuple(self.editor.get_image_size())
        center = self.editor.tcg_to_original_image(*self.center)
        depth_map_mask = None

        from cores.mask2 import inference_runtime as mask2_inference_runtime
        cache_key = cache_keys.depth_cache_key(original_image_size, mask2_inference_runtime.DEPTH_MAP_ALGORITHM_VERSION)
        if self.initializing == False:
            processing_dialog.set_processing_text("Make DepthMapMask...")
            depth_map_mask = self.editor.get_ai_depth_map(
                cache_key,
                lambda: processing_dialog.wait_processing(self.draw_depth_map, original_image_size),
            )
            #depth_map_mask = self.draw_depth_map(original_image_size)

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size))
        if depth_map_mask is not None and (self.depth_map_mask_cache is None or self.depth_map_mask_cache_hash != newhash) and self.initializing == False:
            self.depth_map_mask_cache_hash = newhash

            # Depth Balance: 生 depth の値分布を near↔far で再配分(キャッシュは非破壊)
            depth_map_mask = extended_params.apply_depth_balance(
                depth_map_mask,
                effects.Mask2Effect.get_param(self.effects_param, 'mask2_depth_balance', 0),
            )

            # パラメータに従って画像を変形
            disp_info, rotate_rad, flip, matrix = self.editor.get_hash_items()
            depth_map_mask = core.rotation(depth_map_mask, np.rad2deg(rotate_rad), flip, np.array(matrix).reshape(3, 3))
            depth_map_mask = self._fit_image_mask_to_texture(depth_map_mask)
            if effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_settings') == True:
                if effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert') == True:
                    depth_map_mask = 1.0 - depth_map_mask

            # ルミノシティマスクを作成
            depth_map_mask = self._apply_extened_params(depth_map_mask)

            self.depth_map_mask_cache = depth_map_mask

        if depth_map_mask is None:
            depth_map_mask = self.depth_map_mask_cache

        return depth_map_mask if depth_map_mask is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    def draw_depth_map(self, image_size):
        from cores.mask2 import inference_runtime as mask2_inference_runtime

        return mask2_inference_runtime.predict_depth_map(self.editor.get_original_image_rgb())

class FaceMask(BaseMask):

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Face"
        self.initializing = True  # 初期配置中かどうか
        self.center = (0, 0)

        self.faces_mask_cache = None
        self.faces_mask_cache_hash = None

        with self.canvas:
            KVPushMatrix()
            self.translate = KVTranslate(*self.center)
            KVPopMatrix()

        #self.update_mask()

    def follows_mask_geometry(self):
        return False

    def _edge_refine_support_softness(self):
        return 1.0

    def on_touch_down(self, touch):
        if self.initializing:
            if not self._begin_initial_touch_if_in_placement_area(touch):
                return False
            cx, cy = self.window_to_tcg_for_interaction(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            return True
        else: 
            return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.initializing:
            if not self._initial_touch_can_finish():
                return False
            self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            return True
        else:
            return super().on_touch_up(touch)

    def create_control_points(self):
        self.control_points = []

        # 中心のコントロールポイント
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        if not self.active:
            self.show_center_control_point_only()

    def serialize(self):
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)

        dict = {
            'type': MaskType.FACE,
            'name': self.name,
            'center': [cx, cy],
            'effects_param': param
        }
        # マスクデータはストア参照のみ保存(実体は AIImageCache 共有ストア側にある)
        self._serialize_image_mask_cache(dict)

        return dict

    def deserialize(self, dict):
        self.initializing = False
        cx, cy = dict['center']
        self.name = dict['name']
        self.effects_param.update(dict['effects_param'])
        self.center = params.denorm_param(self.effects_param, (cx, cy))
        # マスクデータ展開(ストア参照。旧形式は移行)
        self._deserialize_image_mask_cache(dict)

        # 描き直し
        self.create_control_points()

    def update_control_points(self):
        cp_center = self.control_points[0]
        cp_center.center = self.center

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            logging.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            cx, cy = self.tcg_to_window_for_overlay(*self.center)
            self.translate.x, self.translate.y = cx, cy

        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        original_image_size = tuple(self.editor.get_image_size())
        center = self.editor.tcg_to_original_image(*self.center)
        exclude_names = []
        if effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_face') == True:
            if effects.Mask2Effect.get_param(self.effects_param, 'mask2_face_face') == False:
                exclude_names.append('face')
            if effects.Mask2Effect.get_param(self.effects_param, 'mask2_face_brows') == False:
                exclude_names.extend(['rb', 'lb'])
            if effects.Mask2Effect.get_param(self.effects_param, 'mask2_face_eyes') == False:
                exclude_names.extend(['re', 'le'])
            if effects.Mask2Effect.get_param(self.effects_param, 'mask2_face_nose') == False:
                exclude_names.append('nose')
            if effects.Mask2Effect.get_param(self.effects_param, 'mask2_face_mouth') == False:
                exclude_names.append('imouth')
            if effects.Mask2Effect.get_param(self.effects_param, 'mask2_face_lips') == False:
                exclude_names.extend(['ulip', 'llip'])
        faces_mask = None

        cache_key = cache_keys.face_cache_key(original_image_size, exclude_names)
        if (self.image_mask_cache is None or self.image_mask_cache_key != cache_key) and self.initializing == False:
            # 描画
            processing_dialog.set_processing_text("Make FaceMask...")
            faces_mask = self._get_or_compute_image_mask_cache(
                cache_key,
                lambda: processing_dialog.wait_processing(self.draw_face, original_image_size, exclude_names),
                "FaceMask",
            )
            #faces_mask = self.draw_face(original_image_size, exclude_names)

        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size))
        if self.image_mask_cache is not None and (self.image_mask_cache is faces_mask or self.faces_mask_cache is None or self.faces_mask_cache_hash != newhash) and self.initializing == False:
            self.faces_mask_cache_hash = newhash

            faces_mask = self.image_mask_cache

            # パラメータに従って画像を変形
            disp_info, rotate_rad, flip, matrix = self.editor.get_hash_items()
            faces_mask = core.rotation(faces_mask, np.rad2deg(rotate_rad), flip, np.array(matrix).reshape(3, 3))
            #faces_mask = core.crop_image_with_disp_info(faces_mask, disp_info)

            faces_mask = self._fit_image_mask_to_texture(faces_mask)

            # ルミノシティマスクを作成
            faces_mask = self._apply_extened_params(faces_mask)

            self.faces_mask_cache = faces_mask

        if faces_mask is None:
            faces_mask = self.faces_mask_cache

        return faces_mask if faces_mask is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    def draw_face(self, image_size, exclude_names):
        from cores.mask2 import inference_runtime as mask2_inference_runtime

        return mask2_inference_runtime.predict_face_mask(
            self.editor.get_original_image_rgb(), exclude_names
        )

    @staticmethod
    def delete_faces():
        from cores.mask2 import inference_runtime as mask2_inference_runtime

        mask2_inference_runtime.delete_faces()

# セグメントマスクのクラス
class TargetTextMask(BaseMask):

    def __init__(self, editor, **kwargs):
        super().__init__(editor, **kwargs)
        self.name = "Target Text"
        self.initializing = True  # 初期配置中かどうか
        self.center = (0, 0)

        self.segment_mask_cache = None
        self.segment_mask_cache_hash = None
        
        self.target_text = ""

        with self.canvas:
            KVPushMatrix()
            self.translate = KVTranslate(*self.center)
            KVPopMatrix()

    def follows_mask_geometry(self):
        return False

    def _edge_refine_support_softness(self):
        return 1.0

    def on_touch_down(self, touch):
        if self.initializing:
            if not self._begin_initial_touch_if_in_placement_area(touch):
                return False
            cx, cy = self.window_to_tcg_for_interaction(*touch.pos)
            self.center_x = cx
            self.center_y = cy
            return True
        else:
            return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.initializing:
            if not self._initial_touch_can_finish():
                return False
            #self.initializing = False
            self.create_control_points()
            self.editor.set_active_mask(self)
            
            # text input dialog（macOS ネイティブ。見た目 + 日本語 IME 対応）
            self._prompt_target_text(self.on_text_entered)

            return True
        else:
            return super().on_touch_up(touch)

    def create_control_points(self):
        # 中心のコントロールポイント（始点）
        cp_center = ControlPoint(self.editor)
        cp_center.center = (self.center_x, self.center_y)
        cp_center.ctrl_center = cp_center.center
        cp_center.is_center = True
        cp_center.color = [0, 1, 0] if self.active else [1, 0, 0]
        cp_center.bind(ctrl_center=self.on_center_control_point_move)
        self.control_points.append(cp_center)
        self.add_widget(cp_center)

        if not self.active:
            self.show_center_control_point_only()

    def _prompt_target_text(self, callback):
        """
        macOS ネイティブのテキスト入力ダイアログ（NSAlert + NSTextField）を出す。
        FileChooser(NSOpenPanel) と同じく **メインスレッドで modal 実行**する必要がある
        （別スレッド/別プロセスだと前面のウィンドウにフォーカスを奪われ、カーソル非表示・
        IME 不可・初回フリーズの原因になる）。on_touch_up はメインスレッドなのでそのまま呼ぶ。

        キャンセルボタンは無く OK のみ。空入力は旧挙動どおり "All" にフォールバックする。
        後続処理（マスク生成）は modal 復帰直後のスタックから切り離すため Clock 経由で呼ぶ。
        """
        try:
            text = device.prompt_native(
                message="Target text (English only)",
                title="Enter Target Text",
                default=self.target_text or "",
                ascii_only=True,  # SAM3 側が日本語非対応のため非 ASCII を抑止
            )
        except Exception as e:
            logging.warning(f"target text prompt failed: {e}")
            text = None
        if not text or text.isspace():
            text = "All"
        KVClock.schedule_once(lambda dt: callback(text), 0)

    def on_text_entered(self, text):
        self.target_text = text
        self.initializing = False
        
        self.update_mask()
        
        self.editor._create_end_new_mask()

    def serialize(self):
        cx, cy = params.norm_param(self.effects_param, (self.center_x, self.center_y))

        param = effects.delete_default_param_all(self.effects, self.effects_param)
        param = params.delete_special_param(param)
        
        dict = {
            'type': MaskType.TARGET_TEXT,
            'name': self.name,
            'center': [cx, cy],
            'target_text': self.target_text,
            'effects_param': param
        }
        # マスクデータはストア参照のみ保存(実体は AIImageCache 共有ストア側にある)
        self._serialize_image_mask_cache(dict)

        return dict

    def deserialize(self, dict):
        self.initializing = False
        cx, cy = dict['center']
        self.name = dict['name']
        self.target_text = dict.get('target_text', "All")
        self.effects_param.update(dict['effects_param'])
        self.center = params.denorm_param(self.effects_param, (cx, cy))
        # マスクデータ展開(ストア参照。旧形式は移行)
        self._deserialize_image_mask_cache(dict)

        # 描き直し
        self.create_control_points()
        #self.update_mask()

    def update_control_points(self):
        cp_center = self.control_points[0]
        cp_center.center = self.center

    def update_mask(self):
        if not self.editor or self.editor.get_image_size()[0] == 0 or self.editor.get_image_size()[1] == 0:
            # image_sizeが正しく設定されていない場合、マスクの更新をスキップ
            logging.warning(f"{self.__class__.__name__}: image_sizeが未設定。マスクの更新をスキップします。")
            return

        with self.canvas:
            cx, cy = self.tcg_to_window_for_overlay(*self.center)
            self.translate.x, self.translate.y = cx, cy
        
        if self.is_draw_mask == True:
            if self.do_draw_composit_mask == True:
                composit_mask = self.editor.find_composit_mask(self)
                if composit_mask is not None:
                    composit_mask.draw_mask_to_fbo(True)
            else:
                self.draw_mask_to_fbo()

    def get_mask_image(self):

        # パラメータ設定
        image_size = (int(self.editor.texture_size[0]), int(self.editor.texture_size[1]))
        original_image_size = tuple(self.editor.get_image_size())
        center = self.editor.tcg_to_original_image(*self.center)
        if effects.Mask2Effect.get_param(self.effects_param, 'switch_mask2_settings') == True:
            invert = effects.Mask2Effect.get_param(self.effects_param, 'mask2_invert')
        else:
            invert = False
        text = self.target_text
        segment_mask = None

        # _draw_segmentを呼び出さなければならない用
        cache_key = cache_keys.target_text_cache_key(original_image_size, text, False)
        if (self.image_mask_cache is None or self.image_mask_cache_key != cache_key) and self.initializing == False:
            # predict_sam3 に渡す box = [x, y, w, h]
            processing_dialog.set_processing_text("Make TargetTextMask SAM3...")
            segment_mask = self._get_or_compute_image_mask_cache(
                cache_key,
                lambda: processing_dialog.wait_processing(self._draw_segment, original_image_size, text, False),
                "TargetTextMask SAM3",
            )
            #segment_mask = self._draw_segment(original_image_size, text)

        # その他更新用
        newhash = hash((self.get_hash_items(), self.editor.get_hash_items(), image_size))
        if self.image_mask_cache is not None and (self.image_mask_cache is segment_mask or self.segment_mask_cache is None or self.segment_mask_cache_hash != newhash) and self.initializing == False:
            self.segment_mask_cache_hash = newhash

            # SegmentMask用のキャッシュ
            segment_mask = self.image_mask_cache
            if invert:
                segment_mask = 1.0 - segment_mask

            # パラメータに従って画像を変形
            disp_info, rotate_rad, flip, matrix = self.editor.get_hash_items()
            segment_mask = core.rotation(segment_mask, np.rad2deg(rotate_rad), flip, np.array(matrix).reshape(3, 3))
            #segment_mask = core.crop_image_with_disp_info(segment_mask, disp_info)

            segment_mask = self._fit_image_mask_to_texture(segment_mask)

            # ルミノシティマスクを作成
            segment_mask = self._apply_extened_params(segment_mask)

            self.segment_mask_cache = segment_mask

        if segment_mask is None:
            segment_mask = self.segment_mask_cache

        return segment_mask if segment_mask is not None else np.zeros((image_size[1], image_size[0]), dtype=np.float32)

    def _draw_segment(self, image_size, text, invert):
        from cores.mask2 import inference_runtime as mask2_inference_runtime

        img = self.editor.get_original_image_rgb()
        return mask2_inference_runtime.predict_sam3_text(img, text, invert)


# メインのエディタークラス
class MaskEditor2(KVFloatLayout, LayerCtrl):
    mask_list = KVListProperty([])
    active_mask = KVObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_event_type('on_structure_change')

        self.mask_container = KVWidget()
        self.add_widget(self.mask_container)
        self.rectangle = None
        # draw_mask_image() 用キャッシュ: サイズが変わらない限りバッファ/テクスチャを
        # 使い回し、ドラッグ中の連続オーバーレイ更新での再アロケーションを避ける。
        self._overlay_la_img = None
        self._overlay_texture = None
        self._overlay_texture_size = None

        self.created_mask = None
        # 作成中マスクはまだ CompositMask.mask_list に未登録なので、
        # mask_list 上の挿入位置から一時的に親 Composit を解決する。
        self.created_mask_index = 0
        self._last_active_mask_id = None
        self._suppress_mask_overlay_draw = False
        self._skip_next_mask_overlay_refresh = False
        self._mask_overlay_enabled = False
        self._overlay_control_points_hidden = False
        self.texture_size = (0, 0)

        self.crop_image_rgb = None
        self.crop_image_hls = None
        self.original_image_rgb = None
        self.original_image_hls = None
        self.ai_image_cache = AIImageCache()
        # AI マスクビットマップ共有ストアの mark-and-sweep 抑止カウンタ。
        # deserialize 中は mask_list が段階的に構築されるため、途中の on_structure_change で
        # sweep すると復元直後・生成前のビットマップを誤って掃除してしまう。>0 の間は sweep を
        # 保留し、resume 時に一度だけ実行する。
        self._bitmap_sweep_suspend_depth = 0
        self._bitmap_sweep_pending = False

        # mask Geometry: image Geom のみの matrix を退避し、active Composit の
        # mask Geom matrix を左乗算したものを tcg_info['matrix'] に書き込む。
        self._image_only_matrix = None
        self._matrix_lock = threads.mask_editor_matrix_lock

        # mask Geometry 軸表示用 (X / Y 軸ともグレーで矢印付き)。
        # Composit active + switch_mask_geometry ON 時に表示。
        # 軸の原点は post-translation 点 (= image_center + translation)。flip 軸は軸線そのもの。
        with self.canvas.after:
            self._axes_scissor = self.push_scissor()
            # 軸線 (本体 + 矢印を 1 つの polyline で描画。先端から左右の羽が伸びる)
            self._axes_color_x = KVColor(0.7, 0.7, 0.7, 0.9)
            self._axis_x_line = KVLine(points=(0, 0, 0, 0), width=max(0.5, 0.75 * device.dpi_scale()))
            self._axes_color_y = KVColor(0.7, 0.7, 0.7, 0.9)
            self._axis_y_line = KVLine(points=(0, 0, 0, 0), width=max(0.5, 0.75 * device.dpi_scale()))
            # 互換用の instruction。中心点表示は使わないので常に size=(0, 0) のまま。
            self._pivot_color = KVColor(1, 1, 0, 0.85)
            self._pivot_marker = KVEllipse(pos=(0, 0), size=(0, 0))
            self.pop_scissor()

        logging.info("MaskEditor: 初期化完了")

    def on_structure_change(self, *args):
        # マスク追加/削除・コンポジット変更・deserialize 完了はすべてここを通るため、
        # AI マスクビットマップ共有ストアの mark-and-sweep 自動削除もここでまとめて行う。
        # ただし deserialize 中(suspend 中)は mask_list が未完成なので保留し、resume 時に走らせる。
        if self._bitmap_sweep_suspend_depth > 0:
            self._bitmap_sweep_pending = True
            return
        self._sweep_mask_bitmaps()

    def suspend_bitmap_sweep(self):
        """AI マスクビットマップ共有ストアの mark-and-sweep を一時停止する(再入可能)。
        deserialize のようにマスクを段階構築する処理を囲い、途中の on_structure_change による
        sweep で復元途中のビットマップが消えるのを防ぐ。resume と対で使う。"""
        self._bitmap_sweep_suspend_depth += 1

    def resume_bitmap_sweep(self):
        """suspend_bitmap_sweep を解除する。最外で解除された時点で保留中の sweep を一度だけ実行。"""
        if self._bitmap_sweep_suspend_depth > 0:
            self._bitmap_sweep_suspend_depth -= 1
        if self._bitmap_sweep_suspend_depth == 0 and self._bitmap_sweep_pending:
            self._bitmap_sweep_pending = False
            self._sweep_mask_bitmaps()

    # 終了処理
    def end(self):
        if self.active_mask is not None:
            self.active_mask.end()

    def push_scissor(self):
        scissor = KVScissorPush()
        self.set_scissor(scissor)
        return scissor

    def set_scissor(self, scissor):
        scissor.x = int(self.pos[0])
        scissor.y = int(self.pos[1])
        scissor.width = int(self.size[0])
        scissor.height = int(self.size[1])

    def pop_scissor(self):
        KVScissorPop()
    
    def set_ref_image(self, crop_image, original_image=None):
        if self.crop_image_rgb is not crop_image:
            self.crop_image_rgb = crop_image
            self.crop_image_hls = None

        if self.original_image_rgb is not original_image:
            self.original_image_rgb = original_image
            self.original_image_hls = None

    def get_crop_image_hls(self):
        if self.crop_image_hls is None and self.crop_image_rgb is not None:
            self.crop_image_hls = hls_mask.rgb_to_selection_hls(self.crop_image_rgb)
            # Keep the RGB crop alive. Edge-refine and its debug views must use
            # the current zoom crop, even after HLS-based masks have run.
        return self.crop_image_hls

    def get_original_image_rgb(self):
        return self.original_image_rgb

    def set_ai_image_cache(self, cache):
        self.ai_image_cache = cache if cache is not None else AIImageCache()

    def serialize_ai_image_cache(self):
        return self.ai_image_cache.serialize() if self.ai_image_cache is not None else None

    def set_serialized_ai_image_cache(self, serialized):
        if self.ai_image_cache is None:
            self.ai_image_cache = AIImageCache()
        self.ai_image_cache.deserialize(serialized)

    def get_ai_depth_map(self, cache_key, compute_func):
        if self.ai_image_cache is None:
            return compute_func()
        return self.ai_image_cache.get_depth_map(cache_key, compute_func)

    def peek_ai_depth_map(self, cache_key):
        # 既に作成済みの depth のみ返す(未作成なら None)。新規推論はしない。
        if self.ai_image_cache is None:
            return None
        return self.ai_image_cache.peek_depth_map(cache_key)

    def get_ai_mask_bitmap(self, cache_key):
        # AI マスク(Segment/Face/TargetText)共有ストアの参照。get_ai_depth_map と対称。
        if self.ai_image_cache is None:
            return None
        return self.ai_image_cache.get_mask_bitmap(cache_key)

    def put_ai_mask_bitmap(self, cache_key, image):
        if self.ai_image_cache is None:
            return
        self.ai_image_cache.put_mask_bitmap(cache_key, image)

    def sweep_ai_mask_bitmaps(self, live_keys):
        if self.ai_image_cache is None:
            return {"mask_bitmap_entries": 0, "mask_bitmap_bytes": 0}
        return self.ai_image_cache.sweep_mask_bitmaps(live_keys)

    def get_original_image_hls(self):
        if self.original_image_hls is None and self.original_image_rgb is not None:
            self.original_image_hls = hls_mask.rgb_to_selection_hls(self.original_image_rgb)
        return self.original_image_hls

    def set_texture_size(self, tx, ty):
        with self._matrix_lock:
            self.texture_size = (tx, ty)
        _mask_zoom_sync_debug("mask_editor.set_texture_size texture_size=%s", self.texture_size)

    def set_primary_param(self, primary_param, disp_info, redraw_mask=True):

        # クロップ編集/full-preview 中の回転コンテンツ四辺形（apply_zero_wrap と共用）。
        # マスクオーバーレイを回転後の有効画像領域にクリップするために保持する。
        # 通常表示では GeometryEffect が None にクリアするので矩形クリップに落ちる。
        self._zero_wrap_content_quad = primary_param.get('_zero_wrap_content_quad')

        # TCG情報を設定
        with self._matrix_lock:
            old_view_key = self._mask_overlay_view_key_locked()
            self.tcg_info = params.param_to_tcg_info(primary_param)
            params.set_disp_info(self.tcg_info, disp_info) # これだけ引数の値を設定

            self.__set_image_info()
            #self.update()

            # mask Geometry: 画像 Geom のみの matrix を退避し、active Composit の
            # mask Geom matrix を合成して tcg_info['matrix'] に反映する。
            self._image_only_matrix = np.array(self.tcg_info['matrix'], dtype=np.float64).copy()
            logged_disp = params.get_disp_info(self.tcg_info)
            logged_orig = self.tcg_info.get('original_img_size')
            new_view_key = self._mask_overlay_view_key_locked()
        # redraw_mask=False は通常フレームでの重い overlay 再描画を抑えるためだが、
        # zoom / scroll / texture resize で viewport が変わった場合は overlay texture
        # 自体を作り直さないと古い crop のマスクが残る。
        redraw_overlay = redraw_mask or (old_view_key is not None and old_view_key != new_view_key)
        self._set_active_composit_matrix(redraw_mask=redraw_overlay)
        _mask_zoom_sync_debug(
            "mask_editor.set_primary_param texture_size=%s disp=%s input_disp=%s orig=%s active=%s redraw=%s view_changed=%s",
            self.texture_size, logged_disp, disp_info, logged_orig,
            _mask_geom_id(self.get_active_mask()), redraw_overlay, old_view_key != new_view_key,
        )

    def _mask_overlay_view_key_locked(self):
        tcg_info = getattr(self, 'tcg_info', None)
        if not isinstance(tcg_info, dict):
            return None
        matrix = tcg_info.get('matrix')
        matrix_key = None
        if matrix is not None:
            matrix_key = tuple(np.asarray(matrix, dtype=np.float64).flatten())
        return (
            tuple(self.texture_size),
            params.get_disp_info(tcg_info),
            tuple(tcg_info.get('original_img_size', ())),
            tcg_info.get('rotation'),
            tcg_info.get('rotation2'),
            tcg_info.get('flip_mode'),
            matrix_key,
        )

    def _call_with_image_only_matrix(self, func, *args, **kwargs):
        if self._image_only_matrix is None or 'matrix' not in self.tcg_info:
            return func(*args, **kwargs)
        with self._matrix_lock:
            saved_matrix = self.tcg_info['matrix']
            self.tcg_info['matrix'] = self._image_only_matrix
            try:
                return func(*args, **kwargs)
            finally:
                self.tcg_info['matrix'] = saved_matrix

    def refresh_active_mask_overlay(self):
        if self._skip_next_mask_overlay_refresh:
            self._skip_next_mask_overlay_refresh = False
            _mask_geom_debug("refresh_active_mask_overlay skipped once")
            return
        if self._overlay_control_points_hidden or not self._mask_overlay_enabled:
            return
        mask = self.overlay_mask_for_active()
        self._draw_overlay_mask(mask)

    def skip_next_mask_overlay_refresh(self, clear=True):
        self._skip_next_mask_overlay_refresh = True
        if clear:
            self.draw_mask_image(None)

    def _get_mask_image_rect(self, texture_size=None):
        if texture_size is None:
            texture_size = tuple(self.texture_size)
        scale = device.dpi_scale()
        px, py = self.to_window(*self.pos)
        marginx = (self.size[0] - texture_size[0] * scale) / 2
        marginy = (self.size[1] - texture_size[1] * scale) / 2
        return (px + marginx, py + marginy), (texture_size[0] * scale, texture_size[1] * scale)

    def _clip_mask_overlay_to_image_area(self, glayimg, disp_info):
        # クロップ編集/full-preview 中は disp_info が正方形全体になり矩形クリップが効かない。
        # 回転コンテンツ四辺形があれば、その内側だけにオーバーレイを残す（四隅のはみ出し除去）。
        # quad は full-preview 中しか格納されない（通常表示でそのまま使うと表示テクスチャが
        # crop 窓のため菱形に誤クリップされる）。通常表示は下の矩形クリップに落ちる。
        quad = getattr(self, '_zero_wrap_content_quad', None)
        if quad is not None:
            h, w = glayimg.shape[:2]
            mask = core.content_quad_mask(h, w, quad)
            return glayimg * mask if glayimg.ndim == 2 else glayimg * mask[..., np.newaxis]

        if disp_info is None:
            return glayimg
        h, w = glayimg.shape[:2]
        new_w, new_h, offset_x, offset_y = core.crop_size_and_offset_from_texture(w, h, disp_info)
        if new_w >= w and new_h >= h:
            return glayimg

        clipped = np.zeros_like(glayimg)
        clipped[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = glayimg[offset_y:offset_y + new_h, offset_x:offset_x + new_w]
        return clipped

    def window_point_in_image_rect(self, x, y):
        if not self.collide_point(x, y):
            return False
        try:
            (rx, ry), (rw, rh) = self._get_mask_image_rect(tuple(self.texture_size))
            if rw <= 0 or rh <= 0:
                return False
            return rx <= float(x) <= rx + rw and ry <= float(y) <= ry + rh
        except Exception:
            return False

    def reposition_mask_image(self):
        if self.rectangle is not None:
            with self._matrix_lock:
                texture_size = tuple(self.texture_size)
                rect = self._get_mask_image_rect(texture_size)
            self.rectangle.pos, self.rectangle.size = rect

    def get_hash_items(self):
        with self._matrix_lock:
            matrix = np.array(self.tcg_info['matrix'], dtype=np.float64, copy=True)
            return (params.get_disp_info(self.tcg_info), self.tcg_info['rotation'] + self.tcg_info['rotation2'], self.tcg_info['flip_mode'], tuple(matrix.flatten()))

    def get_effect_view_param(self):
        with self._matrix_lock:
            tcg_info = getattr(self, 'tcg_info', None)
            if not isinstance(tcg_info, dict):
                return None
            matrix = self._image_only_matrix if self._image_only_matrix is not None else tcg_info.get('matrix')
            view_param = {
                'original_img_size': tuple(tcg_info.get('original_img_size', self.texture_size)),
                'disp_info': tcg_info.get('disp_info'),
                'rotation': math.degrees(float(tcg_info.get('rotation', 0.0))),
                'rotation2': math.degrees(float(tcg_info.get('rotation2', 0.0))),
                'flip_mode': tcg_info.get('flip_mode', 0),
                'matrix': np.array(matrix, dtype=np.float64, copy=True) if matrix is not None else np.eye(3),
            }
            view_param['img_size'] = view_param['original_img_size']
            return view_param

    def __set_image_info(self):
        for mask in reversed(self.mask_list):
            #pass    # 無限ループ対策
            effects.reeffect_all(mask.effects)
        
    def update(self):
        KVClock.schedule_once(self._update, 0)

    def _update(self, dt=0):
        # 既存のマスクに対する更新を処理
        for mask in reversed(self.mask_list):
            mask.update()

    def serialize(self):
        list = []
        for mask in reversed(self.mask_list):
            parent = self.find_composit_mask(mask)
            if parent is not None and parent != mask:
                continue
            list.append(mask.serialize())
        if len(list) <= 0:
            return None

        dict = {
            'mask2': list,
        }
        return dict

    def deserialize(self, dict):
        list = dict['mask2']

        for dict in list:
            type = dict.get('type', None)
            mask = self._create_mask(type)
            mask.deserialize(dict)
            #mask.update()

        self.dispatch('on_structure_change')

    def is_center_click_anyone(self, touch, self_mask):
        for mask in reversed(self.mask_list):
            if mask != self_mask and mask.is_center_click(touch):
                return True
        return False

    def get_created_mask(self):
        return self.created_mask

    def get_active_mask(self):
        if self.disabled == True:
            return None

        return self.active_mask

    def commit_in_progress(self):
        """アクティブマスクが描画中の操作 (Polyline 描画など) を持っていれば確定させる。"""
        mask = self.active_mask
        if mask is None:
            return
        committer = getattr(mask, 'commit_in_progress', None)
        if callable(committer):
            try:
                committer()
            except Exception:
                logging.exception("MaskEditor: commit_in_progress 中に例外")
    
    def find_mask(self, mask_id):
        for mask in reversed(self.mask_list):
            if mask.mask_id == mask_id:
                return mask
        return None
        
    # LayerCtrl用
    def update_layer(self, op, index, op_type, dict):
        match op:
            case "Create":
                mask = self._create_mask(dict['type'], index, dict)
                # 通常マスクなら親にくっつける
                if op_type != "Composit":
                    # なんでもいいから親探す
                    composit_mask = self.find_composit_mask(mask, index)
                    if composit_mask is None:
                        # python -O では assert が消えて add_mask(None, ...) が実行され
                        # AttributeError で落ちる。親が見つからない場合でも mask 自体は
                        # 既に _create_mask() で mask_container / mask_list に登録済みなので、
                        # コンポジットへの所属付けだけ諦めてトップレベルのマスクとして残す。
                        logging.error("update_layer: Create - Composit mask not found for index=%s; leaving mask unattached at top level", index)
                    else:
                        # インデクスがコンポジットマスクの中の何番目かを調べる
                        composit_mask_index = 0
                        for i in range(index-1, -1, -1):
                            composit_mask = self.mask_list[i]
                            if composit_mask.is_composit():
                                composit_mask_index = index - 1 - i
                                break
                        composit_mask.add_mask(mask, op_type, composit_mask_index)
                self.set_active_mask(mask)
                self.request_mask_render_update(
                    mask,
                    reason="history.layer_create",
                    structure_changed=True,
                    redraw_overlay=True,
                    redraw_pipeline=True,
                )

            case "Delete":
                self._remove_mask(self.get_mask(index))

            case "Update":
                mask = self.get_mask(index)
                mask.clear()
                mask.deserialize(dict)
                self.set_active_mask(mask)
                self.request_mask_render_update(
                    mask,
                    reason="history.layer_update",
                    structure_changed=True,
                    redraw_overlay=True,
                    redraw_pipeline=True,
                )

            case _:
                # python -O では assert が消えて何もせず抜けてしまう。ここは元々
                # undo/redo リプレイ経由で来る想定外の op 文字列を検出するためだけの
                # ガードなので、ログを残して何もせずに戻る(呼び出し側は戻り値を見ない)。
                logging.error("Invalid operation: " + op)

    # LayerCtrl用
    def get_layer(self, index):
        return self.get_mask(index)
    
    def get_mask(self, index):
        return self.mask_list[index]
    
    def get_mask_list(self):
        return self.mask_list

    # mask2_content用
    def add_mask(self, mask_type, op_type, index):
        if self._mask_mesh_editor_locks_input():
            return None
        return self._create_start_new_mask(mask_type, op_type, index)

    # mask2_content用
    def add_composit_mask(self, instance):
        if self._mask_mesh_editor_locks_input():
            return
        self._create_start_new_mask(MaskType.COMPOSIT, "Composit")

    # mask2_content用
    def del_mask(self, mask):
        if self._mask_mesh_editor_locks_input():
            return
        index = self.get_mask_list().index(mask)
        is_composit = mask.is_composit()
        if is_composit:
            maskop = 'Composit'
        else:
            # find_composit_mask/find_mask_op が失敗しても None を履歴に記録しない
            # (undo 時に add_mask(mask, None) となり描画が壊れるため 'Add' に倒す)
            maskop = 'Add'
            composit_mask = self.find_composit_mask(mask, index)
            found_op = composit_mask.find_mask_op(mask) if composit_mask else None
            if found_op is not None:
                maskop = found_op
            else:
                logging.error(f"del_mask: maskop not found for mask index={index}; recording 'Add'")

        get_history_ctrl().begin_history_layer_ctrl(self, "Create", index, maskop)
        get_history_ctrl().end_history_layer_ctrl(self, "Delete", index)
        self._remove_mask(mask)

    def set_draw_mask(self, is_draw_mask, refresh=True):
        self._mask_overlay_enabled = bool(is_draw_mask)
        if is_draw_mask == False:
            if self.rectangle is not None:
                try:
                    self.mask_container.canvas.before.remove(self.rectangle)
                except ValueError:
                    # 既にcanvasから除去済みのレース（無害）
                    pass
                self.rectangle = None
            self.draw_mask_image(None)
        mask = self.get_active_mask()
        if mask is not None:
            mask.is_draw_mask = is_draw_mask
            if is_draw_mask == True and refresh:
                self.request_mask_render_update(
                    mask,
                    reason="set_draw_mask",
                    structure_changed=False,
                    redraw_overlay=True,
                    redraw_pipeline=False,
                )
    
    def start_draw_image(self, fast_display=True, skip_histogram=None):
        if self.root is not None:
            if skip_histogram is None:
                skip_histogram = fast_display
            self.root.start_draw_image(fast_display=fast_display, skip_histogram=skip_histogram)

    def _composit_for_render_update(self, mask):
        if mask is None:
            return None
        try:
            return self._mask_parent_for_visibility(mask)
        except Exception:
            # 親Composit解決に失敗するとキャッシュ無効化対象が漏れ得るため記録する
            logging.exception("request_mask_render_update: composit resolution failed")
            return None

    def _invalidate_mask_render_family(self, mask):
        invalidated = []
        if mask is not None:
            try:
                mask.invalidate_render_cache()
                invalidated.append(_mask_geom_id(mask))
            except Exception:
                logging.exception("request_mask_render_update: mask invalidate failed")
        composit = self._composit_for_render_update(mask)
        if composit is not None and composit is not mask:
            try:
                composit.invalidate_render_cache()
                invalidated.append(_mask_geom_id(composit))
            except Exception:
                logging.exception("request_mask_render_update: composit invalidate failed")
        return invalidated

    def _refresh_child_mask_cache_for_overlay(self, mask, overlay_mask):
        if mask is None or overlay_mask is None or mask is overlay_mask:
            return
        try:
            if not overlay_mask.is_composit():
                return
            if self._mask_parent_for_visibility(mask) is not overlay_mask:
                return
            if mask.follows_mask_geometry():
                mask.get_mask_image()
            else:
                self._call_with_image_only_matrix(mask.get_mask_image)
        except Exception:
            logging.exception("request_mask_render_update: child mask cache refresh failed")

    def _draw_overlay_mask(self, overlay_mask):
        if overlay_mask is None:
            self.draw_mask_image(None)
            return
        try:
            overlay_mask.draw_mask_to_fbo(True)
        except Exception:
            logging.exception("mask overlay draw failed")

    def request_mask_render_update(self, mask=None, reason="", structure_changed=False,
                                   refresh_visibility=True, redraw_overlay=True,
                                   redraw_pipeline=True, clear_overlay=False):
        """マスク描画更新の単一入口。

        構造変更・パラメータ変更・作成完了・削除はすべてここを通す。
        ここで cache invalidate、表示状態更新、overlay redraw、pipeline redraw、
        structure event を同じ順序で処理する。
        """
        if mask is None:
            mask = self.get_active_mask()
        invalidated = self._invalidate_mask_render_family(mask)

        if refresh_visibility:
            try:
                self.refresh_mask_visibility()
            except Exception:
                logging.exception("request_mask_render_update: refresh visibility failed")

        if clear_overlay:
            self.draw_mask_image(None)
        elif redraw_overlay and self._mask_overlay_enabled:
            overlay_mask = self.overlay_mask_for_active() or mask
            if overlay_mask is not None:
                try:
                    self._refresh_child_mask_cache_for_overlay(mask, overlay_mask)
                    self._draw_overlay_mask(overlay_mask)
                except Exception:
                    logging.exception("request_mask_render_update: overlay update failed")
            else:
                self.draw_mask_image(None)
        elif redraw_overlay:
            self.draw_mask_image(None)

        if structure_changed:
            self.dispatch('on_structure_change')

        if redraw_pipeline:
            self.start_draw_image(fast_display=False)

        _mask_zoom_sync_debug(
            "request_mask_render_update reason=%s mask=%s active=%s structure=%s overlay=%s pipeline=%s invalidated=%s",
            reason, _mask_geom_id(mask), _mask_geom_id(self.get_active_mask()),
            structure_changed, redraw_overlay, redraw_pipeline, invalidated,
        )

    def draw_mask_image(self, glayimg):
        if self._overlay_control_points_hidden and glayimg is not None:
            return

        if glayimg is None:
            # オーバーレイを消す。バッファ/テクスチャは次回サイズが一致すれば
            # 再利用できるようキャッシュしたまま、Canvas からは Rectangle だけ外す。
            if self.rectangle is not None:
                try:
                    self.mask_container.canvas.before.remove(self.rectangle)
                except ValueError:
                    # canvas からは既に消えているが参照だけ残っていた (state 不整合)
                    pass
                self.rectangle = None
            return

        with self._matrix_lock:
            texture_size = tuple(self.texture_size)
            disp_info = params.get_disp_info(self.tcg_info) if getattr(self, "tcg_info", None) is not None else None
            rect = self._get_mask_image_rect(texture_size)
        _mask_zoom_sync_debug(
            "mask_editor.draw_mask_image mask_shape=%s texture_size=%s disp=%s rect=%s",
            getattr(glayimg, "shape", None), texture_size, disp_info, rect,
        )

        # マスクをアルファとして扱い、ルミナンスを白(1.0)にする
        glayimg = np.clip(glayimg, 0, 1)
        glayimg = self._clip_mask_overlay_to_image_area(glayimg, disp_info)
        h, w = glayimg.shape[:2]

        # ドラッグ中は毎フレーム呼ばれるため、サイズが変わらない間は la_img バッファ /
        # テクスチャ / Rectangle を使い回して確保・GPU アップロード・Canvas 差し替えの
        # churn を避ける。サイズが変わった時だけ作り直す。
        if self._overlay_la_img is None or self._overlay_la_img.shape[:2] != (h, w):
            self._overlay_la_img = np.empty((h, w, 2), dtype=np.float32)
        la_img = self._overlay_la_img
        la_img[..., 0] = 1.0  # Luminance = White
        la_img[..., 1] = glayimg  # Alpha = Mask Value

        size_changed = self._overlay_texture is None or self._overlay_texture_size != (w, h)
        if size_changed:
            texture = KVTexture.create(size=(w, h), colorfmt='luminance_alpha', bufferfmt='float')
            texture.blit_buffer(la_img.tobytes(), colorfmt='luminance_alpha', bufferfmt='float')
            # flip_vertical() はテクスチャ座標を永続的に反転させる副作用があるため、
            # テクスチャ生成直後にちょうど一回だけ呼ぶこと。使い回し時に呼ぶと
            # 二重反転して上下逆に表示されてしまう。
            texture.flip_vertical()
            self._overlay_texture = texture
            self._overlay_texture_size = (w, h)
        else:
            # サイズ不変: 既存テクスチャへ中身だけ書き戻す(flip_vertical は呼ばない)。
            texture = self._overlay_texture
            texture.blit_buffer(la_img.tobytes(), colorfmt='luminance_alpha', bufferfmt='float')

        pos, size = rect
        if self.rectangle is not None and not size_changed:
            # 既存 Rectangle をそのまま使い回す。テクスチャオブジェクト自体は
            # 変わっていないので texture の再代入は不要、pos/size のみ更新する。
            self.rectangle.pos = pos
            self.rectangle.size = size
            # blit_buffer によるテクスチャ内容の更新は Rectangle 側のプロパティ変更を
            # 伴わない(pos/size が前回と同値だと Kivy 側で更新がスキップされ得る)ため、
            # Canvas に明示的に再描画を要求する。
            self.mask_container.canvas.before.ask_update()
        else:
            if self.rectangle is not None:
                try:
                    self.mask_container.canvas.before.remove(self.rectangle)
                except ValueError:
                    # canvas からは既に消えているが参照だけ残っていた (state 不整合)
                    pass
                self.rectangle = None
            with self.mask_container.canvas.before:
                KVColor(1, 0, 0, 0.4)
                self.rectangle = KVRectangle(texture=texture, pos=pos, size=size)

        # cv2.imwrite('combined_mask.png', (glayimg*255).astype(np.uint8))

    def _create_start_new_mask(self, type, op_type, index=0):
        # 画像サイズがまだ設定されていない場合、マスクの作成をスキップ

        mask = self._create_mask(type, index)
        self.set_active_mask(None)
        self.created_mask = mask
        self.created_mask_index = index
        self._mask_overlay_enabled = True
        if self.root is not None:
            self.root.update_mask2_options_enabled()

        # ここで履歴の更新を始める
        get_history_ctrl().begin_history_layer_ctrl(self, "Delete", self.get_mask_list().index(self.created_mask), op_type)

        # mask Geometry: 新規作成時のクリック位置が mask Geom 逆変換越しに正しい
        # TCG 値になるよう、created_mask の所属 Composit の mask Geom matrix を反映。
        self._set_active_composit_matrix()

        # CompositMaskなど初期化が不要な場合は即座に終了処理を行う
        if mask.initializing == False:
            self._create_end_new_mask()

        return mask

    def _create_end_new_mask(self):
        mask = self.created_mask
        if mask is None:
            return
        self.set_active_mask(mask)
        self.created_mask = None
        self.created_mask_index = 0
        self.request_mask_render_update(
            mask,
            reason="create_end",
            structure_changed=True,
            redraw_overlay=True,
            redraw_pipeline=True,
        )
        
        # 履歴記録。 create_maskがレイヤーリストにある場合のみ
        if mask in self.get_mask_list():
            get_history_ctrl().end_history_layer_ctrl(self, "Create", self.get_mask_list().index(mask))

    def _mask_mesh_editor_locks_input(self):
        """Mesh 編集モード中はマスク側の入力を全ロック。
        Mesh widget は preview_widget に直接マウントされているので、
        ここで MaskEditor2 のイベント連鎖を抑止すれば衝突しない。"""
        try:
            from kivy.app import App as _App
            app = _App.get_running_app()
            if app is None or app.root is None:
                return False
            check = getattr(app.root, 'is_mask_mesh_editor_active', None)
            return bool(check()) if callable(check) else False
        except Exception:
            return False

    def _liquify_editor_locks_input(self):
        """Preview直下の Liquify editor が active の間は MaskEditor2 側の入力を止める。"""
        try:
            from kivy.app import App as _App
            app = _App.get_running_app()
            if app is None or app.root is None:
                return False
            check = getattr(app.root, 'is_liquify_editor_active', None)
            return bool(check()) if callable(check) else False
        except Exception:
            return False

    def on_touch_down(self, touch):
        if self.disabled == True:
            return False
        if self._mask_mesh_editor_locks_input() or self._liquify_editor_locks_input():
            # 編集専用 widget が preview_widget 直下にいる間は MaskEditor2 と
            # child mask のイベント連鎖を抑止する。
            return False

        # アクティブなマスクを先に処理
        if self.created_mask is not None:
            if self.created_mask.on_touch_down(touch):
                return True
        """
        # 既存のマスクに対するタッチイベントを処理（新しい方から）
        for mask in self.mask_list:
            if mask.on_touch_down(touch):
                return True
        """
        return KVFloatLayout.on_touch_down(self, touch)

    def on_touch_up(self, touch):
        if self.disabled == True:
            return False
        if self._mask_mesh_editor_locks_input() or self._liquify_editor_locks_input():
            return False
        if (self.created_mask is not None
                and getattr(self.created_mask, 'initializing', False)
                and not getattr(self.created_mask, '_initial_touch_started', False)):
            return False

        result = KVFloatLayout.on_touch_up(self, touch)

        # こっちを後でやらないとまだコントロールポイントが作られてない
        if self.created_mask is not None:
            # Polyline のように複数タッチで初期形状を作るマスクは、初回ストロークが
            # コミットされるまで作成確定を遅らせる(早期確定すると後続ストロークの
            # begin("Update") が Create の begin を上書きし、undo が1つずれる)
            if self.created_mask.initializing == False and not self.created_mask.in_creation_stroke():
                self._create_end_new_mask()

        return result

    def _create_mask_object(self, mask_type):
        # マスクオブジェクト作成のみを行う
        match mask_type:
            case MaskType.CIRCULAR:
                mask = CircularGradientMask(editor=self)
            case MaskType.GRADIENT:
                mask = GradientMask(editor=self)
            case MaskType.FULL:
                mask = FullMask(editor=self)
            case MaskType.FREEDRAW:
                mask = FreeDrawMask(editor=self)
            case MaskType.POLYLINE:
                mask = PolylineMask(editor=self)
            case MaskType.SEGMENT:
                mask = SegmentMask(editor=self)
            case MaskType.DEPTHMAP:
                mask = DepthMapMask(editor=self)
            case MaskType.FACE:
                mask = FaceMask(editor=self)
            case MaskType.TARGET_TEXT:
                mask = TargetTextMask(editor=self)
            case MaskType.COMPOSIT:
                mask = CompositMask(editor=self)
            case _:
                # python -O では assert が消え、mask 未代入のまま return mask に到達して
                # UnboundLocalError という分かりにくい例外になる(通常実行でも assert で
                # 落ちるだけで理由が分かりにくい)。未知のタイプ用に代替マスクを捏造すると
                # mask_container / mask_list に不正なオブジェクトが混入するため、ここは
                # graceful degradation ではなく明示的な例外にして早期に原因を特定できるようにする。
                logging.error(f"MaskEditor: 不明なマスクタイプ: {mask_type}")
                raise ValueError(f"MaskEditor: 不明なマスクタイプ: {mask_type}")

        return mask

    def _create_mask(self, mask_type, index=0, dict=None):
        # マスクオブジェクト作成
        mask = self._create_mask_object(mask_type)

        # コンテナに追加
        self.mask_container.add_widget(mask, index)
        self.mask_list.insert(index, mask)

        # デシリアライズ
        if dict is not None:
            mask.deserialize(dict)
        elif self.root is not None:
            # Quick Select is a sticky drawing tool setting. A freshly created
            # DRAW mask should inherit the last user-set edge-refine controls,
            # while deserialized masks keep their saved values.
            sticky = getattr(self.root, '_sticky_mask2_edge_refine', None)
            if sticky and mask._edge_refine_selection_strategy() == edge_refine.STRATEGY_DRAW:
                for k, v in sticky.items():
                    mask.effects_param[k] = v

        # パラメータをウィジェットに反映
        if self.root is not None:
            self.root.set2widget_all(mask.effects, mask.effects_param)

        #self.dispatch('on_structure_change')
        return mask

    def copy_mask_into(self, src_mask, target_composit, maskop='Add'):
        """src_mask の形状のみを target_composit へコピーする(マスク作成モード「コピー」)。
        調整パラメータは MASK2_SHAPE_PARAM_KEYS ホワイトリストのみ引き継ぎ、それ以外は
        新規デフォルトのまま。AI マスク(Segment/Face/TargetText)は image_mask_cache_key
        をコピーするだけでストア経由に自動共有される(ビットマップの複製は行わない)。
        src_mask の状態は変更しない。"""
        if src_mask is None or src_mask.is_composit():
            logging.error("copy_mask_into: src_mask が None か composit です(copy_composit_children_into を使用)")
            return None

        src_dict = src_mask.serialize()
        mask_type = src_dict['type']

        # type / center / inner_radius / outer_radius / rotate_rad / lines / polylines /
        # corner / image_mask_cache_key 等、タイプ固有の形状キーをそのまま引き継ぐ
        # (name と effects_param は作り直す)。
        new_dict = {k: v for k, v in src_dict.items() if k not in ('name', 'effects_param')}
        new_dict['name'] = _next_copy_name(src_mask.name)
        # 調整パラメータはホワイトリストのみ、コピー元の「現在値」から引き継ぐ
        # (serialize() 済みの sparse dict ではなく src_mask.effects_param を直接参照することで、
        # sticky 設定等の残留値に左右されず常に決定的な値になる)。
        new_dict['effects_param'] = {
            key: effects.Mask2Effect.get_param(src_mask.effects_param, key)
            for key in MASK2_SHAPE_PARAM_KEYS
        }

        index = self.get_mask_list().index(target_composit) + 1
        new_mask = self._create_mask(mask_type, index, new_dict)
        target_composit.add_mask(new_mask, maskop, 0)

        self.request_mask_render_update(
            new_mask,
            reason="copy_mask_into",
            structure_changed=True,
            redraw_overlay=True,
            redraw_pipeline=True,
        )
        return new_mask

    def copy_composit_children_into(self, src_composit, target_composit):
        """src_composit の子マスクを全て target_composit へ展開コピーする(各子の maskop は
        コピー元の値を維持)。src_composit が target_composit 自身(またはその子)であっても
        安全: 現在の子リストのスナップショットを取ってからコピーするため、無限追加にならない。"""
        if src_composit is None or not src_composit.is_composit():
            logging.error("copy_composit_children_into: src_composit が composit ではありません")
            return []

        snapshot = list(src_composit.get_mask_list())  # [(mask, maskop), ...] のスナップショット
        # copy_mask_into は単体追加と同じ挿入方式(target_composit の直後 / composit-local
        # 先頭へ prepend)を使うため、逆順に処理することで最終的にソース順を保持する。
        created = []
        for child, maskop in reversed(snapshot):
            new_mask = self.copy_mask_into(child, target_composit, maskop)
            if new_mask is not None:
                created.append(new_mask)
        created.reverse()
        return created

    def _remove_mask(self, mask):
        if mask is None:
            return
        removed_parent = None
        # 削除する前にアクティブなものを移動する
        if len(self.mask_list) <= 1:
            self.draw_mask_image(None)
            self.set_active_mask(None)
        else:
            i = self.mask_list.index(mask)
            i = i+1 if i+1 < len(self.mask_list) else i-1
            self.set_active_mask(self.mask_list[i])

        # 親探す
        composit_mask = self.find_composit_mask(mask)
        if composit_mask is mask:
            # Compositなら子をすべて削除
            for child, _ in list(composit_mask.get_mask_list()):
                self._remove_mask(child)
            composit_mask.clear()
            removed_parent = composit_mask
        elif composit_mask is not None:
            # Compositでないなら親から削除
            composit_mask.remove_mask(mask)
            removed_parent = composit_mask
        else:
            # python -O では assert が消えて何事もなかったかのように後続処理に進む。
            # ここは元々「親が見つからない」ケースを検出するだけのガードなので、
            # ログだけ残してコンテナ/mask_list からの削除は通常どおり続行する
            # (親への所属解除ができないだけで、エディタ上からは消える)。
            logging.error(f"MaskEditor: 親が見つかりませんでした。所属解除はスキップして削除を続行します。")

        # コンテナから削除
        self.mask_container.remove_widget(mask)
        self.mask_list.remove(mask)
        mask.release()  # Window.bind 等マスク固有のリソースを解放

        # 再描画
        self.request_mask_render_update(
            self.active_mask or removed_parent,
            reason="remove_mask",
            structure_changed=True,
            redraw_overlay=True,
            redraw_pipeline=True,
            clear_overlay=self.active_mask is None,
        )

    def clear_mask(self):
        self.set_active_mask(None)
        self._last_active_mask_id = None
        self.draw_mask_image(None)
        # mask_list はコンポジットの子も含む全マスクのフラットなリスト。
        # 破棄前に全マスクの release() を呼び、Window.bind 等を解除する。
        for mask in self.mask_list:
            mask.release()
        self.mask_container.clear_widgets()
        self.mask_list.clear()
        self._set_active_composit_matrix()
        FaceMask.delete_faces()
        self.request_mask_render_update(
            None,
            reason="clear_mask",
            structure_changed=True,
            refresh_visibility=False,
            redraw_overlay=False,
            redraw_pipeline=True,
            clear_overlay=True,
        )

    @staticmethod
    def _cache_bytes(value, seen=None):
        if seen is None:
            seen = set()
        if value is None:
            return 0
        value_id = id(value)
        if value_id in seen:
            return 0
        seen.add(value_id)
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if isinstance(value, dict):
            return sum(MaskEditor2._cache_bytes(v, seen) for v in value.values())
        if isinstance(value, (list, tuple, set)):
            return sum(MaskEditor2._cache_bytes(v, seen) for v in value)
        return 0

    def _iter_masks_for_memory(self):
        seen = set()

        def visit(mask):
            if mask is None or id(mask) in seen:
                return
            seen.add(id(mask))
            yield mask
            for child, _maskop in getattr(mask, "mask_list", []) or []:
                yield from visit(child)

        for mask in list(self.mask_list):
            yield from visit(mask)

    def _sweep_mask_bitmaps(self):
        """AI マスク(Segment/Face/TargetText)が保持する image_mask_cache_key の生存集合を
        mask_list ツリー全体(コンポジット含む)から収集し、共有ストアの未参照エントリを削除する。"""
        live_keys = [
            key
            for key in (getattr(mask, 'image_mask_cache_key', None) for mask in self._iter_masks_for_memory())
            if key is not None
        ]
        import traceback
        _before = list(getattr(self.ai_image_cache, "_mask_bitmaps", {}).keys())
        result = self.sweep_ai_mask_bitmaps(live_keys)
        _after = list(getattr(self.ai_image_cache, "_mask_bitmaps", {}).keys())
        logging.warning(
            "[DIAG-REINFER] _sweep_mask_bitmaps store_id=%s n_live_keys=%d before=%d after=%d result=%s\n  live=%s\n  caller=%s",
            id(self.ai_image_cache), len(live_keys), len(_before), len(_after), result,
            live_keys, "".join(traceback.format_stack(limit=6)[:-1]),
        )

    def clear_ai_intermediate_caches(self):
        ai_mask_types = (SegmentMask, TargetTextMask, DepthMapMask, FaceMask)
        cache_attrs = (
            "image_mask_cache",
            "segment_mask_cache",
            "depth_map_mask_cache",
            "faces_mask_cache",
        )
        key_attrs = (
            "image_mask_cache_hash",
            "image_mask_cache_key",
            "segment_mask_cache_hash",
            "depth_map_mask_cache_hash",
            "faces_mask_cache_hash",
        )
        removed = 0
        removed_bytes = 0
        seen_values = set()
        for mask in self._iter_masks_for_memory():
            if not isinstance(mask, ai_mask_types):
                continue
            for attr in cache_attrs:
                if not hasattr(mask, attr):
                    continue
                value = getattr(mask, attr)
                if value is None:
                    continue
                removed += 1
                removed_bytes += self._cache_bytes(value, seen_values)
                setattr(mask, attr, None)
            for attr in key_attrs:
                if hasattr(mask, attr):
                    setattr(mask, attr, None)
        if self.ai_image_cache is not None:
            # メモリ逼迫パスでは深度+派生のみ解放し、AI マスクビットマップ共有ストアは保持する
            # (スイープによる mark-and-sweep 削除に委ねる)。
            cache_result = self.ai_image_cache.clear_transient()
            removed += int(cache_result.get("ai_image_cache_entries", 0) or 0)
            removed_bytes += int(cache_result.get("ai_image_cache_bytes", 0) or 0)
        if removed:
            logging.info(
                "MaskEditor2 cleared AI intermediate caches entries=%d bytes=%d",
                removed,
                removed_bytes,
            )
        return {"mask2_entries": removed, "mask2_bytes": removed_bytes}

    def find_composit_mask(self, mask, index=0):
        # 自分の親（コンポジット）を探す
        if mask.is_composit():
            return mask     # 自分がコンポジット

        # 自分がコンポジットでない場合、コンポジットを探す
        for composit_mask in self.mask_list:
            if composit_mask.is_composit():
                if composit_mask.find_mask_op(mask) is not None:
                    return composit_mask

        # リスト内の直前の親にする
        for i in range(index-1, -1, -1):
            composit_mask = self.mask_list[i]
            if composit_mask.is_composit():
                return composit_mask

        return None

    def _find_last_active_mask(self):
        if self._last_active_mask_id is None:
            return None
        return self.find_mask(self._last_active_mask_id)

    def _first_composit_mask(self):
        for mask in self.mask_list:
            if mask.is_composit():
                return mask
        return None

    def _default_active_mask(self):
        for mask in self.mask_list:
            if not mask.is_composit():
                return mask
        return self._first_composit_mask()

    def restore_last_active_mask(self):
        if self.disabled == True:
            return False
        if self.active_mask is not None:
            self._last_active_mask_id = self.active_mask.mask_id
            self._set_active_composit_matrix()
            return True

        mask = self._find_last_active_mask()
        if mask is None:
            mask = self._default_active_mask()
        if mask is None:
            self._set_active_composit_matrix()
            return False

        self.set_active_mask(mask)
        return True

    def _invalidate_composit_mask_render_cache(self, composit):
        if composit is None:
            _mask_geom_debug("invalidate_composit_render_cache composit=None")
            return 0
        invalidated = 0
        for child, _ in getattr(composit, 'mask_list', []):
            try:
                child.invalidate_render_cache()
                invalidated += 1
            except Exception:
                logging.exception("MaskEditor: mask render cache invalidation failed")
        _mask_geom_debug(
            "invalidate_composit_render_cache composit=%s invalidated=%d",
            _mask_geom_id(composit),
            invalidated,
        )
        return invalidated

    def _active_composit_or_none(self):
        """現在 active な Composit を返す。created_mask を優先、なければ active_mask を見る。
        どちらかが Composit ならそれ自身、子マスクなら find_composit_mask で親を辿る。
        新規作成中の mask は Composit の子リストにまだ登録されていないため、
        mask_list 内のインデクスを find_composit_mask に渡して直前 Composit fallback を利かせる。"""
        target = self.created_mask if self.created_mask is not None else self.active_mask
        if target is None:
            target = self._find_last_active_mask()
        if target is None:
            return self._first_composit_mask()
        if target.is_composit():
            return target
        try:
            idx = self.mask_list.index(target)
        except ValueError:
            idx = 0
        return self.find_composit_mask(target, idx)

    def _set_active_composit_matrix(self, redraw_mask=True):
        """image-only matrix に active Composit の mask Geom matrix を左乗算して
        tcg_info['matrix'] を更新する。active mask の overlay と軸 overlay も再描画。
        _image_only_matrix が未初期化なら no-op (set_primary_param 前を保護)。

        matrix 更新は thread-safe (dict mutation) なので同期実行。graphics
        instruction の変更はメインスレッドに schedule する (パイプラインから
        非メインスレッドで呼ばれる経路があるため)。"""
        if self._image_only_matrix is None:
            _mask_geom_debug("set_active_composit_matrix skipped image_only_matrix=None active=%s", _mask_geom_id(self.active_mask))
            return
        with self._matrix_lock:
            previous_matrix = self.tcg_info.get('matrix', None)
            base = self._image_only_matrix.copy()
            composit = self._active_composit_or_none()
            enabled = composit is not None and mask_geometry_mod.is_enabled(composit.effects_param)
            if enabled:
                M_mask = mask_geometry_mod.build_matrix_tcg(
                    composit.effects_param, self.tcg_info['original_img_size'])
                base = M_mask @ base
            self.tcg_info['matrix'] = base
        matrix_changed = previous_matrix is None or not np.array_equal(previous_matrix, base)
        invalidated = 0
        if matrix_changed:
            invalidated = self._invalidate_composit_mask_render_cache(composit)
        _mask_geom_debug(
            "set_active_composit_matrix active=%s composit=%s enabled=%s changed=%s invalidated=%d prev=%s new=%s params=%s",
            _mask_geom_id(self.active_mask),
            _mask_geom_id(composit),
            enabled,
            matrix_changed,
            invalidated,
            _mask_geom_matrix_hash(previous_matrix),
            _mask_geom_matrix_hash(base),
            _mask_geom_param_summary(getattr(composit, 'effects_param', None)),
        )
        # graphics 更新はメインスレッドに委譲
        KVClock.schedule_once(lambda dt: self._refresh_overlays_main_thread(redraw_mask=redraw_mask), 0)

    def _refresh_overlays_main_thread(self, *args, redraw_mask=True):
        """メインスレッドで graphics instruction を更新するヘルパ。

        順序が重要: CP の dispatch (= raw tcg_to_window 経由で update_graphics) を
        update_mask より先に行うこと。逆順にすると Gradient 等の update_mask 内で
        direction-preserving に位置調整した CP translate を dispatch が raw に
        上書きしてしまい、slider drag 中に CP が dir-preserving と raw を交互に
        往復してブルブル震えて見える。

        対象は active composit 配下の全 draw mask。Composit 選択時 (active_mask が
        Composit) は子マスクは全て inactive (中心 CP のみ赤で表示) だが、それらも
        mask Geom 変化に追従させる必要があるため。個別 mask 選択時も兄弟マスクの
        中心 CP を同期させる。
        """
        composit = self._active_composit_or_none()
        targets = []
        if composit is not None:
            # composit 自身は draw mask ではないので子のみ。
            for child, _ in getattr(composit, 'mask_list', []):
                targets.append(child)
        else:
            # composit 解決できない fallback: 旧挙動 (active_mask だけ)
            mask = self.active_mask
            if mask is not None:
                targets.append(mask)

        previous_suppress = self._suppress_mask_overlay_draw
        self._suppress_mask_overlay_draw = previous_suppress or not redraw_mask
        try:
            for mask in targets:
                # CP の center は TCG 値で変わらないため bind(center=...) が発火せず、
                # tcg_to_window が再計算されない (= CP 位置が前 matrix のまま残る)。
                # 全 CP の center プロパティを強制 dispatch して update_graphics を呼ぶ。
                try:
                    mask.refresh_control_points_for_overlay()
                except Exception:
                    logging.exception("_refresh_overlays_main_thread: CP redispatch 失敗")
                try:
                    mask.update_mask()
                except Exception:
                    logging.exception("_refresh_overlays_main_thread: update_mask 失敗")
        finally:
            self._suppress_mask_overlay_draw = previous_suppress
        # mask Geom 軸の表示更新
        self._draw_mask_geom_axes()

    def clear_mask_geom_axes(self):
        try:
            self._axis_x_line.points = (0, 0, 0, 0)
            self._axis_y_line.points = (0, 0, 0, 0)
            self._pivot_marker.size = (0, 0)
        except Exception:
            pass

    def refresh_mask_geom_axes(self):
        self._draw_mask_geom_axes()

    def _is_geometry_tab_active(self):
        try:
            app = KVApp.get_running_app()
            root = getattr(app, "root", None)
            tab_panel = root.ids.get("effects") if root is not None and hasattr(root, "ids") else None
            current_tab = getattr(tab_panel, "current_tab", None)
            return getattr(current_tab, "text", "") == "Ge"
        except Exception:
            return False

    def _draw_mask_geom_axes(self):
        """Ge タブで Composit active かつ switch_mask_geometry ON のとき、mask Geom 座標系を overlay 表示。
        - 軸 (X=赤, Y=緑) の原点 = post-translation 点 (image_center + (tx, ty))
        - 軸の向き = rotation + flip を反映
        - Scale 効果は除外 (= 軸の "向き" だけを伝える)
        - flip 軸線そのものが mirror axis (flip H なら Y軸線、flip V なら X軸線が mirror)
        """
        if self._overlay_control_points_hidden:
            self.clear_mask_geom_axes()
            return
        if not self._is_geometry_tab_active():
            self.clear_mask_geom_axes()
            return
        if self._image_only_matrix is None:
            return
        if self._is_mesh_edit_active():
            self.clear_mask_geom_axes()
            return
        composit = self._active_composit_or_none()
        if composit is None or not mask_geometry_mod.is_enabled(composit.effects_param):
            self.clear_mask_geom_axes()
            return

        try:
            img_w, img_h = self.tcg_info['original_img_size']
            short = max(1.0, float(min(img_w, img_h)))
            half = short / 2.0
            rot_deg = float(effects.Mask2Effect.get_param(composit.effects_param, 'mask_rotation'))
            flip = int(effects.Mask2Effect.get_param(composit.effects_param, 'mask_flip_mode'))
            tx_norm = float(effects.Mask2Effect.get_param(composit.effects_param, 'mask_translation_x'))
            ty_norm = float(effects.Mask2Effect.get_param(composit.effects_param, 'mask_translation_y'))
            sign_x = -1.0 if (flip & 1) else 1.0
            sign_y = -1.0 if (flip & 2) else 1.0
            rad = math.radians(rot_deg)
            c, s = math.cos(rad), math.sin(rad)

            # 軸の原点 = post-translation 点 (TCG image-coord, Y-down)
            ox_tcg, oy_tcg = tx_norm * short, ty_norm * short
            # 軸の終点 = 原点 + R(-rad) @ (sign * half, 0) など (Y-down 空間)
            x_end_tcg = (ox_tcg + sign_x * half * c, oy_tcg + sign_x * half * (-s))
            y_end_tcg = (ox_tcg + sign_y * half * s, oy_tcg + sign_y * half * c)

            # 軸自身が mask geom matrix を表現しているので、変換時は image-only matrix を使う
            # (= mask geom matrix を二重適用しない)。
            def _axis_points():
                return (
                    self.tcg_to_window(ox_tcg, oy_tcg),
                    self.tcg_to_window(*x_end_tcg),
                    self.tcg_to_window(*y_end_tcg),
                )
            origin_win, x_end_win, y_end_win = self._call_with_image_only_matrix(_axis_points)

            self._axis_x_line.points = _axis_polyline_with_arrow(origin_win, x_end_win)
            self._axis_y_line.points = _axis_polyline_with_arrow(origin_win, y_end_win)
            self._pivot_marker.size = (0, 0)
            # scissor はウィジェットの現在サイズに合わせて更新
            self.set_scissor(self._axes_scissor)
        except Exception:
            logging.exception("_draw_mask_geom_axes 失敗")
            self.clear_mask_geom_axes()

    def _is_in_same_composit(self, mask, other):
        """mask と other が同じ Composit ツリーに属するか判定。
        各 mask の「親 Composit」(自身が Composit なら self) を取って同一性比較する。"""
        if mask is other:
            return True
        try:
            return self._mask_parent_for_visibility(mask) is self._mask_parent_for_visibility(other)
        except Exception:
            return False

    def _mask_parent_for_visibility(self, mask):
        if mask is None:
            return None
        try:
            if mask.is_composit():
                return mask
            if mask is self.created_mask:
                return self.find_composit_mask(mask, self.created_mask_index)
            return self.find_composit_mask(mask)
        except Exception:
            return None

    def visibility_reference_mask(self):
        return self.created_mask if self.created_mask is not None else self.get_active_mask()

    def overlay_mask_for_active(self, active_mask=None):
        """現在の選択で描くべき overlay mask を返す。

        - Composit 選択時: Composit の合成 overlay
        - 子マスク選択時: 親 Composit の合成 overlay
        """
        active = active_mask if active_mask is not None else self.visibility_reference_mask()
        if active is None:
            return None
        if active.is_composit():
            return active
        return self._mask_parent_for_visibility(active)

    def mask_visibility_policy_for(self, mask, active_mask=None):
        """mask の CP/overlay 表示方針を返す。

        overlay は最終的に overlay_mask_for_active() の1枚を描くが、各 mask の
        is_draw_mask も合わせておくことで直接 update される経路も破綻しにくくする。
        """
        active = active_mask if active_mask is not None else self.visibility_reference_mask()
        if mask is None or active is None:
            return {"control_points": "hidden", "overlay": False}

        mask_parent = self._mask_parent_for_visibility(mask)
        active_parent = self._mask_parent_for_visibility(active)
        if mask_parent is None or active_parent is None or mask_parent is not active_parent:
            return {"control_points": "hidden", "overlay": False}

        if active.is_composit():
            if mask is active:
                return {"control_points": "hidden", "overlay": True}
            return {"control_points": "all", "overlay": True}

        return {"control_points": "all", "overlay": True}

    def _is_mesh_edit_active(self):
        """Mesh Edit モードが有効か (main MainWidget 側のフラグを参照)。"""
        try:
            from kivy.app import App as _App
            app = _App.get_running_app()
            root = app.root if (app and app.root) else None
            if root is None:
                return False
            check = getattr(root, 'is_mask_mesh_editor_active', None)
            return bool(check()) if callable(check) else False
        except Exception:
            return False

    def refresh_mask_visibility(self, mesh_edit_active=None):
        """全 mask の表示モードを refresh する。set_active_mask の最後や
        Mesh Edit モード切替時に呼ばれる。"""
        if self._overlay_control_points_hidden:
            self._hide_overlay_and_control_points()
            return
        if mesh_edit_active is None:
            mesh_edit_active = self._is_mesh_edit_active()
        active = self.visibility_reference_mask()
        for m in self.mask_list:
            try:
                m.update_visibility_for_active(active, mesh_edit_active)
            except Exception:
                logging.exception("refresh_mask_visibility failed for top mask")
            if m.is_composit():
                for child, _op in getattr(m, 'mask_list', []):
                    try:
                        child.update_visibility_for_active(active, mesh_edit_active)
                    except Exception:
                        logging.exception("refresh_mask_visibility failed for child mask")
        if not self._mask_overlay_enabled:
            for mask in self._iter_visibility_masks():
                try:
                    mask.is_draw_mask = False
                except Exception:
                    logging.exception("refresh_mask_visibility failed to disable overlay flag")
            self.draw_mask_image(None)
        elif not mesh_edit_active:
            overlay_mask = self.overlay_mask_for_active(active)
            self._draw_overlay_mask(overlay_mask)

    def _iter_visibility_masks(self):
        for mask in self.mask_list:
            yield mask
            if mask.is_composit():
                for child, _op in getattr(mask, 'mask_list', []):
                    yield child

    def _hide_overlay_and_control_points(self):
        for mask in self._iter_visibility_masks():
            try:
                mask.opacity = 0
                for cp in getattr(mask, 'control_points', []):
                    cp.opacity = 0
                mask.is_draw_mask = False
                mask.refresh_control_points_for_overlay()
            except Exception:
                logging.exception("hide_overlay_and_control_points failed")
        self.draw_mask_image(None)
        self.clear_mask_geom_axes()

    def set_overlay_control_points_hidden(self, hidden):
        hidden = bool(hidden)
        if self._overlay_control_points_hidden == hidden:
            return
        self._overlay_control_points_hidden = hidden
        if hidden:
            self._hide_overlay_and_control_points()
            return
        self.refresh_mask_visibility()
        self._set_active_composit_matrix(redraw_mask=True)

    def set_active_mask(self, mask):
        if self.active_mask is mask:
            if mask is not None:
                self._last_active_mask_id = mask.mask_id
            _liquify_debug(
                "set_active_mask same mask=%s composit=%s",
                _mask_geom_id(mask),
                mask.is_composit() if mask is not None else None,
            )
            self._set_active_composit_matrix()
            return

        _liquify_debug(
            "set_active_mask change prev=%s next=%s next_composit=%s current_tab=%s",
            _mask_geom_id(self.active_mask),
            _mask_geom_id(mask),
            mask.is_composit() if mask is not None else None,
            getattr(getattr(self.root.ids.get("effects"), "current_tab", None), "text", None)
            if self.root is not None else None,
        )
        if self.active_mask is not None:
            self.active_mask.active = False
            self.active_mask.end()

        self.active_mask = mask
        self._mask_overlay_enabled = mask is not None
        if mask is not None:
            self._last_active_mask_id = mask.mask_id
            mask.active = True
            if mask.is_composit():
                # コンポジットなら通常属性のみ反映
                _liquify_debug(
                    "set_active_mask set2widget composit mask=%s brush=%s strength=%s records=%s",
                    _mask_geom_id(mask),
                    mask.effects_param.get("distortion_brush_size"),
                    mask.effects_param.get("distortion_strength"),
                    len(mask.effects_param.get("distortion_recorded") or []),
                )
                self.root.set2widget_all(mask.effects, mask.effects_param)
            else:
                # コンポジットでないならコンポジットの属性と合わせて反映
                composit_mask = self.find_composit_mask(mask)
                if composit_mask is not None:
                    marge_param = composit_mask.effects_param.copy()
                    marge_param.update(mask.effects_param)
                    _liquify_debug(
                        "set_active_mask set2widget child mask=%s composit=%s brush=%s strength=%s records=%s",
                        _mask_geom_id(mask),
                        _mask_geom_id(composit_mask),
                        marge_param.get("distortion_brush_size"),
                        marge_param.get("distortion_strength"),
                        len(marge_param.get("distortion_recorded") or []),
                    )
                    self.root.set2widget_all(composit_mask.effects, marge_param)
                else:
                    logging.error(f"MaskEditor: 親が見つかりませんでした。マスクを反映できません。")

            try:
                current_tab = self.root.ids["effects"].current_tab if self.root is not None else None
                if getattr(current_tab, "text", None) == "Li":
                    _liquify_debug("set_active_mask tab_sync distortion mask=%s", _mask_geom_id(mask))
                    self.root.apply_effects_lv(
                        1,
                        "distortion",
                        defer_draw=True,
                        overlay_reason="tab_sync",
                    )
            except Exception:
                logging.exception("MaskEditor: failed to switch active Liquify target")

            mask.start()
            #mask.update()
        else:
            self.draw_mask_image(None)
            if self.root is not None:
                self.root.set2widget_all(None, None, reset_effects=False)

        self.start_draw_image(fast_display=False)

        # Mask2パネルのON / OFF
        if self.root is not None:
            self.root.update_mask2_options_enabled()

        # mask Geometry: active Composit の matrix を tcg_info に反映 (overlay も再描画)
        self._set_active_composit_matrix()

        # 仕様変更: active mask が属する Composit のマスクだけ CP / overlay 表示。
        # 別 Composit のマスクは完全に隠す。Mesh Edit モード中は全 CP 非表示。
        try:
            self.refresh_mask_visibility()
        except Exception:
            logging.exception("set_active_mask: refresh_mask_visibility failed")

    def get_rotate_rad(self, rotate_rad):
        # 画像の回転角度を取得する
        rad, flip = self.tcg_info['rotation2'], self.tcg_info['flip_mode']
        
        angle_rad = rotate_rad + rad
        match flip:
            case 0: # 0: normal
                pass
            case 1: # 1: horizontal flip
                angle_rad = -angle_rad
            case 2: # 2: vertical flip
                angle_rad = angle_rad + np.radians(90)
            case 3: # 3: horizontal and vertical flip
                angle_rad = angle_rad - np.radians(180)
        
        return self.tcg_info['rotation'] + angle_rad

    def get_image_size(self):
        return self.tcg_info['original_img_size']
    
    def window_to_tcg_scale(self, x, y):
        # ワールド座標にスケーリングだけ適用する
        with self._matrix_lock:
            return params.window_to_tcg_scale((x, y), self.tcg_info)
    
    def tcg_to_window_scale(self, x, y):
        # TCG座標にスケーリングだけ適用する
        with self._matrix_lock:
            return params.tcg_to_window_scale((x, y), self.tcg_info)

    def tcg_to_image_scale(self, x, y):
        # TCG座標にスケーリングだけ適用する
        with self._matrix_lock:
            return params.tcg_to_image_scale((x, y), self.tcg_info)

    def window_to_tcg(self, cx, cy):
        # ワールド座標からTCG座標に変換する
        with self._matrix_lock:
            cx, cy = params.window_to_tcg(cx, cy, self, self.texture_size, self.tcg_info, normalize=False)
        return (cx, cy)

    def tcg_to_window(self, cx, cy):
        # TCG座標をウィンドウ座標に変換する

        with self._matrix_lock:
            return params.tcg_to_window(cx, cy, self, self.texture_size, self.tcg_info, normalize=False)

    def tcg_to_texture(self, cx, cy):
        #cx, cy = cx * device.dpi_scale(), cy * device.dpi_scale()
        #return params.tcg_to_ref_image(cx, cy, self.original_image_rgb, self.tcg_info, apply_disp_info=True)
        # TCG座標をテクスチャ座標に変換する
        #cx, cy = cx * device.dpi_scale(), cy * device.dpi_scale()
        with self._matrix_lock:
            disp_info = params.get_disp_info(self.tcg_info)
            texture_size = tuple(self.texture_size)
            imax = max(self.tcg_info['original_img_size'][0]/2, self.tcg_info['original_img_size'][1]/2)
            cx, cy = params.center_rotate(cx, cy, self.tcg_info)
        cx, cy = cx + imax, cy + imax
        cx, cy = cx - disp_info[0], cy - disp_info[1]
        cx, cy = cx * disp_info[4], cy * disp_info[4]        
        _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(*texture_size, disp_info)
        cx, cy = cx + offset_x, cy + offset_y
        return (cx, cy)

    def tcg_to_full_image(self, cx, cy):
        # TCG座標をフル画像（pipeline0処理後画像）座標に変換する
        with self._matrix_lock:
            imax = max(self.tcg_info['original_img_size'][0]/2, self.tcg_info['original_img_size'][1]/2)
            cx, cy = params.center_rotate(cx, cy, self.tcg_info)
        cx, cy = cx + imax, cy + imax
        return (cx, cy)

    def tcg_to_crop_image(self, cx, cy):
        # TCG座標をクロップ（pipeline0処理後のクロップ画像）画像座標に変換する
        cx, cy = self.tcg_to_full_image(cx, cy)
        shape_max = max(self.original_image_rgb.shape[0], self.original_image_rgb.shape[1])
        cx = cx * (self.crop_image_hls.shape[1] / shape_max)
        cy = cy * (self.crop_image_hls.shape[0] / shape_max)
        return (cx, cy)

    def tcg_to_original_image(self, cx, cy):
        # 座標変換：TCG座標（回転後） -> Original座標（回転前）
        # 1. TCG座標は元画像の中心を原点とした、回転・反転のない座標系
        # なので、単に左上原点に戻すだけでよい
        h, w = self.get_original_image_rgb().shape[:2]
        cx, cy = cx + w * 0.5, cy + h * 0.5
        cx, cy = min(max(cx, 0), w), min(max(cy, 0), h) # クリップ (範囲外に出ないように)
        return (cx, cy)

# アプリケーションクラス
class MaskEditor2App(KVApp):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.main_widget = self

    def begin_history_layer_ctrl(self, layer_ctrl, op, index):
        pass

    def end_history_layer_ctrl(self, layer_ctrl, op, index):
        pass

    def build(self):
        # 画像ファイルのパスを正しく設定してください
        image_path = 'your_image.JPG'
        if not os.path.exists(image_path):
             image_path = 'your_image.jpg'

        # KVファイルをロード
        from kivy.lang import Builder as KVBuilder
        KVBuilder.load_file(os.path.join(os.path.dirname(__file__), 'mask2_content.kv'))

        box0 = KVBoxLayout(orientation='horizontal') # 全体を横並びに
        
        # エディタ部
        editor = MaskEditor2()
        box0.add_widget(editor)

        # サイドパネル部
        from widgets import mask2_content
        side_panel = mask2_content.create_mask2_content_panel(editor)
        # サイドパネルの幅を制限
        side_panel.size_hint_x = 0.3
        box0.add_widget(side_panel)

        KVClock.schedule_once(partial(editor.imread, image_path), 0.5)

        return box0

if __name__ == '__main__':
    MaskEditor2App().run()
