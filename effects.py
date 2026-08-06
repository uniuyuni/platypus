
import cv2
import numpy as np
from enum import Enum
import os
import logging

import cores.core as core
import cores.cubelut as cubelut
import cores.exposure_fusion_debevec as exposure_fusion_debevec
from effect_backends import film_process_adapter as film_process
import cores.linear_to_log_lut as linear_to_log
import cores.filters as filters
import cores.light_rays as light_rays
import cores.hlsrgb as hlsrgb
from cores.fringe_removal import remove_chromatic_aberration
from cores.distortion_correction import (
    correct_lens_distortion, correct_trapezoid, correct_four_points, correct_with_lines, warp_mesh,
    calculate_trapezoid_homography, calculate_four_point_homography, calculate_lines_homography,
    calculate_mesh_mls_coarse_map
)
from effect_backends import cross_filter_adapter as cross_filter
from effect_backends import color_separation_adapter as color_separation
from effect_backends import coating_adapter
from effect_backends import dehaze_adapter
from effect_backends import film_grain_adapter as film_grain
from effect_backends import image_transform_adapter
from effect_backends import lens_aberration_adapter
from effect_backends import lens_effect_adapter
from effect_backends import lens_blur_adapter
from effect_backends import hls_adjust_adapter
from effect_backends import local_contrast_adapter as local_contrast
from effect_backends import subpixel_shift_adapter as subpixel_shift
from effect_backends import tone_adapter
from effect_backends import vignette_adapter as backend_vignette
import config
import pipeline
import params
import utils.utils as utils
import utils.aiutils as aiutils
import macos as device
from enums import EffectMode, ExecutionMode, ImageFidelity
from image_fidelity import heavy_ai_allowed


_DEBUG_LIQUIFY = os.getenv("PLATYPUS_DEBUG_LIQUIFY", "0").strip().lower() in {"1", "true", "yes", "on"}


def _liquify_debug(message, *args):
    if _DEBUG_LIQUIFY:
        logging.warning("[LIQUIFY] " + message, *args)


def _ai_noise_content_key(nr, upstream_hash):
    """NR 入力に効く upstream（loading_wait までのハッシュ）と nr オンオフ。強度は含めない。"""
    return hash(("scunet_v2", hash(nr), upstream_hash))


def _ai_noise_blend_raw(raw, base, nr_intensity):
    """AI NR 素出力 raw とベース画像を強度でブレンド。"""
    if raw is None or base is None or raw.shape != base.shape:
        return None
    alpha = float(nr_intensity) / 100.0
    if alpha <= 0.0:
        return base if getattr(base, "dtype", None) == np.float32 else np.asarray(base, dtype=np.float32)
    if alpha >= 1.0:
        return raw if getattr(raw, "dtype", None) == np.float32 else np.asarray(raw, dtype=np.float32)
    raw = np.ascontiguousarray(raw, dtype=np.float32)
    base = np.ascontiguousarray(base, dtype=np.float32)
    return cv2.addWeighted(raw, alpha, base, 1.0 - alpha, 0.0)


def _loading_flag_ready_for_heavy_effects(loading_flag):
    """
    image_fidelity.pipeline_loading_flag と整合: None=未ロード、-1 以下で下流の軽い補正を許可。
    """
    if loading_flag is None:
        return False
    return loading_flag <= 0


def _full_fidelity_ready_for_lens_modifier(efconfig):
    fidelity = getattr(efconfig, "image_fidelity", None)
    return fidelity is None or fidelity == ImageFidelity.FULL.value


def _geometry_preview_interpolation(crop_editing):
    if not crop_editing:
        return "area"
    value = os.getenv("PLATYPUS_GE_PREVIEW_INTERPOLATION", "linear").strip().lower()
    # pyramid_linear was too expensive in the synchronous Geometry drag path.
    if value == "pyramid_linear":
        return "linear"
    if value in {"area", "linear", "nearest"}:
        return value
    return "linear"


def _geometry_deferred_cache_enabled():
    """deferred preview transform のフレーム間キャッシュを有効にするか。

    キルスイッチ: PLATYPUS_GEOMETRY_DEFERRED_CACHE=0 で常に再構築（旧挙動）。
    """
    value = os.getenv("PLATYPUS_GEOMETRY_DEFERRED_CACHE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _geometry_signed_area(pts):
    """多角形 (N,2) の符号付き面積 (shoelace)。"""
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _geometry_quad_is_convex(pts):
    """四辺形 (4,2) が凸か (自己交差=bowtie でないか) 判定。"""
    n = len(pts)
    sign = 0
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        c = pts[(i + 2) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _geometry_matrix_is_safe(matrix, size):
    """合成ジオメトリ行列 (入力->出力の前進行列, 中心原点フレーム) が、表示破綻
    (真っ黒 / 裏表反転 / 退化) を起こさないか判定する。

    主な破綻要因:
      - 透視分母 w が画像内で符号反転 = 地平線越え -> 裏返り/黒
      - 変換後四辺形の向き反転 = 裏表逆
      - 面積の潰れ/爆発 = 退化
      - 自己交差 (bowtie) = 非凸
    """
    if matrix is None:
        return True
    M = np.asarray(matrix, dtype=np.float64)
    if M.shape != (3, 3) or not np.all(np.isfinite(M)):
        return False

    half = size / 2.0
    corners = np.array(
        [[-half, -half], [half, -half], [half, half], [-half, half]],
        dtype=np.float64,
    )
    homog = np.concatenate([corners, np.ones((4, 1), dtype=np.float64)], axis=1)
    out = homog @ M.T
    w = out[:, 2]

    # 透視分母 w: 全頂点で同符号でないと地平線をまたいで裏返る/破綻する
    if not (np.all(w > 0.0) or np.all(w < 0.0)):
        return False
    aw = np.abs(w)
    if np.max(aw) <= 0.0:
        return False
    # 一部の頂点だけ w が極端に 0 に近い = 地平線ぎりぎり (極端な引き伸ばし/黒)
    if np.min(aw) / np.max(aw) < 0.02:
        return False

    pts = out[:, :2] / w[:, None]
    if not np.all(np.isfinite(pts)):
        return False

    src_area = _geometry_signed_area(corners)
    dst_area = _geometry_signed_area(pts)
    if dst_area == 0.0:
        return False
    # 向き反転 (裏表逆) を検出
    if (src_area > 0.0) != (dst_area > 0.0):
        return False
    # 面積の極端な潰れ/爆発 (退化) のバックストップ。真の破綻要因は主に w 符号/向き/凸性。
    ratio = abs(dst_area) / abs(src_area)
    if ratio < 1e-3 or ratio > 1e3:
        return False
    if not _geometry_quad_is_convex(pts):
        return False
    return True


def _normalize_homography(H):
    """透視ホモグラフィを h22=1 に正規化 (退化時はそのまま)。"""
    H = np.asarray(H, dtype=np.float64)
    d = H[2, 2]
    if abs(d) > 1e-12:
        return H / d
    return H


def _composite_with_component(prev, comp, half_size):
    """prev (合成済み行列 or None) に comp を offset 付きで合成した行列を返す (非破壊)。"""
    temp = {} if prev is None else {'matrix': np.asarray(prev, dtype=np.float64).copy()}
    params.add_matrix(temp, comp, offset=(half_size, half_size))
    return temp['matrix']


def _largest_safe_fraction(prev, H, half_size, size, iters=28):
    """単位行列 -> H の補間 H(t)=(1-t)I + t*H を prev に足しても安全な最大 t を
    二分探索する。t=0 は無変形 (=prev, 安全)、t=1 は完全補正 (=破綻) を前提とする。"""
    Hn = _normalize_homography(H)
    identity = np.eye(3)
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        Ht = (1.0 - mid) * identity + mid * Hn
        if _geometry_matrix_is_safe(_composite_with_component(prev, Ht, half_size), size):
            lo = mid
        else:
            hi = mid
    return lo


def _safe_add_geometry_matrix(param, H, half_size, size, damp=False):
    """params.add_matrix の安全版。合成後の行列が表示破綻する場合、

      - damp=False: その成分を捨てる (追加を取り消す)。
      - damp=True:  単位行列側へ減衰させた H(t) を「安全な最大限」で足す
                    (=破綻を避けつつ“それっぽく”効かせる)。

    実際に param['matrix'] へ足した成分行列 (offset 適用前の生 H 相当: フル or 減衰版) を
    返す。何も足さなかった場合は None。画像を直接ワープする呼び出し側は、この戻り値で
    ワープすればプレビュー/確定/マスクの整合が保てる。
    """
    if H is None:
        return None
    Harr = np.asarray(H, dtype=np.float64)
    if Harr.shape != (3, 3) or not np.all(np.isfinite(Harr)):
        return None
    prev = param.get('matrix')
    prev = prev.copy() if prev is not None else None

    params.add_matrix(param, Harr, offset=(half_size, half_size))
    if _geometry_matrix_is_safe(param.get('matrix'), size):
        return Harr
    # 破綻 -> 一旦取り消し
    params.set_matrix(param, prev)  # prev=None -> 単位行列
    if not damp:
        return None
    # 安全な最大割合まで減衰させて適用
    t = _largest_safe_fraction(prev, Harr, half_size, size)
    if t <= 1e-3:
        return None  # ほぼ効かない (=無補正) なら足さない
    Ht = (1.0 - t) * np.eye(3) + t * _normalize_homography(Harr)
    params.add_matrix(param, Ht, offset=(half_size, half_size))
    if not _geometry_matrix_is_safe(param.get('matrix'), size):
        params.set_matrix(param, prev)  # 念のためのフォールバック
        return None
    return Ht


def _build_geometry_valid_mask(param):
    width, height = param['original_img_size']
    mask = np.ones((height, width, 3), dtype=np.float32)
    temp_param = param.copy()

    switch_distortion_correction = temp_param.get('switch_distortion_correction', True)
    lens_distortion_strength = temp_param.get('lens_distortion_strength', 0)
    lens_distortion_scale = temp_param.get('lens_distortion_scale', 0)
    correct_horizontal = temp_param.get('correct_horizontal', 0)
    correct_vertical = temp_param.get('correct_vertical', 0)
    focal_length = temp_param.get('focal_length', 20)
    four_points = temp_param.get('four_points', [])
    reference_lines = temp_param.get('reference_lines', [])
    mesh_size = temp_param.get('mesh_size', [4, 4])
    control_points = temp_param.get('control_points', {})

    params.set_matrix(temp_param, None)

    if switch_distortion_correction and (lens_distortion_strength != 0 or lens_distortion_scale != 0):
        mask = correct_lens_distortion(
            mask,
            strength=lens_distortion_strength,
            scale=lens_distortion_scale / 100.0 + 1.0,
            interpolation='bilinear',
            grid_size=4,
        )

    rotation_limit_mask = core.rotation(
        np.ones((height, width, 3), dtype=np.float32),
        temp_param.get('rotation', 0) + temp_param.get('rotation2', 0),
        temp_param.get('flip_mode', 0),
        inter_mode='bilinear',
        border_mode="constant",
    )

    mask = core.rotation(
        mask,
        temp_param.get('rotation', 0) + temp_param.get('rotation2', 0),
        temp_param.get('flip_mode', 0),
        inter_mode='bilinear',
        border_mode="constant",
    )

    tcg_info = params.param_to_tcg_info(temp_param)
    size = max(mask.shape[0], mask.shape[1])
    half_size = size / 2

    if switch_distortion_correction:
        # 表示側 (_build_deferred_preview_transform) と同じ安全ガードを使い、破綻する
        # 成分はマスクにも反映しない (=表示とマスクの整合を保ちつつ真っ黒/退化を防ぐ)。
        if correct_horizontal != 0 or correct_vertical != 0:
            base_f = np.max(mask.shape[:2])
            multiplier = 0.5 + (focal_length * 0.025)
            f_pixel = base_f * multiplier
            th, tw = mask.shape[:2]
            H = calculate_trapezoid_homography(
                tw, th,
                correct_horizontal * 0.5,
                correct_vertical * 0.5,
                0,
                f_pixel,
            )
            applied = _safe_add_geometry_matrix(temp_param, H, half_size, size, damp=True)
            if applied is not None:
                mask, _ = correct_trapezoid(
                    mask,
                    interpolation='bilinear',
                    homography=applied,
                )

        reset_points = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        if four_points != [] and four_points != reset_points:
            src_point = []
            for cx, cy in four_points:
                src_point.append(params.tcg_to_ref_image(cx, cy, mask, tcg_info))
            dst_point = []
            for cx, cy in reset_points:
                dst_point.append(params.tcg_to_ref_image(cx, cy, mask, tcg_info))

            try:
                H = np.linalg.inv(calculate_four_point_homography(src_point, dst_point))
            except Exception:
                H = None
            applied = _safe_add_geometry_matrix(temp_param, H, half_size, size, damp=True)
            if applied is not None:
                mask, _ = correct_four_points(
                    mask,
                    src_point,
                    dst_point,
                    interpolation='bilinear',
                    homography=applied,
                )

        if len(reference_lines) > 0:
            # Lines は破綻時に捨てず、減衰させた行列でマスクもワープする (表示と整合)
            line_tcg_info = _line_homography_tcg_info(tcg_info)
            mh, mw = mask.shape[:2]
            H = calculate_lines_homography(reference_lines, mw, mh, tcg_info=line_tcg_info)
            applied = _safe_add_geometry_matrix(temp_param, H, half_size, size, damp=True)
            if applied is not None:
                mask, _ = correct_with_lines(
                    mask,
                    reference_lines,
                    tcg_info=line_tcg_info,
                    interpolation='bilinear',
                    homography=applied,
                )

        if control_points:
            cp = {}
            for k, v in control_points.items():
                if isinstance(k, str):
                    try:
                        parts = k.strip('()').split(',')
                        key = (int(parts[0]), int(parts[1]))
                    except Exception:
                        continue
                else:
                    key = tuple(k)
                cp[key] = tuple(v)

            mask = warp_mesh(
                mask,
                mesh_size if mesh_size else (4, 4),
                cp,
                tcg_info=tcg_info,
                interpolation='bilinear',
            )

    return np.minimum(mask, rotation_limit_mask)


def _line_homography_tcg_info(tcg_info):
    """Return a TCG copy for Lines homography without image orientation.

    This keeps the already-composed perspective matrix, but evaluates the
    reference line points before rotation/flip. The image pipeline still owns
    the actual rotation step; this only changes how Lines derive their H.
    """
    line_tcg_info = tcg_info.copy()
    line_tcg_info['rotation'] = 0.0
    line_tcg_info['rotation2'] = 0.0
    line_tcg_info['flip_mode'] = 0
    return line_tcg_info


class EffectConfig():

    def __init__(self, **kwargs):
        self.disp_info = None
        self.is_zoom = False
        self.is_zoomed = False
        self.zoom_ratio = 1.0
        self.mode = EffectMode.PREVIEW
        self.resolution_scale = 1.0
        self.processor = None
        self.upstream_status = None
        self.layer_status = None
        self.upstream_hash = 0
        self.loading_flag = -1
        self.image_fidelity = None  # primary_param['image_fidelity'] を pipeline が渡す場合あり
        self.current_tab = None
        self.crop_editing = False
        self.full_preview = False
        self.pipeline_layer_label = "primary"
        self.deferred_geometry_transform = None
        self.file_path = None
        self.get_ai_depth_map = None
        self.current_level = None  # pipeline が現在実行中のレベル(0-4)を入れる。複数レベル登録の効果が判定に使う
        # スライダードラッグ中の half-res プレビュー(pipeline.pipeline2 が制御)。
        # drag_half_res: このフレームがドラッグ品質かどうか(キャンバス同期のスキップ判定用)。
        # preview_res_scale: lv1-lv2 実行中のみ縮小率(例 0.5)。params.get_disp_info(param) を
        # 直接読む効果は、この係数をスケールに掛けて縮小画像と整合させる。
        self.drag_half_res = False
        self.preview_res_scale = 1.0

    def __getstate__(self):
        """ASYNC ワーカー(別プロセス)へ pickle する際の自動仕分け。

        efconfig には毎フレーム get_ai_depth_map 等のクロージャや processor(worker を
        抱える実行時オブジェクト)が付与される。これらは pickle 不可で、放置すると
        submit が全滅し ASYNC 効果が裏で動かなくなる。ここで callable 属性と processor を
        自動で None 化することで、将来また非ピクル属性が増えても submit が壊れないようにする。
        属性名は残す(値だけ None 化)ので、ワーカー側で AttributeError にならない。
        """
        state = self.__dict__.copy()
        state['processor'] = None
        for key, value in state.items():
            if callable(value):
                state[key] = None
        return state


class ParamBinding:
    def __init__(self, key, default, widget_id, widget_attr="active", widget_setter=None):
        self.key = key
        self.default = default
        self.widget_id = widget_id
        self.widget_attr = widget_attr
        self.widget_setter = widget_setter

    def set_widget_value(self, effect, widget, param, value):
        target = widget.ids[self.widget_id]
        if self.widget_setter is not None:
            getattr(target, self.widget_setter)(value)
        else:
            setattr(target, self.widget_attr, value)

    def get_widget_value(self, effect, widget, param):
        return getattr(widget.ids[self.widget_id], self.widget_attr)


class FunctionBinding:
    def __init__(self, key, default, widget_setter, widget_getter, widget_ids=(), method_arg=None):
        self.key = key
        self.default = default
        self.widget_setter = widget_setter
        self.widget_getter = widget_getter
        self.widget_ids = widget_ids
        self.method_arg = method_arg

    def set_widget_value(self, effect, widget, param, value):
        setter = getattr(effect, self.widget_setter)
        if self.method_arg is None:
            setter(widget, param, value)
        else:
            setter(widget, param, value, self.method_arg)

    def get_widget_value(self, effect, widget, param):
        getter = getattr(effect, self.widget_getter)
        if self.method_arg is None:
            return getter(widget, param)
        return getter(widget, param, self.method_arg)


def SwitchBinding(key, default, widget_id, widget_attr="active"):
    return ParamBinding(key, default, widget_id, widget_attr=widget_attr)


def SliderBinding(key, default, widget_id):
    return ParamBinding(key, default, widget_id, widget_attr="value", widget_setter="set_slider_value")


def PointListBinding(key, default, widget_id=None):
    widget_id = key if widget_id is None else widget_id
    return FunctionBinding(
        key,
        default,
        "set_point_list_widget",
        "get_point_list_widget",
        (widget_id,),
        method_arg=widget_id,
    )


def StateBinding(key, default, widget_id, true_state="down", false_state="normal"):
    return FunctionBinding(
        key,
        default,
        "set_state_widget",
        "get_state_widget",
        (widget_id,),
        method_arg=(widget_id, true_state, false_state),
    )


def SpinnerTextBinding(key, default, widget_id):
    return FunctionBinding(
        key,
        default,
        "set_spinner_text_widget",
        "get_spinner_text_widget",
        (widget_id,),
        method_arg=widget_id,
    )


# 補正基底クラス
class Effect():
    param_bindings = ()
    # get_param_dict のデフォルト辞書を1回だけキャッシュしてよいか。
    # デフォルト値が実行時の param に依存する効果（Geometry/Crop/ColorTemperature）だけ
    # True にしてキャッシュを無効化する（古いデフォルトを返さないため）。
    _defaults_depend_on_param = False
    # _get_param のデフォルト経路用キャッシュ。クラス属性 None を既定にしておくことで、
    # super().__init__ を呼ばない subclass でも AttributeError にならない（各 instance が個別に set）。
    _default_param_cache = None

    def __init__(self, **kwargs):
        self.diff = None
        self.hash = None
        self.execution_mode = ExecutionMode.SYNC
        self.keep_async_result = True
        self._last_cache_event = None
    
    def try_async_execution(self, img, param, efconfig, param_hash):
        """
        Attempts to execute the effect asynchronously.
        Returns:
            (bool, object): 
                - handled (bool): True if async handling logic was executed (cached returned or task submitted). 
                  If True, the caller should return `result` immediately.
                - result (object): The value to return if handled is True (usually self.diff).
        """
        if self.execution_mode == ExecutionMode.ASYNC and efconfig.processor is not None and efconfig.mode != EffectMode.EXPORT:
            from enums import PipelineStatus
            
            # Mix upstream hash into the key to ensure cache validity depends on input content
            combined_hash = hash((param_hash, efconfig.upstream_hash))

            # 1. Check Upstream Status FIRST
            # If upstream is not complete, skip heavy processing (return None or Preview)
            if efconfig.upstream_status == PipelineStatus.PREVIEW:
                # Upstream is dirty/preview, so we cannot trust input `img` for heavy calc.
                # Use preview (None for now)
                self.diff = None 
                self.hash = None # Upstream is unstable, so we are unstable
                if efconfig.layer_status is not None:
                        efconfig.layer_status = PipelineStatus.PREVIEW
                self._last_cache_event = "async_upstream_preview"
                return True, self.diff
            
            # 2. Check cache with combined hash
            # We use ClassName + ParamHash + UpstreamHash as key
            cached = efconfig.processor.get_result(self.__class__.__name__, combined_hash)
            
            if cached and cached['status'] == 'COMPLETE':
                self.diff = cached['result']
                self.hash = combined_hash
                if not self.keep_async_result:
                    efconfig.processor.discard_result(self.__class__.__name__, combined_hash)
                self._last_cache_event = "async_hit"
                return True, self.diff

            # Upstream complete, check if we are already running
            if cached and cached['status'] == 'RUNNING':
                if efconfig.layer_status is not None:
                    efconfig.layer_status = PipelineStatus.PREVIEW
                self.hash = None # Running
                self._last_cache_event = "async_running"
                return True, None # Return None as preview while running
                    
            # Submit new task
            submitted = efconfig.processor.submit_task(self.__class__.__name__, img, param, efconfig, combined_hash)
            if not submitted:
                self._last_cache_event = "async_submit_failed"
                return False, None
            if efconfig.layer_status is not None:
                    efconfig.layer_status = PipelineStatus.PREVIEW
            
            self.hash = None # Submitted
            self._last_cache_event = "async_submitted"
            return True, None # Submitted
            
        return False, None

    def check_sync_necessity(self, param_hash, efconfig):
        """
        Check if synchronous recalculation is needed based on params and upstream status.
        Also handles upstream hash mixing validation.
        Returns:
            (bool, int): (needed, combined_hash)
        """
        combined_hash = hash((param_hash, efconfig.upstream_hash))
        if self.hash != combined_hash:
            return True, combined_hash
        return False, combined_hash

    def reeffect(self):
        self.diff = None
        self.hash = None

    # 差分の作成
    def make_diff(self, img, param, efconfig):
        self.diff = img

    def apply_diff(self, img):
        if self.diff is not None:
            return self.diff
        return img

    def finalize(self, param, widget):
        pass

    def get_param_dict(self, param):
        return {binding.key: binding.default for binding in self.param_bindings}

    def _get_param(self, param, key):
        if key in param:
            return param[key]
        # デフォルト値のたびに get_param_dict を丸ごと再構築する（HLS等は1呼びで45キーの
        # f-string辞書）のを避け、定数デフォルトの効果は1回だけ構築してキャッシュする。
        cache = self._default_param_cache
        if cache is None:
            cache = self.get_param_dict(param)
            if not self._defaults_depend_on_param:
                self._default_param_cache = cache
        return cache[key]

    def set2widget(self, widget, param):
        for binding in self.param_bindings:
            binding.set_widget_value(self, widget, param, self._get_param(param, binding.key))
        self.after_set2widget(widget, param)

    def set2param(self, param, widget):
        for binding in self.param_bindings:
            param[binding.key] = binding.get_widget_value(self, widget, param)
        self.after_set2param(param, widget)

    def after_set2widget(self, widget, param):
        pass

    def after_set2param(self, param, widget):
        pass

    def set_point_list_widget(self, widget, param, value, widget_id):
        widget.ids[widget_id].set_point_list(value)

    def get_point_list_widget(self, widget, param, widget_id):
        return widget.ids[widget_id].get_point_list()

    def set_state_widget(self, widget, param, value, state_config):
        widget_id, true_state, false_state = state_config
        widget.ids[widget_id].state = true_state if value else false_state

    def get_state_widget(self, widget, param, state_config):
        widget_id, true_state, _false_state = state_config
        return widget.ids[widget_id].state == true_state

    def set_spinner_text_widget(self, widget, param, value, widget_id):
        widget.ids[widget_id].set_text(value)

    def get_spinner_text_widget(self, widget, param, widget_id):
        spinner = widget.ids[widget_id]
        hovered_item = getattr(spinner, "hovered_item", None)
        return spinner.text if hovered_item is None else hovered_item.text

    def delete_default_param(self, param):
        for p in self.get_param_dict(param).items():
            try:
                if param[p[0]] == p[1]:
                    del param[p[0]]
            except KeyError:
                # 既に削除済みのキーを再度削除しようとするレース（無害）
                pass
            except ValueError:
                # numpy 配列等の値は `==` が配列を返し bool 判定できない。
                # 従来から「デフォルト一致とみなさず param に残す」動作なので維持する。
                pass

# ロード待ちエフェクト
class LoadingWaitEffect(Effect):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_mode = ExecutionMode.ASYNC

    def make_diff(self, img, param, efconfig):
        # main の「表示可能」(flag<=0) まで待つ。-1 のみだとプレビュー(0)で永遠にブロックする。
        if not _loading_flag_ready_for_heavy_effects(efconfig.loading_flag):
            # We are waiting for load completion.
            # Block downstream heavy effects.
            if efconfig.layer_status is not None:
                from enums import PipelineStatus
                efconfig.layer_status = PipelineStatus.PREVIEW
            
            # Since this is an ASYNC effect (conceptually), we assume we return None (Preview/NoOp)
            # while waiting.
            # We DONT submit a task because the Worker cannot see the main thread's loading flag.
            # We simply block here in the main thread pipeline logic.
            # This satisfies "Prevent subsequent heavy processing from starting".
            self.diff = None
            self.hash = None
            return None
        
        # If flag == -1, Loading Complete.
        # Pass through.
        self.diff = None
        self.hash = None
        return None

class RemoveChromaticAberrationEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_fringe_removal', True, "switch_fringe_removal", widget_attr="enabled"),
        SwitchBinding('rca_enabled', False, "switch_rca"),
        SliderBinding('rca_purple_amount', 20, "slider_rca_purple_amount"),
        SliderBinding('rca_green_amount', 20, "slider_rca_green_amount"),
        SliderBinding('rca_fringe_width', 20, "slider_rca_fringe_width"),
        SliderBinding('rca_edge_threshold', 10, "slider_rca_edge_threshold"),
    )
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_mode = ExecutionMode.ASYNC

    def make_diff(self, img, param, efconfig):
        switch_fringe_removal = self._get_param(param, 'switch_fringe_removal')
        rca_enabled = self._get_param(param, 'rca_enabled')
        rca_purple_amount = self._get_param(param, 'rca_purple_amount')
        rca_green_amount = self._get_param(param, 'rca_green_amount')
        rca_fringe_width = self._get_param(param, 'rca_fringe_width')
        rca_edge_threshold = self._get_param(param, 'rca_edge_threshold')
        if switch_fringe_removal == False or rca_enabled == False or not _loading_flag_ready_for_heavy_effects(efconfig.loading_flag):
            if efconfig.processor is not None:
                efconfig.processor.cancel_effect(self.__class__.__name__)
            
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((rca_enabled, rca_purple_amount, rca_green_amount, rca_fringe_width, rca_edge_threshold))

            # Async Processing Logic
            handled, result = self.try_async_execution(img, param, efconfig, param_hash)
            if handled:
                return result

            needed, combined_hash = self.check_sync_necessity(param_hash, efconfig)
            if needed:
                self.hash = combined_hash
                self.diff = remove_chromatic_aberration(
                    img,
                    purple_amount=rca_purple_amount/10,
                    green_amount=rca_green_amount/10,
                    fringe_width=rca_fringe_width,
                    lateral_correction=True,
                    edge_threshold=rca_edge_threshold/1000,
                    min_saturation=0.1
                )
        
        return self.diff

# レンズモディファイア
class LensModifierEffect(Effect):
    
    def __init__(self, lens_modifier_callback=None, **kwargs):
        super().__init__(**kwargs)

        self.mod = None

        self.callback = lens_modifier_callback

    def get_param_dict(self, param):
        return {
            'switch_lens_modifier': True,
            'lens_modifier': True,
            params.LENSFUN_USER_KEY: params.DEFAULT_LENSFUN_USER,
        }

    def set2widget(self, widget, param):
        is_cm, is_sd, is_gd = params.get_lensfun_effective_tuple(param)
        widget.ids["switch_lens_modifier"].active = self._get_param(param, 'switch_lens_modifier')
        widget.ids["checkbox_color_modification"].active = is_cm
        widget.ids["checkbox_subpixel_distortion"].active = is_sd
        widget.ids["checkbox_geometry_distortion"].active = is_gd

    def set2param(self, param, widget):
        nsw = widget.ids["switch_lens_modifier"].active
        ncm = widget.ids["checkbox_color_modification"].active
        nsd = widget.ids["checkbox_subpixel_distortion"].active
        ngd = widget.ids["checkbox_geometry_distortion"].active
        param['switch_lens_modifier'] = nsw
        t = (bool(ncm), bool(nsd), bool(ngd))
        if t == params.DEFAULT_LENSFUN_USER:
            param.pop(params.LENSFUN_USER_KEY, None)
        else:
            param[params.LENSFUN_USER_KEY] = t
        params.set_lensfun_effective_tuple(param, t)

    def delete_default_param(self, param):
        params.collapse_default_lensfun_user(param)
        super().delete_default_param(param)

    def make_diff(self, img, param, efconfig):
        switch_lm = self._get_param(param, 'switch_lens_modifier')        
        lm = self._get_param(param, 'lens_modifier')
        cd, sd, gd = params.get_lensfun_user_tuple(param)
        if (
            switch_lm == False
            or lm == False
            or (cd == False and sd == False and gd == False)
            or not _loading_flag_ready_for_heavy_effects(efconfig.loading_flag)
            or not _full_fidelity_ready_for_lens_modifier(efconfig)
        ):
            lensfun_state_before = dict(param.get(params.LENSFUN_STATE_KEY, {}))
            self.diff = None
            self.hash = None
            params.set_lensfun_effective_tuple(param, (cd, sd, gd))
            params.clear_lensfun_capability(param)
            if self.callback and param.get(params.LENSFUN_STATE_KEY, {}) != lensfun_state_before:
                self.callback()
        else:
            param_hash = hash((cd, sd, gd))
            upstream_key = getattr(efconfig, "stable_upstream_hash", None)
            if upstream_key is None:
                upstream_key = getattr(efconfig, "upstream_hash", None)
            combined_hash = hash((param_hash, upstream_key))
            needed = self.hash != combined_hash
            if needed:
                self.hash = combined_hash

                if self.mod is None:
                    self.mod = core.setup_lensfun(param['original_img_size'], param['exif_data'])
                params.set_lensfun_capability(param, core.get_lensfun_capability(self.mod, img))

                self.diff, is_cm, is_sd, is_gd = core.modify_lensfun(self.mod, img, cd, sd, gd)

                # 適用されなかったパラメータをUIに反映
                params.set_lensfun_effective_tuple(param, (is_cm, is_sd, is_gd))
                if self.callback:
                    self.callback()
        
        return self.diff
    

# サブピクセルシフト合成
class SubpixelShiftEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_details', True, "switch_details"),
        SwitchBinding('subpixel_shift', False, "switch_subpixel_shift"),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_mode = ExecutionMode.ASYNC
        self.keep_async_result = False

    def make_diff(self, img, param, efconfig):
        switch_details = self._get_param(param, 'switch_details')
        ss = self._get_param(param, 'subpixel_shift')
        if switch_details == False or ss == False or not _loading_flag_ready_for_heavy_effects(efconfig.loading_flag):
            if efconfig.processor is not None:
                efconfig.processor.cancel_effect(self.__class__.__name__)
                efconfig.processor.discard_effect_results(self.__class__.__name__)

            self.diff = None
            self.hash = None
        else:
            param_hash = hash((ss))

            needed, combined_hash = self.check_sync_necessity(param_hash, efconfig)
            if not needed and self.diff is not None:
                return self.diff

            native_ready = (
                subpixel_shift.native_enabled()
                and img.ndim == 3
                and img.shape[-1] == 3
            )
            if not native_ready:
                handled, result = self.try_async_execution(img, param, efconfig, param_hash)
                if handled:
                    return result

            if needed or self.diff is None:
                self.hash = combined_hash
                self.diff = subpixel_shift.create_enhanced_image(img)
        
        return self.diff
    

class ExposureFusionDebevecEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_details', True, "switch_details"),
        SwitchBinding('exposure_fusion_debevec', False, "switch_exposure_fusion_debevec"),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_mode = ExecutionMode.ASYNC
        self.keep_async_result = False

    def make_diff(self, img, param, efconfig):
        switch_details = self._get_param(param, 'switch_details')
        hdr = self._get_param(param, 'exposure_fusion_debevec')
        if switch_details == False or hdr == False or not _loading_flag_ready_for_heavy_effects(efconfig.loading_flag):
            if efconfig.processor is not None:
                efconfig.processor.cancel_effect(self.__class__.__name__)
                efconfig.processor.discard_effect_results(self.__class__.__name__)

            self.diff = None
            self.hash = None
        else:
            param_hash = hash((hdr))

            needed, combined_hash = self.check_sync_necessity(param_hash, efconfig)
            if not needed and self.diff is not None:
                return self.diff

            handled, result = self.try_async_execution(img, param, efconfig, param_hash)
            if handled:
                return result

            if needed:
                self.hash = combined_hash
                self.diff, _ = exposure_fusion_debevec.exposure_fusion_debevec(img, out_ldr=False)

        return self.diff


class InpaintDiff:
    def __init__(self, **kwargs):
        self.type = kwargs.get('type', "mask")
        self.disp_info = kwargs.get('disp_info', None)
        self.image = kwargs.get('image', None)
        self._image_key = kwargs.get('image_key', None)
        self._image_key_source_id = id(self.image)

    def __eq__(self, other):
        """内容ベースの等価比較。履歴(history.Operation)は backup/update を deepcopy で
        持つため、これが無いと同一内容のマスクリストでも「変化あり」と判定され、
        undo/redo 中の set2widget → on_state 再入で幽霊 op が積まれてしまう。"""
        if not isinstance(other, InpaintDiff):
            return NotImplemented
        if self.type != other.type:
            return False
        if tuple(self.disp_info or ()) != tuple(other.disp_info or ()):
            return False
        if self.image is None or other.image is None:
            return self.image is other.image
        return self.image_key() == other.image_key()

    # __eq__ 定義で __hash__ が None になるのを避ける(同一性ハッシュを維持)。
    __hash__ = object.__hash__

    def image_key(self):
        image = np.asarray(self.image)
        if self._image_key is not None and self._image_key_source_id == id(self.image):
            return self._image_key
        contiguous = np.ascontiguousarray(image)
        self._image_key = (
            image.shape,
            image.dtype.str,
            hash(contiguous.tobytes()),
        )
        self._image_key_source_id = id(self.image)
        return self._image_key

class InpaintEffect(Effect):
    # inpaint_diff_list / inpaint_mask_list のデフォルト [] は参照で使い回されるため、
    # キャッシュすると共有リストが汚染され別画像へ漏れる。キャッシュ無効化。
    _defaults_depend_on_param = True
    param_bindings = (
        SwitchBinding('switch_details', True, "switch_details"),
        StateBinding('inpaint', False, "switch_inpaint"),
        # NOTE: 'inpaint_predict' is intentionally NOT a StateBinding. The Erase
        # button is a LongPressScaledButton whose state is already "normal" when
        # on_press fires, and (being async) the flag must persist across several
        # pipeline passes until the worker result returns. It is therefore driven
        # as a durable one-shot param flag set by root._trigger_inpaint_predict()
        # and consumed (reset to False) here in make_diff.
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_mode = ExecutionMode.ASYNC
        self.keep_async_result = False
        
        self.inpaint_diff_list = []
        self.inpaint_mask_list = []
        self.mask_editor = None

    def _inpaint_mask_hash(self):
        mask_keys = []
        for inpaint_mask in self.inpaint_mask_list:
            mask_keys.append((
                tuple(inpaint_mask.disp_info),
                inpaint_mask.image_key(),
            ))
        return hash(tuple(mask_keys))

    def _inpaint_diff_hash(self):
        diff_keys = []
        for inpaint_diff in self.inpaint_diff_list:
            diff_keys.append((
                inpaint_diff.type,
                tuple(inpaint_diff.disp_info),
                inpaint_diff.image_key(),
            ))
        return hash(tuple(diff_keys))

    def _build_mask_from_inpaint_list(self, image_shape):
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        for inpaint_mask in self.inpaint_mask_list:
            proc_x, proc_y, proc_w, proc_h = [int(v) for v in inpaint_mask.disp_info]
            src = np.asarray(inpaint_mask.image, dtype=np.float32)
            if src.ndim == 3:
                src = src[:, :, 0]
            if src.size == 0:
                continue
            if float(np.nanmax(src)) > 1.0:
                src = src / 255.0

            x0 = max(proc_x, 0)
            y0 = max(proc_y, 0)
            x1 = min(proc_x + proc_w, w)
            y1 = min(proc_y + proc_h, h)
            if x1 <= x0 or y1 <= y0:
                continue

            sx0 = x0 - proc_x
            sy0 = y0 - proc_y
            sx1 = sx0 + (x1 - x0)
            sy1 = sy0 + (y1 - y0)
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], src[sy0:sy1, sx0:sx1])
        return mask

    def _set_diff_list_from_result(self, result_image):
        self.inpaint_diff_list = []
        h, w = result_image.shape[:2]
        for inpaint_mask in self.inpaint_mask_list:
            proc_x, proc_y, proc_w, proc_h = [int(v) for v in inpaint_mask.disp_info]
            x0 = max(proc_x, 0)
            y0 = max(proc_y, 0)
            x1 = min(proc_x + proc_w, w)
            y1 = min(proc_y + proc_h, h)
            if x1 <= x0 or y1 <= y0:
                continue
            self.inpaint_diff_list.append(
                InpaintDiff(
                    type="image",
                    disp_info=(x0, y0, x1 - x0, y1 - y0),
                    image=result_image[y0:y1, x0:x1].copy(),
                )
            )

    def _clear_pending_inpaint_mask(self, param):
        param['inpaint_mask_list'] = self.inpaint_mask_list = []
        if self.mask_editor is not None:
            self.mask_editor.clear_mask()
            self.mask_editor.delay_update_canvas()

    def _apply_stored_inpaint_diffs(self, img):
        if len(self.inpaint_diff_list) > 0:
            img2 = img.copy()
            h, w = img2.shape[:2]
            for inpaint_diff in self.inpaint_diff_list:
                if inpaint_diff.type == "image":
                    cx, cy, cw, ch = [int(v) for v in inpaint_diff.disp_info]
                    x0 = max(cx, 0)
                    y0 = max(cy, 0)
                    x1 = min(cx + cw, w)
                    y1 = min(cy + ch, h)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    sx0 = x0 - cx
                    sy0 = y0 - cy
                    sx1 = sx0 + (x1 - x0)
                    sy1 = sy0 + (y1 - y0)
                    img2[y0:y1, x0:x1] = inpaint_diff.image[sy0:sy1, sx0:sx1]
            self.diff = img2
        else:
            self.diff = None
        return self.diff

    def get_param_dict(self, param):
        param_dict = super().get_param_dict(param)
        param_dict.update({
            'inpaint_predict': False,
            'inpaint_diff_list': [],
            'inpaint_mask_list': [],
        })
        return param_dict

    def after_set2widget(self, widget, param):
        # 履歴描画
        if self.mask_editor is not None:
            self.mask_editor.clear_mask()
            self.inpaint_mask_list = self._get_param(param, 'inpaint_mask_list')
            for inpaint_mask in self.inpaint_mask_list:
                self.mask_editor.add_mask(inpaint_mask.disp_info, inpaint_mask.image)
            self.mask_editor.delay_update_canvas()

    def after_set2param(self, param, widget):
        if param['inpaint'] == True:
            if hasattr(widget, 'enter_mask1_full_preview_mode'):
                widget.enter_mask1_full_preview_mode('inpaint')
            if self.mask_editor is None:
                from widgets.mask_editor import MaskEditor

                self.mask_editor = MaskEditor(param,
                                              effect_ctrl_param=(0, 'inpaint'),
                                              touch_up_callback=self.mask_editor_touch_up)

                widget.ids["preview_widget"].add_widget(self.mask_editor)
                # StateBinding は widget.state を直接書き換えるため、set2widget 経由の
                # 復元(履歴 undo/redo 等)でここに再入することがある。その場合 param には
                # 既に inpaint_mask_list が(履歴から)復元済みなので [] へ強制上書きせず、
                # 既存の内容をそのまま新しい mask_editor へ描き戻す(after_set2widget と
                # 同じ絵作り)。ここで [] にすると undo で復元したばかりのマスクが
                # 消えてしまう。
                self.inpaint_mask_list = self._get_param(param, 'inpaint_mask_list')
                param['inpaint_mask_list'] = self.inpaint_mask_list
                for inpaint_mask in self.inpaint_mask_list:
                    self.mask_editor.add_mask(inpaint_mask.disp_info, inpaint_mask.image)
                self.mask_editor.delay_update_canvas()

        if param['inpaint'] == False:
            if self.mask_editor is not None:
                widget.ids["preview_widget"].remove_widget(self.mask_editor)
                self.mask_editor.release()
                self.mask_editor = None
                param['inpaint_mask_list'] = self.inpaint_mask_list = []
            if hasattr(widget, 'exit_mask1_full_preview_mode'):
                widget.exit_mask1_full_preview_mode('inpaint')


    def make_diff(self, img, param, efconfig):
        self.inpaint_diff_list = self._get_param(param, 'inpaint_diff_list')
        self.inpaint_mask_list = self._get_param(param, 'inpaint_mask_list')

        switch_details = self._get_param(param, 'switch_details')
        ip = self._get_param(param, 'inpaint')
        ipp = self._get_param(param, 'inpaint_predict')
        if switch_details == True and (ip == True and ipp == True) and heavy_ai_allowed(param):
            if len(self.inpaint_mask_list) == 0:
                param['inpaint_predict'] = False
                return self._apply_stored_inpaint_diffs(img)

            param_hash_async = self._inpaint_mask_hash()
            handled, result = self.try_async_execution(img, param, efconfig, param_hash_async)
            if handled:
                if result is not None:
                    self._set_diff_list_from_result(result)
                    param['inpaint_diff_list'] = self.inpaint_diff_list
                    param['inpaint_predict'] = False
                    self._clear_pending_inpaint_mask(param)
                    self.hash = None
                    return self._apply_stored_inpaint_diffs(img)
                if self._last_cache_event == "async_submitted":
                    param['_mask1_restore_view_after_submit'] = True
                return self.diff

            import helpers.runware_object_eraser_helper as helper
            #import helpers.sdxl_helper as helper

            mask = self._build_mask_from_inpaint_list(img.shape)
            client = helper.setup() # device=config.get_config('gpu_device'))

            # 各バウンディングごとに渡す（predict_helper は image を in-place 更新して返す）
            img_work = img.copy()
            for inpaint_mask in self.inpaint_mask_list:
                proc_x, proc_y, proc_w, proc_h = inpaint_mask.disp_info
                img_work = helper.predict_helper(client, img_work, mask, (proc_x, proc_y, proc_w, proc_h))

            self._set_diff_list_from_result(img_work)
            param['inpaint_diff_list'] = self.inpaint_diff_list
            param['inpaint_predict'] = False
            param['inpaint_mask_list'] = self.inpaint_mask_list = []
        
        param_hash = self._inpaint_diff_hash()
        if self.hash != param_hash:
            self.hash = param_hash
            self._apply_stored_inpaint_diffs(img)

        return self.diff

    def mask_editor_touch_up(self, param, mask):
        
        # イメージが四角く処理されていた場合のオフセット計算
        w, h = param['original_img_size']
        eh, ew = mask.shape[:2]
        x, y = (ew-w)//2, (eh-h)//2

        # 処理
        self.inpaint_mask_list = []
        bboxes = core.get_multiple_mask_bbox(mask)
        for bbox in bboxes:
            proc_x, proc_y, proc_w, proc_h = aiutils.calculate_expanded_crop(
                                                mask.shape[1], mask.shape[0],
                                                bbox[0] + x, bbox[1] + y, bbox[2], bbox[3],
                                                32, 32)

            # 範囲を記録
            self.inpaint_mask_list.append(
                InpaintDiff(type="mask",
                            disp_info=(proc_x, proc_y, proc_w, proc_h),
                            image=mask[proc_y:proc_y+proc_h, proc_x:proc_x+proc_w]))

        param['inpaint_mask_list'] = self.inpaint_mask_list

class PatchmatchInpaintEffect(Effect):
    # make_diff が _get_param で得た diff_list へ直接 append するため、デフォルト [] を
    # キャッシュすると共有リストが汚染され別画像へ漏れる。キャッシュ無効化。
    _defaults_depend_on_param = True
    param_bindings = (
        SwitchBinding('switch_details', True, "switch_details"),
        StateBinding('patchmatch_inpaint', False, "switch_patchmatch_inpaint"),
        # NOTE: 'patchmatch_inpaint_predict' is intentionally NOT a StateBinding.
        # The Erase button is a LongPressScaledButton whose state is already
        # "normal" when on_press fires, so a state-read binding could never see
        # "down". It is driven as a durable one-shot param flag set by
        # root._trigger_patchmatch_inpaint_predict() and consumed in make_diff.
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.inpaint_diff_list = []
        self.inpaint_mask_list = []
        self.mask_editor = None
        # Content-Aware Fill の連続失敗回数。GPU/メモリ逼迫時の一過性失敗を上限までリトライする。
        self._predict_attempts = 0

    def get_param_dict(self, param):
        param_dict = super().get_param_dict(param)
        param_dict.update({
            'patchmatch_inpaint_predict': False,
            'patchmatch_inpaint_diff_list': [],
            'patchmatch_inpaint_mask_list': [],
        })
        return param_dict

    def after_set2widget(self, widget, param):
        # 履歴描画
        if self.mask_editor is not None:
            self.mask_editor.clear_mask()
            self.inpaint_mask_list = self._get_param(param, 'patchmatch_inpaint_mask_list')
            for inpaint_mask in self.inpaint_mask_list:
                self.mask_editor.add_mask(inpaint_mask.disp_info, inpaint_mask.image)
            self.mask_editor.delay_update_canvas()

    def after_set2param(self, param, widget):
        if param['patchmatch_inpaint'] == True:
            if hasattr(widget, 'enter_mask1_full_preview_mode'):
                widget.enter_mask1_full_preview_mode('patchmatch_inpaint')
            if self.mask_editor is None:
                from widgets.mask_editor import MaskEditor

                self.mask_editor = MaskEditor(param,
                                              effect_ctrl_param=(0, 'patchmatch_inpaint'),
                                              touch_up_callback=self.mask_editor_touch_up)

                widget.ids["preview_widget"].add_widget(self.mask_editor)
                # InpaintEffect.after_set2param と同じ理由: StateBinding が widget.state を
                # 直接書き換えるため set2widget(履歴 undo/redo 等)経由でここへ再入することが
                # あり、その時点で param には既に mask_list が復元済みなので [] へ強制上書き
                # せず、既存内容を新しい mask_editor へ描き戻す。
                self.inpaint_mask_list = self._get_param(param, 'patchmatch_inpaint_mask_list')
                param['patchmatch_inpaint_mask_list'] = self.inpaint_mask_list
                for inpaint_mask in self.inpaint_mask_list:
                    self.mask_editor.add_mask(inpaint_mask.disp_info, inpaint_mask.image)
                self.mask_editor.delay_update_canvas()

        if param['patchmatch_inpaint'] == False:
            if self.mask_editor is not None:

                widget.ids["preview_widget"].remove_widget(self.mask_editor)
                self.mask_editor.release()
                self.mask_editor = None
                param['patchmatch_inpaint_mask_list'] = self.inpaint_mask_list = []
            if hasattr(widget, 'exit_mask1_full_preview_mode'):
                widget.exit_mask1_full_preview_mode('patchmatch_inpaint')

    def make_diff(self, img, param, efconfig):
        switch_details = self._get_param(param, 'switch_details')
        patchmatch_inpaint = self._get_param(param, 'patchmatch_inpaint')
        patchmatch_inpaint_predict = self._get_param(param, 'patchmatch_inpaint_predict')
        self.inpaint_diff_list = self._get_param(param, 'patchmatch_inpaint_diff_list')
        self.inpaint_mask_list = self._get_param(param, 'patchmatch_inpaint_mask_list')

        if switch_details == True and patchmatch_inpaint == True and patchmatch_inpaint_predict == True and heavy_ai_allowed(param):
            import processing_dialog

            mask = self.mask_editor.get_mask() if self.mask_editor is not None else None
            if mask is not None and logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(
                    "[INPAINT DEBUG] Image shape: %s, dtype: %s, range: [%.4f, %.4f]",
                    img.shape,
                    img.dtype,
                    img.min(),
                    img.max(),
                )
                logging.debug(
                    "[INPAINT DEBUG] Mask shape: %s, dtype: %s, unique: %s",
                    mask.shape,
                    mask.dtype,
                    np.unique(mask),
                )

            # Content-Aware Fill を実行して差分を作る。content_aware_fill は重い同期処理
            # (初回は torch import も伴う)なので、native processing dialog 配下で worker
            # スレッドに投げ、HUD を回しつつ全入力をブロックして実行する。
            #
            #  A(リトライ安全化): predict フラグは fill と差分作成が成功して初めて消費する。
            #    失敗時はフラグを残して(再アーム)次パスで自動リトライする。以前は fill 前に
            #    フラグを False にしていたため、GPU/メモリ逼迫時の一過性失敗で消し込みが
            #    無言で失われていた(重い時に「ほとんど修復しない」症状の一因)。
            #  B(GPU逼迫フォールバック): MPS 等での失敗時は device="cpu" で再実行し、GPU が
            #    他タスクで逼迫している状況でも確実に修復を完了させる(CPUなので遅い)。
            fill_ok = False
            if mask is None:
                # マスクが無ければ予測不能。無限リトライを避けるため成功扱いで消費する。
                fill_ok = True
            else:
                def _run_content_aware_fill(device=None):
                    from cores.content_aware_fill import content_aware_fill
                    return content_aware_fill(img, mask, device=device)

                processing_dialog.set_processing_text("Content-Aware Fill...")
                img2 = None
                try:
                    img2 = processing_dialog.wait_processing(_run_content_aware_fill)
                except Exception:
                    logging.warning(
                        "[PM_DIAG] content_aware_fill failed on default device (GPU?); retrying on CPU",
                        exc_info=True,
                    )
                    processing_dialog.set_processing_text("Content-Aware Fill (CPU)...")
                    try:
                        img2 = processing_dialog.wait_processing(lambda: _run_content_aware_fill(device="cpu"))
                    except Exception:
                        logging.exception(
                            "[PM_DIAG] content_aware_fill CPU fallback also failed; keeping predict flag for retry"
                        )
                        img2 = None

                if img2 is not None:
                    # [PM_DIAG] 予測時の実値。img と mask/差分座標の解像度整合を確認するためのログ。
                    logging.warning(
                        "[PM_DIAG] PREDICT img.shape=%s dtype=%s range=[%.4f,%.4f] | mask.shape=%s sum=%d | "
                        "original_img_size=%s fidelity=%s | fill.shape=%s | n_masks=%d",
                        getattr(img, "shape", None), getattr(img, "dtype", None),
                        float(img.min()) if img is not None else float("nan"),
                        float(img.max()) if img is not None else float("nan"),
                        getattr(mask, "shape", None), int((mask > 0).sum()),
                        param.get('original_img_size'), param.get('image_fidelity'),
                        getattr(img2, "shape", None), len(self.inpaint_mask_list),
                    )

                    for inpaint_mask in self.inpaint_mask_list:
                        proc_x, proc_y, proc_w, proc_h = inpaint_mask.disp_info
                        _patch = img2[proc_y:proc_y+proc_h, proc_x:proc_x+proc_w]
                        logging.warning(
                            "[PM_DIAG]   diff disp_info=(%s,%s,%s,%s) -> patch.shape=%s (fill in-bounds=%s)",
                            proc_x, proc_y, proc_w, proc_h, getattr(_patch, "shape", None),
                            (proc_y + proc_h <= img2.shape[0] and proc_x + proc_w <= img2.shape[1]),
                        )

                        # 範囲を記録
                        self.inpaint_diff_list.append(
                            InpaintDiff(type="image",
                                        disp_info=(proc_x, proc_y, proc_w, proc_h),
                                        image=_patch))
                    fill_ok = True

            if fill_ok:
                # 成功: フラグ消費・差分確定・マスク消去。
                self._predict_attempts = 0
                param['patchmatch_inpaint_predict'] = False
                param['patchmatch_inpaint_diff_list'] = self.inpaint_diff_list

                # マスク消去
                param['patchmatch_inpaint_mask_list'] = self.inpaint_mask_list = []
                if self.mask_editor:
                    self.mask_editor.clear_mask()
                    self.mask_editor.delay_update_canvas()
            else:
                # 失敗: predict フラグを残して次パスで自動リトライ(マスクも保持)。
                # 連続失敗が続く場合は無限リトライを避けるため上限で打ち切る。
                self._predict_attempts = getattr(self, "_predict_attempts", 0) + 1
                if self._predict_attempts >= 3:
                    logging.error(
                        "[PM_DIAG] content_aware_fill failed %d times; giving up this erase (mask kept for manual retry)",
                        self._predict_attempts,
                    )
                    self._predict_attempts = 0
                    param['patchmatch_inpaint_predict'] = False

        param_hash = hash((len(self.inpaint_diff_list)))
        if self.hash != param_hash:
            self.hash = param_hash

            if len(self.inpaint_diff_list) > 0:
                img2 = img.copy()
                for inpaint_diff in self.inpaint_diff_list:
                    if inpaint_diff.type == "image":
                        cx, cy, cw, ch = inpaint_diff.disp_info
                        # [PM_DIAG] 適用時の実値。予測時と img 解像度が違うと座標ズレ/形状不一致で
                        # 差分が正しく貼り戻せず「消えない」ように見える。原因切り分け用の一時ログ。
                        dst = img2[cy:cy+ch, cx:cx+cw]
                        if dst.shape[:2] != inpaint_diff.image.shape[:2]:
                            logging.warning(
                                "[PM_DIAG] APPLY MISMATCH img.shape=%s disp_info=(%s,%s,%s,%s) "
                                "dst.shape=%s patch.shape=%s -> diff NOT applied",
                                img2.shape, cx, cy, cw, ch, dst.shape, inpaint_diff.image.shape,
                            )
                        else:
                            logging.warning(
                                "[PM_DIAG] APPLY ok img.shape=%s disp_info=(%s,%s,%s,%s)",
                                img2.shape, cx, cy, cw, ch,
                            )
                        img2[cy:cy+ch, cx:cx+cw] = inpaint_diff.image
                self.diff = img2
            else:
                self.diff = None

        return self.diff

    def mask_editor_touch_up(self, param, mask):
        
        # イメージが四角く処理されていた場合のオフセット計算
        w, h = param['original_img_size']
        eh, ew = mask.shape[:2]
        x, y = (ew-w)//2, (eh-h)//2

        # 処理
        self.inpaint_mask_list = []
        bboxes = core.get_multiple_mask_bbox(mask)
        for bbox in bboxes:
            proc_x, proc_y, proc_w, proc_h = aiutils.calculate_expanded_crop(
                                                mask.shape[1], mask.shape[0],
                                                bbox[0] + x, bbox[1] + y, bbox[2], bbox[3],
                                                32, 32)

            # 範囲を記録
            self.inpaint_mask_list.append(
                InpaintDiff(type="mask",
                            disp_info=(proc_x, proc_y, proc_w, proc_h),
                            image=mask[proc_y:proc_y+proc_h, proc_x:proc_x+proc_w]))

        param['patchmatch_inpaint_mask_list'] = self.inpaint_mask_list

class CrossFilterEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_cross_filter', True, "switch_cross_filter", widget_attr="enabled"),
        SliderBinding('cross_filter_num_points', 0, "slider_cross_filter_num_points"),
        SliderBinding('cross_filter_length', 2000, "slider_cross_filter_length"),
        SliderBinding('cross_filter_angle', 0, "slider_cross_filter_angle"),
        SliderBinding('cross_filter_threshold', 70, "slider_cross_filter_threshold"),
        SliderBinding('cross_filter_intensity', 15, "slider_cross_filter_intensity"),
        SliderBinding('cross_filter_spectral', 25, "slider_cross_filter_spectral"),
        SliderBinding('cross_filter_thickness', 1, "slider_cross_filter_thickness"),
        SliderBinding('cross_filter_distance', 100, "slider_cross_filter_distance"),
        SliderBinding('cross_filter_random', 50, "slider_cross_filter_random"),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.execution_mode = ExecutionMode.ASYNC

    def make_diff(self, rgb, param, efconfig):
        switch_cross_filter = self._get_param(param, 'switch_cross_filter')
        num_points = self._get_param(param, 'cross_filter_num_points')
        length = self._get_param(param, 'cross_filter_length') #* efconfig.disp_info[4]
        angle = self._get_param(param, 'cross_filter_angle')
        threshold = self._get_param(param, 'cross_filter_threshold')
        intensity = self._get_param(param, 'cross_filter_intensity') #/ max(0.01, efconfig.disp_info[4])
        spectral = self._get_param(param, 'cross_filter_spectral')
        thickness = max(1.0, self._get_param(param, 'cross_filter_thickness')) #* efconfig.disp_info[4])
        distance = self._get_param(param, 'cross_filter_distance') #* efconfig.disp_info[4]
        random = self._get_param(param, 'cross_filter_random')
        if switch_cross_filter is False or num_points == 0 or length <= 1 or intensity == 0 or not _loading_flag_ready_for_heavy_effects(efconfig.loading_flag):
            if efconfig.processor is not None:
                efconfig.processor.cancel_effect(self.__class__.__name__)

            self.diff = None
            self.hash = None
        else:
            param_hash = hash((num_points, length, angle, threshold, intensity, spectral, thickness, distance, random))

           # Async Processing Logic
            handled, result = self.try_async_execution(rgb, param, efconfig, param_hash)
            if handled:
                return result

            needed, combined_hash = self.check_sync_necessity(param_hash, efconfig)
            if needed:
                self.hash = combined_hash

                self.diff = cross_filter.apply_cross_filter(
                                rgb,
                                num_points=int(num_points),
                                length=int(length),
                                angle_deg=angle,
                                threshold=threshold/50.0,
                                intensity=intensity/100.0,
                                spectral_strength=spectral/100.0,
                                line_thickness=thickness,
                                min_distance=distance,
                                randomness=random/100.0,
                                speed_factor=4)

        return self.diff

# 色合わせ (Color Match)
class ColorMatchEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_color_match', True, "switch_color_match"),
        SwitchBinding('switch_color_match_active', False, "switch_color_match_active"),
        SliderBinding('color_match_intensity', 100, "slider_color_match_intensity"),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # predict 結果はメモリのみ。pmck には保存しない。
        self._cached_predict = None
        self._cached_predict_key = None

    def get_param_dict(self, param):
        param_dict = super().get_param_dict(param)
        param_dict['color_match_source_image'] = None
        return param_dict

    def reeffect(self):
        super().reeffect()
        # predict キャッシュは入力ハッシュで判断するので維持。

    def make_diff(self, img, param, efconfig):
        switch = self._get_param(param, 'switch_color_match')
        active = self._get_param(param, 'switch_color_match_active')
        intensity = self._get_param(param, 'color_match_intensity')
        source = self._get_param(param, 'color_match_source_image')

        if not switch or not active or source is None or not isinstance(source, np.ndarray) or intensity == 0:
            self.diff = None
            self.hash = None
            return self.diff

        # 強度は最終ブレンドにのみ効く。predict 自体は (source, img.shape, upstream) で決まる。
        predict_key = (id(source), tuple(img.shape), efconfig.upstream_hash)
        if self._cached_predict_key != predict_key or self._cached_predict is None:
            import helpers.color_matcher_helper as cmh
            import cores.color as color
            # MKL は知覚均等空間で安定するため sRGB ガンマでエンコードしてから掛ける。
            # source は読み込み時にエンコード済み。
            img_enc = color.srgb_gamma_encode(np.ascontiguousarray(img, dtype=np.float32)).astype(np.float32)
            src_in = np.ascontiguousarray(source, dtype=np.float32)
            try:
                predict_enc = cmh.predict(img_enc, src_in)
                # リニア ProPhoto に戻して以降のブレンドに渡す
                self._cached_predict = color.srgb_gamma_decode(predict_enc).astype(np.float32)
                self._cached_predict_key = predict_key
            except Exception as e:
                logging.warning(f"ColorMatchEffect predict failed: {e}")
                self._cached_predict = None
                self._cached_predict_key = None
                self.diff = None
                self.hash = None
                return self.diff

        final_hash = hash((predict_key, intensity))
        if self.hash == final_hash and self.diff is not None:
            return self.diff

        result = self._cached_predict
        if result is None or result.shape != img.shape:
            self.diff = None
            self.hash = None
            return self.diff

        alpha = float(intensity) / 100.0
        if alpha >= 1.0:
            self.diff = result
        else:
            base = np.ascontiguousarray(img, dtype=np.float32)
            self.diff = cv2.addWeighted(result, alpha, base, 1.0 - alpha, 0.0)

        self.hash = final_hash
        return self.diff

# 変形描画
class DistortionEffect(Effect):
    # distortion_recorded のデフォルト [] はペインタへ参照で渡され in-place に append される。
    # キャッシュすると共有リストが汚染され、param.clear() 後もストロークが復活するため無効化。
    _defaults_depend_on_param = True
    param_bindings = (
        SwitchBinding('switch_distortion', True, "switch_distortion", widget_attr="enabled"),
        SliderBinding('distortion_brush_size', 300, "slider_distortion_brush_size"),
        SliderBinding('distortion_strength', 50, "slider_distortion_strength"),
    )

    def __init__(self, distortion_callback=None, view_param_provider=None, **kwargs):
        super().__init__(**kwargs)
        
        self.distortion_painter = None
        self.is_initial_open = 0
        self.effect_type = 'forward_warp'
        self._painter_ref_key = None
        self.set_distortion_callback(distortion_callback)
        self.set_view_param_provider(view_param_provider)

    def set_distortion_callback(self, callback):
        self.distortion_callback = callback

    def set_view_param_provider(self, provider):
        self.view_param_provider = provider if callable(provider) else None

    def _provider_view_param(self):
        if self.view_param_provider is None:
            return None
        try:
            view_param = self.view_param_provider()
        except Exception:
            logging.exception("view_param_provider failed")
            return None
        return view_param if isinstance(view_param, dict) else None

    def get_param_dict(self, param):
        param_dict = super().get_param_dict(param)
        param_dict['distortion_recorded'] = []
        return param_dict

    def after_set2widget(self, widget, param):
        if self.distortion_painter is not None:
            _liquify_debug(
                "distortion after_set2widget painter=%s brush=%s strength=%s records=%d",
                id(self.distortion_painter),
                self._get_param(param, 'distortion_brush_size'),
                self._get_param(param, 'distortion_strength'),
                len(self._get_param(param, 'distortion_recorded') or []),
            )
            self.distortion_painter.set_recorded(self._get_param(param, 'distortion_recorded'))
            self.distortion_painter.remap_recorded()

    def _view_param(self, param, widget=None, efconfig=None):
        """Return the coordinate context used by Liquify.

        Mask/Composit effect params intentionally do not persist disp_info, but
        Liquify records/replays TCG coordinates in the currently displayed image
        space. Use the live primary/efconfig view while keeping strokes stored in
        the layer param.
        """
        base = None
        if widget is not None:
            base = getattr(widget, 'primary_param', None)
        if not (isinstance(base, dict) and base.get('original_img_size') is not None):
            base = self._provider_view_param()
        if not (isinstance(base, dict) and base.get('original_img_size') is not None):
            if param.get('disp_info') is not None:
                return param
            base = None

        if isinstance(base, dict) and base.get('original_img_size') is not None:
            view_param = base.copy()
        else:
            view_param = param.copy()

        if view_param.get('original_img_size') is None:
            view_param['original_img_size'] = param.get('original_img_size')

        disp_info = getattr(efconfig, 'disp_info', None) if efconfig is not None else None
        if disp_info is not None and view_param.get('original_img_size') is not None:
            params.set_disp_info(view_param, disp_info)
        return view_param

    def after_set2param(self, param, widget):
        # 生成/破棄は 'effect' トグル駆動の _sync_effect_editors() に一本化(toggle-driven へ全面移行)。
        # 以前はここで (tab=='Li' and can_open) を起点に開閉していた。
        sync = getattr(widget, '_sync_effect_editors', None)
        if callable(sync):
            sync()

        if self.distortion_painter is not None:
            if hasattr(self.distortion_painter, 'set_primary_param'):
                self.distortion_painter.set_primary_param(self._view_param(param, widget=widget))
            self.distortion_painter.set_brush_size(param['distortion_brush_size'])
            self.distortion_painter.set_strength(param['distortion_strength'])


    def set2param2(self, param, arg):
        # ペインタ未生成でも effect_type を保持(トグル駆動で同押下時に生成される場合に備える)。
        self.effect_type = arg
        if self.distortion_painter is not None:
            self.distortion_painter.set_effect(arg)

    def _make_painter_ref_key(self, img, param, efconfig):
        view_param = self._view_param(param, efconfig=efconfig)
        matrix = view_param.get('matrix')
        if matrix is not None:
            matrix = tuple(np.asarray(matrix, dtype=np.float64).round(8).ravel())
        return (
            tuple(img.shape),
            str(img.dtype),
            params.get_disp_info(view_param),
            tuple(view_param.get('original_img_size', ())),
            tuple(view_param.get('img_size', ())),
            view_param.get('rotation', 0.0),
            view_param.get('rotation2', 0.0),
            view_param.get('flip_mode', 0),
            matrix,
            getattr(efconfig, 'upstream_hash', None),
        )

    def _sync_distortion_painter_ref(self, img, param, efconfig, force=False):
        distortion_painter = self.distortion_painter
        if distortion_painter is None:
            return
        # half-res ドラッグフレームでは img が縮小画像・efconfig.disp_info が縮小スケールの
        # ため、GUI エディタ(painter)には流さない。ビューはドラッグ中不変なので同期不要。
        if getattr(efconfig, 'drag_half_res', False):
            return
        ref_key = self._make_painter_ref_key(img, param, efconfig)
        if not force and ref_key == self._painter_ref_key:
            return

        _liquify_debug(
            "sync_painter_ref force=%s painter=%s records=%d old_key=%s new_key=%s layer=%s",
            force,
            id(distortion_painter),
            len(self._get_param(param, 'distortion_recorded') or []),
            self._painter_ref_key is not None,
            ref_key is not None,
            getattr(efconfig, "pipeline_layer_label", None),
        )
        distortion_painter.set_effect(self.effect_type)
        distortion_painter.set_primary_param(self._view_param(param, efconfig=efconfig))
        distortion_painter.set_ref_image(img, True)
        distortion_painter.set_recorded(self._get_param(param, 'distortion_recorded'))
        distortion_painter.remap_recorded()
        if self.distortion_painter is distortion_painter:
            self._painter_ref_key = ref_key
        else:
            self.diff = None
            self.hash = None
            self._painter_ref_key = None

    def make_diff(self, img, param, efconfig):
        if self.is_initial_open > 0:
            if self.distortion_painter is not None and efconfig.loading_flag != None:
                self._sync_distortion_painter_ref(img, param, efconfig, force=True)

                if _loading_flag_ready_for_heavy_effects(efconfig.loading_flag):
                    self.is_initial_open = 0
        elif self.distortion_painter is not None:
            self._sync_distortion_painter_ref(img, param, efconfig)
        
        switch_distortion = self._get_param(param, 'switch_distortion')
        if switch_distortion == True and self.distortion_painter is not None:
            self.diff = self.distortion_painter.get_current_image()
            self.hash = hash((
                len(self.distortion_painter.get_recorded()),
                self._painter_ref_key,
                getattr(self.distortion_painter, "get_live_revision", lambda: 0)(),
                id(self.diff),
            ))
            _liquify_debug(
                "make_diff painter records=%d diff_id=%s revision=%s ref_key=%s layer=%s",
                len(self.distortion_painter.get_recorded()),
                id(self.diff) if self.diff is not None else None,
                getattr(self.distortion_painter, "get_live_revision", lambda: 0)(),
                self._painter_ref_key is not None,
                getattr(efconfig, "pipeline_layer_label", None),
            )

        else:
            dr = self._get_param(param, 'distortion_recorded')

            if switch_distortion == False or len(dr) == 0:
                self.diff = None
                self.hash = None
            else:
                param_hash = hash((len(dr)))
                if self.hash != param_hash:
                    from widgets.distortion_painter import DistortionCanvas

                    tcg_info = params.param_to_tcg_info(self._view_param(param, efconfig=efconfig))
                    self.diff = DistortionCanvas.replay_recorded(img, dr, tcg_info)
                self.hash = param_hash
        
        return self.diff

    def apply_diff(self, img):
        if self.diff is not None:
            if self.distortion_painter is not None:
                self.diff = self.distortion_painter.get_current_image()
                if self.diff is not None:
                    return self.diff
            else:
                return self.diff
        return img

    def finalize(self, param, widget):
        self._close_distortion_painter(param, widget)

    def _open_distortion_painter(self, param, widget):
        if self.distortion_painter is None:
            from widgets.distortion_painter import DistortionCanvas

            self.distortion_painter = DistortionCanvas(#image_widget=widget.ids["preview_widget"],
                    recorded=self._get_param(param, 'distortion_recorded'),
                    callback=self._painter_callback,
                    effect_type=self.effect_type,
                    brush_size=widget.ids["slider_distortion_brush_size"].value,
                    strength=widget.ids["slider_distortion_strength"].value)
            self.distortion_painter.set_primary_param(self._view_param(param, widget=widget))
            self.distortion_painter.is_recording = True
            widget.ids["preview_widget"].add_widget(self.distortion_painter, index=0)
            _liquify_debug(
                "open_painter painter=%s parent=%s brush=%s strength=%s records=%d",
                id(self.distortion_painter),
                type(self.distortion_painter.parent).__name__ if self.distortion_painter.parent is not None else None,
                self.distortion_painter.brush_size,
                self.distortion_painter.strength,
                len(self.distortion_painter.get_recorded() or []),
            )

            self.is_initial_open = 1
        self._bring_distortion_painter_to_front(widget)

    def _bring_distortion_painter_to_front(self, widget):
        painter = self.distortion_painter
        if painter is None:
            return
        preview_widget = widget.ids.get("preview_widget")
        if preview_widget is None or painter.parent is not preview_widget:
            return
        if len(preview_widget.children) > 0 and preview_widget.children[0] is painter:
            return
        preview_widget.remove_widget(painter)
        preview_widget.add_widget(painter, index=0)

    def _close_distortion_painter(self, param, widget):
        if self.distortion_painter is not None:
            _liquify_debug(
                "close_painter painter=%s records=%d parent=%s",
                id(self.distortion_painter),
                len(self.distortion_painter.get_recorded() or []),
                type(self.distortion_painter.parent).__name__ if self.distortion_painter.parent is not None else None,
            )
            widget.ids["preview_widget"].remove_widget(self.distortion_painter)
            param['distortion_recorded'] = self.distortion_painter.get_recorded()
            self.distortion_painter = None
            self.diff = None
            self.hash = None
            self._painter_ref_key = None

    def _painter_callback(self, proc, widget):
        if self.distortion_callback is not None:
            self.distortion_callback(proc, widget)

# 画像回転、反転、変形
class GeometryEffect(Effect):
    # get_param_dict のデフォルトが param 依存のためキャッシュ無効化
    _defaults_depend_on_param = True
    param_bindings = (
        SliderBinding('rotation', 0, "slider_rotation"),
        SwitchBinding('switch_distortion_correction', True, "switch_distortion_correction"),
        SliderBinding('lens_distortion_strength', 0, "slider_lens_distortion_strength"),
        SliderBinding('lens_distortion_scale', 0, "slider_lens_distortion_scale"),
        SliderBinding('correct_horizontal', 0, "slider_correct_trapezoid_h"),
        SliderBinding('correct_vertical', 0, "slider_correct_trapezoid_v"),
        SliderBinding('focal_length', 20, "slider_focal_length"),
    )

    def __init__(self, geometry_callback=None, **kwargs):
        super().__init__(**kwargs)

        self.geometry_editor = None
        self.geometry_editor_callback = geometry_callback
        # deferred preview transform のフレーム間キャッシュ (build 結果)。
        # pipeline_lv0 は毎フレーム make_diff を呼ぶため、ガード無しだと
        # 無関係なスライダー操作中も MLS coarse map (~70ms@8K) を再構築してしまう。
        self._deferred_preview_cache = None

    def _editor_update_callback(self, type, widget):
        if self.geometry_editor_callback:
            self.geometry_editor_callback(type, widget)

    def get_param_dict(self, param, subname=None):
        if subname == "rotation":
            return {
                'rotation': 0,
                'rotation2': 0,
                'flip_mode': 0,
            }

        if subname == "distortion_correction":
            return {
                'switch_distortion_correction': True,
                'lens_distortion_strength': 0,
                'lens_distortion_scale': 0,
                'correct_horizontal': 0,
                'correct_vertical': 0,
                'focal_length': 20,
                'four_points': [],
                'reference_lines': [],
                'mesh_size': [4, 4],
                'control_points': {},
                'matrix': np.eye(3),
            }

        default_param = super().get_param_dict(param)
        default_param.update({
            'rotation': 0,
            'rotation2': 0,
            'flip_mode': 0,
            'switch_distortion_correction': True,
            'lens_distortion_strength': 0,
            'lens_distortion_scale': 0,
            'correct_horizontal': 0,
            'correct_vertical': 0,
            'focal_length': 20,
            'four_points': [],
            'reference_lines': [],
            'mesh_size': [4, 4],
            'control_points': {},
            'matrix': np.eye(3),
        })

        original_img_size = param.get('original_img_size')
        if original_img_size is not None:
            param2 = param.copy()
            params.set_crop_rect(param2, core.get_initial_crop_rect(*original_img_size))
            params.set_disp_info(param2, core.convert_rect_to_info(params.get_crop_rect(param2), config.get_preview_texture_side()/max(original_img_size)))
            default_param['crop_rect'] = param2['crop_rect']
            default_param['disp_info'] = param2['disp_info']

        return default_param

    def after_set2widget(self, widget, param):
        if self.geometry_editor is not None:
            self.geometry_editor.set_correction_params(param)

        if hasattr(widget, "sync_distortion_mode_sliders"):
            widget.sync_distortion_mode_sliders()

    def after_set2param(self, param, widget):
        # crop_rect がないのはマスク
        if params.get_crop_rect(param) is not None:

            def get_selected():
                """1行で全確認"""
                for btn_name in ['btn_lens', 'btn_trapezoid', 'btn_four_points', 'btn_mesh', 'btn_lines']:
                    btn = getattr(widget.ids, btn_name)
                    if btn.state == 'down':
                        return btn.text
                return None
            
            # Update params from editor if active (BEFORE opening/syncing)
            if self.geometry_editor is not None:
                d = self.geometry_editor.get_correction_params()
                # Lens オーバーレイの strength/scale は ParamSlider と共有しておらず、
                # 毎回 get で上書きするとスライダーで入れた値が潰れる。レンズ2キーは上書きしない。
                if self.geometry_editor.__class__.__name__ == "LensDistortionWidget":
                    d = {
                        k: v
                        for k, v in d.items()
                        if k not in ("lens_distortion_strength", "lens_distortion_scale")
                    }
                param.update(d)

            if os.getenv("PLATYPUS_DEBUG_4PT", "0").strip().lower() in {"1", "true", "yes", "on"}:
                logging.warning(
                    "[4PT] after_set2param editor=%s selected=%s param.four_points=%s",
                    self.geometry_editor.__class__.__name__ if self.geometry_editor else None,
                    get_selected(), param.get('four_points'),
                )

            # Update Matrix Param based on current params (Visual Fix)
            self._update_matrix_param(param)

            type = get_selected()
            self._open_geometry_editor(widget, type, param)

    def _update_matrix_param(self, param):
        """
        画像処理を行わずにパラメータのみからマトリックスを計算・更新する
        """
        params.set_matrix(param, None)

        if self._get_param(param, 'switch_distortion_correction') == False:
            return

        # パラメータ取得
        correct_horizontal = self._get_param(param, 'correct_horizontal')
        correct_vertical = self._get_param(param, 'correct_vertical')
        focal_length = self._get_param(param, 'focal_length')
        #ang = self._get_param(param, 'rotation')
        #ang2 = self._get_param(param, 'rotation2')
        four_points = self._get_param(param, 'four_points')
        reference_lines = self._get_param(param, 'reference_lines')
        
        # 基準サイズ（回転後を想定して max(w, h) の正方形）
        w_org, h_org = param['original_img_size']
        size = max(w_org, h_org)
        half_size = size / 2

        # 台形補正
        if correct_horizontal != 0 or correct_vertical != 0:
            multiplier = 0.5 + (focal_length * 0.025)
            f_pixel = size * multiplier
            
            H = calculate_trapezoid_homography(
                size, size,
                horizontal=correct_horizontal * 0.5,
                vertical=correct_vertical * 0.5,
                focal_length=f_pixel,
            )
            _safe_add_geometry_matrix(param, H, half_size, size, damp=True)

        # 4点補正
        reset_points = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        if four_points != [] and four_points != reset_points:
            # 座標変換用のダミー画像サイズ
            # Note: tcg_info内のmatrixはここまで（台形補正）の結果を含んでいる必要がある
            # params.add_matrixでparam['matrix']は更新されている
            tcg_info = params.param_to_tcg_info(param)
            
            class DummyShape:
                def __init__(self, s): self.shape = (s, s, 3)
            dummy_img = DummyShape(size)

            src_point = []
            for cx, cy in four_points:
                src_point.append(params.tcg_to_ref_image(cx, cy, dummy_img, tcg_info))
            dst_point = []
            for cx, cy in reset_points:
                dst_point.append(params.tcg_to_ref_image(cx, cy, dummy_img, tcg_info))

            # dst -> src (Inverse) -> src -> dst (Forward)
            # 退化した4点 (共線/自己交差) では getPerspectiveTransform / inv が
            # 例外や特異行列を返すので、失敗時は None にして安全ガードで捨てる。
            try:
                H_inv = calculate_four_point_homography(src_point, dst_point)
                H = np.linalg.inv(H_inv)
            except Exception:
                H = None
            _safe_add_geometry_matrix(param, H, half_size, size, damp=True)

        # Lines (破綻時は捨てず安全な最大限まで減衰させる)
        if len(reference_lines) > 0:
            tcg_info = params.param_to_tcg_info(param)
            line_tcg_info = _line_homography_tcg_info(tcg_info)
            H = calculate_lines_homography(reference_lines, size, size, tcg_info=line_tcg_info)
            if H is not None:
                _safe_add_geometry_matrix(param, H, half_size, size, damp=True)

    def set2param2(self, param, arg):
        if arg == 'hflip':
            param['flip_mode'] = self._get_param(param, 'flip_mode') ^ 1

        elif arg == 'vflip':
            param['flip_mode'] = self._get_param(param, 'flip_mode') ^ 2

        elif arg == 90:
            rot = self._get_param(param, 'rotation2') + 90.0
            if rot >= 90*4:
                rot = 0
            param['rotation2'] = rot

        elif arg == -90:
            rot = self._get_param(param, 'rotation2') - 90.0
            if rot < 0:
                rot = 90*3
            param['rotation2'] = rot


    def _build_deferred_preview_transform(
        self,
        img,
        param,
        ang,
        ang2,
        flp,
        switch_distortion_correction,
        correct_horizontal,
        correct_vertical,
        focal_length,
        four_points,
        reference_lines,
        mesh_size,
        control_points,
    ):
        params.set_matrix(param, None)
        size = max(img.shape[0], img.shape[1])
        half_size = size / 2
        reset_points = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        has_matrix = False

        if switch_distortion_correction:
            if correct_horizontal != 0 or correct_vertical != 0:
                multiplier = 0.5 + (focal_length * 0.025)
                f_pixel = size * multiplier
                H = calculate_trapezoid_homography(
                    size,
                    size,
                    horizontal=correct_horizontal * 0.5,
                    vertical=correct_vertical * 0.5,
                    focal_length=f_pixel,
                )
                has_matrix = (_safe_add_geometry_matrix(param, H, half_size, size, damp=True) is not None) or has_matrix

            if four_points != [] and four_points is not None and four_points != reset_points:
                tcg_info = params.param_to_tcg_info(param)

                class DummyShape:
                    def __init__(self, s):
                        self.shape = (s, s, 3)

                dummy_img = DummyShape(size)
                src_point = []
                for cx, cy in four_points:
                    src_point.append(params.tcg_to_ref_image(cx, cy, dummy_img, tcg_info))
                dst_point = []
                for cx, cy in reset_points:
                    dst_point.append(params.tcg_to_ref_image(cx, cy, dummy_img, tcg_info))

                try:
                    H_inv = calculate_four_point_homography(src_point, dst_point)
                    H = np.linalg.inv(H_inv)
                except Exception:
                    H = None
                has_matrix = (_safe_add_geometry_matrix(param, H, half_size, size, damp=True) is not None) or has_matrix

            if len(reference_lines or []) > 0:
                tcg_info = params.param_to_tcg_info(param)
                line_tcg_info = _line_homography_tcg_info(tcg_info)
                H = calculate_lines_homography(reference_lines, size, size, tcg_info=line_tcg_info)
                if H is not None:
                    # Lines は破綻時に捨てず、安全な最大限まで減衰させて適用する (damp=True)
                    has_matrix = (_safe_add_geometry_matrix(param, H, half_size, size, damp=True) is not None) or has_matrix

        mesh_map_x = None
        mesh_map_y = None
        if control_points:
            cp = {}
            for key, value in control_points.items():
                if isinstance(key, str):
                    try:
                        parts = key.strip('()').split(',')
                        cp_key = (int(parts[0]), int(parts[1]))
                    except Exception:
                        continue
                else:
                    cp_key = tuple(key)
                cp[cp_key] = tuple(value)

            if cp:
                tcg_info = params.param_to_tcg_info(param)
                mesh_maps = calculate_mesh_mls_coarse_map(
                    size,
                    size,
                    mesh_size if mesh_size else (4, 4),
                    cp,
                    tcg_info=tcg_info,
                    grid_step=64,
                )
                if mesh_maps is not None:
                    mesh_map_x, mesh_map_y = mesh_maps

        matrix = param.get("matrix") if has_matrix else None
        transform_matrix, size, transform_type = core.combined_rotation_canvas_matrix(img.shape, ang + ang2, flp, matrix)
        return transform_matrix, size, transform_type, mesh_map_x, mesh_map_y


    def _store_zero_wrap_quad(self, param, full_preview, image_shape, transform_matrix, size, transform_type):
        """apply_zero_wrap 用のコンテンツ四辺形（変換キャンバス size で正規化）を param に格納する。

        full_preview（Ge タブ＝回転正方形パディング表示。Mask2 ON/OFF 問わず）のときだけ
        格納し、それ以外は None でクリアして stale を防ぐ。通常表示はクロップ枠外の矩形
        黒塗りのみ（枠内のミラー画素はエクスポートと同様に残す）で、quad マスクは使わない。
        main.py 側の crop_editing は current_tab=="Ge" で判定され full_preview と一致する。
        """
        if not full_preview:
            param['_zero_wrap_content_quad'] = None
            param['_zero_wrap_canvas_size'] = None
            return
        try:
            quad = core.content_quad_norm(image_shape, transform_matrix, size, transform_type)
            param['_zero_wrap_content_quad'] = quad.tolist()
            param['_zero_wrap_canvas_size'] = float(size)
        except Exception:
            logging.exception("failed to compute zero-wrap content quad")
            param['_zero_wrap_content_quad'] = None
            param['_zero_wrap_canvas_size'] = None

    def make_diff(self, img, param, efconfig):
        ang = self._get_param(param, 'rotation')
        ang2 = self._get_param(param, 'rotation2')
        flp = self._get_param(param, 'flip_mode')
        crop_editing = getattr(efconfig, 'crop_editing', False)
        full_preview = getattr(efconfig, 'full_preview', crop_editing)
        switch_distortion_correction = self._get_param(param, 'switch_distortion_correction')
        lens_distortion_strength = self._get_param(param, 'lens_distortion_strength')
        lens_distortion_scale = self._get_param(param, 'lens_distortion_scale')
        correct_horizontal = self._get_param(param, 'correct_horizontal')
        correct_vertical = self._get_param(param, 'correct_vertical')
        focal_length = self._get_param(param, 'focal_length')
        four_points = self._get_param(param, 'four_points')
        reference_lines = self._get_param(param, 'reference_lines')
        mesh_size = self._get_param(param, 'mesh_size')
        control_points = self._get_param(param, 'control_points') # dict

        # list, convert to tuple for hashing
        fps_hash = tuple(tuple(x) for x in four_points) if four_points else None
        lines_hash = tuple(tuple(tuple(p) for p in line) for line in reference_lines) if reference_lines else None
        cp_hash = tuple(sorted((k, tuple(v)) for k, v in control_points.items())) if control_points else None
        mesh_hash = tuple(mesh_size)
        preview_interpolation = _geometry_preview_interpolation(crop_editing)
        
        param_hash = hash((switch_distortion_correction, ang, ang2, flp, crop_editing, full_preview, preview_interpolation, lens_distortion_strength, lens_distortion_scale, correct_horizontal, correct_vertical, focal_length, fps_hash, lines_hash, mesh_hash, cp_hash))
        lens_active = switch_distortion_correction and (lens_distortion_strength != 0 or lens_distortion_scale != 0)
        deferred_geometry_supported = (
            efconfig.mode != EffectMode.EXPORT
            and image_transform_adapter.native_available()
            and img.dtype == np.float32
            and img.ndim == 3
            and img.shape[2] == 3
            and (not lens_active or lens_distortion_scale == 0)
        )
        if deferred_geometry_supported:
            try:
                # pipeline_lv0 は毎フレーム make_diff を呼ぶため、build（特に
                # calculate_mesh_mls_coarse_map ~70ms@8K）を param 不変時に
                # スキップするフレーム間キャッシュ。キーには param_hash に加え
                # img.shape（size = max(h,w) を決める）と original_img_size
                # （param_to_tcg_info 経由で four_points/lines/mesh の座標変換に
                # 影響）を含める。disp_info は tcg_to_ref_image が
                # apply_disp_info=False で呼ばれるため意図的にキーへ入れない
                # （ズーム/パンで無効化しないことが狙い）。
                cache_key = (
                    param_hash,
                    img.shape,
                    tuple(param.get('original_img_size') or ()),
                )
                cache = self._deferred_preview_cache
                if (
                    _geometry_deferred_cache_enabled()
                    and cache is not None
                    and cache['key'] == cache_key
                ):
                    # build は param['matrix'] を変異させるため、ヒット時も
                    # build 後と同じ値へ復元して他の消費者から見た状態を揃える。
                    params.set_matrix(param, cache['param_matrix'])
                    transform_matrix = cache['transform_matrix']
                    size = cache['size']
                    transform_type = cache['transform_type']
                    mesh_map_x = cache['mesh_map_x']
                    mesh_map_y = cache['mesh_map_y']
                else:
                    transform_matrix, size, transform_type, mesh_map_x, mesh_map_y = self._build_deferred_preview_transform(
                        img,
                        param,
                        ang,
                        ang2,
                        flp,
                        switch_distortion_correction,
                        correct_horizontal,
                        correct_vertical,
                        focal_length,
                        four_points,
                        reference_lines,
                        mesh_size,
                        control_points,
                    )
                    matrix_after_build = param.get('matrix')
                    self._deferred_preview_cache = {
                        'key': cache_key,
                        'param_matrix': None if matrix_after_build is None else np.array(matrix_after_build, copy=True),
                        'transform_matrix': transform_matrix,
                        'size': size,
                        'transform_type': transform_type,
                        # mesh 配列は参照共有でキャッシュ。ヒット時も同一オブジェクト
                        # を渡すことでベースポインタが安定し Metal 側の zero-copy
                        # (newBufferWithBytesNoCopy) が効き続ける。
                        'mesh_map_x': mesh_map_x,
                        'mesh_map_y': mesh_map_y,
                    }
                efconfig.deferred_geometry_transform = {
                    "matrix": transform_matrix,
                    "width": size,
                    "height": size,
                    "transform_type": transform_type,
                    "border_mode": "constant" if full_preview else "reflect",
                    "lens_strength": lens_distortion_strength if lens_active else 0.0,
                    "lens_scale": 1.0,
                    "interpolation": preview_interpolation,
                    "mesh_map_x": mesh_map_x,
                    "mesh_map_y": mesh_map_y,
                    "hash": param_hash,
                }
                self._store_zero_wrap_quad(param, full_preview, img.shape, transform_matrix, size, transform_type)
                self.hash = param_hash
                self.diff = None
                return self.diff
            except Exception:
                logging.exception("deferred geometry transform build failed; falling back to two-pass geometry")
                efconfig.deferred_geometry_transform = None
                self._deferred_preview_cache = None
                self.hash = None

        if self.hash != param_hash:
            self.hash = param_hash
            efconfig.deferred_geometry_transform = None
            # 二パス経路では正確なクォッドを単一行列で得にくいため、安全に旧挙動へ
            # フォールバックさせる（apply_zero_wrap はクォッド無しで矩形ロジックに戻る）。
            param['_zero_wrap_content_quad'] = None
            param['_zero_wrap_canvas_size'] = None

            params.set_matrix(param, None)

            # レンズ歪み補正
            if switch_distortion_correction == True and (lens_distortion_strength != 0 or lens_distortion_scale != 0):
                img = correct_lens_distortion(
                        img,
                        strength=lens_distortion_strength,
                        scale=lens_distortion_scale / 100.0 + 1.0,
                        interpolation='bicubic' if efconfig.mode == EffectMode.EXPORT else 'bilinear',
                        grid_size=2 if efconfig.mode == EffectMode.EXPORT else 4,
                )

            # 回転
            img = core.rotation(img, ang + ang2, flp,
                    inter_mode='bicubic' if efconfig.mode == EffectMode.EXPORT else 'bilinear',
                    border_mode="constant" if full_preview else "reflect")

            tcg_info = params.param_to_tcg_info(param)
            # 基準サイズ（回転後を想定して max(w, h) の正方形）
            size = max(img.shape[0], img.shape[1])
            half_size = size / 2

            if switch_distortion_correction == True:
                # 台形補正
                if correct_horizontal != 0 or correct_vertical != 0:
                    # Focal Length Mapping:
                    # 0-100 Slider -> Multiplier.
                    # Assuming 0 is Wide (High persp), 100 is Tele (Low persp).
                    # Previous default was max(w,h) which corresponds to freq standard lens.
                    # Let's say Slider=20 -> 1.0x (Standard)
                    # Slider=0 -> 0.5x (Super Wide)
                    # Slider=100 -> 5.0x (Super Tele)
                    
                    # Using a linear mapping for simplicity first:
                    # val 20 -> 1.0
                    # val 0 -> 0.5 (delta -20 -> -0.5 => 1 unit = 0.5/20 = 0.025)
                    # val 100 -> 1.0 + (80 * 0.025) = 1.0 + 2.0 = 3.0
                    
                    # So: multiplier = 0.5 + (focal_length* 0.025)
                    # 0 -> 0.5
                    # 20 -> 1.0
                    # 100 -> 3.0
                    # --- Trapezoid Correction ---
                    base_f = np.max(img.shape[:2])
                    multiplier = 0.5 + (focal_length * 0.025)
                    f_pixel = base_f * multiplier # Focal len in pixels

                    # 破綻時は捨てず、安全な最大限まで減衰させた行列で画像もワープする。
                    lh, lw = img.shape[:2]
                    H = calculate_trapezoid_homography(
                        lw, lh,
                        correct_horizontal * 0.5,
                        correct_vertical * 0.5,
                        0,
                        f_pixel,
                    )
                    applied = _safe_add_geometry_matrix(param, H, half_size, size, damp=True)
                    if applied is not None:
                        img, _ = correct_trapezoid(
                            img,
                            interpolation='bicubic' if efconfig.mode == EffectMode.EXPORT else 'bilinear',
                            homography=applied,
                        )

                # 4点補正
                reset_points = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
                if four_points != [] and four_points != reset_points:

                    # 座標をテクスチャ座標へ変換
                    src_point = []
                    for cx, cy in four_points:
                        src_point.append(params.tcg_to_ref_image(cx, cy, img, tcg_info))
                    dst_point = []
                    for cx, cy in reset_points:
                        dst_point.append(params.tcg_to_ref_image(cx, cy, img, tcg_info))

                    # 破綻時は捨てず、安全な最大限まで減衰させた行列で画像もワープする。
                    try:
                        H = np.linalg.inv(calculate_four_point_homography(src_point, dst_point))
                    except Exception:
                        H = None
                    applied = _safe_add_geometry_matrix(param, H, half_size, size, damp=True)
                    if applied is not None:
                        img, _ = correct_four_points(
                                img,
                                src_point,
                                dst_point,
                                interpolation='lanczos' if efconfig.mode == EffectMode.EXPORT else 'bilinear',
                                homography=applied,
                        )

                # Lines (破綻時は捨てず、安全な最大限まで減衰させた行列で画像もワープする)
                if len(reference_lines) > 0:
                    line_tcg_info = _line_homography_tcg_info(tcg_info)
                    lh, lw = img.shape[:2]
                    H = calculate_lines_homography(reference_lines, lw, lh, tcg_info=line_tcg_info)
                    applied = _safe_add_geometry_matrix(param, H, half_size, size, damp=True)
                    if applied is not None:
                        img, _ = correct_with_lines(
                            img,
                            reference_lines,
                            tcg_info=line_tcg_info,
                            interpolation='lanczos' if efconfig.mode == EffectMode.EXPORT else 'bilinear',
                            homography=applied,
                        )

                # Mesh           
                if control_points:
                    # Ensure keys are tuples
                    cp = {}
                    for k, v in control_points.items():
                        if isinstance(k, str):
                            try:
                                parts = k.strip('()').split(',')
                                key = (int(parts[0]), int(parts[1]))
                            except (ValueError, IndexError):
                                continue
                        else:
                            key = tuple(k)
                        cp[key] = tuple(v)
                        
                    img = warp_mesh(
                        img,
                        mesh_size if mesh_size else (4, 4),
                        cp,
                        tcg_info=tcg_info,
                        interpolation='lanczos' if efconfig.mode == EffectMode.EXPORT else 'bilinear'
                    )

            self.diff = img
        
        return self.diff

    def finalize(self, param, widget):
        self._open_geometry_editor(widget, None) # 閉じる

    def _open_geometry_editor(self, widget, type, param=None):
        from widgets.distortion_correction import (
            LensDistortionWidget, LineGuideCorrectionWidget, TrapezoidCorrectionWidget, FourPointCorrectionWidget, MeshWarpWidget
        )
        
        # Check if we can reuse the existing editor
        current_editor_class = self.geometry_editor.__class__.__name__ if self.geometry_editor else None
        target_class = None
        match type:
            case 'Lens': target_class = 'LensDistortionWidget'
            case 'Trapezoid': target_class = 'TrapezoidCorrectionWidget'
            case 'Four Points': target_class = 'FourPointCorrectionWidget'
            case 'Mesh': target_class = 'MeshWarpWidget'
            case 'Lines': target_class = 'LineGuideCorrectionWidget'
            case 'Points': target_class = 'PointWarpWidget'

        # 前のを削除
        if current_editor_class != target_class:
            if self.geometry_editor is not None:
                widget.ids['preview_widget'].remove_widget(self.geometry_editor)
                self.geometry_editor = None

        # 作成
        if self.geometry_editor is None:
            texture_size = config.get_preview_texture_size()
            match type:
                case 'Lens': self.geometry_editor = LensDistortionWidget(texture_size, param)
                case 'Trapezoid': self.geometry_editor = TrapezoidCorrectionWidget(texture_size, param)
                case 'Four Points': self.geometry_editor = FourPointCorrectionWidget(texture_size, param)
                case 'Mesh': self.geometry_editor = MeshWarpWidget(texture_size, param)
                case 'Lines': self.geometry_editor = LineGuideCorrectionWidget(texture_size, param)

            if self.geometry_editor is not None:
                self.geometry_editor.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
                self.geometry_editor.type = type # 消す時必要
                widget.ids['preview_widget'].add_widget(self.geometry_editor)

        # Update parameters and image if applicable
        if self.geometry_editor is not None and param is not None:
            if type == 'Lens':
                #self.geometry_editor.set_image(widget.imgset.img) # なんか重い
                # Sync Params
                #self.geometry_editor.set_correction_params(param)
                pass

            elif type == 'Trapezoid':
                pass

            elif type == 'Four Points':                
                # Sync Params
                self.geometry_editor.set_correction_params(param)
                # Bind callback
                self.geometry_editor.set_callback(self._editor_update_callback)

            elif type == 'Lines':
                # Sync Params
                self.geometry_editor.set_correction_params(param)
                # Bind callback
                self.geometry_editor.set_callback(self._editor_update_callback)

            elif type == 'Mesh':
                # Sync Params
                self.geometry_editor.set_correction_params(param)
                # Bind callback
                self.geometry_editor.set_callback(self._editor_update_callback)

    def close_geometry_editor(self, widget):
        if self.geometry_editor is not None:
            btn_id = f"btn_{self.geometry_editor.type.lower().replace(' ', '_')}"
            widget.ids[btn_id].state = 'normal'
            widget.ids['preview_widget'].remove_widget(self.geometry_editor)
            self.geometry_editor = None

    def update_geometry_editor_texture_size(self, param=None):
        if self.geometry_editor is not None and hasattr(self.geometry_editor, 'set_texture_size'):
            self.geometry_editor.set_texture_size(config.get_preview_texture_size())
            # resize / cmd+F でプレビューサイズが変わると param 側の disp_info が
            # 再計算されるが、エディタ生成時に取り込んだ tcg_info は古いままなので
            # グリッド/CP 表示がズレる。param から view 座標系を再同期してリセットする。
            if param is not None and hasattr(self.geometry_editor, 'set_view_param'):
                self.geometry_editor.set_view_param(param)

# クロップ
class CropEffect(Effect):
    # get_param_dict のデフォルトが param 依存のためキャッシュ無効化
    _defaults_depend_on_param = True

    def __init__(self, crop_callback=None, **kwargs):
        super().__init__(**kwargs)
        
        self.backup_img = None

        self.crop_editor = None
        self.crop_editor_callback = crop_callback
        self._rotation_preview_crop_rect = None

    def set_editing_callback(self, callback):
        self.crop_editor_callback = callback

    def _param_to_aspect_ratio(self, param):
        ar = self._get_param(param, 'aspect_ratio')
        if not ar or ar == "None":
            return 0
        try:
            if "/" in str(ar):
                numerator, denominator = str(ar).split("/", 1)
                return float(numerator) / float(denominator)
            return float(ar)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0

    def get_param_dict(self, param):
        default_param = {
            'rotation': 0,
            'rotation2': 0,
            'aspect_ratio': "None",
            'auto_crop': False,
        }

        original_img_size = param.get('original_img_size')
        if original_img_size is not None:
            param2 = param.copy()
            params.set_crop_rect(param2, core.get_initial_crop_rect(*original_img_size))
            # disp_info は crop_rect から導出される派生値だが、undo/redo は
            # Operation.backup/update に含まれるキーしか書き戻さない
            # (history.py Operation.undo/redo の dict.update)。ここに含めないと
            # crop_rect だけ戻って disp_info が古いまま残り、make_diff (下の
            # crop_editing=False 分岐) が「disp_info はフルプレビュー状態でない
            # ＝既に正しい」と誤判定してハッシュ不変のまま crop_image の再構築が
            # トリガーされず、通常タブでは undo 後もクロップが反映されなかった。
            params.set_disp_info(param2, core.convert_rect_to_info(params.get_crop_rect(param2), config.get_preview_texture_side()/max(original_img_size)))
            default_param['crop_rect'] = param2['crop_rect']
            default_param['disp_info'] = param2['disp_info']

        return default_param

    def set2widget(self, widget, param):
        widget.ids["spinner_acpect_ratio"].set_text(param.get('aspect_ratio', "None"))
        self.sync_crop_editor_from_param(param)

    def set2param(self, param, widget):
        on_ge = widget.ids["effects"].current_tab.text == "Ge"
        try:
            mask2_on = widget.ids["mask2"].state == "down"
        except Exception:
            mask2_on = False
        # マスク Geometry モードではクロップエディタを開かない
        crop_editing = on_ge and not mask2_on
        param['aspect_ratio'] = widget.ids["spinner_acpect_ratio"].text

        # crop_rect がないのはマスク
        if params.get_crop_rect(param) is not None:

            # クロップエディタを開く
            if crop_editing:
                self._open_crop_editor(param, widget)

            # クロップエディタを閉じる
            else:
                self._close_crop_editor(param, widget)

            # クロップ範囲をリセット
            if widget.ids["button_crop_reset"].state == "down":
                self.reset2_crop_editor(param)
                self.reset_crop_editor()

            self.reset2_crop_editor(param)
            if self.crop_editor is not None and self._rotation_preview_crop_rect is not None:
                self.crop_editor.set_to_local_crop_rect(self._rotation_preview_crop_rect)
                self.crop_editor.update_crop_size()

            # 自動クロップ
            if widget.ids["button_crop_auto"].state == "down":
                self.auto_crop_editor(self.backup_img, param)

            # クロップ情報を更新
            if self.crop_editor is not None:
                enforce_bounds = widget.ids["button_crop_auto"].state != "down"
                if self._rotation_preview_crop_rect is None:
                    params.set_crop_rect(param, self.crop_editor.get_crop_rect(enforce_bounds=enforce_bounds))

    def apply_crop_button_action(self, param, widget, action):
        if params.get_crop_rect(param) is None:
            return

        param['aspect_ratio'] = widget.ids["spinner_acpect_ratio"].text
        self._open_crop_editor(param, widget)

        if action == "reset":
            self.reset2_crop_editor(param)
            self.reset_crop_editor()

        self.reset2_crop_editor(param)

        if action == "auto":
            self.auto_crop_editor(self.backup_img, param)

        if self.crop_editor is not None:
            enforce_bounds = action != "auto"
            params.set_crop_rect(param, self.crop_editor.get_crop_rect(enforce_bounds=enforce_bounds))
            params.set_disp_info(param, self.crop_editor.get_disp_info(enforce_bounds=enforce_bounds))

    def sync_crop_editor_mode_from_widget(self, widget, param):
        # マスク Geometry モード (Mask2 ON + Ge タブ) ではクロップエディタを開かない
        on_ge = widget.ids["effects"].current_tab.text == "Ge"
        try:
            mask2_on = widget.ids["mask2"].state == "down"
        except Exception:
            mask2_on = False
        crop_editing = on_ge and not mask2_on
        if params.get_crop_rect(param) is None:
            return

        if crop_editing:
            self._open_crop_editor(param, widget)
            self.sync_crop_editor_from_param(param)
        else:
            self._close_crop_editor(param, widget)

    def _full_preview_disp_info(self, param):
        original_img_size = param.get('original_img_size')
        if original_img_size is None:
            return None
        msize = max(original_img_size[0], original_img_size[1])
        scale = config.get_preview_texture_side() / msize
        return (0, 0, msize, msize, scale)

    def _is_full_preview_disp_info(self, param, disp_info):
        full_disp_info = self._full_preview_disp_info(param)
        if full_disp_info is None or disp_info is None:
            return False
        return (
            int(disp_info[0]) == int(full_disp_info[0]) and
            int(disp_info[1]) == int(full_disp_info[1]) and
            int(disp_info[2]) == int(full_disp_info[2]) and
            int(disp_info[3]) == int(full_disp_info[3]) and
            abs(float(disp_info[4]) - float(full_disp_info[4])) < 1e-6
        )

    def _crop_rect_disp_info(self, param):
        crop_rect = params.get_crop_rect(param)
        original_img_size = param.get('original_img_size')
        if crop_rect is None or original_img_size is None:
            return None
        return core.convert_rect_to_info(
            crop_rect,
            config.get_preview_texture_side() / max(original_img_size),
        )

    def make_diff(self, img, param, efconfig):
        crop_editing = getattr(efconfig, 'crop_editing', False)
        disp_info = params.get_disp_info(param)

        self.backup_img = img

        if crop_editing:
            self.diff = None
            self.hash = None
            param['img_size'] = (param['original_img_size'][0], param['original_img_size'][1])
            params.set_disp_info(param, self._full_preview_disp_info(param))
        else:
            if disp_info is None or self._is_full_preview_disp_info(param, disp_info):
                crop_disp_info = self._crop_rect_disp_info(param)
                if crop_disp_info is not None:
                    params.set_disp_info(param, crop_disp_info)
                    disp_info = params.get_disp_info(param)
                elif disp_info is None:
                    params.set_disp_info(param, self._full_preview_disp_info(param))
                    disp_info = params.get_disp_info(param)

            param_hash = hash((crop_editing, disp_info))
            if self.hash != param_hash:
                self.diff = disp_info
                self.hash = param_hash
                if disp_info is not None:
                    param['img_size'] = (disp_info[2], disp_info[3])
        return self.diff

    def apply_diff(self, img):
        return img

    def _open_crop_editor(self, param, widget):
        if self.crop_editor is None:
            from widgets.crop_editor import CropEditor

            input_width, input_height = param['original_img_size']
            x1, y1, x2, y2 = params.get_crop_rect(param)
            scale = config.get_preview_texture_side() * device.dpi_scale() / max(input_width, input_height)
            self.crop_editor = CropEditor(input_width=input_width, input_height=input_height, scale=scale, crop_rect=[x1, y1, x2, y2], aspect_ratio=self._param_to_aspect_ratio(param))
            self.crop_editor.set_editing_callback(self._crop_editing)
            widget.ids["preview_widget"].add_widget(self.crop_editor)

            # 編集中は一時的に変更
            params.set_disp_info(param, core.get_initial_disp_info(input_width, input_height, scale))

            # 保存しておく
            self.param = param

    def _close_crop_editor(self, param, widget):
        if self.crop_editor is not None:
            params.set_crop_rect(param, self.crop_editor.get_crop_rect())
            params.set_disp_info(param, self.crop_editor.get_disp_info())
            widget.ids["preview_widget"].remove_widget(self.crop_editor)
            self.crop_editor = None

    def _crop_editing(self, proc, widget):
        if self.crop_editor_callback is not None:
            self.crop_editor_callback(proc, widget)

    def begin_rotation_preview(self, param):
        self._rotation_preview_crop_rect = params.get_crop_rect(param)

    def end_rotation_preview(self, param):
        if self.crop_editor is not None and self._rotation_preview_crop_rect is not None:
            self.crop_editor.set_to_local_crop_rect(self._rotation_preview_crop_rect)
            self.crop_editor.update_crop_size()
            params.set_crop_rect(param, self.crop_editor.get_crop_rect())
        self._rotation_preview_crop_rect = None

    def reset_crop_editor(self):
        if self.crop_editor is not None:
            self.crop_editor.set_to_local_crop_rect((0, 0, 0, 0))
            self.crop_editor.update_crop_size()

    def reset2_crop_editor(self, param):
        if self.crop_editor is not None:
            self.crop_editor.input_angle = self._get_param(param, 'rotation') + self._get_param(param, 'rotation2')
            self.crop_editor.set_aspect_ratio(self._param_to_aspect_ratio(param))

    def sync_crop_editor_from_param(self, param):
        if self.crop_editor is None:
            return
        crop_rect = (
            self._rotation_preview_crop_rect
            if self._rotation_preview_crop_rect is not None
            else params.get_crop_rect(param)
        )
        if crop_rect is None:
            return

        input_width, input_height = param['original_img_size']
        self.crop_editor.input_width = input_width
        self.crop_editor.input_height = input_height
        self.crop_editor.scale = config.get_preview_texture_side() * device.dpi_scale() / max(input_width, input_height)
        self.crop_editor.input_angle = self._get_param(param, 'rotation') + self._get_param(param, 'rotation2')

        if self._rotation_preview_crop_rect is not None:
            self.crop_editor.set_aspect_ratio(self._param_to_aspect_ratio(param))
            self.crop_editor.set_to_local_crop_rect(crop_rect)
            self.crop_editor.update_crop_size()
            return

        # set_aspect_ratio may resize the current editor rect; restore the saved param rect last.
        self.crop_editor.set_to_local_crop_rect(crop_rect)
        self.crop_editor.set_aspect_ratio(self._param_to_aspect_ratio(param))
        self.crop_editor.set_to_local_crop_rect(crop_rect)
        self.crop_editor.update_rect()
        self.crop_editor.update_centering()

    def update_crop_editor_preview_size(self, param):
        if self.crop_editor is None:
            return
        self.sync_crop_editor_from_param(param)

    # 自動クロップ
    def auto_crop_editor(self, img, param=None):
        import cores.find_bounding_box as find_bounding_box

        if img is not None:
            # クロップエディタのアスペクト比設定を取得
            aspect_ratio = None
            if self.crop_editor is not None:
                ar = self.crop_editor.aspect_ratio
                # aspect_ratioが0でない場合のみ使用
                if ar is not None and ar > 0:
                    aspect_ratio = ar

            if param is not None:
                valid_mask = _build_geometry_valid_mask(param)
                bbox = find_bounding_box.find_largest_inscribed_rectangle_in_mask(
                    valid_mask,
                    aspect_ratio=aspect_ratio,
                    threshold=0.999,
                    verbose=True,
                )
            else:
                bbox = find_bounding_box.find_bounding_box(
                    img,
                    threshold=0.0001,
                    aspect_ratio=aspect_ratio,
                    verbose=True
                )
            self.crop_editor.set_to_local_crop_rect(bbox, enforce_bounds=param is None)

    def finalize(self, param, widget):
        self._close_crop_editor(param, widget)


# AI ノイズ除去
class AINoiseReductonEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_ai_noise_reduction', True, "switch_ai_noise_reduction"),
        SwitchBinding('ai_noise_reduction', False, "chip_ai_noise_reduction"),
        SliderBinding('ai_noise_reduction_intensity', 70, "slider_ai_noise_reduction_intensity"),
    )

    __net = None
    
    def __init__(self, ai_job_manager=None, **kwargs):
        super().__init__(**kwargs)
        self.execution_mode = ExecutionMode.ASYNC
        self.keep_async_result = False
        self.ai_job_manager = ai_job_manager

    def get_param_dict(self, param):
        param_dict = super().get_param_dict(param)
        param_dict['ai_noise_reduction_result'] = None
        return param_dict

    def make_diff(self, img, param, efconfig):
        switch_ai_noise_reduction = self._get_param(param, 'switch_ai_noise_reduction')
        nr = self._get_param(param, 'ai_noise_reduction')
        nr_intensity = self._get_param(param, 'ai_noise_reduction_intensity') 
        nr_result = self._get_param(param, 'ai_noise_reduction_result')         
        if switch_ai_noise_reduction == False or nr == False:
            ai_job_manager = self.ai_job_manager
            file_path = getattr(efconfig, "file_path", None)
            if ai_job_manager is not None and file_path:
                ai_job_manager.cancel_path(file_path)
            if efconfig.processor is not None:
                efconfig.processor.cancel_effect(self.__class__.__name__)

            self.diff = None
            self.hash = None
        else:
            if not heavy_ai_allowed(param):
                if efconfig.processor is not None:
                    efconfig.processor.cancel_effect(self.__class__.__name__)
                self.diff = None
                self.hash = None
                return None

            # 強度は render のみ。非同期キャッシュの param_hash には含めない（raw 再利用してブレンドのみ）
            param_hash_async = hash((nr,))
            ai_job_manager = self.ai_job_manager
            file_path = getattr(efconfig, "file_path", None)
            if file_path and param.get("image_fidelity") == ImageFidelity.PREVIEW.value:
                logging.info(
                    "AI-NR waiting for FULL_DECODE before inference: file=%s image_fidelity=%s shape=%s",
                    file_path,
                    param.get("image_fidelity"),
                    getattr(img, "shape", None),
                )
                if efconfig.layer_status is not None:
                    from enums import PipelineStatus
                    efconfig.layer_status = PipelineStatus.PREVIEW
                self.diff = None
                self.hash = None
                return None
            source_signature = None
            if file_path:
                from cores.ai_job_manager.ai_noise import (
                    ai_noise_content_key,
                    ai_noise_source_debug_info,
                    ai_noise_source_signature,
                    ai_noise_valid_content_keys,
                )

                source_signature = ai_noise_source_signature(file_path, img, param)
                content_key = ai_noise_content_key(file_path, img, param, source_signature=source_signature)
                valid_content_keys = ai_noise_valid_content_keys(file_path, img, param)
            else:
                content_key = _ai_noise_content_key(nr, efconfig.upstream_hash)
                valid_content_keys = {content_key}
            render_hash = hash((content_key, nr_intensity))

            logging.debug(
                "AINoiseReducton make_diff nr=%s upstream=%s content_key=%s render_hash=%s self.hash=%s",
                nr, efconfig.upstream_hash, content_key, render_hash, self.hash,
            )

            if self.hash == render_hash and self.diff is not None:
                return self.diff

            raw_stored = param.get("ai_noise_reduction_result")
            key_stored = param.get("ai_noise_reduction_content_key")

            # 保存済み raw + upstream 未変化なら AI NR／ワーカーを呼ばずブレンドのみ
            if isinstance(raw_stored, np.ndarray) and raw_stored.shape == img.shape:
                if key_stored is None or key_stored in valid_content_keys:
                    if key_stored is None:
                        param["ai_noise_reduction_content_key"] = content_key
                    blended = _ai_noise_blend_raw(raw_stored, img, nr_intensity)
                    if blended is not None:
                        logging.info("AI-NR reused stored raw result: file=%s content_key=%s", file_path, content_key)
                        self.diff = blended
                        self.hash = render_hash
                        return self.diff
                else:
                    logging.info(
                        "AI-NR discarded stored raw result due to source mismatch: file=%s stored_key=%s valid_keys=%s stored_source=%s current_source=%s raw_shape=%s image_shape=%s current_debug=%s",
                        file_path,
                        key_stored,
                        sorted(valid_content_keys),
                        param.get("ai_noise_reduction_source_signature"),
                        source_signature,
                        getattr(raw_stored, "shape", None),
                        getattr(img, "shape", None),
                        ai_noise_source_debug_info(file_path, img),
                    )
                    param.pop("ai_noise_reduction_result", None)
                    param.pop("ai_noise_reduction_content_key", None)
                    param.pop("ai_noise_reduction_source_signature", None)
            elif raw_stored is not None:
                logging.info(
                    "AI-NR discarded stored raw result due to shape mismatch: file=%s stored_shape=%s image_shape=%s",
                    file_path,
                    getattr(raw_stored, "shape", None),
                    getattr(img, "shape", None),
                )
                param.pop("ai_noise_reduction_result", None)
                param.pop("ai_noise_reduction_content_key", None)
                param.pop("ai_noise_reduction_source_signature", None)

            if param.get("_ai_noise_reduction_result_deferred"):
                if efconfig.layer_status is not None:
                    from enums import PipelineStatus
                    efconfig.layer_status = PipelineStatus.PREVIEW
                return None

            if ai_job_manager is not None and file_path and efconfig.mode != EffectMode.EXPORT:
                from cores.ai_job_manager import merge_ai_noise_result_into_param

                status, raw, content_key, source_signature = ai_job_manager.request_ai_noise(file_path, img, param)
                logging.info(
                    "AI-NR job manager request: file=%s status=%s content_key=%s has_raw=%s",
                    file_path,
                    getattr(status, "value", status),
                    content_key,
                    raw is not None,
                )
                if raw is not None:
                    raw = np.asarray(raw, dtype=np.float32)
                    merge_ai_noise_result_into_param(param, raw, content_key, source_signature)
                    blended = _ai_noise_blend_raw(raw, img, nr_intensity)
                    if blended is None:
                        self.diff = None
                        self.hash = None
                        return None
                    self.diff = blended
                    self.hash = render_hash
                    return self.diff

                if efconfig.layer_status is not None:
                    from enums import PipelineStatus
                    efconfig.layer_status = PipelineStatus.PREVIEW
                self.hash = None
                if nr_result is not None and nr_result.shape == img.shape:
                    blended = _ai_noise_blend_raw(nr_result, img, nr_intensity)
                    if blended is not None:
                        self.diff = blended
                        self.hash = hash((render_hash, "ai_job_preview", str(status)))
                        return self.diff
                return None

            handled, result = self.try_async_execution(img, param, efconfig, param_hash_async)
            if handled:
                logging.debug(
                    "AINoiseReducton try_async handled result=%s combined=%s",
                    id(result) if result is not None else None,
                    hash((param_hash_async, efconfig.upstream_hash)),
                )
                if result is not None:
                    raw = np.asarray(result, dtype=np.float32)
                    param["ai_noise_reduction_result"] = raw
                    param["ai_noise_reduction_content_key"] = content_key
                    blended = _ai_noise_blend_raw(raw, img, nr_intensity)
                    if blended is None:
                        self.diff = None
                        self.hash = None
                        return None
                    self.diff = blended
                    self.hash = render_hash
                    return self.diff

                if nr_result is not None and nr_result.shape == img.shape:
                    alpha = nr_intensity / 100.0
                    if alpha <= 0.0:
                        self.diff = img
                    elif alpha >= 1.0:
                        self.diff = nr_result
                    else:
                        self.diff = cv2.addWeighted(nr_result, alpha, img, 1.0 - alpha, 0.0)
                    self.hash = hash((render_hash, "preview"))
                    return self.diff

                return None

            # 同期（processor なし）：上で raw 無効ならここへ。再計算 or 残り
            raw_diff = param.get("ai_noise_reduction_result")
            if (
                isinstance(raw_diff, np.ndarray)
                and raw_diff.shape == img.shape
                and param.get("ai_noise_reduction_content_key") in valid_content_keys
            ):
                blended = _ai_noise_blend_raw(raw_diff, img, nr_intensity)
                if blended is not None:
                    self.diff = blended
                    self.hash = render_hash
                    return self.diff

            import helpers.scunet_coreml_helper as scunet_helper

            if AINoiseReductonEffect.__net is None:
                AINoiseReductonEffect.__net = scunet_helper.setup()

            raw_diff = scunet_helper.predict_helper(AINoiseReductonEffect.__net, img)
            param["ai_noise_reduction_result"] = raw_diff
            param["ai_noise_reduction_content_key"] = content_key
            blended = _ai_noise_blend_raw(raw_diff, img, nr_intensity)
            if blended is None:
                self.diff = None
                self.hash = None
            else:
                self.diff = blended
                self.hash = render_hash

        return self.diff


class LightNoiseReductionEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_light_noise_reduction', True, "switch_light_noise_reduction"),
        SliderBinding('light_noise_reduction', 0, "slider_light_noise_reduction"),
        SliderBinding('light_color_noise_reduction', 0, "slider_light_color_noise_reduction"),
    )

    def make_diff(self, img, param, efconfig):
        switch_light_noise_reduction = self._get_param(param, 'switch_light_noise_reduction')
        its = int(self._get_param(param, 'light_noise_reduction'))
        col = int(self._get_param(param, 'light_color_noise_reduction'))
        if switch_light_noise_reduction == False or its == 0 and col == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((its, col))
            if self.hash != param_hash:  
                self.hash = param_hash
                from radiance_denoise.native import denoise_native

                self.diff = denoise_native(img, its * efconfig.disp_info[4], col * efconfig.disp_info[4])

        return self.diff

class LensblurFilterEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_lens_simulator', True, "switch_lens_simulator"),
        SliderBinding('lensblur_filter', 0, "slider_lensblur_filter"),
        SliderBinding('lens_focus_depth', 0.5, "slider_lens_focus_depth"),
        SwitchBinding('switch_lensblur_use_depthmap', False, "switch_lensblur_use_depthmap")
    )

    def make_diff(self, img, param, efconfig):
        switch_ls = self._get_param(param, 'switch_lens_simulator')
        lpfr = int(self._get_param(param, 'lensblur_filter'))
        focus_d = float(self._get_param(param, 'lens_focus_depth'))
        use_depthmap = self._get_param(param, 'switch_lensblur_use_depthmap')
        if switch_ls == False or lpfr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((lpfr, focus_d, use_depthmap))
            if self.hash != param_hash:
                self.hash = param_hash

                depth_map = None
                if use_depthmap:
                    getter = getattr(efconfig, "get_ai_depth_map", None)
                    if callable(getter):
                        try:
                            depth_map = getter(space="current", allow_compute=False)
                            w, h = img.shape[1], img.shape[0]
                            if depth_map.ndim == 3:
                                depth_map = depth_map[..., 0]
                            if depth_map.shape[:2] != (h, w):
                                depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

                        except Exception:
                            logging.exception("LensblurFilter: AI depth map の取得に失敗")
                            depth_map = None
                self.diff = lens_blur_adapter.apply_lensblur(img, depth_map, focus_d, int(round(lpfr-1) * 4 * efconfig.resolution_scale))

                #self.diff = filters.lensblur_filter(img, int(round(lpfr-1) * 4 * efconfig.resolution_scale))

        return self.diff

class ScratchEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_filters', True, "switch_filters"),
        SliderBinding('scratch', 0, "slider_scratch"),
    )

    def make_diff(self, img, param, efconfig):
        switch_filters = self._get_param(param, 'switch_filters')
        fr = int(self._get_param(param, 'scratch'))
        if switch_filters == False or fr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((fr))
            if self.hash != param_hash:
                self.hash = param_hash

                # shift_parcent(最終ブラー)は従来通り resolution_scale 込み。
                # 傷サイズ/本数の解像度整合は scratch_effect 側に resolution_scale を渡して行う。
                self.diff = filters.scratch_effect(img, 1.0, fr / 100 * efconfig.resolution_scale, resolution_scale=efconfig.resolution_scale)

        return self.diff

class FrostedGlassEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_filters', True, "switch_filters"),
        SliderBinding('frosted_glass', 0, "slider_frosted_glass"),
    )

    def make_diff(self, img, param, efconfig):
        switch_filters = self._get_param(param, 'switch_filters')
        fr = int(self._get_param(param, 'frosted_glass'))
        if switch_filters == False or fr == 0:
            self.diff = None
            self.hash = None

        else:
            param_hash = hash((fr))
            if self.hash != param_hash:
                self.hash = param_hash

                # noise_scale は内部で *w するため画像比は固定。resolution_scale を掛けると
                # 二重スケールになり preview/export でにじみ量がズレるため blur 側だけに掛ける。
                self.diff = filters.frosted_glass_effect(img, fr / 100 * efconfig.resolution_scale, fr / 1000)

        return self.diff

class MosaicEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_filters', True, "switch_filters"),
        SliderBinding('mosaic', 0, "slider_mosaic"),
    )

    def make_diff(self, img, param, efconfig):
        switch_filters = self._get_param(param, 'switch_filters')
        fr = int(self._get_param(param, 'mosaic'))
        if switch_filters == False or fr == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((fr))
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = filters.mosaic_effect(img, int(fr * efconfig.resolution_scale))

        return self.diff

class OrtonEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_orton_effect', True, "switch_orton_effect"),
        SliderBinding('orton_radius', 30, "slider_orton_radius"),
        SliderBinding('orton_opacity', 75, "slider_orton_opacity"),
        SliderBinding('orton_intensity', 0, "slider_orton_intensity"),
    )

    def make_diff(self, img, param, efconfig):
        switch_orton_effect = self._get_param(param, 'switch_orton_effect')
        oradius = int(self._get_param(param, 'orton_radius'))
        oopacity = int(self._get_param(param, 'orton_opacity'))
        ointensity = int(self._get_param(param, 'orton_intensity'))
        if switch_orton_effect == False or ointensity == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((oradius, oopacity, ointensity))
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = filters.orton_effect(img, oradius * efconfig.disp_info[4], oopacity / 100, ointensity / 100)

        return self.diff

class GlowEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_glow_effect', True, "switch_glow_effect"),
        SliderBinding('glow_black', 0, "slider_glow_black"),
        SliderBinding('glow_gauss', 0, "slider_glow_gauss"),
        SliderBinding('glow_opacity', 0, "slider_glow_opacity"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_glow_effect = self._get_param(param, 'switch_glow_effect')
        gb = self._get_param(param, 'glow_black')
        gg = int(self._get_param(param, 'glow_gauss'))
        go = self._get_param(param, 'glow_opacity')
        if switch_glow_effect == False or go <= 0 or (gb == 0 and gg == 0):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((gb, gg, go))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                hls = hlsrgb.rgb_to_hlc_gain(rgb)
                # グロー源は実際の明るさ=gain(ch3)を基準に作る。L(ch1=正規化輝度)を弄ると暗部/明部を
                # 正しく選べず彩度も漏れるため、L は保持し gain に level 補正する。
                # apply_level_adjustment は HDR(>1)を線形保持するので gain に適用して問題ない。
                hls[:,:,3] = core.apply_level_adjustment(hls[:,:,3], gb, 127+gg/2, 255)
                rgb2 = hlsrgb.hlc_gain_to_rgb(hls)
                if gg > 0:
                    radius = gg * 10 * efconfig.resolution_scale
                    rgb2 = filters.lensblur_filter(rgb2, 1 if radius <= 0 else radius) 
                go = go/100.0
                self.diff = cv2.addWeighted(rgb, 1.0-go, core.blend_screen(rgb, rgb2), go, 0)

        return self.diff

class CleanHighlightEffect(Effect):
    """ハイライトの色被りを抜いて白を白へ寄せる（爽やか・クリア方向）。

    輝度が閾値を超えた領域ほど彩度を中立（=その画素の輝度）へ寄せる。
    中間調・暗部の色は保持し、ハイライトだけをクリーンな白にする。
    Glow/Film と違い情報を足さず「白の純度」を上げるための補正。
    """
    param_bindings = (
        SwitchBinding('switch_clean_highlight', True, "switch_clean_highlight"),
        SliderBinding('clean_highlight', 0, "slider_clean_highlight"),
        SliderBinding('clean_highlight_threshold', 60, "slider_clean_highlight_threshold"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_clean_highlight = self._get_param(param, 'switch_clean_highlight')
        amount = self._get_param(param, 'clean_highlight')
        threshold = self._get_param(param, 'clean_highlight_threshold')
        if switch_clean_highlight == False or amount == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((amount, threshold))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                y = core.cvtColorRGB2Gray(rgb)
                # 閾値→白へ滑らかに立ち上がる重み(smoothstep)。
                t = np.float32(threshold / 100.0)
                denom = np.float32(max(1.0e-4, 1.0 - t))
                x = np.clip((y - t) / denom, 0.0, 1.0).astype(np.float32)
                w = x * x * (np.float32(3.0) - np.float32(2.0) * x)
                w = (w * np.float32(amount / 100.0))[..., np.newaxis]
                # 画素ごとの輝度へ寄せる＝輝度を保ったままハイライトの彩度だけ抜く。
                gray = y[..., np.newaxis]
                self.diff = rgb * (np.float32(1.0) - w) + gray * w

        return self.diff

class AiryGlowEffect(Effect):
    """ハイライトだけを抽出してぼかし、スクリーンで戻す澄んだブルーム。

    Glow と違い暗部を持ち上げないため、コントラストと解像感を保ったまま
    光に空気感(エア感)を足す。爽やか方向のハイライト・ブルーム。
    """
    param_bindings = (
        SwitchBinding('switch_airy_glow', True, "switch_airy_glow"),
        SliderBinding('airy_glow', 0, "slider_airy_glow"),
        SliderBinding('airy_glow_radius', 40, "slider_airy_glow_radius"),
        SliderBinding('airy_glow_threshold', 65, "slider_airy_glow_threshold"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_airy_glow = self._get_param(param, 'switch_airy_glow')
        amount = self._get_param(param, 'airy_glow')
        radius = self._get_param(param, 'airy_glow_radius')
        threshold = self._get_param(param, 'airy_glow_threshold')
        if switch_airy_glow == False or amount == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((amount, radius, threshold))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                y = core.cvtColorRGB2Gray(rgb)
                # ハイライトだけを smoothstep で抽出（暗部は巻き込まない）。
                t = np.float32(threshold / 100.0)
                denom = np.float32(max(1.0e-4, 1.0 - t))
                x = np.clip((y - t) / denom, 0.0, 1.0).astype(np.float32)
                mask = (x * x * (np.float32(3.0) - np.float32(2.0) * x))[..., np.newaxis]
                highlights = (rgb * mask).astype(np.float32)
                # プレビュー/フルで見えを揃えるため resolution_scale で半径をスケール。
                sigma = max(0.5, float(radius) * 0.5 * float(efconfig.resolution_scale))
                bloom = cv2.GaussianBlur(highlights, (0, 0), sigma)
                op = np.float32(amount / 100.0)
                # HDR でも破綻しない screen 合成で光だけを足す。
                self.diff = core.blend_screen(rgb, bloom * op)

        return self.diff

class FaceEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_face', True, "switch_face"),
        SliderBinding('jawline_scale', 0, "slider_jawline_scale"),
        SliderBinding('jaw_scale', 0, "slider_jaw_scale"),
        SliderBinding('left_eye_scale', 0, "slider_left_eye_scale"),
        SliderBinding('right_eye_scale', 0, "slider_right_eye_scale"),
        SliderBinding('lips_scale', 0, "slider_lips_scale"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_face = self._get_param(param, 'switch_face')
        jls = self._get_param(param, 'jawline_scale')
        js = self._get_param(param, 'jaw_scale')
        ls = self._get_param(param, 'left_eye_scale')
        rs = self._get_param(param, 'right_eye_scale')
        lipss = self._get_param(param, 'lips_scale')
        if switch_face == False or (ls == 0 and rs == 0 and jls == 0 and js == 0 and lipss == 0):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((jls, js, ls, rs, lipss))
            if self.hash != param_hash:
                self.hash = param_hash

                import helpers.mediapipe_helper
                fms = helpers.mediapipe_helper.setup_face_mesh(rgb)
                rgb = helpers.mediapipe_helper.adjust_face_jawline(fms, rgb, jls/100, False) #efconfig.mode == EffectMode.PREVIEW)
                rgb = helpers.mediapipe_helper.adjust_face_jaw(fms, rgb, js/100, False)
                rgb = helpers.mediapipe_helper.adjust_left_eye(fms, rgb, ls/100, False)
                rgb = helpers.mediapipe_helper.adjust_right_eye(fms, rgb, rs/100, False)
                rgb = helpers.mediapipe_helper.adjust_lips(fms, rgb, lipss/100, False)
                helpers.mediapipe_helper.clear_face_mesh(fms)
                self.diff = rgb

        return self.diff

class ColorTemperatureEffect(Effect):
    # get_param_dict のデフォルトが param 依存（color_temperature_reset 等）のためキャッシュ無効化
    _defaults_depend_on_param = True
    PRESET_AS_SHOT = "As Shot"
    PRESET_CUSTOM = "Custom"
    PRESET_VALUES = {
        "Daylight": (5500, 0),
        "Cloudy": (6500, 0),
        "Shade": (7500, 0),
        "Tungsten": (2850, 0),
        "Fluorescent": (4000, 20),
        "Flash": (6000, 0),
    }
    PRESET_OPTIONS = (
        PRESET_AS_SHOT,
        "Daylight",
        "Cloudy",
        "Shade",
        "Tungsten",
        "Fluorescent",
        "Flash",
        PRESET_CUSTOM,
    )

    @classmethod
    def preset_options(cls):
        return list(cls.PRESET_OPTIONS)

    @classmethod
    def preset_values(cls, preset, param):
        if preset == cls.PRESET_AS_SHOT:
            return (
                param.get('color_temperature_reset', 5000),
                param.get('color_tint_reset', 0),
            )
        return cls.PRESET_VALUES.get(preset)

    @classmethod
    def infer_preset(cls, param):
        preset = param.get('color_temperature_preset')
        if preset in cls.PRESET_OPTIONS:
            return preset
        temp = param.get('color_temperature', param.get('color_temperature_reset', 5000))
        tint = param.get('color_tint', param.get('color_tint_reset', 0))
        reset_temp = param.get('color_temperature_reset', 5000)
        reset_tint = param.get('color_tint_reset', 0)
        if abs(float(temp) - float(reset_temp)) <= 1.0e-6 and abs(float(tint) - float(reset_tint)) <= 1.0e-6:
            return cls.PRESET_AS_SHOT
        return cls.PRESET_CUSTOM

    def get_param_dict(self, param):
        return {
            'switch_white_balance': True,
            'color_temperature_preset': self.PRESET_AS_SHOT,
            'color_temperature_reset': 5000,
            'color_temperature': param.get('color_temperature_reset', 5000),
            'color_tint_reset': 0,
            'color_tint': param.get('color_tint_reset', 0),
            'color_Y': 1.0,
        }

    def set2widget(self, widget, param):
        widget.ids['switch_white_balance'].enabled = self._get_param(param, 'switch_white_balance')
        widget.ids["spinner_color_temperature_preset"].values = self.preset_options()
        widget.ids["spinner_color_temperature_preset"].set_text(
            self.infer_preset(param)
        )
        widget.ids["slider_color_temperature"].set_slider_value(self._get_param(param, 'color_temperature'))
        widget.ids["slider_color_tint"].set_slider_value(self._get_param(param, 'color_tint'))
        self._set_bar_context(widget, param)
        widget.ids["slider_color_temperature"].set_slider_reset(self._get_param(param, 'color_temperature_reset'))
        widget.ids["slider_color_tint"].set_slider_reset(self._get_param(param, 'color_tint_reset'))
 
    def set2param(self, param, widget):
        param['switch_white_balance'] = widget.ids['switch_white_balance'].enabled
        preset = widget.ids["spinner_color_temperature_preset"].text or self.PRESET_AS_SHOT
        if preset not in self.PRESET_OPTIONS:
            preset = self.PRESET_CUSTOM
        if preset == self.PRESET_AS_SHOT:
            values = (
                widget.ids["slider_color_temperature"].reset_value,
                widget.ids["slider_color_tint"].reset_value,
            )
        else:
            values = self.preset_values(preset, param)
        if values is not None:
            param['color_temperature'], param['color_tint'] = values
        else:
            param['color_temperature'] = widget.ids["slider_color_temperature"].value
            param['color_tint'] = widget.ids["slider_color_tint"].value
        param['color_temperature_preset'] = preset

    def _set_bar_context(self, widget, param):
        reset_temp = self._get_param(param, 'color_temperature_reset')
        reset_tint = self._get_param(param, 'color_tint_reset')
        y = self._get_param(param, 'color_Y')
        widget.ids["slider_color_temperature"].set_bar_context({
            "reset_temp": reset_temp,
            "reset_tint": reset_tint,
            "fixed_tint": reset_tint,
            "Y": y,
        })
        widget.ids["slider_color_tint"].set_bar_context({
            "reset_temp": reset_temp,
            "reset_tint": reset_tint,
            "fixed_temp": reset_temp,
            "Y": y,
        })

    @staticmethod
    def apply_color_temperature(rgb, param):
        temp = param.get('color_temperature', param.get('color_temperature_reset', 5000))
        tint = param.get('color_tint', param.get('color_tint_reset', 0))
        Y = param.get('color_Y', 1.0)
        return rgb * core.invert_TempTint2RGB(temp, tint, Y, 5000)

    def make_diff(self, rgb, param, efconfig):
        switch_white_balance = self._get_param(param, 'switch_white_balance')
        temp = self._get_param(param, 'color_temperature')
        tint = self._get_param(param, 'color_tint')
        Y = self._get_param(param, 'color_Y')
        if switch_white_balance == False:
            self.diff = None
            self.hash = None
        else:
            # Y(色温度変換の輝度基準)と基準白色点(reset)も出力に効くのでハッシュに含める。
            # 含めないと temp/tint を変えずに Y・基準白色点だけ変えたとき結果が更新されない。
            param_hash = hash((temp, tint, Y,
                               param.get('color_temperature_reset'),
                               param.get('color_tint_reset')))
            if self.hash != param_hash:
                trgb = core.convert_TempTint2RGB(param['color_temperature_reset'], param['color_tint_reset'], self._get_param(param, 'color_Y'))
                self.diff = rgb * (trgb / core.convert_TempTint2RGB(temp, tint, Y))
                self.hash = param_hash

        return self.diff

class DehazeEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_precence', True, "switch_precence", widget_attr="enabled"),
        SliderBinding('dehaze', 0, "slider_dehaze"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_precence = self._get_param(param, 'switch_precence')
        de = self._get_param(param, 'dehaze')
        if switch_precence == False or de == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((de))
            if self.hash != param_hash:
                self.hash = param_hash

                if de > 0:
                    de = de / 400 # 効果を1/4に
                else:
                    de = de / 100
                self.diff = dehaze_adapter.dehaze_image(rgb, de)

        return self.diff

class RGB2HLSEffect(Effect):

    @staticmethod
    def _hls_pipeline_active(param):
        color_names = ("red", "skin", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta")
        if param.get("switch_color_mixer", True):
            for color_name in color_names:
                if not param.get(f"switch_hls_{color_name}", True):
                    continue
                if (param.get(f"hls_{color_name}_hue", 0) != 0
                        or param.get(f"hls_{color_name}_lum", 0) != 0
                        or param.get(f"hls_{color_name}_sat", 0) != 0):
                    return True
        if param.get("switch_saturation", True) and (
                param.get("saturation", 0) != 0 or param.get("vibrance", 0) != 0):
            return True
        # vs カーブ（HuevsHue/SatvsSat 等）も HLC 変換が必要。これを判定に含めないと、
        # vs カーブ単独使用時に RGB→HLC 変換がスキップされ、vs が生の R/G/B チャンネルを
        # 処理してしまう（色相が変わる/効かない等の不具合になる）。
        if param.get("switch_color_curves", True):
            for curve_name in ("HuevsHue", "HuevsLum", "HuevsSat",
                               "LumvsLum", "LumvsSat", "SatvsLum", "SatvsSat"):
                if param.get(curve_name) is not None:
                    return True
        return False

    def make_diff(self, rgb, param, efconfig):
        if not self._hls_pipeline_active(param):
            self.diff = None
            self.hash = None
            return self.diff
        if self.diff is None:
            rgb = core.type_convert(rgb, np.ndarray)
            self.diff = hlsrgb.rgb_to_hlc_gain(rgb)
        return self.diff

class HLS2RGBEffect(Effect):

    def make_diff(self, hls, param, efconfig):
        if getattr(hls, "ndim", 0) < 3 or hls.shape[2] < 4:
            self.diff = None
            self.hash = None
            return self.diff
        if self.diff is None:
            hls = core.type_convert(hls, np.ndarray)
            self.diff = hlsrgb.hlc_gain_to_rgb(hls)
        return self.diff

    
class HLSEffect(Effect):
    HLS_COLORS = ("red", "skin", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta")
    FULL_RANGE_HUE_STEP = 0.1
    LOCAL_RANGE_HUE_STEP = 0.01
    HUE_RANGE_DEFAULT = (100.0, 100.0)
    HUE_RANGE_MIN = 0.0
    HUE_RANGE_MAX = 200.0

    @staticmethod
    def _circular_delta(target_hue, source_hue):
        return ((target_hue - source_hue + 180.0) % 360.0) - 180.0

    @classmethod
    def _hue_slider_range(cls, color_name, full_range):
        if full_range:
            return -180.0, 180.0, cls.FULL_RANGE_HUE_STEP

        color_index = cls.HLS_COLORS.index(color_name)
        prev_color = cls.HLS_COLORS[color_index - 1]
        next_color = cls.HLS_COLORS[(color_index + 1) % len(cls.HLS_COLORS)]
        center = core.HLS_COLOR_SETTING[color_name]["center"]
        min_hue = cls._circular_delta(core.HLS_COLOR_SETTING[prev_color]["center"], center)
        max_hue = cls._circular_delta(core.HLS_COLOR_SETTING[next_color]["center"], center)
        return min_hue, max_hue, cls.LOCAL_RANGE_HUE_STEP

    @classmethod
    def _set_hue_slider_range(cls, widget, color_name, full_range):
        min_hue, max_hue, step = cls._hue_slider_range(color_name, full_range)
        slider = widget.ids[f"slider_hls_{color_name}_hue"]
        setting = core.HLS_COLOR_SETTING[color_name]
        slider.bar_renderer = "hls_hue_shift"
        slider.bar_show_active_overlay = False
        slider.bar_show_anchor_marker = True
        slider.set_bar_context({
            "center": setting["center"],
            "l_range": setting["l_range"],
            "s_range": setting["s_range"],
        })
        slider.set_slider_range(min_hue, max_hue, step)

    @classmethod
    def _normalize_hue_range(cls, value):
        try:
            left, right = value[0], value[1]
        except (TypeError, IndexError):
            left, right = cls.HUE_RANGE_DEFAULT
        left = min(cls.HUE_RANGE_MAX, max(cls.HUE_RANGE_MIN, float(left)))
        right = min(cls.HUE_RANGE_MAX, max(cls.HUE_RANGE_MIN, float(right)))
        return left, right

    @staticmethod
    def _scale_hue_range_pair(value, left_scale, right_scale):
        if np.isscalar(value):
            left_value = right_value = float(value)
        else:
            left_value, right_value = float(value[0]), float(value[1])
        return [left_value * left_scale, right_value * right_scale]

    @classmethod
    def _setting_with_hue_range(cls, setting, hue_range):
        left, right = cls._normalize_hue_range(hue_range)
        left_scale = left / 100.0
        right_scale = right / 100.0
        scaled = setting.copy()
        scaled["width"] = cls._scale_hue_range_pair(setting["width"], left_scale, right_scale)
        scaled["fade_width"] = cls._scale_hue_range_pair(setting["fade_width"], left_scale, right_scale)
        return scaled

    @staticmethod
    def _slider_values(slider_widget):
        slider = slider_widget.ids.get("slider") if hasattr(slider_widget, "ids") else None
        values = list(getattr(slider, "values", []) or [])
        if len(values) >= 2:
            return values[0], values[1]
        value = getattr(slider_widget, "value", 100)
        return value, value

    def get_param_dict(self, param, subname=None):
        param_dict = {
            "switch_color_mixer": True,
        }
        for color_name in self.HLS_COLORS:
            param_dict[f"switch_hls_{color_name}"] = True
            param_dict[f"hls_{color_name}_hue"] = 0
            param_dict[f"hls_{color_name}_lum"] = 0
            param_dict[f"hls_{color_name}_sat"] = 0
            param_dict[f"hls_{color_name}_hue_full_range"] = False
            param_dict[f"hls_{color_name}_hue_range"] = list(self.HUE_RANGE_DEFAULT)
        if subname in self.HLS_COLORS:
            return {
                key: param_dict[key]
                for key in (
                    f"switch_hls_{subname}",
                    f"hls_{subname}_hue",
                    f"hls_{subname}_lum",
                    f"hls_{subname}_sat",
                    f"hls_{subname}_hue_full_range",
                    f"hls_{subname}_hue_range",
                )
            }
        return param_dict

    def set2widget(self, widget, param):
        widget.ids["switch_color_mixer"].active = self._get_param(param, "switch_color_mixer")
        for color_name in self.HLS_COLORS:
            full_range = self._get_param(param, f"hls_{color_name}_hue_full_range")
            widget.ids[f"switch_hls_{color_name}"].active = self._get_param(param, f"switch_hls_{color_name}")
            widget.ids[f"checkbox_hls_{color_name}_hue_full_range"].active = full_range
            widget.ids[f"slider_hls_{color_name}_hue"].set_slider_value(self._get_param(param, f"hls_{color_name}_hue"))
            self._set_hue_slider_range(widget, color_name, full_range)
            widget.ids[f"slider_hls_{color_name}_hue_range"].set_slider_value(
                list(self._normalize_hue_range(self._get_param(param, f"hls_{color_name}_hue_range")))
            )
            widget.ids[f"slider_hls_{color_name}_lum"].set_slider_value(self._get_param(param, f"hls_{color_name}_lum"))
            widget.ids[f"slider_hls_{color_name}_sat"].set_slider_value(self._get_param(param, f"hls_{color_name}_sat"))

    def set2param(self, param, widget):
        param["switch_color_mixer"] = widget.ids["switch_color_mixer"].active
        for color_name in self.HLS_COLORS:
            full_range = widget.ids[f"checkbox_hls_{color_name}_hue_full_range"].active
            self._set_hue_slider_range(widget, color_name, full_range)
            param[f"switch_hls_{color_name}"] = widget.ids[f"switch_hls_{color_name}"].active
            param[f"hls_{color_name}_hue_full_range"] = full_range
            param[f"hls_{color_name}_hue"] = widget.ids[f"slider_hls_{color_name}_hue"].value
            param[f"hls_{color_name}_hue_range"] = list(self._normalize_hue_range(
                self._slider_values(widget.ids[f"slider_hls_{color_name}_hue_range"])
            ))
            param[f"hls_{color_name}_lum"] = widget.ids[f"slider_hls_{color_name}_lum"].value
            param[f"hls_{color_name}_sat"] = widget.ids[f"slider_hls_{color_name}_sat"].value

    def make_diff(self, hls, param, efconfig):
        switch_color_mixer = self._get_param(param, "switch_color_mixer")
        params_map = {}
        for color_name in self.HLS_COLORS:
            params_map[color_name] = (
                self._get_param(param, f"switch_hls_{color_name}"),
                self._get_param(param, f"hls_{color_name}_hue"),
                self._get_param(param, f"hls_{color_name}_lum"),
                self._get_param(param, f"hls_{color_name}_sat"),
                self._normalize_hue_range(self._get_param(param, f"hls_{color_name}_hue_range")),
            )

        if (   switch_color_mixer == False
            or all((not switch) or (h == 0 and l == 0 and s == 0) for switch, h, l, s, _hue_range in params_map.values())):
            self.diff = None
            self.hash = None        
        else:
            param_hash = hash(tuple(params_map.items()))
            if self.hash != param_hash:
                self.hash = param_hash

                color_settings = []
                for color_name in self.HLS_COLORS:
                    switch, h, l, s, hue_range = params_map[color_name]
                    
                    if not switch:
                        continue
                    
                    if h == 0 and l == 0 and s == 0:
                        continue
                        
                    if color_name in core.HLS_COLOR_SETTING:
                        setting = self._setting_with_hue_range(core.HLS_COLOR_SETTING[color_name], hue_range)
                        setting['adjust'] = [h, l/100.0, s/100.0]
                        color_settings.append(setting)

                if not color_settings:
                     self.diff = None
                else:
                    self.diff = hls_adjust_adapter.adjust_hls_colors(hls, color_settings, efconfig.resolution_scale)

        return self.diff

class ExposureEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_exposure_contrast', True, "switch_exposure_contrast", widget_attr="enabled"),
        SliderBinding('exposure', 0, "slider_exposure"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_exposure_contrast = self._get_param(param, 'switch_exposure_contrast')
        ev = self._get_param(param, 'exposure')
        if switch_exposure_contrast == False or ev == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((ev))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                self.diff = core.adjust_exposure(rgb, ev)
                #self.diff = core.boost_detail_from_tone_change(rgb, self.diff, detail_strength=1.2, max_comp_stops=2.0)

        return self.diff
    
class ContrastEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_exposure_contrast', True, "switch_exposure_contrast", widget_attr="enabled"),
        SliderBinding('contrast', 0, "slider_contrast"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_exposure_contrast = self._get_param(param, 'switch_exposure_contrast')
        con = self._get_param(param, 'contrast')
        if switch_exposure_contrast == False or con == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((con))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                if con > 0:
                    con *= 0.5
                self.diff = core.adjust_luminance_contrast(rgb, con)

        return self.diff

class ClarityEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_precence', True, "switch_precence", widget_attr="enabled"),
        SliderBinding('clarity', 0, "slider_clarity"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_precence = self._get_param(param, 'switch_precence')
        con = self._get_param(param, 'clarity')
        if switch_precence == False or con == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((con))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                # 半径は入力解像度で相対化されるため、強度はズーム非依存の一定値にする。
                self.diff = local_contrast.apply_clarity(rgb, con / 100 if con > 0 else con / 200)

        return self.diff

class TextureEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_precence', True, "switch_precence", widget_attr="enabled"),
        SliderBinding('texture', 0, "slider_texture"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_precence = self._get_param(param, 'switch_precence')
        con = self._get_param(param, 'texture')
        if switch_precence == False or con == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((con))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                # 半径は入力解像度で相対化されるため、強度はズーム非依存の一定値にする。
                self.diff = local_contrast.apply_texture(rgb, con / 200)

        return self.diff
    
class MicroContrastEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_precence', True, "switch_precence", widget_attr="enabled"),
        SliderBinding('microcontrast', 0, "slider_microcontrast"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_precence = self._get_param(param, 'switch_precence')
        con = self._get_param(param, 'microcontrast')
        if switch_precence == False or con == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((con))
            if self.hash != param_hash:
                self.hash = param_hash
                
                rgb = core.type_convert(rgb, np.ndarray)
                # 半径は入力解像度で相対化されるため、強度はズーム非依存の一定値にする。
                self.diff = local_contrast.apply_microcontrast(rgb, (con * 0.5) / 100)

        return self.diff
    
class ToneEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_tone', True, "switch_tone"),
        SliderBinding('shadow', 0, "slider_shadow"),
        SliderBinding('highlight', 0, "slider_highlight"),
        SliderBinding('midtone', 0, "slider_midtone"),
        SliderBinding('white', 0, "slider_white"),
        SliderBinding('black', 0, "slider_black"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_tone = self._get_param(param, 'switch_tone')
        shadow = self._get_param(param, 'shadow')
        highlight = self._get_param(param, 'highlight')
        mt = self._get_param(param, 'midtone')
        white = self._get_param(param, 'white')
        black = self._get_param(param, 'black')
        if switch_tone == False or (shadow == 0 and highlight == 0 and mt == 0 and white == 0 and black == 0):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((shadow, highlight, mt, white, black))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                self.diff = tone_adapter.adjust_tone(rgb, highlight, shadow, mt, white, black, disp_scale=efconfig.disp_info[4], resolution_scale=efconfig.resolution_scale)

        return self.diff
    
class ColorSeparationEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_global', True, "switch_global"),
        SliderBinding('shadow_chroma_clean', 0.0, "slider_shadow_chroma_clean"),
        SliderBinding('shadow_chroma_threshold', 0.2, "slider_shadow_chroma_threshold"),
        SliderBinding('color_separation', 0.0, "slider_color_separation"),
        SliderBinding('chroma_clarity', 0.0, "slider_chroma_clarity"),
        SliderBinding('color_density', 0.0, "slider_color_density"),
        SliderBinding('subtractive_saturation', 0.0, "slider_subtractive_saturation"),
        SliderBinding('detail_tonemap', 0.0, "slider_detail_tonemap"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_global = self._get_param(param, 'switch_global')
        shadow_clean = self._get_param(param, 'shadow_chroma_clean')
        threshold = self._get_param(param, 'shadow_chroma_threshold')
        separation = self._get_param(param, 'color_separation')
        clarity = self._get_param(param, 'chroma_clarity')
        density = self._get_param(param, 'color_density')
        subtractive_saturation = self._get_param(param, 'subtractive_saturation')
        detail_tonemap = self._get_param(param, 'detail_tonemap')
        if switch_global == False or (shadow_clean == 0.0 and separation == 0.0
                                      and clarity == 0.0 and density == 0.0
                                      and subtractive_saturation == 0.0
                                      and detail_tonemap == 0.0):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((
                shadow_clean,
                threshold,
                separation,
                clarity,
                density,
                subtractive_saturation,
                detail_tonemap,
            ))
            needed, combined_hash = self.check_sync_necessity(param_hash, efconfig)
            if needed:
                rgb = core.type_convert(rgb, np.ndarray)
                shadow_clean_core = shadow_clean / 100.0
                separation_core = separation / 100.0
                clarity_core = clarity / 100.0
                density_core = density / 100.0
                subtractive_saturation_core = subtractive_saturation / 100.0
                detail_tonemap_core = detail_tonemap / 100.0
                if (shadow_clean == 0.0 and separation == 0.0 and clarity == 0.0
                        and density == 0.0 and subtractive_saturation == 0.0):
                    out = rgb
                else:
                    out = color_separation.apply_color_separation(
                        rgb,
                        shadow_chroma_clean=shadow_clean_core,
                        shadow_threshold=threshold,
                        color_separation=separation_core,
                        chroma_clarity=clarity_core,
                        color_density=density_core,
                        subtractive_saturation=subtractive_saturation_core,
                    )
                if detail_tonemap != 0.0:
                    out = core.detail_preserving_tonemap(out, detail_tonemap_core)
                self.diff = out
                self.hash = combined_hash

        return self.diff


class LevelEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_level', True, "switch_level"),
        SliderBinding('black_level', 0, "slider_black_level"),
        SliderBinding('mid_level', 127, "slider_mid_level"),
        SliderBinding('white_level', 255, "slider_white_level"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_level = self._get_param(param, 'switch_level')
        bl = self._get_param(param, 'black_level')
        ml = self._get_param(param, 'mid_level')
        wl = self._get_param(param, 'white_level')
        if switch_level == False or (bl == 0 and wl == 255 and ml == 127):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((bl, ml, wl))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                self.diff = core.apply_level_adjustment(rgb, bl, ml, wl)

        return self.diff
    
class CLAHEEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_precence', True, "switch_precence"),
        SliderBinding('clahe', 0, "slider_clahe"),
    )

    def make_diff(self, img, param, efconfig):
        switch_precence = self._get_param(param, 'switch_precence')
        ci = self._get_param(param, 'clahe')
        if switch_precence == False or ci == 0:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((ci))
            if self.hash != param_hash:
                self.hash = param_hash

                img = core.type_convert(img, np.ndarray)
                img_min = float(img.min())
                img_max = float(img.max())
                img_range = img_max - img_min
                if img_range < 1e-8:
                    # 完全に平坦な画像は CLAHE で変化しない（0除算も防ぐ）。無変化扱い。
                    self.diff = None
                else:
                    # cv2.CLAHE は uint8/uint16 のみ受け付けるため 16bit へ量子化する。
                    # min/max で [0,1] へ正規化してから 16bit 化することで、レンジ全域に
                    # 65536 階調を割り当てる（HDR でもオーバーフローしない）。
                    normalized = (img - img_min) / img_range
                    r, g, b = cv2.split(normalized)
                    target = np.empty_like(normalized)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    for i, n in enumerate([r, g, b]):
                        n = (n * 65535.0).astype(np.uint16)
                        n = clahe.apply(n)
                        target[..., i] = n.astype(np.float32) / 65535.0
                    # 元のレンジへ復元。
                    target = target * img_range + img_min
                    ci = ci / 100
                    # ドライ項は「正規化(=[0,1]へストレッチした)画像」を使う。これが
                    # グローバルなコントラストストレッチとして効き、CLAHE のパンチを生む
                    # （元画像をドライにすると ci<100% で平坦になり「露出を上げただけ」に見える）。
                    self.diff = cv2.addWeighted(target, ci, normalized, 1.0 - ci, 0)

        return self.diff
    
class CurvesEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_tone_curves', True, "switch_tone_curves"),
        SwitchBinding('switch_color_gradings', True, "switch_color_gradings"),
    )

    def get_param_dict(self, param, subname=None):
        if subname == 'tone_curves':
            param_dict = {
                'switch_tone_curves': True,
            }
            for name in ('tonecurve', 'tonecurve_red', 'tonecurve_green', 'tonecurve_blue'):
                param_dict.update(self.effects[name].get_param_dict(param))
            return param_dict
        if subname == 'color_gradings':
            param_dict = {
                'switch_color_gradings': True,
            }
            for name in ('grading1', 'grading2'):
                param_dict.update(self.effects[name].get_param_dict(param))
            return param_dict
        if subname is None:
            return super().get_param_dict(param)
        return self.effects[subname].get_param_dict(param)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        effecs = {}
        effecs['tonecurve'] = TonecurveEffect()
        effecs['tonecurve_red'] = TonecurveRedEffect()
        effecs['tonecurve_green'] = TonecurveGreenEffect()
        effecs['tonecurve_blue'] = TonecurveBlueEffect()
        effecs['grading1'] = GradingEffect("1")
        effecs['grading2'] = GradingEffect("2")
        self.effects = effecs

    @staticmethod
    def _point_hash(point_list):
        if point_list is None:
            return None
        arr = np.asarray(point_list, dtype=np.float32)
        return hash((arr.shape, arr.tobytes()))

    def _param_hash(self, param):
        tonecurve = self.effects['tonecurve']
        tonecurve_red = self.effects['tonecurve_red']
        tonecurve_green = self.effects['tonecurve_green']
        tonecurve_blue = self.effects['tonecurve_blue']
        grading1 = self.effects['grading1']
        grading2 = self.effects['grading2']
        return hash((
            self._get_param(param, 'switch_tone_curves'),
            self._point_hash(tonecurve._get_param(param, 'tonecurve')),
            self._point_hash(tonecurve_red._get_param(param, 'tonecurve_red')),
            self._point_hash(tonecurve_green._get_param(param, 'tonecurve_green')),
            self._point_hash(tonecurve_blue._get_param(param, 'tonecurve_blue')),
            self._get_param(param, 'switch_color_gradings'),
            self._point_hash(grading1._get_param(param, 'grading1')),
            grading1._get_param(param, 'grading1_hue'),
            grading1._get_param(param, 'grading1_lum'),
            grading1._get_param(param, 'grading1_sat'),
            self._point_hash(grading2._get_param(param, 'grading2')),
            grading2._get_param(param, 'grading2_hue'),
            grading2._get_param(param, 'grading2_lum'),
            grading2._get_param(param, 'grading2_sat'),
        ))

    def delete_default_param(self, param):
        super().delete_default_param(param)
        for n in self.effects.values():
            n.delete_default_param(param)

    def reeffect(self):
        super().reeffect()
        for n in self.effects.values():
            n.reeffect()

    def after_set2widget(self, widget, param):
        for n in self.effects.values():
            n.set2widget(widget, param)

    def after_set2param(self, param, widget):
        for n in self.effects.values():
            n.set2param(param, widget)

    def make_diff(self, rgb, param, efconfig):
        needed, combined_hash = self.check_sync_necessity(self._param_hash(param), efconfig)
        if not needed:
            return self.diff
        for n in self.effects.values():
            n.reeffect()
        self.diff = pipeline.pipeline_curve(rgb, self.effects, param, efconfig)
        self.hash = combined_hash

        return self.diff


def _curve_points_hash(points):
    # hash(np.sum(points)) は総和が不変な編集(点を左右対称に動かす等)で衝突し、
    # LUTキャッシュが再計算されずに古い diff が残ってしまう。点座標のバイト列
    # そのものをキーにすることで内容が変われば必ずキーも変わるようにする。
    arr = np.asarray(points, dtype=np.float32)
    return hash((arr.shape, arr.tobytes()))

class TonecurveEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_tone_curves', True, "switch_tone_curves"),
        PointListBinding('tonecurve', None),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_tone_curves = self._get_param(param, 'switch_tone_curves')
        pl = self._get_param(param, 'tonecurve')
        if switch_tone_curves == False or pl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = _curve_points_hash(pl)
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = core.calc_point_list_to_lut(pl)

        return self.diff
    
    def apply_diff(self, rgb):
        rgb =  core.type_convert(rgb, np.ndarray)
        return core.apply_lut(rgb, self.diff, overrange="scale")

class TonecurveRedEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_tone_curves', True, "switch_tone_curves"),
        PointListBinding('tonecurve_red', None),
    )

    def make_diff(self, rgb_r, param, efconfig):
        switch_tone_curves = self._get_param(param, 'switch_tone_curves')
        pl = self._get_param(param, 'tonecurve_red')
        if switch_tone_curves == False or pl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = _curve_points_hash(pl)
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = core.calc_point_list_to_lut(pl)

        return self.diff

    def apply_diff(self, rgb_r):
        rgb_r =  core.type_convert(rgb_r, np.ndarray)
        return core.apply_lut(rgb_r, self.diff, overrange="scale")

class TonecurveGreenEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_tone_curves', True, "switch_tone_curves"),
        PointListBinding('tonecurve_green', None),
    )

    def make_diff(self, rgb_g, param, efconfig):   
        switch_tone_curves = self._get_param(param, 'switch_tone_curves')
        pl = self._get_param(param, 'tonecurve_green')
        if switch_tone_curves == False or pl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = _curve_points_hash(pl)
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = core.calc_point_list_to_lut(pl)

        return self.diff

    def apply_diff(self, rgb_g):
        rgb_g =  core.type_convert(rgb_g, np.ndarray)
        return core.apply_lut(rgb_g, self.diff, overrange="scale")

class TonecurveBlueEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_tone_curves', True, "switch_tone_curves"),
        PointListBinding('tonecurve_blue', None),
    )

    def make_diff(self, rgb_b, param, efconfig):
        switch_tone_curves = self._get_param(param, 'switch_tone_curves')
        pl = self._get_param(param, 'tonecurve_blue')
        if switch_tone_curves == False or pl is None:
            self.diff = None
            self.hash = None

        else:
            param_hash = _curve_points_hash(pl)
            if self.hash != param_hash:
                self.hash = param_hash

                self.diff = core.calc_point_list_to_lut(pl)

        return self.diff

    def apply_diff(self, rgb_b):
        rgb_b =  core.type_convert(rgb_b, np.ndarray)
        return core.apply_lut(rgb_b, self.diff, overrange="scale")

class GradingEffect(Effect):

    def __init__(self, numstr, **kwargs):
        super().__init__(**kwargs)

        self.numstr = numstr

    def get_param_dict(self, param):
        return {
            'switch_color_gradings': True,
            'grading' + self.numstr: None,
            'grading' + self.numstr + '_hue': 0,
            'grading' + self.numstr + '_lum': 50,
            'grading' + self.numstr + '_sat': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["grading" + self.numstr].set_point_list(self._get_param(param, 'grading' + self.numstr))
        widget.ids["grading" + self.numstr + "_color_picker"].set_slider_value(
            [self._get_param(param, 'grading' + self.numstr + '_hue'),
             self._get_param(param, 'grading' + self.numstr + '_lum'),
             self._get_param(param, 'grading' + self.numstr + '_sat')]
        )

    def set2param(self, param, widget):
        param["grading" + self.numstr] = widget.ids["grading" + self.numstr].get_point_list(True)
        param["grading" + self.numstr + "_hue"] = widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_hue'].value
        param["grading" + self.numstr + "_lum"] = widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_lum'].value
        param["grading" + self.numstr + "_sat"] = widget.ids["grading" + self.numstr + "_color_picker"].ids['slider_sat'].value

    def make_diff(self, rgb, param, efconfig):
        switch_color_gradings = self._get_param(param, "switch_color_gradings")
        pl = self._get_param(param, "grading" + self.numstr)
        gh = self._get_param(param, "grading" + self.numstr + "_hue")
        gl = self._get_param(param, "grading" + self.numstr + "_lum")
        gs = self._get_param(param, "grading" + self.numstr + "_sat")
        if switch_color_gradings == False or (gh == 0 and gl == 50 and gs == 0):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((_curve_points_hash(pl), gh, gl, gs))
            if self.hash != param_hash:
                self.hash = param_hash

                import colorsys
                lut = core.calc_point_list_to_lut(pl)
                rgbs = np.array(colorsys.hls_to_rgb(gh/360.0, gl/100.0, gs/100.0), dtype=np.float32)
                self.diff = (lut, rgbs)

        return self.diff
    
    def apply_diff(self, rgb):
        lut, rgbs = self.diff
        rgb = core.type_convert(rgb, np.ndarray)
        gray = core.cvtColorRGB2Gray(rgb)
        blend = core.apply_lut(gray, lut)
        blend = np.array(blend)
        return core.apply_mask(rgb, blend, rgb * rgbs)

class VSandSaturationEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_color_curves', True, "switch_color_curves"),
        SwitchBinding('switch_saturation', True, "switch_saturation"),
    )

    def get_param_dict(self, param, subname=None):
        if subname == 'color_curves':
            param_dict = {
                'switch_color_curves': True,
            }
            for name in ('HuevsHue', 'HuevsLum', 'LumvsLum', 'SatvsLum', 'HuevsSat', 'LumvsSat', 'SatvsSat'):
                param_dict.update(self.effects[name].get_param_dict(param))
            return param_dict
        if subname is None:
            return super().get_param_dict(param)
        return self.effects[subname].get_param_dict(param)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        effecs = {}
        effecs['HuevsHue'] = HuevsHueEffect()
        effecs['HuevsLum'] = HuevsLumEffect()
        effecs['LumvsLum'] = LumvsLumEffect()
        effecs['SatvsLum'] = SatvsLumEffect()
        effecs['HuevsSat'] = HuevsSatEffect()
        effecs['LumvsSat'] = LumvsSatEffect()
        effecs['SatvsSat'] = SatvsSatEffect()
        effecs['saturation'] = SaturationEffect()
        self.effects = effecs

    @staticmethod
    def _point_hash(point_list):
        if point_list is None:
            return None
        arr = np.asarray(point_list, dtype=np.float32)
        return hash((arr.shape, arr.tobytes()))

    def _param_hash(self, param, efconfig):
        hue_hue = self.effects['HuevsHue']
        hue_lum = self.effects['HuevsLum']
        lum_lum = self.effects['LumvsLum']
        sat_lum = self.effects['SatvsLum']
        hue_sat = self.effects['HuevsSat']
        lum_sat = self.effects['LumvsSat']
        sat_sat = self.effects['SatvsSat']
        saturation = self.effects['saturation']
        return hash((
            self._get_param(param, 'switch_color_curves'),
            self._point_hash(hue_hue._get_param(param, 'HuevsHue')),
            self._point_hash(hue_lum._get_param(param, 'HuevsLum')),
            self._point_hash(lum_lum._get_param(param, 'LumvsLum')),
            self._point_hash(sat_lum._get_param(param, 'SatvsLum')),
            self._point_hash(hue_sat._get_param(param, 'HuevsSat')),
            self._point_hash(lum_sat._get_param(param, 'LumvsSat')),
            self._point_hash(sat_sat._get_param(param, 'SatvsSat')),
            _hue_curve_feather_kernel_size(efconfig),
            self._get_param(param, 'switch_saturation'),
            saturation._get_param(param, 'saturation'),
            saturation._get_param(param, 'vibrance'),
        ))

    def delete_default_param(self, param):
        super().delete_default_param(param)
        for n in self.effects.values():
            n.delete_default_param(param)

    def reeffect(self):
        super().reeffect()
        for n in self.effects.values():
            n.reeffect()

    def after_set2widget(self, widget, param):
        for n in self.effects.values():
            n.set2widget(widget, param)

    def after_set2param(self, param, widget):
        for n in self.effects.values():
            n.set2param(param, widget)

    def make_diff(self, hls, param, efconfig):
        needed, combined_hash = self.check_sync_necessity(self._param_hash(param, efconfig), efconfig)
        if not needed:
            return self.diff
        for n in self.effects.values():
            n.reeffect()
        self.diff = pipeline.pipeline_vs_and_saturation(hls, self.effects, param, efconfig)
        self.hash = combined_hash

        return self.diff

HUE_CURVE_FEATHER_RADIUS = 32


def _hue_curve_feather_kernel_size(efconfig):
    resolution_scale = getattr(efconfig, "resolution_scale", 1.0)
    kernel_size = max(3, int((HUE_CURVE_FEATHER_RADIUS * 2 + 1) * resolution_scale))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size


def _blur_hue_curve_map(adjust_map, efconfig):
    kernel_size = _hue_curve_feather_kernel_size(efconfig)
    if kernel_size <= 1:
        return adjust_map
    # HUE_CURVE_FEATHER_RADIUS=32 は kernel_size~65px(表示倍率でさらに拡大)と大きく、
    # 直接の cv2.GaussianBlur が重い(1024pxで~10ms)。cv2 が ksize=0 の時に使う
    # 逆算式で等価な sigma を求め、filters._fast_isotropic_blur に委譲する。
    # sigma が閾値(8.0)以下ならこれまでと同じ cv2.GaussianBlur(0,0,sigma) に帰着し、
    # 閾値超のときだけ縮小→ぼかし→拡大の近似(誤差目安 <1e-3、既存のボケ処理と共通)になる。
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    return filters._fast_isotropic_blur(adjust_map, sigma)
    
class HuevsHueEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_color_curves', True, "switch_color_curves"),
        PointListBinding('HuevsHue', None),
    )

    def make_diff(self, hls_hh, param, efconfig):
        switch_color_curves = self._get_param(param, "switch_color_curves")
        hh = self._get_param(param, "HuevsHue")
        if switch_color_curves == False or hh is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((_curve_points_hash(hh), _hue_curve_feather_kernel_size(efconfig)))
            if self.hash != param_hash:
                self.hash = param_hash

                lut = core.calc_point_list_to_lut(hh)
                lut = ((lut - 0.5) * 2.0) * 360
                hue_offset = core.apply_lut(hls_hh[0] / 360, lut, 1.0)
                hue_offset = _blur_hue_curve_map(hue_offset, efconfig)
                self.diff = hue_offset + hls_hh[1]

        return self.diff

class HuevsLumEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_color_curves', True, "switch_color_curves"),
        PointListBinding('HuevsLum', None),
    )

    def make_diff(self, hls_hl, param, efconfig):
        switch_color_curves = self._get_param(param, "switch_color_curves")
        hl = self._get_param(param, "HuevsLum")
        if switch_color_curves == False or hl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((_curve_points_hash(hl), _hue_curve_feather_kernel_size(efconfig)))
            if self.hash != param_hash:
                self.hash = param_hash

                lut = core.calc_point_list_to_lut(hl)
                lut = (lut - 0.5) * 4.0
                lum_delta = core.apply_lut(hls_hl[0] / 360, lut, 1.0)
                lum_delta = _blur_hue_curve_map(lum_delta, efconfig)
                self.diff = (2.0 ** lum_delta) * hls_hl[1]

        return self.diff

class HuevsSatEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_color_curves', True, "switch_color_curves"),
        PointListBinding('HuevsSat', None),
    )

    def make_diff(self, hls_hs, param, efconfig):
        switch_color_curves = self._get_param(param, "switch_color_curves")
        hs = self._get_param(param, "HuevsSat")
        if switch_color_curves == False or hs is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((_curve_points_hash(hs), _hue_curve_feather_kernel_size(efconfig)))
            if self.hash != param_hash:
                self.hash = param_hash

                lut = core.calc_point_list_to_lut(hs)
                lut = (lut - 0.5) * 2.0
                sat_delta = core.apply_lut(hls_hs[0] / 360.0, lut, 1.0)
                sat_delta = _blur_hue_curve_map(sat_delta, efconfig)
                self.diff = (1.0 + sat_delta) * hls_hs[1]

        return self.diff

class LumvsLumEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_color_curves', True, "switch_color_curves"),
        PointListBinding('LumvsLum', None),
    )

    def make_diff(self, hls_ll, param, efconfig):
        switch_color_curves = self._get_param(param, "switch_color_curves")
        ll = self._get_param(param, "LumvsLum")
        if switch_color_curves == False or ll is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = _curve_points_hash(ll)
            if self.hash != param_hash:
                self.hash = param_hash

                lut = core.calc_point_list_to_lut(ll)
                lut = 2.0 ** ((lut - 0.5) * 4.0)
                self.diff = core.apply_lut(hls_ll[0], lut, 1.0) * hls_ll[1]

        return self.diff

class LumvsSatEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_color_curves', True, "switch_color_curves"),
        PointListBinding('LumvsSat', None),
    )

    def make_diff(self, hls_ls, param, efconfig):
        switch_color_curves = self._get_param(param, "switch_color_curves")
        ls = self._get_param(param, "LumvsSat")
        if switch_color_curves == False or ls is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = _curve_points_hash(ls)
            if self.hash != param_hash:
                self.hash = param_hash

                lut = core.calc_point_list_to_lut(ls)
                lut = (lut - 0.5) * 2.0 + 1.0
                self.diff = core.apply_lut(hls_ls[0], lut, 1.0) * hls_ls[1]

        return self.diff

class SatvsLumEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_color_curves', True, "switch_color_curves"),
        PointListBinding('SatvsLum', None),
    )

    def make_diff(self, hls_sl, param, efconfig):
        switch_color_curves = self._get_param(param, "switch_color_curves")
        sl = self._get_param(param, "SatvsLum")
        if switch_color_curves == False or sl is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = _curve_points_hash(sl)
            if self.hash != param_hash:
                self.hash = param_hash

                lut = core.calc_point_list_to_lut(sl)
                lut = 2.0 ** ((lut - 0.5) * 4.0)
                self.diff = core.apply_lut(hls_sl[0], lut, 1.0) * hls_sl[1]

        return self.diff

class SatvsSatEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_color_curves', True, "switch_color_curves"),
        PointListBinding('SatvsSat', None),
    )

    def make_diff(self, hls_ss, param, efconfig):
        switch_color_curves = self._get_param(param, "switch_color_curves")
        ss = self._get_param(param, "SatvsSat")
        if switch_color_curves == False or ss is None:
            self.diff = None
            self.hash = None
        else:
            param_hash = _curve_points_hash(ss)
            if self.hash != param_hash:
                self.hash = param_hash

                lut = core.calc_point_list_to_lut(ss)
                lut = (lut - 0.5) * 2.0 + 1.0
                self.diff = core.apply_lut(hls_ss[0], lut, 1.0) * hls_ss[1]

        return self.diff

class SaturationEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_saturation', True, "switch_saturation"),
        SliderBinding('saturation', 0, "slider_saturation"),
        SliderBinding('vibrance', 0, "slider_vibrance"),
    )

    def make_diff(self, hls_s, param, efconfig):
        switch_saturation = self._get_param(param, 'switch_saturation')
        sat = self._get_param(param, 'saturation')
        vib = self._get_param(param, 'vibrance')
        if switch_saturation == False or (sat == 0 and vib == 0):
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((sat, vib))
            if self.hash != param_hash:
                self.hash = param_hash

                hls_s = core.type_convert(hls_s, np.ndarray)
                self.diff = core.calc_saturation(hls_s, sat, vib)
        
        return self.diff

class LUTEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_lut', True, "switch_lut"),
        SpinnerTextBinding('lut_name', 'None', "lut_spinner"),
        SliderBinding('lut_intensity', 100, "slider_lut_intensity"),
        SpinnerTextBinding('lut_to_log', 'None', "lut_to_log_spinner"),
    )

    def __init__(self, stage="look", **kwargs):
        super().__init__(**kwargs)

        self.lut = None
        self.lut_key = None
        self.stage = stage

    def get_param_dict(self, param):
        # auto_exposure / rgb_or_raw は UI binding を持たず imageset が EXIF Ev から
        # 設定する（旧 AutoExposureEffect を input stage に統合）。_get_param のフォールバック用。
        param_dict = super().get_param_dict(param)
        param_dict['auto_exposure'] = 0
        param_dict['rgb_or_raw'] = 'raw'
        return param_dict

    def set2param(self, param, widget):
        previous_lut_name = self._get_param(param, 'lut_name')
        super().set2param(param, widget)
        if previous_lut_name != param['lut_name']:
            self.lut = None
            self.lut_key = None

    def make_diff(self, rgb, param, efconfig):
        switch_lut = self._get_param(param, 'switch_lut')
        lut_name = self._get_param(param, 'lut_name')
        lut_to_log = self._get_param(param, 'lut_to_log')
        lut_intensity = self._get_param(param, 'lut_intensity')
        stage_active = (
            (self.stage == "input" and lut_to_log != 'None')
            or (self.stage == "look" and lut_to_log == 'None')
        )
        if switch_lut == False or not stage_active or lut_name == 'None' or lut_intensity == 0:
            self.diff = None
            self.hash = None
        else:
            lut_path = config.get_config('lut_path')
            if lut_path is None:
                self.diff = None
                self.hash = None
                return self.diff

            # 旧 AutoExposureEffect 統合: input stage の raw 画像のみ露出補正を適用する。
            # 露出補正条件は input LUT の描画条件と一致するため、param_hash にも含める。
            rgb_or_raw = self._get_param(param, 'rgb_or_raw')
            auto_exposure = self._get_param(param, 'auto_exposure')
            extra = (rgb_or_raw, auto_exposure) if self.stage == "input" else ()
            param_hash = hash((self.stage, lut_name, lut_path, lut_to_log, lut_intensity, extra))
            if self.hash != param_hash:
                path = os.path.join(lut_path, lut_name)
                lut_key = (path, lut_name)
                if self.lut is None or self.lut_key != lut_key:
                    self.lut = cubelut.read_lut(path)
                    self.lut_key = lut_key

                if self.lut is not None:
                    self.hash = param_hash

                    rgb = core.type_convert(rgb, np.ndarray)
                    # 旧 AutoExposureEffect: raw 画像の input stage のみ、log 変換前に露出補正。
                    if self.stage == "input" and rgb_or_raw == 'raw':
                        rgb = core.adjust_exposure(rgb, auto_exposure)

                    # intensity は「元画像 ↔ (log+)LUT 適用後」のドライ/ウェット。
                    # log 変換は LUT の入力にだけ使い、ブレンドのドライ項には元画像を使う。
                    # （log 画像をドライ項にすると intensity を下げたとき乳白色に明るくなる）
                    lut_input = rgb
                    if lut_to_log != 'None':
                        lut_input = linear_to_log.process_image(rgb, lut_to_log)

                    overrange = "preserve" if lut_to_log == 'None' else "clip"
                    apply_rgb = cubelut.apply_lut(lut_input, self.lut, overrange=overrange)
                    self.diff = rgb * (1-lut_intensity/100) + apply_rgb * lut_intensity/100
                else:
                    self.diff = None
                    self.hash = None

        return self.diff

class LensSimulatorEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_lens_simulator', True, "switch_lens_simulator", widget_attr="enabled"),
        SpinnerTextBinding('coating_preset', 'None', "spinner_coating_preset"),
        SliderBinding('coating_strength', 100, "slider_coating_strength"),
        SliderBinding('coating_light', 1.0, "slider_coating_light"),
        SliderBinding('lateral_ca', 0.0, "slider_lateral_ca"),
        SliderBinding('longitudinal_ca', 0.0, "slider_longitudinal_ca"),
        SliderBinding('spherical_ca', 0.0, "slider_spherical_ca"),
        SliderBinding('lens_focus_depth', 0.5, "slider_lens_focus_depth"),
        SliderBinding('lens_aperture', 1.4, "slider_lens_aperture"),
        # ぐるぐるボケ（Petzval/Helios 風）。渦の向きは半径方向、ボケ選別は depth(あれば)。
        SliderBinding('swirl_bokeh', 0, "slider_swirl_bokeh"),
        # ボケの色縁（前ボケ=マゼンタ / 後ボケ=グリーン）。depth がある時のみ。
        SliderBinding('bokeh_fringe', 0, "slider_bokeh_fringe"),
        # 形状ボケ（絞り形状カーネルで畳み込む被写界ボケ。星/ハート等）。
        SliderBinding('shaped_bokeh', 0, "slider_shaped_bokeh"),
        SpinnerTextBinding('bokeh_shape', 'Hexagon', "spinner_bokeh_shape"),
        # 形状ボケの縁の色ボケ（虹色の輪郭）。R/G/B でボケ径を僅かに変えて出す。
        SliderBinding('bokeh_rim', 0, "slider_bokeh_rim"),
        # サンスター（光条）。クリップ気味のハイライト塊だけに回折スパイクを描く。
        SliderBinding('sunstar_strength', 0, "slider_sunstar_strength"),
        SliderBinding('sunstar_length', 35, "slider_sunstar_length"),
        SliderBinding('sunstar_threshold', 60, "slider_sunstar_threshold"),
        # 絞り羽根枚数。偶数=N本 / 奇数=2N本（物理整合）でスパイク数を決める。
        SpinnerTextBinding('aperture_blades', '9', "spinner_aperture_blades"),
    )

    def _coating_label_to_key(self, label):
        if label == 'None':
            return None
        for key, data in coating_adapter.presets().items():
            if data['name'] == label:
                return key
        return None

    def make_diff(self, rgb, param, efconfig):
        switch = self._get_param(param, 'switch_lens_simulator')
        coat_label = self._get_param(param, 'coating_preset')
        coat_strength = self._get_param(param, 'coating_strength')
        coat_light = float(self._get_param(param, 'coating_light'))
        lateral = float(self._get_param(param, 'lateral_ca'))
        longitudinal = float(self._get_param(param, 'longitudinal_ca'))
        spherical = float(self._get_param(param, 'spherical_ca'))
        focus_d = float(self._get_param(param, 'lens_focus_depth'))
        aperture = float(self._get_param(param, 'lens_aperture'))
        swirl = float(self._get_param(param, 'swirl_bokeh'))
        fringe = float(self._get_param(param, 'bokeh_fringe'))
        shaped = float(self._get_param(param, 'shaped_bokeh'))
        bokeh_shape = self._get_param(param, 'bokeh_shape')
        bokeh_rim = float(self._get_param(param, 'bokeh_rim'))
        sunstar = float(self._get_param(param, 'sunstar_strength'))
        sunstar_length = float(self._get_param(param, 'sunstar_length'))
        sunstar_threshold = float(self._get_param(param, 'sunstar_threshold'))
        aperture_blades = self._get_param(param, 'aperture_blades')

        coat_key = self._coating_label_to_key(coat_label)
        coat_on = coat_key is not None and coat_strength > 0
        lateral_on = lateral > 1e-5
        swirl_on = swirl > 1e-5
        fringe_requested = fringe > 1e-5
        shaped_on = shaped > 1e-5
        sunstar_on = sunstar > 1e-5  # 輝度のみで動作（depth 不要）
        depth_aber_requested = (longitudinal > 1e-5 or spherical > 1e-5)

        # 軸上色収差(longitudinal)・球面収差(spherical)・ボケ色縁(fringe)は AI depth map を要求する。
        # ぐるぐるボケ(swirl)は depth があれば非合焦領域に限定するが、無くても半径のみで動く。
        # ここで生成すると重いので、mask2 側で既に作成済みの depth のみ peek で取得する。
        depth_map = None
        if depth_aber_requested or swirl_on or fringe_requested or shaped_on:
            getter = getattr(efconfig, "get_ai_depth_map", None)
            if callable(getter):
                try:
                    depth_map = getter(space="current", allow_compute=False)
                except Exception:
                    logging.exception("LensSimulator: AI depth map の取得に失敗")
                    depth_map = None
        depth_available = depth_map is not None
        eff_longitudinal = longitudinal if depth_available else 0.0
        eff_spherical = spherical if depth_available else 0.0
        aber_on = lateral_on or (depth_available and depth_aber_requested)
        fringe_on = fringe_requested and depth_available

        if not switch or (not coat_on and not aber_on and not swirl_on and not fringe_on and not shaped_on and not sunstar_on):
            self.diff = None
            self.hash = None
            return self.diff

        # swirl の中心は disp_info（拡大/クロップ窓）に依存するため、変化時に再計算する。
        try:
            disp_snapshot = tuple(params.get_disp_info(param)) if (swirl_on or shaped_on or sunstar_on) else None
        except Exception:
            disp_snapshot = None
        param_hash = hash((
            coat_label, coat_strength, round(coat_light, 4),
            round(lateral, 4), round(longitudinal, 4), round(spherical, 4),
            round(focus_d, 4), round(aperture, 4), round(swirl, 4), round(fringe, 4),
            round(shaped, 4), str(bokeh_shape), round(bokeh_rim, 4),
            round(sunstar, 4), round(sunstar_length, 4), round(sunstar_threshold, 4), str(aperture_blades),
            depth_available, getattr(efconfig, "upstream_hash", 0),
            round(float(getattr(efconfig, "resolution_scale", 1.0)), 4),
            disp_snapshot,
        ))
        if self.hash == param_hash:
            return self.diff

        self.hash = param_hash
        rgb = core.type_convert(rgb, np.ndarray)
        work = np.asarray(rgb, dtype=np.float32).copy()
        res_scale = float(getattr(efconfig, "resolution_scale", 1.0))
        h, w = work.shape[:2]

        # depth を near=0 / far=1 規約に揃えて aber/swirl で共有する。
        dm = None
        if depth_available:
            dm = np.asarray(depth_map, dtype=np.float32)
            if dm.ndim == 3:
                dm = dm[..., 0]
            if dm.shape[:2] != (h, w):
                dm = cv2.resize(dm, (w, h), interpolation=cv2.INTER_LINEAR)
            # Depth Pro は inverse-depth 正規化(近=1.0 / 遠=0.0)。near=0.0 / far=1.0 へ反転。
            dm = np.clip(1.0 - dm, 0.0, 1.0).astype(np.float32, copy=False)

        if aber_on:
            processed = work.copy()

            # 1. 球面収差（最初に適用：ぼかし効果のため）
            processed = lens_aberration_adapter.apply_spherical_aberration(
                processed,
                dm,
                strength=eff_spherical,
                aperture=max(0.5, aperture),
                focus_depth=focus_d,
                highlight_threshold=0.7,
                resolution_scale=res_scale,
            )

            # 2. 軸上色収差（深度マップが必要）
            if dm is not None:
                processed = lens_aberration_adapter.apply_longitudinal_chromatic_aberration(
                    processed,
                    dm,
                    strength=eff_longitudinal,
                    focus_depth=focus_d,
                    resolution_scale=res_scale,
                )

            # 3. 倍率色収差（最後に適用：エッジの色ずれ）
            processed = lens_aberration_adapter.apply_lateral_chromatic_aberration(
                processed,
                strength=lateral,
                resolution_scale=res_scale,
                radial=True,
            )
        else:
            processed = work

        # ボケの色縁（前ボケ=マゼンタ / 後ボケ=グリーン）。depth がある時のみ。
        # 後段の swirl で色ごと smear されると自然なので coat/swirl より前に適用する。
        if fringe_on:
            processed = lens_effect_adapter.apply_bokeh_color_fringe(
                np.ascontiguousarray(processed, dtype=np.float32),
                dm, focus_d, fringe, res_scale,
            )

        # 形状ボケ（絞り形状で畳み込む被写界ボケ）。
        # ボケ径は「シーン(元画像px)で一定」になるよう表示倍率 mag=disp_info[4] を掛ける。
        # これで拡大すると玉ボケも比例して大きくなり、輪郭のボケ具合が一定に保たれる。
        if shaped_on:
            try:
                mag = float(params.get_disp_info(param)[4]) * float(getattr(efconfig, 'preview_res_scale', 1.0))
            except Exception:
                mag = 1.0
            ow, oh = param.get('original_img_size', (processed.shape[1], processed.shape[0]))
            base = float(min(ow, oh))
            ap = float(np.clip(2.0 / max(0.8, aperture), 0.6, 2.0))
            r_scene = (0.004 + 0.016 * (shaped / 100.0)) * base * ap  # 元画像px のボケ半径
            radius = int(np.clip(r_scene * max(mag, 1e-3), 3, 80))
            processed = lens_effect_adapter.apply_shaped_bokeh(
                np.ascontiguousarray(processed, dtype=np.float32),
                dm, focus_d, shaped, radius, bokeh_shape, bokeh_rim,
            )

        if coat_on:
            coated = coating_adapter.apply_preset(
                processed.copy(),
                coat_key,
                light_source_intensity=coat_light,
                resolution_scale=res_scale,
            )
            t = coat_strength / 100.0
            processed = processed * (1.0 - t) + coated * t

        # ぐるぐるボケ（depth があれば非合焦領域に限定、無ければ半径のみ）。
        # 渦中心は拡大/クロップに依らず元画像中心へ固定する。
        if swirl_on:
            disp = original_size = crop_size_offset = None
            try:
                disp = params.get_disp_info(param)
                _res_div = float(getattr(efconfig, 'preview_res_scale', 1.0))
                if disp is not None and _res_div != 1.0:
                    # half-res 実行中は縮小画像に合わせてスケール成分だけ縮める
                    disp = (disp[0], disp[1], disp[2], disp[3], disp[4] * _res_div)
                original_size = param['original_img_size']
                crop_size_offset = core.crop_size_and_offset_from_texture(
                    int(processed.shape[1]),
                    int(processed.shape[0]),
                    disp,
                )
            except Exception:
                # disp/crop情報の取得に失敗しても、以降はNoneのままフォールバック処理される
                pass
            cx, cy, radial_norm = lens_effect_adapter.optical_geometry(
                processed.shape,
                disp_info=disp,
                original_img_size=original_size,
                crop_size_offset=crop_size_offset,
            )
            processed = lens_effect_adapter.apply_swirl_bokeh(
                np.ascontiguousarray(processed, dtype=np.float32),
                dm, focus_d, swirl, res_scale, (cx, cy), radial_norm,
            )

        # サンスター（光条）はパイプライン最後。最終ハイライトの上に加算オーバーレイする。
        # 長さ・太さは表示倍率 mag(=disp_info[4]) と元画像サイズ基準でシーン一定にする。
        if sunstar_on:
            try:
                sun_mag = float(params.get_disp_info(param)[4]) * float(getattr(efconfig, 'preview_res_scale', 1.0))
            except Exception:
                sun_mag = 1.0
            sun_orig = param.get('original_img_size', (processed.shape[1], processed.shape[0]))
            processed = lens_effect_adapter.apply_sunstar(
                np.ascontiguousarray(processed, dtype=np.float32),
                sunstar, sunstar_length, sunstar_threshold,
                aperture_blades, aperture, sun_mag, sun_orig,
            )

        self.diff = processed.astype(np.float32, copy=False)
        return self.diff
    
class FilmSimulationEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_film_simulation', True, "switch_film_simulation", widget_attr="enabled"),
        SpinnerTextBinding('film_mode', 'Off', "spinner_film_mode"),
        SliderBinding('film_latitude', 55, "slider_film_latitude"),
        SliderBinding('film_contrast', 50, "slider_film_contrast"),
        SliderBinding('film_color_bias', 0, "slider_film_color_bias"),
        SliderBinding('film_color_drift', 0, "slider_film_color_drift"),
        SliderBinding('film_dye_purity', 75, "slider_film_dye_purity"),
        SliderBinding('film_layer_crosstalk', 30, "slider_film_layer_crosstalk"),
        SliderBinding('film_halation', 0, "slider_film_halation"),
        SliderBinding('film_aging', 0, "slider_film_aging"),
        SliderBinding('film_intensity', 100, "slider_film_intensity"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_film_simulation = self._get_param(param, 'switch_film_simulation')
        mode = self._get_param(param, 'film_mode')
        latitude = self._get_param(param, 'film_latitude')
        contrast = self._get_param(param, 'film_contrast')
        color_bias = self._get_param(param, 'film_color_bias')
        color_drift = self._get_param(param, 'film_color_drift')
        dye_purity = self._get_param(param, 'film_dye_purity')
        layer_crosstalk = self._get_param(param, 'film_layer_crosstalk')
        halation = self._get_param(param, 'film_halation')
        aging = self._get_param(param, 'film_aging')
        intensity = self._get_param(param, 'film_intensity')
        if switch_film_simulation == False or mode == 'Off' or intensity <= 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((
                mode,
                latitude,
                contrast,
                color_bias,
                color_drift,
                dye_purity,
                layer_crosstalk,
                halation,
                aging,
                intensity,
            ))
            if self.hash != param_hash:
                self.hash = param_hash
                
                rgb = core.type_convert(rgb, np.ndarray)
                film = film_process.apply_film_process(
                    rgb,
                    mode=mode,
                    latitude=latitude,
                    contrast=contrast,
                    color_bias=color_bias,
                    color_drift=color_drift,
                    dye_purity=dye_purity,
                    layer_crosstalk=layer_crosstalk,
                    halation=halation,
                    aging=aging,
                )
                per = intensity / 100.0
                self.diff = cv2.addWeighted(film, per, rgb, 1 - per, 0)

        return self.diff

class SolidColorEffect(Effect):

    def get_param_dict(self, param):
        return {
            'switch_solid_color': True,
            'solid_color': 0,
            'solid_color_hue': 0,
            'solid_color_lum': 50,
            'solid_color_sat': 0,
            'solid_opacity': 0,
        }

    def set2widget(self, widget, param):
        widget.ids["switch_solid_color"].active = self._get_param(param, 'switch_solid_color')
        h, l, s = self._get_param(param, 'solid_color_hue'), self._get_param(param, 'solid_color_lum'), self._get_param(param, 'solid_color_sat')
        widget.ids["cp_solid_color"].ids['slider_hue'].set_slider_value(h)
        widget.ids["cp_solid_color"].ids['slider_lum'].set_slider_value(l)
        widget.ids["cp_solid_color"].ids['slider_sat'].set_slider_value(s)
        widget.ids["slider_solid_color"].set_slider_value(self._get_param(param, 'solid_opacity'))
        # これを後にしないと値が上書きされる
        widget.ids["cp_solid_color"].set_slider_value((h, l, s))

    def set2param(self, param, widget):
        param['switch_solid_color'] = widget.ids["switch_solid_color"].active
        param["solid_color_hue"] = widget.ids["cp_solid_color"].ids['slider_hue'].value
        param["solid_color_lum"] = widget.ids["cp_solid_color"].ids['slider_lum'].value
        param["solid_color_sat"] = widget.ids["cp_solid_color"].ids['slider_sat'].value
        param["solid_opacity"] = widget.ids["slider_solid_color"].value

    def make_diff(self, rgb, param, efconfig):
        switch_solid_color = self._get_param(param, 'switch_solid_color')
        coh = self._get_param(param, "solid_color_hue")
        col = self._get_param(param, "solid_color_lum")
        cos = self._get_param(param, "solid_color_sat")
        coao = self._get_param(param, "solid_opacity")
        if switch_solid_color == False or coao <= 0:
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((coh, cos, col, coao))
            if self.hash != param_hash:
                self.hash = param_hash

                import colorsys
                r, g, b = colorsys.hls_to_rgb(coh/360, col/100, cos/100)
                rgb = core.type_convert(rgb, np.ndarray)
                self.diff = core.apply_solid_color(rgb, solid_color=(r, g, b), opacity=coao/100)

        return self.diff


class LightRaysEffect(Effect):
    """Fog/forest volumetric rays driven by point and line guides.

    The heavy image math lives in ``cores.light_rays``.  This class only bridges
    params, TCG guide conversion, and the optional preview editor.
    """

    _SLIDER_KEYS = (
        'light_ray_intensity',
        'light_ray_length',
        'light_ray_decay',
        'light_ray_width',
        'light_ray_softness',
        'light_ray_edge_bias',
        'light_ray_spread',
        'light_ray_count',
        'light_ray_density',
        'light_ray_variation',
        'light_ray_fog',
        'light_ray_occlusion',
        'light_ray_seed',
    )
    _GUIDE_COLOR_KEYS = (
        'light_ray_color_hue',
        'light_ray_color_lum',
        'light_ray_color_sat',
    )
    _GUIDE_PARAM_KEYS = _SLIDER_KEYS + _GUIDE_COLOR_KEYS
    _PROJECTION_LENGTH_KEY = 'light_ray_projection_length'

    def __init__(self, light_rays_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.light_rays_canvas = None
        self.light_rays_callback = light_rays_callback

    @staticmethod
    def _normalize_editor_type(editor_type):
        return 'Point' if str(editor_type or '').strip().lower() == 'point' else 'Line'

    @classmethod
    def _editor_mode_values(cls, editor_type):
        return ('Radial', 'Directional') if cls._normalize_editor_type(editor_type) == 'Point' else ('Parallel', 'Directional')

    @classmethod
    def _normalize_editor_mode(cls, editor_type, editor_mode):
        values = cls._editor_mode_values(editor_type)
        lowered = str(editor_mode or '').strip().lower()
        for value in values:
            if lowered == value.lower():
                return value
        return values[0]

    @classmethod
    def _sync_editor_mode_spinner(cls, widget, editor_type, editor_mode):
        spinner = widget.ids.get("spinner_light_ray_editor_mode") if hasattr(widget, 'ids') else None
        mode = cls._normalize_editor_mode(editor_type, editor_mode)
        if spinner is not None:
            spinner.values = list(cls._editor_mode_values(editor_type))
            if hasattr(spinner, 'set_text'):
                spinner.set_text(mode)
            else:
                spinner.text = mode
        return mode

    def _guide_params_from_param(self, param):
        out = {}
        for key in self._GUIDE_PARAM_KEYS:
            try:
                out[key] = float(self._get_param(param, key))
            except (KeyError, TypeError, ValueError):
                # キー欠落/数値変換不可のキーは out に含めず、呼び出し側のデフォルトに委ねる
                pass
        return out

    def _is_projected_radial_guide(self, guide):
        if not isinstance(guide, dict):
            return False
        if str(guide.get('type', '')).lower() != 'point':
            return False
        if str(guide.get('mode', 'radial')).lower() != 'radial':
            return False
        p2 = guide.get('p2')
        try:
            float(p2[0])
            float(p2[1])
            return True
        except Exception:
            return False

    def _projection_length_value(self, guide, param):
        guide_params = guide.get('params') if isinstance(guide.get('params'), dict) else {}
        for key in (self._PROJECTION_LENGTH_KEY, 'light_ray_length'):
            if key in guide_params:
                try:
                    return float(guide_params[key])
                except (TypeError, ValueError):
                    # 数値変換不可なら次の候補キー/デフォルトへフォールバック
                    pass
        return float(self._get_param(param, 'light_ray_length'))

    def _ensure_projection_lengths(self, guides, param):
        changed = False
        for guide in guides or []:
            if not self._is_projected_radial_guide(guide):
                continue
            guide_params = guide.get('params') if isinstance(guide.get('params'), dict) else {}
            if self._PROJECTION_LENGTH_KEY not in guide_params:
                guide_params = guide_params.copy()
                guide_params[self._PROJECTION_LENGTH_KEY] = self._projection_length_value(guide, param)
                guide['params'] = guide_params
                changed = True
        return changed

    def _with_preserved_projection_length(self, guide, param, guide_params):
        if self._is_projected_radial_guide(guide):
            old_params = guide.get('params') if isinstance(guide.get('params'), dict) else {}
            if self._PROJECTION_LENGTH_KEY in old_params:
                guide_params[self._PROJECTION_LENGTH_KEY] = old_params[self._PROJECTION_LENGTH_KEY]
            else:
                guide_params[self._PROJECTION_LENGTH_KEY] = self._projection_length_value(guide, param)
        return guide_params

    @staticmethod
    def _copy_light_ray_guides(guides):
        out = []
        for guide in guides or []:
            if isinstance(guide, dict):
                item = guide.copy()
                if isinstance(guide.get('params'), dict):
                    item['params'] = guide['params'].copy()
                out.append(item)
        return out

    @staticmethod
    def _selected_guide_index(param, guides):
        if not guides:
            return -1
        try:
            index = int(param.get('light_ray_selected', -1))
        except Exception:
            index = -1
        if 0 <= index < len(guides):
            return index
        return 0

    def _apply_active_params_to_guide_list(self, param, guides, *, update_mode=False):
        target = self._selected_guide_index(param, guides)
        if target < 0:
            return False
        guide = guides[target]
        guide_params = self._with_preserved_projection_length(guide, param, self._guide_params_from_param(param))
        changed = guide.get('params') != guide_params
        guide['params'] = guide_params
        if update_mode:
            gtype = str(guide.get('type', '')).lower()
            editor_type = self._normalize_editor_type(param.get('light_ray_editor_type', gtype))
            editor_mode = self._normalize_editor_mode(editor_type, param.get('light_ray_editor_mode', guide.get('mode', '')))
            if gtype == editor_type.lower():
                new_mode = editor_mode.lower()
                if guide.get('mode') != new_mode:
                    guide['mode'] = new_mode
                    changed = True
                if gtype == 'point' and new_mode == 'directional' and not isinstance(guide.get('p2'), (tuple, list)):
                    p = guide.get('p')
                    try:
                        guide['p2'] = (float(p[0]) + 0.18, float(p[1]))
                        changed = True
                    except (TypeError, ValueError, IndexError):
                        # p が未設定/不正な形式なら p2 補完はスキップ
                        pass
        param['light_ray_selected'] = target
        return changed

    def _guide_value(self, guide, param, key):
        guide_params = guide.get('params') if isinstance(guide, dict) else None
        if isinstance(guide_params, dict) and key in guide_params:
            return guide_params[key]
        return self._get_param(param, key)

    def _guide_color_rgb(self, guide, param):
        import colorsys
        h = float(self._guide_value(guide, param, 'light_ray_color_hue')) / 360.0
        l = float(self._guide_value(guide, param, 'light_ray_color_lum')) / 100.0
        s = float(self._guide_value(guide, param, 'light_ray_color_sat')) / 100.0
        return colorsys.hls_to_rgb(h, l, s)

    def _apply_guide_params_to_widget(self, widget, guide, param):
        if not isinstance(guide, dict):
            return
        editor_type = self._normalize_editor_type(guide.get('type', self._get_param(param, 'light_ray_editor_type')))
        editor_mode = self._normalize_editor_mode(editor_type, guide.get('mode', self._get_param(param, 'light_ray_editor_mode')))
        type_spinner = widget.ids.get("spinner_light_ray_editor_type")
        if type_spinner is not None:
            if hasattr(type_spinner, 'set_text'):
                type_spinner.set_text(editor_type)
            else:
                type_spinner.text = editor_type
        self._sync_editor_mode_spinner(widget, editor_type, editor_mode)
        guide_params = guide.get('params') if isinstance(guide.get('params'), dict) else {}
        for key in self._SLIDER_KEYS:
            slider = widget.ids.get("slider_" + key)
            if slider is not None:
                slider.set_slider_value(guide_params.get(key, self._get_param(param, key)))
        cp = widget.ids.get("cp_light_ray_color")
        if cp is not None:
            h = guide_params.get('light_ray_color_hue', self._get_param(param, 'light_ray_color_hue'))
            l = guide_params.get('light_ray_color_lum', self._get_param(param, 'light_ray_color_lum'))
            s = guide_params.get('light_ray_color_sat', self._get_param(param, 'light_ray_color_sat'))
            cp.ids['slider_hue'].set_slider_value(h)
            cp.ids['slider_lum'].set_slider_value(l)
            cp.ids['slider_sat'].set_slider_value(s)
            cp.set_slider_value((h, l, s))

    def sync_selected_guide_to_widget(self, widget, param):
        canvas = self.light_rays_canvas
        if canvas is None or not hasattr(canvas, 'selected_index'):
            return False
        index = canvas.selected_index()
        guides = self._get_param(param, 'light_ray_guides') or []
        if index < 0 or index >= len(guides):
            return False
        self._apply_guide_params_to_widget(widget, guides[index], param)
        return True

    def get_param_dict(self, param):
        return {
            'switch_light_rays': True,
            'light_ray_guides': [],
            'light_ray_selected': -1,
            'light_ray_editor_type': 'Line',
            'light_ray_editor_mode': 'Parallel',
            'light_ray_intensity': 60,
            'light_ray_length': 30,
            'light_ray_decay': 50,
            'light_ray_width': 65,
            'light_ray_softness': 45,
            'light_ray_edge_bias': 0,
            'light_ray_spread': 35,
            'light_ray_count': 8,
            'light_ray_density': 10,
            'light_ray_variation': 45,
            'light_ray_fog': 25,
            'light_ray_occlusion': 30,
            'light_ray_color_hue': 42,
            'light_ray_color_lum': 58,
            'light_ray_color_sat': 45,
            'light_ray_seed': 0,
        }

    def set2widget(self, widget, param):
        if "switch_light_rays" not in widget.ids:
            return
        widget.ids["switch_light_rays"].active = self._get_param(param, 'switch_light_rays')
        editor_type = self._normalize_editor_type(self._get_param(param, 'light_ray_editor_type'))
        widget.ids["spinner_light_ray_editor_type"].set_text(editor_type)
        self._sync_editor_mode_spinner(widget, editor_type, self._get_param(param, 'light_ray_editor_mode'))
        for key in self._SLIDER_KEYS:
            slider = widget.ids.get("slider_" + key)
            if slider is not None:
                slider.set_slider_value(self._get_param(param, key))
        h = self._get_param(param, 'light_ray_color_hue')
        l = self._get_param(param, 'light_ray_color_lum')
        s = self._get_param(param, 'light_ray_color_sat')
        cp = widget.ids.get("cp_light_ray_color")
        if cp is not None:
            cp.ids['slider_hue'].set_slider_value(h)
            cp.ids['slider_lum'].set_slider_value(l)
            cp.ids['slider_sat'].set_slider_value(s)
            cp.set_slider_value((h, l, s))
        self.after_set2widget(widget, param)

    def set2param(self, param, widget):
        if "switch_light_rays" not in widget.ids:
            return
        param['switch_light_rays'] = widget.ids["switch_light_rays"].active
        editor_type = self._normalize_editor_type(widget.ids["spinner_light_ray_editor_type"].text)
        editor_mode = self._sync_editor_mode_spinner(widget, editor_type, widget.ids["spinner_light_ray_editor_mode"].text)
        param['light_ray_editor_type'] = editor_type
        param['light_ray_editor_mode'] = editor_mode
        for key in self._SLIDER_KEYS:
            slider = widget.ids.get("slider_" + key)
            if slider is not None:
                param[key] = slider.value
        cp = widget.ids.get("cp_light_ray_color")
        if cp is not None:
            param['light_ray_color_hue'] = cp.ids['slider_hue'].value
            param['light_ray_color_lum'] = cp.ids['slider_lum'].value
            param['light_ray_color_sat'] = cp.ids['slider_sat'].value
        self.after_set2param(param, widget)

    def after_set2widget(self, widget, param):
        if self.light_rays_canvas is not None:
            self.light_rays_canvas.set_creation_params(self._guide_params_from_param(param))
            self.light_rays_canvas.set_guides(self._get_param(param, 'light_ray_guides'))
            self.light_rays_canvas.select_index(self._selected_guide_index(param, self.light_rays_canvas.guides))
            editor_type = self._normalize_editor_type(self._get_param(param, 'light_ray_editor_type'))
            self.light_rays_canvas.set_creation(
                editor_type,
                self._normalize_editor_mode(editor_type, self._get_param(param, 'light_ray_editor_mode')),
            )

    def after_set2param(self, param, widget):
        if self.light_rays_canvas is not None:
            editor_type = self._normalize_editor_type(param.get('light_ray_editor_type', 'Line'))
            changed = self.light_rays_canvas.set_creation(
                editor_type,
                self._normalize_editor_mode(editor_type, param.get('light_ray_editor_mode', 'Parallel')),
                apply_to_selected=True,
            )
            if self.light_rays_canvas.set_active_params(self._guide_params_from_param(param)):
                changed = True
            guides = self.light_rays_canvas.get_guides()
            if self._ensure_projection_lengths(guides, param):
                changed = True
            if changed:
                param['light_ray_guides'] = guides
            selected = self.light_rays_canvas.selected_index()
            param['light_ray_selected'] = selected
        else:
            guides = self._copy_light_ray_guides(self._get_param(param, 'light_ray_guides') or [])
            if self._apply_active_params_to_guide_list(param, guides, update_mode=True):
                param['light_ray_guides'] = guides

    def _view_param(self, param, widget=None, efconfig=None):
        base = None
        if widget is not None:
            base = getattr(widget, 'primary_param', None)
        view = base.copy() if isinstance(base, dict) and base.get('original_img_size') is not None else param.copy()
        if view.get('original_img_size') is None:
            view['original_img_size'] = param.get('original_img_size')
        disp_info = getattr(efconfig, 'disp_info', None) if efconfig is not None else None
        if disp_info is not None and view.get('original_img_size') is not None:
            params.set_disp_info(view, disp_info)
        return view

    def _open_light_rays_canvas(self, param, widget):
        if self.light_rays_canvas is None:
            from widgets.light_rays_canvas import LightRaysCanvas
            self.light_rays_canvas = LightRaysCanvas(
                guides=self._get_param(param, 'light_ray_guides'),
                callback=self._light_rays_callback,
            )
            widget.ids["preview_widget"].add_widget(self.light_rays_canvas, index=0)
        self.light_rays_canvas.set_primary_param(self._view_param(param, widget=widget))
        self.light_rays_canvas.set_guides(self._get_param(param, 'light_ray_guides'))
        self.light_rays_canvas.select_index(self._selected_guide_index(param, self.light_rays_canvas.guides))
        self.light_rays_canvas.set_creation_params(self._guide_params_from_param(param))
        self.light_rays_canvas.set_creation(
            self._get_param(param, 'light_ray_editor_type'),
            self._get_param(param, 'light_ray_editor_mode'),
        )

    def _close_light_rays_canvas(self, param, widget):
        if self.light_rays_canvas is not None:
            pw = widget.ids.get("preview_widget")
            if pw is not None:
                pw.remove_widget(self.light_rays_canvas)
            self.light_rays_canvas = None

    def _light_rays_callback(self, proc, widget):
        if self.light_rays_callback is not None:
            self.light_rays_callback(proc, widget)

    @staticmethod
    def _guide_hash(guides):
        items = []
        for guide in guides or []:
            if not isinstance(guide, dict):
                continue
            item = [str(guide.get('type', '')), str(guide.get('mode', ''))]
            for key in ('p', 'p1', 'p2'):
                p = guide.get(key)
                try:
                    item.append((round(float(p[0]), 6), round(float(p[1]), 6)))
                except Exception:
                    item.append(None)
            if 'angle' in guide:
                item.append(round(float(guide.get('angle')), 4))
            guide_params = guide.get('params')
            if isinstance(guide_params, dict):
                item.append(tuple(sorted((str(k), round(float(v), 4)) for k, v in guide_params.items())))
            items.append(tuple(item))
        return tuple(items)

    def _guides_to_pixels(self, guides, work, view):
        tcg_info = params.param_to_tcg_info(view)
        converted = []
        for guide in guides or []:
            if not isinstance(guide, dict):
                continue
            gtype = str(guide.get('type', '')).lower()
            mode = str(guide.get('mode', '')).lower()
            out = {'type': gtype, 'mode': mode}
            try:
                if gtype == 'line':
                    out['p1'] = params.tcg_to_ref_image(*guide['p1'], work, tcg_info, True, True)
                    out['p2'] = params.tcg_to_ref_image(*guide['p2'], work, tcg_info, True, True)
                elif gtype == 'point':
                    out['p'] = params.tcg_to_ref_image(*guide['p'], work, tcg_info, True, True)
                    if guide.get('p2') is not None:
                        out['p2'] = params.tcg_to_ref_image(*guide['p2'], work, tcg_info, True, True)
                    if 'angle' in guide:
                        out['angle'] = guide['angle']
                else:
                    continue
                if isinstance(guide.get('params'), dict):
                    out['params'] = guide['params'].copy()
                    if self._PROJECTION_LENGTH_KEY in guide['params']:
                        out['projection_length'] = guide['params'][self._PROJECTION_LENGTH_KEY]
            except Exception:
                continue
            converted.append(out)
        return converted

    def _scene_size_px(self, work, view):
        try:
            disp = params.get_disp_info(view)
            mag = float(disp[4])
            original = view.get('original_img_size')
            if original is not None:
                width = float(original[0]) * mag
                height = float(original[1]) * mag
            else:
                width = float(disp[2]) * mag
                height = float(disp[3]) * mag
            if width > 0.0 and height > 0.0:
                return (width, height)
        except Exception:
            # disp情報が取得できない/不正な場合は work のサイズにフォールバック
            pass
        return (float(work.shape[1]), float(work.shape[0]))

    def _color_rgb(self, param):
        import colorsys
        h = self._get_param(param, 'light_ray_color_hue') / 360.0
        l = self._get_param(param, 'light_ray_color_lum') / 100.0
        s = self._get_param(param, 'light_ray_color_sat') / 100.0
        return colorsys.hls_to_rgb(h, l, s)

    def make_diff(self, rgb, param, efconfig):
        if not self._get_param(param, 'switch_light_rays'):
            self.diff = None
            self.hash = None
            return self.diff
        guides = self._get_param(param, 'light_ray_guides') or []
        if not guides:
            self.diff = None
            self.hash = None
            return self.diff
        self._ensure_projection_lengths(guides, param)

        view = self._view_param(param, efconfig=efconfig)
        # half-res ドラッグ中は view の disp_info が縮小スケールなので GUI キャンバスへは流さない
        if self.light_rays_canvas is not None and hasattr(self.light_rays_canvas, 'set_primary_param') and not getattr(efconfig, 'drag_half_res', False):
            self.light_rays_canvas.set_primary_param(view)

        try:
            disp_snapshot = tuple(params.get_disp_info(view))
        except Exception:
            disp_snapshot = None
        values = tuple(round(float(self._get_param(param, key)), 4) for key in self._SLIDER_KEYS)
        color_values = (
            round(float(self._get_param(param, 'light_ray_color_hue')), 4),
            round(float(self._get_param(param, 'light_ray_color_lum')), 4),
            round(float(self._get_param(param, 'light_ray_color_sat')), 4),
        )
        param_hash = hash((
            self._guide_hash(guides),
            values,
            color_values,
            disp_snapshot,
            getattr(efconfig, 'upstream_hash', 0),
            round(float(getattr(efconfig, 'resolution_scale', 1.0)), 4),
        ))
        if self.hash == param_hash:
            return self.diff
        self.hash = param_hash

        work = core.type_convert(rgb, np.ndarray)
        work = np.asarray(work, dtype=np.float32)
        pixel_guides = self._guides_to_pixels(guides, work, view)
        if not pixel_guides:
            self.diff = None
            self.hash = None
            return self.diff

        result = work.copy()
        rendered = False
        scene_size_px = self._scene_size_px(work, view)
        for guide in pixel_guides:
            intensity = float(self._guide_value(guide, param, 'light_ray_intensity'))
            if intensity <= 0.0:
                continue
            rendered_guide = light_rays.apply_light_rays(
                work,
                [guide],
                intensity=intensity,
                length=self._guide_value(guide, param, 'light_ray_length'),
                decay=self._guide_value(guide, param, 'light_ray_decay'),
                width=self._guide_value(guide, param, 'light_ray_width'),
                softness=self._guide_value(guide, param, 'light_ray_softness'),
                edge_bias=self._guide_value(guide, param, 'light_ray_edge_bias'),
                spread=self._guide_value(guide, param, 'light_ray_spread'),
                count=self._guide_value(guide, param, 'light_ray_count'),
                density=self._guide_value(guide, param, 'light_ray_density'),
                variation=self._guide_value(guide, param, 'light_ray_variation'),
                fog=self._guide_value(guide, param, 'light_ray_fog'),
                occlusion=self._guide_value(guide, param, 'light_ray_occlusion'),
                color_rgb=self._guide_color_rgb(guide, param),
                seed=int(round(float(self._guide_value(guide, param, 'light_ray_seed')))),
                resolution_scale=float(getattr(efconfig, 'resolution_scale', 1.0)),
                scene_size_px=scene_size_px,
            )
            result += rendered_guide - work
            rendered = True
        self.diff = result if rendered else None
        return self.diff


class UnsharpMaskEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_unsharp_mask', True, "switch_unsharp_mask"),
        SliderBinding('unsharp_mask_amount', 0, "slider_unsharp_mask_amount"),
        SliderBinding('unsharp_mask_sigma', 50, "slider_unsharp_mask_sigma"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_unsharp_mask = self._get_param(param, 'switch_unsharp_mask')
        amount = self._get_param(param, 'unsharp_mask_amount')
        sigma = self._get_param(param, 'unsharp_mask_sigma')
        if switch_unsharp_mask == False or amount == 0:
            self.diff = None
            self.hash = None
        else:
            param_hash = hash((amount, sigma))
            if self.hash != param_hash:
                self.hash = param_hash

                rgb = core.type_convert(rgb, np.ndarray)
                amount = amount / 100.0 * 1.5
                sigma = sigma / 100.0 * 3.0
                self.diff = core.unsharp_mask(rgb, amount, sigma)

        return self.diff


class Mask2Effect(Effect):

    @staticmethod
    def get_param(param, key, default=None):
        if default is not None:
            return param.get(key, default)
        
        return param.get(key, Mask2Effect.get_param_dict(param)[key])

    @staticmethod
    def get_param_dict(param, subname=None):
        param_dict = {
            'switch_mask2_settings': True,
            'mask2_invert': False,
            'mask2_allow_over_one': False,
            'mask2_allow_under_zero': False,
            'switch_mask2_depth': True,
            'mask2_depth_min': 0,
            'mask2_depth_max': 255,
            'switch_mask2_hue': True,
            'mask2_hue_distance': 179,
            'mask2_hue_min': 0,
            'mask2_hue_max': 359,
            'switch_mask2_lum': True,
            'mask2_lum_distance': 255,
            'mask2_lum_min': 0,
            'mask2_lum_max': 255,
            'switch_mask2_sat': True,
            'mask2_sat_distance': 255,
            'mask2_sat_min': 0,
            'mask2_sat_max': 255,
            'switch_mask2_options': True,
            'mask2_blur': 0,
            'mask2_depth_balance': 0,
            'mask2_open_space': 0,
            'mask2_close_space': 0,
            'mask2_freedraw_brush_size': 300,
            'mask2_freedraw_brush_hardness': 100,
            'mask2_polyline_fill': True,
            'switch_mask2_quick_select': True,
            'mask2_edge_refine_mode': 'Off',
            'mask2_edge_refine_radius': 0,
            'mask2_edge_refine_strength': 0,
            'mask2_edge_refine_bias': 0,
            'switch_mask2_draw_effects': True,
            'mask2_blend_mode': 'Normal',
            'mask2_color_dodge': 0,
            'mask2_color_burn': 0,
            'mask2_mix_black': 0,
            'mask2_mix_white': 0,
            'mask2_skin_smooth_amount': 0,
            'mask2_skin_smooth_radius_bias': 0,
            'switch_mask2_face': True,
            'mask2_face_face': True,
            'mask2_face_brows': True,
            'mask2_face_eyes': True,
            'mask2_face_nose': True,
            'mask2_face_mouth': True,
            'mask2_face_lips': True,
            # マスク Geometry (マスク自身を変形)
            'switch_mask_geometry': True,
            'mask_rotation': 0.0,            # degrees
            'mask_flip_mode': 0,             # 0=none, 1=H, 2=V, 3=both
            'mask_translation_x': 0.0,       # 画像短辺基準の比率 [-1, 1]
            'mask_translation_y': 0.0,
            'mask_scale_x': 1.0,
            'mask_scale_y': 1.0,
            'mask_mesh_size': [4, 4],        # Step 4 用 (placeholder)
            'mask_mesh_control_points': {},  # Step 4 用 (placeholder)
            # True: 画像 mesh の CP を都度参照 (画像 mesh が変わったらマスクも追従)
            # False: 自前 mask_mesh_control_points を使う (独立 = 画像 mesh と切り離し)
            'mask_mesh_link_to_image': True,
        }
        if subname == 'mask2_draw_effects':
            return {
                key: param_dict[key]
                for key in (
                    'switch_mask2_draw_effects',
                    'mask2_blend_mode',
                    'mask2_color_dodge',
                    'mask2_color_burn',
                    'mask2_mix_black',
                    'mask2_mix_white',
                    'mask2_skin_smooth_amount',
                    'mask2_skin_smooth_radius_bias',
                )
            }
        if subname == 'mask2_settings':
            return {
                key: param_dict[key]
                for key in (
                    'switch_mask2_settings',
                    'mask2_invert',
                    'mask2_allow_over_one',
                    'mask2_allow_under_zero',
                )
            }
        if subname == 'mask2_depth':
            return {
                key: param_dict[key]
                for key in (
                    'switch_mask2_depth',
                    'mask2_depth_min',
                    'mask2_depth_max',
                )
            }
        if subname == 'mask2_hue':
            return {
                key: param_dict[key]
                for key in (
                    'switch_mask2_hue',
                    'mask2_hue_distance',
                    'mask2_hue_min',
                    'mask2_hue_max',
                )
            }
        if subname == 'mask2_lum':
            return {
                key: param_dict[key]
                for key in (
                    'switch_mask2_lum',
                    'mask2_lum_distance',
                    'mask2_lum_min',
                    'mask2_lum_max',
                )
            }
        if subname == 'mask2_sat':
            return {
                key: param_dict[key]
                for key in (
                    'switch_mask2_sat',
                    'mask2_sat_distance',
                    'mask2_sat_min',
                    'mask2_sat_max',
                )
            }
        if subname == 'mask2_options':
            return {
                key: param_dict[key]
                for key in (
                    'switch_mask2_options',
                    'mask2_blur',
                    'mask2_depth_balance',
                    'mask2_open_space',
                    'mask2_close_space',
                    'mask2_freedraw_brush_size',
                    'mask2_freedraw_brush_hardness',
                    'mask2_polyline_fill',
                )
            }
        if subname == 'mask2_quick_select':
            return {
                key: param_dict[key]
                for key in (
                    'switch_mask2_quick_select',
                    'mask2_edge_refine_mode',
                    'mask2_edge_refine_radius',
                    'mask2_edge_refine_strength',
                    'mask2_edge_refine_bias',
                )
            }
        if subname == 'mask_geometry':
            return {
                key: param_dict[key]
                for key in (
                    'switch_mask_geometry',
                    'mask_rotation',
                    'mask_flip_mode',
                    'mask_translation_x',
                    'mask_translation_y',
                    'mask_scale_x',
                    'mask_scale_y',
                    'mask_mesh_size',
                    'mask_mesh_control_points',
                    'mask_mesh_link_to_image',
                )
            }
        if subname == 'mask2_face':
            return {
                key: param_dict[key]
                for key in (
                    'switch_mask2_face',
                    'mask2_face_face',
                    'mask2_face_brows',
                    'mask2_face_eyes',
                    'mask2_face_nose',
                    'mask2_face_mouth',
                    'mask2_face_lips',
                )
            }
        return param_dict

    def set2widget(self, widget, param):
        widget.ids["switch_mask2_settings"].active = self._get_param(param, 'switch_mask2_settings')
        widget.ids["checkbox_mask2_invert"].active = self._get_param(param, 'mask2_invert')
        widget.ids["checkbox_mask2_allow_over_one"].active = self._get_param(param, 'mask2_allow_over_one')
        widget.ids["checkbox_mask2_allow_under_zero"].active = False
        widget.ids["switch_mask2_depth"].active = self._get_param(param, 'switch_mask2_depth')
        widget.ids["slider_mask2_depth_min"].set_slider_value(self._get_param(param, 'mask2_depth_min'))
        widget.ids["slider_mask2_depth_max"].set_slider_value(self._get_param(param, 'mask2_depth_max'))
        widget.ids["switch_mask2_hue"].active = self._get_param(param, 'switch_mask2_hue')
        widget.ids["slider_mask2_hue_distance"].set_slider_value(self._get_param(param, 'mask2_hue_distance'))
        widget.ids["slider_mask2_hue_range"].set_slider_value([
            self._get_param(param, 'mask2_hue_min'),
            self._get_param(param, 'mask2_hue_max'),
        ])
        widget.ids["switch_mask2_lum"].active = self._get_param(param, 'switch_mask2_lum')
        widget.ids["slider_mask2_lum_distance"].set_slider_value(self._get_param(param, 'mask2_lum_distance'))
        widget.ids["slider_mask2_lum_range"].set_slider_value([
            self._get_param(param, 'mask2_lum_min'),
            self._get_param(param, 'mask2_lum_max'),
        ])
        widget.ids["switch_mask2_sat"].active = self._get_param(param, 'switch_mask2_sat')
        widget.ids["slider_mask2_sat_distance"].set_slider_value(self._get_param(param, 'mask2_sat_distance'))
        widget.ids["slider_mask2_sat_range"].set_slider_value([
            self._get_param(param, 'mask2_sat_min'),
            self._get_param(param, 'mask2_sat_max'),
        ])
        widget.ids["switch_mask2_options"].active = self._get_param(param, 'switch_mask2_options')
        widget.ids["slider_mask2_blur"].set_slider_value(self._get_param(param, 'mask2_blur'))
        widget.ids["slider_mask2_depth_balance"].set_slider_value(self._get_param(param, 'mask2_depth_balance'))
        widget.ids["slider_mask2_open_space"].set_slider_value(self._get_param(param, 'mask2_open_space'))
        widget.ids["slider_mask2_close_space"].set_slider_value(self._get_param(param, 'mask2_close_space'))
        widget.ids["slider_mask2_freedraw_brush_size"].set_slider_value(self._get_param(param, 'mask2_freedraw_brush_size'))
        widget.ids["slider_mask2_freedraw_brush_hardness"].set_slider_value(self._get_param(param, 'mask2_freedraw_brush_hardness'))
        widget.ids["checkbox_mask2_polyline_fill"].active = self._get_param(param, 'mask2_polyline_fill')
        widget.ids["switch_mask2_quick_select"].active = self._get_param(param, 'switch_mask2_quick_select')
        edge_refine_mode = self._get_param(param, 'mask2_edge_refine_mode')
        if edge_refine_mode in ('Refine', 'Grow', 'Grow + Islands', 'Lock'):
            edge_refine_mode = 'Quick Select'
        widget.ids["spinner_mask2_edge_refine_mode"].set_text(edge_refine_mode)
        widget.ids["slider_mask2_edge_refine_radius"].set_slider_value(self._get_param(param, 'mask2_edge_refine_radius'))
        widget.ids["slider_mask2_edge_refine_strength"].set_slider_value(self._get_param(param, 'mask2_edge_refine_strength'))
        widget.ids["slider_mask2_edge_refine_bias"].set_slider_value(self._get_param(param, 'mask2_edge_refine_bias'))
        widget.ids["switch_mask2_draw_effects"].active = self._get_param(param, 'switch_mask2_draw_effects')
        widget.ids["spinner_mask2_blend_mode"].set_text(self._get_param(param, 'mask2_blend_mode'))
        widget.ids["slider_mask2_color_dodge"].set_slider_value(self._get_param(param, 'mask2_color_dodge'))
        widget.ids["slider_mask2_color_burn"].set_slider_value(self._get_param(param, 'mask2_color_burn'))
        widget.ids["slider_mask2_mix_black"].set_slider_value(self._get_param(param, 'mask2_mix_black'))
        widget.ids["slider_mask2_mix_white"].set_slider_value(self._get_param(param, 'mask2_mix_white'))
        widget.ids["slider_mask2_skin_smooth_amount"].set_slider_value(self._get_param(param, 'mask2_skin_smooth_amount'))
        widget.ids["slider_mask2_skin_smooth_radius_bias"].set_slider_value(self._get_param(param, 'mask2_skin_smooth_radius_bias'))
        widget.ids["switch_mask2_face"].active = self._get_param(param, 'switch_mask2_face')
        widget.ids["checkbox_mask2_face_face"].active = self._get_param(param, 'mask2_face_face')
        widget.ids["checkbox_mask2_face_brows"].active = self._get_param(param, 'mask2_face_brows')
        widget.ids["checkbox_mask2_face_eyes"].active = self._get_param(param, 'mask2_face_eyes')
        widget.ids["checkbox_mask2_face_nose"].active = self._get_param(param, 'mask2_face_nose')
        widget.ids["checkbox_mask2_face_mouth"].active = self._get_param(param, 'mask2_face_mouth')
        widget.ids["checkbox_mask2_face_lips"].active = self._get_param(param, 'mask2_face_lips')
        # Mask Geometry は MaskGeometryEffect が担当するためここでは扱わない

    def set2param(self, param, widget):
        param['switch_mask2_settings'] = widget.ids["switch_mask2_settings"].active
        param['mask2_invert'] = widget.ids["checkbox_mask2_invert"].active
        param['mask2_allow_over_one'] = widget.ids["checkbox_mask2_allow_over_one"].active
        param['mask2_allow_under_zero'] = False
        param['switch_mask2_depth'] = widget.ids["switch_mask2_depth"].active
        param['mask2_depth_min'] = widget.ids["slider_mask2_depth_min"].value
        param['mask2_depth_max'] = widget.ids["slider_mask2_depth_max"].value
        param['switch_mask2_hue'] = widget.ids["switch_mask2_hue"].active
        param['mask2_hue_distance'] = widget.ids["slider_mask2_hue_distance"].value
        hue_values = list(widget.ids["slider_mask2_hue_range"].ids["slider"].values)
        if len(hue_values) >= 2:
            param['mask2_hue_min'] = hue_values[0]
            param['mask2_hue_max'] = hue_values[-1]
        else:
            param['mask2_hue_min'] = widget.ids["slider_mask2_hue_range"].value
            param['mask2_hue_max'] = widget.ids["slider_mask2_hue_range"].value
        param['switch_mask2_lum'] = widget.ids["switch_mask2_lum"].active
        param['mask2_lum_distance'] = widget.ids["slider_mask2_lum_distance"].value
        lum_values = list(widget.ids["slider_mask2_lum_range"].ids["slider"].values)
        if len(lum_values) >= 2:
            param['mask2_lum_min'] = lum_values[0]
            param['mask2_lum_max'] = lum_values[-1]
        else:
            param['mask2_lum_min'] = widget.ids["slider_mask2_lum_range"].value
            param['mask2_lum_max'] = widget.ids["slider_mask2_lum_range"].value
        param['switch_mask2_sat'] = widget.ids["switch_mask2_sat"].active
        param['mask2_sat_distance'] = widget.ids["slider_mask2_sat_distance"].value
        sat_values = list(widget.ids["slider_mask2_sat_range"].ids["slider"].values)
        if len(sat_values) >= 2:
            param['mask2_sat_min'] = sat_values[0]
            param['mask2_sat_max'] = sat_values[-1]
        else:
            param['mask2_sat_min'] = widget.ids["slider_mask2_sat_range"].value
            param['mask2_sat_max'] = widget.ids["slider_mask2_sat_range"].value
        param['switch_mask2_options'] = widget.ids["switch_mask2_options"].active
        param['mask2_blur'] = widget.ids["slider_mask2_blur"].value
        param['mask2_depth_balance'] = widget.ids["slider_mask2_depth_balance"].value
        param['mask2_open_space'] = widget.ids["slider_mask2_open_space"].value
        param['mask2_close_space'] = widget.ids["slider_mask2_close_space"].value
        param['mask2_freedraw_brush_size'] = widget.ids["slider_mask2_freedraw_brush_size"].value
        param['mask2_freedraw_brush_hardness'] = widget.ids["slider_mask2_freedraw_brush_hardness"].value
        param['mask2_polyline_fill'] = widget.ids["checkbox_mask2_polyline_fill"].active
        param['switch_mask2_quick_select'] = widget.ids["switch_mask2_quick_select"].active
        param['mask2_edge_refine_mode'] = widget.ids["spinner_mask2_edge_refine_mode"].text
        param['mask2_edge_refine_radius'] = widget.ids["slider_mask2_edge_refine_radius"].value
        param['mask2_edge_refine_strength'] = widget.ids["slider_mask2_edge_refine_strength"].value
        param['mask2_edge_refine_bias'] = widget.ids["slider_mask2_edge_refine_bias"].value
        param['switch_mask2_draw_effects'] = widget.ids["switch_mask2_draw_effects"].active
        param['mask2_blend_mode'] = widget.ids["spinner_mask2_blend_mode"].text
        param['mask2_color_dodge'] = widget.ids["slider_mask2_color_dodge"].value
        param['mask2_color_burn'] = widget.ids["slider_mask2_color_burn"].value
        param['mask2_mix_black'] = widget.ids["slider_mask2_mix_black"].value
        param['mask2_mix_white'] = widget.ids["slider_mask2_mix_white"].value
        param['mask2_skin_smooth_amount'] = widget.ids["slider_mask2_skin_smooth_amount"].value
        param['mask2_skin_smooth_radius_bias'] = widget.ids["slider_mask2_skin_smooth_radius_bias"].value
        param['switch_mask2_face'] = widget.ids["switch_mask2_face"].active
        param['mask2_face_face'] = widget.ids["checkbox_mask2_face_face"].active
        param['mask2_face_brows'] = widget.ids["checkbox_mask2_face_brows"].active
        param['mask2_face_eyes'] = widget.ids["checkbox_mask2_face_eyes"].active
        param['mask2_face_nose'] = widget.ids["checkbox_mask2_face_nose"].active
        param['mask2_face_mouth'] = widget.ids["checkbox_mask2_face_mouth"].active
        param['mask2_face_lips'] = widget.ids["checkbox_mask2_face_lips"].active
        # Mask Geometry は MaskGeometryEffect が担当するためここでは扱わない

    def make_diff(self, rgb, param, efconfig):
        """
        invert = self._get_param(param, 'mask2_invert')
        dmin = self._get_param(param, 'mask2_depth_min')
        dmax = self._get_param(param, 'mask2_depth_max')
        hdis = self._get_param(param, 'mask2_hue_distance')
        hmin = self._get_param(param, 'mask2_hue_min')
        hmax = self._get_param(param, 'mask2_hue_max')
        ldis = self._get_param(param, 'mask2_lum_distance')
        lmin = self._get_param(param, 'mask2_lum_min')
        lmax = self._get_param(param, 'mask2_lum_max')
        sdis = self._get_param(param, 'mask2_sat_distance')
        smin = self._get_param(param, 'mask2_sat_min')
        smax = self._get_param(param, 'mask2_sat_max')
        blur = self._get_param(param, 'mask2_blur')
        face_face = self._get_param(param, 'mask2_face_face')
        face_brows = self._get_param(param, 'mask2_face_brows')
        face_eyes = self._get_param(param, 'mask2_face_eyes')
        face_nose = self._get_param(param, 'mask2_face_nose')
        face_mouth = self._get_param(param, 'mask2_face_mouth')
        face_lips = self._get_param(param, 'mask2_face_lips')
        open_space = self._get_param(param, 'mask2_open_space')
        close_space = self._get_param(param, 'mask2_close_space')
        if  (invert == False and dmin == 0 and dmax == 255 and
             hdis == 179 and hmin == 0 and hmax == 359 and
             ldis == 255 and lmin == 0 and lmax == 255 and
             sdis == 255 and smin == 0 and smax == 255 and
             blur == 0):
            self.diff = None
            self.hash = None
        else:        
            param_hash = hash((invert, dmin, dmax, hdis, hmin, hmax, ldis, lmin, lmax, sdis, smin, smax, blur))
            if self.hash != param_hash:
                self.hash = param_hash
                
                self.diff = None
        """
        return self.diff

class MaskGeometryEffect(Effect):
    """マスク自身の Geometry 変形パラメータ専用 Effect。
    Mask2Effect とは分離して、CompositMask 直下に書き込むことで
    『どのマスクを選択していても Composit のパラメータとして機能』する設計に合わせる。"""

    @staticmethod
    def get_param(param, key, default=None):
        if default is not None:
            return param.get(key, default)
        # 共通の既定値は Mask2Effect.get_param_dict が保有しているのでそちらに委譲
        return Mask2Effect.get_param(param, key)

    def get_param_dict(self, param, subname=None):
        # 既定値は Mask2Effect 側に一括定義されている。重複定義を避けるため流用する。
        full = Mask2Effect.get_param_dict(param)
        return {
            key: full[key]
            for key in (
                'switch_mask_geometry',
                'mask_rotation',
                'mask_flip_mode',
                'mask_translation_x',
                'mask_translation_y',
                'mask_scale_x',
                'mask_scale_y',
                'mask_mesh_size',
                'mask_mesh_control_points',
                'mask_mesh_link_to_image',
            )
        }

    def set2widget(self, widget, param):
        if "switch_mask_geometry" not in widget.ids:
            return
        widget.ids["switch_mask_geometry"].active = self._get_param(param, 'switch_mask_geometry')
        widget.ids["slider_mask_rotation"].set_slider_value(self._get_param(param, 'mask_rotation'))
        flip = int(self._get_param(param, 'mask_flip_mode'))
        widget.ids["checkbox_mask_flip_h"].active = bool(flip & 1)
        widget.ids["checkbox_mask_flip_v"].active = bool(flip & 2)
        widget.ids["slider_mask_translation_x"].set_slider_value(self._get_param(param, 'mask_translation_x') * 100.0)
        widget.ids["slider_mask_translation_y"].set_slider_value(self._get_param(param, 'mask_translation_y') * 100.0)
        widget.ids["slider_mask_scale_x"].set_slider_value(self._get_param(param, 'mask_scale_x') * 100.0)
        widget.ids["slider_mask_scale_y"].set_slider_value(self._get_param(param, 'mask_scale_y') * 100.0)

    def set2param(self, param, widget):
        if "switch_mask_geometry" not in widget.ids:
            return
        param['switch_mask_geometry'] = widget.ids["switch_mask_geometry"].active
        param['mask_rotation'] = float(widget.ids["slider_mask_rotation"].value)
        flip = 0
        if widget.ids["checkbox_mask_flip_h"].active:
            flip |= 1
        if widget.ids["checkbox_mask_flip_v"].active:
            flip |= 2
        param['mask_flip_mode'] = flip
        param['mask_translation_x'] = float(widget.ids["slider_mask_translation_x"].value) / 100.0
        param['mask_translation_y'] = float(widget.ids["slider_mask_translation_y"].value) / 100.0
        param['mask_scale_x'] = float(widget.ids["slider_mask_scale_x"].value) / 100.0
        param['mask_scale_y'] = float(widget.ids["slider_mask_scale_y"].value) / 100.0

    def make_diff(self, img, param, efconfig):
        # 実際の変形は cores/mask2/mask_geometry.py が CompositMask.get_mask_image から呼ぶ。
        return None


class GrainEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_grain', True, "switch_grain"),
        SliderBinding('grain_amount', 0, "slider_grain_amount"),
        SliderBinding('grain_size', 0, "slider_grain_size"),
        SliderBinding('grain_roughness', 50, "slider_grain_roughness"),
        SliderBinding('grain_shadow', 60, "slider_grain_shadow"),
        SliderBinding('grain_highlight', 30, "slider_grain_highlight"),
        SliderBinding('grain_color', 10, "slider_grain_color"),
        SliderBinding('grain_seed', 0, "slider_grain_seed"),
    )

    def make_diff(self, rgb, param, efconfig):
        switch_grain = self._get_param(param, 'switch_grain')
        amount = self._get_param(param, 'grain_amount')
        gs = self._get_param(param, 'grain_size')
        roughness = self._get_param(param, 'grain_roughness')
        shadow = self._get_param(param, 'grain_shadow')
        highlight = self._get_param(param, 'grain_highlight')
        color = self._get_param(param, 'grain_color')
        seed = self._get_param(param, 'grain_seed')
        if switch_grain == False or amount == 0:
            self.diff = None
            self.hash = None
        else:
            rgb = core.type_convert(rgb, np.ndarray)
            param_hash = hash((amount, gs, roughness, shadow, highlight, color, seed, rgb.shape, efconfig.resolution_scale))
            if self.hash != param_hash:
                self.hash = param_hash

                size_px = 0.60 + 6.40 * ((float(gs) / 100.0) ** 1.4)
                size_px *= max(0.35, float(efconfig.resolution_scale))
                self.diff = film_grain.apply_film_grain(
                    rgb,
                    amount=amount,
                    grain_size=size_px,
                    roughness=roughness,
                    shadow=shadow,
                    highlight=highlight,
                    color=color,
                    seed=seed,
                )
        
        return self.diff
    
class VignetteEffect(Effect):
    param_bindings = (
        SwitchBinding('switch_vignette', True, "switch_vignette"),
        SliderBinding('vignette_intensity', 0, "slider_vignette_intensity"),
        SliderBinding('vignette_radius_percent', 80, "slider_vignette_radius_percent"),
        SliderBinding('vignette_softness', 80, "slider_vignette_softness"),
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vignette_mask_key = None
        self._vignette_mask = None

    def make_diff(self, rgb, param, efconfig):
        switch_vignette = self._get_param(param, 'switch_vignette')
        vi = self._get_param(param, 'vignette_intensity')
        vr = self._get_param(param, 'vignette_radius_percent')
        vs = self._get_param(param, 'vignette_softness')
        pce = getattr(efconfig, 'crop_editing', False)
        if switch_vignette == False or vi == 0 or pce == True:
            self.diff = None
            self.hash = None
            self._vignette_mask_key = None
            self._vignette_mask = None

        else:
            if efconfig.mode == EffectMode.EXPORT:
                offset_x, offset_y = 0, 0
            else:
                _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(config.get_config('preview_width'), config.get_config('preview_height'), efconfig.disp_info)

            rgb = core.type_convert(rgb, np.ndarray)
            crop_rect = params.get_crop_rect(param)
            if crop_rect is None:
                ow, oh = param.get('original_img_size', (rgb.shape[1], rgb.shape[0]))
                crop_rect = (0, 0, ow, oh)
            vs = (100 - vs) / 100.0 * 3.0 + 1.0  # 1.0-4.0
            disp_info = tuple(float(v) for v in efconfig.disp_info)
            crop_rect = tuple(float(v) for v in crop_rect)
            offset = (float(offset_x), float(offset_y))
            mask_key = (rgb.shape[0], rgb.shape[1], float(vr), float(vs), disp_info, crop_rect, offset)
            if self._vignette_mask_key != mask_key or self._vignette_mask is None:
                self._vignette_mask_key = mask_key
                self._vignette_mask = backend_vignette.create_vignette_mask(
                    rgb.shape[0],
                    rgb.shape[1],
                    vr,
                    disp_info,
                    crop_rect,
                    offset,
                    vs,
                )

            self.hash = hash((vi, mask_key, getattr(efconfig, 'upstream_hash', None)))
            self.diff = backend_vignette.apply_vignette_mask(rgb, self._vignette_mask, vi)
        
        return self.diff
    

def bind_ai_job_manager(effect_sets, ai_job_manager):
    try:
        effect_sets[0]['ai_noise_reduction'].ai_job_manager = ai_job_manager
    except Exception:
        logging.exception("failed to bind AI job manager to AI-NR effect")


class LensGhostEffect(Effect):
    """レンズゴースト(配線検証用ダミー)。

    1つのインスタンスを lv1(lensblur_filter の前)と lv2(最後)の両方に登録する。
    どちらで実行するかは param['lens_ghost_position'] ('lv1' / 'lv2')でユーザーが選択し、
    efconfig.current_level が自身の選択レベルと一致するときだけ実行する(2回実行しない)。
    UI は main.kv の Li(Liquify) タブ末尾に配置。make_diff は現状ダミー(赤み加算)で、
    本実装で create_ghost 呼び出しに差し替える。
    """

    param_bindings = (
        SwitchBinding('lens_ghost_enabled', True, 'switch_lens_ghost', widget_attr='enabled'),
        SpinnerTextBinding('lens_ghost_position', 'lv2', 'spinner_lens_ghost_position'),
        # CP配置直後の初期値: 単純(単環・トゲ無し・ビンテージ無し)で少し大きめ・はっきり見える設定。
        SliderBinding('global_intensity', 0.5, 'slider_global_intensity'),
        SliderBinding('vintage_amount', 0.0, 'slider_vintage_amount'),
        SliderBinding('vintage_hue', 0.0, 'slider_vintage_hue'),
        SliderBinding('base_radius', 400, 'slider_base_radius'),
        SliderBinding('num_components', 1, 'slider_num_components'),
        SliderBinding('chromatic_aberration_strength', 0.6, 'slider_chromatic_aberration_strength'),
        SliderBinding('ghost_ring_thickness', 0.35, 'slider_ghost_ring_thickness'),
        SliderBinding('offset_ratio', 2.0, 'slider_offset_ratio'),
        SliderBinding('spike_strength', 0.0, 'slider_spike_strength'),
        SliderBinding('spike_density', 600, 'slider_spike_density'),
        SliderBinding('spike_randomness', 0.6, 'slider_spike_randomness'),
        SliderBinding('source_fade', 0.6, 'slider_source_fade'),
        SliderBinding('blur_sigma', 3.0, 'slider_blur_sigma'),
        SliderBinding('max_eccentricity', 0.0, 'slider_max_eccentricity'),
        SliderBinding('perspective_distortion', 0.0, 'slider_perspective_distortion'),
        SliderBinding('spherical_aberration_strength', 0.0, 'slider_spherical_aberration_strength'),
        SliderBinding('random_seed', 45, 'slider_random_seed'),
    )

    LEVEL_BY_POSITION = {'lv1': 1, 'lv2': 2}

    # create_ghost に渡すスライダー系パラメータ名
    _SLIDER_KEYS = (
        'global_intensity', 'vintage_amount', 'vintage_hue', 'base_radius', 'num_components',
        'chromatic_aberration_strength', 'ghost_ring_thickness', 'offset_ratio', 'spike_strength',
        'spike_density', 'spike_randomness', 'source_fade', 'blur_sigma', 'max_eccentricity',
        'perspective_distortion', 'spherical_aberration_strength', 'random_seed',
    )
    _INT_KEYS = ('base_radius', 'num_components', 'spike_density', 'random_seed')

    def __init__(self, ghost_callback=None, **kwargs):
        super().__init__(**kwargs)
        self.ghost_canvas = None
        self.ghost_callback = ghost_callback

    def set_ghost_callback(self, callback):
        self.ghost_callback = callback

    def get_param_dict(self, param):
        d = super().get_param_dict(param)
        # 光源ごとに独立パラメータ: {'pos': (cx,cy 正規化TCG), 'params': {16スライダーキー}}
        d['lens_ghost_lights'] = []
        d['lens_ghost_selected'] = -1   # 選択中の光源 index(スライダーに反映)
        d['lens_ghost_coords'] = []     # 旧形式(移行用に残置)
        return d

    def _lights(self, param):
        """光源リストを取得(旧 lens_ghost_coords からの移行込み)。各要素 {'pos','params'}。"""
        lights = param.get('lens_ghost_lights')
        if not isinstance(lights, list):
            lights = []
            param['lens_ghost_lights'] = lights
        if not lights:
            coords = param.get('lens_ghost_coords')
            if coords:
                base = {k: self._get_param(param, k) for k in self._SLIDER_KEYS}
                for c in coords:
                    lights.append({'pos': tuple(c), 'params': dict(base)})
                param['lens_ghost_selected'] = len(lights) - 1
                param['lens_ghost_coords'] = []
        return lights

    def light_params(self, param, idx):
        """指定 index の光源 params(無ければ None)。"""
        lights = self._lights(param)
        if 0 <= idx < len(lights):
            return lights[idx].get('params')
        return None

    def set2param(self, param, widget):
        super().set2param(param, widget)
        # flat スライダーキー = 選択中光源のバッファ。編集を選択光源へ反映。
        sel = param.get('lens_ghost_selected', -1)
        lights = self._lights(param)
        if 0 <= sel < len(lights):
            lights[sel]['params'] = {k: param.get(k) for k in self._SLIDER_KEYS}

    def target_level(self, param):
        return self.LEVEL_BY_POSITION.get(self._get_param(param, 'lens_ghost_position'), 2)

    # 編集中の canvas へ param の CP を復元
    def after_set2widget(self, widget, param):
        if self.ghost_canvas is not None and hasattr(self.ghost_canvas, 'set_coords'):
            self.ghost_canvas.set_coords([L['pos'] for L in self._lights(param)])
            self.ghost_canvas.selected = param.get('lens_ghost_selected', -1)

    def _view_param(self, param, widget=None, efconfig=None):
        """TCG 変換用の座標コンテキスト。disp_info はライブビュー(efconfig/primary)から取る(liquify と同方針)。"""
        base = None
        if widget is not None:
            base = getattr(widget, 'primary_param', None)
        view = base.copy() if isinstance(base, dict) and base.get('original_img_size') is not None else param.copy()
        if view.get('original_img_size') is None:
            view['original_img_size'] = param.get('original_img_size')
        disp_info = getattr(efconfig, 'disp_info', None) if efconfig is not None else None
        if disp_info is not None and view.get('original_img_size') is not None:
            params.set_disp_info(view, disp_info)
        return view

    # --- canvas ライフサイクル(main._sync_effect_editors から呼ぶ。DistortionEffect を範に) ---
    def _open_ghost_canvas(self, param, widget):
        # param は「保存先」(mask2時は Composit param)。canvas はこの param の CP/tcg_info を反映する。
        positions = [L['pos'] for L in self._lights(param)]
        if self.ghost_canvas is None:
            from widgets.ghost_canvas import LensGhostCanvas
            self.ghost_canvas = LensGhostCanvas(
                coords=positions,
                callback=self._ghost_callback,
            )
            widget.ids["preview_widget"].add_widget(self.ghost_canvas, index=0)
        self.ghost_canvas.set_primary_param(self._view_param(param, widget=widget))
        self.ghost_canvas.set_coords(positions)
        self.ghost_canvas.selected = param.get('lens_ghost_selected', -1)

    def _close_ghost_canvas(self, param, widget):
        if self.ghost_canvas is not None:
            pw = widget.ids.get("preview_widget")
            if pw is not None:
                pw.remove_widget(self.ghost_canvas)
            self.ghost_canvas = None

    def _ghost_callback(self, proc, widget):
        if self.ghost_callback is not None:
            self.ghost_callback(proc, widget)

    def make_diff(self, rgb, param, efconfig):
        target = self.target_level(param)
        current = getattr(efconfig, 'current_level', None)
        # 選択レベル以外では何もしない(他レベルの cache 保持のため hash/diff は触らない)。
        if current != target:
            return None

        # ライブビュー(zoom/pan/Ge)の座標系。毎描画で canvas を同期して CP マーカーを追従させる。
        # half-res ドラッグ中は view の disp_info が縮小スケールなので GUI キャンバスへは流さない。
        view = self._view_param(param, efconfig=efconfig)
        if self.ghost_canvas is not None and hasattr(self.ghost_canvas, 'set_primary_param') and not getattr(efconfig, 'drag_half_res', False):
            self.ghost_canvas.set_primary_param(view)

        enabled = self._get_param(param, 'lens_ghost_enabled')
        lights = self._lights(param)
        if not enabled or not lights:
            return None

        res = float(getattr(efconfig, 'resolution_scale', 1.0))
        param_hash = hash((
            'lens_ghost', target,
            tuple((
                round(float(L['pos'][0]), 6), round(float(L['pos'][1]), 6),
                tuple(round(float((L.get('params') or {}).get(k, 0.0)), 6) for k in self._SLIDER_KEYS),
            ) for L in lights),
            round(res, 4),
            params.get_disp_info(view),   # zoom/pan/Ge でビューが変われば再描画(表示位置追従)
            getattr(efconfig, 'upstream_hash', 0),
        ))
        if self.hash == param_hash:
            return self.diff
        self.hash = param_hash

        work = core.type_convert(rgb, np.ndarray)
        work = np.asarray(work, dtype=np.float32).copy()

        tcg_info = params.param_to_tcg_info(view)
        # レンズ光学中心(=画像中心 TCG(0,0))を work 座標へ。拡大時に work の (w/2,h/2) ではズレる。
        lens_cx, lens_cy = params.tcg_to_ref_image(0.0, 0.0, work, tcg_info, True, True)

        from cores.lens_ghost import create_ghost
        result = work
        for L in lights:
            cx, cy = L['pos']
            # Liquify(replay_recorded)と同じ変換(apply_ref_img_divide=True)。拡大/Ge追従。
            x, y = params.tcg_to_ref_image(cx, cy, work, tcg_info, True, True)
            sl = dict(L.get('params') or {})
            for k in self._SLIDER_KEYS:
                if k not in sl:
                    sl[k] = self._get_param(param, k)   # 欠損キーはデフォルト
            for k in self._INT_KEYS:
                sl[k] = int(round(sl[k]))
            sl['base_radius'] = max(1, int(round(sl['base_radius'] * res)))
            sl['blur_sigma'] = float(sl['blur_sigma']) * res
            # 光源ごとに独立パラメータでレンダーし、走行結果へ合成。
            result = create_ghost(result, [(int(round(x)), int(round(y)))],
                                  lens_center=(lens_cx, lens_cy), **sl)
        self.diff = result
        return self.diff


def create_effects(lens_modifier_callback=None, geometry_callback=None, distortion_callback=None, crop_callback=None, ai_job_manager=None, view_param_provider=None, ghost_callback=None, light_rays_callback=None):
    effects = [{}, {}, {}, {}, {}]

    lv0 = effects[0]
    lv0['loading_wait'] = LoadingWaitEffect()
    lv0['ai_noise_reduction'] = AINoiseReductonEffect(ai_job_manager=ai_job_manager)
    lv0['remove_chromatic_aberration'] = RemoveChromaticAberrationEffect()
    lv0['lens_modifier'] = LensModifierEffect(lens_modifier_callback=lens_modifier_callback)
    lv0['subpixel_shift'] = SubpixelShiftEffect()
    lv0['exposure_fusion_debevec'] = ExposureFusionDebevecEffect()
    lv0['inpaint'] = InpaintEffect()
    lv0['patchmatch_inpaint'] = PatchmatchInpaintEffect()
    lv0['cross_filter'] = CrossFilterEffect()
    lv0['color_match'] = ColorMatchEffect()
    lv0['geometry'] = GeometryEffect(geometry_callback=geometry_callback)
    lv0['crop'] = CropEffect(crop_callback=crop_callback)

    lv1 = effects[1]
    lv1['face'] = FaceEffect()
    lv1['distortion'] = DistortionEffect(
        distortion_callback=distortion_callback,
        view_param_provider=view_param_provider,
    )
    # レンズゴースト: 1インスタンスを lv1(lensblur前)と lv2(最後)に登録。実行レベルは設定で選択。
    _lens_ghost = LensGhostEffect(ghost_callback=ghost_callback)
    lv1['lens_ghost'] = _lens_ghost
    lv1['lensblur_filter'] = LensblurFilterEffect()
    lv1['scratch'] = ScratchEffect()
    lv1['frosted_glass'] = FrostedGlassEffect()
    lv1['mosaic'] = MosaicEffect()
    
    lv2 = effects[2]
    lv2['color_temperature'] = ColorTemperatureEffect()

    # 旧 lv2['auto_exposure'] = AutoExposureEffect() は input_lut(stage="input") に統合済み。
    lv2['input_lut'] = LUTEffect(stage="input")

    lv2['exposure'] = ExposureEffect()
    lv2['contrast'] = ContrastEffect()
    lv2['tone'] = ToneEffect()
    lv2['level'] = LevelEffect()
    lv2['curves'] = CurvesEffect()

    lv2['dehaze'] = DehazeEffect()
    lv2['light_noise_reduction'] = LightNoiseReductionEffect()
    lv2['clarity'] = ClarityEffect()
    lv2['texture'] = TextureEffect()
    lv2['microcontrast'] = MicroContrastEffect()
    lv2['color_separation'] = ColorSeparationEffect()

    # ここでクリッピング

    #lv2['rgb2hls1'] = RGB2HLSEffect()
    #lv2['hls2rgb1'] = HLS2RGBEffect()

    lv2['clahe'] = CLAHEEffect()

    lv2['rgb2hls2'] = RGB2HLSEffect()
    lv2['hls'] = HLSEffect()
    lv2['vs_and_saturation'] = VSandSaturationEffect()
    lv2['hls2rgb2'] = HLS2RGBEffect()

    # 彩度処理の後でハイライトの色被りを抜き、白をクリーンにする（爽やか方向の補正）
    lv2['clean_highlight'] = CleanHighlightEffect()

    lv2['look_lut'] = LUTEffect(stage="look")
    lv2['lens_simulator'] = LensSimulatorEffect()
    lv2['light_rays'] = LightRaysEffect(light_rays_callback=light_rays_callback)
    lv2['film_emulation'] = FilmSimulationEffect()
    lv2['solid_color'] = SolidColorEffect()
    lv2['orton'] = OrtonEffect()
    lv2['glow'] = GlowEffect()
    lv2['airy_glow'] = AiryGlowEffect()
    lv2['unsharp_mask'] = UnsharpMaskEffect()
    lv2['lens_ghost'] = _lens_ghost  # 同一インスタンスを lv2 の最後にも登録(実行レベルは設定で選択)

    lv3 = effects[3]
    lv3['mask2'] = Mask2Effect()
    lv3['mask_geometry'] = MaskGeometryEffect()

    lv4 = effects[4]
    lv4['vignette'] = VignetteEffect()
    lv4['grain'] = GrainEffect()

    return effects


def set_composit_mask_noop_defaults(param):
    """Composit mask layers start with no image adjustment unless explicitly edited."""
    param.setdefault('ai_noise_reduction', False)
    param.setdefault('light_noise_reduction', 0)
    param.setdefault('light_color_noise_reduction', 0)


def set2widget_all(widget, effects, param, reset_effects=True):
    for dict in effects:
        for l in dict.values():
            l.set2widget(widget, param)
            if reset_effects:
                l.reeffect()

def set2param_all(effects, param, widget):
    for dict in effects:
        for l in dict.values():
            l.set2param(param, widget)
            l.reeffect()

def reeffect_all(effects, lv=0):
    for i, dict in enumerate(effects):
        if i >= lv:
            for l in dict.values():
               l.reeffect()

def finalize_all(effects, param, widget):
    for dict in effects:
        for l in dict.values():
            l.finalize(param, widget)

def delete_default_param_all(effects, param):
    param2 = param.copy()
    for dict in effects:
        for l in dict.values():
            l.delete_default_param(param2)
    return param2

def get_default_param(effects, key, param):
    for dict in effects:
        for l in dict.values():
            if hasattr(l, 'effects'):
                for l2 in l.effects.values():
                    if key in l2.get_param_dict(param):
                        return l2.get_param_dict(param)[key]
            if key in l.get_param_dict(param):
                return l.get_param_dict(param)[key]
    return None
