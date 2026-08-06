if __name__ == '__main__':
    import sys as _sys_early
    import os as _os_early
    import multiprocessing as _mp_early
    import logging as _logging_early

    _SPLASH_ENV = "PLATYPUS_SPLASH_SCREEN"
    _SPLASH_IMAGE = "assets/Shade Wave.png"
    _splash_close_screen = None

    def _display_startup_splash():
        if not env_flag(_SPLASH_ENV, default=True):
            return None
        try:
            from splashscreen import display_splash_screen, close_splash_screen
            display_splash_screen(_SPLASH_IMAGE)
            return close_splash_screen
        except Exception:
            _logging_early.exception("Failed to display startup splash screen")
            return None

    def _close_startup_splash():
        global _splash_close_screen
        if _splash_close_screen is None:
            return
        try:
            _splash_close_screen()
        except Exception:
            _logging_early.exception("Failed to close startup splash screen")
        finally:
            _splash_close_screen = None

    def _install_frozen_startup_logging():
        if not getattr(_sys_early, "frozen", False):
            return None
        try:
            log_dir = _os_early.path.join(
                _os_early.path.expanduser("~/Library/Logs"),
                "Shade Wave",
            )
            _os_early.makedirs(log_dir, exist_ok=True)
            log_path = _os_early.path.join(log_dir, "startup.log")
            root_logger = _logging_early.getLogger()
            root_logger.setLevel(min(root_logger.level or _logging_early.INFO, _logging_early.INFO))
            has_startup_handler = any(
                getattr(handler, "_shade_wave_startup_log", False)
                for handler in root_logger.handlers
            )
            if not has_startup_handler:
                handler = _logging_early.FileHandler(log_path, mode="a", encoding="utf-8")
                handler._shade_wave_startup_log = True
                handler.setFormatter(_logging_early.Formatter(
                    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
                ))
                root_logger.addHandler(handler)

            def _log_uncaught_exception(exc_type, exc_value, exc_traceback):
                _logging_early.critical(
                    "Uncaught exception during frozen app execution",
                    exc_info=(exc_type, exc_value, exc_traceback),
                )

            _sys_early.excepthook = _log_uncaught_exception
            _logging_early.info("Frozen app startup logging initialized")
            return log_path
        except Exception:
            return None

    _frozen_startup_log_path = _install_frozen_startup_logging()
    # PyInstaller: kv/json 等は sys._MEIPASS 配下に同梱される
    if getattr(_sys_early, "frozen", False) and hasattr(_sys_early, "_MEIPASS"):
        _os_early.chdir(_sys_early._MEIPASS)
    # OpenMP ランタイムの情報メッセージを抑える（llvm-openmp / 混在時）
    _os_early.environ.setdefault("OMP_DISPLAY_ENV", "FALSE")
    _os_early.environ.setdefault("KMP_WARNINGS", "0")
    _os_early.environ.setdefault("LIBOMP_VERBOSE", "0")
    _os_early.environ.setdefault("KIVY_NO_ARGS", "1")  # 子プロセスでの Kivy 引数誤解釈を防ぐ
    # Finder から起動した .app は PATH が /usr/bin:/bin:/usr/sbin:/sbin に絞られ、
    # /usr/local/bin (公式 ExifTool) や /opt/homebrew/bin (Homebrew) が含まれない
    if _sys_early.platform == "darwin":
        _extra_paths = [p for p in ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin")
                        if _os_early.path.isdir(p)]
        _path_parts = _os_early.environ.get("PATH", "").split(_os_early.pathsep)
        _missing_paths = [p for p in _extra_paths if p not in _path_parts]
        if _missing_paths:
            _os_early.environ["PATH"] = _os_early.pathsep.join(_missing_paths + _path_parts)
    _mp_early.freeze_support()  # frozen 実行時の multiprocessing 子プロセス分岐を早期処理

    try:
        #import matplotlib
        #matplotlib.use('Agg', force=True)  # matplotlib読み込み前にAgg固定
        #matplotlib.interactive(False)
        pass
    except ImportError:
        # matplotlib import自体が無効化されているため実質到達しない（無害）
        pass

    # PyInstaller + Kivy 既定フックでは tkinter が除外される。バンドル実行時はスキップ。
    """
    if not getattr(_sys_early, "frozen", False):
        import tkinter as tk

        # tk.Tk()で落ちるのを回避するためのパッチ
        tk = tk.Tk()
        tk.withdraw()
        tk.destroy()
    """
    from utils.envutils import env_flag

    _splash_close_screen = _display_startup_splash()

    from kivy.config import Config
    Config.set('input', 'mouse', 'mouse,disable_multitouch')  # 右クリック赤丸消去
    Config.set('kivy', 'exit_on_escape', '0')  # kivy ESC無効
    Config.set('kivy', 'kivy_clock', 'interrupt')
    # Kivy/SDL2 はウィンドウ生成時に SDL_SetWindowIcon を呼び、macOS では
    # [NSApp setApplicationIconImage:] として Dock アイコンを上書きする。未指定だと
    # 既定の Kivy ロゴになるため、ShadeWave アイコンを明示してロゴ化けを防ぐ。
    _APP_ICON = "assets/Shade Wave icon.png"
    if _os_early.path.exists(_APP_ICON):
        Config.set('kivy', 'window_icon', _APP_ICON)

    from kivy.app import App as KVApp
    from kivy.core.window import Window as KVWindow
    from kivy.graphics.texture import Texture as KVTexture
    from kivy.properties import (
        BooleanProperty as KVBooleanProperty,
        ListProperty as KVListProperty,
        NumericProperty as KVNumericProperty,
        StringProperty as KVStringProperty,
    )
    from kivy.clock import Clock as KVClock, mainthread as kvmainthread
    from kivy.graphics.transformation import Matrix as KVMatrix
    from kivy.uix.label import Label as KVLabel
    from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
    from kivy.uix.textinput import TextInput as KVTextInput
    from kivy.metrics import dp as kvdp

    import gc
    import threading
    import threads
    import os

    from effect_backends import coating_adapter
    from effect_backends import colour_functions_adapter as colour_functions
    import re
    import time
    import multiprocessing
    import math
    import logging
    # ログレベルの設定
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("pyvips").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    #logging.getLogger("PIL.TiffImagePlugin").setLevel(logging.WARNING)

    import define
    import auto_adjust
    import cores.core as core
    import params
    from enums import ImageFidelity, LoadStage, coerce_load_stage
    from image_fidelity import pipeline_loading_flag
    import effects
    import pipeline
    import utils.utils as utils
    import utils.kvutils as kvutils
    from utils import perf_trace
    from utils import preset_utils
    from utils import rating_utils
    from utils import rating_io
    import macos as device

    from cores.ai_image_cache import AIImageCache
    from cores import pmck_store
    import config
    import export
    import processing_dialog
    from processing_dialog import create_processing_dialog
    from async_worker import AsyncWorker
    from cores.ai_job_manager import (
        AIJobStatus,
        AI_NOISE_KIND,
        AISidecarMergeQueue,
        AIJobManager,
        ai_noise_enabled,
        current_param_accepts_ai_noise_result,
        merge_ai_noise_result_into_param,
    )
    import waitinfo
    import history

    import widgets.metainfo
    import widgets.float_input
    import widgets.param_slider
    import widgets.color_picker
    import widgets.hover_spinner
    import widgets.histogram
    import widgets.viewer
    import widgets.curve
    import widgets.bbox_viewer
    import widgets.mask_editor2
    import widgets.history_content as history_content
    import widgets.mask2_content as mask2_content
    import widgets.preset_content as preset_content
    from widgets.effect_selector import EffectSelector
    from widgets.export_dialog import ExportDialog, ExportConfirmDialog
    from widgets.sort_filter_dialog import SortFilterDialog
    import widgets.collapsible_box
    import widgets.compact_switch
    import widgets.modern_checkbox
    from widgets.switch_reset_map import build_switch_reset_targets

if __name__ != '__main__':
    class ImportBlocker:
        """特定のモジュールのインポートをブロックする"""
        def __init__(self, blocked_modules):
            self.blocked_modules = set(blocked_modules)
        
        def find_module(self, fullname, path=None):
            # ブロック対象のモジュールかチェック
            for blocked in self.blocked_modules:
                if fullname == blocked or fullname.startswith(blocked + '.'):
                    return self
            return None
        
        def load_module(self, fullname):
            raise ImportError(f"Module '{fullname}' is blocked in child process")

    def init_worker():
        """子プロセス初期化時に実行される"""
        import sys
        import multiprocessing
        import logging
        # インポートフックを設定
        blocker = ImportBlocker(['kivy', 'kivymd', 'matplotlib', 'tkinter'])
        sys.meta_path.insert(0, blocker)
        logging.debug("子プロセス %s: インポートブロッカー設定完了", multiprocessing.current_process().name)
    
    #init_worker()

import os
import copy
import dataclasses
import numpy as np
import cv2

import file_cache_system
import memory_manager

# ヒストグラム計算の間引き率(各軸)。2 なら画素数 1/4 で計算する。
# 512bin のヒストグラム表示にはサンプル数として十分で、視認差は出ない。1 で無効化。
HISTOGRAM_DECIMATION = 2

# drain-all プレビュー(中間バージョンを飛ばさず順に描くモード)で許容する
# 未処理バージョンの最大ラグ。これを超えたら中間フレームを捨てて最新版へ追いつく。
# ドラッグを止めた後に積み残しの描画で UI が固まる(クロップ枠が動かせない)のを防ぐ。
_PREVIEW_DRAIN_MAX_LAG = 2


@dataclasses.dataclass(frozen=True)
class _DrawRequest:
    """draw_image ワーカーに渡す「version + 描画フラグ束」の不変スナップショット。

    アトミック性の不変条件: version とそれに対応する4つのフラグ(center/
    fast_display/skip_histogram/drag_quality)は必ずこのオブジェクト1つに
    まとめて生成し、UI スレッド側は `self._draw_request = _DrawRequest(...)`
    という1回の代入で公開する。ワーカースレッド側も `request = self._draw_request`
    という1回の代入で丸ごと取得してから各フィールドを読む。version とフラグを
    個別に代入/読み取りすると、その間に次の start_draw_image が割り込んで
    「新しい version なのにフラグは古い(またはその逆)」という中途半端な組み
    合わせを観測しうる(GIL はバイトコード単位でしか排他しないため)。
    """
    version: int
    center: object = None
    fast_display: bool = False
    skip_histogram: bool = False
    drag_quality: bool = False


def _debug_display_stats(label, img):
    if os.getenv("PLATYPUS_DEBUG_PIPELINE_STATS", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    try:
        import logging
        arr = np.asarray(img)
        if arr.size == 0:
            logging.warning("[DISPLAY_STATS] %s shape=%s empty", label, getattr(arr, "shape", None))
            return
        finite = np.isfinite(arr)
        finite_count = int(np.count_nonzero(finite))
        total_count = int(arr.size)
        if finite_count == 0:
            logging.warning("[DISPLAY_STATS] %s shape=%s dtype=%s finite=0/%d", label, arr.shape, arr.dtype, total_count)
            return
        finite_values = arr[finite]
        parts = [
            f"[DISPLAY_STATS] {label}",
            f"shape={arr.shape}",
            f"dtype={arr.dtype}",
            f"finite={finite_count}/{total_count}",
            f"min={float(np.min(finite_values)):.6g}",
            f"max={float(np.max(finite_values)):.6g}",
            f"mean={float(np.mean(finite_values)):.6g}",
            f"neg={int(np.count_nonzero(arr < 0))}",
            f"over1={int(np.count_nonzero(arr > 1))}",
        ]
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[..., :3]
            any_neg = np.any(rgb < 0, axis=2)
            any_over = np.any(rgb > 1, axis=2)
            all_over = np.all(rgb > 1, axis=2)
            ch_min = np.nanmin(rgb, axis=(0, 1))
            ch_max = np.nanmax(rgb, axis=(0, 1))
            ch_neg = np.count_nonzero(rgb < 0, axis=(0, 1))
            ch_over = np.count_nonzero(rgb > 1, axis=(0, 1))
            parts.extend([
                "ch_min=({:.6g},{:.6g},{:.6g})".format(*ch_min),
                "ch_max=({:.6g},{:.6g},{:.6g})".format(*ch_max),
                "ch_neg=({},{},{})".format(*ch_neg),
                "ch_over1=({},{},{})".format(*ch_over),
                f"px_any_neg={int(np.count_nonzero(any_neg))}",
                f"px_any_over1={int(np.count_nonzero(any_over))}",
                f"px_all_over1={int(np.count_nonzero(all_over))}",
            ])
        logging.warning(" ".join(parts))
    except Exception:
        import logging
        logging.exception("[DISPLAY_STATS] failed to inspect %s", label)


# OpenCVの設定
cv2.ocl.setUseOpenCL(True)
cv2.setUseOptimized(True)

if __name__ == '__main__':

    def pillow_init():
        import PIL.Image as PILImage
        import PIL.Jpeg2KImagePlugin
        import PIL.JpegImagePlugin
        import PIL.PngImagePlugin
        import PIL.TiffImagePlugin
        import PIL.GifImagePlugin
        import PIL.BmpImagePlugin
        PILImage._initialized = 2
        PILImage.init()

    def _load_stage_allows_ui(stage, imgset):
        """
        プレビュー（RAW 埋め込み）または単発 RGB が表示可能になった段階でのみ True。
        RAW フルデコード完了は fidelity が FULL のときのみ True（それ以外はローディング継続）。
        """
        if stage in (LoadStage.FIRST_PAINTABLE, LoadStage.RGB_DONE):
            return True
        if stage == LoadStage.FULL_DECODE and getattr(imgset, 'fidelity', None) == ImageFidelity.FULL:
            return True
        return False

    def _load_stage_ends_file_loading_indicator(stage, imgset):
        """
        ファイル読み込み用インジケータ（_actively_loading / is_processing）を消す段階。
        プレビュー到達でパラメータを触れる loading 解除とは分離し、RAW はフルデコード完了まで
        スピナーを表示し続ける。
        """
        if stage == LoadStage.FULL_DECODE:
            return True
        if stage == LoadStage.RGB_DONE:
            return True
        if stage == LoadStage.FIRST_PAINTABLE and getattr(imgset, 'fidelity', None) == ImageFidelity.FULL:
            return True
        return False

    class GeModePanel(KVBoxLayout):
        visible = KVBooleanProperty(True)

        def on_touch_down(self, touch):
            if not self.visible:
                return False
            return super().on_touch_down(touch)

        def on_touch_move(self, touch):
            if not self.visible:
                return False
            return super().on_touch_move(touch)

        def on_touch_up(self, touch):
            if not self.visible:
                return False
            return super().on_touch_up(touch)

    class MainWidget(KVBoxLayout):
        # === 読み込み状態フラグの役割分担 ===
        # loading:              on_select 直後 〜 最初のピクセル表示可能 (FIRST_PAINTABLE/RGB_DONE)
        #                       Select/Export 等の「ロード中は触らせたくない」UI を無効化する短期フラグ。
        # image_loaded:         imgset が確定して編集可能な状態か (起動時/フォルダ切替時=False)。
        #                       全パラメータ UI の gate。loading とは独立で、loading=False でも
        #                       image_loaded=False の局面が存在する（起動直後・フォルダ切替直後）。
        # mask2_wait_full_load: RAW プレビュー段階では True、フル復号完了で False。Mask2 専用 gate。
        # _actively_loading:    ファイル選択開始 〜 フル復号完了。スピナーアニメ用 is_processing 算出の入力。
        # is_processing:        _actively_loading or 非同期タスク有り（スピナー表示）。
        # === 内部状態（UI 非連動） ===
        # _last_image_fidelity: 直前 fid。PREVIEW→FULL 遷移時の pmck heavy merge 判定。
        # _expected_file_path:  期待しているファイル。遅延 FCS コールバックを破棄するため。
        loading = KVBooleanProperty(False)
        mask2_wait_full_load = KVBooleanProperty(True)
        preview_size = KVListProperty([100, 100])
        is_processing = KVBooleanProperty(False)
        ai_inpaint_processing = KVBooleanProperty(False)
        export_in_progress = KVBooleanProperty(False)
        export_done = KVNumericProperty(0)
        export_total = KVNumericProperty(0)
        image_loaded = KVBooleanProperty(False)
        is_zoomed = KVBooleanProperty(False)
        zoom_ratio = KVNumericProperty(1.0)
        preview_focus_mode = KVBooleanProperty(False)
        preview_pixel_visible = KVBooleanProperty(False)
        preview_pixel_text = KVStringProperty("")
        preview_pixel_color = KVListProperty([0, 0, 0, 1])

        LEFT_INFO_FRAC_NORMAL = 0.2
        PREVIEW_COL_FRAC_NORMAL = 0.55
        PREVIEW_COL_FRAC_FOCUS = 0.75
        VIEWER_REF_HEIGHT_NORMAL = 160.0
        PREVIEW_BAR_REF_HEIGHT = 30.0
        PREVIEW_CLICK_MARGIN_DP = 6.0

        def on_touch_down(self, touch):
            # A native processing dialog (e.g. Content-Aware Fill) lives outside
            # Kivy's widget tree, so Kivy has no natural way to know it should be
            # modal. Swallow all input here instead of relying on the AppKit
            # run-loop event pump (macos.py:_pump_runloop) to drop clicks, which
            # proved unreliable.
            if processing_dialog.is_active():
                return True
            return super().on_touch_down(touch)

        def on_touch_move(self, touch):
            if processing_dialog.is_active():
                return True
            return super().on_touch_move(touch)

        def on_touch_up(self, touch):
            if processing_dialog.is_active():
                return True
            return super().on_touch_up(touch)

        def __init__(self, cache_system, **kwargs):
            super().__init__(**kwargs)

            self.texture = None
            self.preview_sample_image = None
            self._preview_pixel_last_xy = None
            self.imgset = None
            self.click_x = 0
            self.click_y = 0        
            self.crop_image = None
            self.crop_image_view_key = None
            self._mask1_full_preview_backup = None
            self._mask1_full_preview_sources = set()
            self.is_zoomed = False
            self.zoom_ratio = 1.0
            self.drag_center_start = None
            self.is_press_space = False

            self.primary_param = {}
            self.primary_effects = effects.create_effects(
                lens_modifier_callback=self.lens_modifier_callback,
                distortion_callback=self.distortion_callback,
                geometry_callback=self.geometry_callback,
                crop_callback=self.crop_callback,
                ghost_callback=self.ghost_callback,
                light_rays_callback=self.light_rays_callback)
            #self.primary_effects[0]['crop'].set_editing_callback(self.crop_editing)
            self.inpaint_edit = None
            self.patchmatch_inpaint_edit = None
            # mask Mesh edit エディタ (MeshWarpWidget) のインスタンスと、編集対象 Composit。
            self.mask_mesh_editor = None
            self._mask_mesh_target_composit = None
            self.cache_system = cache_system
            self.ids['viewer'].set_cache_system(self.cache_system)
            self._rgb_xmp_rating_had = {}

            self.async_worker = AsyncWorker()
            # self.async_worker.start() # Start explicitly after config init
            self.processor = pipeline.AsyncPipelineManager(self.async_worker)
            self.ai_job_manager = AIJobManager(viewer_state_callback=self._set_ai_job_viewer_state)
            effects.bind_ai_job_manager(self.primary_effects, self.ai_job_manager)
            self.ai_image_cache = AIImageCache()
            self.ids['mask_editor2'].set_ai_image_cache(self.ai_image_cache)
            self.ai_sidecar_merge_queue = AISidecarMergeQueue()
            KVClock.schedule_interval(self.update_async_results, 0.1)
            self.pipeline_version = 0
            self._draw_image_core_active = False
            self._last_processed_pipeline_version = self.pipeline_version
            self._pending_preview_focus_mode = None
            self._preview_focus_retry_event = None
            self._preview_focus_refresh_event = None
            self._preview_focus_late_refresh_event = None
            self._preview_focus_layout_pending = False

            # version + 描画フラグ束のアトミックなスナップショット(_DrawRequest 参照)。
            self._draw_request = _DrawRequest(version=self.pipeline_version)
            # ParamSlider の連続ドラッグ中フラグ。ドラッグ中のフレームは lv1-lv2 を
            # 半解像度で回し(pipeline 側)、リリース時に必ずフル解像度で描き直す。
            self._param_slider_dragging = False
            self._drag_quality_frame_pending = False
            self._fast_display_transform_cache = {}
            self._auto_display_color_gamut = "sRGB"
            self._auto_display_color_gamut_last_reason = None
            KVClock.schedule_once(lambda dt: self._sync_zoom_ratio_slider(), 0)
            self.draw_event = threading.Event()
            self.apply_thread = threading.Thread(target=self.draw_image, daemon=False)
            self.apply_thread.start()
            self.enabledelay = None
            self._actively_loading = False  # ファイル選択によるロード中フラグ（起動時のloading: Trueとは別管理）
            self._memory_last_load_stage = None
            self._memory_last_report_key = None
            self._pending_final_display_cache = None
            self._last_display_memory_policy_at = 0.0

            self.history = history.History()
            self.current_op = None
            self._copied_effect_param = None

            self.run_set2widget_all = False
            self._sticky_distortion_tool_values = {
                'distortion_brush_size': None,
                'distortion_strength': None,
            }

            self.is_press_space = False
            # on_select で選んだパス。FCS の遅延コールバックが別ファイル向けなら無視する（primary_param と imgset の不整合防止）
            self._expected_file_path = None
            self._deferred_select_card = None
            self._deferred_select_event = None
            KVClock.schedule_interval(self._check_memory_pressure, 5.0)

            self._export_cancel_event = threading.Event()
            self._export_thread = None
            self._clamping_preview_window = False
            self._preview_min_w = 0
            self._preview_min_h = 0
            self._debug_resize_label = None
            preset_utils.ensure_preset_dir()

            KVWindow.bind(on_key_down=self.on_key_down)
            KVWindow.bind(on_key_up=self.on_key_up)
            KVWindow.bind(mouse_pos=self.on_preview_mouse_pos)
            KVClock.schedule_once(lambda _dt: self.update_mask2_options_enabled(), 0)

        def on_start(self, *args, **kwargs):
            #self.ids['preview_widget'].ref_size_hint_min = (config.get_config("preview_width"), config.get_config("preview_height"))
            #self.ids['preview_widget'].ref_size_hint_max = (config.get_config("preview_width") * 1.1, config.get_config("preview_height") * 1.1)
            pass

        def _check_memory_pressure(self, dt=0):
            try:
                if self._draw_image_core_active or self._last_processed_pipeline_version < self.pipeline_version:
                    return
                self.cache_system.enforce_memory_policy(owner=self, reason="periodic")
            except Exception:
                logging.exception("memory pressure check failed")

        def _mask_editor2_for_memory(self):
            try:
                return self.ids.get("mask_editor2")
            except Exception:
                try:
                    return self.ids["mask_editor2"]
                except Exception:
                    return None

        def force_release_memory_caches(self, reason="manual_shortcut"):
            if not threads.primary_param_lock.acquire(blocking=False):
                logging.info("manual memory release skipped reason=%s primary_param_lock_busy=True", reason)
                return {"skipped": True, "reason": "primary_param_lock_busy"}
            try:
                mask_editor2 = self._mask_editor2_for_memory()
                result = memory_manager.clear_effect_intermediate_caches(
                    self.primary_effects,
                    self.processor,
                    self.primary_param,
                    mask_editor2,
                    reason=reason,
                    clear_mask2_results=False,
                    release_ai_models=True,
                )
                if self.cache_system is not None:
                    keep_file_path = getattr(self.imgset, "file_path", None) if self.imgset is not None else None
                    result["final_display_evicted"] = self.cache_system.clear_final_display_cache(
                        keep_file_path=keep_file_path,
                    )
                pending = self._pending_final_display_cache is not None
                self._pending_final_display_cache = None
                self.crop_image = None
                gc.collect()
                logging.info(
                    "manual memory release reason=%s result=%s pending_final_display_cleared=%s",
                    reason,
                    result,
                    pending,
                )
                return result
            finally:
                threads.primary_param_lock.release()

        def _is_force_memory_release_shortcut(self, key, codepoint, modifier):
            modifiers = set(modifier or [])
            if "shift" not in modifiers:
                return False
            if "meta" not in modifiers and "ctrl" not in modifiers:
                return False
            if key in (8, 127, 266):
                return True
            return (codepoint or "") in ("\x08", "\x7f")

        def _log_display_ready_memory(self, file_path, stage, frame_version, img_shape, display_shape):
            if not memory_manager.debug_enabled():
                return
            key = (file_path, str(stage), frame_version, tuple(display_shape or ()))
            if self._memory_last_report_key == key:
                return
            self._memory_last_report_key = key
            try:
                self.cache_system.log_display_ready_memory(
                    owner=self,
                    file_path=file_path,
                    stage=stage,
                    extra={
                        "pipeline_version": frame_version,
                        "source_shape": tuple(img_shape or ()),
                        "display_shape": tuple(display_shape or ()),
                    },
                )
            except Exception:
                logging.exception("display-ready memory report failed")

        def _show_cached_final_display_image(self, file_path):
            cached = self.cache_system.get_final_display_image(file_path)
            if cached is None:
                return False
            try:
                self.blit_image(cached, frame_version=self.pipeline_version, allow_stale=True)
            except Exception:
                logging.exception("cached final display blit failed for %s", file_path)
                return False
            logging.debug("displayed cached final image for %s", file_path)
            return True

        @kvmainthread
        def _set_ai_job_viewer_state(self, file_path, state, progress_text=""):
            viewer = self.ids.get("viewer")
            if viewer is None or not file_path:
                return
            try:
                viewer.set_ai_job_state_for_path(file_path, state, progress_text)
            except Exception:
                logging.exception("failed to update viewer AI job state for %s", file_path)

        def _handle_ai_job_result(self, job_result):
            job = job_result.job
            if job.kind != AI_NOISE_KIND:
                return False
            if job_result.status != AIJobStatus.COMPLETE or job_result.result is None:
                logging.error("AI job failed: %s %s", job.file_path, job_result.error)
                return False

            current_path = self.imgset.file_path if self.imgset is not None else None
            if current_path == job.file_path:
                with threads.primary_param_lock:
                    image = self.imgset.img if self.imgset is not None else None
                    if current_param_accepts_ai_noise_result(
                        self.primary_param,
                        file_path=job.file_path,
                        image=image,
                        content_key=job.content_key,
                        source_signature=job.source_signature,
                    ):
                        merge_ai_noise_result_into_param(
                            self.primary_param,
                            job_result.result,
                            job.content_key,
                            job.source_signature,
                        )
                        logging.info("AI-NR job merged into current param: %s", job.file_path)
                        return True
                logging.info("AI-NR job result is stale for current param: %s", job.file_path)
                return False

            future = self.ai_sidecar_merge_queue.submit_ai_noise_result(job, job_result.result)
            if future is None:
                logging.warning("AI-NR job sidecar pmck merge skipped by backpressure: %s", job.file_path)
                self.ai_job_manager.discard_completed_result(job)
            else:
                cache_state = "kept" if self.ai_job_manager.has_completed_result(job) else "not_cached"
                logging.info(
                    "AI-NR job queued for sidecar pmck merge: file=%s content_key=%s completed_cache=%s",
                    job.file_path,
                    job.content_key,
                    cache_state,
                )
            return False

        def _handle_ai_sidecar_merge_result(self, merge_result):
            job = merge_result.job
            if merge_result.error:
                logging.error("AI-NR sidecar pmck merge failed: %s %s", job.file_path, merge_result.error)
                self.ai_job_manager.discard_completed_result(job)
                return False
            if merge_result.merged:
                logging.info("AI-NR job merged into pmck: %s", job.file_path)
                self._refresh_pmck_indicator_for_image_path(job.file_path)
            else:
                logging.info("AI-NR job result discarded as stale for pmck: %s", job.file_path)
                self.ai_job_manager.discard_completed_result(job)
                return False
            self.ai_job_manager.discard_completed_result(job)

            current_path = self.imgset.file_path if self.imgset is not None else None
            if current_path != job.file_path or merge_result.result is None:
                return False
            with threads.primary_param_lock:
                image = self.imgset.img if self.imgset is not None else None
                if not current_param_accepts_ai_noise_result(
                    self.primary_param,
                    file_path=job.file_path,
                    image=image,
                    content_key=job.content_key,
                    source_signature=job.source_signature,
                ):
                    return False
                merge_ai_noise_result_into_param(
                    self.primary_param,
                    merge_result.result,
                    job.content_key,
                    job.source_signature,
                )
                logging.info("AI-NR sidecar merge also applied to current param: %s", job.file_path)
            return True

        def update_async_results(self, dt):
            if self.async_worker:
                timed_out_effects = self.async_worker.cancel_timed_out_effects()
                if "InpaintEffect" in timed_out_effects:
                    self._reset_ai_inpaint_processing_ui()

                results = self.async_worker.poll_results()
                dirty = False
                inpaint_completed = False
                inpaint_failed = False
                inpaint_error_msg = None
                for task_id, result_image, error_msg in results:
                    if error_msg:
                        logging.error(f"Async Task {task_id} failed: {error_msg}")
                        if self.primary_param.get('inpaint_predict'):
                            inpaint_failed = True
                            inpaint_error_msg = error_msg
                    elif result_image is not None:
                        # Update cache in manager
                        key = self.processor.update_result(task_id, result_image)
                        if key:
                            dirty = True
                            if key[0] == "InpaintEffect":
                                inpaint_completed = True
                            logging.info(f"Async Task {task_id} ({key}) completed.")
                
                if dirty:
                    # Trigger redraw
                    # We need to make sure we don't spam redraws?
                    self.start_draw_image()
                if inpaint_completed:
                    self._schedule_ai_inpaint_ui_sync()
                if inpaint_failed:
                    self._reset_ai_inpaint_processing_ui()
                    # Show the failure verbatim (the Runware helper raises with the
                    # real HTTP status/body, which names the actual problem) rather
                    # than only logging it, so the user isn't left thinking Erase
                    # silently did nothing.
                    self.show_warning_dialog(f"AI Erase (Inpaint) failed:\n\n{inpaint_error_msg}")

            if self.ai_job_manager:
                ai_dirty = False
                for job_result in self.ai_job_manager.poll_results():
                    ai_dirty = self._handle_ai_job_result(job_result) or ai_dirty
                    if (
                        job_result.status == AIJobStatus.COMPLETE
                        and not self.ai_sidecar_merge_queue.has_pending_job(job_result.job)
                    ):
                        self.ai_job_manager.discard_completed_result(job_result.job)
                    elif job_result.status == AIJobStatus.COMPLETE:
                        logging.info(
                            "AI-NR sidecar merge is pending: file=%s content_key=%s completed_cache=%s",
                            job_result.job.file_path,
                            job_result.job.content_key,
                            "kept" if self.ai_job_manager.has_completed_result(job_result.job) else "not_cached",
                        )
                for merge_result in self.ai_sidecar_merge_queue.poll_results():
                    ai_dirty = self._handle_ai_sidecar_merge_result(merge_result) or ai_dirty
                if ai_dirty:
                    self.start_draw_image()

            if self.async_worker:
                for msg in self.async_worker.poll_messages():
                    if msg['type'] == 'waitinfo':
                        waitinfo.set_text(msg['tag'], msg['text'], self)
            
            # 処理状態の更新
            if self.async_worker:
                has_tasks = self.async_worker.has_pending_tasks()
                if self.ai_job_manager:
                    current_path = self.imgset.file_path if self.imgset is not None else None
                    current_ai_status = (
                        self.ai_job_manager.get_status_for_path(current_path)
                        if current_path
                        else None
                    )
                    has_tasks = has_tasks or (
                        bool(current_path)
                        and current_ai_status in (AIJobStatus.QUEUED, AIJobStatus.RUNNING)
                    )
                ai_inpaint_processing = self.async_worker.has_pending_effect("InpaintEffect")
                if self.ai_inpaint_processing != ai_inpaint_processing:
                    self.ai_inpaint_processing = ai_inpaint_processing
                queue_empty = self.async_worker.input_queue.empty()
                active_count = len(self.async_worker.active_shms)
                should_processing = has_tasks or self._actively_loading
                if self.is_processing != should_processing:
                    logging.info(
                        "is_processing changed: %s -> %s (queue_empty=%s, active_shms=%s, actively_loading=%s, current_ai_status=%s, ai_sidecar_pending=%s)",
                        self.is_processing,
                        should_processing,
                        queue_empty,
                        active_count,
                        self._actively_loading,
                        getattr(current_ai_status, "value", current_ai_status) if self.ai_job_manager else None,
                        self.ai_sidecar_merge_queue.pending_count() if getattr(self, "ai_sidecar_merge_queue", None) else 0,
                    )
                    self.is_processing = should_processing

        def _schedule_ai_inpaint_ui_sync(self, delay=0.15, retries=8):
            def _sync(_dt):
                if self.async_worker and self.async_worker.has_pending_effect("InpaintEffect"):
                    return
                if (
                    retries > 0
                    and self.primary_param.get('inpaint_predict')
                    and len(self.primary_param.get('inpaint_mask_list', [])) > 0
                ):
                    self._schedule_ai_inpaint_ui_sync(delay=delay, retries=retries - 1)
                    return
                try:
                    self.primary_effects[0]['inpaint'].set2widget(self, self.primary_param)
                    self._set_diff_list_to_inpaint_edit()
                    self._finish_ai_inpaint_mask_mode()
                except Exception:
                    logging.exception("failed to sync AI inpaint UI after async completion")

            KVClock.schedule_once(_sync, delay)

        def _finish_ai_inpaint_mask_mode(self):
            try:
                self._cancel_mask1_mode(sources=("inpaint",), redraw=False)
            except Exception:
                logging.exception("failed to finish AI inpaint mask mode")

        @kvmainthread
        def _remove_mask1_editor_for_effect(self, effect_name):
            effect = self.primary_effects[0].get(effect_name)
            editor = getattr(effect, "mask_editor", None) if effect is not None else None
            if editor is not None:
                try:
                    if getattr(editor, "parent", None) is not None:
                        self.ids['preview_widget'].remove_widget(editor)
                    editor.release()
                except Exception:
                    logging.exception("failed to remove mask1 editor for %s", effect_name)
                effect.mask_editor = None
                effect.inpaint_mask_list = []

        def _cancel_mask1_mode(self, sources=("inpaint", "patchmatch_inpaint"), redraw=False):
            # mask1 (inpaint/patchmatch_inpaint) を閉じる操作は、タブ切替 (on_current_tab) や
            # 画像切替 (on_select) から無条件に呼ばれる。以前は begin/end_history_effect_ctrl
            # を挟まずに primary_param を直接書き換えていた (かつ switch の state を "normal" に
            # 戻すこと自体が on_state 経由で apply_effects_lv を誘発する) ため、マスクを開いた
            # まま閉じる操作が履歴に一切残らず、実際の param 状態と履歴スタックが食い違って
            # 以降の undo/redo が壊れる不具合があった。実際に閉じる変化があるときだけ、
            # 通常のトグルボタン操作(on_press/on_state/on_release)と同じ begin/end で囲む。
            if "inpaint" in sources and (
                self.primary_param.get('inpaint') or self.ids['switch_inpaint'].state == "down"
            ):
                own_op = self.current_op is None and self.begin_history_effect_ctrl(0, 'inpaint')
                self.primary_param['inpaint'] = False
                self.primary_param['inpaint_predict'] = False
                self.primary_param['inpaint_mask_list'] = []
                self.ids['switch_inpaint'].state = "normal"
                self.ids['button_inpaint_predict'].state = "normal"
                self._remove_mask1_editor_for_effect("inpaint")
                self.exit_mask1_full_preview_mode('inpaint')
                if own_op:
                    self.end_history_effect_ctrl(0, 'inpaint')

            if "patchmatch_inpaint" in sources and (
                self.primary_param.get('patchmatch_inpaint') or self.ids['switch_patchmatch_inpaint'].state == "down"
            ):
                own_op = self.current_op is None and self.begin_history_effect_ctrl(0, 'patchmatch_inpaint')
                self.primary_param['patchmatch_inpaint'] = False
                self.primary_param['patchmatch_inpaint_predict'] = False
                self.primary_param['patchmatch_inpaint_mask_list'] = []
                self.ids['switch_patchmatch_inpaint'].state = "normal"
                self.ids['button_patchmatch_inpaint_predict'].state = "normal"
                self._remove_mask1_editor_for_effect("patchmatch_inpaint")
                self.exit_mask1_full_preview_mode('patchmatch_inpaint')
                if own_op:
                    self.end_history_effect_ctrl(0, 'patchmatch_inpaint')

            if redraw and self._image_interaction_ready():
                self.start_draw_image_and_crop(self.imgset)

        def _apply_mask1_exclusive_buttons(self, effect):
            if effect == 'inpaint' and self.ids['switch_inpaint'].state == "down":
                self.ids['switch_patchmatch_inpaint'].state = "normal"
                self.ids['button_patchmatch_inpaint_predict'].state = "normal"
            elif effect == 'patchmatch_inpaint' and self.ids['switch_patchmatch_inpaint'].state == "down":
                self.ids['switch_inpaint'].state = "normal"
                self.ids['button_inpaint_predict'].state = "normal"

        def on_mask1_make_mask_state(self, effect_name):
            """switch_inpaint / switch_patchmatch_inpaint (Make mask) の on_state ハンドラ。

            Kivy の ToggleButton は on_press が発火する前に state が変わる(= on_state が
            先に走る)ため、kv で on_press: begin → on_state: apply → on_release: end と
            並べる方式では、begin の backup が「開閉が反映された後」の param を撮ってしまい
            diff が空になって Make mask の開閉が履歴に一切残らなかった。その結果、線を描く
            op に含まれる inpaint=True だけが履歴に残り、undo/redo でボタン状態が実際の
            操作履歴と無関係に切り替わっていた。

            state 変化の時点では param がまだ旧状態なので、ここで begin → apply → end を
            まとめて行う。set2widget からのプログラム的な state 復元(履歴 undo/redo)でも
            発火するが、その場合は param が既に復元済みで diff が空になり履歴には追加され
            ない。既に別の op が進行中(_cancel_mask1_mode の own_op 等)の場合は begin を
            重ねず apply だけ行う。"""
            if self.run_set2widget_all:
                return
            own_op = self.current_op is None and self.begin_history_effect_ctrl(0, effect_name)
            self.apply_effects_lv(0, effect_name)
            if own_op:
                self.end_history_effect_ctrl(0, effect_name)

        def _restore_mask1_view_after_submit(self):
            if not self.primary_param.pop('_mask1_restore_view_after_submit', False):
                return False
            try:
                self._remove_mask1_editor_for_effect("inpaint")
                self.exit_mask1_full_preview_mode('inpaint')
                self.start_draw_image_and_crop(self.imgset)
                return True
            except Exception:
                logging.exception("failed to restore mask1 view after submit")
                return False

        def _reset_ai_inpaint_processing_ui(self):
            self.primary_param['inpaint_predict'] = False
            self.ai_inpaint_processing = False
            try:
                self.primary_effects[0]['inpaint'].set2widget(self, self.primary_param)
            except Exception:
                logging.exception("failed to reset AI inpaint UI")

        def get_preview_window_minimum_size(self):
            """
            返す min_w, min_h は Kivy 窓（通常は macOS 論理 pt 系）向け。

            m は ref*dpi（バッキング幅）のまま。窓の最小は min(w, h)/dpi_scale と同じ m_log = m/dpi
            を make し、0.55 列・上段+bar+viewer(ref) の論理高で揃える。ceil(m/0.55) を
            バッキングのまま使うと min・cap が画面いっぱいに張り付きがち（Retina）。
            """
            col_frac = self._preview_column_fraction()
            viewer_ref = self._viewer_ref_height_for_layout()
            m = int(kvutils.preview_min_edge_for_window(
                config.get_preview_min_size(),
                preview_col_frac=col_frac,
                viewer_ref=viewer_ref,
            ))
            if m < 1:
                return 0, 0, m
            dps = float(device.dpi_scale())
            if dps < 0.01:
                dps = 1.0
            m_log = m / dps
            min_w = int(math.ceil(m_log / col_frac))
            # main.kv: プレビュー下 bar ref 30, 下段 viewer ref 160（focus 時は 0）
            min_h = int(math.ceil(m_log + self.PREVIEW_BAR_REF_HEIGHT + viewer_ref))
            return min_w, min_h, m

        def _preview_column_fraction(self):
            return self.PREVIEW_COL_FRAC_FOCUS if self.preview_focus_mode else self.PREVIEW_COL_FRAC_NORMAL

        def _viewer_ref_height_for_layout(self):
            return 0.0 if self.preview_focus_mode else self.VIEWER_REF_HEIGHT_NORMAL

        def apply_preview_focus_layout(self):
            left = self.ids.get("left_info_pane")
            if left is not None:
                left.size_hint_x = 0 if self.preview_focus_mode else self.LEFT_INFO_FRAC_NORMAL
                left.opacity = 0 if self.preview_focus_mode else 1
                if self.preview_focus_mode:
                    left.width = 0

            preview_column = self.ids.get("preview_column")
            if preview_column is not None:
                preview_column.size_hint_x = self._preview_column_fraction()

            viewer = self.ids.get("viewer")
            if viewer is not None:
                ref_height = self._viewer_ref_height_for_layout()
                viewer.ref_height = ref_height
                viewer.height = kvutils.dpi_scale_height(ref_height)
                viewer.opacity = 0 if self.preview_focus_mode else 1
                viewer.disabled = self.preview_focus_mode

        def set_preview_focus_mode(self, enabled):
            enabled = bool(enabled)
            if self._is_preview_pipeline_busy():
                self._pending_preview_focus_mode = enabled
                self._schedule_pending_preview_focus_mode()
                return
            if self.preview_focus_mode == enabled:
                self._pending_preview_focus_mode = None
                return
            self._pending_preview_focus_mode = None
            self.preview_focus_mode = enabled

        def toggle_preview_focus_mode(self):
            current = self._pending_preview_focus_mode
            if current is None:
                current = self.preview_focus_mode
            self.set_preview_focus_mode(not current)

        def _is_preview_pipeline_busy(self):
            return self._has_preview_draw_in_flight() or self._preview_focus_layout_pending

        def _has_preview_draw_in_flight(self):
            return (
                self._draw_image_core_active
                or self._last_processed_pipeline_version < self.pipeline_version
            )

        def _schedule_pending_preview_focus_mode(self):
            if self._preview_focus_retry_event is None:
                self._preview_focus_retry_event = KVClock.schedule_once(
                    self._apply_pending_preview_focus_mode, 0.05)

        def _apply_pending_preview_focus_mode(self, dt=0):
            self._preview_focus_retry_event = None
            pending = self._pending_preview_focus_mode
            if pending is None:
                return
            if self._is_preview_pipeline_busy():
                self._schedule_pending_preview_focus_mode()
                return
            self.set_preview_focus_mode(pending)

        def on_preview_focus_mode(self, *args):
            self.apply_preview_focus_layout()
            self._preview_focus_layout_pending = True
            for event_attr in ("_preview_focus_refresh_event", "_preview_focus_late_refresh_event"):
                event = getattr(self, event_attr, None)
                if event is not None:
                    event.cancel()
                    setattr(self, event_attr, None)
            self._preview_focus_refresh_event = KVClock.schedule_once(self._refresh_preview_focus_layout, 0)
            self._preview_focus_late_refresh_event = KVClock.schedule_once(self._refresh_preview_focus_layout_late, 0.05)

        def _refresh_preview_focus_layout(self, dt=0):
            self._preview_focus_refresh_event = None
            if self._has_preview_draw_in_flight():
                self._preview_focus_refresh_event = KVClock.schedule_once(
                    self._refresh_preview_focus_layout, 0.05)
                return
            self.apply_preview_focus_layout()
            self._force_preview_layout_update()
            self.resize()
            self._preview_focus_layout_pending = False
            if self._pending_preview_focus_mode is not None:
                self._schedule_pending_preview_focus_mode()

        def _refresh_preview_focus_layout_late(self, dt=0):
            self._preview_focus_late_refresh_event = None
            self._refresh_preview_focus_layout(dt)

        def _force_preview_layout_update(self):
            targets = (
                self.ids.get("left_info_pane"),
                self.ids.get("preview_column"),
                self.ids.get("viewer"),
                self.ids.get("preview_widget"),
            )
            chain = []
            seen = set()
            for target in targets:
                widget = target
                local_chain = []
                while widget is not None:
                    local_chain.append(widget)
                    if widget is self:
                        break
                    widget = getattr(widget, "parent", None)
                for widget in reversed(local_chain):
                    key = id(widget)
                    if key not in seen:
                        seen.add(key)
                        chain.append(widget)

            for widget in chain:
                do_layout = getattr(widget, "do_layout", None)
                if callable(do_layout):
                    do_layout()

        def _clamp_window_to_preview_minimum(self):
            """SDL の minimum_* が効かない環境向け: 手動で窓を既定最小以上に戻す。"""
            if self._clamping_preview_window:
                return
            min_w, min_h = self._preview_min_w, self._preview_min_h
            if min_w < 1 or min_h < 1:
                return
            w, h = KVWindow.size
            if w + 0.5 >= min_w and h + 0.5 >= min_h:
                return
            self._clamping_preview_window = True
            try:
                KVWindow.size = (max(int(w), min_w), max(int(h), min_h))
            finally:
                self._clamping_preview_window = False

        def sync_preview_widget_min_size(self):
            """
            m0=ref*dpi: kvutils.preview_min_edge_for_window（枠は NSScreen ポイント * dpi して m0 と同系で比較）。

            minimum_* / cap の sw, sh は get_window_screen_size() のポイント。min_w, min_h は
            m/dpi（論理）基準（get_preview_window_minimum_size）で、Retina ではバッキングのまま
            ceil(m/0.55) を使うと cap が画面いっぱいに張り付きがちになるのを避ける。
            """
            min_w, min_h, m = self.get_preview_window_minimum_size()
            self._preview_min_w, self._preview_min_h = min_w, min_h
            pw = self.ids.get("preview_widget")
            if pw is not None:
                pw.size_hint_min = (m, m)
            if min_w > 0 and min_h > 0:
                sw, sh = kvutils.get_window_screen_size()
                KVWindow.minimum_width = min(min_w, max(1, int(sw * 0.99)))
                KVWindow.minimum_height = min(min_h, max(1, int(sh * 0.99)))

        def _resize_debug_enabled(self):
            return os.getenv("PLATYPUS_RESIZE_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}

        def _update_resize_debug_display(self):
            if not self._resize_debug_enabled():
                return
            try:
                ww, wh = int(KVWindow.size[0]), int(KVWindow.size[1])
            except Exception:
                ww, wh = 0, 0
            try:
                wkmin = int(getattr(KVWindow, "minimum_width", 0) or 0)
                wkhmin = int(getattr(KVWindow, "minimum_height", 0) or 0)
            except Exception:
                wkmin, wkhmin = 0, 0
            try:
                sys_w, sys_h = int(KVWindow.system_size[0]), int(KVWindow.system_size[1])
            except Exception:
                sys_w, sys_h = 0, 0
            pw = self.ids.get("preview_widget")
            pww, pwh = (int(pw.size[0]), int(pw.size[1])) if pw is not None else (0, 0)
            tgeo = config.get_preview_texture_size()
            tw, th = int(tgeo[0]), int(tgeo[1])
            # get_config("preview_width/height") も同じ（process_pipeline に渡る値）
            tcfg_w = int(config.get_config("preview_width"))
            tcfg_h = int(config.get_config("preview_height"))
            sw, sh = kvutils.get_window_screen_size()
            pminw, pminh = int(self._preview_min_w), int(self._preview_min_h)
            mon = None
            if hasattr(device, "get_app_window_screen_size_points"):
                try:
                    mon = device.get_app_window_screen_size_points()
                except Exception:
                    mon = None
            mon_s = f"{int(mon[0])}x{int(mon[1])}" if mon and len(mon) >= 2 else "—"
            try:
                dps = float(device.dpi_scale())
            except Exception:
                dps = 0.0
            monb = None
            if hasattr(device, "get_app_window_screen_backing_pixel_size"):
                try:
                    monb = device.get_app_window_screen_backing_pixel_size()
                except Exception:
                    monb = None
            monb_s = f"{int(monb[0])}x{int(monb[1])}" if monb and len(monb) >= 2 else "—"
            line1 = (
                f"win {ww}x{wh}  |  kivy_sys {sys_w}x{sys_h}  scn_pt {sw}x{sh}  scn_bak {monb_s}  nss {mon_s}  dpi {dps:.2f}"
            )
            if tcfg_w == tw and tcfg_h == th:
                texpart = f"tex&pipeline {tw}x{th}"
            else:
                texpart = f"tex_cfg {tcfg_w}x{tcfg_h}  tex_store {tw}x{th}  (!)"
            line2 = (
                f"Kmin {wkmin}x{wkhmin}  pmin {pminw}x{pminh}  |  previewWgt {pww}x{pwh}  |  {texpart}"
            )
            logging.info("RESIZE-DEBUG %s  %s", line1, line2)
            if self._debug_resize_label is not None and pw is not None:
                self._debug_resize_label.text = f"{line1}\n{line2}"
                tw0 = max(200.0, float(pw.size[0]) - 8.0)
                self._debug_resize_label.text_size = (tw0, None)
                self._debug_resize_label.width = tw0
                self._debug_resize_label.height = self._debug_resize_label.texture_size[1] + 6.0
                self._debug_resize_label.pos = (4, 4)

        def on_kv_post(self, *args, **kwargs):
            super().on_kv_post(*args, **kwargs)

            self.ids['mask_editor2'].opacity = 0
            self.ids['mask_editor2'].disabled = True
            self._unmount_mask_editor2_from_preview()
            self._set_lens_presets()

            self.apply_preview_focus_layout()
            KVClock.schedule_once(lambda dt: self.sync_distortion_mode_sliders(), 0)
            KVClock.schedule_once(lambda dt: self.sync_preview_widget_min_size(), 0)

            # Geometry 編集タブでのダブルタップを、子 (CropEditor / geometry editor) へ
            # 伝播する前に消費するフィルタ。KV の `on_touch_down:` ルールは式の戻り値を
            # 伝播制御に使わないため、戻り値を尊重する Python bind を choke point にする。
            # (これが無いとダブルタップが CropEditor 等に届き、クロップ枠がずれる)
            self.ids['preview_widget'].bind(
                on_touch_down=self._consume_geometry_double_tap
            )

            self.mask2_panel = mask2_content.create_mask2_content_panel(self.ids['mask_editor2'])
            self.ids['masks_box'].add_widget(self.mask2_panel)
            #self.ids['masks_box'].ids['content'].add_widget(self.mask2_panel)

            self.preset_panel = preset_content.create_preset_content_panel()
            self.ids['presets_box'].add_widget(self.preset_panel)

            self.history_panel = history_content.create_history_content_panel(self._on_history_selected)
            self.ids['history_box'].add_widget(self.history_panel)
            #self.ids['history_box'].ids['content'].add_widget(self.history_panel)
            self.update_load_dependent_panels_enabled()

            if self._resize_debug_enabled():
                self._debug_resize_label = KVLabel(
                    text="",
                    size_hint=(None, None),
                    halign="left",
                    valign="bottom",
                    color=(1, 0.92, 0.15, 0.95),
                    font_size=10,
                )
                self.ids["preview_widget"].add_widget(self._debug_resize_label)
                KVClock.schedule_once(lambda _dt: self._update_resize_debug_display(), 0.05)

        def _preview_widget_logical_size(self, preview_widget):
            dpi_scale = max(float(device.dpi_scale()), 1e-6)
            margin = max(0.0, float(kvdp(self.PREVIEW_CLICK_MARGIN_DP)))
            usable_width = max(1.0, float(preview_widget.width) - margin * 2.0)
            usable_height = max(1.0, float(preview_widget.height) - margin * 2.0)
            return (
                usable_width / dpi_scale,
                usable_height / dpi_scale,
            )

        def _preview_source_image_size(self):
            original_img_size = self.primary_param.get('original_img_size')
            if original_img_size and len(original_img_size) >= 2:
                if not self._preview_uses_full_image_size():
                    try:
                        crop_rect = params.get_crop_rect(self.primary_param)
                    except Exception:
                        crop_rect = None
                    if crop_rect is not None and len(crop_rect) >= 4:
                        x1, y1, x2, y2 = crop_rect
                        crop_width = max(1.0, float(x2) - float(x1))
                        crop_height = max(1.0, float(y2) - float(y1))
                        return (crop_width, crop_height)
                return (float(original_img_size[0]), float(original_img_size[1]))
            if self.imgset is not None and getattr(self.imgset, "img", None) is not None:
                height, width = self.imgset.img.shape[:2]
                return (float(width), float(height))
            return None

        def _preview_uses_full_image_size(self):
            if getattr(self, "_mask1_full_preview_sources", None):
                return True
            return self._is_image_geometry_mode()

        def _preview_texture_side_for_widget(self, preview_widget):
            min_side = config.get_config('preview_size')
            widget_width, widget_height = self._preview_widget_logical_size(preview_widget)
            base_side = min(widget_width, widget_height)

            image_size = self._preview_source_image_size()
            if image_size is None:
                return max(min_side, int(round(base_side)))

            image_width, image_height = image_size
            image_long = max(image_width, image_height)
            image_short = min(image_width, image_height)
            if image_short <= 0 or math.isclose(image_long, image_short):
                return max(min_side, int(round(base_side)))

            display_long = max(widget_width, widget_height)
            display_short = min(widget_width, widget_height)
            image_aspect = image_long / image_short
            same_orientation = (image_width >= image_height) == (widget_width >= widget_height)
            if same_orientation:
                side = min(display_long, display_short * image_aspect)
            else:
                side = display_short
            return max(min_side, int(round(side)))

        def get_preview_texture_size(self):
            preview_widget = self.ids.get('preview_widget')
            min_side = config.get_config('preview_size')
            if preview_widget is None:
                return (min_side, min_side)

            if self.is_zoomed:
                widget_width, widget_height = self._preview_widget_logical_size(preview_widget)
                return (
                    max(1, int(round(widget_width))),
                    max(1, int(round(widget_height))),
                )

            side = self._preview_texture_side_for_widget(preview_widget)
            return (side, side)

        def update_preview_texture_size(self, force=False):
            size = self.get_preview_texture_size()
            changed = force or size != config.get_preview_texture_size()
            if changed:
                config.set_preview_texture_size(*size)
            return changed

        def sync_distortion_mode_sliders(self):
            """
            group 'distortion' 内のトグル: Lens 時はレンズ2本、Trapezoid 時は H/V/焦点、
            それ以外（Lines / Mesh / Four 等）では5本とも無効。
            """
            if "btn_lens" not in self.ids or "btn_trapezoid" not in self.ids:
                return

            lens_on = self.ids["btn_lens"].state == "down"
            trap_on = self.ids["btn_trapezoid"].state == "down"
            self.ids["slider_lens_distortion_strength"].disabled = not lens_on
            self.ids["slider_lens_distortion_scale"].disabled = not lens_on
            self.ids["slider_correct_trapezoid_h"].disabled = not trap_on
            self.ids["slider_correct_trapezoid_v"].disabled = not trap_on
            self.ids["slider_focal_length"].disabled = not trap_on

        def _preview_overlay_after_blit_enabled(self):
            value = os.getenv("PLATYPUS_PREVIEW_OVERLAY_AFTER_BLIT", "0").strip().lower()
            return value in {"1", "true", "yes", "on"}

        @kvmainthread
        def refresh_preview_overlays(self, dt=0):
            # apply_thread 上の draw_image_core から呼ばれるため、ここをメインスレッド化する
            # （Canvas / グラフィックス操作は Kivy ではメインスレッド専用）
            texture_size = config.get_preview_texture_size()
            mask_editor = self.ids['mask_editor2']
            mask_editor.set_texture_size(*texture_size)
            if self._is_mask2_enabled():
                mask_editor.reposition_mask_image()

            if self.inpaint_edit is not None:
                self.inpaint_edit.set_display_size(texture_size)
            if self.patchmatch_inpaint_edit is not None:
                self.patchmatch_inpaint_edit.set_display_size(texture_size)

            geometry_effect = self.primary_effects[0].get('geometry')
            if geometry_effect is not None:
                geometry_effect.update_geometry_editor_texture_size(self.primary_param)

            self._sync_mask_mesh_editor_view(mask_editor=mask_editor, texture_size=texture_size)

            crop_effect = self.primary_effects[0].get('crop')
            if crop_effect is not None and params.has_original_img_size(self.primary_param):
                crop_effect.update_crop_editor_preview_size(self.primary_param)

        @kvmainthread
        def refresh_mask2_overlay(self, dt=0):
            mask_editor = self.ids.get('mask_editor2')
            if mask_editor is not None and self._is_mask2_enabled():
                mask_editor.refresh_active_mask_overlay()

        def empty_image(self):
            with threads.primary_param_lock:
                # 画像が無い状態。編集系 UI は全て無効化する。
                self.image_loaded = False
                # mask2/preset/history パネルも未選択状態に同期させる。
                # update_mask2_options_enabled 内で update_load_dependent_panels_enabled も
                # 走るので、両方の連動 UI が一度で正しく無効化される。
                self.update_mask2_options_enabled()
                self.update_preview_texture_size()
                self.texture = KVTexture.create(size=(config.get_config('preview_width'), config.get_config('preview_height')), colorfmt='rgb', bufferfmt='float')
                self.texture.flip_vertical()
                self.ids["preview"].texture = None

                self.imgset = None
                self.click_x = 0
                self.click_y = 0
                self.is_zoomed = False
                self.crop_image = None
                self._last_image_fidelity = None

                #core.clean_lensfun()

                self.primary_effects = effects.create_effects(
                    lens_modifier_callback=self.lens_modifier_callback,
                    distortion_callback=self.distortion_callback,
                    geometry_callback=self.geometry_callback,
                    crop_callback=self.crop_callback,
                    ghost_callback=self.ghost_callback,
                    light_rays_callback=self.light_rays_callback)
                effects.bind_ai_job_manager(self.primary_effects, self.ai_job_manager)
                self.ai_image_cache.clear()
                self.reset_param(self.primary_param)
                self.ids['mask_editor2'].clear_mask()
        
        def start_draw_image_and_crop(self, imgset, center_pos=None, fast_display=False, skip_histogram=False):
            if self.imgset is imgset:
                self.start_draw_image(
                    center_pos,
                    invalidate_crop=True,
                    fast_display=fast_display,
                    skip_histogram=skip_histogram,
                )

        def sync_draw_image_and_crop(self, imgset):
            if self.imgset is imgset:
                self.sync_draw_image(invalidate_crop=True)

        def _debug_mask_geom_image_stats(self, image):
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

        def _same_disp_info(self, left, right, eps=1e-6):
            if left is None or right is None:
                return left == right
            if len(left) != len(right):
                return False
            return all(abs(float(a) - float(b)) <= eps for a, b in zip(left, right))

        @kvmainthread
        def blit_image(self, img, frame_version=None, allow_stale=False, dt=0, disp_snapshot=None):
            stale_frame = frame_version is not None and frame_version < self.pipeline_version
            current_disp = params.get_disp_info(self.primary_param)
            if stale_frame and allow_stale and self._is_mask2_on() and not self._same_disp_info(current_disp, disp_snapshot):
                self._mask_zoom_sync_log(
                    "skip_stale_fast_viewport_mismatch frame=%s current=%s snapshot_disp=%s current_disp=%s",
                    frame_version, self.pipeline_version, disp_snapshot, current_disp,
                )
                return
            if stale_frame and not allow_stale:
                if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and self._is_mask2_enabled():
                    logging.warning(
                        "[MASK_GEOM] blit_image skipped stale frame_version=%s current_version=%s",
                        frame_version,
                        self.pipeline_version,
                    )
                return
            if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and self._is_mask2_enabled():
                logging.warning(
                    "[MASK_GEOM] blit_image frame_version=%s current_version=%s img_draw=%s",
                    frame_version,
                    self.pipeline_version,
                    self._debug_mask_geom_image_stats(img),
                )
            logging.debug("[PERF] blit_image: Start. Time: %s", time.time())
            perf_trace.event("blit_image.enter", shape=list(img.shape))
            # Texture Resizing logic
            is_dither = config.get_config('display_output_dither')
            is_downscale = config.get_config('display_output_downscale')
            target_fmt = 'ubyte' if (is_dither or is_downscale) else 'float'
            self.preview_sample_image = img
            self._preview_pixel_last_xy = None

            if self.texture is None or self.texture.size != (img.shape[1], img.shape[0]):
                self.texture = KVTexture.create(size=(img.shape[1], img.shape[0]), colorfmt='rgb', bufferfmt=target_fmt)
                self.texture.flip_vertical()

            if self.is_zoomed and self.zoom_ratio >= 1.0:
                self.texture.mag_filter = 'nearest'
                self.texture.min_filter = 'nearest'
            else:
                self.texture.mag_filter = 'linear'
                self.texture.min_filter = 'linear'

            if is_dither:
                img = core.jjn_dither_uint8(img)
                self.texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            elif is_downscale:
                img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                self.texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
            else:
                self.texture.blit_buffer(img.tobytes(), colorfmt='rgb', bufferfmt='float')

            # Update Preview Widget Size
            self.ids["preview"].texture = None
            self.ids["preview"].texture = self.texture
            self._mask_zoom_sync_log(
                "blit frame=%s current=%s allow_stale=%s img_shape=%s texture_size=%s disp=%s snapshot_disp=%s zoomed=%s zoom_ratio=%.3f",
                frame_version, self.pipeline_version, allow_stale, getattr(img, "shape", None),
                tuple(self.texture.size), params.get_disp_info(self.primary_param), disp_snapshot,
                self.is_zoomed, self.zoom_ratio,
            )
            try:
                self.ids["preview"].canvas.ask_update()
                self.ids["transform_wrapper"].canvas.ask_update()
                self.ids["preview_widget"].canvas.ask_update()
            except Exception:
                logging.exception("preview canvas ask_update failed")

            self.resize()
            self.refresh_mask2_overlay()
            self.update_preview_pixel_info(KVWindow.mouse_pos)

            #Singnalを送る
            import signals
            signals.blit_image.emit()

            # 1 トレース = 1 画像表示。ここで JSONL に書き出す。
            perf_trace.event("blit_image.done")
            perf_trace.flush(reason="blit_done")

        def on_preview_mouse_pos(self, window, pos):
            self.update_preview_pixel_info(pos)

        def update_preview_pixel_info(self, pos):
            sample = self.preview_sample_image
            preview = self.ids.get("preview")
            if sample is None or preview is None or getattr(preview, "texture", None) is None:
                self.preview_pixel_visible = False
                self._preview_pixel_last_xy = None
                return

            try:
                tex_x, tex_y = utils.to_texture(pos, preview)
            except Exception:
                self.preview_pixel_visible = False
                self._preview_pixel_last_xy = None
                return

            h, w = sample.shape[:2]
            x = int(tex_x)
            y = int(tex_y)
            if x < 0 or y < 0 or x >= w or y >= h:
                self.preview_pixel_visible = False
                self._preview_pixel_last_xy = None
                return

            if self.preview_pixel_visible and self._preview_pixel_last_xy == (x, y):
                return
            self._preview_pixel_last_xy = (x, y)

            r, g, b = [float(np.clip(v, 0, 1)) for v in sample[y, x, :3]]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            r8, g8, b8, l8 = [int(round(np.clip(v, 0, 1) * 255)) for v in (r, g, b, luminance)]
            self.preview_pixel_color = [r, g, b, 1]
            self.preview_pixel_text = f"R {r8:3d}  G {g8:3d}  B {b8:3d}  L {l8:3d}"
            self.preview_pixel_visible = True

        def _maybe_enforce_display_memory_policy(self, reason="display_ready"):
            now = time.monotonic()
            if now - self._last_display_memory_policy_at < 1.0:
                return
            self._last_display_memory_policy_at = now
            self.cache_system.enforce_memory_policy(owner=self, reason=reason)

        def _remember_final_display_image_or_defer(self, file_path, image, *, stage=None, frame_version=None):
            if self.current_op is not None:
                self._pending_final_display_cache = (file_path, image, stage, frame_version)
                return False
            remembered = self.cache_system.remember_final_display_image(
                file_path,
                image,
                stage=stage,
                frame_version=frame_version,
            )
            if remembered:
                self._pending_final_display_cache = None
                self._maybe_enforce_display_memory_policy()
            return remembered

        def _flush_pending_final_display_cache(self):
            pending = self._pending_final_display_cache
            if pending is None:
                return False
            self._pending_final_display_cache = None
            file_path, image, stage, frame_version = pending
            remembered = self.cache_system.remember_final_display_image(
                file_path,
                image,
                stage=stage,
                frame_version=frame_version,
            )
            if remembered:
                self._maybe_enforce_display_memory_policy("display_ready_deferred")
            return remembered

        @kvmainthread
        def draw_histogram_view(self, hist_data):
            #logging.debug(f"draw_histogram_view")
            self.ids["histogram"].draw_histogram_from_data(hist_data)

        def _get_fast_display_basis(self, src_space, dst_space, cat):
            key = (src_space, dst_space, cat)
            basis = self._fast_display_transform_cache.get(key)
            if basis is None:
                basis = colour_functions.display_color_transform_basis(
                    src_space,
                    dst_space,
                    cat,
                )
                self._fast_display_transform_cache[key] = basis
            return basis

        def _fast_display_color_transform(self, img, src_space, dst_space, cat):
            basis = self._get_fast_display_basis(src_space, dst_space, cat)
            return colour_functions.apply_display_color_transform(img, basis, dst_space)

        def _configured_display_color_gamut(self):
            try:
                return str(config.get_config('display_color_gamut')).strip()
            except Exception:
                return "sRGB"

        def _is_display_color_gamut_auto(self):
            return self._configured_display_color_gamut().lower() == "auto"

        def _supported_display_color_gamut(self, name):
            if not name:
                return None
            supported = {
                "sRGB",
                "Display P3",
                "Adobe RGB (1998)",
                "ProPhoto RGB",
                "Rec.2020",
                "Rec.709",
            }
            return name if name in supported else None

        def refresh_auto_display_color_gamut(self, reason="display_change", redraw=True):
            if not self._is_display_color_gamut_auto():
                return False

            previous = self._auto_display_color_gamut or "sRGB"
            try:
                detected = device.get_app_window_screen_color_space_name(
                    default=previous,
                    normalize=True,
                )
            except Exception:
                logging.exception("auto display color gamut detection failed")
                detected = previous

            resolved = self._supported_display_color_gamut(detected) or previous or "sRGB"
            if resolved == previous:
                self._auto_display_color_gamut_last_reason = reason
                return False

            self._auto_display_color_gamut = resolved
            self._auto_display_color_gamut_last_reason = reason
            self._fast_display_transform_cache.clear()
            logging.info(
                "auto display color gamut updated reason=%s detected=%s resolved=%s",
                reason,
                detected,
                resolved,
            )
            if redraw and self.imgset is not None and self.imgset.img is not None:
                self.start_draw_image(fast_display=False)
            return True

        def _effective_display_color_gamut(self):
            configured = self._configured_display_color_gamut()
            if configured.lower() != "auto":
                return configured
            if not self._auto_display_color_gamut:
                self.refresh_auto_display_color_gamut(reason="lazy", redraw=False)
            return self._auto_display_color_gamut or "sRGB"

        def draw_image_core(self, center_pos=None, fast_display=False, skip_histogram=False, frame_version_override=None, drag_quality=False):
            self._draw_image_core_active = True
            try:
                with threads.primary_param_lock:
                    if (self.imgset is not None) and (self.imgset.img is not None):
                        if not params.has_original_img_size(self.primary_param):
                            logging.warning("draw_image_core: original_img_size 未定義のため描画しません")
                            return

                        if self.update_preview_texture_size():
                            self.crop_image = None

                        frame_version = self.pipeline_version if frame_version_override is None else frame_version_override
                        current_tab = self.ids["effects"].current_tab.text
                        overlay_after_blit = self._preview_overlay_after_blit_enabled()
                        if not overlay_after_blit:
                            self.refresh_preview_overlays()
                        mask2_on = self._is_mask2_on()
                        force_full_preview_render = pipeline.preview_full_render_enabled(current_tab)
                        full_preview_allow_stale = force_full_preview_render and pipeline.preview_allow_stale_enabled(current_tab)
                        effective_fast_display = False if force_full_preview_render else fast_display
                        effective_skip_histogram = False if force_full_preview_render else skip_histogram
                        effective_is_drag = self.is_press_space and not force_full_preview_render
                        effective_drag_quality = drag_quality and not force_full_preview_render
                        effective_allow_stale = effective_fast_display or full_preview_allow_stale
                        crop_image_view_key = "full" if current_tab == "Ge" else "crop"
                        if self.crop_image_view_key != crop_image_view_key:
                            self.crop_image = None
                            self.crop_image_view_key = crop_image_view_key
                        if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and self._is_mask2_enabled():
                            logging.warning(
                                "[MASK_GEOM] draw_image_core start frame_version=%s current_tab=%s center_pos=%s fast_display=%s skip_histogram=%s force_full=%s effective_fast=%s effective_skip_histogram=%s effective_is_drag=%s allow_stale=%s",
                                frame_version,
                                current_tab,
                                center_pos,
                                fast_display,
                                skip_histogram,
                                force_full_preview_render,
                                effective_fast_display,
                                effective_skip_histogram,
                                effective_is_drag,
                                effective_allow_stale,
                            )
                        img, self.crop_image = pipeline.process_pipeline(self.imgset.img, self.crop_image, self.is_zoomed, self.zoom_ratio, config.get_config('preview_width'), config.get_config('preview_height'), self.click_x, self.click_y, self.primary_effects, self.primary_param, self.ids['mask_editor2'], self.processor, frame_version, current_tab=current_tab, loading_flag=pipeline_loading_flag(self.imgset), is_drag=effective_is_drag, center_pos=center_pos, mask2_active=mask2_on, ai_image_cache=self.ai_image_cache, drag_quality=effective_drag_quality)
                        self._refresh_mask1_editors()
                        # depth はパイプラインのマスク描画中にキャッシュされるため、
                        # 描画後に lens の depth 連動 UI を再評価する(構造変化イベント外で
                        # depth が確定するケースに追従)。
                        self.update_lens_simulator_depth_enabled()
                        logging.debug("[PERF] draw_image_core: process_pipeline finished. Time: %s", time.time())
                        perf_trace.event("draw_image_core.pipeline_done")
                        if self._restore_mask1_view_after_submit():
                            return
                        if img is None:
                            return
                        stale_frame = frame_version < self.pipeline_version
                        if stale_frame and not effective_allow_stale:
                            self._mask_zoom_sync_log(
                                "skip_stale frame=%s current=%s fast=%s allow_stale=%s disp=%s",
                                frame_version, self.pipeline_version, effective_fast_display, effective_allow_stale,
                                params.get_disp_info(self.primary_param),
                            )
                            if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and self._is_mask2_enabled():
                                logging.warning(
                                    "[MASK_GEOM] draw_image_core skipped stale frame_version=%s current_version=%s",
                                    frame_version,
                                    self.pipeline_version,
                                )
                            return
                        elif stale_frame:
                            self._mask_zoom_sync_log(
                                "allow_stale frame=%s current=%s fast=%s allow_stale=%s disp=%s",
                                frame_version, self.pipeline_version, effective_fast_display, effective_allow_stale,
                                params.get_disp_info(self.primary_param),
                            )
                            if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and self._is_mask2_enabled():
                                logging.warning(
                                    "[MASK_GEOM] draw_image_core allowing stale frame frame_version=%s current_version=%s fast=%s allow_stale=%s",
                                    frame_version,
                                    self.pipeline_version,
                                    effective_fast_display,
                                    effective_allow_stale,
                                )

                        debug_mask_geom = os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and self._is_mask2_enabled()
                        post_t0 = time.perf_counter() if debug_mask_geom else None
                        img = np.asarray(img)
                        utils.print_nan_inf(img, "output")
                        _debug_display_stats("input", img)

                        src_space = getattr(self.imgset, 'color_space', 'ProPhoto RGB')
                        dst_space = self._effective_display_color_gamut()
                        cat = config.get_config('cat')
                        color_t0 = time.perf_counter() if debug_mask_geom else None
                        if effective_fast_display:
                            img = self._fast_display_color_transform(img, src_space, dst_space, cat)
                        else:
                            img = colour_functions.display_color_transform(img, src_space, dst_space, cat)
                        _debug_display_stats("converted fast=%s %s->%s" % (effective_fast_display, src_space, dst_space), img)
                        color_ms = (time.perf_counter() - color_t0) * 1000.0 if debug_mask_geom else 0.0

                        # Ge タブでは Mask2 モードでも full-preview なので zero-wrap しない。
                        # CropEditor の起動可否は CropEffect 側で Mask2 ON/OFF を見て制御する。
                        crop_editing = current_tab == "Ge"

                        # プレビュー表示
                        preview_t0 = time.perf_counter() if debug_mask_geom else None
                        img_draw = core.apply_out_of_range_exposure(img, self.ids['toggle_overexposure'].state == 'down', self.ids['toggle_underexposure'].state == 'down')
                        img_draw, _ = core.apply_zero_wrap(img_draw, self.primary_param, crop_editing=crop_editing)
                        img_draw = np.clip(img_draw, 0, 1)
                        _debug_display_stats("clipped", img_draw)
                        preview_ms = (time.perf_counter() - preview_t0) * 1000.0 if debug_mask_geom else 0.0
                        if debug_mask_geom:
                            logging.warning(
                                "[MASK_GEOM] draw_image_core ready_to_blit frame_version=%s current_version=%s img=%s img_draw=%s",
                                frame_version,
                                self.pipeline_version,
                                self._debug_mask_geom_image_stats(img),
                                self._debug_mask_geom_image_stats(img_draw),
                            )

                        #描画をスケジューリング
                        disp_snapshot = params.get_disp_info(self.primary_param)
                        self._mask_zoom_sync_log(
                            "ready_to_blit frame=%s current=%s fast=%s allow_stale=%s disp=%s img_draw_shape=%s",
                            frame_version, self.pipeline_version, effective_fast_display,
                            effective_allow_stale,
                            disp_snapshot, getattr(img_draw, "shape", None),
                        )
                        self.blit_image(
                            img_draw,
                            frame_version,
                            allow_stale=effective_allow_stale,
                            disp_snapshot=disp_snapshot,
                        )

                        # ヒストグラムは表示画像を投げてから計算する。Mask2 操作中の体感遅延を
                        # 減らすため、プレビュー反映をヒストグラム更新で待たせない。
                        hist_ms = 0.0
                        if not effective_skip_histogram:
                            hist_t0 = time.perf_counter() if debug_mask_geom else None
                            img_hist, exclude_count = core.apply_zero_wrap(img, self.primary_param, crop_editing=crop_editing)
                            if HISTOGRAM_DECIMATION > 1:
                                # 間引きで画素数を 1/D^2 に。zero_wrap 後に行うことでレターボックス
                                # ジオメトリ(disp_info 基準)を壊さない。exclude_count(パッド画素数)も
                                # 同率で縮める(bin0 のみに効く値で、sqrt スケール表示では誤差不可視)。
                                img_hist = np.ascontiguousarray(img_hist[::HISTOGRAM_DECIMATION, ::HISTOGRAM_DECIMATION])
                                exclude_count = exclude_count // (HISTOGRAM_DECIMATION * HISTOGRAM_DECIMATION)
                            hist_data = widgets.histogram.HistogramWidget.calculate_histogram_data(img_hist, 0, exclude_count)
                            hist_ms = (time.perf_counter() - hist_t0) * 1000.0 if debug_mask_geom else 0.0
                            if frame_version == self.pipeline_version:
                                self.draw_histogram_view(hist_data)
                        if frame_version == self.pipeline_version and not effective_fast_display and not effective_is_drag and not effective_drag_quality:
                            self._remember_final_display_image_or_defer(
                                getattr(self.imgset, "file_path", None),
                                img_draw,
                                stage=self._memory_last_load_stage,
                                frame_version=frame_version,
                            )
                            self._log_display_ready_memory(
                                getattr(self.imgset, "file_path", None),
                                self._memory_last_load_stage,
                                frame_version,
                                getattr(self.imgset.img, "shape", None),
                                getattr(img_draw, "shape", None),
                            )
                        if debug_mask_geom:
                            logging.warning(
                                "[MASK_GEOM] draw_image_core post timings frame_version=%s fast_display=%s skip_histogram=%s color_ms=%.1f preview_ms=%.1f hist_ms=%.1f total_ms=%.1f",
                                frame_version,
                                effective_fast_display,
                                effective_skip_histogram,
                                color_ms,
                                preview_ms,
                                hist_ms,
                                (time.perf_counter() - post_t0) * 1000.0,
                            )
                        """
                        try:
                            if self.enabledelay is not None:
                                self.enabledelay.cancel()  # 既にスケジュール済みならキャンセル
                        except:
                            pass  # 未スケジュール時は無視
                        self.enabledelay = KVClock.schedule_once(partial(self.blit_image, img_draw), -1)
                        """
            finally:
                self._draw_image_core_active = False
                if self._pending_preview_focus_mode is not None:
                    self._schedule_pending_preview_focus_mode()

        def draw_image(self):
            last_processed_version = -1
            while self.apply_thread is not None:

                if last_processed_version >= self.pipeline_version:
                    self.draw_event.wait(timeout=0.1)
                    self.draw_event.clear()

                # version とフラグ束を _draw_request 1個の代入でまとめて取得する。
                # 個別に self.pipeline_version / self.apply_draw_* を読むと、その間に
                # 別の start_draw_image が割り込んで「version は新しいのにフラグは
                # 古い(またはその逆)」という中途半端な組み合わせを観測しうる。
                request = self._draw_request
                current_version = request.version
                if last_processed_version < current_version:
                    center_pos = request.center
                    fast_display = request.fast_display
                    skip_histogram = request.skip_histogram
                    drag_quality = request.drag_quality
                    try:
                        current_tab = self.ids["effects"].current_tab.text
                    except Exception:
                        current_tab = ""
                    full_preview_render = pipeline.preview_full_render_enabled(current_tab)
                    drain_all_preview = pipeline.preview_drain_all_enabled(current_tab)
                    # drain-all は中間バージョンを飛ばさず「1バージョンずつ順に」描くモード
                    # (Ge タブで既定 ON)。だが回転補正スライダー等をドラッグし続けて
                    # フルレンダーが追いつかないと、未処理バージョンが際限なく溜まる。ドラッグを
                    # 止めた後もこの積み残しを最後まで描き切るまで apply_thread が CPU を占有し、
                    # UI スレッド(クロップ枠の移動処理)が飢餓状態になって「枠が震えて動かせない」
                    # 状態になる。復帰時間がドラッグ時間に比例するのはこのため。
                    # 対策として積み残し(ラグ)に上限を設け、上限を超えたら中間を捨てて最新版へ
                    # 一気に追いつく。追いついている間は従来どおり中間フレームも描くので
                    # プレビューの滑らかさは保ちつつ、遅延だけを定数フレームに抑える。
                    if (full_preview_render and drain_all_preview and last_processed_version >= 0
                            and current_version - last_processed_version <= _PREVIEW_DRAIN_MAX_LAG):
                        target_version = last_processed_version + 1
                    else:
                        target_version = current_version
                    self.draw_image_core(
                        center_pos,
                        fast_display=fast_display,
                        skip_histogram=skip_histogram,
                        frame_version_override=target_version,
                        drag_quality=drag_quality,
                    )
                    last_processed_version = target_version
                    self._last_processed_pipeline_version = target_version
            
        def _is_liquify_painter_open(self):
            try:
                return self.primary_effects[1]['distortion'].distortion_painter is not None
            except Exception:
                return False

        def set_param_slider_drag(self, active):
            """ParamSlider からのドラッグ開始/終了通知。

            ドラッグ中は start_draw_image(drag_quality=True) 経由で half-res プレビューを
            許可し、終了時に half-res フレームを描いていたら必ずフル解像度で描き直す。
            """
            active = bool(active)
            if self._param_slider_dragging == active:
                return
            self._param_slider_dragging = active
            if not active and self._drag_quality_frame_pending:
                self._drag_quality_frame_pending = False
                self.start_draw_image()

        def start_draw_image(self, center_pos=None, invalidate_crop=False, fast_display=False, skip_histogram=False, drag_quality=False):
            if invalidate_crop:
                self.crop_image = None
            self.pipeline_version += 1
            # version + フラグ束を1回の代入でアトミックに公開する(_DrawRequest 参照)。
            self._draw_request = _DrawRequest(
                version=self.pipeline_version,
                center=center_pos,
                fast_display=fast_display,
                skip_histogram=skip_histogram,
                drag_quality=drag_quality,
            )
            if drag_quality:
                self._drag_quality_frame_pending = True
            self.processor.set_pipeline_version(self.pipeline_version)
            if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and self._is_mask2_enabled():
                logging.warning(
                    "[MASK_GEOM] start_draw_image version=%s center_pos=%s invalidate_crop=%s fast_display=%s skip_histogram=%s active=%s",
                    self.pipeline_version,
                    center_pos,
                    invalidate_crop,
                    fast_display,
                    skip_histogram,
                    getattr(self.ids['mask_editor2'].get_active_mask(), 'mask_id', None),
                )
            self.draw_event.set()

        def sync_draw_image(self, invalidate_crop=False):
            if invalidate_crop:
                self.crop_image = None
            self.pipeline_version += 1
            # draw_image ワーカーが current_version として参照するのは
            # self.pipeline_version ではなく self._draw_request.version なので、
            # ここでも version だけ追従させておく(フラグは直前の値を維持)。
            # 揃えないと version だけ進んでワーカー側の request.version が古いまま
            # 停滞し、last_processed_version に追いつけずビジーループしうる。
            self._draw_request = dataclasses.replace(self._draw_request, version=self.pipeline_version)
            self.draw_image_core()
            self._last_processed_pipeline_version = self.pipeline_version
                
        def crop_editing(self):
            self.apply_effects_lv(4, 'vignette')

        def lens_modifier_callback(self):
            self.run_set2widget_all = True
            try:
                self.primary_effects[0]['lens_modifier'].set2widget(self, self.primary_param)
            finally:
                self.run_set2widget_all = False

        def _liquify_debug(self, message, *args):
            if os.getenv("PLATYPUS_DEBUG_LIQUIFY", "0").strip().lower() in {"1", "true", "yes", "on"}:
                logging.warning("[LIQUIFY] " + message, *args)

        def _distortion_callback_target(self, proc, widget):
            if proc != 'focus':
                effect, current_param, owner_is_active = self._get_distortion_effect_and_param_for_painter(widget)
                if not owner_is_active:
                    self._liquify_debug("ignored inactive painter callback proc=%s", proc)
                    active_effect, _active_param = self._get_active_distortion_effect_and_param()
                    self._close_inactive_distortion_painters(active_effect)
                    self._sync_liquify_mask_editor_input_lock()
                    return None, None, False
                return effect, current_param, True
            return None, None, True

        def _log_distortion_callback(self, proc, widget, owner_is_active):
            try:
                editor = self.ids.get('mask_editor2')
                active = editor.get_active_mask() if editor is not None else None
                self._liquify_debug(
                    "callback proc=%s mask2=%s active=%s owner_active=%s records=%s buffer=%s current_id=%s pipeline_version=%s",
                    proc,
                    self._is_mask2_enabled(),
                    getattr(active, 'mask_id', None),
                    owner_is_active,
                    len(getattr(widget, 'recorded', []) or []),
                    len(getattr(widget, 'points_buffer', []) or []),
                    id(getattr(widget, 'current_image', None)) if getattr(widget, 'current_image', None) is not None else None,
                    self.pipeline_version,
                )
            except Exception:
                logging.exception("[LIQUIFY] callback debug failed")

        def _apply_distortion_painter_params(self, widget, effect, current_param):
            current_param.update(widget.get_distortion_params())

        def _sync_distortion_brush_from_painter(self, widget, current_param):
            current_param['distortion_brush_size'] = widget.brush_size
            slider = self.ids.get("slider_distortion_brush_size")
            if slider is not None:
                slider.set_slider_value(widget.brush_size)
            self._remember_distortion_tool_values_from_widgets(current_param)

        def distortion_callback(self, proc, widget):
            effect, current_param, owner_is_active = self._distortion_callback_target(proc, widget)
            if not owner_is_active:
                return
            self._log_distortion_callback(proc, widget, owner_is_active)
            match proc:
                case 'focus':
                    self._clear_text_input_focus()
                case 'start':
                    self.begin_history_effect_ctrl(1, 'distortion')
                    current_param['switch_distortion'] = True
                    switch = self.ids.get("switch_distortion")
                    if switch is not None:
                        switch.enabled = True
                case 'update' | 'apply':
                    self._apply_distortion_painter_params(widget, effect, current_param)
                    if self._is_mask2_enabled():
                        self._request_active_liquify_mask_render_update(redraw_pipeline=True)
                    else:
                        self.apply_effects_lv(1, 'distortion')
                case 'end':
                    self._apply_distortion_painter_params(widget, effect, current_param)
                    self._request_active_liquify_mask_render_update(redraw_pipeline=True)
                    self.end_history_effect_ctrl(1, 'distortion')
                case 'brush_size':
                    self._sync_distortion_brush_from_painter(widget, current_param)

        def ghost_callback(self, proc, widget):
            """LensGhostCanvas からの通知。光源は per-light 独立パラメータ(lens_ghost_lights)。
            選択(select)で flat スライダーへ反映、追加/移動/削除で lights を更新。実行レベルは lens_ghost_level()。"""
            eff = self._lens_ghost_effect()
            if eff is None or widget is not getattr(eff, 'ghost_canvas', None):
                return
            lv = self.lens_ghost_level()
            # 保存先: mask2 起動時は Composit の effect_param(マスクの影響を受ける)、それ以外は primary_param。
            _ce, cp, _mid = self._get_active_effects(lv=lv)
            sel = getattr(widget, 'selected', -1)
            match proc:
                case 'focus':
                    self._clear_text_input_focus()
                case 'start':
                    self.begin_history_effect_ctrl(lv, 'lens_ghost')
                    cp['lens_ghost_enabled'] = True
                    sw = self.ids.get('switch_lens_ghost')
                    if sw is not None:
                        sw.enabled = True
                case 'select':
                    # 選択 CP のパラメータをスライダーへ(履歴なし・描画なし)。
                    cp['lens_ghost_selected'] = sel
                    self._load_ghost_light_to_sliders(cp, sel)
                case 'add':
                    lights = eff._lights(cp)
                    base = {k: cp.get(k) for k in eff._SLIDER_KEYS}   # 現在のスライダー値を引き継ぐ
                    pos = tuple(widget.coords[-1]) if widget.coords else (0.0, 0.0)
                    lights.append({'pos': pos, 'params': base})
                    cp['lens_ghost_selected'] = len(lights) - 1
                    self.apply_effects_lv(lv, 'lens_ghost', sync=True)
                case 'move':
                    lights = eff._lights(cp)
                    if 0 <= sel < len(lights) and sel < len(widget.coords):
                        lights[sel]['pos'] = tuple(widget.coords[sel])
                    self.apply_effects_lv(lv, 'lens_ghost', sync=True)
                case 'delete':
                    lights = eff._lights(cp)
                    i = getattr(widget, '_del_index', -1)
                    if 0 <= i < len(lights):
                        del lights[i]
                    cp['lens_ghost_selected'] = -1
                    self.apply_effects_lv(lv, 'lens_ghost')
                case 'end':
                    # スロットルで取りこぼした最終位置を確定 + フル品質描画。
                    lights = eff._lights(cp)
                    if 0 <= sel < len(lights) and sel < len(widget.coords):
                        lights[sel]['pos'] = tuple(widget.coords[sel])
                    self.apply_effects_lv(lv, 'lens_ghost')
                    self.end_history_effect_ctrl(lv, 'lens_ghost')

        def _load_ghost_light_to_sliders(self, cp, sel):
            """選択中の光源 params を flat スライダーキー + スライダー widget へ反映(エコー抑止)。"""
            eff = self._lens_ghost_effect()
            if eff is None:
                return
            lp = eff.light_params(cp, sel)
            if lp is None:
                return
            self.run_set2widget_all = True
            try:
                for k in eff._SLIDER_KEYS:
                    if k in lp:
                        cp[k] = lp[k]
                        slider = self.ids.get('slider_' + k)
                        if slider is not None:
                            slider.set_slider_value(lp[k])
            finally:
                self.run_set2widget_all = False

        def light_rays_callback(self, proc, widget):
            """LightRaysCanvas からの通知。ガイドを active param に書き戻して再描画。"""
            eff = self._light_rays_effect()
            if eff is None or widget is not getattr(eff, 'light_rays_canvas', None):
                return
            lv = 2
            _ce, cp, _mid = self._get_active_effects(lv=lv)
            match proc:
                case 'focus':
                    self._clear_text_input_focus()
                case 'select':
                    cp['light_ray_guides'] = widget.get_guides()
                    cp['light_ray_selected'] = widget.selected_index()
                    if eff.sync_selected_guide_to_widget(self, cp):
                        cp['light_ray_editor_type'] = self.ids['spinner_light_ray_editor_type'].text
                        cp['light_ray_editor_mode'] = self.ids['spinner_light_ray_editor_mode'].text
                case 'start':
                    self.begin_history_effect_ctrl(lv, 'light_rays')
                    cp['switch_light_rays'] = True
                    sw = self.ids.get('switch_light_rays')
                    if sw is not None:
                        sw.enabled = True
                case 'update' | 'apply':
                    cp['light_ray_guides'] = widget.get_guides()
                    cp['light_ray_selected'] = widget.selected_index()
                    cp['light_ray_editor_type'] = self.ids['spinner_light_ray_editor_type'].text
                    cp['light_ray_editor_mode'] = self.ids['spinner_light_ray_editor_mode'].text
                    self.apply_effects_lv(lv, 'light_rays', sync=True)
                case 'end':
                    cp['light_ray_guides'] = widget.get_guides()
                    cp['light_ray_selected'] = widget.selected_index()
                    cp['light_ray_editor_type'] = self.ids['spinner_light_ray_editor_type'].text
                    cp['light_ray_editor_mode'] = self.ids['spinner_light_ray_editor_mode'].text
                    self.apply_effects_lv(lv, 'light_rays')
                    self.end_history_effect_ctrl(lv, 'light_rays')

        def apply_ghost_preset(self, name):
            """プリセット選択 → lens_ghost の全スライダーへ反映(履歴対応)。
            プリセットは 700x500 基準なので、サイズ系(base_radius/blur_sigma)は画像サイズに
            合わせてスケールし「スケール感」を保つ。比率・強度系はそのまま。CP は変更しない。"""
            from cores.lens_ghost import GHOST_PRESETS
            preset = GHOST_PRESETS.get(name)
            if preset is None:
                return
            lv = self.lens_ghost_level()
            if not self.begin_history_effect_ctrl(lv, 'lens_ghost'):
                return
            ce, cp, _mid = self._get_active_effects(lv=lv)
            eff = ce[lv].get('lens_ghost')
            if eff is None:
                self.current_op = None
                return
            sz = cp.get('original_img_size') or self.primary_param.get('original_img_size')
            size_scale = (min(sz) / 500.0) if sz else 1.0   # プリセットは min 辺 500 基準
            # スライダーへの反映で on_slider が空振りするよう run_set2widget_all で抑止(履歴は外側で取る)。
            self.run_set2widget_all = True
            try:
                for key in eff._SLIDER_KEYS:
                    if key not in preset:
                        continue
                    val = float(preset[key])
                    if key in ('base_radius', 'blur_sigma'):
                        val *= size_scale
                    slider = self.ids.get('slider_' + key)
                    if slider is not None:
                        val = min(slider.max, max(slider.min, val))
                    if key in eff._INT_KEYS:
                        val = int(round(val))
                    cp[key] = val
                    if slider is not None:
                        slider.set_slider_value(val)
            finally:
                self.run_set2widget_all = False
            cp['lens_ghost_enabled'] = True
            sw = self.ids.get('switch_lens_ghost')
            if sw is not None:
                sw.enabled = True
            self.apply_effects_lv(lv, 'lens_ghost')
            self.end_history_effect_ctrl(lv, 'lens_ghost')

        def _request_active_liquify_mask_render_update(self, redraw_pipeline=False):
            if not self._is_mask2_enabled():
                return
            # mask2 モードが有効でも、mask リストで Primary が選択されている(= どの
            # Composit マスクもアクティブでない)ときは get_active_mask() が None を返す。
            # その場合でも Liquify は primary の distortion を直接編集しているので、
            # マスクレンダーキャッシュの無効化はスキップしてよいが、パイプライン再描画
            # 自体は必ず要求する。以前はここで早期 return しており、mask2 表示中に
            # primary の Liquify を Reset しても画面に反映されず、履歴クリック(別経路の
            # start_draw_image_and_crop)でしか更新が見えない不具合があった。
            editor = self.ids.get('mask_editor2')
            mask = None
            if editor is not None:
                try:
                    mask = editor.get_active_mask()
                    if mask is not None and not mask.is_composit():
                        composit = editor.find_composit_mask(mask)
                        if composit is not None:
                            mask = composit
                    if mask is not None:
                        updater = getattr(editor, 'request_mask_render_update', None)
                        if callable(updater):
                            self._liquify_debug(
                                "request_mask_render_update mask=%s redraw_pipeline=%s",
                                getattr(mask, 'mask_id', None),
                                redraw_pipeline,
                            )
                            updater(
                                mask,
                                reason="liquify",
                                structure_changed=False,
                                refresh_visibility=False,
                                redraw_overlay=False,
                                redraw_pipeline=False,
                            )
                except Exception:
                    logging.exception("failed to invalidate liquify mask render cache")
            if redraw_pipeline:
                self.start_draw_image(fast_display=False)

        def reset_distortion_painter_action(self):
            if not self.can_open_liquify_editor():
                return
            effect, current_param = self._get_active_distortion_effect_and_param()
            if effect is None:
                return
            painter = getattr(effect, 'distortion_painter', None)
            # ペインタ未生成(ツールボタン非押下)でも記録済みストロークは param に残るので消す。
            if painter is None and not current_param.get('distortion_recorded'):
                return
            self.begin_history_effect_ctrl(1, 'distortion')
            if painter is not None:
                painter.reset_image(notify=False)
            current_param['distortion_recorded'] = []
            current_param['switch_distortion'] = True
            switch = self.ids.get("switch_distortion")
            if switch is not None:
                switch.enabled = True
            effect.reeffect()
            self._request_active_liquify_mask_render_update(redraw_pipeline=True)
            if not self._is_mask2_enabled():
                self.start_draw_image()
            self.end_history_effect_ctrl(1, 'distortion')

        def geometry_callback(self, proc, widget):
            if os.getenv("PLATYPUS_DEBUG_4PT", "0").strip().lower() in {"1", "true", "yes", "on"}:
                logging.warning(
                    "[4PT] geometry_callback proc=%s | primary_param.four_points=%s",
                    proc, self.primary_param.get('four_points'),
                )
            match proc:
                case 'start':
                    if self.begin_history_effect_ctrl(0, 'geometry'):
                        self.crop_image = None
                case 'update' | 'apply':
                    self.crop_image = None
                    self.apply_effects_lv(0, 'geometry', sync=True)
                    # Update widget with new params (especially matrix for correct display)
                    #widget.set_correction_params(self.primary_param)
                    if os.getenv("PLATYPUS_DEBUG_4PT", "0").strip().lower() in {"1", "true", "yes", "on"}:
                        logging.warning(
                            "[4PT] geometry_callback after apply_effects_lv | primary_param.four_points=%s",
                            self.primary_param.get('four_points'),
                        )
                case 'end':
                    self.primary_param.update(widget.get_correction_params())
                    self.end_history_effect_ctrl(0, 'geometry')
                    self.start_draw_image(invalidate_crop=True)

        def crop_callback(self, proc, widget):
            match proc:
                case 'start':
                    if self.begin_history_effect_ctrl(0, 'crop'):
                        self.crop_image = None
                case 'update' | 'apply':
                    params.set_crop_rect(self.primary_param, widget.get_crop_rect())
                    # geometry_callback などの他のエディタコールバックと違い、ここだけ
                    # redraw を一切トリガーしていなかった。CropEditor はドラッグ中は
                    # 自前の矩形を canvas に描くだけで、パイプラインで実際にクロップ
                    # された「通常プレビュー」は touch_up まで古いままだった上、
                    # touch_up 後もタブ切替など別の操作が起きるまで更新されなかった。
                    self.crop_image = None
                    self.apply_effects_lv(0, 'crop', sync=True)
                case 'end':
                    self.end_history_effect_ctrl(0, 'crop')
                    self.start_draw_image(invalidate_crop=True)

        def _get_active_effects(self, mask_id=None, lv=None, subname=None):
            editor = self.ids['mask_editor2']
            if mask_id is None and not self._is_mask2_enabled():
                return (self.primary_effects, self.primary_param, None)
            if mask_id is None:
                mask = editor.get_created_mask() or editor.get_active_mask()
            else:
                mask = editor.find_mask(mask_id)

            if mask is None:
                return (self.primary_effects, self.primary_param, None)

            # マスクパラメータの振り分け
            if mask.is_composit():
                composit_mask = mask
            else:
                try:
                    mask_index = editor.get_mask_list().index(mask)
                except ValueError:
                    mask_index = 0
                composit_mask = editor.find_composit_mask(mask, mask_index)
            if lv is not None:
                if lv == 3:
                    if subname == 'mask2_draw_effects' and not mask.is_composit() and composit_mask is not None:
                        mask = composit_mask
                    elif subname == 'mask_geometry' and not mask.is_composit() and composit_mask is not None:
                        # マスク自身の Geometry 変形は Composit 直下に保存・適用する
                        mask = composit_mask
                else:
                    # それ以外は親のCompositMaskへ（自分がCompositMaskなら自分へ）
                    if not mask.is_composit():
                        composit_mask = editor.find_composit_mask(mask)
                        if composit_mask is not None:
                            mask = composit_mask

            effects_owner = composit_mask if composit_mask is not None else mask
            return (effects_owner.effects, mask.effects_param, mask.mask_id)

        def _get_active_distortion_effect_and_param(self):
            if not self._is_mask2_enabled():
                effect = None
                try:
                    effect = self.primary_effects[1].get('distortion')
                except Exception:
                    effect = None
                return effect, self.primary_param
            current_effects, current_param, _mask_id = self._get_active_effects(
                lv=1,
                subname='distortion',
            )
            effect = None
            try:
                effect = current_effects[1].get('distortion')
            except Exception:
                effect = None
            return effect, current_param

        def _get_distortion_effect_and_param_for_painter(self, painter):
            active_effect, active_param = self._get_active_distortion_effect_and_param()
            for effect, param in self._iter_distortion_effect_targets():
                if effect is not None and getattr(effect, 'distortion_painter', None) is painter:
                    return effect, param, effect is active_effect and param is active_param
            return active_effect, active_param, True

        def _iter_distortion_effect_targets(self):
            try:
                yield self.primary_effects[1].get('distortion'), self.primary_param
            except Exception:
                # primary_effects未整備等の場合はこのターゲットをスキップして次へ進む
                pass

            editor = self.ids.get('mask_editor2')
            if editor is None:
                return
            try:
                mask_list = list(editor.get_mask_list())
            except Exception:
                mask_list = []
            for mask in mask_list:
                try:
                    if not mask.is_composit():
                        continue
                    yield mask.effects[1].get('distortion'), mask.effects_param
                except Exception:
                    continue

        def _close_inactive_distortion_painters(self, active_effect):
            for effect, param in self._iter_distortion_effect_targets():
                if effect is None or effect is active_effect:
                    continue
                if getattr(effect, 'distortion_painter', None) is None:
                    continue
                try:
                    effect._close_distortion_painter(param, self)
                except Exception:
                    logging.exception("failed to close inactive distortion painter")

        def _lens_ghost_effect(self):
            try:
                return self.primary_effects[1].get('lens_ghost')
            except Exception:
                return None

        def _light_rays_effect(self):
            try:
                return self.primary_effects[2].get('light_rays')
            except Exception:
                return None

        def _get_active_ghost_canvas(self):
            eff = self._lens_ghost_effect()
            return getattr(eff, 'ghost_canvas', None) if eff is not None else None

        def _get_active_light_rays_canvas(self):
            eff = self._light_rays_effect()
            return getattr(eff, 'light_rays_canvas', None) if eff is not None else None

        def _effect_editor_allowed(self):
            """preview エディタ(liquify / ghost / light rays)を開いてよい状態か。"""
            eff_panel = self.ids.get('effects')
            in_li = eff_panel is not None and getattr(eff_panel.current_tab, 'text', '') == 'Li'
            return bool(in_li and self.can_open_liquify_editor())

        def _sync_effect_editors(self):
            """'effect' トグルグループの押下状態で preview 上のエディタを生成/破棄する。
            tab=='Li' かつ can_open_liquify_editor() の下でのみ。どちらか一方のみ(排他)。"""
            if getattr(self, 'run_set2widget_all', False):
                return
            # トグルの state を書き換えると on_state→本メソッドが再入するので一度だけ通す。
            if getattr(self, '_in_sync_effect_editors', False):
                return
            self._in_sync_effect_editors = True
            try:
                self._sync_effect_editors_impl()
            finally:
                self._in_sync_effect_editors = False

        def _sync_effect_editors_impl(self):
            from kivy.uix.behaviors import ToggleButtonBehavior
            effect_toggles = ToggleButtonBehavior.get_widgets('effect')
            eff_panel = self.ids.get('effects')
            in_li = eff_panel is not None and getattr(eff_panel.current_tab, 'text', '') == 'Li'
            if not in_li:
                # Li タブから離れたら 'effect' グループのトグルを全解除(押下状態を残さない)。
                for w in effect_toggles:
                    if getattr(w, 'state', 'normal') == 'down':
                        w.state = 'normal'
            allowed = self._effect_editor_allowed()
            ghost_btn = self.ids.get('toggle_lens_ghost_open')
            light_rays_btn = self.ids.get('toggle_light_rays_open')
            active_distortion = None
            for w in effect_toggles:
                if getattr(w, 'state', 'normal') != 'down':
                    continue
                if type(w).__name__ == 'DistortionEffectButton':
                    active_distortion = w
                    break
            is_ghost_toggle = ghost_btn is not None and getattr(ghost_btn, 'state', 'normal') == 'down'
            is_light_rays_toggle = light_rays_btn is not None and getattr(light_rays_btn, 'state', 'normal') == 'down'
            want_ghost = bool(allowed and is_ghost_toggle)
            want_light_rays = bool(allowed and is_light_rays_toggle)
            want_liquify = bool(allowed and active_distortion is not None)

            # ghost canvas
            ghost_effect = self._lens_ghost_effect()
            if ghost_effect is not None:
                if want_ghost:
                    # 保存先 param: mask2時は Composit の effect_param。canvas はそこを編集する。
                    _ce, cp, _mid = self._get_active_effects(lv=self.lens_ghost_level())
                    ghost_effect._open_ghost_canvas(cp, self)
                else:
                    ghost_effect._close_ghost_canvas(self.primary_param, self)

            # light rays canvas
            light_rays_effect = self._light_rays_effect()
            if light_rays_effect is not None:
                if want_light_rays:
                    _ce, cp, _mid = self._get_active_effects(lv=2)
                    light_rays_effect._open_light_rays_canvas(cp, self)
                else:
                    light_rays_effect._close_light_rays_canvas(self.primary_param, self)

            # liquify painter
            if want_liquify:
                ce, cp, _ = self._get_active_effects(lv=1)
                distortion = ce[1].get('distortion')
                if distortion is not None:
                    distortion._open_distortion_painter(cp, self)
                    self._close_inactive_distortion_painters(distortion)
            else:
                self._close_inactive_distortion_painters(None)

            self._sync_liquify_mask_editor_input_lock()

        def is_liquify_editor_active(self):
            try:
                return self._get_active_distortion_painter() is not None
            except Exception:
                return False

        def _get_active_distortion_painter(self):
            if not self.can_open_liquify_editor():
                return None
            current_tab = self.ids["effects"].current_tab
            if getattr(current_tab, "text", None) != "Li":
                return None
            effect, _current_param = self._get_active_distortion_effect_and_param()
            return getattr(effect, 'distortion_painter', None) if effect is not None else None

        def _sync_liquify_mask_editor_input_lock(self):
            # Overlay editors follow the same availability as liquify.
            if not self._effect_editor_allowed():
                ge = self._lens_ghost_effect()
                if ge is not None:
                    ge._close_ghost_canvas(self.primary_param, self)
                le = self._light_rays_effect()
                if le is not None:
                    le._close_light_rays_canvas(self.primary_param, self)
            editor = self.ids.get('mask_editor2')
            if editor is None:
                return
            locked = (
                (self._get_active_distortion_painter() is not None)
                or (self._get_active_ghost_canvas() is not None)
                or (self._get_active_light_rays_canvas() is not None)
            )
            self._liquify_debug("mask_editor_input_lock=%s", locked)
            if locked:
                # Liquify owns preview touch/scroll while its painter is active.
                # Keep MaskEditor2 state readable, but remove it from Kivy dispatch.
                self._unmount_mask_editor2_from_preview()
            elif self._is_mask2_enabled():
                self._mount_mask_editor2_to_preview()

        def _has_distortion_effect(self, effects_map):
            try:
                return effects_map is not None and effects_map[1].get('distortion') is not None
            except Exception:
                return False

        def _distortion_tool_specs(self):
            return (
                ('distortion_brush_size', 'slider_distortion_brush_size'),
                ('distortion_strength', 'slider_distortion_strength'),
            )

        def _remember_distortion_tool_values_from_widgets(self, param=None):
            sticky = getattr(self, '_sticky_distortion_tool_values', None)
            if sticky is None:
                return
            for key, widget_id in self._distortion_tool_specs():
                slider = self.ids.get(widget_id)
                if slider is None:
                    continue
                value = getattr(slider, 'value', None)
                if value is None:
                    continue
                sticky[key] = value
                if param is not None:
                    param[key] = value

        def _restore_distortion_tool_values_to_widgets(self, param=None):
            sticky = getattr(self, '_sticky_distortion_tool_values', None)
            if sticky is None:
                return
            for key, widget_id in self._distortion_tool_specs():
                slider = self.ids.get(widget_id)
                if slider is None:
                    continue
                value = sticky.get(key)
                if value is None:
                    value = param.get(key) if isinstance(param, dict) else None
                    if value is not None:
                        sticky[key] = value
                    continue
                if param is not None:
                    param[key] = value
                setter = getattr(slider, 'set_slider_value', None)
                if callable(setter):
                    setter(value)
                else:
                    slider.value = value

        def _mount_mask_editor2_to_preview(self):
            editor = self.ids.get('mask_editor2')
            preview = self.ids.get('preview_widget')
            if editor is None or preview is None:
                return
            if editor.parent is preview:
                return
            if editor.parent is not None:
                try:
                    editor.parent.remove_widget(editor)
                except Exception:
                    logging.exception("failed to detach MaskEditor2 before remount")
            preview.add_widget(editor, index=0)

        def _unmount_mask_editor2_from_preview(self):
            editor = self.ids.get('mask_editor2')
            preview = self.ids.get('preview_widget')
            if editor is None or preview is None or editor.parent is not preview:
                return
            try:
                preview.remove_widget(editor)
            except Exception:
                logging.exception("failed to unmount MaskEditor2 from preview")

        def lens_ghost_level(self):
            """レンズゴーストの実行レベル(1 or 2)を Position スピナーの現在値から解決する。
            kv コールバックは apply/履歴ともこの値を渡す(位置変更時のみ kv 側で固定 lv1 を使う)。"""
            try:
                pos = self.ids['spinner_lens_ghost_position'].text
            except Exception:
                pos = self.primary_param.get('lens_ghost_position', 'lv2')
            return 1 if pos == 'lv1' else 2

        def apply_effects_lv(self, lv, effect, sync=False, subname=None, defer_draw=False, overlay_reason="param_change"):
            if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and (
                lv == 3 or subname == 'mask_geometry' or effect == 'mask_geometry'
            ):
                logging.warning(
                    "[MASK_GEOM] apply_effects_lv enter lv=%s effect=%s subname=%s sync=%s defer_draw=%s run_set2widget_all=%s",
                    lv,
                    effect,
                    subname,
                    sync,
                    defer_draw,
                    self.run_set2widget_all,
                )
            if self.run_set2widget_all == True:
                if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"} and (
                    lv == 3 or subname == 'mask_geometry' or effect == 'mask_geometry'
                ):
                    logging.warning(
                        "[MASK_GEOM] apply_effects_lv skipped run_set2widget_all lv=%s effect=%s subname=%s",
                        lv,
                        effect,
                        subname,
                    )
                return

            if lv == 1 and (effect is None or effect == 'distortion' or (
                    isinstance(effect, list) and 'distortion' in effect)):
                if not self.can_open_liquify_editor():
                    self._close_inactive_distortion_painters(None)
                    self._sync_liquify_mask_editor_input_lock()
                    return

            current_effects, current_param, mask_id = self._get_active_effects(lv=lv, subname=subname or effect)
            if effect is None:
                effects.set2param_all(current_effects, current_param, self)
            else:
                effect = effect if isinstance(effect, list) else [effect]
                if lv == 0:
                    for e in effect:
                        self._apply_mask1_exclusive_buttons(e)
                for e in effect:
                    current_effects[lv][e].set2param(current_param, self)
            if lv == 1 and (effect is None or 'distortion' in effect):
                self._remember_distortion_tool_values_from_widgets(current_param)
                active_distortion = None
                try:
                    active_distortion = current_effects[1].get('distortion')
                except Exception:
                    active_distortion = None
                self._close_inactive_distortion_painters(active_distortion)
                self._sync_liquify_mask_editor_input_lock()
            # Remember Quick Select / edge-refine as a sticky tool setting so a
            # freshly created draw mask inherits it (it is a tool mode, not a
            # per-mask one). Only genuine user edits reach here; set2widget echoes
            # are gated out above by run_set2widget_all, so this never captures a
            # spinner reset. Read after set2param so it reflects the new value.
            if lv == 3 and current_param is not None:
                self._sticky_mask2_edge_refine = {
                    k: effects.Mask2Effect.get_param(current_param, k)
                    for k in (
                        'switch_mask2_quick_select',
                        'mask2_edge_refine_mode',
                        'mask2_edge_refine_radius',
                        'mask2_edge_refine_strength',
                        'mask2_edge_refine_bias',
                    )
                }
            if lv == 0:
                self.sync_distortion_mode_sliders()

            mask_geometry_update = False
            # Mask Geometry: slider 変更後 active Composit の mask Geom matrix を再構築。
            # set_draw_mask (= mask.update_mask 同期呼出) より前に行うことで、
            # update_mask 内の direction-preserving 等の matrix 依存計算が新 matrix
            # を使うようになる (古い matrix で 1 frame 描画 → schedule 経由で新 matrix
            # で再描画、という見た目のジャンプを回避)。
            if lv == 3:
                _eff_list = effect if isinstance(effect, list) else [effect] if effect is not None else []
                mask_geometry_update = subname == 'mask_geometry' or 'mask_geometry' in _eff_list
                if mask_geometry_update:
                    if os.getenv("PLATYPUS_DEBUG_MASK_GEOMETRY", "0").strip().lower() in {"1", "true", "yes", "on"}:
                        _keys = (
                            "switch_mask_geometry",
                            "mask_rotation",
                            "mask_translation_x",
                            "mask_translation_y",
                            "mask_scale_x",
                            "mask_scale_y",
                            "mask_flip_mode",
                        )
                        logging.warning(
                            "[MASK_GEOM] apply_effects_lv lv=%s effect=%s subname=%s target_mask=%s params=%s",
                            lv,
                            effect,
                            subname,
                            mask_id,
                            {key: current_param.get(key) for key in _keys if key in current_param},
                        )
                    self.ids['mask_editor2']._set_active_composit_matrix(redraw_mask=True)
                    self.ids['mask_editor2'].skip_next_mask_overlay_refresh(clear=False)

            self._apply_mask_overlay_policy(
                lv,
                effect,
                subname=subname,
                reason=overlay_reason,
                refresh=not mask_geometry_update,
            )
            #self.apply_rotation_flip_for_wrapper()

            # defer_draw=True のとき呼び出し側は後でまとめて start/sync を発火する想定。
            # pipeline_version の無駄な多段進行と apply_thread の捨て描画を避ける。
            if defer_draw:
                return
            if sync == False:
                # half-res ドラッグは primary の lv0-lv2 スライダーに限定する。
                # lv3/lv4 やマスクレイヤーの操作では lv1-lv2 がキャッシュヒットするため、
                # 解像度切替の境界再計算コストの方が高くつき、マスク系にも触れたくない。
                # Liquify エディタ表示中も無効(painter が参照画像を持つため縮小画像を流せない)。
                drag_quality = (
                    self._param_slider_dragging and lv <= 2 and mask_id is None
                    and not self._is_liquify_painter_open()
                )
                self.start_draw_image(
                    fast_display=mask_geometry_update,
                    skip_histogram=mask_geometry_update,
                    drag_quality=drag_quality,
                )
            else:
                self.sync_draw_image()

        def on_color_temperature_preset_value(self, preset):
            if preset == effects.ColorTemperatureEffect.PRESET_AS_SHOT:
                values = (
                    self.ids["slider_color_temperature"].reset_value,
                    self.ids["slider_color_tint"].reset_value,
                )
            else:
                values = effects.ColorTemperatureEffect.preset_values(preset, self.primary_param)
            if values is None:
                return
            temp, tint = values
            self.ids["slider_color_temperature"].set_slider_value(temp)
            self.ids["slider_color_tint"].set_slider_value(tint)

        def on_color_temperature_slider_changed(self):
            spinner = self.ids.get("spinner_color_temperature_preset")
            if spinner is not None:
                spinner.set_text(effects.ColorTemperatureEffect.PRESET_CUSTOM)

        def apply_mask2_edge_refine_slider(self, settle=False):
            """Debounce expensive Quick Select slider redraws while dragging."""
            event = getattr(self, "_mask2_edge_refine_slider_event", None)
            if event is not None:
                event.cancel()
                self._mask2_edge_refine_slider_event = None

            if settle:
                self.apply_effects_lv(3, "mask2")
                return

            self.apply_effects_lv(3, "mask2", defer_draw=True)

            def _draw(_dt):
                self._mask2_edge_refine_slider_event = None
                self.apply_effects_lv(3, "mask2")

            self._mask2_edge_refine_slider_event = KVClock.schedule_once(_draw, 0.18)

        def set_effect_param(self, lv, effect, arg):
            if self.run_set2widget_all == True:
                return
            if lv == 1 and effect == 'distortion' and not self.can_open_liquify_editor():
                self._close_inactive_distortion_painters(None)
                self._sync_liquify_mask_editor_input_lock()
                return

            current_effects, current_param, _ = self._get_active_effects(lv=lv)
            current_effects[lv][effect].set2param2(current_param, arg)
            if lv == 1 and effect == 'distortion':
                self._close_inactive_distortion_painters(current_effects[lv][effect])
                self._sync_liquify_mask_editor_input_lock()
            self._apply_mask_overlay_policy(lv, effect)
            #self.apply_rotation_flip_for_wrapper()
            self.start_draw_image()

        def _mask_overlay_policy(self, lv, effect=None, subname=None, reason="param_change"):
            """Return 'show', 'hide', or 'preserve' for the active Mask2 overlay."""
            if reason == "tab_sync":
                return "preserve"

            effect_list = effect if isinstance(effect, list) else [effect] if effect is not None else []
            mask2_group = subname or (effect_list[0] if len(effect_list) == 1 else None)

            if lv == 1 and mask2_group == "distortion" and self._is_mask2_enabled():
                return "preserve"
            if lv == 3 and mask2_group in ("mask2", "mask2_quick_select", "mask_geometry"):
                return "show"
            if lv == 3 and mask2_group == "mask2_draw_effects":
                return "hide"
            return "hide"

        def _apply_mask_overlay_policy(self, lv, effect=None, subname=None, reason="param_change", refresh=True):
            policy = self._mask_overlay_policy(lv, effect, subname=subname, reason=reason)
            if policy == "preserve":
                return
            self.ids['mask_editor2'].set_draw_mask(policy == "show", refresh=refresh)

        def apply_rotation_flip_for_wrapper(self):
            # Calculate Rotation/Flip for Hardware
            #if self.ids["effects"].current_tab.text == "Ge":
            if False:
                rotation_effect = self.primary_effects[0]['geometry']
                angle = rotation_effect._get_param(self.primary_param,'rotation') + rotation_effect._get_param(self.primary_param,'rotation2')
                flip = rotation_effect._get_param(self.primary_param,'flip_mode')
            else:
                angle = 0
                flip = 0

            # Apply Hardware Rotation/Flip to Wrapper
            wrapper = self.ids['transform_wrapper']
            if wrapper:
                wrapper.size = self.texture.size
                
                # Scale for Flip
                sx = -1 if (flip & 1) else 1
                sy = -1 if (flip & 2) else 1

                # Reset transform
                wrapper.transform = KVMatrix()
                
                # Apply Rotation & Flip via Matrix
                mat = KVMatrix()
                mat.scale(sx, sy, 1)
                mat.rotate(math.radians(angle), 0, 0, 1)
                wrapper.transform = mat
                
                # Center wrapper (KV binding should handle this, but re-triggering might be needed if transform affected pos)
                # Setting transform updates pos, so KV binding might fight. 
                # Ideally KV binding `center: self.parent.center` wins on next layout cycle.
                # To be safe, we can manually set center after transform if KV doesn't catch it immediately.
                if self.ids['preview_widget']:
                    wrapper.center = self.ids['preview_widget'].center

        def begin_history_layer_ctrl(self, layer_ctrl, op, index, op_type):
            self.current_op = history.Operation(type="Layer")
            self.current_op.set_backup_layer(layer_ctrl, op, index, op_type)

        def end_history_layer_ctrl(self, layer_ctrl, op, index):
            if self.current_op is None:
                logging.warning("MainWidget.end_history_layer_ctrl None.")
                return

            # current_op が Layer 型でない場合 (effect 系の begin が混入したケース等) は安全に抜ける
            if getattr(self.current_op, 'type', None) != 'Layer' or not hasattr(self.current_op, 'layer_ctrl'):
                logging.warning(f"MainWidget.end_history_layer_ctrl Type Mismatch (got type={getattr(self.current_op, 'type', None)}).")
                self.current_op = None
                return

            if self.current_op.layer_ctrl is not layer_ctrl:
                logging.warning("MainWidget.end_history_layer_ctrl Unmatching.")
                return

            if self.current_op.set_update_layer(layer_ctrl, op, index) is not None:
                self.history.append(self.current_op)
                self.history_panel.set_history(self.history)
            self.current_op = None
            self._flush_pending_final_display_cache()

        def begin_history_reset_all(self):
            self.current_op = history.Operation(type="All")
            self.current_op.set_backup_all(self.primary_param, self.ids['mask_editor2'])

        def end_history_reset_all(self):
            if self.current_op is None:
                return

            if self.current_op.type != "All":
                logging.warning(f"MainWidget.end_history_reset_all Type Unmatching. {self.current_op.type}")
                return

            if self.current_op.check_backup_all(self.primary_param, self.ids['mask_editor2']) == False:
                self.history.append(self.current_op)
                self.history_panel.set_history(self.history)
            self.current_op = None
            self._flush_pending_final_display_cache()

        def reset_all(self):
            
            # セーブしないパラメータ（メタデータ等）は維持する。
            # 全体Resetでは crop_rect などの編集内容は維持しない。
            temp_param = {}
            params.copy_special_param(temp_param, self.primary_param)
            
            self.primary_param.clear()
            self.primary_param.update(temp_param)
            
            # 初期化パラメータ設定
            params.set_image_param(self.primary_param, self.imgset.img)

            # マスク関連全消去
            self.ids['mask2'].state = 'normal' # マスクモードを抜けないとおかしくなる
            self._disable_mask2()
            self.ids['mask_editor2'].clear_mask()
            
            # UIを先に初期値へ同期しないと、apply_effects_lv が古いウィジェット値をparamへ戻してしまう。
            self.set2widget_all(self.primary_effects, self.primary_param)

            # クロップエディタ起動時はそれの初期化も行う
            self.primary_effects[0]['crop'].reset2_crop_editor(self.primary_param)
            self.primary_effects[0]['crop'].reset_crop_editor()
            self.apply_effects_lv(0, 'crop') # 描画を走らせる

            # これでファイルが消えるはず
            self.save_current_sidecar()
           
            # UIと表示の更新
            self.set2widget_all(self.primary_effects, self.primary_param)

        def begin_history_effect_ctrl(self, lv, effect, subname=None):
            if self.run_set2widget_all == True:
                return False
            current_effects, current_param, mask_id = self._get_active_effects(lv=lv, subname=subname or effect)
            effect_list = effect if isinstance(effect, list) else [effect]
            self.current_op = history.Operation(lv, effect_list, subname, mask_id)
            self.current_op.set_backup(current_effects, current_param, subname)
            return True
        
        def end_history_effect_ctrl(self, lv, effect, subname=None):            
            effect_list = effect if isinstance(effect, list) else [effect]
            redraw_full_after_edit = lv == 3 and (
                subname == 'mask_geometry' or 'mask_geometry' in effect_list
            )
            
            if self.current_op is None:
                logging.warning(f"MainWidget.end_history_effect_ctrl None. {effect_list}")
                return
            
            if self.current_op.subname != subname:
                logging.warning(f"MainWidget.end_history_effect_ctrl Subname Unmatching. {effect_list}")
                return

            if self.current_op.lv != lv or self.current_op.effect_list != effect_list:
                logging.warning(f"MainWidget.end_history_effect_ctrl LV or Effect Unmatching. {effect_list}")

            current_effects, current_param, mask_id = self._get_active_effects(
                self.current_op.mask_id, lv=lv, subname=subname or effect
            )
            if self.current_op.set_update(current_effects, current_param, subname) is not None:
                self.history.append(self.current_op)
                self.history_panel.set_history(self.history)
            self.current_op = None
            self._flush_pending_final_display_cache()
            if redraw_full_after_edit:
                self.start_draw_image()

        def apply_crop_button_action(self, action):
            if not self.begin_history_effect_ctrl(0, 'crop'):
                return
            self.primary_effects[0]['crop'].apply_crop_button_action(self.primary_param, self, action)
            self._apply_mask_overlay_policy(0, 'crop')
            self.start_draw_image_and_crop(self.imgset)
            self.end_history_effect_ctrl(0, 'crop')
            self.save_current_sidecar() # 要るかどうかは微妙。CropEditorの操作はHistoryに入るので、Undo/RedoでCrop状態も変わる。

        def on_auto_adjust_press(self):
            if self.mask2_wait_full_load or getattr(self, "_actively_loading", False):
                return
            if self.imgset is None or getattr(self.imgset, "img", None) is None:
                return
            if not params.has_original_img_size(self.primary_param):
                return
            effect_list = [
                'exposure',
                'contrast',
                'tone',
                'dehaze',
                'clarity',
                'texture',
                'microcontrast',
                'vs_and_saturation',
                'color_separation',
            ]
            if not self.begin_history_effect_ctrl(2, effect_list):
                return
            try:
                adjustment = auto_adjust.compute_basic_auto_adjustment(
                    self.imgset.img,
                    crop_rect=params.get_crop_rect(self.primary_param),
                )
                self.primary_param.update(adjustment)
                self.set2widget_all(self.primary_effects, self.primary_param)
                effects.reeffect_all(self.primary_effects, 2)
                self.start_draw_image()
            except Exception:
                self.current_op = None
                logging.exception("auto adjust failed")
                return
            self.end_history_effect_ctrl(2, effect_list)

        def begin_rotation_crop_preview(self):
            self.primary_effects[0]['crop'].begin_rotation_preview(self.primary_param)

        def end_rotation_crop_preview(self):
            self.primary_effects[0]['crop'].end_rotation_preview(self.primary_param)

        def reset_switch_defaults_for_label(self, head_label):
            switch_id = None
            # self.ids の値はしばしば WeakProxy なので `is head_label` では一致しない。
            # 実体との比較は `==` に任せつつ、既に無効になったプロキシは ReferenceError とする。
            for key in self.ids:
                try:
                    value = self.ids[key]
                except ReferenceError:
                    continue
                if value is head_label:
                    switch_id = key
                    break
                try:
                    if value == head_label:
                        switch_id = key
                        break
                except ReferenceError:
                    continue
            if switch_id is None:
                return False

            target = self._switch_reset_targets().get(switch_id)
            if target is None:
                return False

            lv, effect, subname = target
            if switch_id == 'switch_lens_ghost':
                lv = self.lens_ghost_level()   # 実行レベル(lv1/lv2)に合わせる
            self._reset_effect_defaults(lv, effect, subname)
            return True

        def _switch_reset_targets(self):
            return build_switch_reset_targets()

        def _reset_effect_defaults(self, lv, effect, subname=None):
            effect_list = effect if isinstance(effect, list) else [effect]
            if not self.begin_history_effect_ctrl(lv, effect, subname):
                return

            current_effects, current_param, _ = self._get_active_effects(lv=lv, subname=subname or effect)
            if subname is not None:
                default_params = current_effects[lv][effect_list[0]].get_param_dict(current_param, subname)
            else:
                default_params = {}
                for effect_name in effect_list:
                    default_params.update(current_effects[lv][effect_name].get_param_dict(current_param))

            for key, value in default_params.items():
                current_param[key] = value.copy() if isinstance(value, list) else value

            for effect_name in effect_list:
                current_effects[lv][effect_name].set2widget(self, current_param)

            self.apply_effects_lv(lv, effect, subname=subname)
            self.end_history_effect_ctrl(lv, effect, subname)

        def _undo(self):        
            if self.history.can_undo():
                if self.history.undo(self):
                    self.history_panel.set_history(self.history)
                    #self.ids['mask_editor2'].set_draw_mask(lv == 3)
                    self.ids['mask_editor2'].update()       # MaskEditor2の表示を更新
                    self._set_diff_list_to_inpaint_edit()
                    self._sync_editor_modes_after_history()
                    self.start_draw_image_and_crop(self.imgset)

        def _redo(self):        
            if self.history.can_redo():
                if self.history.redo(self):
                    self.history_panel.set_history(self.history)
                    #self.ids['mask_editor2'].set_draw_mask(lv == 3)
                    self.ids['mask_editor2'].update()       # MaskEditor2の表示を更新
                    self._set_diff_list_to_inpaint_edit()
                    self._sync_editor_modes_after_history()
                    self.start_draw_image_and_crop(self.imgset)

        def _on_history_selected(self, index):
            if self.mask2_wait_full_load:
                return

            if index < self.history.current_index:
                n = self.history.current_index - index
                for _ in range(n):
                    self.history.undo(self)
                self.history_panel.set_history(self.history)
                #self.ids['mask_editor2'].set_draw_mask(lv == 3)
                self.ids['mask_editor2'].update()       # MaskEditor2の表示を更新
                self._set_diff_list_to_inpaint_edit()
                self._sync_editor_modes_after_history()
                self.start_draw_image_and_crop(self.imgset)

            elif index >= self.history.current_index:
                n = index - self.history.current_index
                for _ in range(n):
                    self.history.redo(self)
                self.history_panel.set_history(self.history)
                #self.ids['mask_editor2'].set_draw_mask(lv == 3)
                self.ids['mask_editor2'].update()       # MaskEditor2の表示を更新
                self._set_diff_list_to_inpaint_edit()
                self._sync_editor_modes_after_history()
                self.start_draw_image_and_crop(self.imgset)

        def _sync_editor_modes_after_history(self):
            self.primary_effects[0]['crop'].sync_crop_editor_mode_from_widget(self, self.primary_param)

        def reset_param(self, param):
            param.clear()

        def set2widget_all(self, _effects, param, reset_effects=True):
            if _effects is None:
                _effects = self.primary_effects
                param = self.primary_param

            self.run_set2widget_all = True
            try:
                effects.set2widget_all(self, _effects, param, reset_effects=reset_effects)
                if self._has_distortion_effect(_effects):
                    self._restore_distortion_tool_values_to_widgets(param)
            finally:
                self.run_set2widget_all = False

        def _viewer_snapshot_rating(self, file_path: str) -> int:
            if not file_path:
                return 0
            v = self.ids.get("viewer")
            if v:
                r = v.get_rating_for_path(file_path)
                if r is not None:
                    return r
            return 0

        def save_current_sidecar(self):
            if self.imgset is not None:
                param2 = effects.delete_default_param_all(self.primary_effects, self.primary_param) # プライマリのデフォルト値は消す
                param2['image_fidelity'] = getattr(self.imgset, 'fidelity', ImageFidelity.FULL).value
                param2.pop("rating", None)
                raw_r = 0
                if rating_utils.is_raw_path(self.imgset.file_path):
                    raw_r = self._viewer_snapshot_rating(self.imgset.file_path)
                result = params.save_json(
                    self.imgset.file_path, param2, self.ids['mask_editor2'], raw_sidecar_rating=raw_r
                )
                if result == False:
                    # 失敗時はファイルを削除
                    params.delete_empty_param_json(self.imgset.file_path)
                viewer = self.ids.get("viewer")
                if viewer:
                    viewer.set_pmck_indicator_for_path(self.imgset.file_path)

        def _enqueue_unfinished_ai_noise_for_path(self, file_path, param, *, reason):
            if not self.ai_job_manager or not file_path or not isinstance(param, dict):
                return
            if not ai_noise_enabled(param):
                return
            if param.get("ai_noise_reduction_result") is not None:
                return
            if param.get("_ai_noise_reduction_result_deferred"):
                return
            try:
                self.ai_job_manager.enqueue_ai_noise_file(file_path, param)
                logging.info(
                    "AI-NR unfinished file job enqueued: file=%s reason=%s image_fidelity=%s",
                    file_path,
                    reason,
                    param.get("image_fidelity"),
                )
            except Exception:
                logging.exception("failed to enqueue unfinished AI-NR job: %s", file_path)

        def _snapshot_current_param(self):
            snap = params.serialize(self.primary_param, self.ids['mask_editor2']) or {"primary_param": {}}
            params.copy_special_param(snap["primary_param"], self.primary_param)
            params.copy_remain_param(snap["primary_param"], self.primary_param)
            return snap

        def _open_effect_selector(self, on_decide):
            selected = preset_utils.get_saved_selector_switch_keys(config)
            dialog = EffectSelector(selected_switch_keys=selected)

            def _decide(inst, _selection):
                switch_keys = inst.get_selected_switch_keys()
                preset_utils.save_selector_switch_keys(config, switch_keys)
                on_decide(switch_keys)

            dialog.bind(on_decide=_decide)
            dialog.open()

        def _can_use_effect_settings_transfer(self):
            return (
                self.imgset is not None
                and self.loading is False
                and self.mask2_wait_full_load is False
                and self.primary_param.get("image_fidelity") == ImageFidelity.FULL.value
            )

        def _warn_effect_settings_transfer_not_ready(self):
            self.show_warning_dialog("Please wait until the image finishes loading.")

        def copy_effect_settings(self):
            if not self._can_use_effect_settings_transfer() or not self.primary_param:
                self._warn_effect_settings_transfer_not_ready()
                return

            def _copy(switch_keys):
                partial = preset_utils.collect_selected_primary_param(
                    self.primary_effects, self.primary_param, switch_keys
                )
                self._copied_effect_param = {
                    "switch_keys": list(switch_keys),
                    "primary_param": partial,
                }

            self._open_effect_selector(_copy)

        def _apply_partial_to_current(self, partial_param, history_name="Paste Settings"):
            if self.imgset is None:
                return False
            op = history.Operation(type="All")
            op.set_backup_all(self.primary_param, self.ids['mask_editor2'])
            preset_utils.apply_partial_primary_param(self.primary_param, partial_param)
            self.set2widget_all(self.primary_effects, self.primary_param)
            self.start_draw_image()
            if op.set_update_all(self.primary_param, self.ids['mask_editor2'], history_name) is not None:
                self.history.append(op)
                self.history_panel.set_history(self.history)
            self.save_current_sidecar()
            return True

        def paste_effect_settings(self):
            if not self._can_use_effect_settings_transfer():
                self._warn_effect_settings_transfer_not_ready()
                return
            copied = self._copied_effect_param
            if not copied or not copied.get("primary_param"):
                self.show_warning_dialog("No copied settings.")
                return
            cards = self.ids['viewer'].get_selected_cards()
            if not cards:
                if self.imgset is None:
                    return
                self._apply_partial_to_current(copied["primary_param"])
                return
            if len(cards) == 1:
                card = cards[0]
                if self.imgset is not None and card.file_path == self.imgset.file_path:
                    self._apply_partial_to_current(copied["primary_param"])
                    return
            self._paste_effect_settings_to_cards(cards, copied["primary_param"])

        def _paste_effect_settings_to_cards(self, cards, partial_param):
            current_path = self.imgset.file_path if self.imgset is not None else None
            current_backup = self._snapshot_current_param() if current_path and any(c.file_path == current_path for c in cards) else None

            def _job():
                items = []
                for card in cards:
                    item = preset_utils.backup_pmck_for_batch(card.file_path)
                    preset_utils.apply_partial_to_pmck_file(card.file_path, partial_param)
                    items.append(item)
                return items

            processing_dialog.set_processing_text("Applying settings to selected images...")
            items = processing_dialog.wait_processing(_job)
            if current_backup is not None:
                preset_utils.apply_partial_primary_param(self.primary_param, partial_param)
                self.set2widget_all(self.primary_effects, self.primary_param)
                self.start_draw_image()
            current_update = self._snapshot_current_param() if current_backup is not None else None
            op = history.Operation(type="BatchPaste")
            op.set_batch_paste(items, current_backup=current_backup, current_update=current_update)
            self.history.append(op)
            self.history_panel.set_history(self.history)
            for item in items:
                self._refresh_pmck_indicator_for_image_path(item.get("image_path"))
            self._enqueue_ai_noise_jobs_after_batch_paste(items)

        def _enqueue_ai_noise_jobs_after_batch_paste(self, items):
            if not self.ai_job_manager:
                return
            current_path = self.imgset.file_path if self.imgset is not None else None
            for item in items or []:
                image_path = item.get("image_path")
                if not image_path:
                    continue
                if image_path == current_path:
                    continue
                try:
                    data = preset_utils.read_pmck_dict(pmck_store.image_pmck_path(image_path))
                    primary = data.get("primary_param") or {}
                    if ai_noise_enabled(primary):
                        self.ai_job_manager.enqueue_ai_noise_file(image_path, primary)
                except Exception:
                    logging.exception("failed to enqueue AI-NR job after batch paste: %s", image_path)

        def on_import_path_applied(self, directory):
            if not self.ai_job_manager or not directory:
                return
            threading.Thread(
                target=self._enqueue_resumable_ai_noise_jobs_for_folder,
                args=(directory,),
                daemon=True,
            ).start()

        def _enqueue_resumable_ai_noise_jobs_for_folder(self, directory):
            try:
                root = os.path.abspath(directory)
                file_names = sorted(os.listdir(root))
            except OSError:
                return
            current_path = self.imgset.file_path if self.imgset is not None else None
            supported = define.SUPPORTED_FORMATS_RGB + define.SUPPORTED_FORMATS_RAW + define.SUPPORTED_FORMATS_EXR
            restored = 0
            for file_name in file_names:
                image_path = os.path.join(root, file_name)
                if not image_path.lower().endswith(supported):
                    continue
                if current_path and os.path.abspath(image_path) == os.path.abspath(current_path):
                    continue
                try:
                    data = preset_utils.read_pmck_dict(pmck_store.image_pmck_path(image_path))
                    primary = data.get("primary_param") or {}
                    if not ai_noise_enabled(primary):
                        continue
                    if primary.get("ai_noise_reduction_result") is not None:
                        continue
                    self.ai_job_manager.enqueue_ai_noise_file(image_path, primary)
                    restored += 1
                except Exception:
                    logging.exception("failed to restore AI-NR job for %s", image_path)
            if restored:
                logging.info("restored %d resumable AI-NR job(s) for folder: %s", restored, root)

        def _refresh_pmck_indicator_for_image_path(self, image_path):
            viewer = self.ids.get("viewer")
            if viewer and image_path:
                viewer.set_pmck_indicator_for_path(
                    image_path, exists=pmck_store.exists_image(image_path)
                )

        def show_warning_dialog(self, message):
            device.alert(str(message), title="Warning", icon="caution")

        def start_add_preset(self):
            if self.mask2_wait_full_load:
                return
            if not self._can_use_effect_settings_transfer() or not self.primary_param:
                self._warn_effect_settings_transfer_not_ready()
                return

            def _selected(switch_keys):
                partial = preset_utils.collect_selected_primary_param(
                    self.primary_effects, self.primary_param, switch_keys
                )
                self._open_preset_name_dialog(partial)

            self._open_effect_selector(_selected)

        def _open_preset_name_dialog(self, partial_param):
            try:
                preset_name = device.prompt_native(
                    message="Preset name",
                    title="Save Preset",
                    default="",
                    show_cancel=True,
                    ascii_only=False,
                )
            except Exception as e:
                logging.warning("preset name prompt failed: %s", e)
                self.show_warning_dialog(str(e))
                return
            if not preset_name or not preset_name.strip():
                return
            try:
                path = preset_utils.preset_path_for_name(preset_name)
                preset_utils.save_preset_json(path, preset_utils.build_preset_dict(partial_param))
            except Exception as e:
                self.show_warning_dialog(str(e))
                return
            self.refresh_preset_panel()

        def refresh_preset_panel(self):
            panel = getattr(self, "preset_panel", None)
            if panel is not None:
                panel.refresh_list()

        def apply_preset_path(self, preset_path):
            if self.mask2_wait_full_load:
                return
            if self.imgset is None:
                return
            try:
                partial = preset_utils.load_preset_json(preset_path)
            except Exception as e:
                self.show_warning_dialog(str(e))
                return
            self._apply_partial_to_current(partial, history_name="Apply Preset")

        def confirm_delete_preset(self, preset_name, preset_path):
            if self.mask2_wait_full_load:
                return
            if not device.confirm(
                f'Delete preset "{preset_name}"?',
                title="Delete Preset",
                ok_label="Delete",
                cancel_label="Cancel",
                icon="caution",
            ):
                return
            try:
                os.remove(preset_path)
            except FileNotFoundError:
                # 既に削除済みなら目的は達成済み（無害）
                pass
            except OSError as e:
                self.show_warning_dialog(str(e))
                return
            self.refresh_preset_panel()

        def _log_slow_load_step(self, label, start_time, threshold=0.05):
            elapsed = time.perf_counter() - start_time
            if elapsed >= threshold:
                logging.warning("[LOAD_PERF] %s took %.3fs", label, elapsed)
            return time.perf_counter()
        
        @kvmainthread
        def on_select(self, card):
            logging.debug("[PERF] on_select: Start. Time: %s", time.time())
            perf_trace.select_start(card.file_path if card is not None else None)
            perf_trace.event("on_select.enter")
            previous_file_path = self.imgset.file_path if self.imgset is not None else None
            current_file_path = card.file_path if card is not None else None
            # ロード開始
            self.loading = True
            self.mask2_wait_full_load = True
            if 'mask2' in self.ids:
                # ファイル切替時: 状態を戻してから無効化
                self.ids['mask2'].state = 'normal'
            self.update_mask2_options_enabled()
            self._actively_loading = True  # アニメーション表示開始
            if not threads.primary_param_lock.acquire(blocking=False):
                self._deferred_select_card = card
                if self._deferred_select_event is None:
                    logging.warning(
                        "on_select deferred while draw/update owns primary_param_lock: %s",
                        current_file_path,
                    )

                    def _retry_select(_dt):
                        self._deferred_select_event = None
                        retry_card = self._deferred_select_card
                        self._deferred_select_card = None
                        self.on_select(retry_card)

                    self._deferred_select_event = KVClock.schedule_once(_retry_select, 0.05)
                return
            try:
                # Mask1 edit uses temporary original-image geometry; restore it before saving/switching.
                self._cancel_mask1_mode(redraw=False)
                # 前の設定を保存
                self.save_current_sidecar()
                self._enqueue_unfinished_ai_noise_for_path(
                    previous_file_path,
                    self.primary_param,
                    reason="selection_changed",
                )
                self.cache_system.on_image_selection_changed(
                    owner=self,
                    previous_file_path=previous_file_path,
                    current_file_path=current_file_path,
                )
                # 前のエフェクトを終了
                effects.finalize_all(self.primary_effects, self.primary_param, self)
                # 空のイメージをセット
                self.empty_image()
            finally:
                threads.primary_param_lock.release()

            if card is not None:
                self._expected_file_path = card.file_path
                self._clear_exif_data()
                self._show_cached_final_display_image(card.file_path)
                self.cache_system.register_for_preload(card.file_path, card.exif_data, None, True)
                try:
                    self.cache_system.get_file(card.file_path, lambda f1, f2, f3, f4, f5, f6: file_cache_system.run_method(self, "on_fcs_get_file", config._config, f1, f2, f3, f4, f5, f6))
                except FileNotFoundError:
                    logging.exception("FCS selection load disappeared before callback registration: %s", card.file_path)
                    self.loading = False
                    self._actively_loading = False
                    self.update_mask2_options_enabled()
            else:
                self._expected_file_path = None
                self._clear_exif_data()
                # カードなし（フォルダ空など）— get_file が呼ばれず loading が解除されないのを防ぐ
                self.loading = False
                self._actively_loading = False
                self.update_mask2_options_enabled()
        
        @kvmainthread
        def on_fcs_get_file(self, file_path, imgset, exif_data, param, history_obj, stage):
            stage = coerce_load_stage(stage)
            _load_perf_t = time.perf_counter()
            if file_path != getattr(self, '_expected_file_path', None):
                logging.info(
                    "FCS: 現在の選択と異なるファイルのコールバックを無視します: %r (expected %r)",
                    file_path,
                    getattr(self, '_expected_file_path', None),
                )
                return
            logging.debug("[PERF] on_fcs_get_file: Called. stage: %s, Time: %s", stage, time.time())
            perf_trace.event("on_fcs_get_file.enter", stage=str(stage))
            self._memory_last_load_stage = stage
            _img = getattr(imgset, "img", None) if imgset is not None else None
            shape_str = getattr(_img, "shape", None) if _img is not None else None
            logging.debug(
                "Load image SHAPE: %s fidelity: %s, Proc: %s",
                shape_str,
                getattr(imgset, 'fidelity', None),
                stage,
            )

            if imgset is None or getattr(imgset, "img", None) is None:
                logging.error(f"画像データがありません: {file_path} (stage={stage})")
                self.empty_image()
                if self.processor:
                    self.processor.cancel_all()
                self.loading = False
                self._actively_loading = False
                # 失敗パスはここで終わり blit_image に到達しない。トレースを取りこぼさないよう
                # ここで明示的に flush する。
                perf_trace.event("on_fcs_get_file.load_failed", stage=str(stage))
                perf_trace.flush(reason="load_failed")
                return

            if _load_stage_allows_ui(stage, imgset):
                self.loading = False
                # ここで初めて param と imgset が編集可能な対になる。UI 全体を解禁。
                self.image_loaded = True
            if _load_stage_ends_file_loading_indicator(stage, imgset):
                self._actively_loading = False
            # Mask2 はフル読み込み完了までは無効化
            if stage == LoadStage.RGB_DONE or (
                stage == LoadStage.FULL_DECODE and getattr(imgset, 'fidelity', None) == ImageFidelity.FULL
            ):
                self.mask2_wait_full_load = False
            self.update_load_dependent_panels_enabled()
            self.update_mask2_options_enabled()
            _load_perf_t = self._log_slow_load_step(f"on_fcs.{stage.name}.update_load_ui", _load_perf_t)

            if stage in (LoadStage.FIRST_PAINTABLE, LoadStage.RGB_DONE):
                card = self.ids['viewer'].get_card(file_path)
                _load_perf_t = self._log_slow_load_step(f"on_fcs.{stage.name}.get_card", _load_perf_t)
                if card is not None:
                    # 一度も描画してないので値が設定されてない。暫定処置
                    self.update_preview_texture_size()
                    _load_perf_t = self._log_slow_load_step(f"on_fcs.{stage.name}.update_preview_texture_size", _load_perf_t)
                    self.ids['mask_editor2'].set_texture_size(config.get_config('preview_width'), config.get_config('preview_height'))
                    _load_perf_t = self._log_slow_load_step(f"on_fcs.{stage.name}.mask_set_texture_size", _load_perf_t)
                    self.ids['mask_editor2'].set_primary_param(param, params.get_disp_info(param))
                    _load_perf_t = self._log_slow_load_step(f"on_fcs.{stage.name}.mask_set_primary_param", _load_perf_t)

                    # フル解像のときだけ pmck から重い結果を復元（RAW プレビュー段階ではスキップ）
                    param['image_fidelity'] = getattr(imgset, 'fidelity', ImageFidelity.FULL).value
                    load_heavy = param['image_fidelity'] == ImageFidelity.FULL.value
                    pmck_dict = params.load_json(file_path, param, self.ids['mask_editor2'], load_heavy=load_heavy)
                    _load_perf_t = self._log_slow_load_step(f"on_fcs.{stage.name}.load_pmck", _load_perf_t)
                    # RAW: プレビュー段階で読んだ dict を保持しておき、FULL_DECODE 遷移時の
                    # merge_heavy_from_pmck で再パースを避ける。
                    self._last_pmck_dict = (file_path, pmck_dict)

            # Cancel previous background tasks
            if self.processor:
                restart_idle_worker = not (
                    stage == LoadStage.FIRST_PAINTABLE
                    and getattr(imgset, 'fidelity', None) == ImageFidelity.PREVIEW
                )
                self.processor.cancel_all(restart_idle=restart_idle_worker)
                self.ids['histogram'].set_histogram_data(None)  # Reset histogram?
                _load_perf_t = self._log_slow_load_step(f"on_fcs.{stage.name}.cancel_async_worker", _load_perf_t)

            with threads.primary_param_lock:
                if stage in (LoadStage.FIRST_PAINTABLE, LoadStage.RGB_DONE):
                    # １回目の時だけパラメータを反映して、編集できる様にする
                    self.primary_param.clear()
                    self.primary_param.update(param)
                    self.primary_param['_source_file_path'] = file_path
                    if file_path and rating_utils.is_rgb_path(file_path):
                        # サムネ用 get_metadata 由来は param / exif_data 引数（同一参照想定）に。空なら exiftool 追読だけが頼り
                        pex = self.primary_param.get("exif_data")
                        if not isinstance(pex, dict):
                            pex = {}
                        if not pex and isinstance(exif_data, dict) and exif_data:
                            pex = exif_data
                        self.primary_param["exif_data"] = {**pex}
                        ex = self.primary_param["exif_data"]
                        rating_io.merge_xmp_star_tags_into_exif(file_path, ex)
                        try:
                            r = int(rating_utils.parse_exif_rating_value(ex) or 0)
                            self.ids["viewer"].set_rating_for_path(file_path, r)
                        except Exception:
                            logging.exception("failed to sync rating for %s", file_path)
                    logging.debug("[PERF] on_fcs_get_file: Merged Params. Time: %s", time.time())
                    perf_trace.event("on_fcs_get_file.params_merged")
                    self.set2widget_all(self.primary_effects, self.primary_param)
                    _load_perf_t = self._log_slow_load_step(f"on_fcs.{stage.name}.set2widget_all", _load_perf_t)
                    self.update_mask2_options_enabled()
                    _load_perf_t = self._log_slow_load_step(f"on_fcs.{stage.name}.update_mask2_options_after_widgets", _load_perf_t)

                    # 特別あつかいでエディタを起動できるなら起動する。
                    # ここでは描画キックを抑止し、後段の start_draw_image_and_crop に集約する。
                    self.apply_effects_lv(1, 'distortion', defer_draw=True)
                    _load_perf_t = self._log_slow_load_step(f"on_fcs.{stage.name}.apply_distortion_editor", _load_perf_t)
                    self.apply_effects_lv(0, 'crop', defer_draw=True)
                    _load_perf_t = self._log_slow_load_step(f"on_fcs.{stage.name}.apply_crop_editor", _load_perf_t)

                    # ヒストリーの設定
                    if history_obj is None:
                        self.history = history.History()
                        self.cache_system.set_history(file_path, self.history)
                    else:
                        self.history = history_obj

                    self.history_panel.set_history(self.history)
                    _load_perf_t = self._log_slow_load_step(f"on_fcs.{stage.name}.history_panel", _load_perf_t)

                # RAW フルデコード完了時にレンズ補正まわりを param から反映
                if stage == LoadStage.FULL_DECODE and param.get('rgb_or_raw') == 'raw':
                    self.primary_param['lens_modifier'] = param['lens_modifier']
                    if param['lens_modifier'] == True:
                        self.primary_param['exif_data'] = param['exif_data']
                    self.primary_param['rgb_or_raw'] = param['rgb_or_raw']
                    self.primary_param['auto_exposure'] = param['auto_exposure']
                    # プレビュー段階で入った original_img_size / img_size のままだと
                    # フル解像バッファとずれて crop_image が空領域になり黒画面になる。
                    # FULL_DECODEは履歴外で来るため、ユーザー編集値 crop_rect 自体は上書きしない。
                    for _k in ('original_img_size', 'img_size'):
                        if _k in param:
                            self.primary_param[_k] = param[_k]
                    if params.get_crop_rect(self.primary_param) is not None:
                        disp_info = core.convert_rect_to_info(
                            params.get_crop_rect(self.primary_param),
                            config.get_preview_texture_side()/max(self.primary_param['original_img_size'])
                        )
                        params.set_disp_info(self.primary_param, disp_info)
                    self.ids['mask_editor2'].set_primary_param(
                        self.primary_param, params.get_disp_info(self.primary_param)
                    )
                    self.primary_effects[0]['crop'].sync_crop_editor_mode_from_widget(self, self.primary_param)

                self.imgset = imgset
                self.primary_param['_source_file_path'] = file_path

                fid = getattr(imgset, 'fidelity', ImageFidelity.FULL)
                self.primary_param['image_fidelity'] = fid.value
                prev_fid = getattr(self, '_last_image_fidelity', None)
                if fid == ImageFidelity.FULL and prev_fid == ImageFidelity.PREVIEW:
                    cached_pmck = None
                    last = getattr(self, '_last_pmck_dict', None)
                    if last is not None and last[0] == file_path:
                        cached_pmck = last[1]
                    params.merge_heavy_from_pmck(
                        file_path, self.primary_param, self.ids['mask_editor2'], cached_dict=cached_pmck
                    )
                    _load_perf_t = self._log_slow_load_step(f"on_fcs.{stage.name}.merge_heavy_pmck", _load_perf_t)
                self._last_image_fidelity = fid

                params.apply_original_geometry_if_missing(self.primary_param, imgset.img)
                _load_perf_t = self._log_slow_load_step(f"on_fcs.{stage.name}.apply_original_geometry", _load_perf_t)
                if not params.has_original_img_size(self.primary_param):
                    logging.error("on_fcs_get_file: デコード画像からも original_img_size を確定できません")
                    self.loading = False
                    self._actively_loading = False
                    perf_trace.event("on_fcs_get_file.no_geometry", stage=str(stage))
                    perf_trace.flush(reason="no_geometry")
                    return

                effects.reeffect_all(self.primary_effects)
                _load_perf_t = self._log_slow_load_step(f"on_fcs.{stage.name}.reeffect_all", _load_perf_t)
                preview_fast_display = (
                    stage == LoadStage.FIRST_PAINTABLE
                    and getattr(imgset, 'fidelity', None) == ImageFidelity.PREVIEW
                )
                self.start_draw_image_and_crop(
                    imgset,
                    fast_display=preview_fast_display,
                    skip_histogram=preview_fast_display,
                )
                self._log_slow_load_step(f"on_fcs.{stage.name}.start_draw", _load_perf_t)
            if stage in (LoadStage.FIRST_PAINTABLE, LoadStage.RGB_DONE) or (
                stage == LoadStage.FULL_DECODE and param.get('rgb_or_raw') == 'raw'
            ):
                display_exif = self.primary_param.get("exif_data", exif_data)
                if isinstance(display_exif, dict) and display_exif:
                    self._set_exif_data(display_exif, file_path=file_path)
            self._sync_exif_rating_row()

        def _image_interaction_ready(self):
            """ズーム／ドラッグ等のプレビュー上の操作を受け付けて良い状態か。

            画像が確定していない間（起動直後・フォルダ切替直後・ロード途中）に操作させると、
            is_zoomed 等の内部状態だけが変化して描画は走らず、見た目の整合性が崩れる。
            """
            return (
                self.image_loaded
                and self.imgset is not None
                and getattr(self.imgset, 'img', None) is not None
            )

        def _debug_mask_zoom_sync_enabled(self):
            return os.getenv("PLATYPUS_DEBUG_MASK_ZOOM_SYNC", "0").strip().lower() in {"1", "true", "yes", "on"}

        def _mask_zoom_sync_log(self, message, *args):
            if self._debug_mask_zoom_sync_enabled():
                logging.warning("[MASK_ZOOM_SYNC] " + message, *args)

        def _preview_texture_pos_or_none(self, touch):
            """preview Image 上の texture 座標を返す。レターボックス/外側なら None。

            preview_widget は黒帯を含むため、そこで double-tap すると click_x/y が
            texture 外になり、zoom crop が端へ clamp される。マスク overlay とは別の
            「ズーム中心が意図せず飛ぶ」原因になるので、zoom-in 時は実画像領域だけ許可する。
            """
            return self._preview_texture_pos_from_window_pos(touch.pos)

        def _preview_texture_pos_from_window_pos(self, pos):
            preview = self.ids.get('preview')
            if preview is None:
                return None
            tex_x, tex_y = utils.to_texture(pos, preview)
            tw, th = getattr(preview, 'texture_size', (0, 0))
            if tex_x < 0 or tex_y < 0 or tex_x >= tw or tex_y >= th:
                self._mask_zoom_sync_log(
                    "ignore texture-outside pos=%s tex=(%.2f,%.2f) texture_size=%s preview_pos=%s preview_size=%s",
                    pos, tex_x, tex_y, (tw, th), tuple(preview.pos), tuple(preview.size),
                )
                return None
            return tex_x, tex_y

        def _preview_texture_pos_to_image_pos(self, tex_pos):
            disp_info = params.get_disp_info(self.primary_param)
            if disp_info is None or tex_pos is None:
                return None
            preview = self.ids.get('preview')
            if preview is not None:
                texture_width, texture_height = getattr(preview, 'texture_size', config.get_preview_texture_size())
            else:
                texture_width, texture_height = config.get_preview_texture_size()
            if texture_width <= 0 or texture_height <= 0:
                return None
            _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(
                texture_width,
                texture_height,
                disp_info,
            )
            if disp_info[2] >= disp_info[3]:
                scale = texture_width / max(1, disp_info[2])
            else:
                scale = texture_height / max(1, disp_info[3])
            if scale <= 0:
                return None
            image_x = disp_info[0] + (tex_pos[0] - offset_x) / scale
            image_y = disp_info[1] + (tex_pos[1] - offset_y) / scale
            return image_x, image_y

        def _clamp_zoom_ratio(self, value):
            return min(4.0, max(0.1, float(value)))

        def _sync_zoom_ratio_slider(self, *args):
            try:
                slider = self.ids['slider_zoom_ratio']
            except Exception:
                return
            disabled = (not self.image_loaded) or (not self.is_zoomed)
            percent = int(round(self.zoom_ratio * 100))
            if hasattr(slider, 'ids') and 'slider' in slider.ids:
                current = slider.ids['slider'].value
                if not math.isclose(current, percent, rel_tol=0.0, abs_tol=1e-4):
                    slider.disabled = True
                    slider.ids['slider'].value = percent
                    slider.value = percent
                    slider.ids['input'].set_value(percent)
            slider.disabled = disabled

        def on_image_loaded(self, *args):
            self._sync_zoom_ratio_slider()

        def on_is_zoomed(self, *args):
            self._sync_zoom_ratio_slider()

        def on_zoom_ratio(self, *args):
            self._sync_zoom_ratio_slider()

        def _current_zoom_center_pos(self):
            disp_info = None
            # Mask2 Geometry 中は pipeline が実表示用 disp_info を primary_param に
            # 書き戻さない経路があるため、表示と同期済みの MaskEditor2 を優先する。
            try:
                mask_editor = self.ids.get('mask_editor2')
                tcg_info = getattr(mask_editor, 'tcg_info', None)
                if self._is_mask2_on() and isinstance(tcg_info, dict):
                    disp_info = params.get_disp_info(tcg_info)
            except Exception:
                disp_info = None
            if disp_info is None:
                disp_info = params.get_disp_info(self.primary_param)
            if disp_info is None:
                return None
            dx, dy, dw, dh, _ = disp_info
            return (dx + dw / 2, dy + dh / 2)

        def on_zoom_ratio_slider(self, percent):
            ratio = self._clamp_zoom_ratio(percent / 100.0)
            if math.isclose(self.zoom_ratio, ratio, rel_tol=0.0, abs_tol=1e-4):
                return
            center_pos = self._current_zoom_center_pos() if self.is_zoomed else None
            self.zoom_ratio = ratio
            if self.is_zoomed and self._image_interaction_ready():
                effects.reeffect_all(self.primary_effects, 1)
                self.start_draw_image_and_crop(
                    self.imgset,
                    center_pos=center_pos,
                    fast_display=True,
                    skip_histogram=True,
                )

        def on_zoom_ratio_after_edit(self):
            if self.is_zoomed and self._image_interaction_ready():
                self.start_draw_image_and_crop(
                    self.imgset,
                    center_pos=self._current_zoom_center_pos(),
                )

        def _reset_preview_zoom(self):
            if not self.is_zoomed or not self._image_interaction_ready():
                return False
            self.is_zoomed = False
            self.click_x, self.click_y = 0, 0
            self.drag_center_start = None
            self.update_preview_texture_size()
            disp_info = core.convert_rect_to_info(
                params.get_crop_rect(self.primary_param),
                config.get_preview_texture_side() / max(self.primary_param['original_img_size']),
            )
            params.set_disp_info(self.primary_param, disp_info)
            effects.reeffect_all(self.primary_effects, 1)
            self.start_draw_image_and_crop(self.imgset)
            return True

        def _mask1_full_preview_disp_info(self):
            original_img_size = self.primary_param.get('original_img_size')
            if not original_img_size:
                return None
            width, height = original_img_size
            scale = config.get_preview_texture_side() / max(width, height)
            return core.get_initial_disp_info(width, height, scale)

        _MASK1_GEOMETRY_BYPASS_KEYS = (
            'rotation',
            'rotation2',
            'flip_mode',
            'switch_distortion_correction',
            'lens_distortion_strength',
            'lens_distortion_scale',
            'correct_horizontal',
            'correct_vertical',
            'focal_length',
            'four_points',
            'reference_lines',
            'mesh_size',
            'control_points',
            'matrix',
            'crop_rect',
            'disp_info',
            'img_size',
            'switch_distortion',
        )

        def _backup_mask1_geometry_params(self):
            return {
                key: (key in self.primary_param, copy.deepcopy(self.primary_param.get(key)))
                for key in self._MASK1_GEOMETRY_BYPASS_KEYS
            }

        def _restore_mask1_geometry_params(self, backup):
            for key, (had_key, value) in backup.items():
                if had_key:
                    self.primary_param[key] = copy.deepcopy(value)
                else:
                    self.primary_param.pop(key, None)

        def _apply_mask1_geometry_bypass(self, disp_info):
            original_img_size = self.primary_param.get('original_img_size')
            if not original_img_size:
                return
            width, height = original_img_size
            params.set_crop_rect(self.primary_param, core.get_initial_crop_rect(width, height))
            params.set_disp_info(self.primary_param, disp_info)
            self.primary_param['img_size'] = (width, height)
            self.primary_param['rotation'] = 0
            self.primary_param['rotation2'] = 0
            self.primary_param['flip_mode'] = 0
            self.primary_param['switch_distortion_correction'] = False
            self.primary_param['lens_distortion_strength'] = 0
            self.primary_param['lens_distortion_scale'] = 0
            self.primary_param['correct_horizontal'] = 0
            self.primary_param['correct_vertical'] = 0
            self.primary_param['focal_length'] = 20
            self.primary_param['four_points'] = []
            self.primary_param['reference_lines'] = []
            self.primary_param['mesh_size'] = [4, 4]
            self.primary_param['control_points'] = {}
            self.primary_param['matrix'] = np.eye(3)
            self.primary_param['switch_distortion'] = False

        @staticmethod
        def _mask1_identity_tcg_info():
            """mask1 バイパス空間(回転/行列/フリップなし)の tcg_info。"""
            return {
                'rotation': 0.0,
                'rotation2': 0.0,
                'flip_mode': 0,
                'matrix': np.eye(3),
            }

        def _mask1_map_view_center(self, center, tcg_from, tcg_to):
            """padded-square 座標の表示中心 center を、tcg_from のジオメトリ空間から
            tcg_to のジオメトリ空間へ写像する(同じ画像内容の位置を指し続ける)。"""
            imax = max(self.primary_param['original_img_size']) / 2
            cx, cy = center[0] - imax, center[1] - imax
            cx, cy = params.center_rotate_invert(cx, cy, tcg_from)  # from空間 → オリジナル
            cx, cy = params.center_rotate(cx, cy, tcg_to)           # オリジナル → to空間
            return cx + imax, cy + imax

        def _mask1_view_anchor_click(self, target_disp_info, content_pos):
            """zoom_crop_source_info が content_pos(target 空間の padded-square 座標)を
            中心にズーム窓を作るような click_x/y を逆算する。
            crop 側: crop_center = disp[0:2] + (click - offset) / base_scale"""
            tex_w = config.get_config('preview_width')
            tex_h = config.get_config('preview_height')
            _, _, offset_x, offset_y = core.crop_size_and_offset_from_texture(tex_w, tex_h, target_disp_info)
            if target_disp_info[2] >= target_disp_info[3]:
                base_scale = tex_w / target_disp_info[2]
            else:
                base_scale = tex_h / target_disp_info[3]
            click_x = offset_x + (content_pos[0] - target_disp_info[0]) * base_scale
            click_y = offset_y + (content_pos[1] - target_disp_info[1]) * base_scale
            return click_x, click_y

        def enter_mask1_full_preview_mode(self, source, redraw=False):
            if not self._image_interaction_ready():
                return
            if self._mask1_full_preview_backup is None:
                self._mask1_full_preview_backup = {
                    'crop_image_view_key': self.crop_image_view_key,
                    'geometry_params': self._backup_mask1_geometry_params(),
                }

            self._mask1_full_preview_sources.add(source)
            disp_info = self._mask1_full_preview_disp_info()
            if disp_info is None:
                return

            # ズーム中は「見ていた画像内容」を維持する。現在のズーム窓中心(現ジオメトリ
            # 空間の padded-square 座標)を回転/行列の逆変換でバイパス(オリジナル)空間へ
            # 写像し、同じ内容位置を指す click アンカーに引き直す。以前は常に画像中心へ
            # 飛ばしていたため、ジオメトリの状態によって見た目上「違う場所」へ移動していた。
            if self.is_zoomed:
                cur_disp = params.get_disp_info(self.primary_param)
                if cur_disp is not None:
                    tcg_from = params.param_to_tcg_info(self.primary_param)
                    center = (cur_disp[0] + cur_disp[2] / 2, cur_disp[1] + cur_disp[3] / 2)
                    center = self._mask1_map_view_center(center, tcg_from, self._mask1_identity_tcg_info())
                    self.click_x, self.click_y = self._mask1_view_anchor_click(disp_info, center)
            else:
                self.click_x, self.click_y = 0, 0
            self.drag_center_start = None
            self.crop_image = None
            self.crop_image_view_key = None
            self._apply_mask1_geometry_bypass(disp_info)
            effects.reeffect_all(self.primary_effects, 0)
            if redraw:
                self.start_draw_image_and_crop(self.imgset)

        def exit_mask1_full_preview_mode(self, source, redraw=False):
            self._mask1_full_preview_sources.discard(source)
            if self._mask1_full_preview_sources or self._mask1_full_preview_backup is None:
                return

            backup = self._mask1_full_preview_backup
            self._mask1_full_preview_backup = None
            # is_zoomed/zoom_ratio は mask1 編集中にユーザーが変更している可能性があるため、
            # 突入前の値へ強制的に戻さない。ジオメトリ復元前に、バイパス空間での現在の
            # 表示中心を控えておく(ズーム継続時のアンカー引き直しに使う)。
            pre_disp = params.get_disp_info(self.primary_param)
            self.drag_center_start = None
            self.crop_image = None
            self.crop_image_view_key = backup.get('crop_image_view_key')
            self._restore_mask1_geometry_params(backup.get('geometry_params', {}))
            # 退避してあった disp_info は「突入時のビューポート」のスナップショットなので
            # そのまま使わない(編集中にズームを変えても突入時の拡大表示へ戻ってしまう)。
            # 復元した crop_rect から非ズーム基準の disp_info を作り直す。
            original_img_size = self.primary_param.get('original_img_size')
            crop_rect = params.get_crop_rect(self.primary_param)
            if original_img_size and crop_rect is not None:
                base_disp = core.convert_rect_to_info(
                    crop_rect,
                    config.get_preview_texture_side() / max(original_img_size),
                )
                params.set_disp_info(self.primary_param, base_disp)
                # ズーム継続中は、バイパス空間で見ていた内容位置を復元後ジオメトリ空間へ
                # 写像し、同じ内容を指す click アンカーへ引き直す。
                if self.is_zoomed and pre_disp is not None:
                    tcg_to = params.param_to_tcg_info(self.primary_param)
                    center = (pre_disp[0] + pre_disp[2] / 2, pre_disp[1] + pre_disp[3] / 2)
                    center = self._mask1_map_view_center(center, self._mask1_identity_tcg_info(), tcg_to)
                    self.click_x, self.click_y = self._mask1_view_anchor_click(base_disp, center)
            effects.reeffect_all(self.primary_effects, 0)
            if redraw and self._image_interaction_ready():
                self.start_draw_image_and_crop(self.imgset)

        def _preview_texture_center_pos(self):
            preview = self.ids.get('preview')
            if preview is None:
                return None
            tw, th = getattr(preview, 'texture_size', (0, 0))
            if tw <= 0 or th <= 0:
                return None
            return tw / 2.0, th / 2.0

        def _invalidate_mask2_overlay_for_view_change(self):
            mask_editor = self.ids.get('mask_editor2')
            if mask_editor is None or not self._is_mask2_enabled():
                return
            mask = mask_editor.get_active_mask()
            if mask is None:
                return
            invalidator = getattr(mask_editor, '_invalidate_mask_render_family', None)
            try:
                if callable(invalidator):
                    invalidator(mask)
                else:
                    mask.invalidate_render_cache()
            except Exception:
                logging.exception('failed to invalidate mask2 overlay for view change')

        def _zoom_preview_to_texture_pos(self, tex_pos):
            if tex_pos is None:
                return False
            self.is_zoomed = True
            self.click_x, self.click_y = tex_pos
            self.drag_center_start = None
            effects.reeffect_all(self.primary_effects, 1)
            self._invalidate_mask2_overlay_for_view_change()
            self.start_draw_image_and_crop(
                self.imgset,
                center_pos=self._preview_texture_pos_to_image_pos(tex_pos),
            )
            return True

        def _zoom_preview_from_keyboard(self):
            if self.is_zoomed:
                return self._reset_preview_zoom()
            if (not self._image_interaction_ready()
                    or self._is_image_geometry_mode()
                    or self.is_mask_mesh_editor_active()):
                return False
            tex_pos = self._preview_texture_pos_from_window_pos(KVWindow.mouse_pos)
            if tex_pos is None:
                tex_pos = self._preview_texture_center_pos()
            return self._zoom_preview_to_texture_pos(tex_pos)

        def _text_input_has_focus(self):
            return self._focused_text_input() is not None

        def _focused_text_input(self):
            stack = [self]
            try:
                stack.extend(KVWindow.children)
            except Exception:
                # children属性を持たないウィンドウ実装への防御（走査対象が減るだけ）
                pass
            seen = set()
            while stack:
                widget = stack.pop()
                if widget is None:
                    continue
                wid = id(widget)
                if wid in seen:
                    continue
                seen.add(wid)
                if isinstance(widget, KVTextInput) and getattr(widget, "focus", False):
                    return widget
                try:
                    stack.extend(widget.children)
                except Exception:
                    # children属性を持たないウィジェットへの防御（走査対象が減るだけ）
                    pass
            return None

        def _clear_text_input_focus(self):
            focused = self._focused_text_input()
            if focused is not None:
                focused.focus = False

        def _refresh_mask1_editors(self):
            try:
                for effect_name in ("inpaint", "patchmatch_inpaint"):
                    effect = self.primary_effects[0].get(effect_name)
                    editor = getattr(effect, "mask_editor", None) if effect is not None else None
                    if editor is not None:
                        editor.delay_update_canvas()
                # inpaint/fill 結果の bbox 枠も、現在の disp_info(ズーム/パン)に追従させて再描画する
                if self.inpaint_edit is not None:
                    self.inpaint_edit.redraw()
                if self.patchmatch_inpaint_edit is not None:
                    self.patchmatch_inpaint_edit.redraw()
            except Exception:
                logging.exception("failed to refresh mask1 editors")

        def _is_mask2_on(self):
            """Mask2 トグルが ON 状態か。マスク編集モードの判定軸。
            個別マスクが Active かどうかではなく、Mask2 パネル全体が有効か否かで判定する
            (ON 中ならマスク未選択でも『マスク Geometry モード』として扱う)。"""
            try:
                return self.ids['mask2'].state == 'down'
            except Exception:
                return False

        def _is_image_geometry_mode(self):
            """Geometry タブを開いている状態。
            Ge タブ中は画像/マスクどちらの Geometry 編集でもズーム操作を禁止する。
            Mesh CP の double tap / Shift+Reset と preview double-tap zoom が競合するため。"""
            try:
                return self.ids["effects"].current_tab.text == "Ge"
            except Exception:
                return False

        def _active_mask_consumes_double_tap(self, touch):
            """アクティブマスクがダブルタップを自分の動作で消費するかどうか。
            (例: PolylineMask の開放確定。これが True なら preview の zoom 切替を抑制する)"""
            try:
                mask = self.ids["mask_editor2"].get_active_mask()
            except Exception:
                return False
            if mask is None:
                return False
            consumer = getattr(mask, 'consumes_double_tap', None)
            if not callable(consumer):
                return False
            try:
                return bool(consumer(touch))
            except Exception:
                return False

        def _consume_geometry_double_tap(self, instance, touch):
            """preview_widget.on_touch_down に Python bind するフィルタ。
            Geometry 編集タブ (画像/マスク mesh) 中のダブルタップを、子ウィジェット
            (CropEditor / geometry editor) に伝播する前にここで消費 (return True) する。
            これにより Ge タブでのダブルクリックがクロップ枠を動かす不具合を防ぐ。
            戻り値 True が伝播を止めるのは Python bind だから (KV ルールは戻り値を無視する)。"""
            if not getattr(touch, 'is_double_tap', False):
                return False
            if not instance.collide_point(*touch.pos):
                return False
            if not self._image_interaction_ready():
                return False
            # マスクが自前でダブルタップを処理する場合 (Polyline 開放確定) は消費しない
            if self._active_mask_consumes_double_tap(touch):
                return False
            # 通常プレビュー (非 Geometry) のダブルタップは従来どおりズーム切替に使うので消費しない
            if not (self._is_image_geometry_mode() or self.is_mask_mesh_editor_active()):
                return False
            return True

        def on_image_touch_down(self, touch):
            if self.ids['preview_widget'].collide_point(*touch.pos):
                self._clear_text_input_focus()
                # 画像未確定の間は preview 上のジェスチャを受け付けない
                if not self._image_interaction_ready():
                    return False
                # ズーム操作: 画像 Geometry モード時のみ抑制 (マスク Geometry モード時は許可)
                # ただしアクティブマスクがダブルタップを消費 (= PolylineMask の確定) する場合も抑制
                if (touch.is_double_tap == True
                        and not self._is_image_geometry_mode()
                        and not self.is_mask_mesh_editor_active()
                        and not self._active_mask_consumes_double_tap(touch)):
                    next_zoomed = not self.is_zoomed
                    tex_pos = self._preview_texture_pos_or_none(touch) if next_zoomed else None
                    if next_zoomed and tex_pos is None:
                        return False
                    self._mask_zoom_sync_log(
                        "double_tap next_zoomed=%s touch=%s tex=%s current_disp=%s zoom_ratio=%.3f",
                        next_zoomed, touch.pos, tex_pos,
                        params.get_disp_info(self.primary_param), self.zoom_ratio,
                    )
                    if not next_zoomed:
                        self._reset_preview_zoom()
                        return False

                    self._zoom_preview_to_texture_pos(tex_pos)

                # ドラッグ操作
                elif self.is_zoomed == True:
                    # ドラッグ開始時の中心位置を計算して保存
                    self.drag_center_start = self._current_zoom_center_pos()
                    if self.drag_center_start is not None:
                        self._mask_zoom_sync_log(
                            "drag_start disp=%s center=%s touch=%s",
                            params.get_disp_info(self.primary_param), self.drag_center_start, touch.pos,
                        )

            return False

        def on_image_touch_move(self, touch):
            if self.collide_point(*touch.pos):
                if self.is_zoomed == True:
                    if self.is_press_space == True and self.drag_center_start is not None:
                        # 表示倍率を取得
                        disp_info = params.get_disp_info(self.primary_param)
                        scale = disp_info[4] * device.dpi_scale()
                        
                        # 画面上の移動量
                        diff_screen_x = touch.pos[0] - touch.opos[0]
                        diff_screen_y = touch.pos[1] - touch.opos[1]

                        # 画像上の移動量に変換
                        offset_x = -diff_screen_x / scale
                        offset_y = diff_screen_y / scale # KivyはY軸上向き、画像座標系の移動としては...

                        # 新しい中心位置の計算
                        new_cx = self.drag_center_start[0] + offset_x
                        new_cy = self.drag_center_start[1] + offset_y 

                        self._mask_zoom_sync_log(
                            "drag_move diff=(%.2f,%.2f) scale=%.4f start=%s new_center=(%.2f,%.2f) current_disp=%s",
                            diff_screen_x, diff_screen_y, scale,
                            self.drag_center_start, new_cx, new_cy,
                            params.get_disp_info(self.primary_param),
                        )
                        effects.reeffect_all(self.primary_effects, 1)
                        self.start_draw_image_and_crop(
                            self.imgset,
                            center_pos=(new_cx, new_cy),
                            fast_display=True,
                            skip_histogram=True,
                        )

            return False
                    
        def on_image_touch_up(self, touch):
            if self.is_zoomed == True:
                if self.drag_center_start is not None:
                    self.drag_center_start = None

            return False

        def on_select_press(self):
            self.set_preview_focus_mode(False)
            self.save_current_sidecar()
            device.FileChooser(title="Select Folder", mode="dir", filters=[("Jpeg Files", "*.jpg")], on_selection=self.handle_for_dir_selection).run()

        def on_sort_filter_press(self):
            dialog = SortFilterDialog(viewer=self.ids['viewer'])
            dialog.open()

        def on_export_press(self):
            self.save_current_sidecar()

            dialog = ExportDialog(callback=self.handle_export_dialog)
            dialog.open()

        def handle_export_dialog(self, preset):
            if self.export_in_progress:
                return
            # 保存先ファイルの存在チェック
            cards = self.ids['viewer'].get_selected_cards()
            isfile = False
            for x in cards:
                ex_path = self._make_export_path(x.file_path, preset)
                if os.path.isfile(ex_path):
                    isfile = True
                    break

            if isfile == True:
                dialog = ExportConfirmDialog(preset=preset, callback=self.handle_confirm_dialog)
                dialog.open()

            elif len(cards) > 0:
                self.handle_confirm_dialog('Overwrite', preset)            

        def handle_confirm_dialog(self, select, preset):
            if select in ['Overwrite', 'Rename']:
                cards = self.ids['viewer'].get_selected_cards()
                if not cards or self.export_in_progress:
                    return

                self._export_cancel_event.clear()
                n = len(cards)
                self.export_total = n
                self.export_done = 0
                self.export_in_progress = True

                rating_by_path = {
                    c.file_path: self._export_snapshot_rating(c.file_path) for c in cards
                }

                def _export_job():
                    exported_ok = []
                    failed_paths = []
                    try:
                        for i, x in enumerate(cards):
                            if self._export_cancel_event.is_set():
                                break
                            # C: 待機中にソースが消えた/リネームされたファイルは、その1枚だけ
                            # スキップして継続（バッチ全体を止めない）。
                            if not os.path.isfile(x.file_path):
                                logging.warning("export skipped (source missing): %s", x.file_path)
                                failed_paths.append(x.file_path)
                                done = i + 1
                                KVClock.schedule_once(
                                    lambda dt, d=done: setattr(self, "export_done", d), 0
                                )
                                continue
                            ok = False
                            try:
                                ex_path = self._make_export_path(x.file_path, preset)
                                if select == 'Rename':
                                    ex_path = self._find_not_duplicate_filename(ex_path)
                                if select == 'Overwrite':
                                    if os.path.isfile(ex_path): # あっても無くても'Overwrite'
                                        self.cache_system.delete_file(ex_path)
                                        os.remove(ex_path) # ほんとは消さなくても良さそうだけど、通知がおかしいので

                                resize_str = ""
                                if preset['size_mode'] == "Long Edge":
                                    _, _, width, height = core.get_exif_image_size(x.exif_data)
                                    if width >= height:
                                        resize_str = preset['size_value'] + "x"
                                    else:
                                        resize_str = "x" + preset['size_value']
                                if preset['size_mode'] == "Pixels": resize_str = preset['size_value']
                                if preset['size_mode'] == "Percentage": resize_str = preset['size_value'] + "%"

                                ex_r = rating_by_path.get(x.file_path, 0)
                                exfile = export.ExportFile(x.file_path, x.exif_data, export_rating=ex_r)
                                ok = exfile.write_to_file(
                                    ex_path,
                                    preset['quality'],
                                    resize_str,
                                    preset['sharpen']/100,
                                    preset['icc_profile'],
                                    preset['metadata'],
                                    preset['gps'],
                                    preset['dithering'],
                                    cancel_event=self._export_cancel_event,
                                )
                            except Exception:
                                logging.exception("export failed for %s", x.file_path)
                                ok = False
                            if ok:
                                exported_ok.append(ex_path)
                            else:
                                failed_paths.append(x.file_path)
                            done = i + 1
                            KVClock.schedule_once(
                                lambda dt, d=done: setattr(self, "export_done", d),
                                0,
                            )
                            # A: 1件の失敗でバッチ全体を止めない。中断はキャンセル時(上の break)のみ。
                    finally:
                        if failed_paths:
                            logging.warning(
                                "export finished with %d failure(s) (skipped/continued): %s",
                                len(failed_paths), failed_paths,
                            )
                        done_paths = list(exported_ok)
                        KVClock.schedule_once(
                            lambda dt, ep=done_paths: self._export_finish_ui(dt, ep), 0
                        )

                self._export_thread = threading.Thread(target=_export_job, daemon=True)
                self._export_thread.start()

        def _make_export_path(self, path, preset):
            dirname, basename = os.path.split(path)
            basename_with_out_ext, ext = os.path.splitext(basename)
            if len(preset['output_path']) > 0:
                if preset['output_path'] != '/':
                    ex_path = os.path.join(dirname, preset['output_path'], basename_with_out_ext) + preset['format']
                else:
                    ex_path = os.path.join(preset['output_path'], basename_with_out_ext) + preset['format']
            else:
                ex_path = os.path.join(dirname, basename_with_out_ext) + preset['format']
            return ex_path

        def _find_not_duplicate_filename(self, path):
            addnum = -1
            while os.path.isfile(path) == True:
                path_with_out_ext, ext = os.path.splitext(path)
                path_with_out_ext = re.split('-[0-9]+$', path_with_out_ext)
                path = path_with_out_ext[0] + str(addnum) + ext
                addnum -= 1

            return path

        #--------------------------------
        # パラメータの全消去

        def on_reset_press(self):
            # パラメータバックアップ
            temp = self.primary_param['color_temperature_reset']
            tint = self.primary_param['color_tint_reset']
            Y = self.primary_param['color_Y']

            # 全消去
            self.primary_param.clear()

            # 初期化パラメータ設定
            params.set_image_param(self.primary_param, self.imgset.img)
            params.set_temperature_to_param(self.primary_param, temp, tint, Y)

            # マスク関連全消去
            is_mask2_enabled = self._is_mask2_enabled()
            self._disable_mask2()
            self.ids['mask_editor2'].clear_mask()
            if is_mask2_enabled:
                self._enable_mask2()
            
            # クロップエディタ起動時はそれの初期化も行う
            self.primary_effects[0]['crop'].reset2_crop_editor(self.primary_param)
            self.primary_effects[0]['crop'].reset_crop_editor()
            self.apply_effects_lv(0, 'crop') # 描画を走らせる

            # これでファイルが消えるはず
            self.save_current_sidecar()

            # widget更新
            self.set2widget_all(self.primary_effects, self.primary_param)

        #--------------------------------
        # Mask2関連

        def _is_mask2_enabled(self):
            return self.ids['mask_editor2'].opacity == 1 and self.ids['mask_editor2'].disabled == False

        def _find_effect_tab(self, text):
            effects_panel = self.ids.get("effects")
            if effects_panel is None:
                return None
            for tab in getattr(effects_panel, "tab_list", []):
                if getattr(tab, "text", None) == text:
                    return tab
            return None

        def can_open_liquify_editor(self):
            if not self._is_mask2_enabled():
                return True
            editor = self.ids.get('mask_editor2')
            if editor is None:
                return False
            try:
                active = editor.get_active_mask()
            except Exception:
                active = None
            if active is None or not active.is_composit():
                return False
            if self.is_mask_mesh_editor_active():
                return False
            if self._has_initializing_mask():
                return False
            try:
                created = editor.get_created_mask()
                return not bool(created is not None and getattr(created, 'initializing', False))
            except Exception:
                return True

        def _is_li_tab_blocked_for_mask2(self):
            return False

        def _set_li_tab_disabled_for_mask2(self):
            li_tab = self.ids.get("tab_li") or self._find_effect_tab("Li")
            if li_tab is None:
                return False

            disabled = self._is_li_tab_blocked_for_mask2()
            li_tab.disabled = disabled
            if not self.can_open_liquify_editor():
                self._close_inactive_distortion_painters(None)
            # マスク状態変化(作成/選択)に追従して liquify/ghost を開閉・再ロード(active param で開き直す)
            self._sync_effect_editors()
            self._sync_liquify_mask_editor_input_lock()
            return False

        def _set_disabled_for_ids(self, id_names, disabled):
            for id_name in id_names:
                widget = self.ids.get(id_name)
                if widget is not None:
                    widget.disabled = disabled

        def update_mask2_options_enabled(self):
            self.update_load_dependent_panels_enabled()
            self._set_li_tab_disabled_for_mask2()
            editor = self.ids.get('mask_editor2')
            mask2_button = self.ids.get('mask2')
            mask2_panel = self.ids.get('mask2_panel')
            if editor is None or mask2_button is None or mask2_panel is None:
                return
            mask2_enabled = (
                mask2_button.state == 'down'
                and self.mask2_wait_full_load == False
                and editor.disabled == False
            )
            active_mask = editor.get_active_mask() if mask2_enabled else None
            created_mask = editor.get_created_mask() if mask2_enabled else None
            current_mask = created_mask if created_mask is not None else active_mask
            has_mask_context = mask2_enabled and current_mask is not None
            is_composit = bool(has_mask_context and current_mask.is_composit())
            class_name = current_mask.__class__.__name__ if current_mask is not None else ''
            is_freedraw = class_name == 'FreeDrawMask'
            # brush hardness は FreeDraw と Polyline で共有 (PolylineMask は線幅エッジの soft 制御に流用)
            has_brush_hardness = class_name in ('FreeDrawMask', 'PolylineMask')
            is_polyline = class_name == 'PolylineMask'
            is_face = class_name == 'FaceMask'
            mask_specific_enabled = has_mask_context and not is_composit

            mask2_panel.disabled = not has_mask_context
            self._set_disabled_for_ids(
                (
                    'switch_lens_modifier',
                    'checkbox_color_modification',
                    'checkbox_subpixel_distortion',
                    'checkbox_geometry_distortion',
                ),
                mask2_button.state == 'down',
            )
            self._set_disabled_for_ids(
                (
                    'switch_fringe_removal',
                    'switch_rca',
                    'slider_rca_purple_amount',
                    'slider_rca_green_amount',
                    'slider_rca_fringe_width',
                    'slider_rca_edge_threshold',
                ),
                mask2_button.state == 'down',
            )

            self._set_disabled_for_ids(
                (
                    'switch_mask2_draw_effects',
                    'spinner_mask2_blend_mode',
                    'slider_mask2_color_dodge',
                    'slider_mask2_color_burn',
                    'slider_mask2_mix_black',
                    'slider_mask2_mix_white',
                    'slider_mask2_skin_smooth_amount',
                    'slider_mask2_skin_smooth_radius_bias',
                ),
                not has_mask_context,
            )
            self._set_disabled_for_ids(
                (
                    'switch_mask2_settings',
                    'checkbox_mask2_invert',
                    'switch_mask2_depth',
                    'slider_mask2_depth_min',
                    'slider_mask2_depth_max',
                    'switch_mask2_hue',
                    'slider_mask2_hue_distance',
                    'slider_mask2_hue_range',
                    'switch_mask2_lum',
                    'slider_mask2_lum_distance',
                    'slider_mask2_lum_range',
                    'switch_mask2_sat',
                    'slider_mask2_sat_distance',
                    'slider_mask2_sat_range',
                    'switch_mask2_options',
                    'slider_mask2_blur',
                    'slider_mask2_depth_balance',
                    'slider_mask2_open_space',
                    'slider_mask2_close_space',
                    'switch_mask2_quick_select',
                    'spinner_mask2_edge_refine_mode',
                    'slider_mask2_edge_refine_radius',
                    'slider_mask2_edge_refine_strength',
                    'slider_mask2_edge_refine_bias',
                ),
                not mask_specific_enabled,
            )
            # Mask Geometry: Composit を含む has_mask_context のとき有効。
            # CompositMask は集約後のマスク画像に matrix を適用する。
            self._set_disabled_for_ids(
                (
                    'switch_mask_geometry',
                    'slider_mask_rotation',
                    'checkbox_mask_flip_h',
                    'checkbox_mask_flip_v',
                    'slider_mask_translation_x',
                    'slider_mask_translation_y',
                    'slider_mask_scale_x',
                    'slider_mask_scale_y',
                ),
                not has_mask_context,
            )
            # Mesh Edit: Composit が選択されていて、かつマスク作成中ではないとき有効。
            # 作成中マスクが存在する間に Mesh モードに入るのを禁止する。
            self._set_disabled_for_ids(
                ('btn_mask_mesh_edit',),
                (not has_mask_context) or self._has_initializing_mask(),
            )
            self._set_disabled_for_ids(
                (
                    'slider_mask2_freedraw_brush_size',
                    'slider_mask2_freedraw_brush_hardness',
                ),
                not (mask_specific_enabled and has_brush_hardness),
            )
            self._set_disabled_for_ids(
                (
                    'checkbox_mask2_polyline_fill',
                ),
                not (mask_specific_enabled and is_polyline),
            )
            self._set_disabled_for_ids(
                (
                    'switch_mask2_face',
                    'grid_mask2_face',
                    'checkbox_mask2_face_face',
                    'checkbox_mask2_face_brows',
                    'checkbox_mask2_face_eyes',
                    'checkbox_mask2_face_nose',
                    'checkbox_mask2_face_mouth',
                    'checkbox_mask2_face_lips',
                ),
                not (mask_specific_enabled and is_face),
            )
            self._set_disabled_for_ids(
                (
                    'checkbox_mask2_allow_under_zero',
                ),
                True,
            )

        def update_load_dependent_panels_enabled(self):
            # ファイル未選択時はプリセット／ヒストリ系パネルも無効化する。
            disabled = bool(self.mask2_wait_full_load) or not bool(self.image_loaded)
            for panel_name in ("preset_panel", "history_panel"):
                panel = getattr(self, panel_name, None)
                if panel is not None:
                    panel.disabled = disabled
            self._set_disabled_for_ids(
                (
                    "switch_ai_noise_reduction",
                    "chip_ai_noise_reduction",
                    "slider_ai_noise_reduction_intensity",
                ),
                disabled,
            )
            self.update_lens_simulator_depth_enabled()

        def _ai_depth_map_cached(self):
            """AI depth map が既にキャッシュ済みか(App 保有の ai_image_cache を直接 peek)。
            マスクの存在ではなく『実際に推論済みか』を見るので、作成途中(キャンセル可能)
            では有効にならない。"""
            cache = getattr(self, "ai_image_cache", None)
            if cache is None:
                return False
            try:
                from cores.mask2 import cache_keys, inference_runtime
                size = self.primary_param.get("original_img_size")
                if not size or len(size) < 2:
                    return False
                key = cache_keys.depth_cache_key(
                    tuple(size), inference_runtime.DEPTH_MAP_ALGORITHM_VERSION
                )
                return cache.peek_depth_map(key) is not None
            except Exception:
                return False

        def update_lens_simulator_depth_enabled(self):
            # LensSimulator の depth 連動コントロール(軸上CA/球面収差/フォーカス)は、
            # AI depth map が実際にキャッシュされている時だけ有効化する。coating / lateral CA /
            # 絞り(サンスター・形状ボケで depth 非依存に使う) は常時有効。
            disabled = (
                bool(self.mask2_wait_full_load)
                or not bool(self.image_loaded)
                or not self._ai_depth_map_cached()
            )
            self._set_disabled_for_ids(
                (
                    "slider_longitudinal_ca",
                    "slider_spherical_ca",
                    "slider_lens_focus_depth",
                    "label_lensblur_use_depthmap",
                    "switch_lensblur_use_depthmap",
                ),
                disabled,
            )

        def _enable_mask2(self):
            self._mount_mask_editor2_to_preview()
            self.ids['mask_editor2'].opacity = 1
            self.ids['mask_editor2'].disabled = False
            self._set_li_tab_disabled_for_mask2()
            self.update_preview_texture_size()
            self.ids['mask_editor2'].set_texture_size(config.get_config('preview_width'), config.get_config('preview_height'))
            self.ids['mask_editor2'].set_primary_param(self.primary_param, params.get_disp_info(self.primary_param))
            self.ids['mask_editor2'].restore_last_active_mask()
            self.ids['mask_editor2'].update()
            self.update_mask2_options_enabled()

        def _disable_mask2(self):
            primary_distortion = None
            try:
                primary_distortion = self.primary_effects[1].get('distortion')
            except Exception:
                primary_distortion = None
            self._close_inactive_distortion_painters(primary_distortion)
            self.ids['mask_editor2'].opacity = 0
            self.ids['mask_editor2'].disabled = True
            self.ids['mask_editor2'].set_active_mask(None)
            self.ids['mask_editor2'].end()
            self._unmount_mask_editor2_from_preview()
            self._set_li_tab_disabled_for_mask2()
            try:
                current_tab = self.ids["effects"].current_tab
                if getattr(current_tab, "text", None) == "Li":
                    self.apply_effects_lv(
                        1,
                        "distortion",
                        defer_draw=True,
                        overlay_reason="tab_sync",
                    )
            except Exception:
                logging.exception("failed to restore primary Liquify editor after Mask2 off")
            self.update_mask2_options_enabled()

        # vignette / grain は範囲マスク(Mask2)に未対応のため、Mask2 ON 中は UI を操作不可にする。
        _MASK_INCOMPATIBLE_EFFECT_IDS = (
            'switch_vignette',
            'slider_vignette_intensity',
            'slider_vignette_radius_percent',
            'slider_vignette_softness',
            'switch_grain',
            'slider_grain_amount',
            'slider_grain_size',
            'slider_grain_roughness',
            'slider_grain_shadow',
            'slider_grain_highlight',
            'slider_grain_color',
            'slider_grain_seed',
        )

        def on_mask2_press(self, value):
            if self.mask2_wait_full_load:
                self.ids['mask2'].state = 'normal'
                kvutils.find_widget(self, 'mask2_content_panel').disabled = True
                # Mesh Edit モード中なら強制終了
                self._force_close_mask_mesh_editor()
                self._disable_mask2()
                self._set_disabled_for_ids(self._MASK_INCOMPATIBLE_EFFECT_IDS, False)
                return
            if value == "down":
                self.set_preview_focus_mode(False)
                self._enable_mask2()
                self._set_disabled_for_ids(self._MASK_INCOMPATIBLE_EFFECT_IDS, True)
                kvutils.find_widget(self, 'mask2_content_panel').disabled = False
                # マスク Geometry モードへ遷移: Ge タブ上のクロップ枠/lens エディタを閉じる
                try:
                    if self.ids["effects"].current_tab.text == "Ge":
                        self.primary_effects[0]['geometry'].close_geometry_editor(self)
                        self.primary_effects[0]['crop'].sync_crop_editor_mode_from_widget(self, self.primary_param)
                except Exception:
                    # 失敗するとGeタブのエディタがMask2 ON中も開いたままになり得るため記録する
                    logging.exception("failed to close Ge tab editor on Mask2 on")
            else:
                kvutils.find_widget(self, 'mask2_content_panel').disabled = True
                # Mask2 OFF: Mesh Edit モード中なら強制終了 (Mesh Edit は Mask2 ON 前提)
                self._force_close_mask_mesh_editor()
                self._disable_mask2()
                self._set_disabled_for_ids(self._MASK_INCOMPATIBLE_EFFECT_IDS, False)
                # マスク Geometry モードから画像 Geometry モードへ抜けるとき、
                # Ge タブ上で拡大表示中ならリセットする (画像 Geometry はズーム禁止のため)。
                try:
                    if self.is_zoomed and self.ids["effects"].current_tab.text == "Ge":
                        self.is_zoomed = False
                        self.click_x, self.click_y = 0, 0
                        self.drag_center_start = None
                        self.update_preview_texture_size()
                        disp_info = core.convert_rect_to_info(
                            params.get_crop_rect(self.primary_param),
                            config.get_preview_texture_side() / max(self.primary_param['original_img_size']),
                        )
                        params.set_disp_info(self.primary_param, disp_info)
                        effects.reeffect_all(self.primary_effects, 1)
                        self.start_draw_image_and_crop(self.imgset)
                except Exception:
                    # ズーム解除が途中失敗すると表示状態が不整合になり得るため記録する
                    logging.exception("failed to reset zoom on Mask2 off")
                # 画像 Geometry モードへ復帰: Ge タブ上ならクロップ枠を再展開する
                try:
                    if self.ids["effects"].current_tab.text == "Ge":
                        self.apply_effects_lv(0, "crop")
                except Exception:
                    # 失敗するとGeタブ復帰時にクロップ枠が再展開されず操作不能になり得る
                    logging.exception("failed to reopen crop editor on Mask2 off")

        def set_mask2_hue_range(self, color_str):
            # イベント発火させる代入
            hmin = core.HLS_COLOR_SETTING[color_str]['center'] - core.HLS_COLOR_SETTING[color_str]['width'][0] - core.HLS_COLOR_SETTING[color_str]['fade_width'][0]
            hmax = core.HLS_COLOR_SETTING[color_str]['center'] + core.HLS_COLOR_SETTING[color_str]['width'][1] + core.HLS_COLOR_SETTING[color_str]['fade_width'][1]
            slider = self.ids['slider_mask2_hue_range'].ids['slider']
            slider.active_index = 0
            slider.values = [hmin, hmax]

        #--------------------------------

        def _enable_inpaint_edit(self):
            if self.inpaint_edit is None:
                self.update_preview_texture_size()
                self.inpaint_edit = widgets.bbox_viewer.BoundingBoxViewer(param=self.primary_param,
                                    size=config.get_preview_texture_size(),
                                    initial_view=params.get_disp_info(self.primary_param),
                                    on_delete=self._on_inpaint_edit)
                self._set_diff_list_to_inpaint_edit()
                self.ids['preview_widget'].add_widget(self.inpaint_edit)
                #print(f"Inpaint x:{self.inpaint_edit.x}, y:{self.inpaint_edit.y}")
                #print(f"Preview x:{self.ids['preview'].x}, y:{self.ids['preview'].y}")
                #print(f"Mask2 x:{self.ids['mask_editor2'].x}, y:{self.ids['mask_editor2'].y}")

        def _disable_inpaint_edit(self):
            if self.inpaint_edit is not None:
                self.ids['preview_widget'].remove_widget(self.inpaint_edit)
                del self.inpaint_edit
                self.inpaint_edit = None

        def _set_diff_list_to_inpaint_edit(self):
            if self.inpaint_edit is not None:
                boxes = []
                for inpaint_diff in self.primary_param.get('inpaint_diff_list', []):
                    boxes.append(inpaint_diff.disp_info)
                self.inpaint_edit.set_boxes(boxes)

        def _on_inpaint_edit(self, deleted_index, deleted_box):
            self.begin_history_effect_ctrl(0, 'inpaint')
            self.primary_param['inpaint_diff_list'].pop(deleted_index)
            self.end_history_effect_ctrl(0, 'inpaint')
            self.apply_effects_lv(0, 'inpaint')

        def on_inpaint_edit_press(self, value):
            if value == "down":
                self._enable_inpaint_edit()
            else:
                self._disable_inpaint_edit()

        #--------------------------------

        def _enable_patchmatch_inpaint_edit(self):
            if self.patchmatch_inpaint_edit is None:
                self.update_preview_texture_size()
                self.patchmatch_inpaint_edit = widgets.bbox_viewer.BoundingBoxViewer(param=self.primary_param,
                                    size=config.get_preview_texture_size(),
                                    initial_view=params.get_disp_info(self.primary_param),
                                    on_delete=self._on_patchmatch_inpaint_edit)
                self._set_diff_list_to_patchmatch_inpaint_edit()
                self.ids['preview_widget'].add_widget(self.patchmatch_inpaint_edit)
                #print(f"Inpaint x:{self.inpaint_edit.x}, y:{self.inpaint_edit.y}")
                #print(f"Preview x:{self.ids['preview'].x}, y:{self.ids['preview'].y}")
                #print(f"Mask2 x:{self.ids['mask_editor2'].x}, y:{self.ids['mask_editor2'].y}")

        def _disable_patchmatch_inpaint_edit(self):
            if self.patchmatch_inpaint_edit is not None:
                self.ids['preview_widget'].remove_widget(self.patchmatch_inpaint_edit)
                del self.patchmatch_inpaint_edit
                self.patchmatch_inpaint_edit = None

        def _set_diff_list_to_patchmatch_inpaint_edit(self):
            if self.patchmatch_inpaint_edit is not None:
                boxes = []
                for inpaint_diff in self.primary_param.get('patchmatch_inpaint_diff_list', []):
                    boxes.append(inpaint_diff.disp_info)
                self.patchmatch_inpaint_edit.set_boxes(boxes)

        def _on_patchmatch_inpaint_edit(self, deleted_index, deleted_box):
            self.begin_history_effect_ctrl(0, 'patchmatch_inpaint')
            self.primary_param['patchmatch_inpaint_diff_list'].pop(deleted_index)
            self.end_history_effect_ctrl(0, 'patchmatch_inpaint')
            self.apply_effects_lv(0, 'patchmatch_inpaint')

        def on_patchmatch_inpaint_edit_press(self, value):
            if value == "down":
                self._enable_patchmatch_inpaint_edit()
            else:
                self._disable_patchmatch_inpaint_edit()

        def _trigger_patchmatch_inpaint_predict(self):
            # The Erase chip's transient button state can't be read reliably (it is a
            # LongPressScaledButton), so set the one-shot predict flag durably here and
            # let the effect's make_diff consume it on the next pipeline pass.
            self.primary_param['patchmatch_inpaint_predict'] = True
            self.apply_effects_lv(0, 'patchmatch_inpaint')

        def _trigger_inpaint_predict(self):
            # Same as above for the AI (async) inpaint. The flag must persist across
            # pipeline passes until the async worker result returns, so it is a durable
            # param flag rather than a transient button-state read.
            self.primary_param['inpaint_predict'] = True
            self.apply_effects_lv(0, 'inpaint')

        #--------------------------------
        # Mask Mesh edit (Composit 単位の TPS 変形)。MeshWarpWidget をプレビュー上に
        # マウントし、active Composit の effects_param['mask_mesh_*'] と双方向同期する。

        def _get_active_mask_geom_composit(self):
            """Mask Geom 編集の対象 Composit を返す。未選択なら None。"""
            me = self.ids.get('mask_editor2')
            if me is None:
                return None
            active = me.get_active_mask()
            if active is None:
                return None
            if active.is_composit():
                return active
            return me.find_composit_mask(active)

        def _has_initializing_mask(self):
            """マスク作成中 (initializing=True) のマスクが存在するかを判定。
            Mesh トグルの disable 制御に使う。"""
            me = self.ids.get('mask_editor2')
            if me is None:
                return False
            try:
                for m in getattr(me, 'mask_list', []):
                    if getattr(m, 'initializing', False):
                        return True
                    if m.is_composit():
                        for child, _op in getattr(m, 'mask_list', []):
                            if getattr(child, 'initializing', False):
                                return True
            except Exception:
                return False
            return False

        def is_mask_mesh_editor_active(self):
            """Mesh 編集モード中かどうかの判定。MaskEditor2 の on_touch_* ガードから参照される。"""
            return self.mask_mesh_editor is not None

        def update_mask_mesh_button_enabled(self):
            """マスク作成中は Mesh トグルを disable に。状態変化時に呼ぶ。"""
            btn = self.ids.get('btn_mask_mesh_edit')
            if btn is None:
                return
            btn.disabled = self._has_initializing_mask()

        def _set_mask2_content_panel_disabled(self, disabled):
            """Mesh モード中はサイドの mask2_content_panel (マスクリスト+追加/削除ボタン)
            を無効化して、選択・追加・削除を全ブロックする。"""
            try:
                panel = kvutils.find_widget(self, 'mask2_content_panel')
            except Exception:
                panel = None
            if panel is not None:
                panel.disabled = bool(disabled)

        def _sync_mask_mesh_editor_view(self, mask_editor=None, texture_size=None):
            """Mask Mesh editor の view を MaskEditor2 の表示座標系に揃える。

            control_points は MeshWarpWidget 側の編集状態として保持するが、zoom / scroll /
            crop / matrix は MaskEditor2 を正にする。primary_param は Mask2 Geometry
            full-preview 中に最新 disp_info を持たない経路があるため、fallback にだけ使う。
            """
            if self.mask_mesh_editor is None:
                return
            if mask_editor is None:
                mask_editor = self.ids.get('mask_editor2')
            try:
                if mask_editor is not None and getattr(mask_editor, 'tcg_info', None) is not None:
                    if hasattr(self.mask_mesh_editor, 'set_view_context'):
                        self.mask_mesh_editor.set_view_context(mask_editor, image_only_matrix=True)
                    elif hasattr(self.mask_mesh_editor, 'set_tcg_info'):
                        self.mask_mesh_editor.set_tcg_info(mask_editor.tcg_info)
                    return

                if texture_size is not None and hasattr(self.mask_mesh_editor, 'set_texture_size'):
                    self.mask_mesh_editor.set_texture_size(texture_size)
                if hasattr(self.mask_mesh_editor, 'set_view_param'):
                    self.mask_mesh_editor.set_view_param(self.primary_param)
            except Exception:
                logging.exception("mask mesh editor viewport sync failed")

        def _enable_mask_mesh_editor(self):
            if self.mask_mesh_editor is not None:
                return
            # マスク作成途中、または対象 Composit が無いなら Mesh モードに入らない。
            # 1 箇所で UI 状態 (toggle button) を戻して早期 return する。
            composit = None
            if not self._has_initializing_mask():
                composit = self._get_active_mask_geom_composit()
            if composit is None:
                btn = self.ids.get('btn_mask_mesh_edit')
                if btn is not None:
                    btn.state = 'normal'
                return
            from widgets.distortion_correction import MeshWarpWidget
            texture_size = config.get_preview_texture_size()
            # Draw 系マスクの CP を消して画面を整理するため、active を Composit に切替
            me = self.ids['mask_editor2']
            try:
                if me.get_active_mask() is not composit:
                    me.set_active_mask(composit)
            except Exception:
                # active mask切替失敗はMesh Edit中の表示対象がずれた状態になり得るため記録する
                logging.exception("failed to set active mask before mesh editor open")
            # 画像 mesh editor (effects.py GeometryEffect._open_geometry_editor) と
            # 完全に同じ順序: pos_hint 設定 → add_widget → 最後に set_correction_params。
            # set_correction_params は内部で _redraw_mesh を呼ぶため、parent 未 attach の
            # 段階 (= widget.size=default(100,100)) で実行すると初回描画が画面外になる。
            # Mask Mesh editor は拡大/スクロール中の preview crop 上で操作するため、
            # attach 後に MaskEditor2.tcg_info (実表示と同期済み) を入れて viewport を合わせる。
            mw = MeshWarpWidget(
                texture_size,
                self.primary_param,
                force_square_disp_info=False,
                show_shift_reset_hint=True,
            )
            mw.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
            self.mask_mesh_editor = mw
            self._mask_mesh_target_composit = composit
            self.ids['preview_widget'].add_widget(mw)
            self._sync_mask_mesh_editor_view(mask_editor=me, texture_size=texture_size)
            try:
                me.clear_mask_geom_axes()
            except Exception:
                logging.exception("_enable_mask_mesh_editor: clear_mask_geom_axes failed")
            # 初期 CP の決定: mask_mesh_link_to_image=True なら画像 mesh の CP を表示、
            # False なら Composit 自前の CP を表示。
            linked = composit.effects_param.get('mask_mesh_link_to_image', True)
            if linked:
                init_size = self.primary_param.get('mesh_size', [4, 4])
                init_cps = self.primary_param.get('control_points', {})
            else:
                init_size = composit.effects_param.get('mask_mesh_size', [4, 4])
                init_cps = composit.effects_param.get('mask_mesh_control_points', {})
            mw.set_correction_params({
                'mesh_size': init_size,
                'control_points': init_cps,
            })
            mw.set_callback(self._on_mask_mesh_editor_callback)
            # サイドパネル経由のマスク選択/追加/削除をブロック
            self._set_mask2_content_panel_disabled(True)
            # Mesh Edit モード中は全マスクの CP を非表示にする
            try:
                me.refresh_mask_visibility(mesh_edit_active=True)
            except Exception:
                logging.exception("_enable_mask_mesh_editor: refresh_mask_visibility failed")

        def _disable_mask_mesh_editor(self):
            if self.mask_mesh_editor is None:
                return
            self.ids['preview_widget'].remove_widget(self.mask_mesh_editor)
            self.mask_mesh_editor = None
            self._mask_mesh_target_composit = None
            # サイドパネルのロック解除
            self._set_mask2_content_panel_disabled(False)
            # Mesh Edit モード解除: 通常の visibility に戻す (active mask ベース)
            try:
                me = self.ids.get('mask_editor2')
                if me is not None:
                    me.refresh_mask_visibility(mesh_edit_active=False)
            except Exception:
                logging.exception("_disable_mask_mesh_editor: refresh_mask_visibility failed")

        def _force_close_mask_mesh_editor(self):
            """Mask2 OFF など、Mesh Edit が成立しなくなる遷移で外部から強制終了する。
            まず toggle button の state を 'normal' に戻し (Kivy の on_state bind が
            _disable_mask_mesh_editor を間接的に走らせる)、そのフックが効かない
            ケースのために最後に直接 _disable_mask_mesh_editor を呼ぶ。"""
            btn = self.ids.get('btn_mask_mesh_edit')
            if btn is not None and btn.state == 'down':
                btn.state = 'normal'
            if self.mask_mesh_editor is not None:
                self._disable_mask_mesh_editor()

        def _on_mask_mesh_editor_callback(self, event, mesh_widget):
            """MeshWarpWidget の lifecycle callback。
            'apply' が両方の trigger (通常 Apply / Reset / Shift+Reset) なので、
            CP の中身と modifier から状態を判別する:
              - CP 非空: 通常 Apply → local モード (linked=False, 自前 CP を保存)
              - CP 空 + Shift 押下: Shift+Reset → local モード + 空 CP (画像も無視)
              - CP 空 + Shift なし: 通常 Reset → linked モードに戻す
            """
            if event != 'apply':
                return
            composit = self._mask_mesh_target_composit
            if composit is None:
                return
            out = mesh_widget.get_correction_params()
            mesh_size = list(out.get('mesh_size', [4, 4]))
            cps = dict(out.get('control_points', {}))
            composit.effects_param['mask_mesh_size'] = mesh_size

            if cps:
                # 通常 Apply: 自前 CP を保存して linked 解除
                composit.effects_param['mask_mesh_control_points'] = cps
                composit.effects_param['mask_mesh_link_to_image'] = False
            else:
                # CP 空 = Reset。Shift キー押下なら local-empty、それ以外は linked へ戻す
                from kivy.core.window import Window as _KVWindow
                is_shift_reset = 'shift' in (_KVWindow.modifiers or set())
                composit.effects_param['mask_mesh_control_points'] = {}
                composit.effects_param['mask_mesh_link_to_image'] = not is_shift_reset
                # 通常 Reset (linked モードに戻す) 時は、MeshWarpWidget の UI 表示も
                # 画像 mesh の CP に再同期する (= 効果と UI を一致させる)
                if not is_shift_reset:
                    primary_cps = self.primary_param.get('control_points', {}) or {}
                    primary_size = self.primary_param.get('mesh_size', [4, 4])
                    if primary_cps:
                        # 再 set すると bind 経由で _redraw_mesh が走るので UI も更新される
                        mesh_widget.set_correction_params({
                            'mesh_size': primary_size,
                            'control_points': primary_cps,
                        })
            # overlay / pipeline / cache invalidate は MaskEditor2 側の単一入口に寄せる。
            # Mask Mesh は source viewport を一時拡張するため、直接 update_mask() すると
            # AI 系 child mask の表示キャッシュ順に依存して空フレームが出やすい。
            try:
                me = self.ids.get('mask_editor2')
                if me is not None and hasattr(me, 'request_mask_render_update'):
                    me.request_mask_render_update(
                        composit,
                        reason="mask_mesh_apply",
                        refresh_visibility=False,
                        redraw_overlay=True,
                        redraw_pipeline=True,
                    )
                else:
                    composit.update_mask()
                    self.start_draw_image()
            except Exception:
                logging.exception("mask mesh editor: render update failed")

        def on_mask_mesh_edit_press(self, value):
            if value == "down":
                self._enable_mask_mesh_editor()
            else:
                self._disable_mask_mesh_editor()

        #--------------------------------

        def handle_for_dir_selection(self, selection):
            if selection is not None:
                # フォルダ切替で viewer.data は刷新されるが、MainWidget は直前の imgset を
                # 保持し続けてしまう。新フォルダから新たに選択されるまで編集 UI を無効化する。
                with threads.primary_param_lock:
                    self.save_current_sidecar()
                    effects.finalize_all(self.primary_effects, self.primary_param, self)
                    self.empty_image()  # image_loaded=False / imgset=None
                self._expected_file_path = None
                self._last_pmck_dict = None
                self._clear_exif_data()
                # フォルダ切替時は「何もロードしていない」状態に戻す。
                # 前回 on_select 直後にフォルダ切替が走った場合に loading=True が残るのを防ぐ。
                self.loading = False
                self._actively_loading = False
                self.update_mask2_options_enabled()
                config.set_config('import_path', selection[0].decode())

        #--------------------------------

        def on_lut_select_folder(self):
            device.FileChooser(title="Select LUT Folder", mode="dir", filters=[("CUBE Files", "*.cube")], on_selection=self.handle_for_lut).run()

        def handle_for_lut(self, selection):
            if selection is not None:
                path = selection[0].decode()
                config.set_config('lut_path', path)

        #--------------------------------

        def on_color_match_select_source(self):
            device.FileChooser(
                title="Select Color Match Source Image",
                mode="open",
                filters=[("Image Files", "*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.webp")],
                on_selection=self.handle_for_color_match_source,
            ).run()

        def handle_for_color_match_source(self, selection):
            if selection is None or len(selection) == 0:
                return
            path = selection[0]
            if isinstance(path, bytes):
                path = path.decode()
            try:
                import io as _io
                import pyvips
                from PIL import ImageCms
                from effect_backends import colour_functions_adapter as colour_functions
                with pyvips.Image.new_from_file(path) as vips_image:
                    long_side = max(vips_image.width, vips_image.height)
                    if long_side > 1024:
                        vips_image = vips_image.resize(1024.0 / long_side)
                    # imageset._load_rgb と同じ手順で ICC プロファイル名を取得
                    try:
                        icc_data = vips_image.get("icc-profile-data")
                        if icc_data is None:
                            src_icc_profile_name = "sRGB"
                        else:
                            profile = ImageCms.ImageCmsProfile(_io.BytesIO(icc_data))
                            src_icc_profile_name = ImageCms.getProfileDescription(profile).strip()
                    except Exception:
                        src_icc_profile_name = "sRGB"
                    arr = np.array(vips_image)
                if arr.ndim == 3 and arr.shape[2] > 3:
                    arr = arr[:, :, :3]
                arr = core.convert_to_float32(arr)
                if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                src_space = core.ICC_PROFILE_TO_COLOR_SPACE.get(src_icc_profile_name, 'sRGB')
                arr = colour_functions.RGB_to_RGB(
                    arr, src_space, 'ProPhoto RGB', config.get_config('cat'),
                    apply_cctf_decoding=True, apply_gamut_mapping=False,
                )
                arr = colour_functions.apply_RGB_gamut_mapping(arr).astype(np.float32)
                # MKL を知覚均等空間で走らせるため、保存時点で sRGB ガンマエンコード済みにしておく
                import cores.color as color
                arr = color.srgb_gamma_encode(arr).astype(np.float32)
                arr = np.ascontiguousarray(arr)
            except Exception as e:
                logging.warning(f"on_color_match_select_source failed to load {path}: {e}")
                return

            if self.imgset is None:
                return

            self.begin_history_effect_ctrl(0, 'color_match')
            self.primary_param['color_match_source_image'] = arr
            self.apply_effects_lv(0, 'color_match')
            self.end_history_effect_ctrl(0, 'color_match')
            self.save_current_sidecar()

        #--------------------------------

        def on_current_tab(self, current):
            self._clear_text_input_focus()
            self._cancel_mask1_mode(redraw=False)
            if getattr(current, "text", None) == "Li" and self._is_li_tab_blocked_for_mask2():
                if self._set_li_tab_disabled_for_mask2():
                    return

            # 描画中の操作 (Polyline の途中など) はタブ切替で確定させる。
            try:
                self.ids['mask_editor2'].commit_in_progress()
            except Exception:
                # 失敗すると描画途中の操作がコミットされずタブ切替後に不整合が残り得る
                logging.exception("on_current_tab: commit_in_progress failed")

            # Mask Mesh edit モード中にタブを切り替えたら必ず解除する。
            # (Mesh Edit は特定タブ上の編集 UI 前提なので、他タブへ移ったら成立しない)
            try:
                self._force_close_mask_mesh_editor()
            except Exception:
                logging.exception("on_current_tab: _force_close_mask_mesh_editor failed")

            if current.text == "Ge":
                # Ge タブ中は画像/マスクどちらの Geometry 編集でもズーム禁止。
                if self.is_zoomed:
                    self.is_zoomed = False
                    self.crop_image = None
                    self.click_x, self.click_y = 0, 0
                    self.drag_center_start = None
                # マスク Geometry モードに入った時は lens/four-points 系エディタも閉じる
                if self._is_mask2_on():
                    self.primary_effects[0]['geometry'].close_geometry_editor(self)
            else:
                self.primary_effects[0]['geometry'].close_geometry_editor(self)

            if self.update_preview_texture_size():
                self.crop_image = None

            try:
                self.ids['mask_editor2'].refresh_mask_geom_axes()
            except Exception:
                logging.exception("on_current_tab: refresh_mask_geom_axes failed")

            if self.imgset is not None:
                # apply_effects_lv(0, 'crop') 内の sync_crop_editor_mode_from_widget は
                # Mask2 ON 時はクロップエディタを閉じるよう CropEffect 側で対応済み。
                self.apply_effects_lv(1, "distortion", overlay_reason="tab_sync")
                self.apply_effects_lv(0, "geometry", overlay_reason="tab_sync")
                self.apply_effects_lv(0, "crop", overlay_reason="tab_sync")



        def set_lut_path(self, path):
            lut_values = ['None']
            effects.LUTEffect.file_pathes = { 'None': None, }

            file_list = os.listdir(path)
            file_list.sort()
            for file_name in file_list:
                file_path = os.path.join(path, file_name)
                if file_name.lower().endswith(('.cube')):
                    lut_values.append(file_name)
                    effects.LUTEffect.file_pathes[file_name] = file_path
            self.ids['lut_spinner'].values = lut_values

        def _set_lens_presets(self):
            presets = ['None']
            for _key, data in coating_adapter.presets().items():
                presets.append(data['name'])
            self.ids['spinner_coating_preset'].values = presets

        def _clear_exif_data(self):
            for exif_id in (
                'exif_file_name',
                'exif_file_size',
                'exif_create_date',
                'exif_image_size',
                'exif_iso_speed',
                'exif_aperture',
                'exif_shutter_speed',
                'exif_exposure_compensation',
                'exif_flash',
                'exif_white_balance',
                'exif_focal_length',
                'exif_exposure_program',
                'exif_make',
                'exif_model',
                'exif_lens_model',
                'exif_software',
            ):
                self.ids[exif_id].value = '-'

            row = self.ids.get("exif_rating_row", None)
            if row is not None:
                row.rating = 0

        def _set_exif_data(self, exif_data, file_path=None):
            fp = file_path or (self.imgset.file_path if self.imgset is not None else None)
            if exif_data is not None and fp and rating_utils.is_rgb_path(fp):
                rating_io.merge_xmp_star_tags_into_exif(fp, exif_data)
            self.ids['exif_file_name'].value = exif_data.get("FileName", "-")
            self.ids['exif_file_size'].value = exif_data.get("FileSize", "-")
            self.ids['exif_create_date'].value = exif_data.get("CreateDate", "-")
            _, _, width, height = core.get_exif_image_size_with_orientation(exif_data)
            self.ids['exif_image_size'].value = str(width) + "x" + str(height)
            self.ids['exif_iso_speed'].value = str(exif_data.get("ISO", "-"))
            self.ids['exif_aperture'].value = str(exif_data.get("ApertureValue", exif_data.get("Aperture", "-")))
            self.ids['exif_shutter_speed'].value = str(exif_data.get("ShutterSpeedValue", "-"))
            self.ids['exif_exposure_compensation'].value = str(exif_data.get("ExposureCompensation", "-"))
            self.ids['exif_flash'].value = exif_data.get("Flash", "-")
            self.ids['exif_white_balance'].value = exif_data.get("WhiteBalance", "-")
            self.ids['exif_focal_length'].value = exif_data.get("FocalLength", "-")
            self.ids['exif_exposure_program'].value = exif_data.get("PictureMode", "-")
            self.ids['exif_make'].value = exif_data.get("Make", "-")
            self.ids['exif_model'].value = exif_data.get("Model", "-")
            self.ids['exif_lens_model'].value = exif_data.get("LensModel", "-")
            self.ids['exif_software'].value = exif_data.get("Software", "-")
            #self.ids['exif_'].value = exif_data.get("", "-")
            if self.imgset and self.imgset.file_path and rating_utils.is_rgb_path(self.imgset.file_path):
                self._rgb_xmp_rating_had[self.imgset.file_path] = rating_utils.exif_had_xmp_rating_tag(exif_data)
            self._sync_exif_rating_row()

        def _sync_exif_rating_row(self):
            row = self.ids.get("exif_rating_row", None)
            if row is None or self.imgset is None or not self.imgset.file_path:
                if row is not None:
                    row.rating = 0
                return
            fp = self.imgset.file_path
            r = self.ids["viewer"].get_rating_for_path(fp)
            if r is None:
                r = int(
                    rating_utils.effective_rating_display(
                        fp, self.primary_param.get("exif_data", {}) or {}, self.primary_param
                    )
                )
            row.rating = int(r)

        def apply_exif_pane_rating_slot(self, slot: int):
            if not self.imgset or not self.imgset.file_path:
                return
            cur = int(self.ids["exif_rating_row"].rating or 0)
            new_r = rating_utils.new_rating_on_slot_click(cur, int(slot))
            self.apply_paths_rating([self.imgset.file_path], new_r)

        def apply_paths_rating(self, file_paths, new_r: int):
            new_r = int(new_r) if new_r is not None else 0
            for fp in file_paths:
                if not fp:
                    continue
                if rating_utils.is_raw_path(fp):
                    # RAW: 星は viewer の data のみ即時更新。pmck へは他パラメータと同様、
                    # ファイル切替・エクスポート前・手動/終了保存など save_current_sidecar 経由のみ。
                    self.ids["viewer"].set_rating_for_path(fp, new_r)
                    if self.imgset and self.imgset.file_path == fp:
                        with threads.primary_param_lock:
                            self.primary_param.pop("rating", None)
                else:
                    had = self._rgb_xmp_rating_had.get(fp, rating_utils.exif_had_xmp_rating_tag(
                        (self.primary_param.get("exif_data") or {}) if (self.imgset and self.imgset.file_path == fp) else (self._viewer_exif_for_path(fp) or {})))
                    try:
                        rating_io.write_rgb_file_xmp_rating(fp, new_r, had)
                    except Exception as e:
                        logging.exception("RGB rating")
                        rating_io.notify_write_error(str(e))
                        continue
                    exif = None
                    if fp in self.cache_system.cache:
                        exif = self.cache_system.cache[fp][1]
                        rating_io.update_exif_dict_after_rgb_write(exif, new_r, had)
                    with threads.primary_param_lock:
                        if self.imgset and self.imgset.file_path == fp and self.primary_param.get("exif_data"):
                            rating_io.update_exif_dict_after_rgb_write(self.primary_param["exif_data"], new_r, had)
                    if new_r > 0:
                        self._rgb_xmp_rating_had[fp] = True
                    elif new_r == 0 and had:
                        self._rgb_xmp_rating_had[fp] = True
                    else:
                        self._rgb_xmp_rating_had[fp] = False
                    if exif is not None:
                        self.ids["viewer"].set_exif_for_path(fp, exif)
                    self.ids["viewer"].set_rating_for_path(fp, new_r)
            self._sync_exif_rating_row()

        def _viewer_exif_for_path(self, file_path: str):
            return self.ids["viewer"].get_exif_for_path(file_path)

        def _export_snapshot_rating(self, file_path: str) -> int:
            """Export 用の 0～5。選択カードは常に viewer にある。"""
            return self._viewer_snapshot_rating(file_path)

        def request_export_cancel(self):
            self._export_cancel_event.set()

        def _export_finish_ui(self, dt=None, exported_ok=None):
            self.export_in_progress = False
            self.export_done = 0
            self.export_total = 0
            self._export_thread = None
            viewer = self.ids['viewer']
            # エクスポート直後: watchfiles の追加イベントより先に Viewer を明示同期する。
            viewer.refresh_exported_paths(exported_ok or [])
            if exported_ok:
                retry_paths = list(exported_ok)
                KVClock.schedule_once(
                    lambda _dt, pl=retry_paths: self._export_retry_viewer_exif(list(pl)), 0.35
                )
            fp = self.imgset.file_path if self.imgset is not None else None
            if fp:
                viewer.set_selection_silent(fp)
            else:
                viewer.clear_selection()

        def _export_retry_viewer_exif(self, paths):
            v = self.ids["viewer"]
            v.refresh_exported_paths(paths or [])

        def on_export_bar_press(self):
            if self.export_in_progress:
                self.request_export_cancel()
            else:
                self.on_export_press()

        def shutdown(self):
            #self.processor.stop()
            viewer = self.ids.get("viewer")
            if viewer and getattr(viewer, "watch_directory", None):
                preset_utils.cleanup_pmck_backup_files(viewer.watch_directory)
            for op in getattr(self.history, "operations", []):
                if getattr(op, "type", None) == "BatchPaste":
                    for item in getattr(op, "batch_items", []):
                        preset_utils.finalize_batch_item(item)
            if self.async_worker:
                self.async_worker.stop()
            if self.ai_job_manager:
                self.ai_job_manager.stop()
            if self.ai_sidecar_merge_queue:
                self.ai_sidecar_merge_queue.shutdown()
            
            if self.apply_thread is not None:
                t = self.apply_thread
                self.apply_thread = None
                self.draw_event.set()  # 待機中のスレッドを起こす
                t.join()

        def resize(self):
            self.apply_preview_focus_layout()
            self.sync_preview_widget_min_size()
            self._clamp_window_to_preview_minimum()
            preview_changed = self.update_preview_texture_size()
            if self._resize_debug_enabled():
                KVClock.schedule_once(lambda _dt: self._update_resize_debug_display(), 0.05)
            if self.imgset is not None and self.imgset.img is not None:
                h, w = self.imgset.img.shape[:2]
                self.preview_size = [kvutils.dpi_scale_width(w), kvutils.dpi_scale_height(h)]
            self.ids["transform_wrapper"].scale = device.dpi_scale()
            self.ids["transform_wrapper"].center = self.ids['preview_widget'].center
            self.refresh_preview_overlays()
            if preview_changed and self.imgset is not None and self.imgset.img is not None:
                self.start_draw_image_and_crop(self.imgset)
                
        def on_key_down(self, window, key, scancode, codepoint, modifier):
            #print(f"key:{key}, scancode:{scancode}, codepoint:{codepoint}, modifier:{modifier}")
            if processing_dialog.is_active():
                return True

            if self._is_force_memory_release_shortcut(key, codepoint, modifier):
                if not self._text_input_has_focus():
                    self.force_release_memory_caches(reason="manual_shortcut")
                    return True
                return False

            if (codepoint or "").lower() == 'm' or key == 109:
                if not self._text_input_has_focus():
                    self.ids['mask_editor2'].set_overlay_control_points_hidden(True)
                    return True
                return False

            if codepoint == '0' or key == 48:
                if not self._text_input_has_focus():
                    return self._zoom_preview_from_keyboard()
                return False

            if key == 32 or codepoint == ' ':
                self._clear_text_input_focus()
                if self.is_press_space == False:
                    self.sync_draw_image_and_crop(self.imgset)
                self.is_press_space = True
                return True

            if (key == 115 and ('ctrl' in modifier or 'meta' in modifier)):  # Sキー
                self.save_current_sidecar()
                return True

            if (key == 99 and ('ctrl' in modifier or 'meta' in modifier)):  # Cキー
                self.copy_effect_settings()
                return True

            if (key == 118 and ('ctrl' in modifier or 'meta' in modifier)):  # Vキー
                self.paste_effect_settings()
                return True

            if (key == 102 and ('ctrl' in modifier or 'meta' in modifier)):  # Fキー
                self.toggle_preview_focus_mode()
                return True
                                
            if (key == 122 and ('shift' not in modifier) and ('ctrl' in modifier or 'meta' in modifier)):  # Zキー
                self._undo()
                return True
                    
            if (key == 122 and ('shift' in modifier) and ('ctrl' in modifier or 'meta' in modifier)):  # shift-Zキー
                self._redo()
                return True

        def on_key_up(self, window, key, *args):
            if processing_dialog.is_active():
                return True

            if key == 109:
                self.ids['mask_editor2'].set_overlay_control_points_hidden(False)
                return True

            if key == 32:
                self.is_press_space = False
                self.sync_draw_image_and_crop(self.imgset)
                return True
        
    class MainApp(KVApp):
        def __init__(self, cache_system, **kwargs):
            super(MainApp, self).__init__(**kwargs)
            
            self.title = define.APPNAME
            
            self.cache_system = cache_system
            self._setup_window_handle = None

        def build(self):
            self.main_widget = MainWidget(self.cache_system)

            config.init_config(self.main_widget)
            config.load_config()
            self.main_widget.refresh_auto_display_color_gamut(reason="startup", redraw=False)
            
            # Start worker after config is loaded
            self.main_widget.async_worker.start()

            # window setup
            self._setup_window_handle = KVClock.schedule_interval(self._setup_window, 0.02)
            
            return self.main_widget

        def _setup_window(self, dt):
            if device.set_window_autosave(define.APPNAME, "PlatypusMainWindow"):
                KVClock.unschedule(self._setup_window_handle)
                self.on_window_resize(KVWindow, KVWindow.width, KVWindow.height)
        
        def on_start(self):
            self.main_widget.on_start()

            KVWindow.bind(on_resize=self.on_window_resize)
            self._bind_window_move_events()
            """
            display = device.get_current_display()
            KVWindow.size = (display["width"] * 0.9, display["height"] * 0.9)
            KVWindow.left = (display["width"] - display["width"] * 0.9) // 2
            KVWindow.top = (display["height"] - display["height"] * 0.9) // 2
            """
            _close_startup_splash()
            return super().on_start()

        def _bind_window_move_events(self):
            for kwargs in (
                {"on_move": self.on_window_move},
                {"left": self.on_window_move},
                {"top": self.on_window_move},
            ):
                try:
                    KVWindow.bind(**kwargs)
                except Exception:
                    # プロバイダによっては一部イベント名が未対応（他のbindで代替できる）
                    pass

        def on_stop(self):
            self.main_widget.request_export_cancel()
            t = self.main_widget._export_thread
            if t is not None and t.is_alive():
                t.join(timeout=3.0)
            self.main_widget._export_finish_ui()
            self.main_widget.save_current_sidecar()
            self.main_widget.shutdown()

        def on_window_resize(self, window, width, height):
            color_changed = self.main_widget.refresh_auto_display_color_gamut(reason="resize", redraw=False)
            kvutils.traverse_widget(self.root)
            self.main_widget.resize()
            if color_changed and self.main_widget.imgset is not None and self.main_widget.imgset.img is not None:
                self.main_widget.start_draw_image(fast_display=False)

        def on_window_move(self, *args):
            self.main_widget.refresh_auto_display_color_gamut(reason="move", redraw=True)

        def _widget_pos(self, root, pos):
            kvutils.traverse_widget(root)
            self.main_widget.resize()

if __name__ == '__main__':
    from kivy.factory import Factory
    from widgets.rating_row import RatingRow
    Factory.register("RatingRow", cls=RatingRow)
    # 処理中ダイアログ作成
    create_processing_dialog()

    # PILイメージプラグイン抑制
    pillow_init()
    
    cache_system = None
    try:
        # メインプロセスでマネージャーを作成
        cache_system = file_cache_system.FileCacheSystem(max_cache_size=100, max_concurrent_loads=20)

        # ここでシステムを使用...
        MainApp(cache_system).run()
    except Exception:
        logging.exception("Unhandled error in app main loop")
        raise
    finally:
        # 終了時にクリーンアップ
        _close_startup_splash()
        if cache_system is not None:
            try:
                cache_system.shutdown()
            except Exception:
                logging.exception("Failed to shut down cache system")
