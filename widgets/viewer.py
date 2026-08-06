import os
import threading
import concurrent.futures
import base64
import io
import numpy as np
import cv2
from watchfiles import watch
import time
import pyvips
from PIL import Image as PILImage, ImageOps as PILImageOps

from kivy.app import App as KVApp
from kivy.core.window import Window as KVWindow
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.uix.image import Image as KVImage
from kivy.uix.label import Label as KVLabel
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget as KVWidget
from kivy.graphics.texture import Texture as KVTexture
from kivy.graphics import Color as KVColor, Rectangle as KVRectangle
from kivy.metrics import dp as kvdp
from kivy.properties import Property as KVProperty, StringProperty as KVStringProperty, NumericProperty as KVNumericProperty, ObjectProperty as KVObjectProperty, BooleanProperty as KVBooleanProperty, ListProperty as KVListProperty
from kivy.clock import Clock as KVClock
from kivy.clock import mainthread as kvmainthread
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior

import logging

import libraw_enhanced as lre
import define
import cores.core as core
import utils.kvutils as kvutils
from utils import rating_utils
from utils import rating_io
from utils import viewer_query
from utils.exiftool_safe import safe_get_metadata
from widgets.draggable_widget import DraggableWidget
from widgets.plain_card import PlainCard
from widgets.rating_row import RatingRow
from utils.paths import rel
from utils import preset_utils
from utils.rename_detect import detect_rename_pair
from cores import pmck_store


_PMCK_ICON_REF_SIZE = 12
_PMCK_ICON_MARGIN_REF = 2
_THUMBNAIL_CARD_WIDTH_RATIO = 0.7
_THUMBNAIL_DISPLAY_MAX_SIDE = 240
# サイズ未確定でジオメトリ焼き付けを保留したときの再試行上限。これを超えたら諦める
# （カードが再アタッチされれば refresh_view_attrs が改めてスケジュールし直すため安全）。
_THUMBNAIL_GEOMETRY_MAX_RETRIES = 8
_HOVER_HINT_DELAY = 0.7
_EMBEDDED_PREVIEW_KEYS = ("PreviewImage", "JpgFromRaw", "PreviewTIFF", "OtherImage")
_EMBEDDED_THUMBNAIL_KEYS = ("ThumbnailImage", "ThumbnailTIFF")
# サムネイル生成（EXIF取得/デコード/デモザイク）を並列実行するワーカー数。
_THUMBNAIL_WORKER_COUNT = max(2, min(8, os.cpu_count() or 4))


def _first_value(data, *keys):
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _format_file_size(file_path):
    try:
        size = os.path.getsize(file_path)
    except OSError:
        return None
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.1f} {unit}"


def _format_image_size(exif_data):
    try:
        _, _, width, height = core.get_exif_image_size_with_orientation(exif_data or {})
    except Exception:
        width = _first_value(exif_data, "ImageWidth", "ExifImageWidth")
        height = _first_value(exif_data, "ImageHeight", "ExifImageHeight")
    try:
        width = int(str(width).split()[0])
        height = int(str(height).split()[0])
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    mp = width * height / 1_000_000.0
    return f"{width} x {height} · {mp:.1f} MP"


def _format_aperture(value):
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.lower().startswith("f/"):
        return text
    try:
        return f"f/{float(text):g}"
    except ValueError:
        return text


def _build_file_hint_text(file_path, exif_data):
    exif_data = exif_data or {}
    lines = [os.path.basename(file_path or "")]
    directory = os.path.dirname(file_path or "")
    if directory:
        lines.append(directory)

    date = _first_value(exif_data, "CreateDate", "DateCreated", "FileModifyDate", "ModifyDate")
    size_text = _format_image_size(exif_data)
    file_size = _format_file_size(file_path)
    if date:
        lines.extend(["", str(date)])
    metrics = " · ".join(part for part in (size_text, file_size) if part)
    if metrics:
        lines.append(metrics)

    make = _first_value(exif_data, "Make")
    model = _first_value(exif_data, "Model")
    camera = " ".join(str(part).strip() for part in (make, model) if part)
    lens = _first_value(exif_data, "LensModel", "Lens", "LensInfo")
    if camera or lens:
        lines.append("")
    if camera:
        lines.append(camera)
    if lens:
        lines.append(str(lens))

    exposure_parts = [
        f"ISO {exif_data.get('ISO')}" if exif_data.get("ISO") not in (None, "") else None,
        _first_value(exif_data, "ExposureTime", "ShutterSpeedValue"),
        _format_aperture(_first_value(exif_data, "Aperture", "FNumber", "ApertureValue")),
        _first_value(exif_data, "FocalLength"),
    ]
    exposure = " · ".join(str(part) for part in exposure_parts if part)
    if exposure:
        lines.append(exposure)
    return "\n".join(line for line in lines if line is not None)


class FileHint(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.opacity = 0
        self.label = KVLabel(
            text="",
            color=(0.96, 0.96, 0.96, 1),
            font_size="10pt",
            halign="left",
            valign="middle",
            size_hint=(None, None),
        )
        self.add_widget(self.label)
        with self.canvas.before:
            KVColor(0.04, 0.04, 0.04, 0.94)
            self._bg = KVRectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_bg, size=self._update_bg)

    def _update_bg(self, *_args):
        self._bg.pos = self.pos
        self._bg.size = self.size
        pad = kvdp(8)
        self.label.pos = (self.x + pad, self.y + pad)
        self.label.size = (max(1, self.width - pad * 2), max(1, self.height - pad * 2))
        self.label.text_size = self.label.size

    def show(self, text, mouse_pos):
        if not text:
            self.hide()
            return
        pad = kvdp(8)
        self.label.text = text
        self.label.text_size = (kvdp(360), None)
        self.label.texture_update()
        width = min(max(kvdp(220), self.label.texture_size[0] + pad * 2), kvdp(420))
        self.label.text_size = (width - pad * 2, None)
        self.label.texture_update()
        height = self.label.texture_size[1] + pad * 2
        self.size = (width, height)
        gap = kvdp(14)
        left_margin = kvdp(4)
        right_limit = max(left_margin, KVWindow.width - width - left_margin)
        x = mouse_pos[0] + gap
        if x > right_limit:
            x = mouse_pos[0] - width - gap
        y = mouse_pos[1] - height - kvdp(14)
        x = min(max(left_margin, x), right_limit)
        y = min(max(kvdp(4), y), max(kvdp(4), KVWindow.height - height - kvdp(4)))
        self.pos = (x, y)
        self.opacity = 1

    def hide(self):
        self.opacity = 0


class ThumbnailImage(KVWidget):
    texture = KVObjectProperty(None, allownone=True, force_dispatch=True)
    norm_image_size = KVListProperty([0, 0])
    max_display_side = KVNumericProperty(_THUMBNAIL_DISPLAY_MAX_SIDE)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            KVColor(1, 1, 1, 1)
            self._rect = KVRectangle(pos=self.pos, size=(0, 0))
        self.bind(
            pos=self._update_rect,
            size=self._update_rect,
            texture=self._update_rect,
            max_display_side=self._update_rect,
        )
        KVClock.schedule_once(lambda _dt: self._update_rect(), 0)

    def _update_rect(self, *_args):
        texture = self.texture
        if texture is None or self.width <= 0 or self.height <= 0:
            self._rect.texture = None
            self._rect.pos = self.pos
            self._rect.size = (0, 0)
            self.norm_image_size = [0, 0]
            return

        tex_w, tex_h = texture.size
        if tex_w <= 0 or tex_h <= 0:
            self._rect.texture = None
            self._rect.pos = self.pos
            self._rect.size = (0, 0)
            self.norm_image_size = [0, 0]
            return

        scale = min(1.0, self.width / tex_w, self.height / tex_h)
        if self.max_display_side > 0:
            scale = min(scale, self.max_display_side / max(tex_w, tex_h))
        draw_w = tex_w * scale
        draw_h = tex_h * scale
        self._rect.texture = texture
        self._rect.size = (draw_w, draw_h)
        self._rect.pos = (
            self.x + (self.width - draw_w) / 2,
            self.y + (self.height - draw_h) / 2,
        )
        self.norm_image_size = [draw_w, draw_h]


class ThumbnailCard(RecycleDataViewBehavior, PlainCard):
    file_path = KVStringProperty()
    thumb_source = KVObjectProperty(None, allownone=True, force_dispatch=True)
    rating = KVNumericProperty(0)
    pmck_exists = KVBooleanProperty(False)
    ai_job_state = KVStringProperty("")
    ai_job_progress = KVStringProperty("")
    load_pending = KVBooleanProperty(True)
    selected = KVBooleanProperty(False)
    ctx = KVObjectProperty(None)
    index = KVNumericProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._bound_layout_parent = None
        self._thumbnail_geometry_event = None
        self._thumbnail_geometry_late_event = None
        self._thumbnail_geometry_retries = 0
        self.exif_data = None
        self.orientation = 'vertical'
        self.size_hint = (None, 1)
        self.bg_color = [0.1, 0.1, 0.1, 1]
        self.radius = [5, 5, 5, 5]
        self.shadow_color = [0, 0, 0, 0.5]
        self.shadow_offset = [0, -3]
        self.shadow_spread = [2, 2]

        self.content_box = KVBoxLayout(orientation='vertical')
        self.content_box.ref_layout_padding = 8
        self._sync_content_box_layout_metrics()

        # サムネイル表示
        self.image_box = FloatLayout(size_hint_y=0.62)
        self.loading_spinner = KVImage(
            source=rel("assets", "spinner.gif"),
            anim_delay=0.02,
            size_hint=(1, 1),
            pos_hint={"x": 0, "y": 0},
            fit_mode="scale-down",
        )
        self.image_box.add_widget(self.loading_spinner)
        self.image = ThumbnailImage(
            max_display_side=_THUMBNAIL_DISPLAY_MAX_SIDE,
            size_hint=(1, 1),
            pos_hint={"x": 0, "y": 0},
        )
        self._configure_thumbnail_image_widget()
        self.image_box.add_widget(self.image)
        self.pmck_icon = KVImage(
            source=rel("assets", "pmck_indicator.png"),
            size_hint=(None, None),
            size=(
                kvutils.dpi_scale_width(_PMCK_ICON_REF_SIZE),
                kvutils.dpi_scale_height(_PMCK_ICON_REF_SIZE),
            ),
            allow_stretch=True,
            keep_ratio=True,
            mipmap=True,
            opacity=0,
        )
        self.pmck_icon.ref_width = _PMCK_ICON_REF_SIZE
        self.pmck_icon.ref_height = _PMCK_ICON_REF_SIZE
        self.image_box.add_widget(self.pmck_icon)
        self.ai_job_icon = KVImage(
            source=rel("assets", "spinner.gif"),
            anim_delay=0.03,
            size_hint=(1, 1),
            pos_hint={"x": 0, "y": 0},
            opacity=0,
        )
        self.image_box.add_widget(self.ai_job_icon)
        self.ai_job_progress_label = KVLabel(
            text="",
            bold=True,
            font_size='12sp',
            size_hint=(None, None),
            size=(kvdp(54), kvdp(22)),
            halign="right",
            valign="middle",
            color=(1, 1, 1, 0.95),
            outline_width=1,
            outline_color=(0, 0, 0, 0.9),
            opacity=0,
        )
        self.ai_job_progress_label.bind(size=self.ai_job_progress_label.setter("text_size"))
        self.image_box.add_widget(self.ai_job_progress_label)
        self.image_box.bind(pos=self._update_pmck_icon_layout, size=self._update_pmck_icon_layout)
        self.image.bind(
            pos=self._update_pmck_icon_layout,
            size=self._update_pmck_icon_layout,
            norm_image_size=self._update_pmck_icon_layout,
        )
        self.content_box.add_widget(self.image_box)

        # ファイル名ラベル
        self.label = KVLabel(
            text="",
            bold=True,
            font_size='9pt',
            size_hint_y=0.28,
            shorten=True,
            shorten_from="center",
            max_lines=1,
            halign="center",
            valign="middle",
        )
        self.label.bind(size=self.label.setter("text_size"))
        self.content_box.add_widget(self.label)

        self.rating_row = RatingRow(size_hint_y=0.1)
        self.content_box.add_widget(self.rating_row)

        self.add_widget(self.content_box)

        self.bind(file_path=self.update_filename)

    def _schedule_thumbnail_geometry_refresh(self):
        if self._thumbnail_geometry_event is None:
            self._thumbnail_geometry_event = KVClock.schedule_once(
                self._refresh_thumbnail_geometry, 0
            )
        if self._thumbnail_geometry_late_event is None:
            self._thumbnail_geometry_late_event = KVClock.schedule_once(
                self._refresh_thumbnail_geometry, 0.05
            )

    def _sync_content_box_layout_metrics(self):
        if hasattr(self, "content_box"):
            self.content_box.padding = kvutils.dpi_scale_width(self.content_box.ref_layout_padding)

    def _refresh_thumbnail_geometry(self, *_args):
        self._thumbnail_geometry_event = None
        self._thumbnail_geometry_late_event = None
        # RecycleView がカードサイズを確定する前／デタッチ中に do_layout() を走らせると、
        # 過渡的な誤サイズで image_box を焼き付けてしまう（export の refresh 連打で顕在化）。
        # 横並び RecycleView ではカード高さ＝親レイアウト高さなので、サイズが妥当に確定して
        # いる時だけ焼き付け、未確定なら焼き付けず後続フレームで再試行する。
        parent = self.parent
        if (
            parent is None
            or self.width <= 0
            or self.height <= 0
            or abs(self.height - parent.height) > 1
        ):
            self._thumbnail_geometry_retries += 1
            if self._thumbnail_geometry_retries <= _THUMBNAIL_GEOMETRY_MAX_RETRIES:
                self._schedule_thumbnail_geometry_refresh()
            return
        self._thumbnail_geometry_retries = 0
        if hasattr(self, "content_box"):
            self._sync_content_box_layout_metrics()
            self.do_layout()
            self.content_box.do_layout()
        if hasattr(self, "image_box") and hasattr(self.image_box, "do_layout"):
            self.image_box.do_layout()
        if hasattr(self, "image"):
            self.image._update_rect()
        self._update_pmck_icon_layout()

    def _configure_thumbnail_image_widget(self):
        for widget in (self.loading_spinner, self.image):
            widget.size_hint = (1, 1)
            widget.pos_hint = {"x": 0, "y": 0}
            if hasattr(widget, "allow_stretch"):
                widget.allow_stretch = False
            if hasattr(widget, "keep_ratio"):
                widget.keep_ratio = True
            if hasattr(widget, "fit_mode"):
                widget.fit_mode = "scale-down"
        self.image.max_display_side = _THUMBNAIL_DISPLAY_MAX_SIDE

    def on_parent(self, instance, value):
        if self._bound_layout_parent is not None:
            self._bound_layout_parent.unbind(height=self._set_width)
        self._bound_layout_parent = value
        if value is not None:
            value.bind(height=self._set_width)
        self._set_width()
        KVClock.schedule_once(lambda _dt: self._set_width(), 0)
    
    def on_size(self, instance, value):
        self._set_width()
    
    def _set_width(self, *_args):
        layout_height = self.parent.height if self.parent else self.height
        if layout_height <= 0:
            return
        width = layout_height * _THUMBNAIL_CARD_WIDTH_RATIO
        if abs(self.width - width) > 0.5:
            self.width = width

    def _update_pmck_icon_layout(self, *_args):
        if not hasattr(self, "pmck_icon"):
            return
        margin = kvutils.dpi_scale_width(_PMCK_ICON_MARGIN_REF)
        try:
            image_w, image_h = self.image.norm_image_size
        except (TypeError, ValueError):
            # 高頻度経路のためログ抑制（スクロール中のレイアウト更新で頻発しうる）
            image_w, image_h = self.image_box.size
        if image_w <= 0 or image_h <= 0:
            image_w, image_h = self.image_box.size
        image_x = self.image_box.x + max(0, (self.image_box.width - image_w) / 2)
        image_y = self.image_box.y + max(0, (self.image_box.height - image_h) / 2)
        self.pmck_icon.pos = (
            image_x + image_w - self.pmck_icon.width - margin,
            image_y + margin,
        )
        if hasattr(self, "ai_job_progress_label"):
            self.ai_job_progress_label.pos = (
                image_x + image_w - self.ai_job_progress_label.width - margin,
                image_y + margin + self.pmck_icon.height + kvutils.dpi_scale_height(2),
            )

    def update_filename(self, instance, value):
        if value:
            self.label.text = os.path.basename(value)

    def refresh_view_attrs(self, rv, index, data):
        """ Catch and handle the view changes """
        self.index = index
        # 新しいアイテムへ割り当て直されたので保留リトライ予算をリセット。
        self._thumbnail_geometry_retries = 0
        self._set_width()
        self._configure_thumbnail_image_widget()
        r = super(ThumbnailCard, self).refresh_view_attrs(rv, index, data)
        self.rating_row.rating = int(data.get("rating", 0) or 0)
        self.rating_row.card_index = index
        self.rating_row.ctx = data.get("ctx")
        self.rating_row.exif_pane = False
        self.exif_data = data.get("exif_data")
        self.pmck_exists = bool(data.get("pmck_exists", False))
        self.ai_job_state = str(data.get("ai_job_state") or "")
        self.ai_job_progress = str(data.get("ai_job_progress") or "")
        self.load_pending = bool(data.get("load_pending", False))
        self.pmck_icon.opacity = 1.0 if self.pmck_exists else 0.0
        self.ai_job_icon.opacity = 1.0 if self.ai_job_state in {"queued", "running"} else (0.65 if self.ai_job_state == "error" else 0.0)
        self.ai_job_progress_label.text = self.ai_job_progress
        self.ai_job_progress_label.opacity = 1.0 if self.ai_job_state == "running" and self.ai_job_progress else 0.0
        self._update_pmck_icon_layout()
        self._schedule_thumbnail_geometry_refresh()
        return r

    def refresh_view_layout(self, rv, index, layout, viewport):
        r = super().refresh_view_layout(rv, index, layout, viewport)
        self._set_width()
        self.image._update_rect()
        self._update_pmck_icon_layout()
        self._schedule_thumbnail_geometry_refresh()
        return r

    def on_selected(self, instance, value):
        self.bg_color = [0.32, 0.32, 0.32, 1] if value else [0.1, 0.1, 0.1, 1]

    def on_thumb_source(self, instance, thumb):
        self._configure_thumbnail_image_widget()
        if thumb is None:
            self.texture = None
            self.image.texture = None
            self.loading_spinner.opacity = 1.0
            return

        # float32(4byte/ch)ではなくubyte(1byte/ch)でGPUに転送する（転送量1/4、メインスレッド負荷軽減）。
        thumb_u8 = np.clip(thumb * 255.0, 0, 255).astype(np.uint8)
        self.texture = KVTexture.create(size=(thumb_u8.shape[1], thumb_u8.shape[0]), colorfmt='rgb', bufferfmt='ubyte')
        self.texture.flip_vertical()
        self.texture.blit_buffer(thumb_u8.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.image.texture = self.texture
        self.loading_spinner.opacity = 0.0
        self._update_pmck_icon_layout()
        self._schedule_thumbnail_geometry_refresh()

    def on_touch_down(self, touch):
        # 子（星スロット）へ先に伝播。ここで丸呑みするとタッチが RatingRow に届かない。
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        if self.load_pending and not touch.is_mouse_scrolling and touch.button == 'left':
            return True
        for child in reversed(self.children):
            if child.dispatch("on_touch_down", touch):
                return True
        if self.ctx:
            self.ctx.handle_selection(self.index, touch)
            return True
        return super().on_touch_down(touch)

class ViewerWidget(RecycleView, DraggableWidget):
    last_selected_index = KVNumericProperty(None, allownone=True)
    cols = KVNumericProperty(4)
    card_width = KVNumericProperty(112)
    thumb_width = KVNumericProperty(120*2)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 正データは _all_items（path キーで _items_by_key から O(1) 参照）。
        # self.data は view_settings（ソート/フィルタ）適用後の派生ビューで、
        # _rebuild_view() でのみ再構築する。item dict 自体は両者で共有される。
        self.data = []
        self._all_items = []
        self._items_by_key = {}
        self.view_settings = dict(viewer_query.DEFAULT_SETTINGS)
        # 選択はソート/フィルタで view インデックスが変わっても維持できるよう
        # norm path キー集合が正。selected_indices は view 上の派生インデックス。
        self.selected_paths = set()
        self.selected_indices = set()
        self._last_selected_path = None
        # Shift range のアンカーとは別に、preview に表示している画像を保持する。
        # 修飾キーによる複数選択では、この path が選択内に残る限り再ロードしない。
        self._current_path = None
        self.watch_directory = None
        self._watch_directory_lock = threading.Lock()
        self._watch_stop_event = None
        self._card_width_layout_event = None
        self._hover_recheck_event = None
        self._hover_hint_event = None
        self._hover_index = None
        self._file_hint = None
        self._file_hint_path = None
        self._coalesced_refresh_event = None

        threading.Thread(target=self._watchfiles_thread, daemon=True).start()
        KVWindow.bind(on_key_down=self.on_key_down)
        KVWindow.bind(mouse_pos=self._on_window_mouse_pos)
        self.bind(height=self._schedule_card_width_sync)
        self.bind(scroll_x=self._on_viewer_scroll_position)
        KVClock.schedule_once(lambda _dt: self._sync_card_width(), 0)

    def on_kv_post(self, base_widget):
        self._sync_card_width()

    def show_file_hint(self, file_path, exif_data, mouse_pos):
        if self._file_hint is None:
            self._file_hint = FileHint()
        if self._file_hint.parent is None:
            KVWindow.add_widget(self._file_hint)
        self._file_hint.show(_build_file_hint_text(file_path, exif_data), mouse_pos)
        self._file_hint_path = file_path

    def move_file_hint(self, mouse_pos):
        if self._file_hint is not None and self._file_hint.opacity > 0:
            self._file_hint.show(self._file_hint.label.text, mouse_pos)

    def hide_file_hint(self, file_path=None):
        if file_path is not None and self._file_hint_path != file_path:
            return
        if self._file_hint is not None:
            self._file_hint.hide()
        self._file_hint_path = None

    def _visible_thumbnail_cards(self):
        stack = list(self.children)
        while stack:
            widget = stack.pop()
            if isinstance(widget, ThumbnailCard):
                yield widget
            try:
                stack.extend(widget.children)
            except Exception:
                # 高頻度経路のためログ抑制（マウスホバー判定で頻繁に呼ばれる）
                pass

    def hover_index_at_window_pos(self, pos):
        card = self.hover_card_at_window_pos(pos)
        if card is None:
            return None
        try:
            index = int(getattr(card, "index", -1))
        except (TypeError, ValueError):
            # 高頻度経路のためログ抑制
            return None
        if 0 <= index < len(self.data):
            return index
        return None

    def hover_card_at_window_pos(self, pos):
        if not self.collide_point(*pos):
            return None

        local_pos = self.to_local(pos[0], pos[1])
        for card in self._visible_thumbnail_cards():
            if card.collide_point(*local_pos):
                return card
        return None

    def _cancel_hover_hint(self):
        if self._hover_hint_event is not None:
            self._hover_hint_event.cancel()
            self._hover_hint_event = None

    def _schedule_hover_hint(self, index, mouse_pos):
        self._cancel_hover_hint()
        if not (0 <= index < len(self.data)):
            return
        item = self.data[index]
        if item.get("load_pending") or not item.get("file_path"):
            return
        expected_path = item.get("file_path")
        self._hover_hint_event = KVClock.schedule_once(
            lambda _dt: self._show_hover_hint(index, expected_path, mouse_pos),
            _HOVER_HINT_DELAY,
        )

    def _show_hover_hint(self, index, expected_path, mouse_pos):
        self._hover_hint_event = None
        if self._hover_index != index or self.hover_index_at_window_pos(KVWindow.mouse_pos) != index:
            return
        if not (0 <= index < len(self.data)):
            return
        item = self.data[index]
        if item.get("file_path") != expected_path or item.get("load_pending"):
            return
        self.show_file_hint(item.get("file_path"), item.get("exif_data"), KVWindow.mouse_pos or mouse_pos)

    def _on_window_mouse_pos(self, _window, pos, force=False):
        if force:
            self._hover_index = None
        index = self.hover_index_at_window_pos(pos)
        if index is None:
            self._hover_index = None
            self._cancel_hover_hint()
            self.hide_file_hint()
            return
        if self._hover_index != index:
            self._hover_index = index
            self.hide_file_hint()
            self._schedule_hover_hint(index, pos)
            return
        if self._file_hint is not None and self._file_hint.opacity > 0:
            self.move_file_hint(pos)

    def _schedule_hover_recheck(self, delay=0.05):
        if self._hover_recheck_event is not None:
            self._hover_recheck_event.cancel()
        self._hover_recheck_event = KVClock.schedule_once(self._recheck_hover_cards, delay)

    def _recheck_hover_cards(self, _dt):
        self._hover_recheck_event = None
        self._on_window_mouse_pos(KVWindow, KVWindow.mouse_pos, force=True)

    def _on_viewer_scroll_position(self, *_args):
        self._hover_index = None
        self._cancel_hover_hint()
        self.hide_file_hint()
        self._schedule_hover_recheck()

    def _schedule_card_width_sync(self, *_args):
        if self._card_width_layout_event is None:
            self._card_width_layout_event = KVClock.schedule_once(
                lambda _dt: self._sync_card_width(), 0
            )

    def _sync_card_width(self):
        self._card_width_layout_event = None
        if self.height <= 0:
            return
        width = max(1, self.height * _THUMBNAIL_CARD_WIDTH_RATIO)
        if abs(self.card_width - width) > 0.5:
            self.card_width = width
        layout = getattr(self, "layout_manager", None)
        if layout is None and self.children:
            layout = self.children[0]
        if layout is not None and hasattr(layout, "default_size"):
            layout.default_size = (width, None)
        self.refresh_from_layout()

    def _watchfiles_thread(self):
        while True:
            with self._watch_directory_lock:
                watch_directory = self.watch_directory
                stop_event = self._watch_stop_event
            if watch_directory is None or stop_event is None:
                time.sleep(1)
                continue
            try:
                for changes in watch(watch_directory, stop_event=stop_event):
                    if stop_event.is_set():
                        break
                    self._dispatch_changes(changes)
            except Exception:
                # 失敗するとファイル変更検知が停止し一覧が古いままになり得るため記録する
                logging.exception("watchfiles: watch loop failed for %s", watch_directory)
            time.sleep(1)

    def _dispatch_changes(self, changes):
        action_type_map = {
            1: self._added_file,
            2: self._modified_file,
            3: self._deleted_file,
        }
        # リネーム相関(best-effort): watchfiles は rename を delete(old)+add(new) で通知する。
        # 同一バッチ・同一ディレクトリで対応画像が 1対1 のときだけ rename とみなし、
        # 通常ディスパッチ(一覧更新)の前に追従処理(.pmck移動 / AI-NRキャンセル / imgset remap)を行う。
        added = [p for a, p in changes if a == 1 and self.is_visible_image(p)]
        deleted = [p for a, p in changes if a == 3 and self.is_visible_image(p)]
        pair = detect_rename_pair(added, deleted)
        if pair is not None:
            self._follow_rename(pair[0], pair[1])

        for action, path in changes:
            fn = action_type_map.get(action)
            if fn:
                fn(path)

    def _follow_rename(self, old_path, new_path):
        """リネーム追従。ワーカースレッドから呼ばれる（imgset remap のみ UI スレッドへ）。"""
        try:
            # 1) .pmck サイドカーを新名へ追従（new 側に既存が無いときだけ。move は gateway 直列化済み）。
            old_pmck = old_path + pmck_store.PMCK_SUFFIX
            new_pmck = new_path + pmck_store.PMCK_SUFFIX
            if os.path.exists(old_pmck) and not os.path.exists(new_pmck):
                pmck_store.move_path_to_path(old_pmck, new_pmck)
        except Exception:
            logging.exception("rename follow: .pmck move failed: %s -> %s", old_path, new_path)

        app = KVApp.get_running_app()
        main_widget = getattr(app, "main_widget", None) if app else None

        # 2) AI-NR の in-flight を安全停止（旧 .pmck への誤書込・例外を防ぐ）。
        mgr = getattr(main_widget, "ai_job_manager", None) if main_widget else None
        if mgr is not None:
            try:
                if mgr.get_status_for_path(old_path) is not None:
                    mgr.cancel_path(old_path)
            except Exception:
                logging.exception("rename follow: AI-NR cancel failed for %s", old_path)

        # 3) 編集中ファイルなら imgset.file_path を新名へ（保存先ずれ防止）。
        if main_widget is not None:
            self._remap_imgset_if_current(main_widget, old_path, new_path)

    @kvmainthread
    def _remap_imgset_if_current(self, main_widget, old_path, new_path):
        imgset = getattr(main_widget, "imgset", None)
        if imgset is None:
            return
        if self._norm_path_key(getattr(imgset, "file_path", "") or "") != self._norm_path_key(old_path):
            return
        hook = getattr(main_widget, "remap_imgset_file_path", None)
        if callable(hook):
            hook(old_path, new_path)
        else:
            try:
                imgset.file_path = new_path
            except Exception:
                logging.exception("rename follow: imgset remap failed: %s -> %s", old_path, new_path)

    def _set_watch_directory(self, directory):
        directory = os.path.abspath(directory) if directory else None
        with self._watch_directory_lock:
            if self.watch_directory and self._norm_path_key(self.watch_directory) == self._norm_path_key(directory):
                return
            if self._watch_stop_event is not None:
                self._watch_stop_event.set()
            self.watch_directory = directory
            self._watch_stop_event = threading.Event() if directory else None

    def _new_image_item(self, file_path):
        return {
            'file_path': file_path,
            'thumb_source': None,
            'exif_data': None,
            'load_pending': True,
            'selected': False,
            'ctx': self,
            'rating': 0,
            'pmck_exists': os.path.exists(file_path + ".pmck"),
            'ai_job_state': "",
            'ai_job_progress': "",
        }

    def _item_for_path(self, file_path):
        return self._items_by_key.get(self._norm_path_key(file_path or ""))

    def _add_item_if_missing(self, file_path):
        key = self._norm_path_key(file_path)
        if key in self._items_by_key:
            return False
        item = self._new_image_item(file_path)
        self._all_items.append(item)
        self._items_by_key[key] = item
        return True

    def _remove_item(self, file_path):
        key = self._norm_path_key(file_path)
        item = self._items_by_key.pop(key, None)
        if item is None:
            return False
        try:
            self._all_items.remove(item)
        except ValueError:
            # _all_items に見つからないと一覧に古いアイテムが残り得るため記録する
            logging.exception("_remove_item: item not found in _all_items for %s", file_path)
        self.selected_paths.discard(key)
        if self._last_selected_path == key:
            self._last_selected_path = None
        return True

    def _rebuild_view(self):
        """_all_items から view_settings 適用済みの self.data を再構築し、
        選択状態（path 正）を view インデックスへ再マップする。"""
        for item in self._all_items:
            item['selected'] = (
                self._norm_path_key(item.get('file_path') or "") in self.selected_paths
            )
        view = viewer_query.build_view(self._all_items, self.view_settings)
        self.data = view
        self.cols = max(1, len(view))
        self.selected_indices = {i for i, item in enumerate(view) if item['selected']}
        last_index = None
        if self._last_selected_path is not None:
            last_index = next(
                (
                    i for i, item in enumerate(view)
                    if self._norm_path_key(item.get('file_path') or "") == self._last_selected_path
                ),
                None,
            )
        self.last_selected_index = last_index
        self.refresh_from_data()

    def set_view_settings(self, **changes):
        """ソート/フィルタ設定を更新し、変化があれば view を再構築する。"""
        changed = False
        for key, value in changes.items():
            if key not in self.view_settings:
                continue
            if self.view_settings[key] != value:
                self.view_settings[key] = value
                changed = True
        if changed:
            self._hover_index = None
            self._cancel_hover_hint()
            self.hide_file_hint()
            self._rebuild_view()
            self._schedule_hover_recheck()
        return changed

    def _is_in_current_watch_directory(self, file_path):
        if not self.watch_directory:
            return False
        try:
            file_dir = os.path.dirname(os.path.abspath(file_path))
            watch_dir = os.path.abspath(self.watch_directory)
        except OSError:
            return False
        return self._norm_path_key(file_dir) == self._norm_path_key(watch_dir)

    def refresh_exported_paths(self, file_paths):
        paths = []
        seen = set()
        for file_path in file_paths or []:
            if not file_path or not self.is_visible_image(file_path):
                continue
            if not self._is_in_current_watch_directory(file_path):
                continue
            key = self._norm_path_key(file_path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(file_path)

        if not paths:
            return False

        changed = False
        for file_path in paths:
            changed = self._add_item_if_missing(file_path) or changed

        load_paths = [fp for fp in paths if self._item_for_path(fp) is not None]

        if changed:
            self._rebuild_view()
        self.load_images(load_paths)
        return bool(load_paths)

    @kvmainthread
    def _added_file(self, file_path):
        pmck_image_path = self._image_path_for_pmck_sidecar(file_path)
        if pmck_image_path is not None:
            self.set_pmck_indicator_for_path(pmck_image_path, True)
            return
        if self.is_visible_image(file_path):
            self.refresh_exported_paths([file_path])

    @kvmainthread
    def _deleted_file(self, file_path):
        pmck_image_path = self._image_path_for_pmck_sidecar(file_path)
        if pmck_image_path is not None:
            self.set_pmck_indicator_for_path(pmck_image_path, False)
            return
        if not self._is_in_current_watch_directory(file_path):
            return
        if self._remove_item(file_path):
            self._rebuild_view()

    @kvmainthread
    def _modified_file(self, file_path):
        """
        エクスポート等で「先にファイル作成 → 後から exiftool で星」となると、
        追加 (watch) 時点では星が無い。追記後の modify でメタ＆星表示を再取得する。
        """
        pmck_image_path = self._image_path_for_pmck_sidecar(file_path)
        if pmck_image_path is not None:
            self.set_pmck_indicator_for_path(pmck_image_path, os.path.exists(file_path))
            return
        if not self.is_visible_image(file_path):
            return
        self.refresh_exported_paths([file_path])

    def set_path(self, directory):
        self._hover_index = None
        self._cancel_hover_hint()
        self.hide_file_hint()
        preset_utils.cleanup_pmck_backup_files(directory)
        self._all_items = []
        self._items_by_key = {}
        self.selected_paths = set()
        self.selected_indices = set()
        self._last_selected_path = None
        self._current_path = None
        self.last_selected_index = None
        self.data = []

        file_list = os.listdir(directory)
        file_list.sort()

        load_paths = []
        for file_name in file_list:
            if self.is_visible_image(file_name):
                file_path = os.path.join(directory, file_name)
                if self._add_item_if_missing(file_path):
                    load_paths.append(file_path)

        self._rebuild_view()
        self.load_images(load_paths)
        self._set_watch_directory(directory)

    def load_images(self, file_paths):
        file_paths = list(file_paths or [])
        if len(file_paths) > 0:
            self._set_load_pending(file_paths, True)
            threading.Thread(target=self.load_images_thread, args=(file_paths, 16), daemon=True).start()

    @kvmainthread
    def _set_load_pending(self, file_paths, pending):
        changed = False
        for file_path in file_paths:
            item = self._item_for_path(file_path)
            if item is not None:
                item['load_pending'] = bool(pending)
                changed = True
        if changed:
            self.refresh_from_data()

    @staticmethod
    def _norm_path_key(p: str) -> str:
        try:
            return os.path.normcase(os.path.abspath(p))
        except OSError:
            # 高頻度経路のためログ抑制（一覧再構築時に大量呼び出しされるパス正規化）
            return os.path.normcase(p or "")

    def _process_metadata_chunk(self, chunk, deferred_raw, deferred_lock):
        """1チャンク分の EXIF 取得＋軽量サムネイル生成。ワーカープールから並列実行される想定。"""
        try:
            # -a -G1 keeps duplicate Rating tags as group-qualified keys.
            # safe_get_metadata also adds short-name aliases for existing UI code.
            exif_data_list = safe_get_metadata(
                chunk,
                common_args=["-b", "-s", "-a", "-G1", "-x", "IFD1:PreviewTIFF", "-x", "SubIFD1:PreviewTIFF"],
            )
        except Exception:
            logging.exception("load_images_thread: EXIF取得失敗。スキップして続行 (chunk size=%d)", len(chunk))
            self._finish_failed_chunk(chunk)
            return

        for k, file_path in enumerate(chunk):
            # 既に一覧から消えた(削除/リネーム)ファイルは処理しない（best-effort）。
            if self._item_for_path(file_path) is None:
                continue
            exif_data = exif_data_list[k]
            try:
                # 1) EXIF/rating/pmck を先に反映（サムネイル生成を待たない）。
                #    rating/pmck のディスクI/O はワーカースレッドで済ませてUIスレッドを塞がない。
                ex0 = exif_data or {}
                if rating_utils.is_raw_path(file_path):
                    rating = rating_io.read_raw_pmck_rating_value(file_path)
                else:
                    rating = rating_utils.parse_exif_rating_value(ex0)
                pmck_exists = os.path.exists(file_path + ".pmck")
                self._apply_metadata(file_path, exif_data, rating, pmck_exists)
                # 2) 軽い経路で作れるサムネイルは即時生成・反映。RAWデモザイクは後回し。
                try:
                    thumb, deferred = self._build_thumbnail(file_path, exif_data, allow_demosaic=False)
                except Exception:
                    logging.exception("load_images_thread: thumbnail load failed for %s", file_path)
                    thumb, deferred = None, False
                if deferred:
                    with deferred_lock:
                        deferred_raw.append((file_path, exif_data))
                else:
                    self._apply_thumbnail(file_path, thumb)
            except Exception:
                logging.exception("load_images_thread: chunk item processing failed for %s", file_path)

    def _process_deferred_raw(self, file_path, exif_data):
        """後回しにした RAW デモザイクを1件処理。ワーカープールから並列実行される想定。"""
        # 待機中にファイルが消えた/リネームされたら、無駄なデモザイクと例外ログを避ける。
        if self._item_for_path(file_path) is None:
            return
        try:
            thumb, _ = self._build_thumbnail(file_path, exif_data, allow_demosaic=True)
        except Exception:
            logging.exception("load_images_thread: RAW demosaic thumbnail failed for %s", file_path)
            thumb = None
        self._apply_thumbnail(file_path, thumb)

    def load_images_thread(self, file_paths, chunk_size):
        file_path_list = list(file_paths)
        chunks = [file_path_list[i:i + chunk_size] for i in range(0, len(file_path_list), chunk_size)]

        # 埋め込みプレビューが無く、フルデモザイクが必要な RAW は後回しにする。
        # これらが軽いファイル(埋め込みプレビュー有り / 非RAW)の表示をブロックしないようにする。
        deferred_raw = []  # [(file_path, exif_data), ...]
        deferred_lock = threading.Lock()

        # チャンク単位（EXIF取得＋軽量サムネイル生成）を並列実行する。
        # 単一スレッド逐次処理だと exiftool のプロセス起動やデコード/デモザイクが
        # 1コア分の速度に律速されるため、ワーカープールでオーバーラップさせる。
        with concurrent.futures.ThreadPoolExecutor(max_workers=_THUMBNAIL_WORKER_COUNT) as pool:
            list(pool.map(
                lambda chunk: self._process_metadata_chunk(chunk, deferred_raw, deferred_lock),
                chunks,
            ))

            # 3) 後回しにした RAW デモザイクを並列処理（軽いファイルが出揃った後）
            list(pool.map(
                lambda item: self._process_deferred_raw(item[0], item[1]),
                deferred_raw,
            ))

    @kvmainthread
    def _apply_metadata(self, file_path, exif_data, rating, pmck_exists):
        """EXIF/rating/pmck を反映（load_pending は維持＝サムネイル待ち）。

        rating/pmck はワーカースレッドで算出済みの値を受け取り、UIスレッドでは代入のみ。
        """
        item = self._item_for_path(file_path)
        if item is None:
            return
        item['exif_data'] = exif_data
        item['rating'] = rating
        item['pmck_exists'] = pmck_exists
        self._schedule_coalesced_refresh()

    @kvmainthread
    def _apply_thumbnail(self, file_path, thumb):
        """サムネイルを反映し、load_pending を解除する。"""
        item = self._item_for_path(file_path)
        if item is None:
            return
        item['thumb_source'] = thumb
        item['load_pending'] = False
        self._schedule_coalesced_refresh()

    def _schedule_coalesced_refresh(self):
        """refresh_from_data をコアレスして過剰再描画を防ぐ（UIスレッドから呼ぶ前提）。"""
        if self._coalesced_refresh_event is None:
            self._coalesced_refresh_event = KVClock.schedule_once(self._do_coalesced_refresh, 0.05)

    def _do_coalesced_refresh(self, *_args):
        self._coalesced_refresh_event = None
        self._hover_index = None
        self._cancel_hover_hint()
        self.hide_file_hint()
        # メタデータ到着でソート/フィルタ結果が変わり得るため view ごと再構築する。
        self._rebuild_view()
        self._schedule_hover_recheck()

    @kvmainthread
    def _finish_failed_chunk(self, chunk):
        for file_path in chunk:
            item = self._item_for_path(file_path)
            if item is None:
                continue
            item['load_pending'] = False
            if item.get('exif_data') is None:
                item['exif_data'] = {}
        self._rebuild_view()

    def is_supported_image(self, file_name):
        return (file_name.lower().endswith(define.SUPPORTED_FORMATS_RGB)
                or file_name.lower().endswith(define.SUPPORTED_FORMATS_RAW)
                or file_name.lower().endswith(define.SUPPORTED_FORMATS_EXR))

    def is_visible_image(self, file_name):
        basename = os.path.basename(str(file_name or ""))
        return bool(basename) and not basename.startswith(".") and self.is_supported_image(file_name)

    def _image_path_for_pmck_sidecar(self, file_path):
        if not file_path:
            return None
        s = str(file_path)
        if not s.lower().endswith(".pmck"):
            return None
        image_path = s[:-5]
        return image_path if self.is_supported_image(image_path) else None

    def set_pmck_indicator_for_path(self, file_path, exists=None):
        if not file_path:
            return False
        want = self._norm_path_key(file_path)
        found = False
        pmck_exists = os.path.exists(file_path + ".pmck") if exists is None else bool(exists)
        d = self._item_for_path(file_path)
        if d is not None:
            d["pmck_exists"] = pmck_exists
            if rating_utils.is_raw_path(file_path):
                d["rating"] = rating_io.read_raw_pmck_rating_value(file_path)
            found = True
        if found:
            # pmck/rating は「編集済み」ソート/フィルタの入力なので view を再構築。
            self._rebuild_view()
            app = KVApp.get_running_app()
            main_widget = getattr(app, "main_widget", None) if app else None
            imgset = getattr(main_widget, "imgset", None) if main_widget else None
            if imgset and self._norm_path_key(getattr(imgset, "file_path", "") or "") == want:
                sync = getattr(main_widget, "_sync_exif_rating_row", None)
                if sync:
                    sync()
        return found

    @kvmainthread
    def set_ai_job_state_for_path(self, file_path, state, progress_text=""):
        if not file_path:
            return False
        found = False
        clean_state = state if state in {"queued", "running", "error"} else ""
        clean_progress = str(progress_text or "") if clean_state == "running" else ""
        d = self._item_for_path(file_path)
        if d is not None:
            if d.get("ai_job_state") == clean_state and d.get("ai_job_progress", "") == clean_progress:
                return True
            d["ai_job_state"] = clean_state
            d["ai_job_progress"] = clean_progress
            found = True
        if found:
            self.refresh_from_data()
        return found

    def _build_thumbnail(self, file_path, exif_data, allow_demosaic=True):
        """1ファイルのサムネイルを生成して返す。

        戻り値: (thumb_or_None, deferred)
        - deferred=True は「埋め込みプレビューが無く、フルデモザイクが必要な RAW を、
          allow_demosaic=False のため生成せず後回しにした」ことを示す（thumb は None）。
        """
        exif_data = exif_data or {}

        thumb, thumb_source_key = self._decode_embedded_thumbnail(exif_data)
        if thumb is None:
            if file_path.lower().endswith(define.SUPPORTED_FORMATS_RAW):
                if not allow_demosaic:
                    # 重いデモザイクは後回し（軽いファイルの表示をブロックしないため）。
                    return None, True
                with lre.imread(file_path) as raw:
                    # サムネイル用途なので half_size で高速デモザイク（約4倍）。
                    # half_size 非対応のビルドにはフォールバック。
                    try:
                        thumb = raw.postprocess(
                            demosaic_algorithm=lre.DemosaicAlgorithm.Linear,
                            output_bps=8, half_size=True,
                        )
                    except TypeError:
                        # half_size 非対応のlibrawビルドへのフォールバック（既知の互換シム）
                        thumb = raw.postprocess(
                            demosaic_algorithm=lre.DemosaicAlgorithm.Linear, output_bps=8,
                        )
            elif file_path.lower().endswith(define.SUPPORTED_FORMATS_EXR):
                # EXR は pyvips 非対応。OpenEXR で読み、表示用にトーンマップ済み float32[0,1] を得る。
                import cores.exr_io as exr_io
                thumb = exr_io.read_exr_thumbnail(file_path)
            else:
                with pyvips.Image.new_from_file(file_path) as vips_image:
                    thumb = np.array(vips_image)
                    if thumb.ndim == 3 and thumb.shape[2] > 3:
                        thumb = thumb[:, :, :3]
        # 先に（元の dtype のまま）縮小してから float32 化する。
        # 逆順だとフルサイズ画像を float32 化してから縮小することになり、
        # メモリ帯域・CPU コストが解像度に比例して無駄に増える。
        thumb_size = self._calc_resize_image((thumb.shape[1], thumb.shape[0]), self.thumb_width)
        thumb = cv2.resize(thumb, thumb_size)
        thumb = core.convert_to_float32(thumb)

        # Orientation
        orientation = exif_data.get('Orientation')
        if orientation is not None and self._should_apply_parent_orientation(thumb_source_key):
            if orientation == 'Rotate 180':
                thumb = cv2.rotate(thumb, cv2.ROTATE_180)
            elif orientation == 'Rotate 270 CW':
                thumb = cv2.rotate(thumb, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif orientation == 'Rotate 90 CW':
                thumb = cv2.rotate(thumb, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 'Mirror horizontal':
                thumb = cv2.flip(thumb, 1)
            elif orientation == 'Mirror vertical':
                thumb = cv2.flip(thumb, 0)
            elif orientation == 'Mirror horizontal and rotate 270 CW':
                thumb = cv2.flip(thumb, 1)
                thumb = cv2.rotate(thumb, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif orientation == 'Mirror horizontal and rotate 90 CW':
                thumb = cv2.flip(thumb, 1)
                thumb = cv2.rotate(thumb, cv2.ROTATE_90_CLOCKWISE)

        return thumb, False

    def _should_apply_parent_orientation(self, embedded_key):
        return embedded_key not in _EMBEDDED_PREVIEW_KEYS

    def _decode_embedded_bytes(self, encoded):
        if isinstance(encoded, str) and encoded.startswith("base64:"):
            encoded = encoded[7:]
        elif isinstance(encoded, bytes) and encoded.startswith(b"base64:"):
            encoded = encoded[7:]
        return base64.b64decode(encoded)

    def _decode_embedded_preview(self, encoded):
        data = self._decode_embedded_bytes(encoded)
        with PILImage.open(io.BytesIO(data)) as img:
            img = PILImageOps.exif_transpose(img)
            img = img.convert("RGB")
            return np.array(img)

    def _decode_embedded_thumbnail_image(self, encoded):
        data = self._decode_embedded_bytes(encoded)
        image = np.frombuffer(data, dtype=np.uint8)
        thumb = cv2.imdecode(image, 1)
        if thumb is None:
            return None
        if thumb.ndim == 2:
            return cv2.cvtColor(thumb, cv2.COLOR_GRAY2RGB)
        if thumb.shape[2] == 4:
            return cv2.cvtColor(thumb, cv2.COLOR_BGRA2RGB)
        if thumb.shape[2] > 4:
            return cv2.cvtColor(thumb[:, :, :3], cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)

    def _decode_embedded_thumbnail(self, exif_data):
        for key in _EMBEDDED_PREVIEW_KEYS:
            encoded = exif_data.get(key, None)
            if not encoded:
                continue
            try:
                return self._decode_embedded_preview(encoded), key
            except Exception:
                # このキーのプレビューが壊れている等の場合は次候補キーへフォールバック
                continue

        for key in _EMBEDDED_THUMBNAIL_KEYS:
            encoded = exif_data.get(key, None)
            if not encoded:
                continue
            try:
                thumb = self._decode_embedded_thumbnail_image(encoded)
            except Exception:
                # このキーのサムネイルが壊れている等の場合は次候補キーへフォールバック
                continue
            if thumb is None:
                continue
            return thumb, key
        return None, None

    def handle_selection(self, index, touch):
        if not self._is_item_ready(index):
            return

        if not touch.is_mouse_scrolling and touch.button == 'left':
            current_index = self._current_view_index()
            already_single_selected = (
                index in self.selected_indices
                and len(self.selected_indices) == 1
            )
            if (
                'shift' in KVWindow.modifiers
                and self.last_selected_index is not None
                and index is not None
            ):
                anchor = self.last_selected_index
                if not( 'ctrl' in KVWindow.modifiers or 'meta' in KVWindow.modifiers ):
                    self.clear_selection(notify=False)
                
                start = min(anchor, index)
                end = max(anchor, index)
                
                for i in range(start, end + 1):
                    self.select_at(i)
                # clear_selection() で消える Shift range のアンカーを維持する。
                self._set_last_selected(anchor)
                    
            else:
                if 'ctrl' in KVWindow.modifiers or 'meta' in KVWindow.modifiers:
                    self.toggle_at(index)
                else:
                    if already_single_selected:
                        self._set_last_selected(index)
                        return
                    self.clear_selection(notify=False)
                    self.select_at(index)

                self._set_last_selected(index)

            # 複数選択にカレントが残っていれば preview は変えない。外れた時だけ、
            # 元のカレント位置に最も近い選択項目へ移す。初回選択ではクリック位置を使う。
            reference_index = current_index if current_index is not None else index
            self._reconcile_current_selection(reference_index)

    def _current_view_index(self):
        if self._current_path is None:
            return None
        return next(
            (
                i for i, item in enumerate(self.data)
                if self._norm_path_key(item.get('file_path') or "") == self._current_path
            ),
            None,
        )

    def _reconcile_current_selection(self, reference_index=None):
        if self._current_path is not None and self._current_path in self.selected_paths:
            return

        next_index = viewer_query.nearest_selected_index(
            self.selected_indices,
            reference_index,
        )
        if next_index is None:
            if self._current_path is not None:
                self.notify_selection_change(None)
            return
        self.notify_selection_change(next_index)

    def notify_selection_change(self, index):
        if index is not None and not self._is_item_ready(index):
            return
        selected_data = self.data[index] if index is not None else None
        self._current_path = (
            self._norm_path_key(selected_data.get('file_path') or "")
            if selected_data is not None
            else None
        )
        app = KVApp.get_running_app()
        if app and hasattr(app, 'main_widget'):
            if selected_data is None:
                app.main_widget.on_select(None)
                return

            class MockCard:
                def __init__(self, d):
                    self.file_path = d['file_path']
                    self.exif_data = d['exif_data']

            app.main_widget.on_select(MockCard(selected_data))

    def _set_last_selected(self, index):
        self.last_selected_index = index
        if index is not None and 0 <= index < len(self.data):
            self._last_selected_path = self._norm_path_key(self.data[index].get('file_path') or "")
        else:
            self._last_selected_path = None

    def select_at(self, index):
        if self._is_item_ready(index):
            self.data[index]['selected'] = True
            self.selected_indices.add(index)
            self.selected_paths.add(self._norm_path_key(self.data[index].get('file_path') or ""))
            self.refresh_from_data()

    def toggle_at(self, index):
        if self._is_item_ready(index):
            val = not self.data[index]['selected']
            self.data[index]['selected'] = val
            key = self._norm_path_key(self.data[index].get('file_path') or "")
            if val:
                self.selected_indices.add(index)
                self.selected_paths.add(key)
            else:
                self.selected_indices.discard(index)
                self.selected_paths.discard(key)
            self.refresh_from_data()

    def clear_selection(self, *, notify=True):
        current_index = self._current_view_index()
        for d in self._all_items:
            d['selected'] = False
        self.selected_paths.clear()
        self.selected_indices.clear()
        self._set_last_selected(None)
        self.refresh_from_data()
        if notify:
            self._reconcile_current_selection(current_index)

    def refresh_exif_for_exported_path(self, file_path: str) -> bool:
        """
        エクスポート直後: watch より前でも新規カードを追加し、メタ＆星を取り直す。
        """
        return self.refresh_exported_paths([file_path])

    def _is_item_ready(self, index):
        return (
            index is not None
            and 0 <= index < len(self.data)
            and not bool(self.data[index].get("load_pending", False))
            and self.data[index].get("exif_data") is not None
        )

    def set_selection_silent(self, file_path):
        """サムネの選択表示だけを合わせる。on_select（画像の再ロード）は呼ばない。"""
        if not self._all_items or file_path is None:
            self.clear_selection(notify=False)
            self._current_path = None
            return
        key = self._norm_path_key(file_path)
        if key in self._items_by_key:
            self.selected_paths = {key}
            self._last_selected_path = key
            self._current_path = key
        else:
            self.selected_paths = set()
            self._last_selected_path = None
            self._current_path = None
        self._rebuild_view()

    def get_selected_cards(self):
        res = []
        class MockCard:
             def __init__(self, d):
                 self.file_path = d['file_path']
                 self.exif_data = d['exif_data']
                 self.thumb_source = d['thumb_source']
        
        for idx in self.selected_indices:
            if idx < len(self.data):
                res.append(MockCard(self.data[idx]))
        return res

    def set_rating_for_path(self, file_path, rating_value: int):
        d = self._item_for_path(file_path)
        if d is not None:
            d["rating"] = int(rating_value)
        # rating はソート/フィルタの入力なので view を再構築。
        self._rebuild_view()

    def get_rating_for_path(self, file_path):
        """rating を返す。Viewer が知らないパスは None（呼び出し側でフォールバック）。"""
        d = self._item_for_path(file_path)
        if d is None:
            return None
        return int(d.get("rating", 0) or 0)

    def get_exif_for_path(self, file_path):
        d = self._item_for_path(file_path)
        return None if d is None else d.get("exif_data")

    def set_exif_for_path(self, file_path, exif_data):
        d = self._item_for_path(file_path)
        if d is None:
            return False
        d["exif_data"] = exif_data
        return True

    def get_card(self, file_path):
        d = self._item_for_path(file_path)
        if d is not None:
            class MockCard:
                 def __init__(self, d):
                     self.file_path = d['file_path']
                     self.exif_data = d['exif_data']
            return MockCard(d)
        return None

    def set_cache_system(self, cache_system):
        self.cache_system = cache_system
        self.bind(scroll_x=self._request_current_view_cards)

    @kvmainthread
    def _request_current_view_cards(self, instance, value):
        pass

    def get_drag_files(self):
        file_paths = []
        for card in self.get_selected_cards():
            if card.thumb_source is not None:
                file_paths.append((card.file_path, (card.thumb_source * 255).astype(np.uint8)))
        return file_paths

    def _calc_resize_image(self, original_size, max_length):
        width, height = original_size
        width = max(1, int(width))
        height = max(1, int(height))
        scale_factor = min(1.0, max_length / max(width, height))
        return (
            max(1, int(round(width * scale_factor))),
            max(1, int(round(height * scale_factor))),
        )

    def on_scroll_start(self, touch, check_children=True):
        # マウスホイールの縦スクロールを横スクロールに変換する
        # touch.buttonを一時的に書き換え、super()呼び出し後に元に戻す
        # （touchオブジェクトは共有のため、他のウィジェットへの副作用を防ぐ）
        if touch.is_mouse_scrolling:
            self._hover_index = None
            self._cancel_hover_hint()
            self.hide_file_hint()
            original_button = touch.button
            if touch.button == 'scrolldown':
                touch.button = 'scrollright'
            elif touch.button == 'scrollup':
                touch.button = 'scrollleft'
            result = super().on_scroll_start(touch, check_children)
            touch.button = original_button  # 他のウィジェットのために元の値に戻す
            self._schedule_hover_recheck()
            return result
        return super().on_scroll_start(touch, check_children)

    def on_touch_move(self, touch):
        if (
            not touch.is_mouse_scrolling
            and self.collide_point(*touch.pos)
            and not self.dragging
            and self.get_drag_files()
        ):
            self._hover_index = None
            self._cancel_hover_hint()
            self.hide_file_hint()
            self.dragging = True
            self.start_drag(touch)
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        self.dragging = False
        return super().on_touch_up(touch)

    def on_key_down(self, window, key, scancode, codepoint, modifier):
        if (key == 97 and ('ctrl' in modifier or 'meta' in modifier)):  # A
            current_index = self._current_view_index()
            self.clear_selection(notify=False)
            for i in range(len(self.data)):
                self.select_at(i)
            self._reconcile_current_selection(current_index)
            return True

    def on_rating_slot(self, index, slot: int):
        if not self._is_item_ready(index):
            return
        cur = int(self.data[index].get("rating", 0) or 0)
        new_r = rating_utils.new_rating_on_slot_click(cur, slot)
        if len(self.selected_indices) > 1 and index in self.selected_indices:
            target_paths = [self.data[i]["file_path"] for i in sorted(self.selected_indices)]
        else:
            target_paths = [self.data[index]["file_path"]]
        app = KVApp.get_running_app()
        if app and hasattr(app, "main_widget"):
            app.main_widget.apply_paths_rating(target_paths, new_r)
