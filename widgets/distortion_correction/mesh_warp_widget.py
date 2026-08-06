"""
メッシュワープWidget
"""

from kivy.uix.floatlayout import FloatLayout as KVFloatLayout
from kivy.uix.widget import Widget as KVWidget
from kivy.uix.label import Label as KVLabel
from kivy.uix.boxlayout import BoxLayout as KVBoxLayout
from kivy.properties import ListProperty as KVListProperty, DictProperty as KVDictProperty
from kivy.graphics import Color, Line
from kivy.clock import mainthread as kvmainthread
import os
import copy
import numpy as np
import cv2

from cores.distortion_correction.warp_correction import get_mesh_coordinates
import params
from utils import kvutils
from widgets.scaled_button import ScaledButton


_MESH_DEBUG = os.getenv("PLATYPUS_DEBUG_MESH_WARP", "0").strip().lower() in ("1", "true", "yes", "on")

# コントロールポイントの表示半径(通常/選択)と当たり半径(いずれもウィンドウ px)。
# 当たり判定は表示円より少しだけ大きい程度にする (以前は TCG 0.05 で大幅に広かった)。
_CP_SIZE_NORMAL = 5
_CP_SIZE_SELECTED = 8
_CP_HIT_RADIUS_PX = 12

class MeshWarpWidget(KVFloatLayout):
    """メッシュワープWidget"""
    
    # プロパティ
    mesh_size = KVListProperty([4, 4]) # [rows, cols]
    control_offsets_tcg = KVDictProperty({}) # {(row, col): (off_x, off_y)}
    
    def __init__(
            self,
            texture_size,
            param,
            force_square_disp_info=False,
            show_shift_reset_hint=False,
            **kwargs):
        """
        Args:
            texture_size: preview texture サイズ (W, H)
            param: 画像 param (primary_param)
            force_square_disp_info: True のとき、param の disp_info を「正方形全範囲」
                に強制上書きする旧互換用オプション。
                Mask Geometry 編集モード (Mask2 ON + Ge タブ) では pipeline 側が
                primary_param['disp_info'] を画像 aspect (非正方形) に書き換える。
                このとき MeshWarpWidget 内部の座標計算スケールが画像 mesh と 2 倍違い、
                CP 表示が右下に拡大するため、square 範囲の disp_info を明示する必要が
                ある。画像 mesh editor (= 通常用途) では False のままで OK。
            show_shift_reset_hint: True のとき Reset 下に Shift+ ヒントを表示する。
        """
        super().__init__(**kwargs)

        self.texture_size = texture_size
        self.tcg_info = params.param_to_tcg_info(param)
        self._view_context = None
        self._view_context_image_only_matrix = False

        if force_square_disp_info:
            import config as _config
            orig = self.tcg_info.get('original_img_size') or (0, 0)
            imax = max(orig)
            if imax > 0:
                # normalized 表現の「画像全体正方形」を強制
                self.tcg_info['disp_info'] = (
                    0.0, 0.0, 1.0, 1.0, _config.get_preview_texture_side() / imax,
                )

        self.on_callback = None
        
        # 描画用オーバーレイ
        self.draw_overlay = KVWidget()
        self.add_widget(self.draw_overlay)
        
        # 状態変数
        self.selected_point = None # (row, col)
        self.dragging = False
        
        # UIコントロール（リセットボタンなど）
        button_layout = KVBoxLayout(
            orientation='horizontal',
            size_hint=(None, None),
            pos_hint={'center_x': 0.5, 'y': 0.035},
        )
        button_layout.ref_layout_spacing = 5
        button_layout.ref_layout_padding = 5
        button_layout.bind(
            minimum_width=button_layout.setter('width'),
            minimum_height=button_layout.setter('height'),
        )
        kvutils.traverse_widget(button_layout)

        self.reset_btn = ScaledButton(text="Reset")
        self.reset_btn.set_ref_metrics()
        self.reset_btn.bind(on_press=self.reset_mesh)

        if show_shift_reset_hint:
            hint_font_px = '12px'
            hint_height_ref = 8
            reset_layout_height_ref = 22
            reset_layout = KVFloatLayout(
                size_hint=(None, None),
                width=kvutils.dpi_scale_width(50),
                height=kvutils.dpi_scale_height(reset_layout_height_ref),
            )
            reset_layout.add_widget(self.reset_btn)

            def _sync_reset_button_pos(*_args):
                reset_layout.width = self.reset_btn.width
                reset_layout.height = kvutils.dpi_scale_height(reset_layout_height_ref)
                self.reset_btn.pos = reset_layout.pos

            reset_shift_hint = KVLabel(
                text="Shift+",
                size_hint=(None, None),
                width=kvutils.dpi_scale_width(50),
                height=kvutils.dpi_scale_height(hint_height_ref),
                halign='center',
                valign='middle',
                font_size=hint_font_px,
                color=(1, 1, 1, 0.55),
            )

            def _sync_reset_shift_hint(*_args):
                _sync_reset_button_pos()
                reset_shift_hint.width = self.reset_btn.width
                reset_shift_hint.height = kvutils.dpi_scale_height(hint_height_ref)
                reset_shift_hint.pos = (
                    reset_layout.x,
                    reset_layout.y - reset_shift_hint.height,
                )
                reset_shift_hint.text_size = reset_shift_hint.size

            _sync_reset_shift_hint()
            self.reset_btn.bind(size=_sync_reset_shift_hint)
            reset_layout.bind(pos=_sync_reset_shift_hint, size=_sync_reset_shift_hint)
            reset_layout.add_widget(reset_shift_hint)
            button_layout.add_widget(reset_layout)
        else:
            button_layout.add_widget(self.reset_btn)

        # Rows Control
        button_layout.add_widget(KVLabel(text="Y:", size_hint_x=None, width=40, halign='right', color=(1,1,1,1)))
        btn_row_minus = ScaledButton(text="-", on_release=lambda x: self.change_mesh_size(-2, 0))
        btn_row_minus.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
        btn_row_minus.set_ref_metrics(width_ref=20, height_ref=20, font_ref=14, padding_x_ref=1, padding_y_ref=0)
        button_layout.add_widget(btn_row_minus)
        
        self.label_rows = KVLabel(text=str(self.mesh_size[0]), size_hint_x=None, width=40, halign='center', color=(1,1,1,1))
        self.label_rows.size_hint_x = None
        self.label_rows.ref_width = 40
        button_layout.add_widget(self.label_rows)
        
        btn_row_plus = ScaledButton(text="+", on_release=lambda x: self.change_mesh_size(2, 0))
        btn_row_plus.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
        btn_row_plus.set_ref_metrics(width_ref=20, height_ref=20, font_ref=14, padding_x_ref=1, padding_y_ref=0)
        button_layout.add_widget(btn_row_plus)

        # Cols Control
        button_layout.add_widget(KVLabel(text="X:", size_hint_x=None, width=40, halign='right', color=(1,1,1,1)))
        btn_col_minus = ScaledButton(text="-", on_release=lambda x: self.change_mesh_size(0, -2))
        btn_col_minus.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
        btn_col_minus.set_ref_metrics(width_ref=20, height_ref=20, font_ref=14, padding_x_ref=1, padding_y_ref=0)
        button_layout.add_widget(btn_col_minus)
        
        self.label_cols = KVLabel(text=str(self.mesh_size[1]), size_hint_x=None, width=40, halign='center', color=(1,1,1,1))
        self.label_cols.size_hint_x = None
        self.label_cols.ref_width = 40
        button_layout.add_widget(self.label_cols)
        
        btn_col_plus = ScaledButton(text="+", on_release=lambda x: self.change_mesh_size(0, 2))
        btn_col_plus.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
        btn_col_plus.set_ref_metrics(width_ref=20, height_ref=20, font_ref=14, padding_x_ref=1, padding_y_ref=0)
        button_layout.add_widget(btn_col_plus)
        
        self.apply_btn = ScaledButton(text="Apply")
        self.apply_btn.set_ref_metrics()
        self.apply_btn.bind(on_press=lambda x: self._apply())
        button_layout.add_widget(self.apply_btn)

        self.add_widget(button_layout)
        
        # Update labels on property change
        self.bind(mesh_size=self._update_labels)

        # タッチイベント
        self.draw_overlay.bind(
            on_touch_down=self._on_touch_down,
            on_touch_move=self._on_touch_move,
            on_touch_up=self._on_touch_up
        )
        
        # プロパティ監視
        self.bind(mesh_size=self._redraw_mesh)
        self.bind(control_offsets_tcg=self._redraw_mesh)
        self.bind(size=self._redraw_mesh)
        self.bind(pos=self._redraw_mesh)
        kvutils.install_ref_scaling(self)

    def _update_labels(self, instance, value):
        self.label_rows.text = str(value[0])
        self.label_cols.text = str(value[1])

    def change_mesh_size(self, d_row, d_col):
        """メッシュサイズを変更し、オフセットを補間する"""
        rows, cols = self.mesh_size
        new_rows = rows + d_row
        new_cols = cols + d_col
        
        # 制限 (偶数のみ、最小4、最大20程度)
        if new_rows < 4: new_rows = 4
        if new_cols < 4: new_cols = 4
        if new_rows > 16: new_rows = 16
        if new_cols > 16: new_cols = 16
        
        if new_rows == rows and new_cols == cols:
            return

        # 現在のオフセットを密な配列に変換 (rows+1, cols+1, 2)
        current_field = np.zeros((rows + 1, cols + 1, 2), dtype=np.float32)
        for (r, c), offset in self.control_offsets_tcg.items():
            current_field[r, c] = offset
            
        # リサイズ (補間)
        # cv2.resize takes (width, height) -> (cols, rows)
        new_field = cv2.resize(current_field, (new_cols + 1, new_rows + 1), interpolation=cv2.INTER_LINEAR)
        
        # 配列を新しいDictに戻す
        new_offsets = {}
        for r in range(new_rows + 1):
            for c in range(new_cols + 1):
                off = new_field[r, c]
                if abs(off[0]) > 1e-6 or abs(off[1]) > 1e-6:
                    new_offsets[(r, c)] = (float(off[0]), float(off[1]))
        
        # 一括更新
        self.control_offsets_tcg = new_offsets
        self.on_edit_start()
        self.mesh_size = [new_rows, new_cols]
        self.on_edit_end()

    def set_callback(self, callback):
        self.on_callback = callback

    def set_texture_size(self, texture_size):
        if self._view_context is not None:
            self.sync_view_from_context()
            return
        self.texture_size = texture_size
        self._redraw_mesh()

    @staticmethod
    def _copy_tcg_info(tcg_info):
        copied = {}
        for key, value in tcg_info.items():
            if key == 'matrix':
                copied[key] = np.array(value, dtype=np.float64, copy=True)
            else:
                copied[key] = copy.deepcopy(value)
        return copied

    def set_tcg_info(self, tcg_info):
        """表示中 preview と同じ座標系へ同期する。

        control_offsets_tcg は正規化TCG上の編集値なので保持し、disp_info / matrix など
        view 変換だけを差し替える。Mask Mesh editor が zoom/scroll 後も表示cropに追従
        するために使う。
        """
        if not isinstance(tcg_info, dict):
            return
        self._view_context = None
        self._view_context_image_only_matrix = False
        self.tcg_info = self._copy_tcg_info(tcg_info)
        self._redraw_mesh()

    def set_view_param(self, param):
        self.set_tcg_info(params.param_to_tcg_info(param))

    def set_view_context(self, context, image_only_matrix=False):
        """外部の表示座標 context をこの widget の view source にする。

        MeshWarpWidget 自身は control_offsets_tcg だけを編集状態として持ち、
        zoom / scroll / crop は context 側を正とする。Mask Mesh editor では
        MaskEditor2 を渡しつつ image_only_matrix=True にすることで、Mask Geom ではなく
        画像 Geom の座標系にメッシュグリッドを固定する。
        """
        self._view_context = context
        self._view_context_image_only_matrix = bool(image_only_matrix)
        self.sync_view_from_context()

    def sync_view_from_context(self):
        if self._sync_view_from_context():
            self._redraw_mesh()

    def _sync_view_from_context(self):
        context = self._view_context
        if context is None:
            return False

        lock = getattr(context, '_matrix_lock', None)
        if lock is None:
            texture_size = getattr(context, 'texture_size', None)
            tcg_info = getattr(context, 'tcg_info', None)
        else:
            with lock:
                texture_size = getattr(context, 'texture_size', None)
                tcg_info = getattr(context, 'tcg_info', None)
                texture_size = tuple(texture_size) if texture_size is not None else None
                tcg_info = self._copy_tcg_info(tcg_info) if isinstance(tcg_info, dict) else None
                if self._view_context_image_only_matrix and tcg_info is not None:
                    image_only = getattr(context, '_image_only_matrix', None)
                    if image_only is not None:
                        tcg_info['matrix'] = np.array(image_only, dtype=np.float64, copy=True)

        if lock is None:
            texture_size = tuple(texture_size) if texture_size is not None else None
            tcg_info = self._copy_tcg_info(tcg_info) if isinstance(tcg_info, dict) else None
            if self._view_context_image_only_matrix and tcg_info is not None:
                image_only = getattr(context, '_image_only_matrix', None)
                if image_only is not None:
                    tcg_info['matrix'] = np.array(image_only, dtype=np.float64, copy=True)

        changed = False
        if texture_size is not None and texture_size != tuple(self.texture_size):
            self.texture_size = texture_size
            changed = True
        if tcg_info is not None:
            old_info = self.tcg_info if isinstance(self.tcg_info, dict) else {}
            old_matrix = old_info.get('matrix')
            new_matrix = tcg_info.get('matrix')
            matrix_changed = (
                old_matrix is None
                or new_matrix is None
                or not np.array_equal(np.asarray(old_matrix), np.asarray(new_matrix))
            )
            view_changed = any(
                old_info.get(key) != tcg_info.get(key)
                for key in ('disp_info', 'original_img_size', 'rotation', 'rotation2', 'flip_mode')
            )
            if view_changed or matrix_changed:
                self.tcg_info = tcg_info
                changed = True
        return changed

    def on_edit_start(self):
        if self.on_callback:
            self.on_callback('start', self)

    def on_edit_end(self):
        if self.on_callback:
            self.on_callback('end', self)

    def reset_mesh(self, instance=None):
        """メッシュをリセット"""
        self.on_edit_start()
        self.control_offsets_tcg = {}
        self.on_edit_end()
        self._redraw_mesh()
        self._apply()

    def get_correction_params(self) -> dict:
        """現在のパラメータを取得"""
        # JSON互換のためキーを文字列に変換
        cp_str_keys = {
            f"{k[0]},{k[1]}": (float(v[0]), float(v[1]))
            for k, v in self.control_offsets_tcg.items()
        }
        return {
            "mesh_size": list(self.mesh_size),
            "control_points": cp_str_keys,
        }
    
    def set_correction_params(self, params: dict):
        """パラメータを設定"""
        self.mesh_size = params.get('mesh_size', [4, 4])
        # Dict key must be tuple, json might give list/string
        raw_offsets = params.get('control_points', {})
        # Ensure keys are tuples
        cleaned_offsets = {}
        for k, v in raw_offsets.items():
            if isinstance(k, str):
                try:
                    # eval is dangerous but simple context depends on usage
                    # Better parse "1,2" -> (1,2)
                    parts = k.strip('()').split(',')
                    key = (int(parts[0]), int(parts[1]))
                except (ValueError, IndexError):
                    continue
            else:
                key = tuple(k)
            cleaned_offsets[key] = tuple(v)
            
        self.control_offsets_tcg = cleaned_offsets
        self._redraw_mesh()

    def _apply(self):
        """適用"""
        if self.on_callback:
            self.on_callback('apply', self)

    def _get_tcg_pos(self, touch_pos):
        self._sync_view_from_context()
        tx, ty = params.window_to_tcg(touch_pos[0], touch_pos[1], self, self.texture_size, self.tcg_info)
        return tx, ty

    def _get_window_pos(self, tx, ty):
        wx, wy = params.tcg_to_window(tx, ty, self, self.texture_size, self.tcg_info)
        return wx, wy
        
    def _hit_test(self, tcg_x, tcg_y, touch_tcg_x, touch_tcg_y):
        # TCG空間での距離判定 
        # メッシュの密度によるが、適当な閾値
        dist = np.sqrt((tcg_x - touch_tcg_x)**2 + (tcg_y - touch_tcg_y)**2)
        return dist < 0.04 

    def _on_touch_down(self, instance, touch):
        if not self.collide_point(*touch.pos):
            return False
            
        touch.grab(self)
        
        tx, ty = self._get_tcg_pos(touch.pos)

        # ヒットテスト (ウィンドウ座標=表示円と同じ空間で px 判定)
        rows, cols = self.mesh_size
        img_shape = (self.texture_size[1], self.texture_size[0]) # H, W
        base_coords = get_mesh_coordinates(img_shape, (rows, cols)) # Shape: (rows+1, cols+1, 2)

        min_dist = _CP_HIT_RADIUS_PX
        hit_pt = None

        for r in range(rows + 1):
            for c in range(cols + 1):
                bx, by = base_coords[r, c]
                off_x, off_y = self.control_offsets_tcg.get((r, c), (0, 0))
                wx, wy = self._get_window_pos(bx + off_x, by + off_y)
                dist = np.hypot(wx - touch.pos[0], wy - touch.pos[1])
                if dist < min_dist:
                    min_dist = dist
                    hit_pt = (r, c)
        
        if hit_pt:
            # 右クリック判定
            if getattr(touch, 'button', 'left') == 'right':
                # リセット
                if hit_pt in self.control_offsets_tcg:
                    self.on_edit_start()
                    del self.control_offsets_tcg[hit_pt]
                    self.on_edit_end()
                    self._redraw_mesh()
                return True

            self.selected_point = hit_pt
            self.dragging = True
            self.on_edit_start()
            self._redraw_mesh()
            return True
            
        return True # Consume touch to prevent propagation

    def _on_touch_move(self, instance, touch):
        if touch.grab_current is not self:
            return False
            
        if self.dragging and self.selected_point:
            tx, ty = self._get_tcg_pos(touch.pos)
            
            # 元の座標を取得してオフセットを計算
            rows, cols = self.mesh_size
            img_shape = (self.texture_size[1], self.texture_size[0])
            base_coords = get_mesh_coordinates(img_shape, (rows, cols))
            r, c = self.selected_point
            bx, by = base_coords[r, c]
            
            off_x = tx - bx
            off_y = ty - by
            
            # 更新 (DictProperty update hack)
            new_offsets = dict(self.control_offsets_tcg)
            new_offsets[self.selected_point] = (off_x, off_y)
            self.control_offsets_tcg = new_offsets
            
            # リアルタイム反映する場合はここで _apply だが重いので描画のみ
            self._redraw_mesh()
            
        return True

    def _on_touch_up(self, instance, touch):
        if touch.grab_current is not self:
            return False
        
        touch.ungrab(self)
        
        if self.dragging:
            self.dragging = False
            self.selected_point = None
            self.on_edit_end()
            self._redraw_mesh()
            
        return True

    def _redraw_mesh(self, *args):
        self._sync_view_from_context()
        # 環境変数 PLATYPUS_DEBUG_MESH_WARP=1 のときだけ debug log を出力。
        # 通常は早期 if で skip するので hot path に影響しない。
        if _MESH_DEBUG:
            try:
                import logging
                import macos as _device
                _disp = self.tcg_info.get('disp_info') if isinstance(self.tcg_info, dict) else None
                logging.warning(
                    "[MESH_DEBUG] _redraw_mesh widget_id=%s parent=%s self.size=%s self.pos=%s "
                    "texture_size=%s mesh_size=%s n_cps=%d disp_info=%s orig=%s dpi=%s",
                    id(self),
                    type(self.parent).__name__ if self.parent else "None",
                    tuple(self.size), tuple(self.pos),
                    tuple(self.texture_size), tuple(self.mesh_size),
                    len(self.control_offsets_tcg or {}),
                    _disp,
                    self.tcg_info.get('original_img_size') if isinstance(self.tcg_info, dict) else None,
                    _device.dpi_scale() if hasattr(_device, 'dpi_scale') else None,
                )
            except Exception:
                # 高頻度経路のためログ抑制（デバッグログ自体の失敗でredrawを止めない）
                pass

        self.draw_overlay.canvas.clear()

        rows, cols = self.mesh_size
        if self.texture_size[0] == 0: return

        img_shape = (self.texture_size[1], self.texture_size[0])
        base_coords = get_mesh_coordinates(img_shape, (rows, cols))
        
        # 現在の変形後座標を計算
        current_points = np.zeros_like(base_coords)
        for r in range(rows + 1):
            for c in range(cols + 1):
                bx, by = base_coords[r, c]
                off_x, off_y = self.control_offsets_tcg.get((r, c), (0, 0))
                current_points[r, c] = [bx + off_x, by + off_y]

        # 描画
        with self.draw_overlay.canvas:
            Color(0.0, 1.0, 1.0, 0.6) # Cyan lines
            
            # Horizontal lines
            for r in range(rows + 1):
                pts = []
                for c in range(cols + 1):
                    ctx, cty = current_points[r, c]
                    wx, wy = self._get_window_pos(ctx, cty)
                    pts.extend([wx, wy])
                Line(points=pts, width=1.2)
                
            # Vertical lines
            for c in range(cols + 1):
                pts = []
                for r in range(rows + 1):
                    ctx, cty = current_points[r, c]
                    wx, wy = self._get_window_pos(ctx, cty)
                    pts.extend([wx, wy])
                Line(points=pts, width=1.2)
            
            # Control points
            for r in range(rows + 1):
                for c in range(cols + 1):
                    if self.selected_point == (r, c):
                         Color(1.0, 0.8, 0.2, 1.0)
                         size = _CP_SIZE_SELECTED
                    else:
                         Color(1.0, 1.0, 1.0, 0.8)
                         size = _CP_SIZE_NORMAL
                         
                    ctx, cty = current_points[r, c]
                    wx, wy = self._get_window_pos(ctx, cty)
                    Line(circle=(wx, wy, size), width=1.5)
