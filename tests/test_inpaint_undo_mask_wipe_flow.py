"""InpaintEffect/PatchmatchInpaintEffect: 履歴 undo で mask1 を再オープンする際に
描画済みマスクが握りつぶされる不具合の回帰テスト。

不具合の仕組み:
  1. StateBinding は widget の ToggleButton の `.state` を直接書き換える
     (effects.py: set_state_widget)。
  2. `.state` の変更は kv の `on_state: root.apply_effects_lv(...)` を同期的に
     再発火させる。
  3. 履歴 undo/redo で Operation.undo()/redo() が
     `effect.set2widget(widget, effects_param)` を呼ぶと、backup で
     `effects_param` に inpaint_mask_list を復元した直後に
     StateBinding が switch_inpaint.state を 'down' に変えてしまい、
     それが再入で apply_effects_lv → set2param → after_set2param を発火する。
  4. 旧実装の after_set2param は `mask_editor is None` かつ `inpaint == True`
     を「新規オープン」と誤認し、直前に復元したばかりの inpaint_mask_list を
     [] へ強制上書きしていた。

widgets.mask_editor.MaskEditor は kivy Window に依存するため、
sys.modules へスタブを差し込んで effects.InpaintEffect の実コードを
そのままヘッドレスで検証する。
"""

import pathlib
import sys
import types
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class _StubMaskEditor:
    """widgets.mask_editor.MaskEditor の最小スタブ。"""

    def __init__(self, param, **kwargs):
        self.param = param
        self.effect_ctrl_param = kwargs.get('effect_ctrl_param')
        self.touch_up_callback = kwargs.get('touch_up_callback')
        self.added_masks = []
        self.cleared = 0
        self.canvas_updates = 0
        self.released = 0

    def release(self):
        # 実装側は破棄時に Window.mouse_pos の unbind として呼ぶ
        self.released += 1

    def clear_mask(self):
        self.added_masks = []
        self.cleared += 1

    def add_mask(self, disp_info, image):
        self.added_masks.append((tuple(disp_info), image))

    def delay_update_canvas(self):
        self.canvas_updates += 1


def _install_stub_mask_editor_module():
    stub_module = types.ModuleType("widgets.mask_editor")
    stub_module.MaskEditor = _StubMaskEditor
    sys.modules["widgets.mask_editor"] = stub_module


_install_stub_mask_editor_module()

import effects  # noqa: E402  (import after stubbing widgets.mask_editor)
import history  # noqa: E402


class _ReactiveSwitch:
    """Kivy ToggleButton.state の代わり。値が実際に変わったときだけ
    on_state バインディング相当のコールバックを同期的に発火する。"""

    def __init__(self, widget, on_state_callback):
        self._state = "normal"
        self._widget = widget
        self._on_state_callback = on_state_callback

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if value == self._state:
            return
        self._state = value
        self._on_state_callback()


class _ActiveSwitch:
    def __init__(self, active=True):
        self.active = active


class _PreviewWidget:
    def __init__(self):
        self.children = []

    def add_widget(self, widget, index=0):
        self.children.insert(index, widget)

    def remove_widget(self, widget):
        if widget in self.children:
            self.children.remove(widget)


def _load_on_mask1_make_mask_state():
    """main.py から on_mask1_make_mask_state をソース抽出する。
    kv の on_state はこのハンドラを呼ぶ(begin → apply → end を state 変化時点で一括)。"""
    import ast
    import textwrap

    source_text = (PROJECT_ROOT / "main.py").read_text()
    tree = ast.parse(source_text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MainWidget":
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == "on_mask1_make_mask_state":
                    ns = {}
                    exec(textwrap.dedent(ast.get_source_segment(source_text, child)), ns)
                    return ns["on_mask1_make_mask_state"]
    raise AssertionError("on_mask1_make_mask_state not found in main.py")


_ON_MASK1_MAKE_MASK_STATE = _load_on_mask1_make_mask_state()


class _StubMainWidget:
    """effects.InpaintEffect が要求する root widget API の最小実装。

    on_state バインディングを switch_inpaint に配線し、実際の main.kv と同じく
    on_mask1_make_mask_state (begin → apply_effects_lv → end) を呼ぶ。
    begin/end は本物の history.Operation を使い、diff が空でない op だけ
    history_ops へ積む(main.py の begin/end_history_effect_ctrl 相当)。
    """

    def __init__(self, effect, param):
        self.primary_effects = [{'inpaint': effect}]
        self.primary_param = param
        self.ids = {
            'switch_inpaint': _ReactiveSwitch(self, self._on_switch_inpaint_state),
            'switch_details': _ActiveSwitch(True),
            'preview_widget': _PreviewWidget(),
        }
        self._effect = effect
        self.enter_calls = []
        self.exit_calls = []
        self.run_set2widget_all = False
        self.current_op = None
        self.history_ops = []

    def _on_switch_inpaint_state(self):
        # main.kv: on_state: root.on_mask1_make_mask_state('inpaint')
        _ON_MASK1_MAKE_MASK_STATE(self, 'inpaint')

    def begin_history_effect_ctrl(self, lv, effect_name, subname=None):
        if self.run_set2widget_all:
            return False
        self.current_op = history.Operation(lv, [effect_name], subname, None)
        self.current_op.set_backup(self.primary_effects, self.primary_param, subname)
        return True

    def end_history_effect_ctrl(self, lv, effect_name, subname=None):
        if self.current_op is None:
            return
        if self.current_op.set_update(self.primary_effects, self.primary_param, subname) is not None:
            self.history_ops.append(self.current_op)
        self.current_op = None

    def apply_effects_lv(self, lv, effect_name, subname=None):
        self._effect.set2param(self.primary_param, self)

    def enter_mask1_full_preview_mode(self, source, redraw=False):
        self.enter_calls.append(source)

    def exit_mask1_full_preview_mode(self, source, redraw=False):
        self.exit_calls.append(source)


def _make_mask(tag):
    image = np.full((10, 10), 255, dtype=np.uint8)
    return effects.InpaintDiff(type="mask", disp_info=(tag, 0, 10, 10), image=image)


class InpaintUndoMaskWipeTest(unittest.TestCase):
    def test_undo_of_close_preserves_drawn_masks(self):
        effect = effects.InpaintEffect()
        param = effect.get_param_dict({})
        param['original_img_size'] = (100, 100)
        widget = _StubMainWidget(effect, param)

        # 1) mask1 を開く(ユーザーが Make mask ボタンを押す)。
        widget.ids['switch_inpaint'].state = "down"
        self.assertIsNotNone(effect.mask_editor)
        self.assertEqual(param['inpaint_mask_list'], [])

        # 2) 線を2本描く(MaskEditor.on_touch_up が effect_ctrl_param 経由で
        #    begin/end_history_effect_ctrl(0, 'inpaint') を挟んで呼ぶ想定を、
        #    履歴 Operation で直接模擬する)。
        op_draw1 = history.Operation(0, ['inpaint'], None, None)
        op_draw1.set_backup(widget.primary_effects, param)
        effect.mask_editor_touch_up(param, np.zeros((100, 100), dtype=np.uint8))
        param['inpaint_mask_list'] = effect.inpaint_mask_list = [_make_mask(1)]
        op_draw1.set_update(widget.primary_effects, param)

        op_draw2 = history.Operation(0, ['inpaint'], None, None)
        op_draw2.set_backup(widget.primary_effects, param)
        param['inpaint_mask_list'] = effect.inpaint_mask_list = [_make_mask(1), _make_mask(2)]
        op_draw2.set_update(widget.primary_effects, param)

        self.assertEqual(len(param['inpaint_mask_list']), 2)

        # 3) editor1 を閉じる(タブ切替などで inpaint=False になる)。
        op_close = history.Operation(0, ['inpaint'], None, None)
        op_close.set_backup(widget.primary_effects, param)
        widget.ids['switch_inpaint'].state = "normal"
        op_close.set_update(widget.primary_effects, param)

        self.assertIsNone(effect.mask_editor)
        self.assertEqual(param['inpaint_mask_list'], [])

        # 4) 履歴を1つ戻す = op_close を undo する。
        #    StateBinding が switch_inpaint.state を "down" に戻すと、on_state
        #    経由で apply_effects_lv が再入し、旧実装ではここで
        #    inpaint_mask_list が [] に握りつぶされていた。
        op_close.undo(widget)

        self.assertEqual(param['inpaint'], True)
        self.assertIsNotNone(effect.mask_editor)
        self.assertEqual(len(param['inpaint_mask_list']), 2, "undo で復元した2本のマスクが握りつぶされてはいけない")
        # 新しく作られた mask_editor にも復元したマスクが描き戻されていること。
        self.assertEqual(len(effect.mask_editor.added_masks), 2)

    def test_undo_further_back_and_redo_restores_two_masks(self):
        # ユーザー報告の「さらに2個戻して2個進めると2本ある」を再現し、
        # 修正後は最初の undo 1回でも同じ状態になることを確認する。
        effect = effects.InpaintEffect()
        param = effect.get_param_dict({})
        param['original_img_size'] = (100, 100)
        widget = _StubMainWidget(effect, param)

        widget.ids['switch_inpaint'].state = "down"

        op_draw1 = history.Operation(0, ['inpaint'], None, None)
        op_draw1.set_backup(widget.primary_effects, param)
        param['inpaint_mask_list'] = effect.inpaint_mask_list = [_make_mask(1)]
        op_draw1.set_update(widget.primary_effects, param)

        op_draw2 = history.Operation(0, ['inpaint'], None, None)
        op_draw2.set_backup(widget.primary_effects, param)
        param['inpaint_mask_list'] = effect.inpaint_mask_list = [_make_mask(1), _make_mask(2)]
        op_draw2.set_update(widget.primary_effects, param)

        op_close = history.Operation(0, ['inpaint'], None, None)
        op_close.set_backup(widget.primary_effects, param)
        widget.ids['switch_inpaint'].state = "normal"
        op_close.set_update(widget.primary_effects, param)

        # undo close, then undo draw2, then redo draw2: 2本あるはず。
        op_close.undo(widget)
        op_draw2.undo(widget)
        self.assertEqual(len(param['inpaint_mask_list']), 1)
        op_draw2.redo(widget)
        self.assertEqual(len(param['inpaint_mask_list']), 2)


class MakeMaskButtonHistoryTest(unittest.TestCase):
    """Make mask ボタン(ToggleButton)による開閉が履歴に記録されることの回帰テスト。

    Kivy ToggleButton は on_press の前に state が変わる(= on_state が先に発火する)ため、
    旧 kv 配線(on_press: begin → on_state: apply → on_release: end)では backup が
    「開閉反映後」の param を撮ってしまい diff が空になり、開閉が履歴に一切残らなかった。
    現在は on_state から on_mask1_make_mask_state が begin → apply → end を一括で行う。
    """

    def _setup(self):
        effect = effects.InpaintEffect()
        param = effect.get_param_dict({})
        param['original_img_size'] = (100, 100)
        widget = _StubMainWidget(effect, param)
        return effect, param, widget

    def test_open_via_button_is_recorded(self):
        effect, param, widget = self._setup()

        widget.ids['switch_inpaint'].state = "down"

        self.assertIsNotNone(effect.mask_editor)
        self.assertEqual(len(widget.history_ops), 1)
        op = widget.history_ops[0]
        self.assertEqual(op.backup.get('inpaint'), False)
        self.assertEqual(op.update.get('inpaint'), True)

    def test_close_via_button_is_recorded_with_masks_in_backup(self):
        effect, param, widget = self._setup()
        widget.ids['switch_inpaint'].state = "down"
        param['inpaint_mask_list'] = effect.inpaint_mask_list = [_make_mask(1), _make_mask(2)]

        widget.ids['switch_inpaint'].state = "normal"

        self.assertIsNone(effect.mask_editor)
        self.assertEqual(len(widget.history_ops), 2)
        op_close = widget.history_ops[1]
        self.assertEqual(op_close.backup.get('inpaint'), True)
        self.assertEqual(len(op_close.backup.get('inpaint_mask_list')), 2)
        self.assertEqual(op_close.update.get('inpaint'), False)
        self.assertEqual(op_close.update.get('inpaint_mask_list'), [])

    def test_undo_of_recorded_close_reopens_editor_with_masks(self):
        effect, param, widget = self._setup()
        widget.ids['switch_inpaint'].state = "down"
        param['inpaint_mask_list'] = effect.inpaint_mask_list = [_make_mask(1), _make_mask(2)]
        widget.ids['switch_inpaint'].state = "normal"
        op_close = widget.history_ops[-1]
        ops_before_undo = len(widget.history_ops)

        # undo: set2widget が state を 'down' に戻す → on_state → ハンドラ再入。
        # param は既に復元済みなので diff は空になり、幽霊 op を積んではいけない。
        op_close.undo(widget)

        self.assertEqual(param['inpaint'], True)
        self.assertEqual(widget.ids['switch_inpaint'].state, "down")
        self.assertIsNotNone(effect.mask_editor)
        self.assertEqual(len(param['inpaint_mask_list']), 2)
        self.assertEqual(len(effect.mask_editor.added_masks), 2)
        self.assertEqual(len(widget.history_ops), ops_before_undo, "undo が新しい履歴を作ってはいけない")

    def test_redo_of_recorded_close_closes_editor_without_ghost_ops(self):
        effect, param, widget = self._setup()
        widget.ids['switch_inpaint'].state = "down"
        param['inpaint_mask_list'] = effect.inpaint_mask_list = [_make_mask(1)]
        widget.ids['switch_inpaint'].state = "normal"
        op_close = widget.history_ops[-1]
        op_close.undo(widget)
        ops_before_redo = len(widget.history_ops)

        op_close.redo(widget)

        self.assertEqual(param['inpaint'], False)
        self.assertEqual(widget.ids['switch_inpaint'].state, "normal")
        self.assertIsNone(effect.mask_editor)
        self.assertEqual(param['inpaint_mask_list'], [])
        self.assertEqual(len(widget.history_ops), ops_before_redo, "redo が新しい履歴を作ってはいけない")


if __name__ == "__main__":
    unittest.main()
