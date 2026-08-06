import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import effects


class DummySlider:
    def __init__(self, value=None):
        self.value = value
        self.set_calls = []

    def set_slider_value(self, value):
        self.value = value
        self.set_calls.append(value)


class DummySwitch:
    def __init__(self, active=None, enabled=None):
        self.active = active
        self.enabled = enabled


class DummySpinner:
    def __init__(self, text=""):
        self.text = text
        self.hovered_item = None
        self.set_calls = []

    def set_text(self, value):
        self.text = value
        self.set_calls.append(value)


class DummyPointList:
    def __init__(self):
        self.point_list = None
        self.set_calls = []

    def set_point_list(self, point_list):
        self.point_list = point_list
        self.set_calls.append(point_list)

    def get_point_list(self, *args):
        return self.point_list


class DummyColorPicker:
    def __init__(self):
        self.ids = {
            "slider_hue": DummySlider(),
            "slider_lum": DummySlider(),
            "slider_sat": DummySlider(),
        }
        self.set_calls = []

    def set_slider_value(self, value):
        self.set_calls.append(value)
        self.ids["slider_hue"].value = value[0]
        self.ids["slider_lum"].value = value[1]
        self.ids["slider_sat"].value = value[2]


class DummyIds(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


class DummyMaskEditor:
    def __init__(self):
        self.cleared = 0
        self.added_masks = []
        self.delay_updates = 0
        self.released = 0

    def clear_mask(self):
        self.cleared += 1

    def release(self):
        # 実装側は破棄時に Window.mouse_pos の unbind として呼ぶ
        self.released += 1

    def add_mask(self, disp_info, image):
        self.added_masks.append((disp_info, image))

    def delay_update_canvas(self):
        self.delay_updates += 1


class DummyPreviewWidget:
    def __init__(self):
        self.added = []
        self.removed = []

    def add_widget(self, widget):
        self.added.append(widget)

    def remove_widget(self, widget):
        self.removed.append(widget)


class DummyDistortionPainter:
    def __init__(self):
        self.recorded = None
        self.remap_calls = 0
        self.brush_size = None
        self.strength = None

    def set_recorded(self, recorded):
        self.recorded = recorded

    def remap_recorded(self):
        self.remap_calls += 1

    def set_brush_size(self, value):
        self.brush_size = value

    def set_strength(self, value):
        self.strength = value


class TestableDistortionEffect(effects.DistortionEffect):
    def _open_distortion_painter(self, param, widget):
        if self.distortion_painter is None:
            self.distortion_painter = DummyDistortionPainter()

    def _close_distortion_painter(self, param, widget):
        self.distortion_painter = None


class LensDistortionWidget:
    def __init__(self):
        self.set_params = None

    def set_correction_params(self, param):
        self.set_params = param

    def get_correction_params(self):
        return {
            "lens_distortion_strength": 999,
            "lens_distortion_scale": 888,
            "correct_horizontal": 12,
        }


class TestableGeometryEffect(effects.GeometryEffect):
    def __init__(self):
        super().__init__()
        self.opened_editor = None

    def _open_geometry_editor(self, widget, type, param):
        self.opened_editor = (type, param.copy())


class EffectParamBindingTest(unittest.TestCase):
    def _make_widget(self, effect):
        ids = {}
        for binding in effect.param_bindings:
            if isinstance(binding, effects.FunctionBinding):
                for widget_id in binding.widget_ids:
                    if binding.widget_setter == "set_point_list_widget":
                        ids[widget_id] = DummyPointList()
                    elif binding.widget_setter == "set_state_widget":
                        ids[widget_id] = SimpleNamespace(state="normal")
                    else:
                        ids[widget_id] = DummySpinner()
            elif binding.widget_setter == "set_slider_value":
                ids[binding.widget_id] = DummySlider()
            else:
                ids[binding.widget_id] = DummySwitch()
        return SimpleNamespace(ids=ids)

    def _set_widget_value(self, effect, widget, binding, value):
        if isinstance(binding, effects.FunctionBinding):
            target = widget.ids[binding.widget_ids[0]]
            if isinstance(target, DummyPointList):
                target.point_list = value
            elif hasattr(target, "state"):
                target.state = "down" if value else "normal"
            else:
                target.text = value
            return
        target = widget.ids[binding.widget_id]
        setattr(target, binding.widget_attr, value)

    def _assert_widget_value(self, effect, widget, param, binding, expected):
        if isinstance(binding, effects.FunctionBinding):
            self.assertEqual(binding.get_widget_value(effect, widget, param), expected)
            if binding.widget_setter == "set_state_widget":
                self.assertEqual(
                    widget.ids[binding.widget_ids[0]].state,
                    "down" if expected else "normal",
                )
                return
            for widget_id in binding.widget_ids:
                self.assertEqual(widget.ids[widget_id].set_calls, [expected])
            return

        target = widget.ids[binding.widget_id]
        if binding.widget_setter == "set_slider_value":
            self.assertEqual(target.value, expected)
            self.assertEqual(target.set_calls, [expected])
        else:
            self.assertEqual(getattr(target, binding.widget_attr), expected)

    def _exercise_binding_round_trip(self, effect_cls, override_param, widget_values):
        effect = effect_cls()
        defaults = effect.get_param_dict({})
        binding_defaults = {binding.key: binding.default for binding in effect.param_bindings}
        self.assertEqual(
            {key: defaults[key] for key in binding_defaults},
            binding_defaults,
        )

        widget = self._make_widget(effect)
        effect.set2widget(widget, override_param)
        for binding in effect.param_bindings:
            expected = override_param.get(binding.key, binding.default)
            self._assert_widget_value(effect, widget, override_param, binding, expected)

        for key, value in widget_values.items():
            binding = next(b for b in effect.param_bindings if b.key == key)
            self._set_widget_value(effect, widget, binding, value)

        param = {}
        effect.set2param(param, widget)
        expected_param = binding_defaults.copy()
        expected_param.update(widget_values)
        self.assertEqual(param, expected_param)

    def test_lut_spinner_binding_uses_hovered_item_and_resets_cache_on_name_change(self):
        effect = effects.LUTEffect()
        widget = self._make_widget(effect)

        effect.set2widget(widget, {
            "switch_lut": False,
            "lut_name": "input.cube",
            "lut_intensity": 75,
            "lut_to_log": "LogC",
        })

        self.assertFalse(widget.ids["switch_lut"].active)
        self.assertEqual(widget.ids["lut_spinner"].text, "input.cube")
        self.assertEqual(widget.ids["slider_lut_intensity"].value, 75)
        self.assertEqual(widget.ids["lut_to_log_spinner"].text, "LogC")

        effect.lut = object()
        effect.lut_key = ("old", "old.cube")
        widget.ids["switch_lut"].active = True
        widget.ids["lut_spinner"].hovered_item = SimpleNamespace(text="hovered.cube")
        widget.ids["slider_lut_intensity"].value = 55
        widget.ids["lut_to_log_spinner"].text = "None"

        param = {"lut_name": "input.cube"}
        effect.set2param(param, widget)

        self.assertEqual(
            param,
            {
                "lut_name": "hovered.cube",
                "switch_lut": True,
                "lut_intensity": 55,
                "lut_to_log": "None",
            },
        )
        self.assertIsNone(effect.lut)
        self.assertIsNone(effect.lut_key)

        effect.lut = object()
        effect.lut_key = ("same", "hovered.cube")
        widget.ids["lut_spinner"].hovered_item = None
        widget.ids["lut_spinner"].text = "hovered.cube"
        effect.set2param(param, widget)

        self.assertIsNotNone(effect.lut)
        self.assertEqual(effect.lut_key, ("same", "hovered.cube"))

    def test_color_match_keeps_source_image_out_of_ui_bindings(self):
        effect = effects.ColorMatchEffect()
        defaults = effect.get_param_dict({})
        self.assertEqual(defaults["color_match_source_image"], None)
        self.assertNotIn(
            "color_match_source_image",
            {binding.key for binding in effect.param_bindings},
        )

        widget = self._make_widget(effect)
        effect.set2widget(widget, {
            "switch_color_match": False,
            "switch_color_match_active": True,
            "color_match_intensity": 45,
            "color_match_source_image": object(),
        })

        self.assertFalse(widget.ids["switch_color_match"].active)
        self.assertTrue(widget.ids["switch_color_match_active"].active)
        self.assertEqual(widget.ids["slider_color_match_intensity"].value, 45)

        widget.ids["switch_color_match"].active = True
        widget.ids["switch_color_match_active"].active = False
        widget.ids["slider_color_match_intensity"].value = 55
        param = {}
        effect.set2param(param, widget)
        self.assertEqual(param, {
            "switch_color_match": True,
            "switch_color_match_active": False,
            "color_match_intensity": 55,
        })

    def test_ai_noise_keeps_result_out_of_ui_bindings(self):
        effect = effects.AINoiseReductonEffect()
        defaults = effect.get_param_dict({})
        self.assertEqual(defaults["ai_noise_reduction_result"], None)
        self.assertNotIn(
            "ai_noise_reduction_result",
            {binding.key for binding in effect.param_bindings},
        )

        widget = self._make_widget(effect)
        effect.set2widget(widget, {
            "switch_ai_noise_reduction": False,
            "ai_noise_reduction": True,
            "ai_noise_reduction_intensity": 45,
            "ai_noise_reduction_result": object(),
        })

        self.assertFalse(widget.ids["switch_ai_noise_reduction"].active)
        self.assertTrue(widget.ids["chip_ai_noise_reduction"].active)
        self.assertEqual(widget.ids["slider_ai_noise_reduction_intensity"].value, 45)

        widget.ids["switch_ai_noise_reduction"].active = True
        widget.ids["chip_ai_noise_reduction"].active = False
        widget.ids["slider_ai_noise_reduction_intensity"].value = 55
        param = {}
        effect.set2param(param, widget)
        self.assertEqual(param, {
            "switch_ai_noise_reduction": True,
            "ai_noise_reduction": False,
            "ai_noise_reduction_intensity": 55,
        })

    def test_film_mode_binding_uses_hovered_item(self):
        effect = effects.FilmSimulationEffect()
        widget = self._make_widget(effect)
        widget.ids["switch_film_simulation"].enabled = True
        widget.ids["spinner_film_mode"].text = "Negative"
        widget.ids["spinner_film_mode"].hovered_item = SimpleNamespace(text="Slide")
        widget.ids["slider_film_latitude"].value = 70
        widget.ids["slider_film_contrast"].value = 55
        widget.ids["slider_film_color_bias"].value = -12
        widget.ids["slider_film_color_drift"].value = 34
        widget.ids["slider_film_dye_purity"].value = 82
        widget.ids["slider_film_layer_crosstalk"].value = 25
        widget.ids["slider_film_halation"].value = 18
        widget.ids["slider_film_aging"].value = 9
        widget.ids["slider_film_intensity"].value = 65

        param = {}
        effect.set2param(param, widget)

        self.assertEqual(param["film_mode"], "Slide")
        self.assertEqual(param["film_latitude"], 70)
        self.assertEqual(param["film_contrast"], 55)
        self.assertEqual(param["film_color_bias"], -12)
        self.assertEqual(param["film_color_drift"], 34)
        self.assertEqual(param["film_dye_purity"], 82)
        self.assertEqual(param["film_layer_crosstalk"], 25)
        self.assertEqual(param["film_halation"], 18)
        self.assertEqual(param["film_aging"], 9)
        self.assertEqual(param["film_intensity"], 65)

    def test_film_off_mode_makes_no_diff(self):
        effect = effects.FilmSimulationEffect()
        image = np.full((2, 2, 3), 0.5, dtype=np.float32)

        diff = effect.make_diff(image, effect.get_param_dict({}), effects.EffectConfig())

        self.assertIsNone(diff)
        self.assertIsNone(effect.diff)

    def test_patchmatch_set2widget_does_not_touch_predict_button(self):
        # The Erase (predict) button is a momentary one-shot, not state-bound, so
        # set2widget must sync only the persistent "Make mask" toggle and leave the
        # Erase button state alone.
        effect = effects.PatchmatchInpaintEffect()
        widget = SimpleNamespace(ids={
            "switch_details": DummySwitch(),
            "switch_patchmatch_inpaint": SimpleNamespace(state="normal"),
            "button_patchmatch_inpaint_predict": SimpleNamespace(state="normal"),
        })

        effect.set2widget(widget, {
            "switch_details": True,
            "patchmatch_inpaint": True,
            "patchmatch_inpaint_predict": True,
        })

        self.assertEqual(widget.ids["switch_patchmatch_inpaint"].state, "down")
        self.assertEqual(widget.ids["button_patchmatch_inpaint_predict"].state, "normal")

    def test_inpaint_state_bindings_run_mask_editor_hooks(self):
        effect = effects.InpaintEffect()
        effect.mask_editor = DummyMaskEditor()
        mask = SimpleNamespace(disp_info=(1, 2, 3, 4), image="mask")
        preview = DummyPreviewWidget()
        exit_calls = []
        widget = SimpleNamespace(
            ids={
                "switch_details": DummySwitch(),
                "switch_inpaint": SimpleNamespace(state="normal"),
                "button_inpaint_predict": SimpleNamespace(state="normal"),
                "preview_widget": preview,
            },
            exit_mask1_full_preview_mode=lambda name: exit_calls.append(name),
        )

        effect.set2widget(widget, {
            "switch_details": False,
            "inpaint": True,
            "inpaint_predict": True,
            "inpaint_mask_list": [mask],
        })

        self.assertFalse(widget.ids["switch_details"].active)
        self.assertEqual(widget.ids["switch_inpaint"].state, "down")
        # predict is a durable one-shot flag, not state-bound: set2widget leaves the
        # momentary Erase button untouched.
        self.assertEqual(widget.ids["button_inpaint_predict"].state, "normal")
        self.assertEqual(effect.mask_editor.added_masks, [((1, 2, 3, 4), "mask")])
        self.assertEqual(effect.mask_editor.delay_updates, 1)

        param = {}
        widget.ids["switch_details"].active = True
        widget.ids["switch_inpaint"].state = "normal"
        widget.ids["button_inpaint_predict"].state = "down"
        old_editor = effect.mask_editor
        effect.set2param(param, widget)

        self.assertEqual(param["switch_details"], True)
        self.assertEqual(param["inpaint"], False)
        # set2param must NOT resurrect the predict flag from the transient button state.
        self.assertNotIn("inpaint_predict", param)
        self.assertEqual(preview.removed, [old_editor])
        self.assertIsNone(effect.mask_editor)
        self.assertEqual(exit_calls, ["inpaint"])

    def test_patchmatch_state_bindings_close_patchmatch_editor(self):
        effect = effects.PatchmatchInpaintEffect()
        effect.mask_editor = DummyMaskEditor()
        preview = DummyPreviewWidget()
        exit_calls = []
        widget = SimpleNamespace(
            ids={
                "switch_details": DummySwitch(active=True),
                "switch_patchmatch_inpaint": SimpleNamespace(state="normal"),
                "button_patchmatch_inpaint_predict": SimpleNamespace(state="down"),
                "preview_widget": preview,
            },
            exit_mask1_full_preview_mode=lambda name: exit_calls.append(name),
        )
        old_editor = effect.mask_editor
        param = {}

        effect.set2param(param, widget)

        self.assertTrue(param["switch_details"])
        self.assertFalse(param["patchmatch_inpaint"])
        # set2param must NOT resurrect the predict flag from the transient button state.
        self.assertNotIn("patchmatch_inpaint_predict", param)
        self.assertEqual(preview.removed, [old_editor])
        self.assertIsNone(effect.mask_editor)
        self.assertEqual(exit_calls, ["patchmatch_inpaint"])

    def test_distortion_binding_runs_painter_hooks_after_value_transfer(self):
        effect = TestableDistortionEffect()
        effect.distortion_painter = DummyDistortionPainter()
        widget = SimpleNamespace(ids={
            "switch_distortion": DummySwitch(enabled=None),
            "slider_distortion_brush_size": DummySlider(),
            "slider_distortion_strength": DummySlider(),
            "effects": SimpleNamespace(current_tab=SimpleNamespace(text="Li")),
        })
        recorded = [("stroke", 1)]

        effect.set2widget(widget, {
            "switch_distortion": False,
            "distortion_recorded": recorded,
            "distortion_brush_size": 12,
            "distortion_strength": 34,
        })

        self.assertFalse(widget.ids["switch_distortion"].enabled)
        self.assertEqual(widget.ids["slider_distortion_brush_size"].value, 12)
        self.assertEqual(widget.ids["slider_distortion_strength"].value, 34)
        self.assertEqual(effect.distortion_painter.recorded, recorded)
        self.assertEqual(effect.distortion_painter.remap_calls, 1)

        widget.ids["switch_distortion"].enabled = True
        widget.ids["slider_distortion_brush_size"].value = 56
        widget.ids["slider_distortion_strength"].value = 78
        param = {}
        effect.distortion_painter = None
        widget._sync_effect_editors = lambda: effect._open_distortion_painter(param, widget)

        effect.set2param(param, widget)

        self.assertEqual(
            param,
            {
                "switch_distortion": True,
                "distortion_brush_size": 56,
                "distortion_strength": 78,
            },
        )
        self.assertIsNotNone(effect.distortion_painter)
        self.assertEqual(effect.distortion_painter.brush_size, 56)
        self.assertEqual(effect.distortion_painter.strength, 78)

    def test_geometry_binding_runs_editor_hooks_after_value_transfer(self):
        effect = TestableGeometryEffect()
        effect.geometry_editor = LensDistortionWidget()
        sync_calls = []
        ids = DummyIds({
            "slider_rotation": DummySlider(),
            "switch_distortion_correction": DummySwitch(),
            "slider_lens_distortion_strength": DummySlider(),
            "slider_lens_distortion_scale": DummySlider(),
            "slider_correct_trapezoid_h": DummySlider(),
            "slider_correct_trapezoid_v": DummySlider(),
            "slider_focal_length": DummySlider(),
            "btn_lens": SimpleNamespace(state="down", text="Lens"),
            "btn_trapezoid": SimpleNamespace(state="normal", text="Trapezoid"),
            "btn_four_points": SimpleNamespace(state="normal", text="Four Points"),
            "btn_mesh": SimpleNamespace(state="normal", text="Mesh"),
            "btn_lines": SimpleNamespace(state="normal", text="Lines"),
        })
        widget = SimpleNamespace(
            ids=ids,
            sync_distortion_mode_sliders=lambda: sync_calls.append("sync"),
        )

        effect.set2widget(widget, {
            "rotation": 11,
            "switch_distortion_correction": False,
            "lens_distortion_strength": 22,
            "lens_distortion_scale": 33,
            "correct_horizontal": 44,
            "correct_vertical": 55,
            "focal_length": 66,
        })

        self.assertEqual(ids["slider_rotation"].value, 11)
        self.assertFalse(ids["switch_distortion_correction"].active)
        self.assertEqual(ids["slider_lens_distortion_strength"].value, 22)
        self.assertEqual(effect.geometry_editor.set_params["correct_vertical"], 55)
        self.assertEqual(sync_calls, ["sync"])

        ids["slider_rotation"].value = 1
        ids["switch_distortion_correction"].active = False
        ids["slider_lens_distortion_strength"].value = 2
        ids["slider_lens_distortion_scale"].value = 3
        ids["slider_correct_trapezoid_h"].value = 4
        ids["slider_correct_trapezoid_v"].value = 5
        ids["slider_focal_length"].value = 6
        param = {
            "original_img_size": (100, 80),
            "crop_rect": (0.0, 0.0, 1.0, 0.8),
        }

        effect.set2param(param, widget)

        self.assertEqual(param["rotation"], 1)
        self.assertFalse(param["switch_distortion_correction"])
        self.assertEqual(param["lens_distortion_strength"], 2)
        self.assertEqual(param["lens_distortion_scale"], 3)
        self.assertEqual(param["correct_horizontal"], 12)
        self.assertIn("matrix", param)
        self.assertEqual(effect.opened_editor[0], "Lens")

    def test_parent_effect_hooks_delegate_to_child_effects(self):
        curves = effects.CurvesEffect()
        curves_widget = SimpleNamespace(ids=DummyIds({
            "switch_tone_curves": DummySwitch(),
            "switch_color_gradings": DummySwitch(),
            "tonecurve": DummyPointList(),
            "tonecurve_red": DummyPointList(),
            "tonecurve_green": DummyPointList(),
            "tonecurve_blue": DummyPointList(),
            "grading1": DummyPointList(),
            "grading2": DummyPointList(),
            "grading1_color_picker": DummyColorPicker(),
            "grading2_color_picker": DummyColorPicker(),
        }))
        curves.set2widget(curves_widget, {
            "switch_tone_curves": False,
            "switch_color_gradings": True,
            "tonecurve": [[0, 0], [1, 1]],
            "tonecurve_red": [[0, 0], [1, 1]],
            "tonecurve_green": [[0, 0], [1, 1]],
            "tonecurve_blue": [[0, 0], [1, 1]],
            "grading1": [[0, 0.5], [1, 0.5]],
            "grading1_hue": 10,
            "grading1_lum": 60,
            "grading1_sat": 20,
            "grading2": [[0, 0.4], [1, 0.6]],
            "grading2_hue": 30,
            "grading2_lum": 40,
            "grading2_sat": 50,
        })
        self.assertFalse(curves_widget.ids["switch_tone_curves"].active)
        self.assertEqual(curves_widget.ids["tonecurve"].point_list, [[0, 0], [1, 1]])
        self.assertEqual(curves_widget.ids["grading1_color_picker"].set_calls, [[10, 60, 20]])

        curves_widget.ids["switch_tone_curves"].active = True
        curves_widget.ids["switch_color_gradings"].active = False
        curves_widget.ids["tonecurve"].point_list = [[0, 0.2], [1, 0.8]]
        curves_widget.ids["grading1_color_picker"].ids["slider_hue"].value = 1
        curves_widget.ids["grading1_color_picker"].ids["slider_lum"].value = 2
        curves_widget.ids["grading1_color_picker"].ids["slider_sat"].value = 3
        param = {}
        curves.set2param(param, curves_widget)
        self.assertTrue(param["switch_tone_curves"])
        self.assertFalse(param["switch_color_gradings"])
        self.assertEqual(param["tonecurve"], [[0, 0.2], [1, 0.8]])
        self.assertEqual(param["grading1_hue"], 1)

        vs = effects.VSandSaturationEffect()
        vs_widget = SimpleNamespace(ids=DummyIds({
            "switch_color_curves": DummySwitch(),
            "switch_saturation": DummySwitch(),
            "HuevsHue": DummyPointList(),
            "HuevsLum": DummyPointList(),
            "HuevsSat": DummyPointList(),
            "LumvsLum": DummyPointList(),
            "LumvsSat": DummyPointList(),
            "SatvsLum": DummyPointList(),
            "SatvsSat": DummyPointList(),
            "slider_saturation": DummySlider(),
            "slider_vibrance": DummySlider(),
        }))
        vs.set2widget(vs_widget, {
            "switch_color_curves": False,
            "switch_saturation": True,
            "HuevsHue": [[0, 0.5], [1, 0.5]],
            "HuevsLum": [[0, 0.5], [1, 0.5]],
            "HuevsSat": [[0, 0.5], [1, 0.5]],
            "LumvsLum": [[0, 0.5], [1, 0.5]],
            "LumvsSat": [[0, 0.5], [1, 0.5]],
            "SatvsLum": [[0, 0.5], [1, 0.5]],
            "SatvsSat": [[0, 0.5], [1, 0.5]],
            "saturation": 7,
            "vibrance": 8,
        })
        self.assertFalse(vs_widget.ids["switch_color_curves"].active)
        self.assertEqual(vs_widget.ids["HuevsHue"].point_list, [[0, 0.5], [1, 0.5]])
        self.assertEqual(vs_widget.ids["slider_saturation"].value, 7)

        vs_widget.ids["switch_color_curves"].active = True
        vs_widget.ids["switch_saturation"].active = False
        vs_widget.ids["HuevsHue"].point_list = [[0, 0.1], [1, 0.9]]
        vs_widget.ids["slider_saturation"].value = 9
        vs_widget.ids["slider_vibrance"].value = 10
        param = {}
        vs.set2param(param, vs_widget)
        self.assertTrue(param["switch_color_curves"])
        self.assertFalse(param["switch_saturation"])
        self.assertEqual(param["HuevsHue"], [[0, 0.1], [1, 0.9]])
        self.assertEqual(param["saturation"], 9)

    def test_param_bindings_round_trip(self):
        cases = (
            (
                effects.SubpixelShiftEffect,
                {"switch_details": False, "subpixel_shift": True},
                {"switch_details": True, "subpixel_shift": False},
            ),
            (
                effects.ExposureEffect,
                {"switch_exposure_contrast": False, "exposure": 1.25},
                {"switch_exposure_contrast": True, "exposure": -0.75},
            ),
            (
                effects.LightNoiseReductionEffect,
                {
                    "switch_light_noise_reduction": False,
                    "light_noise_reduction": 12,
                    "light_color_noise_reduction": 34,
                },
                {
                    "switch_light_noise_reduction": True,
                    "light_noise_reduction": 4,
                    "light_color_noise_reduction": 5,
                },
            ),
            (
                effects.LUTEffect,
                {
                    "switch_lut": False,
                    "lut_name": "warm.cube",
                    "lut_intensity": 80,
                    "lut_to_log": "LogC",
                },
                {
                    "switch_lut": True,
                    "lut_name": "cool.cube",
                    "lut_intensity": 60,
                    "lut_to_log": "None",
                },
            ),
            (
                effects.CrossFilterEffect,
                {
                    "switch_cross_filter": False,
                    "cross_filter_num_points": 4,
                    "cross_filter_length": 1200,
                    "cross_filter_angle": 15,
                    "cross_filter_threshold": 44,
                    "cross_filter_intensity": 22,
                    "cross_filter_spectral": 33,
                    "cross_filter_thickness": 2,
                    "cross_filter_distance": 80,
                    "cross_filter_random": 12,
                },
                {
                    "switch_cross_filter": True,
                    "cross_filter_num_points": 5,
                    "cross_filter_length": 1300,
                    "cross_filter_angle": 30,
                    "cross_filter_threshold": 55,
                    "cross_filter_intensity": 25,
                    "cross_filter_spectral": 35,
                    "cross_filter_thickness": 3,
                    "cross_filter_distance": 90,
                    "cross_filter_random": 20,
                },
            ),
            (
                effects.OrtonEffect,
                {
                    "switch_orton_effect": False,
                    "orton_radius": 20,
                    "orton_opacity": 40,
                    "orton_intensity": 60,
                },
                {
                    "switch_orton_effect": True,
                    "orton_radius": 25,
                    "orton_opacity": 45,
                    "orton_intensity": 65,
                },
            ),
            (
                effects.GlowEffect,
                {
                    "switch_glow_effect": False,
                    "glow_black": 10,
                    "glow_gauss": 20,
                    "glow_opacity": 30,
                },
                {
                    "switch_glow_effect": True,
                    "glow_black": 15,
                    "glow_gauss": 25,
                    "glow_opacity": 35,
                },
            ),
            (
                effects.RemoveChromaticAberrationEffect,
                {
                    "switch_fringe_removal": False,
                    "rca_enabled": True,
                    "rca_purple_amount": 21,
                    "rca_green_amount": 22,
                    "rca_fringe_width": 23,
                    "rca_edge_threshold": 24,
                },
                {
                    "switch_fringe_removal": True,
                    "rca_enabled": False,
                    "rca_purple_amount": 31,
                    "rca_green_amount": 32,
                    "rca_fringe_width": 33,
                    "rca_edge_threshold": 34,
                },
            ),
            (
                effects.FaceEffect,
                {
                    "switch_face": False,
                    "jawline_scale": 1,
                    "jaw_scale": 2,
                    "left_eye_scale": 3,
                    "right_eye_scale": 4,
                    "lips_scale": 5,
                },
                {
                    "switch_face": True,
                    "jawline_scale": 6,
                    "jaw_scale": 7,
                    "left_eye_scale": 8,
                    "right_eye_scale": 9,
                    "lips_scale": 10,
                },
            ),
            (
                effects.DehazeEffect,
                {"switch_precence": False, "dehaze": 11},
                {"switch_precence": True, "dehaze": 12},
            ),
            (
                effects.MicroContrastEffect,
                {"switch_precence": False, "microcontrast": 13},
                {"switch_precence": True, "microcontrast": 14},
            ),
            (
                effects.ToneEffect,
                {
                    "switch_tone": False,
                    "shadow": 1,
                    "highlight": 2,
                    "midtone": 3,
                    "white": 4,
                    "black": 5,
                },
                {
                    "switch_tone": True,
                    "shadow": 6,
                    "highlight": 7,
                    "midtone": 8,
                    "white": 9,
                    "black": 10,
                },
            ),
            (
                effects.ColorSeparationEffect,
                {
                    "switch_global": False,
                    "shadow_chroma_clean": 0.1,
                    "shadow_chroma_threshold": 0.2,
                    "color_separation": 0.3,
                    "chroma_clarity": 0.4,
                    "color_density": 0.5,
                    "subtractive_saturation": 0.6,
                    "detail_tonemap": 0.7,
                },
                {
                    "switch_global": True,
                    "shadow_chroma_clean": 1.1,
                    "shadow_chroma_threshold": 1.2,
                    "color_separation": 1.3,
                    "chroma_clarity": 1.4,
                    "color_density": 1.5,
                    "subtractive_saturation": 1.6,
                    "detail_tonemap": 1.7,
                },
            ),
            (
                effects.LevelEffect,
                {
                    "switch_level": False,
                    "black_level": 2,
                    "mid_level": 128,
                    "white_level": 250,
                },
                {
                    "switch_level": True,
                    "black_level": 3,
                    "mid_level": 129,
                    "white_level": 251,
                },
            ),
            (
                effects.CLAHEEffect,
                {"switch_precence": False, "clahe": 15},
                {"switch_precence": True, "clahe": 16},
            ),
            (
                effects.SaturationEffect,
                {
                    "switch_saturation": False,
                    "saturation": 17,
                    "vibrance": 18,
                },
                {
                    "switch_saturation": True,
                    "saturation": 19,
                    "vibrance": 20,
                },
            ),
            (
                effects.UnsharpMaskEffect,
                {
                    "switch_unsharp_mask": False,
                    "unsharp_mask_amount": 21,
                    "unsharp_mask_sigma": 22,
                },
                {
                    "switch_unsharp_mask": True,
                    "unsharp_mask_amount": 23,
                    "unsharp_mask_sigma": 24,
                },
            ),
            (
                effects.GrainEffect,
                {
                    "switch_grain": False,
                    "grain_amount": 11,
                    "grain_size": 22,
                    "grain_roughness": 33,
                    "grain_shadow": 44,
                    "grain_highlight": 55,
                    "grain_color": 66,
                    "grain_seed": 77,
                },
                {
                    "switch_grain": True,
                    "grain_amount": 1,
                    "grain_size": 2,
                    "grain_roughness": 3,
                    "grain_shadow": 4,
                    "grain_highlight": 5,
                    "grain_color": 6,
                    "grain_seed": 7,
                },
            ),
            (
                effects.VignetteEffect,
                {
                    "switch_vignette": False,
                    "vignette_intensity": 10,
                    "vignette_radius_percent": 70,
                    "vignette_softness": 60,
                },
                {
                    "switch_vignette": True,
                    "vignette_intensity": -20,
                    "vignette_radius_percent": 85,
                    "vignette_softness": 90,
                },
            ),
            (
                effects.LensSimulatorEffect,
                {
                    "switch_lens_simulator": False,
                    "coating_preset": "Vintage",
                    "coating_strength": 80,
                    "coating_light": 1.5,
                    "lateral_ca": 0.1,
                    "longitudinal_ca": 0.2,
                    "spherical_ca": 0.3,
                    "lens_focus_depth": 0.4,
                    "lens_aperture": 2.0,
                },
                {
                    "switch_lens_simulator": True,
                    "coating_preset": "Modern",
                    "coating_strength": 70,
                    "coating_light": 1.2,
                    "lateral_ca": 0.4,
                    "longitudinal_ca": 0.5,
                    "spherical_ca": 0.6,
                    "lens_focus_depth": 0.7,
                    "lens_aperture": 2.8,
                },
            ),
            (
                effects.FilmSimulationEffect,
                {
                    "switch_film_simulation": False,
                    "film_mode": "Off",
                    "film_latitude": 55,
                    "film_contrast": 50,
                    "film_color_bias": 0,
                    "film_color_drift": 0,
                    "film_dye_purity": 75,
                    "film_layer_crosstalk": 30,
                    "film_halation": 0,
                    "film_aging": 0,
                    "film_intensity": 75,
                },
                {
                    "switch_film_simulation": True,
                    "film_mode": "Slide",
                    "film_latitude": 65,
                    "film_contrast": 58,
                    "film_color_bias": 10,
                    "film_color_drift": 35,
                    "film_dye_purity": 88,
                    "film_layer_crosstalk": 18,
                    "film_halation": 12,
                    "film_aging": 6,
                    "film_intensity": 60,
                },
            ),
            (
                effects.TonecurveEffect,
                {"switch_tone_curves": False, "tonecurve": [[0, 0], [1, 1]]},
                {"switch_tone_curves": True, "tonecurve": [[0, 0.1], [1, 0.9]]},
            ),
            (
                effects.TonecurveRedEffect,
                {"switch_tone_curves": False, "tonecurve_red": [[0, 0], [1, 1]]},
                {"switch_tone_curves": True, "tonecurve_red": [[0, 0.2], [1, 0.8]]},
            ),
            (
                effects.HuevsHueEffect,
                {"switch_color_curves": False, "HuevsHue": [[0, 0.5], [1, 0.5]]},
                {"switch_color_curves": True, "HuevsHue": [[0, 0.4], [1, 0.6]]},
            ),
            (
                effects.SatvsSatEffect,
                {"switch_color_curves": False, "SatvsSat": [[0, 0.5], [1, 0.5]]},
                {"switch_color_curves": True, "SatvsSat": [[0, 0.3], [1, 0.7]]},
            ),
        )

        for effect_cls, override_param, widget_values in cases:
            with self.subTest(effect=effect_cls.__name__):
                self._exercise_binding_round_trip(effect_cls, override_param, widget_values)


if __name__ == "__main__":
    unittest.main()
