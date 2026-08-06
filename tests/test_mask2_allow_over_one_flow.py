from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import effects
from cores.mask2 import extended_params, hls_mask


MAIN_KV_PATH = ROOT / "main.kv"
MAIN_PY_PATH = ROOT / "main.py"


class _RangeSlider:
    def __init__(self, values):
        self.value = values[0] if values else 0
        self.ids = {"slider": SimpleNamespace(values=list(values))}


def _control(active=False, value=0, text=""):
    return SimpleNamespace(active=active, value=value, text=text)


def _mask2_widget(*, allow_over_one):
    return SimpleNamespace(
        ids={
            "switch_mask2_settings": _control(active=True),
            "checkbox_mask2_invert": _control(active=False),
            "checkbox_mask2_allow_over_one": _control(active=allow_over_one),
            "spinner_mask2_blend_mode": _control(text="Normal"),
            "switch_mask2_depth": _control(active=False),
            "slider_mask2_depth_min": _control(value=0),
            "slider_mask2_depth_max": _control(value=255),
            "switch_mask2_hue": _control(active=False),
            "slider_mask2_hue_distance": _control(value=179),
            "slider_mask2_hue_range": _RangeSlider([0, 359]),
            "switch_mask2_lum": _control(active=False),
            "slider_mask2_lum_distance": _control(value=255),
            "slider_mask2_lum_range": _RangeSlider([0, 255]),
            "switch_mask2_sat": _control(active=False),
            "slider_mask2_sat_distance": _control(value=255),
            "slider_mask2_sat_range": _RangeSlider([0, 255]),
            "switch_mask2_options": _control(active=True),
            "slider_mask2_blur": _control(value=0),
            "slider_mask2_depth_balance": _control(value=0),
            "slider_mask2_open_space": _control(value=0),
            "slider_mask2_close_space": _control(value=0),
            "slider_mask2_freedraw_brush_size": _control(value=300),
            "slider_mask2_freedraw_brush_hardness": _control(value=100),
            "checkbox_mask2_polyline_fill": _control(active=True),
            "switch_mask2_quick_select": _control(active=True),
            "spinner_mask2_edge_refine_mode": _control(text="Off"),
            "slider_mask2_edge_refine_radius": _control(value=0),
            "slider_mask2_edge_refine_strength": _control(value=0),
            "slider_mask2_edge_refine_bias": _control(value=0),
            "switch_mask2_draw_effects": _control(active=True),
            "slider_mask2_color_dodge": _control(value=0),
            "slider_mask2_color_burn": _control(value=0),
            "slider_mask2_mix_black": _control(value=0),
            "slider_mask2_mix_white": _control(value=0),
            "slider_mask2_skin_smooth_amount": _control(value=0),
            "slider_mask2_skin_smooth_radius_bias": _control(value=0),
            "switch_mask2_face": _control(active=True),
            "checkbox_mask2_face_face": _control(active=True),
            "checkbox_mask2_face_brows": _control(active=True),
            "checkbox_mask2_face_eyes": _control(active=True),
            "checkbox_mask2_face_nose": _control(active=True),
            "checkbox_mask2_face_mouth": _control(active=True),
            "checkbox_mask2_face_lips": _control(active=True),
        }
    )


class _Ctx:
    def __init__(self, rgb):
        self._hls = hls_mask.rgb_to_selection_hls(rgb)

    def get_crop_image_hls(self):
        return self._hls


class Mask2AllowOverOneFlowTest(unittest.TestCase):
    def test_mask2_allow_over_one_checkbox_is_user_editable(self):
        kv = MAIN_KV_PATH.read_text(encoding="utf-8")
        block_start = kv.index("id: checkbox_mask2_allow_over_one")
        block_end = kv.index("id: checkbox_mask2_allow_under_zero")
        over_one_block = kv[block_start:block_end]

        self.assertNotIn("disabled: True", over_one_block)

    def test_mask2_allow_over_one_is_not_forced_disabled_at_runtime(self):
        source = MAIN_PY_PATH.read_text(encoding="utf-8")

        self.assertNotIn("'checkbox_mask2_allow_over_one'", source)
        self.assertIn("'checkbox_mask2_allow_under_zero'", source)

    def test_mask2_set2param_preserves_allow_over_one_selection(self):
        param = {}

        effects.Mask2Effect().set2param(param, _mask2_widget(allow_over_one=True))

        self.assertTrue(param["mask2_allow_over_one"])
        self.assertFalse(param["mask2_allow_under_zero"])

    def test_selection_hls_stores_hdr_brightness_as_l_times_gain(self):
        rgb = np.array([[[0.25, 0.5, 0.75], [1.4, 1.4, 1.4]]], dtype=np.float32)

        hls = hls_mask.rgb_to_selection_hls(rgb)

        expected_sdr_luminance = 0.2126 * 0.25 + 0.7152 * 0.5 + 0.0722 * 0.75
        self.assertAlmostEqual(float(hls[0, 0, 3]), expected_sdr_luminance, places=6)
        self.assertAlmostEqual(float(hls[0, 1, 3]), 1.4, places=6)

    def test_mask2_allow_over_one_selects_only_hdr_highlight_pixels(self):
        rgb = np.array(
            [[[0.8, 0.8, 0.8], [1.0, 1.0, 1.0], [1.2, 1.1, 0.9]]],
            dtype=np.float32,
        )
        mask = np.ones((1, 3), dtype=np.float32)
        param = effects.Mask2Effect.get_param_dict({})
        param.update(
            {
                "mask2_allow_over_one": True,
                "switch_mask2_depth": False,
                "switch_mask2_hue": False,
                "switch_mask2_lum": False,
                "switch_mask2_sat": False,
                "mask2_blur": 0,
            }
        )

        out = extended_params.apply_post_edge_params(_Ctx(rgb), param, mask, center_tcg=(0, 0))

        np.testing.assert_array_equal(out, np.array([[0.0, 0.0, 1.0]], dtype=np.float32))

    def test_mask2_settings_off_does_not_apply_hdr_highlight_gate(self):
        rgb = np.array([[[0.8, 0.8, 0.8], [1.2, 1.1, 0.9]]], dtype=np.float32)
        mask = np.ones((1, 2), dtype=np.float32)
        param = effects.Mask2Effect.get_param_dict({})
        param.update(
            {
                "switch_mask2_settings": False,
                "mask2_allow_over_one": True,
                "switch_mask2_depth": False,
                "switch_mask2_hue": False,
                "switch_mask2_lum": False,
                "switch_mask2_sat": False,
                "mask2_blur": 0,
            }
        )

        out = extended_params.apply_post_edge_params(_Ctx(rgb), param, mask, center_tcg=(0, 0))

        np.testing.assert_array_equal(out, mask)


if __name__ == "__main__":
    unittest.main()
