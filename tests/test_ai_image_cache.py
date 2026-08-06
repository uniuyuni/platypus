import pathlib
import sys
import unittest

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import effects
import pipeline
from cores.ai_image_cache import AIImageCache


class AIImageCacheTest(unittest.TestCase):
    def test_depth_map_is_reused_for_same_key(self):
        cache = AIImageCache()
        calls = []

        def compute():
            calls.append(1)
            return np.array([[0.25, 0.75]], dtype=np.float32)

        key = ["mask2-ai-cache", 1, "depth", [2, 1], 2]
        first = cache.get_depth_map(key, compute)
        second = cache.get_depth_map(list(key), compute)

        self.assertIs(first, second)
        self.assertEqual(1, len(calls))

    def test_depth_map_round_trips_through_serialized_cache(self):
        cache = AIImageCache()
        depth = np.array([[0.1, 0.9]], dtype=np.float32)
        key = ["mask2-ai-cache", 1, "depth", [2, 1], 2]
        cache.get_depth_map(key, lambda: depth)

        restored = AIImageCache(cache.serialize())
        actual = restored.get_depth_map(key, lambda: self.fail("cache miss"))

        np.testing.assert_allclose(depth, actual)

    def test_clear_reports_removed_depth_map(self):
        cache = AIImageCache()
        cache.get_depth_map(["depth"], lambda: np.zeros((2, 3), dtype=np.float32))

        result = cache.clear()

        self.assertEqual(1, result["ai_image_cache_entries"])
        self.assertGreater(result["ai_image_cache_bytes"], 0)

    def test_derived_depth_map_is_reused_for_same_key(self):
        cache = AIImageCache()
        calls = []

        def compute():
            calls.append(1)
            return np.array([[1, 2]], dtype=np.float32)

        first = cache.get_derived_depth_map(["depth-current", [2, 1]], compute)
        second = cache.get_derived_depth_map(["depth-current", [2, 1]], compute)

        self.assertIs(first, second)
        self.assertEqual(1, len(calls))

    def test_result_is_not_stored_after_cache_generation_changes(self):
        cache = AIImageCache()
        first = np.array([[1]], dtype=np.float32)
        second = np.array([[2]], dtype=np.float32)
        calls = []

        def stale_compute():
            calls.append("stale")
            cache.clear()
            return first

        result = cache.get_depth_map(["depth"], stale_compute)
        self.assertIs(result, first)

        cached = cache.get_depth_map(["depth"], lambda: calls.append("fresh") or second)

        self.assertIs(cached, second)
        self.assertEqual(["stale", "fresh"], calls)

    def test_effect_config_depth_getter_returns_original_space_without_exposing_source(self):
        from cores.mask2 import inference_runtime

        cache = AIImageCache()
        efconfig = effects.EffectConfig()
        source = np.zeros((2, 3, 3), dtype=np.float32)
        depth = np.array([[0.2, 0.8, 0.6], [0.1, 0.4, 0.9]], dtype=np.float32)
        param = {"original_img_size": [3, 2]}
        calls = []
        original_predict = inference_runtime.predict_depth_map

        def fake_predict(image):
            calls.append(image)
            return depth

        try:
            inference_runtime.predict_depth_map = fake_predict
            pipeline._install_ai_depth_map_getter(
                efconfig,
                source,
                param,
                ai_image_cache=cache,
            )

            first = efconfig.get_ai_depth_map(space="original")
            second = efconfig.get_ai_depth_map(space="original")
        finally:
            inference_runtime.predict_depth_map = original_predict

        self.assertFalse(hasattr(efconfig, "original_image_rgb"))
        self.assertIs(first, second)
        self.assertEqual(1, len(calls))
        self.assertIs(source, calls[0])

    def test_effect_config_depth_getter_returns_current_cropped_space_and_caches_it(self):
        from cores.mask2 import inference_runtime

        cache = AIImageCache()
        source = np.zeros((4, 4, 3), dtype=np.float32)
        depth = np.arange(16, dtype=np.float32).reshape(4, 4)
        param = {
            "original_img_size": [4, 4],
            "crop_rect": [1, 1, 3, 3],
        }
        calls = []
        original_predict = inference_runtime.predict_depth_map

        def fake_predict(image):
            calls.append(image)
            return depth

        try:
            inference_runtime.predict_depth_map = fake_predict
            efconfig = effects.EffectConfig()
            pipeline._install_ai_depth_map_getter(efconfig, source, param, ai_image_cache=cache)
            pipeline._set_ai_depth_map_current_context(
                efconfig,
                space="current",
                mode="preview",
                disp_info=(1, 1, 2, 2, 1.0),
                crop_rect=param["crop_rect"],
                texture_width=2,
                texture_height=2,
                click_x=0,
                click_y=0,
                is_zoomed=False,
                center_pos=None,
                zoom_ratio=1.0,
                deferred_geometry=None,
            )

            first = efconfig.get_ai_depth_map()

            efconfig2 = effects.EffectConfig()
            pipeline._install_ai_depth_map_getter(efconfig2, source, param, ai_image_cache=cache)
            pipeline._set_ai_depth_map_current_context(
                efconfig2,
                space="current",
                mode="preview",
                disp_info=(1, 1, 2, 2, 1.0),
                crop_rect=param["crop_rect"],
                texture_width=2,
                texture_height=2,
                click_x=0,
                click_y=0,
                is_zoomed=False,
                center_pos=None,
                zoom_ratio=1.0,
                deferred_geometry=None,
            )
            second = efconfig2.get_ai_depth_map()
        finally:
            inference_runtime.predict_depth_map = original_predict

        np.testing.assert_allclose(np.array([[5, 6], [9, 10]], dtype=np.float32), first)
        self.assertIs(first, second)
        self.assertEqual(1, len(calls))


if __name__ == "__main__":
    unittest.main()
