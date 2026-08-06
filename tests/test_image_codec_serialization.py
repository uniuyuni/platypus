import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pathlib
import unittest

import numpy as np

import config
from utils import utils


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


class ImageCodecSerializationTest(unittest.TestCase):
    def tearDown(self):
        config._config = None

    def test_default_image_codec_version_is_radiance_codec(self):
        source = (PROJECT_ROOT / "config.py").read_text(encoding="utf-8")

        self.assertIn("'image_codec_version': 2", source)

    def test_radiance_codec_version2_roundtrips_float32_images_exactly(self):
        config._config = {"image_codec_version": 2}
        image = np.linspace(-1.0, 2.0, 19 * 17 * 3, dtype=np.float32).reshape(19, 17, 3)

        encoded = utils.convert_image_to_list(image)
        decoded = utils.convert_image_from_list(encoded)

        self.assertEqual(2, encoded["version"])
        self.assertEqual("radiance_codec", encoded["codec"])
        self.assertEqual("lossless", encoded["mode"])
        self.assertEqual("compact", encoded["preset"])
        self.assertEqual(image.shape, decoded.shape)
        self.assertEqual(np.float32, decoded.dtype)
        self.assertEqual(image.tobytes(), decoded.tobytes())

    def test_radiance_codec_version2_preserves_2d_shape(self):
        config._config = {"image_codec_version": 2}
        image = np.arange(23 * 11, dtype=np.float32).reshape(23, 11)

        encoded = utils.convert_image_to_list(image)
        decoded = utils.convert_image_from_list(encoded)

        self.assertEqual(2, encoded["version"])
        self.assertEqual(image.shape, decoded.shape)
        self.assertEqual(image.tobytes(), decoded.tobytes())

    def test_zstd_version1_still_roundtrips_for_compatibility(self):
        config._config = {"image_codec_version": 1}
        image = np.random.default_rng(3).standard_normal((8, 9, 3)).astype(np.float32)

        encoded = utils.convert_image_to_list(image)
        decoded = utils.convert_image_from_list(encoded)

        self.assertEqual(1, encoded["version"])
        self.assertEqual("zstd", encoded["codec"])
        self.assertEqual(image.shape, decoded.shape)
        self.assertEqual(image.tobytes(), decoded.tobytes())

    def test_pyinstaller_bundles_radiance_codec_dylib_explicitly(self):
        source = (PROJECT_ROOT / "scripts" / "build_macos_app_pyinstaller.py").read_text(encoding="utf-8")
        hook_source = (
            PROJECT_ROOT / "scripts" / "pyinstaller" / "rth_radiance_codec.py"
        ).read_text(encoding="utf-8")

        self.assertIn("def _radiance_codec_binary_args", source)
        self.assertIn("libradiance_codec.dylib", source)
        self.assertIn("args.extend(_radiance_codec_binary_args(root))", source)
        self.assertIn("rth_radiance_codec.py", source)
        self.assertIn("[\"--hidden-import\", \"radiance_codec\"]", source)
        self.assertNotIn("\"radiance_codec\", \"radiance_denoise\"", source)
        self.assertIn("RADIANCE_CODEC_LIBRARY", hook_source)
        self.assertIn("libradiance_codec.dylib", hook_source)


if __name__ == "__main__":
    unittest.main()
