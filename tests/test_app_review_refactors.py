import pathlib
import sys

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import ast
import copy
import pathlib
import unittest
from unittest.mock import patch

from effect_backends.backend_utils import (
    backend_preference,
    import_error_detail,
    native_backend_enabled,
    optional_backend,
    strict_enabled,
)
from utils import rating_io


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
EFFECTS_PATH = PROJECT_ROOT / "effects.py"
FACER_HELPER_PATH = PROJECT_ROOT / "helpers" / "facer_helper.py"
DISTORTION_PAINTER_PATH = PROJECT_ROOT / "widgets" / "distortion_painter.py"


def _load_class_function(path, class_name, function_name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == function_name:
                    return child
    raise AssertionError(f"{class_name}.{function_name} was not found")


def _load_function(path, function_name):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise AssertionError(f"{function_name} was not found")


def _compile_function_from_node(node):
    node = copy.deepcopy(node)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "<ast>", "exec"), namespace)
    return namespace[node.name]


class AppReviewRefactorsTest(unittest.TestCase):
    def test_backend_utils_normalize_common_adapter_decisions(self):
        with patch.dict("os.environ", {"BACKEND": " Reference ", "STRICT": "yes"}):
            self.assertEqual("reference", backend_preference("BACKEND"))
            self.assertTrue(strict_enabled("STRICT"))
        self.assertFalse(native_backend_enabled(object(), "reference"))
        self.assertTrue(native_backend_enabled(object(), "auto"))
        missing, error = optional_backend("effect_backends", "_definitely_missing_backend")
        self.assertIsNone(missing)
        self.assertIn("_definitely_missing_backend", import_error_detail(error))

    def test_crop_aspect_ratio_parser_does_not_use_eval(self):
        node = _load_class_function(EFFECTS_PATH, "CropEffect", "_param_to_aspect_ratio")
        self.assertFalse(
            any(isinstance(call.func, ast.Name) and call.func.id == "eval" for call in ast.walk(node) if isinstance(call, ast.Call))
        )
        method = _compile_function_from_node(node)

        class Dummy:
            _param_to_aspect_ratio = method

            def _get_param(self, param, key):
                return param.get(key)

        dummy = Dummy()
        self.assertEqual(0, dummy._param_to_aspect_ratio({"aspect_ratio": "None"}))
        self.assertAlmostEqual(16 / 9, dummy._param_to_aspect_ratio({"aspect_ratio": "16/9"}))
        self.assertEqual(1.25, dummy._param_to_aspect_ratio({"aspect_ratio": "1.25"}))
        self.assertEqual(0, dummy._param_to_aspect_ratio({"aspect_ratio": "__import__('os').system('echo bad')"}))
        self.assertEqual(0, dummy._param_to_aspect_ratio({"aspect_ratio": "1/0"}))

    def test_mutable_default_arguments_are_removed_from_review_targets(self):
        draw_face_mask = _load_function(FACER_HELPER_PATH, "draw_face_mask")
        distortion_init = _load_class_function(DISTORTION_PAINTER_PATH, "DistortionCanvas", "__init__")

        for node in (draw_face_mask, distortion_init):
            for default in node.args.defaults:
                self.assertNotIsInstance(default, (ast.List, ast.Dict, ast.Set))

    def test_rating_io_uses_contract_specific_pmck_reader_name(self):
        self.assertFalse(hasattr(rating_io, "read_pmck_dict"))
        self.assertTrue(hasattr(rating_io, "read_pmck_dict_or_none"))


if __name__ == "__main__":
    unittest.main()
