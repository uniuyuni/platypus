import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest

from utils.envutils import env_flag


class EnvUtilsTests(unittest.TestCase):
    def test_env_flag_accepts_common_true_values(self):
        for value in ("1", "true", "TRUE", " yes ", "on"):
            with self.subTest(value=value):
                self.assertTrue(env_flag("FLAG", environ={"FLAG": value}))

    def test_env_flag_rejects_false_or_unknown_values(self):
        for value in ("", "0", "false", "no", "off", "maybe"):
            with self.subTest(value=value):
                self.assertFalse(env_flag("FLAG", environ={"FLAG": value}))

    def test_env_flag_uses_default_for_missing_values(self):
        self.assertTrue(env_flag("FLAG", default=True, environ={}))
        self.assertFalse(env_flag("FLAG", default=False, environ={}))


if __name__ == "__main__":
    unittest.main()
