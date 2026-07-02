import importlib
import unittest
from unittest import mock

from spider.tasks.g1_wbc import mjx_runtime


class MjxRuntimeTest(unittest.TestCase):
    def test_runtime_status_reports_missing_modules_without_raising(self) -> None:
        real_import_module = importlib.import_module

        def fake_import_module(name, package=None):
            if name == "jax" or name == "mujoco.mjx":
                raise ModuleNotFoundError(name)
            return real_import_module(name, package)

        with mock.patch.object(importlib, "import_module", fake_import_module):
            status = mjx_runtime.probe_mjx_runtime()

        self.assertFalse(status.available)
        self.assertFalse(status.jax_available)
        self.assertFalse(status.mjx_available)
        self.assertIn("jax", str(status.error).lower())

    def test_require_mjx_runtime_raises_clear_error_when_unavailable(self) -> None:
        status = mjx_runtime.MjxRuntimeStatus(
            available=False,
            jax_available=False,
            mjx_available=False,
            jax_version=None,
            mujoco_version="3.7.0",
            mjx_module=None,
            visible_devices=(),
            error="No module named 'jax'",
        )

        with mock.patch.object(mjx_runtime, "probe_mjx_runtime", return_value=status):
            with self.assertRaisesRegex(RuntimeError, "--mpc-backend mjx") as cm:
                mjx_runtime.require_mjx_runtime()

        self.assertIn("No module named 'jax'", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
