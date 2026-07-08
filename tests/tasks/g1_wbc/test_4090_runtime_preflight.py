from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PREFLIGHT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "check_g1_wbc_4090_runtime.py"
)


def load_preflight():
    spec = importlib.util.spec_from_file_location("g1_wbc_4090_preflight", PREFLIGHT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load preflight from {PREFLIGHT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RuntimePreflightTest(unittest.TestCase):
    def test_passes_with_single_visible_4090_and_runtime_imports(self) -> None:
        preflight = load_preflight()

        report = preflight.build_report(
            required_gpu_name_fragment="4090",
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            import_module=_fake_import_module(),
            run_command=_fake_nvidia_smi(
                "0, NVIDIA GeForce RTX 4090, 49140 MiB, 779 MiB, 580.126.09\n"
            ),
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["cuda_visible_devices"], ["0"])
        self.assertEqual(report["imports"]["jax"]["version"], "0.test")
        self.assertEqual(report["jax_devices"][0]["device_kind"], "NVIDIA GeForce RTX 4090")
        self.assertEqual(report["nvidia_smi"]["gpus"][0]["name"], "NVIDIA GeForce RTX 4090")

    def test_fails_without_single_visible_gpu_even_when_imports_work(self) -> None:
        preflight = load_preflight()

        report = preflight.build_report(
            required_gpu_name_fragment="4090",
            environ={"CUDA_VISIBLE_DEVICES": ""},
            import_module=_fake_import_module(),
            run_command=_fake_nvidia_smi(
                "0, NVIDIA GeForce RTX 4090, 49140 MiB, 779 MiB, 580.126.09\n"
            ),
        )

        self.assertFalse(report["passed"])
        self.assertIn("cuda_visible_devices_count", report["failures"])

    def test_records_missing_jax_without_raising(self) -> None:
        preflight = load_preflight()

        def fake_import_module(name):
            if name == "jax":
                raise ModuleNotFoundError(name)
            return _fake_import_module()(name)

        report = preflight.build_report(
            required_gpu_name_fragment="4090",
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            import_module=fake_import_module,
            run_command=_fake_nvidia_smi(
                "0, NVIDIA GeForce RTX 4090, 49140 MiB, 779 MiB, 580.126.09\n"
            ),
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["imports"]["jax"]["available"])
        self.assertIn("import:jax", report["failures"])


def _fake_import_module():
    jax = SimpleNamespace(
        __version__="0.test",
        devices=lambda: [
            SimpleNamespace(
                id=0,
                platform="gpu",
                device_kind="NVIDIA GeForce RTX 4090",
            )
        ],
    )
    modules = {
        "torch": SimpleNamespace(__version__="2.test"),
        "jax": jax,
        "jax.numpy": SimpleNamespace(__name__="jax.numpy"),
        "mujoco": SimpleNamespace(__version__="3.7.0"),
        "mujoco.mjx": SimpleNamespace(__name__="mujoco.mjx"),
        "mujoco_warp": SimpleNamespace(__version__="3.7.0.1"),
        "warp": SimpleNamespace(__version__="1.12.1"),
    }

    def fake_import_module(name):
        return modules[name]

    return fake_import_module


def _fake_nvidia_smi(stdout: str):
    def fake_run_command(command):
        self = None
        del self
        return {
            "command": command,
            "returncode": 0,
            "stdout": stdout,
            "stderr": "",
            "error": None,
        }

    return fake_run_command


if __name__ == "__main__":
    unittest.main()
