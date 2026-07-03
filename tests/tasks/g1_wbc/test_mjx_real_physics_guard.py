import os
from types import SimpleNamespace
import unittest
from unittest import mock

from tests.tasks.g1_wbc import test_mjx_real_physics as real_physics_tests


class MjxRealPhysicsGuardTest(unittest.TestCase):
    def test_guard_skips_without_opt_in_before_probe(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(real_physics_tests, "probe_mjx_runtime") as probe,
        ):
            with self.assertRaisesRegex(unittest.SkipTest, "SPIDER_RUN_REAL_MJX_TESTS"):
                real_physics_tests._require_real_mjx_test_runtime()

        probe.assert_not_called()

    def test_guard_fails_without_exactly_one_visible_gpu_before_probe(self) -> None:
        for visible_devices in ("", "0,1", " , "):
            with self.subTest(visible_devices=visible_devices):
                with (
                    mock.patch.dict(
                        os.environ,
                        {
                            "SPIDER_RUN_REAL_MJX_TESTS": "1",
                            "CUDA_VISIBLE_DEVICES": visible_devices,
                        },
                        clear=True,
                    ),
                    mock.patch.object(real_physics_tests, "probe_mjx_runtime") as probe,
                ):
                    with self.assertRaisesRegex(AssertionError, "exactly one"):
                        real_physics_tests._require_real_mjx_test_runtime()

                probe.assert_not_called()

    def test_guard_skips_unavailable_runtime_after_env_gate(self) -> None:
        with (
            mock.patch.dict(
                os.environ,
                {
                    "SPIDER_RUN_REAL_MJX_TESTS": "1",
                    "CUDA_VISIBLE_DEVICES": "2",
                },
                clear=True,
            ),
            mock.patch.object(
                real_physics_tests,
                "probe_mjx_runtime",
                return_value=SimpleNamespace(available=False),
            ) as probe,
            mock.patch.object(real_physics_tests, "require_mjx_runtime") as require,
        ):
            with self.assertRaisesRegex(unittest.SkipTest, "runtime is not available"):
                real_physics_tests._require_real_mjx_test_runtime()

        probe.assert_called_once_with()
        require.assert_not_called()

    def test_guard_requires_runtime_when_opted_in_with_single_visible_gpu(self) -> None:
        runtime = object()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "SPIDER_RUN_REAL_MJX_TESTS": "1",
                    "CUDA_VISIBLE_DEVICES": "2",
                },
                clear=True,
            ),
            mock.patch.object(
                real_physics_tests,
                "probe_mjx_runtime",
                return_value=SimpleNamespace(available=True),
            ) as probe,
            mock.patch.object(
                real_physics_tests,
                "require_mjx_runtime",
                return_value=runtime,
            ) as require,
        ):
            result = real_physics_tests._require_real_mjx_test_runtime()

        self.assertIs(result, runtime)
        probe.assert_called_once_with()
        require.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
