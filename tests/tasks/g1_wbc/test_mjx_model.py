import unittest

from spider.tasks.g1_wbc.constants import ACTION_DIM, QPOS_DIM, QVEL_DIM
from spider.tasks.g1_wbc.mjx_model import (
    assert_mjx_model_parity,
    build_mjx_model_bundle,
)
from spider.tasks.g1_wbc.mjx_runtime import probe_mjx_runtime


class MjxModelBundleTest(unittest.TestCase):
    def test_builds_wxy_parity_cpu_model_bundle(self) -> None:
        bundle = build_mjx_model_bundle(profile_name="wxy_parity")

        self.assertEqual(bundle.profile.name, "wxy_parity")
        self.assertEqual(bundle.cpu_model.nq, QPOS_DIM)
        self.assertEqual(bundle.cpu_model.nv, QVEL_DIM)
        self.assertEqual(bundle.cpu_model.nu, ACTION_DIM)
        self.assertGreaterEqual(len(bundle.body_name_to_id), 30)
        self.assertGreaterEqual(len(bundle.joint_name_to_id), 29)
        self.assertIn("robot/left_hip_pitch_joint", bundle.joint_name_to_id)
        self.assertIn("robot/left_hip_pitch_joint", bundle.actuator_name_to_id)
        self.assertIn("terrain", bundle.geom_name_to_id)
        self.assertIsNone(bundle.mjx_model)

    def test_asserts_parity_profile(self) -> None:
        bundle = build_mjx_model_bundle(profile_name="wxy_parity")

        assert_mjx_model_parity(bundle)

    def test_rejects_unknown_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown MJX contact profile"):
            build_mjx_model_bundle(profile_name="missing")

    def test_reference_profile_runtime_is_not_parity_eligible(self) -> None:
        with self.assertRaisesRegex(ValueError, "not eligible for parity"):
            build_mjx_model_bundle(
                profile_name="hgpt_track_reference",
                require_runtime=True,
            )

    def test_runtime_put_model_smoke_when_available(self) -> None:
        if not probe_mjx_runtime().available:
            self.skipTest("jax/mujoco.mjx runtime is not available")

        bundle = build_mjx_model_bundle(
            profile_name="wxy_parity",
            require_runtime=True,
        )

        self.assertIsNotNone(bundle.mjx_model)


if __name__ == "__main__":
    unittest.main()
