import unittest
from dataclasses import replace

from spider.tasks.g1_wbc.constants import ACTION_DIM, QPOS_DIM, QVEL_DIM
from spider.tasks.g1_wbc.mjx_model import (
    MjxModelBundle,
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
        for floor_geom_name in bundle.profile.floor_geom_names:
            self.assertIn(floor_geom_name, bundle.geom_name_to_id)
        for foot_geom_name in bundle.profile.foot_collision_geom_names:
            self.assertIn(foot_geom_name, bundle.geom_name_to_id)
        self.assertIsNone(bundle.mjx_model)

    def test_asserts_parity_profile(self) -> None:
        bundle = build_mjx_model_bundle(profile_name="wxy_parity")

        assert_mjx_model_parity(bundle)

    def test_parity_assertion_checks_body_actuator_and_contact_names(self) -> None:
        bundle = build_mjx_model_bundle(profile_name="wxy_parity")

        missing_body = replace(
            bundle,
            body_name_to_id={
                key: value
                for key, value in bundle.body_name_to_id.items()
                if key != "robot/pelvis"
            },
        )
        with self.assertRaisesRegex(ValueError, "robot/pelvis"):
            assert_mjx_model_parity(missing_body)

        missing_actuator = replace(
            bundle,
            actuator_name_to_id={
                key: value
                for key, value in bundle.actuator_name_to_id.items()
                if key != "robot/left_hip_pitch_joint"
            },
        )
        with self.assertRaisesRegex(ValueError, "robot/left_hip_pitch_joint"):
            assert_mjx_model_parity(missing_actuator)

        missing_floor = replace(
            bundle,
            geom_name_to_id={
                key: value
                for key, value in bundle.geom_name_to_id.items()
                if key != "terrain"
            },
        )
        with self.assertRaisesRegex(ValueError, "terrain"):
            assert_mjx_model_parity(missing_floor)

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
