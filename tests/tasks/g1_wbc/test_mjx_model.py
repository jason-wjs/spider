import unittest
from dataclasses import replace
from types import SimpleNamespace

import mujoco

from spider.tasks.g1_wbc.constants import ACTION_DIM, QPOS_DIM, QVEL_DIM
from spider.tasks.g1_wbc.mjx_model import (
    MjxModelBundle,
    assert_mjx_model_parity,
    build_mjx_model_bundle,
    make_mjx_data,
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
        self.assertEqual(bundle.cpu_model.npair, 0)
        self.assertEqual(bundle.profile.explicit_pair_names, ())
        for floor_geom_name in bundle.profile.floor_geom_names:
            self.assertIn(floor_geom_name, bundle.geom_name_to_id)
        for foot_geom_name in bundle.profile.foot_collision_geom_names:
            self.assertIn(foot_geom_name, bundle.geom_name_to_id)
        self.assertIsNone(bundle.mjx_model)
        self.assertIsNone(bundle.mjx_impl)
        self.assertIsNone(bundle.mjx_warp_naconmax)
        self.assertIsNone(bundle.mjx_warp_njmax)

    def test_builds_wxy_explicit_pair_profile_without_changing_dynamics(self) -> None:
        parity = build_mjx_model_bundle(profile_name="wxy_parity")
        explicit = build_mjx_model_bundle(profile_name="wxy_explicit_pairs_7caps")

        self.assertEqual(explicit.profile.name, "wxy_explicit_pairs_7caps")
        self.assertTrue(explicit.profile.eligible_for_parity)
        self.assertEqual(explicit.cpu_model.nq, QPOS_DIM)
        self.assertEqual(explicit.cpu_model.nv, QVEL_DIM)
        self.assertEqual(explicit.cpu_model.nu, ACTION_DIM)
        self.assertGreater(explicit.cpu_model.npair, 0)
        self.assertEqual(
            explicit.cpu_model.opt.timestep,
            parity.cpu_model.opt.timestep,
        )
        self.assertEqual(
            explicit.cpu_model.opt.integrator,
            parity.cpu_model.opt.integrator,
        )
        self.assertEqual(
            explicit.cpu_model.opt.iterations,
            parity.cpu_model.opt.iterations,
        )
        self.assertEqual(
            explicit.cpu_model.opt.ls_iterations,
            parity.cpu_model.opt.ls_iterations,
        )
        for actuator_id in range(explicit.cpu_model.nu):
            self.assertEqual(
                list(explicit.cpu_model.actuator_forcerange[actuator_id]),
                list(parity.cpu_model.actuator_forcerange[actuator_id]),
            )
            self.assertEqual(
                list(explicit.cpu_model.actuator_gainprm[actuator_id]),
                list(parity.cpu_model.actuator_gainprm[actuator_id]),
            )
            self.assertEqual(
                list(explicit.cpu_model.actuator_biasprm[actuator_id]),
                list(parity.cpu_model.actuator_biasprm[actuator_id]),
            )

        terrain_id = explicit.geom_name_to_id["terrain"]
        self.assertEqual(int(explicit.cpu_model.geom_contype[terrain_id]), 1)
        self.assertEqual(int(explicit.cpu_model.geom_conaffinity[terrain_id]), 1)
        for geom_name, geom_id in explicit.geom_name_to_id.items():
            if geom_name == "terrain" or not geom_name.endswith("_collision"):
                continue
            self.assertEqual(int(explicit.cpu_model.geom_contype[geom_id]), 0)
            self.assertEqual(int(explicit.cpu_model.geom_conaffinity[geom_id]), 0)

        pair_names = {
            mujoco.mj_id2name(explicit.cpu_model, mujoco.mjtObj.mjOBJ_PAIR, pair_id)
            for pair_id in range(explicit.cpu_model.npair)
        }
        self.assertIn("robot/left_foot1_collision__terrain", pair_names)
        self.assertIn("robot/right_foot7_collision__terrain", pair_names)
        self.assertIn("robot/left_hand_collision__terrain", pair_names)

    def test_builds_wxy_explicit_floor_leg_profile_without_changing_dynamics(
        self,
    ) -> None:
        parity = build_mjx_model_bundle(profile_name="wxy_parity")
        explicit = build_mjx_model_bundle(
            profile_name="wxy_explicit_floor_leg_pairs_7caps"
        )

        self.assertEqual(explicit.profile.name, "wxy_explicit_floor_leg_pairs_7caps")
        self.assertTrue(explicit.profile.eligible_for_parity)
        self.assertGreater(explicit.cpu_model.npair, 0)
        self.assertLess(explicit.cpu_model.npair, 65)
        self.assertEqual(explicit.cpu_model.opt.timestep, parity.cpu_model.opt.timestep)
        self.assertEqual(explicit.cpu_model.opt.integrator, parity.cpu_model.opt.integrator)
        self.assertEqual(explicit.cpu_model.opt.iterations, parity.cpu_model.opt.iterations)
        self.assertEqual(
            explicit.cpu_model.opt.ls_iterations,
            parity.cpu_model.opt.ls_iterations,
        )
        for actuator_id in range(explicit.cpu_model.nu):
            self.assertEqual(
                list(explicit.cpu_model.actuator_forcerange[actuator_id]),
                list(parity.cpu_model.actuator_forcerange[actuator_id]),
            )
            self.assertEqual(
                list(explicit.cpu_model.actuator_gainprm[actuator_id]),
                list(parity.cpu_model.actuator_gainprm[actuator_id]),
            )
            self.assertEqual(
                list(explicit.cpu_model.actuator_biasprm[actuator_id]),
                list(parity.cpu_model.actuator_biasprm[actuator_id]),
            )

        pair_names = {
            mujoco.mj_id2name(explicit.cpu_model, mujoco.mjtObj.mjOBJ_PAIR, pair_id)
            for pair_id in range(explicit.cpu_model.npair)
        }
        self.assertIn("robot/left_foot1_collision__terrain", pair_names)
        self.assertIn("robot/right_foot7_collision__terrain", pair_names)
        self.assertIn(
            "robot/right_shin_collision__robot/left_thigh_collision",
            pair_names,
        )
        self.assertNotIn("robot/torso_collision__robot/left_hand_collision", pair_names)
        self.assertNotIn(
            "robot/left_hand_collision__robot/right_hand_collision",
            pair_names,
        )

    def test_builds_wxy_explicit_floor_profile_without_self_pairs_or_dynamics_changes(
        self,
    ) -> None:
        parity = build_mjx_model_bundle(profile_name="wxy_parity")
        explicit = build_mjx_model_bundle(profile_name="wxy_explicit_floor_pairs_7caps")

        self.assertEqual(explicit.profile.name, "wxy_explicit_floor_pairs_7caps")
        self.assertTrue(explicit.profile.eligible_for_parity)
        self.assertEqual(explicit.cpu_model.npair, 31)
        self.assertEqual(explicit.cpu_model.opt.timestep, parity.cpu_model.opt.timestep)
        self.assertEqual(explicit.cpu_model.opt.integrator, parity.cpu_model.opt.integrator)
        self.assertEqual(explicit.cpu_model.opt.iterations, parity.cpu_model.opt.iterations)
        self.assertEqual(
            explicit.cpu_model.opt.ls_iterations,
            parity.cpu_model.opt.ls_iterations,
        )
        for actuator_id in range(explicit.cpu_model.nu):
            self.assertEqual(
                list(explicit.cpu_model.actuator_forcerange[actuator_id]),
                list(parity.cpu_model.actuator_forcerange[actuator_id]),
            )
            self.assertEqual(
                list(explicit.cpu_model.actuator_gainprm[actuator_id]),
                list(parity.cpu_model.actuator_gainprm[actuator_id]),
            )
            self.assertEqual(
                list(explicit.cpu_model.actuator_biasprm[actuator_id]),
                list(parity.cpu_model.actuator_biasprm[actuator_id]),
            )

        pair_names = {
            mujoco.mj_id2name(explicit.cpu_model, mujoco.mjtObj.mjOBJ_PAIR, pair_id)
            for pair_id in range(explicit.cpu_model.npair)
        }
        self.assertIn("robot/left_foot1_collision__terrain", pair_names)
        self.assertIn("robot/right_foot7_collision__terrain", pair_names)
        self.assertIn("robot/left_hand_collision__terrain", pair_names)
        self.assertNotIn(
            "robot/right_shin_collision__robot/left_thigh_collision",
            pair_names,
        )
        self.assertNotIn(
            "robot/left_foot4_collision__robot/right_foot4_collision",
            pair_names,
        )

    def test_wxy_pair_param_parity_keeps_pair_set_but_relaxes_nonfoot_floor_dim(
        self,
    ) -> None:
        base = build_mjx_model_bundle(
            profile_name="wxy_explicit_floor_leg_pairs_7caps"
        )
        parity_params = build_mjx_model_bundle(
            profile_name="wxy_explicit_floor_leg_pair_param_parity_7caps"
        )

        self.assertEqual(
            parity_params.profile.explicit_pair_names,
            base.profile.explicit_pair_names,
        )
        self.assertEqual(parity_params.cpu_model.npair, base.cpu_model.npair)
        pair_ids = {
            mujoco.mj_id2name(
                parity_params.cpu_model,
                mujoco.mjtObj.mjOBJ_PAIR,
                pair_id,
            ): pair_id
            for pair_id in range(parity_params.cpu_model.npair)
        }
        foot_floor_id = pair_ids["robot/left_foot1_collision__terrain"]
        hand_floor_id = pair_ids["robot/left_hand_collision__terrain"]
        self.assertEqual(int(parity_params.cpu_model.pair_dim[foot_floor_id]), 3)
        self.assertEqual(int(parity_params.cpu_model.pair_dim[hand_floor_id]), 1)
        self.assertEqual(int(base.cpu_model.pair_dim[hand_floor_id]), 3)

    def test_builds_wxy_floor_leg_foot147_profile_without_changing_dynamics(
        self,
    ) -> None:
        parity = build_mjx_model_bundle(profile_name="wxy_parity")
        explicit = build_mjx_model_bundle(
            profile_name="wxy_explicit_floor_leg_foot147_pairs_7caps"
        )

        self.assertEqual(
            explicit.profile.name,
            "wxy_explicit_floor_leg_foot147_pairs_7caps",
        )
        self.assertTrue(explicit.profile.eligible_for_parity)
        self.assertEqual(explicit.cpu_model.npair, 36)
        self.assertEqual(explicit.cpu_model.opt.timestep, parity.cpu_model.opt.timestep)
        self.assertEqual(explicit.cpu_model.opt.integrator, parity.cpu_model.opt.integrator)
        self.assertEqual(explicit.cpu_model.opt.iterations, parity.cpu_model.opt.iterations)
        self.assertEqual(
            explicit.cpu_model.opt.ls_iterations,
            parity.cpu_model.opt.ls_iterations,
        )
        for actuator_id in range(explicit.cpu_model.nu):
            self.assertEqual(
                list(explicit.cpu_model.actuator_forcerange[actuator_id]),
                list(parity.cpu_model.actuator_forcerange[actuator_id]),
            )
            self.assertEqual(
                list(explicit.cpu_model.actuator_gainprm[actuator_id]),
                list(parity.cpu_model.actuator_gainprm[actuator_id]),
            )
            self.assertEqual(
                list(explicit.cpu_model.actuator_biasprm[actuator_id]),
                list(parity.cpu_model.actuator_biasprm[actuator_id]),
            )

        pair_names = {
            mujoco.mj_id2name(explicit.cpu_model, mujoco.mjtObj.mjOBJ_PAIR, pair_id)
            for pair_id in range(explicit.cpu_model.npair)
        }
        pair_ids = {
            mujoco.mj_id2name(explicit.cpu_model, mujoco.mjtObj.mjOBJ_PAIR, pair_id): pair_id
            for pair_id in range(explicit.cpu_model.npair)
        }
        for side in ("left", "right"):
            for index in (1, 4, 7):
                self.assertIn(
                    f"robot/{side}_foot{index}_collision__terrain",
                    pair_names,
                )
            for index in (2, 3, 5, 6):
                self.assertNotIn(
                    f"robot/{side}_foot{index}_collision__terrain",
                    pair_names,
                )
        self.assertIn("robot/left_hand_collision__terrain", pair_names)
        self.assertIn(
            "robot/right_shin_collision__robot/left_thigh_collision",
            pair_names,
        )
        foot_floor_id = pair_ids["robot/left_foot4_collision__terrain"]
        leg_self_id = pair_ids[
            "robot/right_shin_collision__robot/left_thigh_collision"
        ]
        self.assertEqual(int(explicit.cpu_model.pair_dim[foot_floor_id]), 3)
        self.assertEqual(int(explicit.cpu_model.pair_dim[leg_self_id]), 1)

    def test_records_explicit_mjx_impl_without_runtime(self) -> None:
        bundle = build_mjx_model_bundle(
            profile_name="wxy_parity",
            mjx_impl="warp",
            mjx_warp_naconmax=4096,
            mjx_warp_njmax=512,
        )

        self.assertEqual(bundle.mjx_impl, "warp")
        self.assertEqual(bundle.mjx_warp_naconmax, 4096)
        self.assertEqual(bundle.mjx_warp_njmax, 512)

    def test_applies_supported_model_option_overrides(self) -> None:
        bundle = build_mjx_model_bundle(
            profile_name="wxy_parity",
            mjx_model_options={
                "iterations": 6,
                "ls_iterations": 10,
            },
        )

        self.assertEqual(bundle.cpu_model.opt.iterations, 6)
        self.assertEqual(bundle.cpu_model.opt.ls_iterations, 10)
        self.assertEqual(
            bundle.mjx_model_options,
            {"iterations": 6, "ls_iterations": 10},
        )

    def test_rejects_unsupported_model_option_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported MJX model option"):
            build_mjx_model_bundle(
                profile_name="wxy_parity",
                mjx_model_options={"solver": 1},
            )

    def test_rejects_unknown_mjx_impl(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported MJX impl"):
            build_mjx_model_bundle(profile_name="wxy_parity", mjx_impl="not-an-impl")

    def test_make_mjx_data_uses_cpu_model_and_buffers_for_warp(self) -> None:
        bundle = build_mjx_model_bundle(
            profile_name="wxy_parity",
            mjx_impl="warp",
            mjx_warp_naconmax=4096,
            mjx_warp_njmax=512,
        )
        mjx_model = object()
        bundle = replace(bundle, mjx_model=mjx_model)
        calls = []

        class FakeMjx:
            def make_data(self, model, **kwargs):
                calls.append((model, kwargs))
                return "data"

        data = make_mjx_data(
            bundle,
            runtime=SimpleNamespace(mjx=FakeMjx()),
        )

        self.assertEqual(data, "data")
        self.assertIs(calls[0][0], bundle.cpu_model)
        self.assertEqual(
            calls[0][1],
            {"impl": "warp", "naconmax": 4096, "njmax": 512},
        )

    def test_make_mjx_data_uses_mjx_model_for_jax(self) -> None:
        bundle = build_mjx_model_bundle(profile_name="wxy_parity", mjx_impl="jax")
        mjx_model = object()
        bundle = replace(bundle, mjx_model=mjx_model)
        calls = []

        class FakeMjx:
            def make_data(self, model, **kwargs):
                calls.append((model, kwargs))
                return "data"

        data = make_mjx_data(
            bundle,
            runtime=SimpleNamespace(mjx=FakeMjx()),
        )

        self.assertEqual(data, "data")
        self.assertIs(calls[0][0], mjx_model)
        self.assertEqual(calls[0][1], {})

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

    def test_reference_profile_cannot_bind_to_wxy_bundle(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be bound"):
            build_mjx_model_bundle(profile_name="hgpt_track_reference")
        with self.assertRaisesRegex(ValueError, "cannot be bound"):
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
