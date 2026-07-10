import os
from types import SimpleNamespace
import unittest

import numpy as np

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    DECIMATION,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    PHYSICS_DT,
    QPOS_DIM,
    QVEL_DIM,
    TASK_EE_BODY_NAMES,
)
from spider.tasks.g1_wbc.mjx_model import build_mjx_model_bundle
from spider.tasks.g1_wbc.mjx_physics import (
    FootContactGeomGroups,
    _warp_floor_contact_summary,
    action_to_model_ctrl,
    foot_contact_geom_groups,
    contact_count_diagnostics_from_warp_impl,
    floor_contact_force_from_contact,
    floor_contact_force_first_row_from_contact,
    floor_contact_force_from_warp_impl,
    floor_contact_force_first_row_from_warp_impl,
    foot_contact_indicator_from_contact,
    floor_contact_indicator_from_contact,
    floor_contact_indicator_from_warp_impl,
    joint_order_to_model_ctrl,
    make_mjx_command_reference_fn,
    make_mjx_physics_step_fn,
    reset_forward_step_smoke,
)
from spider.tasks.g1_wbc.mjx_runtime import probe_mjx_runtime, require_mjx_runtime
from spider.tasks.g1_wbc.rollout import default_joint_pos_tensor, joint_actuator_specs


class _NumpyJnp:
    @staticmethod
    def asarray(value):
        array = np.asarray(value)
        if array.dtype.kind in {"i", "u"}:
            return array
        return array.astype(np.float32)

    @staticmethod
    def zeros_like(value):
        return np.zeros_like(value, dtype=np.float32)

    @staticmethod
    def zeros(shape):
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def any(value, axis=None):
        return np.any(value, axis=axis)

    @staticmethod
    def arange(stop):
        return np.arange(stop, dtype=np.int32)

    @staticmethod
    def clip(value, low, high):
        return np.clip(value, low, high)

    @staticmethod
    def cumsum(value):
        return np.cumsum(value)

    @staticmethod
    def maximum(left, right):
        return np.maximum(left, right)

    @staticmethod
    def minimum(left, right):
        return np.minimum(left, right)

    @staticmethod
    def argmax(value):
        return np.argmax(value)

    @staticmethod
    def sum(value, axis=None):
        return np.sum(value, axis=axis)

    @staticmethod
    def where(condition, left, right):
        return np.where(condition, left, right)


class _FakeCommandReferenceJnp:
    @staticmethod
    def asarray(value):
        array = np.asarray(value)
        if array.dtype.kind in {"i", "u"}:
            return array
        return array.astype(np.float32)

    @staticmethod
    def take(value, indices, axis=0):
        return np.take(value, indices, axis=axis)

    @staticmethod
    def cross(left, right):
        return np.cross(left, right)


class _FakeCommandReferenceJax:
    @staticmethod
    def vmap(fn):
        def mapped(*values):
            rows = [
                fn(*(value[index] for value in values))
                for index in range(len(values[0]))
            ]
            return {
                name: np.stack([row[name] for row in rows], axis=0)
                for name in rows[0]
            }

        return mapped


class _FakeCommandReferenceMjx:
    @staticmethod
    def make_data(model):
        return _FakeCommandReferenceData(model["body_count"])

    @staticmethod
    def forward(model, data):
        del model
        body_count = int(data.xpos.shape[0])
        data.xpos = np.zeros((body_count, 3), dtype=np.float32)
        data.xpos[:, 0] = data.qpos[0]
        data.xpos[:, 2] = np.arange(body_count, dtype=np.float32)
        data.xquat = np.zeros((body_count, 4), dtype=np.float32)
        data.xquat[:, 0] = 1.0
        data.cvel = np.zeros((body_count, 6), dtype=np.float32)
        data.cvel[:, :3] = data.qvel[3:6]
        data.cvel[:, 3:6] = data.qvel[:3]
        data.subtree_com = np.zeros((body_count, 3), dtype=np.float32)
        data.subtree_com[0] = data.qpos[:3]
        return data


class _FakeCommandReferenceData:
    def __init__(self, body_count: int) -> None:
        self.qpos = np.zeros(QPOS_DIM, dtype=np.float32)
        self.qvel = np.zeros(QVEL_DIM, dtype=np.float32)
        self.xpos = np.zeros((body_count, 3), dtype=np.float32)
        self.xquat = np.zeros((body_count, 4), dtype=np.float32)
        self.cvel = np.zeros((body_count, 6), dtype=np.float32)
        self.subtree_com = np.zeros((body_count, 3), dtype=np.float32)

    def replace(self, **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, value)
        return self


class _FakeCommandReferenceRuntime:
    jnp = _FakeCommandReferenceJnp()
    jax = _FakeCommandReferenceJax()
    mjx = _FakeCommandReferenceMjx()


def _require_real_mjx_test_runtime():
    if os.environ.get("SPIDER_RUN_REAL_MJX_TESTS") != "1":
        raise unittest.SkipTest(
            "set SPIDER_RUN_REAL_MJX_TESTS=1 and CUDA_VISIBLE_DEVICES to exactly "
            "one GPU to run real MJX tests"
        )
    visible_devices = tuple(
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    )
    if len(visible_devices) != 1:
        raise AssertionError(
            "SPIDER_RUN_REAL_MJX_TESTS=1 requires exactly one non-empty "
            "CUDA_VISIBLE_DEVICES entry"
        )
    if not probe_mjx_runtime().available:
        raise unittest.SkipTest("jax/mujoco.mjx runtime is not available")
    return require_mjx_runtime()


def _warp_contact_impl_fixture():
    return SimpleNamespace(
        contact__geom=np.array(
            [
                [3, 11],
                [3, 25],
                [31, 3],
                [3, 25],
            ],
            dtype=np.int32,
        ),
        contact__dist=np.array([-0.001, -0.002, -0.003, -0.004], dtype=np.float32),
        contact__includemargin=np.zeros(4, dtype=np.float32),
        contact__worldid=np.array([0, 1, 0, 0], dtype=np.int32),
        contact__dim=np.array([1, 1, 3, 1], dtype=np.int32),
        contact__efc_address=np.array(
            [
                [0, -1, -1, -1],
                [1, -1, -1, -1],
                [1, 2, 3, 4],
                [5, -1, -1, -1],
            ],
            dtype=np.int32,
        ),
        nacon=np.array([3], dtype=np.int32),
        efc__force=np.array(
            [
                [5.0, 1.0, 2.0, 3.0, 4.0, 100.0],
                [0.0, 7.0, 100.0, 100.0, 100.0, 100.0],
            ],
            dtype=np.float32,
        ),
    )


class MjxRealPhysicsTest(unittest.TestCase):
    def test_joint_order_to_model_ctrl_maps_non_identity_actuator_ids(self) -> None:
        bundle = SimpleNamespace(
            actuator_name_to_id={
                "robot/left_hip_pitch_joint": 2,
                "robot/left_hip_roll_joint": 0,
                "robot/left_hip_yaw_joint": 1,
                **{
                    f"robot/joint_{index}": index
                    for index in range(3, ACTION_DIM)
                },
            }
        )
        # Fill the remaining expected G1 names after the deliberately permuted prefix.
        for index, joint_name in enumerate(MUJOCO_JOINT_NAMES[3:], start=3):
            bundle.actuator_name_to_id[f"robot/{joint_name}"] = index
        joint_ctrl = np.linspace(-1.0, 1.0, ACTION_DIM, dtype=np.float32)

        model_ctrl = joint_order_to_model_ctrl(bundle, joint_ctrl, jnp=_NumpyJnp)

        self.assertEqual(model_ctrl.shape, (ACTION_DIM,))
        self.assertEqual(float(model_ctrl[2]), float(joint_ctrl[0]))
        self.assertEqual(float(model_ctrl[0]), float(joint_ctrl[1]))
        self.assertEqual(float(model_ctrl[1]), float(joint_ctrl[2]))
        np.testing.assert_allclose(model_ctrl[3:], joint_ctrl[3:])

    def test_action_to_model_ctrl_applies_wbc_scale_before_actuator_mapping(self) -> None:
        bundle = SimpleNamespace(
            actuator_name_to_id={
                f"robot/{joint_name}": ACTION_DIM - index - 1
                for index, joint_name in enumerate(MUJOCO_JOINT_NAMES)
            }
        )
        action = np.linspace(-0.5, 0.5, ACTION_DIM, dtype=np.float32)
        default_joint_pos = np.linspace(0.1, 0.2, ACTION_DIM, dtype=np.float32)
        action_scale = np.linspace(0.01, 0.03, ACTION_DIM, dtype=np.float32)

        model_ctrl = action_to_model_ctrl(
            bundle,
            action,
            default_joint_pos,
            action_scale,
            jnp=_NumpyJnp,
        )

        expected_joint_ctrl = action * action_scale + default_joint_pos
        actuator_ids = [
            bundle.actuator_name_to_id[f"robot/{joint_name}"]
            for joint_name in MUJOCO_JOINT_NAMES
        ]
        np.testing.assert_allclose(model_ctrl[actuator_ids], expected_joint_ctrl)

    def test_foot_contact_geom_groups_resolve_wxy_profile_ids(self) -> None:
        geom_name_to_id = {"terrain": 3}
        geom_name_to_id.update(
            {
                f"robot/left_foot{index}_collision": 10 + index
                for index in range(1, 8)
            }
        )
        geom_name_to_id.update(
            {
                f"robot/right_foot{index}_collision": 20 + index
                for index in range(1, 8)
            }
        )
        geom_name_to_id["robot/left_hand_collision"] = 31
        bundle = SimpleNamespace(
            profile=SimpleNamespace(
                floor_geom_names=("terrain",),
                foot_collision_geom_names=tuple(
                    f"robot/{side}_foot{index}_collision"
                    for side in ("left", "right")
                    for index in range(1, 8)
                ),
            ),
            geom_name_to_id=geom_name_to_id,
        )

        groups = foot_contact_geom_groups(bundle)

        self.assertEqual(groups.floor_geom_ids, (3,))
        self.assertEqual(groups.left_foot_geom_ids, tuple(range(11, 18)))
        self.assertEqual(groups.right_foot_geom_ids, tuple(range(21, 28)))
        self.assertIn(31, groups.other_robot_geom_ids)

    def test_foot_contact_indicator_from_contact_uses_active_floor_pairs(self) -> None:
        contact = SimpleNamespace(
            geom=np.array(
                [
                    [3, 11],
                    [25, 3],
                    [12, 26],
                    [3, 14],
                    [3, 25],
                    [3, 16],
                ],
                dtype=np.int32,
            ),
            dist=np.array([-0.001, 0.0, -0.1, 0.02, 0.006, 0.01], dtype=np.float32),
            includemargin=np.array(
                [0.0, 0.0, 0.0, 0.015, 0.005, 0.02],
                dtype=np.float32,
            ),
        )

        indicator = foot_contact_indicator_from_contact(
            contact,
            floor_geom_ids=(3,),
            left_foot_geom_ids=tuple(range(11, 18)),
            right_foot_geom_ids=tuple(range(21, 28)),
            jnp=_NumpyJnp,
        )

        np.testing.assert_allclose(indicator, np.array([1.0, 1.0], dtype=np.float32))

    def test_foot_contact_indicator_from_contact_ignores_inactive_pairs(self) -> None:
        contact = SimpleNamespace(
            geom=np.array(
                [
                    [3, 11],
                    [25, 3],
                    [12, 26],
                ],
                dtype=np.int32,
            ),
            dist=np.array([0.02, 0.03, -0.1], dtype=np.float32),
            includemargin=np.array([0.0, 0.01, 0.0], dtype=np.float32),
        )

        indicator = foot_contact_indicator_from_contact(
            contact,
            floor_geom_ids=(3,),
            left_foot_geom_ids=tuple(range(11, 18)),
            right_foot_geom_ids=tuple(range(21, 28)),
            jnp=_NumpyJnp,
        )

        np.testing.assert_allclose(indicator, np.zeros(2, dtype=np.float32))

    def test_foot_contact_indicator_from_contact_ignores_negative_geom_slots(self) -> None:
        contact = SimpleNamespace(
            geom=np.array(
                [
                    [-1, 11],
                    [3, -1],
                ],
                dtype=np.int32,
            ),
            dist=np.array([-1.0, -1.0], dtype=np.float32),
            includemargin=np.zeros(2, dtype=np.float32),
        )

        indicator = foot_contact_indicator_from_contact(
            contact,
            floor_geom_ids=(3,),
            left_foot_geom_ids=tuple(range(11, 18)),
            right_foot_geom_ids=tuple(range(21, 28)),
            jnp=_NumpyJnp,
        )

        np.testing.assert_allclose(indicator, np.zeros(2, dtype=np.float32))

    def test_floor_contact_indicator_from_contact_marks_other_robot_floor_contact(
        self,
    ) -> None:
        contact = SimpleNamespace(
            geom=np.array(
                [
                    [3, 11],
                    [3, 25],
                    [31, 3],
                ],
                dtype=np.int32,
            ),
            dist=np.array([-0.001, -0.002, -0.003], dtype=np.float32),
            includemargin=np.zeros(3, dtype=np.float32),
        )

        indicator = floor_contact_indicator_from_contact(
            contact,
            floor_geom_ids=(3,),
            left_foot_geom_ids=tuple(range(11, 18)),
            right_foot_geom_ids=tuple(range(21, 28)),
            other_robot_geom_ids=(31, 32),
            jnp=_NumpyJnp,
        )

        np.testing.assert_allclose(
            indicator,
            np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

    def test_floor_contact_force_from_contact_sums_pyramidal_normal_rows(self) -> None:
        contact = SimpleNamespace(
            geom=np.array(
                [
                    [3, 11],
                    [3, 25],
                    [31, 3],
                    [31, 25],
                ],
                dtype=np.int32,
            ),
            dist=np.array([-0.001, -0.002, -0.003, -0.004], dtype=np.float32),
            includemargin=np.zeros(4, dtype=np.float32),
            dim=np.array([3, 1, 3, 3], dtype=np.int32),
            efc_address=np.array([0, 8, 9, 13], dtype=np.int32),
        )
        efc_force = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                50.0,
                50.0,
                50.0,
                50.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ],
            dtype=np.float32,
        )

        force = floor_contact_force_from_contact(
            contact,
            efc_force,
            floor_geom_ids=(3,),
            left_foot_geom_ids=tuple(range(11, 18)),
            right_foot_geom_ids=tuple(range(21, 28)),
            other_robot_geom_ids=(31, 32),
            jnp=_NumpyJnp,
        )

        np.testing.assert_allclose(
            force,
            np.array([10.0, 5.0, 30.0], dtype=np.float32),
        )

    def test_floor_contact_force_first_row_from_contact_matches_mujoco_warp_probe(
        self,
    ) -> None:
        contact = SimpleNamespace(
            geom=np.array(
                [
                    [3, 11],
                    [3, 25],
                    [31, 3],
                    [31, 25],
                ],
                dtype=np.int32,
            ),
            dist=np.array([-0.001, -0.002, -0.003, -0.004], dtype=np.float32),
            includemargin=np.zeros(4, dtype=np.float32),
            dim=np.array([3, 1, 3, 3], dtype=np.int32),
            efc_address=np.array([0, 8, 9, 13], dtype=np.int32),
        )
        efc_force = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                50.0,
                50.0,
                50.0,
                50.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                100.0,
            ],
            dtype=np.float32,
        )

        force = floor_contact_force_first_row_from_contact(
            contact,
            efc_force,
            floor_geom_ids=(3,),
            left_foot_geom_ids=tuple(range(11, 18)),
            right_foot_geom_ids=tuple(range(21, 28)),
            other_robot_geom_ids=(31, 32),
            jnp=_NumpyJnp,
        )

        np.testing.assert_allclose(
            force,
            np.array([1.0, 5.0, 6.0], dtype=np.float32),
        )

    def test_floor_contact_indicator_from_warp_impl_filters_world_and_padding(
        self,
    ) -> None:
        impl = _warp_contact_impl_fixture()

        world0 = floor_contact_indicator_from_warp_impl(
            impl,
            0,
            floor_geom_ids=(3,),
            left_foot_geom_ids=(11,),
            right_foot_geom_ids=(25,),
            other_robot_geom_ids=(31,),
            jnp=_NumpyJnp,
        )
        world1 = floor_contact_indicator_from_warp_impl(
            impl,
            1,
            floor_geom_ids=(3,),
            left_foot_geom_ids=(11,),
            right_foot_geom_ids=(25,),
            other_robot_geom_ids=(31,),
            jnp=_NumpyJnp,
        )

        np.testing.assert_allclose(world0, np.array([1.0, 0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(world1, np.array([0.0, 1.0, 0.0], dtype=np.float32))

    def test_floor_contact_force_from_warp_impl_sums_2d_efc_rows_per_world(
        self,
    ) -> None:
        impl = _warp_contact_impl_fixture()

        world0 = floor_contact_force_from_warp_impl(
            impl,
            0,
            floor_geom_ids=(3,),
            left_foot_geom_ids=(11,),
            right_foot_geom_ids=(25,),
            other_robot_geom_ids=(31,),
            jnp=_NumpyJnp,
        )
        world1 = floor_contact_force_from_warp_impl(
            impl,
            1,
            floor_geom_ids=(3,),
            left_foot_geom_ids=(11,),
            right_foot_geom_ids=(25,),
            other_robot_geom_ids=(31,),
            jnp=_NumpyJnp,
        )

        np.testing.assert_allclose(world0, np.array([5.0, 0.0, 10.0], dtype=np.float32))
        np.testing.assert_allclose(world1, np.array([0.0, 7.0, 0.0], dtype=np.float32))

    def test_warp_contact_helpers_use_global_nacon_prefix_when_broadcast(self) -> None:
        impl = SimpleNamespace(
            contact__geom=np.array(
                [
                    [3, 11],
                    [3, 25],
                    [3, 11],
                    [3, 25],
                    [-1, -1],
                ],
                dtype=np.int32,
            ),
            contact__dist=np.array(
                [-0.001, -0.002, -0.003, -0.004, 0.0],
                dtype=np.float32,
            ),
            contact__includemargin=np.zeros(5, dtype=np.float32),
            contact__worldid=np.array([0, 0, 1, 1, 1], dtype=np.int32),
            contact__dim=np.array([1, 1, 1, 1, 1], dtype=np.int32),
            contact__efc_address=np.array(
                [
                    [0, -1, -1, -1],
                    [1, -1, -1, -1],
                    [0, -1, -1, -1],
                    [1, -1, -1, -1],
                    [-1, -1, -1, -1],
                ],
                dtype=np.int32,
            ),
            nacon=np.array([2, 5], dtype=np.int32),
            efc__force=np.array(
                [
                    [10.0, 20.0],
                    [30.0, 40.0],
                ],
                dtype=np.float32,
            ),
        )

        world1 = floor_contact_force_from_warp_impl(
            impl,
            1,
            floor_geom_ids=(3,),
            left_foot_geom_ids=(11,),
            right_foot_geom_ids=(25,),
            other_robot_geom_ids=(),
            jnp=_NumpyJnp,
        )
        counts = contact_count_diagnostics_from_warp_impl(impl, 1, jnp=_NumpyJnp)

        np.testing.assert_allclose(
            world1,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
        )
        self.assertEqual(float(counts["active_contact_count"]), 0.0)
        self.assertEqual(float(counts["contact_pair_count"]), 0.0)

    def test_floor_contact_force_first_row_from_warp_impl_uses_first_solver_row(
        self,
    ) -> None:
        impl = _warp_contact_impl_fixture()

        world0 = floor_contact_force_first_row_from_warp_impl(
            impl,
            0,
            floor_geom_ids=(3,),
            left_foot_geom_ids=(11,),
            right_foot_geom_ids=(25,),
            other_robot_geom_ids=(31,),
            jnp=_NumpyJnp,
        )
        world1 = floor_contact_force_first_row_from_warp_impl(
            impl,
            1,
            floor_geom_ids=(3,),
            left_foot_geom_ids=(11,),
            right_foot_geom_ids=(25,),
            other_robot_geom_ids=(31,),
            jnp=_NumpyJnp,
        )

        np.testing.assert_allclose(world0, np.array([5.0, 0.0, 1.0], dtype=np.float32))
        np.testing.assert_allclose(world1, np.array([0.0, 7.0, 0.0], dtype=np.float32))

    def test_contact_count_diagnostics_from_warp_impl_counts_per_world(self) -> None:
        impl = _warp_contact_impl_fixture()

        world0 = contact_count_diagnostics_from_warp_impl(impl, 0, jnp=_NumpyJnp)
        world1 = contact_count_diagnostics_from_warp_impl(impl, 1, jnp=_NumpyJnp)

        self.assertEqual(float(world0["active_contact_count"]), 2.0)
        self.assertEqual(float(world0["contact_pair_count"]), 2.0)
        self.assertEqual(float(world1["active_contact_count"]), 1.0)
        self.assertEqual(float(world1["contact_pair_count"]), 1.0)

    def test_warp_floor_contact_summary_matches_public_helpers(self) -> None:
        impl = _warp_contact_impl_fixture()
        contact_groups = FootContactGeomGroups(
            floor_geom_ids=(3,),
            left_foot_geom_ids=(11,),
            right_foot_geom_ids=(25,),
            other_robot_geom_ids=(31,),
        )

        for world_id in (0, 1):
            summary = _warp_floor_contact_summary(
                impl,
                world_id,
                contact_groups=contact_groups,
                jnp=_NumpyJnp,
            )
            indicator = floor_contact_indicator_from_warp_impl(
                impl,
                world_id,
                floor_geom_ids=contact_groups.floor_geom_ids,
                left_foot_geom_ids=contact_groups.left_foot_geom_ids,
                right_foot_geom_ids=contact_groups.right_foot_geom_ids,
                other_robot_geom_ids=contact_groups.other_robot_geom_ids,
                jnp=_NumpyJnp,
            )
            force = floor_contact_force_from_warp_impl(
                impl,
                world_id,
                floor_geom_ids=contact_groups.floor_geom_ids,
                left_foot_geom_ids=contact_groups.left_foot_geom_ids,
                right_foot_geom_ids=contact_groups.right_foot_geom_ids,
                other_robot_geom_ids=contact_groups.other_robot_geom_ids,
                jnp=_NumpyJnp,
            )
            counts = contact_count_diagnostics_from_warp_impl(
                impl,
                world_id,
                jnp=_NumpyJnp,
            )

            np.testing.assert_allclose(summary["floor_contact"], indicator)
            np.testing.assert_allclose(summary["floor_contact_force"], force)
            np.testing.assert_allclose(
                summary["floor_contact_force_first_row"],
                floor_contact_force_first_row_from_warp_impl(
                    impl,
                    world_id,
                    floor_geom_ids=contact_groups.floor_geom_ids,
                    left_foot_geom_ids=contact_groups.left_foot_geom_ids,
                    right_foot_geom_ids=contact_groups.right_foot_geom_ids,
                    other_robot_geom_ids=contact_groups.other_robot_geom_ids,
                    jnp=_NumpyJnp,
                ),
            )
            self.assertEqual(
                float(summary["contact_counts"]["active_contact_count"]),
                float(counts["active_contact_count"]),
            )
            self.assertEqual(
                float(summary["contact_counts"]["contact_pair_count"]),
                float(counts["contact_pair_count"]),
            )

    def test_warp_floor_contact_summary_reports_peak_source_rows(self) -> None:
        impl = _warp_contact_impl_fixture()
        contact_groups = FootContactGeomGroups(
            floor_geom_ids=(3,),
            left_foot_geom_ids=(11,),
            right_foot_geom_ids=(25,),
            other_robot_geom_ids=(31,),
        )

        summary = _warp_floor_contact_summary(
            impl,
            0,
            contact_groups=contact_groups,
            include_peak_source=True,
            jnp=_NumpyJnp,
        )

        source = summary["floor_contact_force_peak_source"]
        self.assertEqual(source.shape, (3, 8))
        # Columns: row_id, geom0, geom1, normal_force, first_row_force,
        # group_sum_force, dim, efc0.
        np.testing.assert_allclose(
            source[0],
            np.array([0.0, 3.0, 11.0, 5.0, 5.0, 5.0, 1.0, 0.0]),
        )
        np.testing.assert_allclose(
            source[1],
            np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
        )
        np.testing.assert_allclose(
            source[2],
            np.array([2.0, 31.0, 3.0, 10.0, 1.0, 10.0, 3.0, 1.0]),
        )

    def test_warp_floor_contact_summary_reports_top_rows(self) -> None:
        impl = _warp_contact_impl_fixture()
        contact_groups = FootContactGeomGroups(
            floor_geom_ids=(3,),
            left_foot_geom_ids=(11,),
            right_foot_geom_ids=(25,),
            other_robot_geom_ids=(31,),
        )

        summary = _warp_floor_contact_summary(
            impl,
            0,
            contact_groups=contact_groups,
            include_top_rows=True,
            jnp=_NumpyJnp,
        )

        top_rows = summary["floor_contact_force_top_rows"]
        self.assertEqual(top_rows.shape, (3, 4, 21))
        # Columns: row_id, worldid, nacon0, active, geom0, geom1, dist,
        # margin, dim, efc0..efc3, force0..force3, first_row_force,
        # normal_sum_force, group_sum_force, active_row_count.
        np.testing.assert_allclose(
            top_rows[0, 0],
            np.array(
                [
                    0.0,
                    0.0,
                    3.0,
                    1.0,
                    3.0,
                    11.0,
                    -0.001,
                    0.0,
                    1.0,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    5.0,
                    0.0,
                    0.0,
                    0.0,
                    5.0,
                    5.0,
                    5.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            top_rows[1, 0],
            np.array(
                [
                    -1.0,
                    0.0,
                    3.0,
                    0.0,
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32,
            ),
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            top_rows[2, 0],
            np.array(
                [
                    2.0,
                    0.0,
                    3.0,
                    1.0,
                    31.0,
                    3.0,
                    -0.003,
                    0.0,
                    3.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    1.0,
                    10.0,
                    10.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            atol=1.0e-6,
        )

    def test_command_reference_fn_returns_command_body_kinematics(self) -> None:
        bundle = SimpleNamespace(
            mjx_model={"body_count": 3},
            body_name_to_id={
                "robot/pelvis": 0,
                "robot/torso_link": 2,
            },
        )
        qpos = np.zeros((2, 2, QPOS_DIM), dtype=np.float32)
        qpos[..., 3] = 1.0
        qpos[1, 1, 0] = 0.4
        qvel = np.zeros((2, 2, QVEL_DIM), dtype=np.float32)
        qvel[1, 1, 3:6] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        command_reference_fn = make_mjx_command_reference_fn(
            command_body_names=("pelvis", "torso_link"),
        )

        reference = command_reference_fn(
            bundle,
            qpos,
            qvel,
            runtime=_FakeCommandReferenceRuntime(),
        )

        self.assertEqual(reference["body_pos_w"].shape, (2, 2, 2, 3))
        self.assertEqual(reference["body_quat_w"].shape, (2, 2, 2, 4))
        self.assertEqual(reference["body_ang_vel_w"].shape, (2, 2, 2, 3))
        self.assertEqual(reference["body_lin_vel_w"].shape, (2, 2, 2, 3))
        self.assertAlmostEqual(float(reference["body_pos_w"][1, 1, 0, 0]), 0.4)
        self.assertAlmostEqual(float(reference["body_pos_w"][1, 1, 1, 2]), 2.0)
        np.testing.assert_allclose(
            reference["body_ang_vel_w"][1, 1, 0],
            [0.1, 0.2, 0.3],
        )

    def test_reset_forward_step_smoke_runs_real_mjx_when_runtime_available(self) -> None:
        runtime = _require_real_mjx_test_runtime()
        bundle = build_mjx_model_bundle(profile_name="wxy_parity", require_runtime=True)
        qpos = np.zeros(QPOS_DIM, dtype=np.float32)
        qpos[3] = 1.0
        qvel = np.zeros(QVEL_DIM, dtype=np.float32)
        joint_ctrl = np.linspace(-0.2, 0.2, ACTION_DIM, dtype=np.float32)

        result = reset_forward_step_smoke(
            bundle,
            qpos,
            qvel,
            joint_ctrl,
            runtime=runtime,
        )

        self.assertEqual(result["qpos"].shape, (QPOS_DIM,))
        self.assertEqual(result["qvel"].shape, (QVEL_DIM,))
        self.assertEqual(result["ctrl"].shape, (ACTION_DIM,))
        actuator_ids = [
            bundle.actuator_name_to_id[f"robot/{joint_name}"]
            for joint_name in MUJOCO_JOINT_NAMES
        ]
        np.testing.assert_allclose(result["ctrl"][actuator_ids], joint_ctrl, atol=1.0e-6)
        self.assertAlmostEqual(
            float(result["time"]),
            PHYSICS_DT * DECIMATION,
            places=5,
        )
        self.assertAlmostEqual(float(np.linalg.norm(result["qpos"][3:7])), 1.0, places=4)
        self.assertTrue(np.all(np.isfinite(result["qpos"])))
        self.assertTrue(np.all(np.isfinite(result["qvel"])))

    def test_real_mjx_physics_step_returns_rollout_state_fields(self) -> None:
        runtime = _require_real_mjx_test_runtime()
        bundle = build_mjx_model_bundle(profile_name="wxy_parity", require_runtime=True)
        default_joint_pos = default_joint_pos_tensor("cpu").numpy()
        action_scale = joint_actuator_specs("cpu")["action_scale"].numpy()
        physics_step = make_mjx_physics_step_fn(
            default_joint_pos=default_joint_pos,
            action_scale=action_scale,
        )
        sample_count = 2
        qpos = np.zeros((sample_count, QPOS_DIM), dtype=np.float32)
        qpos[:, 3] = 1.0
        qpos[:, 7:] = default_joint_pos
        qpos[:, 0] = np.array([0.0, 0.05], dtype=np.float32)
        qvel = np.zeros((sample_count, QVEL_DIM), dtype=np.float32)
        robot_state = {
            "qpos": qpos,
            "qvel": qvel,
            "body_pos_w": np.zeros(
                (sample_count, len(MUJOCO_BODY_NAMES), 3),
                dtype=np.float32,
            ),
            "body_quat_w": np.zeros(
                (sample_count, len(MUJOCO_BODY_NAMES), 4),
                dtype=np.float32,
            ),
            "body_ang_vel_w": np.zeros(
                (sample_count, len(MUJOCO_BODY_NAMES), 3),
                dtype=np.float32,
            ),
        }
        action = np.zeros((sample_count, ACTION_DIM), dtype=np.float32)
        action[:, 0] = np.array([0.2, -0.2], dtype=np.float32)

        next_robot, score_state = physics_step(
            bundle,
            robot_state,
            qpos,
            action,
            0,
            runtime=runtime,
        )
        next_robot = {
            name: runtime.jax.device_get(value)
            for name, value in next_robot.items()
            if value is not None
        }
        score_state = {
            name: runtime.jax.device_get(value)
            for name, value in score_state.items()
        }

        self.assertEqual(next_robot["qpos"].shape, (sample_count, QPOS_DIM))
        self.assertEqual(next_robot["qvel"].shape, (sample_count, QVEL_DIM))
        self.assertEqual(
            next_robot["body_pos_w"].shape,
            (sample_count, len(MUJOCO_BODY_NAMES), 3),
        )
        self.assertEqual(
            next_robot["body_quat_w"].shape,
            (sample_count, len(MUJOCO_BODY_NAMES), 4),
        )
        self.assertEqual(
            next_robot["body_lin_vel_w"].shape,
            (sample_count, len(MUJOCO_BODY_NAMES), 3),
        )
        self.assertEqual(
            next_robot["body_ang_vel_w"].shape,
            (sample_count, len(MUJOCO_BODY_NAMES), 3),
        )
        self.assertEqual(score_state["root_pos"].shape, (sample_count, 3))
        self.assertEqual(
            score_state["body_pos"].shape,
            (sample_count, len(MUJOCO_BODY_NAMES), 3),
        )
        self.assertEqual(
            score_state["ee_pos"].shape,
            (sample_count, len(TASK_EE_BODY_NAMES), 3),
        )
        self.assertEqual(score_state["contact"].shape, (sample_count, 2))
        self.assertEqual(score_state["floor_contact"].shape, (sample_count, 3))
        self.assertEqual(score_state["contact_force"].shape, (sample_count, 2))
        self.assertEqual(
            score_state["floor_contact_force"].shape,
            (sample_count, 3),
        )
        self.assertNotIn("contact_force_first_row", score_state)
        self.assertNotIn("floor_contact_force_first_row", score_state)
        self.assertEqual(score_state["model_ctrl"].shape, (sample_count, ACTION_DIM))
        self.assertEqual(score_state["time"].shape, (sample_count,))

        actuator_ids = [
            bundle.actuator_name_to_id[f"robot/{joint_name}"]
            for joint_name in MUJOCO_JOINT_NAMES
        ]
        expected_joint_ctrl = action * action_scale.reshape(1, -1) + default_joint_pos
        np.testing.assert_allclose(
            score_state["model_ctrl"][:, actuator_ids],
            expected_joint_ctrl,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            score_state["time"],
            np.full(sample_count, PHYSICS_DT * DECIMATION, dtype=np.float32),
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            np.linalg.norm(next_robot["qpos"][:, 3:7], axis=-1),
            np.ones(sample_count, dtype=np.float32),
            atol=1.0e-4,
        )
        for values in (*next_robot.values(), *score_state.values()):
            self.assertTrue(np.all(np.isfinite(values)))

    def test_real_mjx_physics_step_can_emit_first_row_force_diagnostics(self) -> None:
        runtime = _require_real_mjx_test_runtime()
        bundle = build_mjx_model_bundle(profile_name="wxy_parity", require_runtime=True)
        default_joint_pos = default_joint_pos_tensor("cpu").numpy()
        action_scale = joint_actuator_specs("cpu")["action_scale"].numpy()
        physics_step = make_mjx_physics_step_fn(
            default_joint_pos=default_joint_pos,
            action_scale=action_scale,
            contact_force_first_row_diagnostics=True,
        )
        sample_count = 1
        qpos = np.zeros((sample_count, QPOS_DIM), dtype=np.float32)
        qpos[:, 3] = 1.0
        qpos[:, 7:] = default_joint_pos
        qvel = np.zeros((sample_count, QVEL_DIM), dtype=np.float32)
        robot_state = {
            "qpos": qpos,
            "qvel": qvel,
            "body_pos_w": np.zeros(
                (sample_count, len(MUJOCO_BODY_NAMES), 3),
                dtype=np.float32,
            ),
            "body_quat_w": np.zeros(
                (sample_count, len(MUJOCO_BODY_NAMES), 4),
                dtype=np.float32,
            ),
            "body_ang_vel_w": np.zeros(
                (sample_count, len(MUJOCO_BODY_NAMES), 3),
                dtype=np.float32,
            ),
        }

        _next_robot, score_state = physics_step(
            bundle,
            robot_state,
            qpos,
            np.zeros((sample_count, ACTION_DIM), dtype=np.float32),
            0,
            runtime=runtime,
        )
        score_state = {
            name: runtime.jax.device_get(value)
            for name, value in score_state.items()
        }

        self.assertEqual(
            score_state["contact_force_first_row"].shape,
            (sample_count, 2),
        )
        self.assertEqual(
            score_state["floor_contact_force_first_row"].shape,
            (sample_count, 3),
        )
        self.assertTrue(np.all(np.isfinite(score_state["contact_force_first_row"])))

    def test_real_mjx_physics_step_uses_carried_robot_qpos_not_command_qpos(
        self,
    ) -> None:
        runtime = _require_real_mjx_test_runtime()
        bundle = build_mjx_model_bundle(profile_name="wxy_parity", require_runtime=True)
        default_joint_pos = default_joint_pos_tensor("cpu").numpy()
        action_scale = joint_actuator_specs("cpu")["action_scale"].numpy()
        physics_step = make_mjx_physics_step_fn(
            default_joint_pos=default_joint_pos,
            action_scale=action_scale,
            decimation=0,
        )
        qpos = np.zeros((1, QPOS_DIM), dtype=np.float32)
        qpos[:, 3] = 1.0
        qpos[:, 7:] = default_joint_pos
        qpos[:, 0] = 0.25
        command_qpos = qpos.copy()
        command_qpos[:, 0] = 1.25
        qvel = np.zeros((1, QVEL_DIM), dtype=np.float32)
        robot_state = {
            "qpos": qpos,
            "qvel": qvel,
            "body_pos_w": np.zeros((1, len(MUJOCO_BODY_NAMES), 3), dtype=np.float32),
            "body_quat_w": np.zeros((1, len(MUJOCO_BODY_NAMES), 4), dtype=np.float32),
            "body_ang_vel_w": np.zeros((1, len(MUJOCO_BODY_NAMES), 3), dtype=np.float32),
        }

        next_robot, _score_state = physics_step(
            bundle,
            robot_state,
            command_qpos,
            np.zeros((1, ACTION_DIM), dtype=np.float32),
            0,
            runtime=runtime,
        )
        next_qpos = runtime.jax.device_get(next_robot["qpos"])

        self.assertAlmostEqual(float(next_qpos[0, 0]), 0.25, places=6)

    def test_real_mjx_physics_step_carries_mjx_data_without_time_reset(self) -> None:
        runtime = _require_real_mjx_test_runtime()
        bundle = build_mjx_model_bundle(profile_name="wxy_parity", require_runtime=True)
        default_joint_pos = default_joint_pos_tensor("cpu").numpy()
        action_scale = joint_actuator_specs("cpu")["action_scale"].numpy()
        physics_step = make_mjx_physics_step_fn(
            default_joint_pos=default_joint_pos,
            action_scale=action_scale,
        )
        qpos = np.zeros((1, QPOS_DIM), dtype=np.float32)
        qpos[:, 3] = 1.0
        qpos[:, 7:] = default_joint_pos
        qvel = np.zeros((1, QVEL_DIM), dtype=np.float32)
        robot_state = {
            "qpos": qpos,
            "qvel": qvel,
            "body_pos_w": np.zeros((1, len(MUJOCO_BODY_NAMES), 3), dtype=np.float32),
            "body_quat_w": np.zeros((1, len(MUJOCO_BODY_NAMES), 4), dtype=np.float32),
            "body_ang_vel_w": np.zeros((1, len(MUJOCO_BODY_NAMES), 3), dtype=np.float32),
        }
        robot_state = physics_step.initialize_robot_state(
            bundle,
            robot_state,
            1,
            runtime=runtime,
        )
        action = np.zeros((1, ACTION_DIM), dtype=np.float32)

        next_robot, first_score = physics_step(
            bundle,
            robot_state,
            qpos,
            action,
            0,
            runtime=runtime,
        )
        _next_robot, second_score = physics_step(
            bundle,
            next_robot,
            qpos,
            action,
            1,
            runtime=runtime,
        )

        first_time = runtime.jax.device_get(first_score["time"])
        second_time = runtime.jax.device_get(second_score["time"])
        np.testing.assert_allclose(
            first_time,
            np.array([PHYSICS_DT * DECIMATION], dtype=np.float32),
            atol=1.0e-5,
        )
        np.testing.assert_allclose(
            second_time,
            np.array([2 * PHYSICS_DT * DECIMATION], dtype=np.float32),
            atol=1.0e-5,
        )

    def test_real_warp_batched_physics_step_keeps_carried_contact_counts_bounded(
        self,
    ) -> None:
        runtime = _require_real_mjx_test_runtime()
        bundle = build_mjx_model_bundle(
            profile_name="wxy_parity",
            require_runtime=True,
            mjx_impl="warp",
            mjx_warp_naconmax=20000,
            mjx_warp_njmax=256,
            mjx_model_options={"iterations": 4, "ls_iterations": 5},
        )
        default_joint_pos = default_joint_pos_tensor("cpu").numpy()
        action_scale = joint_actuator_specs("cpu")["action_scale"].numpy()
        physics_step = make_mjx_physics_step_fn(
            default_joint_pos=default_joint_pos,
            action_scale=action_scale,
        )
        sample_count = 4
        qpos = np.zeros((sample_count, QPOS_DIM), dtype=np.float32)
        qpos[:, 3] = 1.0
        qpos[:, 7:] = default_joint_pos
        qvel = np.zeros((sample_count, QVEL_DIM), dtype=np.float32)
        robot_state = {
            "qpos": qpos,
            "qvel": qvel,
            "body_pos_w": np.zeros(
                (sample_count, len(MUJOCO_BODY_NAMES), 3),
                dtype=np.float32,
            ),
            "body_quat_w": np.zeros(
                (sample_count, len(MUJOCO_BODY_NAMES), 4),
                dtype=np.float32,
            ),
            "body_ang_vel_w": np.zeros(
                (sample_count, len(MUJOCO_BODY_NAMES), 3),
                dtype=np.float32,
            ),
        }
        robot_state = physics_step.initialize_robot_state(
            bundle,
            robot_state,
            sample_count,
            runtime=runtime,
        )
        self.assertIn("mjx_data", robot_state)
        action = np.zeros((sample_count, ACTION_DIM), dtype=np.float32)

        next_robot, first_score = physics_step(
            bundle,
            robot_state,
            qpos,
            action,
            0,
            runtime=runtime,
        )
        self.assertIn("mjx_data", next_robot)
        _next_robot, second_score = physics_step(
            bundle,
            next_robot,
            qpos,
            action,
            1,
            runtime=runtime,
        )

        first_force = runtime.jax.device_get(first_score["contact_force"])
        first_pairs = runtime.jax.device_get(first_score["contact_pair_count"])
        second_pairs = runtime.jax.device_get(second_score["contact_pair_count"])
        self.assertGreater(float(np.min(first_force)), 1.0)
        self.assertLess(float(np.max(first_pairs)), 512.0)
        self.assertLess(float(np.max(second_pairs)), 512.0)


if __name__ == "__main__":
    unittest.main()
