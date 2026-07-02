import unittest

import numpy as np
import torch

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    ANCHOR_BODY_NAME,
    COMMAND_BODY_NAMES,
    LIMB_EE_BODY_NAMES,
    MUJOCO_BODY_NAMES,
    OBS_DIM,
    OBS_HISTORY_LENGTH,
    QVEL_DIM,
    TRACKING_ANCHOR_BODY_NAME,
)
from spider.tasks.g1_wbc.motion import G1Motion
from spider.tasks.g1_wbc.mjx_obs import (
    JaxObsIndices,
    JaxObsState,
    LIMB_POSE_DIM,
    OBS_FIELD_ORDER,
    OBS_FIELD_SPECS,
    build_wbc_observation,
    build_wbc_observation_from_state,
    update_obs_history,
)
from spider.tasks.g1_wbc.obs import G1WbcObservationBuilder, RobotState


class _NumpyJnp:
    int32 = np.int32

    @staticmethod
    def asarray(value, dtype=None):
        return np.asarray(value, dtype=dtype or np.float32)

    @staticmethod
    def concatenate(values, axis=0):
        return np.concatenate(values, axis=axis)

    @staticmethod
    def expand_dims(value, axis):
        return np.expand_dims(value, axis=axis)

    @staticmethod
    def repeat(value, repeats, axis=0):
        return np.repeat(value, repeats, axis=axis)

    @staticmethod
    def where(condition, x, y):
        return np.where(condition, x, y)

    @staticmethod
    def stack(values, axis=0):
        return np.stack(values, axis=axis)

    @staticmethod
    def zeros(shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def take(value, indices, axis=0):
        return np.take(value, indices, axis=axis)

    @staticmethod
    def sum(value, axis=None, keepdims=False):
        return np.sum(value, axis=axis, keepdims=keepdims)


class _StrictTakeJnp(_NumpyJnp):
    @staticmethod
    def take(value, indices, axis=0):
        if isinstance(indices, (list, tuple)):
            raise TypeError("take requires ndarray indices")
        return np.take(value, indices, axis=axis)


def _values(shape: tuple[int, ...], start: int) -> np.ndarray:
    size = int(np.prod(shape))
    return np.arange(start, start + size, dtype=np.float32).reshape(shape)


def _synthetic_fields(batch_shape: tuple[int, ...] = ()) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    offset = 0
    for name in OBS_FIELD_ORDER:
        shape = (*batch_shape, *OBS_FIELD_SPECS[name])
        fields[name] = _values(shape, offset)
        offset += int(np.prod(OBS_FIELD_SPECS[name]))
    return fields


def _flatten_reference(fields: dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for name in OBS_FIELD_ORDER:
        value = fields[name]
        width = int(np.prod(OBS_FIELD_SPECS[name]))
        parts.append(value.reshape((*value.shape[: -len(OBS_FIELD_SPECS[name])], width)))
    return np.concatenate(parts, axis=-1)


def _synthetic_motion(frames: int = 3) -> G1Motion:
    bodies = len(MUJOCO_BODY_NAMES)
    joint_pos = torch.linspace(-0.2, 0.2, frames * ACTION_DIM).reshape(
        frames,
        ACTION_DIM,
    )
    joint_vel = torch.linspace(0.1, 0.4, frames * ACTION_DIM).reshape(
        frames,
        ACTION_DIM,
    )
    body_pos = torch.zeros(frames, bodies, 3)
    for frame in range(frames):
        for body in range(bodies):
            body_pos[frame, body] = torch.tensor(
                [0.1 * body + 0.01 * frame, -0.02 * body, 0.8 + 0.03 * frame]
            )
    body_quat = torch.zeros(frames, bodies, 4)
    body_quat[..., 0] = 1.0
    body_lin_vel = torch.zeros(frames, bodies, 3)
    body_ang_vel = torch.linspace(-0.3, 0.3, frames * bodies * 3).reshape(
        frames,
        bodies,
        3,
    )
    return G1Motion(
        path=None,
        motion_type="mujoco",
        fps=50.0,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
        contact=torch.zeros(frames, 2),
    )


class MjxObsTest(unittest.TestCase):
    def test_field_specs_sum_to_actor_observation_dim(self) -> None:
        width = sum(int(np.prod(OBS_FIELD_SPECS[name])) for name in OBS_FIELD_ORDER)

        self.assertEqual(width, OBS_DIM)
        self.assertEqual(OBS_FIELD_ORDER[0], "command")
        self.assertEqual(OBS_FIELD_SPECS["command"], (ACTION_DIM * 2,))
        self.assertEqual(
            OBS_FIELD_SPECS["ref_limb_ee_pose_b"],
            (OBS_HISTORY_LENGTH, LIMB_POSE_DIM),
        )

    def test_build_wbc_observation_matches_existing_actor_slice_order(self) -> None:
        fields = _synthetic_fields()

        obs = build_wbc_observation(fields, jnp=_NumpyJnp)

        expected = _flatten_reference(fields)
        self.assertEqual(obs.shape, (OBS_DIM,))
        np.testing.assert_allclose(obs, expected)

        offset = 0
        for name in OBS_FIELD_ORDER:
            width = int(np.prod(OBS_FIELD_SPECS[name]))
            np.testing.assert_allclose(
                obs[offset : offset + width],
                fields[name].reshape(width),
                err_msg=f"bad slice for {name}",
            )
            offset += width

    def test_state_observation_matches_torch_builder_first_frame(self) -> None:
        motion = _synthetic_motion()
        ref_indices = torch.tensor([1])
        default_joint_pos = torch.linspace(-0.05, 0.05, ACTION_DIM)
        torch_builder = G1WbcObservationBuilder(
            motion=motion,
            num_envs=1,
            default_joint_pos=default_joint_pos,
            device="cpu",
        )
        qpos = motion.qpos()[1:2].clone()
        qpos[:, 7:] += 0.03
        qvel = torch.zeros(1, QVEL_DIM)
        qvel[:, 6:] = motion.joint_vel[1:2] + 0.02
        robot = RobotState(
            qpos=qpos,
            qvel=qvel,
            body_pos_w=motion.body_pos_w[1:2] + 0.01,
            body_quat_w=motion.body_quat_w[1:2],
            body_lin_vel_w=motion.body_lin_vel_w[1:2],
            body_ang_vel_w=motion.body_ang_vel_w[1:2],
        )
        last_action = torch.linspace(-0.1, 0.1, ACTION_DIM).view(1, ACTION_DIM)

        torch_obs = torch_builder.compute(robot, ref_indices, last_action)

        command_body_indices = [motion.body_index[name] for name in COMMAND_BODY_NAMES]
        reference = {
            "joint_pos": motion.joint_pos[ref_indices].numpy(),
            "joint_vel": motion.joint_vel[ref_indices].numpy(),
            "body_pos_w": motion.body_pos_w[ref_indices][
                :, command_body_indices
            ].numpy(),
            "body_quat_w": motion.body_quat_w[ref_indices][
                :, command_body_indices
            ].numpy(),
            "body_ang_vel_w": motion.body_ang_vel_w[ref_indices][
                :, command_body_indices
            ].numpy(),
        }
        obs, next_state = build_wbc_observation_from_state(
            robot_state={
                "qpos": robot.qpos.numpy(),
                "qvel": robot.qvel.numpy(),
                "body_pos_w": robot.body_pos_w.numpy(),
                "body_quat_w": robot.body_quat_w.numpy(),
                "body_ang_vel_w": robot.body_ang_vel_w.numpy(),
            },
            reference_state=reference,
            obs_state=JaxObsState(history=None, last_action=last_action.numpy()),
            indices=JaxObsIndices(
                command_body_indices=command_body_indices,
                limb_indices=[
                    COMMAND_BODY_NAMES.index(name) for name in LIMB_EE_BODY_NAMES
                ],
                anchor_index=COMMAND_BODY_NAMES.index(ANCHOR_BODY_NAME),
                tracking_anchor_index=COMMAND_BODY_NAMES.index(
                    TRACKING_ANCHOR_BODY_NAME
                ),
            ),
            default_joint_pos=default_joint_pos.numpy(),
            initialized=False,
            jnp=_NumpyJnp,
        )

        self.assertEqual(obs.shape, (1, OBS_DIM))
        self.assertEqual(
            set(next_state.history),
            set(OBS_FIELD_ORDER) - {"command", "motion_ref_ang_vel"},
        )
        np.testing.assert_allclose(obs, torch_obs.numpy(), atol=1.0e-5, rtol=1.0e-5)

    def test_state_observation_materializes_body_indices_for_jax_take(self) -> None:
        motion = _synthetic_motion()
        ref_indices = torch.tensor([1])
        default_joint_pos = torch.linspace(-0.05, 0.05, ACTION_DIM)
        qpos = motion.qpos()[1:2].clone()
        qvel = torch.zeros(1, QVEL_DIM)
        command_body_indices = [motion.body_index[name] for name in COMMAND_BODY_NAMES]
        reference = {
            "joint_pos": motion.joint_pos[ref_indices].numpy(),
            "joint_vel": motion.joint_vel[ref_indices].numpy(),
            "body_pos_w": motion.body_pos_w[ref_indices][
                :, command_body_indices
            ].numpy(),
            "body_quat_w": motion.body_quat_w[ref_indices][
                :, command_body_indices
            ].numpy(),
            "body_ang_vel_w": motion.body_ang_vel_w[ref_indices][
                :, command_body_indices
            ].numpy(),
        }

        obs, _ = build_wbc_observation_from_state(
            robot_state={
                "qpos": qpos.numpy(),
                "qvel": qvel.numpy(),
                "body_pos_w": motion.body_pos_w[1:2].numpy(),
                "body_quat_w": motion.body_quat_w[1:2].numpy(),
                "body_ang_vel_w": motion.body_ang_vel_w[1:2].numpy(),
            },
            reference_state=reference,
            obs_state=JaxObsState(
                history=None,
                last_action=torch.zeros(1, ACTION_DIM).numpy(),
            ),
            indices=JaxObsIndices(
                command_body_indices=command_body_indices,
                limb_indices=[
                    COMMAND_BODY_NAMES.index(name) for name in LIMB_EE_BODY_NAMES
                ],
                anchor_index=COMMAND_BODY_NAMES.index(ANCHOR_BODY_NAME),
                tracking_anchor_index=COMMAND_BODY_NAMES.index(
                    TRACKING_ANCHOR_BODY_NAME
                ),
            ),
            default_joint_pos=default_joint_pos.numpy(),
            initialized=False,
            jnp=_StrictTakeJnp,
        )

        self.assertEqual(obs.shape, (1, OBS_DIM))

    def test_build_wbc_observation_preserves_batch_axis(self) -> None:
        fields = _synthetic_fields(batch_shape=(2,))

        obs = build_wbc_observation(fields, jnp=_NumpyJnp)

        self.assertEqual(obs.shape, (2, OBS_DIM))
        np.testing.assert_allclose(obs, _flatten_reference(fields))

    def test_build_wbc_observation_accepts_preflattened_history_fields(self) -> None:
        fields = _synthetic_fields()
        fields["ref_limb_ee_pose_b"] = fields["ref_limb_ee_pose_b"].reshape(
            OBS_HISTORY_LENGTH * LIMB_POSE_DIM
        )

        obs = build_wbc_observation(fields, jnp=_NumpyJnp)

        self.assertEqual(obs.shape, (OBS_DIM,))

    def test_build_wbc_observation_rejects_missing_field(self) -> None:
        fields = _synthetic_fields()
        del fields["joint_vel"]

        with self.assertRaisesRegex(KeyError, "joint_vel"):
            build_wbc_observation(fields, jnp=_NumpyJnp)

    def test_build_wbc_observation_rejects_wrong_tail_shape(self) -> None:
        fields = _synthetic_fields()
        fields["actions"] = np.ones((OBS_HISTORY_LENGTH, ACTION_DIM + 1), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "actions"):
            build_wbc_observation(fields, jnp=_NumpyJnp)

    def test_build_wbc_observation_rejects_mismatched_batch_shape(self) -> None:
        fields = _synthetic_fields(batch_shape=(2,))
        fields["motion_ref_ang_vel"] = np.zeros((3,), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "batch shape"):
            build_wbc_observation(fields, jnp=_NumpyJnp)

    def test_update_obs_history_backfills_first_frame(self) -> None:
        obs = np.linspace(-1.0, 1.0, 4, dtype=np.float32)

        history = update_obs_history(None, obs, initialized=False, jnp=_NumpyJnp)

        self.assertEqual(history.shape[-2:], (OBS_HISTORY_LENGTH, 4))
        np.testing.assert_allclose(
            history,
            np.repeat(obs[None, :], OBS_HISTORY_LENGTH, axis=0),
        )

    def test_update_obs_history_shifts_left_and_inserts_latest(self) -> None:
        obs = np.array([9.0, 10.0, 11.0], dtype=np.float32)
        history = np.arange(OBS_HISTORY_LENGTH * 3, dtype=np.float32).reshape(
            OBS_HISTORY_LENGTH,
            3,
        )

        updated = update_obs_history(history, obs, initialized=True, jnp=_NumpyJnp)

        self.assertEqual(updated.shape[-2:], (OBS_HISTORY_LENGTH, 3))
        np.testing.assert_allclose(updated[:-1], history[1:])
        np.testing.assert_allclose(updated[-1], obs)

    def test_update_obs_history_supports_batched_mixed_initialized_mask(self) -> None:
        obs = np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32)
        history = np.arange(2 * OBS_HISTORY_LENGTH * 2, dtype=np.float32).reshape(
            2,
            OBS_HISTORY_LENGTH,
            2,
        )
        initialized = np.array([True, False])

        updated = update_obs_history(history, obs, initialized=initialized, jnp=_NumpyJnp)

        expected_first = np.concatenate([history[0, 1:], obs[0][None, :]], axis=0)
        expected_second = np.repeat(obs[1][None, :], OBS_HISTORY_LENGTH, axis=0)
        np.testing.assert_allclose(updated[0], expected_first)
        np.testing.assert_allclose(updated[1], expected_second)

    def test_update_obs_history_rejects_wrong_history_shape(self) -> None:
        obs = np.zeros(3, dtype=np.float32)
        bad_history = np.zeros((OBS_HISTORY_LENGTH - 1, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "Expected history shape"):
            update_obs_history(bad_history, obs, initialized=True, jnp=_NumpyJnp)


if __name__ == "__main__":
    unittest.main()
