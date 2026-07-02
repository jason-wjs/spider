from types import SimpleNamespace
import unittest

import numpy as np
import torch

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    COMMAND_BODY_NAMES,
    KNEES_BENT_JOINT_POS,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    QPOS_DIM,
    QVEL_DIM,
    TASK_EE_BODY_NAMES,
)
from spider.tasks.g1_wbc.mjx_obs import JaxObsIndices
from spider.tasks.g1_wbc.mjx_reference import build_mjx_rollout_reference
from spider.tasks.g1_wbc.motion import G1Motion


class _NumpyJnp:
    @staticmethod
    def asarray(value):
        return np.asarray(value, dtype=np.float32)


class _Runtime:
    jnp = _NumpyJnp()


def _motion(frames: int = 6) -> G1Motion:
    bodies = len(MUJOCO_BODY_NAMES)
    qpos = torch.zeros(frames, QPOS_DIM)
    qpos[:, 0] = torch.arange(frames, dtype=torch.float32)
    qpos[:, 3] = 1.0
    qpos[:, 7:] = torch.arange(frames * ACTION_DIM, dtype=torch.float32).view(
        frames,
        ACTION_DIM,
    )
    body_pos = torch.arange(frames * bodies * 3, dtype=torch.float32).view(
        frames,
        bodies,
        3,
    )
    body_quat = torch.zeros(frames, bodies, 4)
    body_quat[..., 0] = 1.0
    body_ang_vel = torch.full((frames, bodies, 3), 0.25)
    return G1Motion(
        path=None,
        motion_type="mujoco",
        fps=50.0,
        joint_pos=qpos[:, 7:].clone(),
        joint_vel=torch.ones(frames, ACTION_DIM),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=torch.zeros(frames, bodies, 3),
        body_ang_vel_w=body_ang_vel,
        contact=torch.tensor(
            [[index % 2, (index + 1) % 2] for index in range(frames)],
            dtype=torch.float32,
        ),
    )


def _model_bundle():
    low = np.linspace(-1.0, -0.1, ACTION_DIM, dtype=np.float32)
    high = np.linspace(0.1, 1.0, ACTION_DIM, dtype=np.float32)
    return SimpleNamespace(
        cpu_model=SimpleNamespace(
            jnt_limited=np.ones(ACTION_DIM, dtype=np.int32),
            jnt_range=np.stack([low, high], axis=-1),
        ),
        joint_name_to_id={
            f"robot/{joint_name}": index
            for index, joint_name in enumerate(MUJOCO_JOINT_NAMES)
        },
    )


class MjxReferenceTest(unittest.TestCase):
    def test_build_rollout_reference_matches_motion_window_shapes(self) -> None:
        motion = _motion(frames=6)
        controls = torch.zeros(4, QPOS_DIM - 1)
        controls[0, 0] = 0.4

        reference = build_mjx_rollout_reference(
            start=2,
            motion=motion,
            controls=controls,
            actor_params=object(),
            model_bundle=_model_bundle(),
            runtime=_Runtime(),
        )

        self.assertEqual(reference["initial_robot_state"]["qpos"].shape, (QPOS_DIM,))
        self.assertEqual(reference["initial_robot_state"]["qvel"].shape, (QVEL_DIM,))
        np.testing.assert_allclose(
            reference["initial_robot_state"]["qpos"],
            motion.qpos()[2].numpy(),
        )
        self.assertEqual(
            reference["initial_robot_state"]["body_pos_w"].shape,
            (len(MUJOCO_BODY_NAMES), 3),
        )
        self.assertEqual(reference["obs_reference"]["joint_pos"].shape, (4, ACTION_DIM))
        self.assertEqual(reference["obs_reference"]["joint_vel"].shape, (4, ACTION_DIM))
        self.assertEqual(
            reference["obs_reference"]["body_pos_w"].shape,
            (4, len(COMMAND_BODY_NAMES), 3),
        )
        self.assertEqual(
            reference["obs_reference"]["body_quat_w"].shape,
            (4, len(COMMAND_BODY_NAMES), 4),
        )
        self.assertEqual(
            reference["score_reference"]["body_pos"].shape,
            (4, len(MUJOCO_BODY_NAMES), 3),
        )
        self.assertEqual(
            reference["score_reference"]["ee_pos"].shape,
            (4, len(TASK_EE_BODY_NAMES), 3),
        )
        self.assertEqual(reference["score_reference"]["contact"].shape, (4, 2))
        self.assertEqual(reference["prev_control"].shape, (QPOS_DIM - 1,))
        self.assertAlmostEqual(float(reference["prev_control"][0]), 0.4)
        self.assertEqual(reference["joint_low"].shape, (ACTION_DIM,))
        self.assertEqual(reference["joint_high"].shape, (ACTION_DIM,))
        self.assertIsInstance(reference["obs_indices"], JaxObsIndices)
        self.assertFalse(reference["obs_initialized"])

    def test_rollout_reference_clamps_window_indices_at_motion_end(self) -> None:
        motion = _motion(frames=4)

        reference = build_mjx_rollout_reference(
            start=2,
            motion=motion,
            controls=torch.zeros(5, QPOS_DIM - 1),
            actor_params=object(),
            model_bundle=_model_bundle(),
            runtime=_Runtime(),
        )

        expected = motion.qpos()[[2, 3, 3, 3, 3], 0].numpy()
        np.testing.assert_allclose(
            reference["score_reference"]["root_pos"][:, 0],
            expected,
        )

    def test_default_joint_pos_matches_wbc_knees_bent_pose(self) -> None:
        reference = build_mjx_rollout_reference(
            start=0,
            motion=_motion(),
            controls=torch.zeros(3, QPOS_DIM - 1),
            actor_params=object(),
            model_bundle=_model_bundle(),
            runtime=_Runtime(),
        )

        default_joint_pos = reference["default_joint_pos"]
        for joint_name, value in KNEES_BENT_JOINT_POS.items():
            joint_index = MUJOCO_JOINT_NAMES.index(joint_name)
            self.assertAlmostEqual(float(default_joint_pos[joint_index]), float(value))


if __name__ == "__main__":
    unittest.main()
