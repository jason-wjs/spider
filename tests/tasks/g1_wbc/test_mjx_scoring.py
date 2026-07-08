from types import SimpleNamespace
import unittest

import numpy as np
import torch

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    ANCHOR_BODY_NAME,
    HAND_EE_BODY_NAMES,
    MUJOCO_BODY_NAMES,
    POLICY_DT,
    QPOS_DIM,
    QVEL_DIM,
    TASK_EE_BODY_NAMES,
)
from spider.tasks.g1_wbc.metrics import compute_rollout_scores
from spider.tasks.g1_wbc.mjx_scoring import (
    ACCUMULATOR_KEYS,
    JaxScoreWeights,
    finalize_score,
    init_score_accumulator,
    score_step,
)
from spider.tasks.g1_wbc.motion import G1Motion


class _NumpyJnp:
    @staticmethod
    def asarray(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def zeros(shape):
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def mean(value, axis=None):
        return np.mean(value, axis=axis)

    @staticmethod
    def abs(value):
        return np.abs(value)

    @staticmethod
    def maximum(x, y):
        return np.maximum(x, y)

    @staticmethod
    def max(value, axis=None):
        return np.max(value, axis=axis)

    @staticmethod
    def argmax(value, axis=None):
        return np.argmax(value, axis=axis)

    @staticmethod
    def arange(stop):
        return np.arange(stop)

    @staticmethod
    def sum(value, axis=None, keepdims=False):
        return np.sum(value, axis=axis, keepdims=keepdims)

    @staticmethod
    def sqrt(value):
        return np.sqrt(value)

    @staticmethod
    def clip(value, low, high):
        return np.clip(value, low, high)

    @staticmethod
    def arccos(value):
        return np.arccos(value)

    @staticmethod
    def take(value, indices, axis=None):
        return np.take(value, indices, axis=axis)

    @staticmethod
    def expand_dims(value, axis):
        return np.expand_dims(value, axis=axis)

    @staticmethod
    def zeros_like(value):
        return np.zeros_like(value, dtype=np.float32)

    @staticmethod
    def concatenate(values, axis=0):
        return np.concatenate(values, axis=axis)

    @staticmethod
    def stack(values, axis=0):
        return np.stack(values, axis=axis)

    @staticmethod
    def where(condition, x, y):
        return np.where(condition, x, y)

    @staticmethod
    def cross(x, y, axis=None):
        return np.cross(x, y, axis=axis)


def _step_state(offset: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "root_pos": np.array([1.0 + offset, 2.0, 3.0], dtype=np.float32),
        "body_pos": np.array(
            [[0.0, 1.0 + offset, 2.0], [3.0, 4.0, 5.0]],
            dtype=np.float32,
        ),
        "ee_pos": np.array([[1.0, -1.0 + offset, 0.5]], dtype=np.float32),
        "contact": np.array([1.0, 0.0], dtype=np.float32),
        "control": np.array([0.2, 0.4, 0.6], dtype=np.float32),
        "prev_control": np.array([0.1, 0.5, 0.3], dtype=np.float32),
        "joint_vel": np.array([-0.2, 0.1, 0.3], dtype=np.float32),
        "prev_joint_vel": np.array([-0.1, 0.0, 0.5], dtype=np.float32),
        "prev_joint_acc": np.array([0.05, -0.1, 0.2], dtype=np.float32),
    }


def _reference_state() -> dict[str, np.ndarray]:
    return {
        "root_pos": np.array([1.5, 1.0, 3.0], dtype=np.float32),
        "body_pos": np.array(
            [[0.0, 2.0, 2.0], [2.0, 4.0, 7.0]],
            dtype=np.float32,
        ),
        "ee_pos": np.array([[0.5, -1.0, 1.5]], dtype=np.float32),
        "contact": np.array([0.0, 0.0], dtype=np.float32),
    }


def _expected_terms(step_state, reference_state) -> dict[str, float]:
    root_error = np.linalg.norm(step_state["root_pos"] - reference_state["root_pos"])
    body_error = np.mean(
        np.linalg.norm(step_state["body_pos"] - reference_state["body_pos"], axis=-1)
    )
    ee_error = np.mean(
        np.linalg.norm(step_state["ee_pos"] - reference_state["ee_pos"], axis=-1)
    )
    contact_error = np.mean(np.abs(step_state["contact"] - reference_state["contact"]))
    control_delta = np.linalg.norm(step_state["control"] - step_state["prev_control"])
    joint_acc_delta = step_state["joint_vel"] - step_state["prev_joint_vel"]
    joint_acc = np.linalg.norm(joint_acc_delta) / POLICY_DT
    joint_jerk = (
        np.linalg.norm(joint_acc_delta - step_state["prev_joint_acc"]) / POLICY_DT
    )
    return {
        "root_pos_error_mean": float(root_error),
        "body_global_pos_error_mean": float(body_error),
        "ee_global_pos_error_mean": float(ee_error),
        "contact_mismatch_rate": float(contact_error),
        "control_delta_mean": float(control_delta),
        "joint_acc_mean": float(joint_acc),
        "joint_jerk_mean": float(joint_jerk),
    }


class MjxScoringTest(unittest.TestCase):
    def test_score_step_matches_numpy_expected_terms(self) -> None:
        step_state = _step_state()
        reference_state = _reference_state()
        weights = JaxScoreWeights(
            {
                "root_pos": 1.5,
                "body_global_pos": 4.0,
                "ee_global_pos": 3.0,
                "contact": 2.0,
                "control_delta": 0.5,
                "joint_acc": 0.25,
            }
        )

        accumulator = init_score_accumulator((), jnp=_NumpyJnp)
        accumulator = score_step(
            accumulator,
            step_state,
            reference_state,
            weights,
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        expected = _expected_terms(step_state, reference_state)
        for name, value in expected.items():
            self.assertAlmostEqual(float(metrics[name]), value, places=6)

        expected_penalty = (
            1.5 * expected["root_pos_error_mean"]
            + 4.0 * expected["body_global_pos_error_mean"]
            + 3.0 * expected["ee_global_pos_error_mean"]
            + 2.0 * expected["contact_mismatch_rate"]
            + 0.5 * expected["control_delta_mean"]
            + 0.25 * expected["joint_acc_mean"]
        )
        self.assertAlmostEqual(float(metrics["score"]), -expected_penalty, places=6)

    def test_score_step_accumulates_joint_jerk_from_previous_joint_acc(self) -> None:
        step_state = {
            **_step_state(),
            "joint_vel": np.array([0.3, -0.2, 0.5], dtype=np.float32),
            "prev_joint_vel": np.array([0.1, -0.2, 0.1], dtype=np.float32),
            "prev_joint_acc": np.array([0.05, 0.0, -0.1], dtype=np.float32),
        }
        reference_state = _reference_state()
        weights = JaxScoreWeights({"joint_jerk": 0.2})

        accumulator = score_step(
            init_score_accumulator((), jnp=_NumpyJnp),
            step_state,
            reference_state,
            weights,
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        joint_acc_delta = step_state["joint_vel"] - step_state["prev_joint_vel"]
        expected = np.linalg.norm(joint_acc_delta - step_state["prev_joint_acc"])
        expected /= POLICY_DT
        self.assertAlmostEqual(float(metrics["joint_jerk_mean"]), expected, places=6)
        self.assertEqual(metrics["joint_jerk"], metrics["joint_jerk_mean"])
        self.assertAlmostEqual(float(metrics["score"]), -0.2 * expected, places=6)

    def test_score_step_penalizes_only_active_contact_force(self) -> None:
        step_state = {
            **_step_state(),
            "contact": np.array([1.0, 0.0], dtype=np.float32),
            "contact_force": np.array([600.0, 100.0], dtype=np.float32),
        }
        reference_state = {
            **_reference_state(),
            "contact": np.array([1.0, 0.0], dtype=np.float32),
        }

        accumulator = score_step(
            init_score_accumulator((), jnp=_NumpyJnp),
            step_state,
            reference_state,
            JaxScoreWeights({"contact_force_active": 0.5}),
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        self.assertAlmostEqual(float(metrics["contact_force_active_mean"]), 600.0)
        self.assertAlmostEqual(float(metrics["contact_force_active"]), 2.0)
        self.assertAlmostEqual(float(metrics["score"]), -1.0)

    def test_contact_force_active_weight_zero_keeps_score_unchanged(self) -> None:
        step_state = {
            **_step_state(),
            "contact_force": np.array([600.0, 100.0], dtype=np.float32),
        }
        reference_state = _reference_state()

        accumulator = score_step(
            init_score_accumulator((), jnp=_NumpyJnp),
            step_state,
            reference_state,
            JaxScoreWeights({}),
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        self.assertAlmostEqual(float(metrics["contact_force_active_mean"]), 600.0)
        self.assertAlmostEqual(float(metrics["contact_force_active"]), 2.0)
        self.assertAlmostEqual(float(metrics["score"]), 0.0)

    def test_score_step_penalizes_contact_force_peak_excess(self) -> None:
        step_state = {
            **_step_state(),
            "contact_force": np.array([600.0, 100.0], dtype=np.float32),
        }

        accumulator = score_step(
            init_score_accumulator((), jnp=_NumpyJnp),
            step_state,
            _reference_state(),
            JaxScoreWeights({"contact_force_peak_excess": 0.5}),
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        self.assertAlmostEqual(float(metrics["contact_force_peak"]), 600.0)
        self.assertAlmostEqual(float(metrics["contact_force_peak_excess_mean"]), 1.0)
        self.assertAlmostEqual(float(metrics["contact_force_peak_excess"]), 1.0)
        self.assertAlmostEqual(float(metrics["score"]), -0.5)

    def test_score_step_accumulates_rotation_error_terms(self) -> None:
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        half_turn_x = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        step_state = {
            **_step_state(),
            "root_quat": half_turn_x,
            "body_quat": np.stack([half_turn_x, identity], axis=0),
            "ee_quat": np.stack([half_turn_x], axis=0),
        }
        reference_state = {
            **_reference_state(),
            "root_quat": identity,
            "body_quat": np.stack([identity, identity], axis=0),
            "ee_quat": np.stack([identity], axis=0),
        }
        weights = JaxScoreWeights(
            {
                "root_rot": 0.5,
                "body_global_rot": 0.8,
                "ee_global_rot": 0.3,
            }
        )

        accumulator = score_step(
            init_score_accumulator((), jnp=_NumpyJnp),
            step_state,
            reference_state,
            weights,
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        self.assertAlmostEqual(
            float(metrics["root_rot_error_mean"]),
            np.pi,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["body_global_rot_error_mean"]),
            np.pi / 2.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["ee_global_rot_error_mean"]),
            np.pi,
            places=6,
        )
        expected_penalty = 0.5 * np.pi + 0.8 * (np.pi / 2.0) + 0.3 * np.pi
        self.assertAlmostEqual(float(metrics["score"]), -expected_penalty, places=6)

    def test_score_step_accumulates_joint_and_local_body_hand_terms(self) -> None:
        body_count = len(MUJOCO_BODY_NAMES)
        anchor_index = MUJOCO_BODY_NAMES.index(ANCHOR_BODY_NAME)
        local_indices = [
            index
            for index, name in enumerate(MUJOCO_BODY_NAMES)
            if name != ANCHOR_BODY_NAME
        ]
        ee_indices = [MUJOCO_BODY_NAMES.index(name) for name in TASK_EE_BODY_NAMES]
        hand_indices = [MUJOCO_BODY_NAMES.index(name) for name in HAND_EE_BODY_NAMES]
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        half_turn_x = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        step_body_pos = np.zeros((body_count, 3), dtype=np.float32)
        ref_body_pos = np.zeros((body_count, 3), dtype=np.float32)
        step_body_quat = np.tile(identity, (body_count, 1))
        ref_body_quat = np.tile(identity, (body_count, 1))
        step_body_pos[anchor_index] = np.array([2.0, -1.0, 0.5], dtype=np.float32)
        ref_body_pos[anchor_index] = step_body_pos[anchor_index]
        for body_index in hand_indices:
            step_body_pos[body_index] = step_body_pos[anchor_index] + np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            )
            ref_body_pos[body_index] = ref_body_pos[anchor_index]
            step_body_quat[body_index] = half_turn_x
        for body_index in ee_indices:
            if body_index not in hand_indices:
                step_body_pos[body_index] = step_body_pos[anchor_index] + np.array(
                    [0.0, 2.0, 0.0],
                    dtype=np.float32,
                )
                ref_body_pos[body_index] = ref_body_pos[anchor_index]

        step_state = {
            **_step_state(),
            "joint_pos": np.full(ACTION_DIM, 0.5, dtype=np.float32),
            "body_pos": step_body_pos,
            "body_quat": step_body_quat,
            "ee_pos": step_body_pos[ee_indices],
            "ee_quat": step_body_quat[ee_indices],
        }
        reference_state = {
            **_reference_state(),
            "joint_pos": np.zeros(ACTION_DIM, dtype=np.float32),
            "body_pos": ref_body_pos,
            "body_quat": ref_body_quat,
            "ee_pos": ref_body_pos[ee_indices],
            "ee_quat": ref_body_quat[ee_indices],
        }
        weights = JaxScoreWeights(
            {
                "joint_pos": 0.5,
                "body_local_pos": 2.0,
                "ee_local_pos": 3.0,
                "hand_global_pos": 5.0,
                "hand_global_rot": 0.25,
                "hand_local_pos": 7.0,
                "hand_local_rot": 0.75,
            }
        )

        accumulator = score_step(
            init_score_accumulator((), jnp=_NumpyJnp),
            step_state,
            reference_state,
            weights,
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        pos_delta = step_body_pos - ref_body_pos
        expected_joint_pos = float(np.linalg.norm(step_state["joint_pos"]))
        expected_body_local_pos = float(
            np.mean(np.linalg.norm(pos_delta[local_indices], axis=-1))
        )
        expected_ee_local_pos = float(
            np.mean(np.linalg.norm(pos_delta[ee_indices], axis=-1))
        )
        expected_hand_pos = float(
            np.mean(np.linalg.norm(pos_delta[hand_indices], axis=-1))
        )
        expected_hand_rot = np.pi
        self.assertAlmostEqual(
            float(metrics["joint_pos_error_mean"]),
            expected_joint_pos,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["body_local_pos_error_mean"]),
            expected_body_local_pos,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["ee_local_pos_error_mean"]),
            expected_ee_local_pos,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["hand_global_pos_error_mean"]),
            expected_hand_pos,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["hand_local_pos_error_mean"]),
            expected_hand_pos,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["hand_global_rot_error_mean"]),
            expected_hand_rot,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["hand_local_rot_error_mean"]),
            expected_hand_rot,
            places=6,
        )
        expected_penalty = (
            0.5 * expected_joint_pos
            + 2.0 * expected_body_local_pos
            + 3.0 * expected_ee_local_pos
            + 5.0 * expected_hand_pos
            + 0.25 * expected_hand_rot
            + 7.0 * expected_hand_pos
            + 0.75 * expected_hand_rot
        )
        self.assertAlmostEqual(float(metrics["score"]), -expected_penalty, places=5)

    def test_score_step_matches_torch_rollout_terms_for_v14_pose_surface(self) -> None:
        frames = 3
        body_count = len(MUJOCO_BODY_NAMES)
        anchor_index = MUJOCO_BODY_NAMES.index(ANCHOR_BODY_NAME)
        ee_indices = [MUJOCO_BODY_NAMES.index(name) for name in TASK_EE_BODY_NAMES]
        hand_indices = [MUJOCO_BODY_NAMES.index(name) for name in HAND_EE_BODY_NAMES]
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0])
        half_turn_x = torch.tensor([0.0, 1.0, 0.0, 0.0])

        ref_body_pos = torch.zeros(frames, 1, body_count, 3)
        body_pos = ref_body_pos.clone()
        ref_body_quat = identity.view(1, 1, 1, 4).repeat(frames, 1, body_count, 1)
        body_quat = ref_body_quat.clone()
        body_pos[..., anchor_index, :] = torch.tensor([1.0, -2.0, 0.5])
        ref_body_pos[..., anchor_index, :] = body_pos[..., anchor_index, :]
        for body_index in hand_indices:
            body_pos[..., body_index, :] = body_pos[..., anchor_index, :] + torch.tensor(
                [0.5, 0.0, 0.0]
            )
            ref_body_pos[..., body_index, :] = ref_body_pos[..., anchor_index, :]
            body_quat[..., body_index, :] = half_turn_x
        for body_index in ee_indices:
            if body_index not in hand_indices:
                body_pos[..., body_index, :] = body_pos[
                    ..., anchor_index, :
                ] + torch.tensor([0.0, 0.25, 0.0])
                ref_body_pos[..., body_index, :] = ref_body_pos[..., anchor_index, :]

        qpos = torch.zeros(frames, 1, QPOS_DIM)
        qpos[..., :3] = body_pos[..., anchor_index, :]
        qpos[..., 3:7] = body_quat[..., anchor_index, :]
        qpos[..., 7:] = 0.25
        ref_joint_pos = torch.zeros(frames, ACTION_DIM)
        contact = torch.zeros(frames, 2)
        motion = G1Motion(
            path=None,
            motion_type="mujoco",
            fps=50.0,
            joint_pos=ref_joint_pos,
            joint_vel=torch.zeros(frames, ACTION_DIM),
            body_pos_w=ref_body_pos[:, 0],
            body_quat_w=ref_body_quat[:, 0],
            body_lin_vel_w=torch.zeros(frames, body_count, 3),
            body_ang_vel_w=torch.zeros(frames, body_count, 3),
            contact=contact,
        )
        rollout = SimpleNamespace(
            qpos=qpos,
            qvel=torch.zeros(frames, 1, QVEL_DIM),
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            body_lin_vel_w=torch.zeros(frames, 1, body_count, 3),
            body_ang_vel_w=torch.zeros(frames, 1, body_count, 3),
            actions=torch.zeros(frames, 1, ACTION_DIM),
            controls=torch.zeros(frames, 1, ACTION_DIM),
            contact_indicator=torch.zeros(frames, 1, 2),
            contact_force=torch.zeros(frames, 1, 2),
            floor_contact_indicator=torch.zeros(frames, 1, 3),
            floor_contact_force=torch.zeros(frames, 1, 3),
            ref_indices=torch.arange(frames).view(frames, 1),
            dt=POLICY_DT,
        )
        _, torch_terms = compute_rollout_scores(motion, rollout)
        weights = JaxScoreWeights(
            {
                "joint_pos": 1.0,
                "body_global_pos": 1.0,
                "body_global_rot": 1.0,
                "body_local_pos": 1.0,
                "body_local_rot": 1.0,
                "ee_global_pos": 1.0,
                "ee_global_rot": 1.0,
                "ee_local_pos": 1.0,
                "ee_local_rot": 1.0,
                "hand_global_pos": 1.0,
                "hand_global_rot": 1.0,
                "hand_local_pos": 1.0,
                "hand_local_rot": 1.0,
            }
        )
        accumulator = init_score_accumulator((1,), jnp=_NumpyJnp)
        for frame in range(frames):
            step_state = {
                "root_pos": qpos[frame, :, :3].numpy(),
                "root_quat": qpos[frame, :, 3:7].numpy(),
                "joint_pos": qpos[frame, :, 7:].numpy(),
                "body_pos": body_pos[frame].numpy(),
                "body_quat": body_quat[frame].numpy(),
                "ee_pos": body_pos[frame, :, ee_indices, :].numpy(),
                "ee_quat": body_quat[frame, :, ee_indices, :].numpy(),
                "contact": np.zeros((1, 2), dtype=np.float32),
                "control": np.zeros((1, ACTION_DIM), dtype=np.float32),
                "prev_control": np.zeros((1, ACTION_DIM), dtype=np.float32),
                "joint_vel": np.zeros((1, ACTION_DIM), dtype=np.float32),
                "prev_joint_vel": np.zeros((1, ACTION_DIM), dtype=np.float32),
                "prev_joint_acc": np.zeros((1, ACTION_DIM), dtype=np.float32),
            }
            reference_state = {
                "root_pos": motion.qpos()[frame : frame + 1, :3].numpy(),
                "root_quat": motion.qpos()[frame : frame + 1, 3:7].numpy(),
                "joint_pos": motion.qpos()[frame : frame + 1, 7:].numpy(),
                "body_pos": ref_body_pos[frame].numpy(),
                "body_quat": ref_body_quat[frame].numpy(),
                "ee_pos": ref_body_pos[frame, :, ee_indices, :].numpy(),
                "ee_quat": ref_body_quat[frame, :, ee_indices, :].numpy(),
                "contact": np.zeros((1, 2), dtype=np.float32),
            }
            accumulator = score_step(
                accumulator,
                step_state,
                reference_state,
                weights,
                jnp=_NumpyJnp,
            )
        mjx_metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        for name in (
            "joint_pos_error",
            "body_global_pos_error",
            "body_global_rot_error",
            "body_local_pos_error",
            "body_local_rot_error",
            "ee_global_pos_error",
            "ee_global_rot_error",
            "ee_local_pos_error",
            "ee_local_rot_error",
            "hand_global_pos_error",
            "hand_global_rot_error",
            "hand_local_pos_error",
            "hand_local_rot_error",
        ):
            np.testing.assert_allclose(
                mjx_metrics[name],
                torch_terms[name].numpy(),
                rtol=1.0e-5,
                atol=1.0e-6,
            )

    def test_score_step_accumulates_contact_classification_and_action_delta(
        self,
    ) -> None:
        step_state = {
            **_step_state(),
            "contact": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "action": np.array([0.2, -0.4], dtype=np.float32),
            "prev_action": np.array([-0.1, 0.1], dtype=np.float32),
        }
        reference_state = {
            **_reference_state(),
            "contact": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        }
        weights = JaxScoreWeights(
            {
                "contact": 2.0,
                "contact_false_positive": 3.0,
                "contact_false_negative": 5.0,
                "action_delta": 7.0,
            }
        )

        accumulator = score_step(
            init_score_accumulator((), jnp=_NumpyJnp),
            step_state,
            reference_state,
            weights,
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        self.assertAlmostEqual(float(metrics["contact_mismatch_rate"]), 0.5)
        self.assertAlmostEqual(float(metrics["contact_false_positive"]), 0.25)
        self.assertAlmostEqual(float(metrics["contact_false_negative"]), 0.25)
        expected_action_delta = float(
            np.linalg.norm(step_state["action"] - step_state["prev_action"])
        )
        self.assertAlmostEqual(float(metrics["action_delta_mean"]), expected_action_delta)
        expected_penalty = (
            2.0 * 0.5
            + 3.0 * 0.25
            + 5.0 * 0.25
            + 7.0 * expected_action_delta
        )
        self.assertAlmostEqual(float(metrics["score"]), -expected_penalty, places=6)

    def test_score_step_accumulates_contact_switch_from_previous_contact(self) -> None:
        step_state = {
            **_step_state(),
            "contact": np.array([1.0, 0.0, 1.0], dtype=np.float32),
            "prev_contact": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }
        reference_state = {
            **_reference_state(),
            "contact": np.array([1.0, 0.0, 1.0], dtype=np.float32),
        }
        weights = JaxScoreWeights({"contact_switch": 2.0})

        accumulator = score_step(
            init_score_accumulator((), jnp=_NumpyJnp),
            step_state,
            reference_state,
            weights,
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        expected = np.linalg.norm(step_state["contact"] - step_state["prev_contact"])
        self.assertAlmostEqual(float(metrics["contact_switch_rate"]), expected)
        self.assertEqual(metrics["contact_switch"], metrics["contact_switch_rate"])
        self.assertAlmostEqual(float(metrics["score"]), -2.0 * expected)

    def test_score_step_accumulates_bad_floor_contact_from_third_channel(self) -> None:
        step_state = {
            **_step_state(),
            "floor_contact": np.array([1.0, 0.0, 1.0], dtype=np.float32),
        }
        weights = JaxScoreWeights({"bad_floor_contact": 3.0})

        accumulator = score_step(
            init_score_accumulator((), jnp=_NumpyJnp),
            step_state,
            _reference_state(),
            weights,
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        self.assertAlmostEqual(float(metrics["bad_floor_contact_rate"]), 1.0)
        self.assertEqual(
            metrics["bad_floor_contact"],
            metrics["bad_floor_contact_rate"],
        )
        self.assertEqual(
            metrics["bad_floor_force_excess"],
            metrics["bad_floor_force_excess_mean"],
        )
        self.assertEqual(
            metrics["contact_force_delta"],
            metrics["contact_force_delta_mean"],
        )
        self.assertAlmostEqual(float(metrics["score"]), -3.0)

    def test_score_step_accumulates_contact_force_terms(self) -> None:
        step_state = {
            **_step_state(),
            "contact_force": np.array([600.0, 0.0], dtype=np.float32),
            "prev_contact_force": np.array([300.0, 400.0], dtype=np.float32),
            "prev_contact_force_valid": np.array(1.0, dtype=np.float32),
            "floor_contact_force": np.array([10.0, 20.0, 600.0], dtype=np.float32),
        }
        weights = JaxScoreWeights(
            {
                "contact_force_delta": 1.5,
                "bad_floor_force_excess": 2.0,
            }
        )

        accumulator = score_step(
            init_score_accumulator((), jnp=_NumpyJnp),
            step_state,
            _reference_state(),
            weights,
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        expected_delta = (
            np.linalg.norm(step_state["contact_force"] - step_state["prev_contact_force"])
            / 300.0
        )
        self.assertAlmostEqual(float(metrics["contact_force_delta_mean"]), expected_delta)
        self.assertEqual(
            metrics["contact_force_delta"],
            metrics["contact_force_delta_mean"],
        )
        self.assertAlmostEqual(float(metrics["bad_floor_force_excess_mean"]), 1.0)
        self.assertEqual(
            metrics["bad_floor_force_excess"],
            metrics["bad_floor_force_excess_mean"],
        )
        expected_penalty = 1.5 * expected_delta + 2.0
        self.assertAlmostEqual(float(metrics["score"]), -expected_penalty)

    def test_finalize_score_returns_compute_rollout_scores_term_aliases(self) -> None:
        step_state = _step_state()
        reference_state = _reference_state()
        accumulator = init_score_accumulator((), jnp=_NumpyJnp)
        accumulator = score_step(
            accumulator,
            step_state,
            reference_state,
            JaxScoreWeights({"root_pos_error": 1.0}),
            jnp=_NumpyJnp,
        )

        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        self.assertEqual(metrics["root_pos_error"], metrics["root_pos_error_mean"])
        self.assertEqual(
            metrics["body_global_pos_error"],
            metrics["body_global_pos_error_mean"],
        )
        self.assertEqual(
            metrics["body_local_pos_error"],
            metrics["body_local_pos_error_mean"],
        )
        self.assertEqual(
            metrics["body_local_rot_error"],
            metrics["body_local_rot_error_mean"],
        )
        self.assertEqual(
            metrics["ee_global_pos_error"],
            metrics["ee_global_pos_error_mean"],
        )
        self.assertEqual(
            metrics["ee_local_pos_error"],
            metrics["ee_local_pos_error_mean"],
        )
        self.assertEqual(
            metrics["ee_local_rot_error"],
            metrics["ee_local_rot_error_mean"],
        )
        self.assertEqual(
            metrics["hand_global_pos_error"],
            metrics["hand_global_pos_error_mean"],
        )
        self.assertEqual(
            metrics["hand_global_rot_error"],
            metrics["hand_global_rot_error_mean"],
        )
        self.assertEqual(
            metrics["hand_local_pos_error"],
            metrics["hand_local_pos_error_mean"],
        )
        self.assertEqual(
            metrics["hand_local_rot_error"],
            metrics["hand_local_rot_error_mean"],
        )
        self.assertEqual(metrics["contact_mismatch"], metrics["contact_mismatch_rate"])
        self.assertEqual(metrics["control_delta"], metrics["control_delta_mean"])
        self.assertEqual(metrics["joint_acc"], metrics["joint_acc_mean"])
        self.assertEqual(metrics["joint_jerk"], metrics["joint_jerk_mean"])
        self.assertEqual(metrics["contact_switch"], metrics["contact_switch_rate"])
        self.assertEqual(
            metrics["bad_floor_contact"],
            metrics["bad_floor_contact_rate"],
        )

    def test_accumulator_averages_multiple_steps(self) -> None:
        reference_state = _reference_state()
        first = _step_state(offset=0.0)
        second = _step_state(offset=0.3)
        accumulator = init_score_accumulator((), jnp=_NumpyJnp)

        accumulator = score_step(
            accumulator,
            first,
            reference_state,
            JaxScoreWeights({"root_pos": 1.0}),
            jnp=_NumpyJnp,
        )
        accumulator = score_step(
            accumulator,
            second,
            reference_state,
            JaxScoreWeights({"root_pos": 1.0}),
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        expected_root = (
            _expected_terms(first, reference_state)["root_pos_error_mean"]
            + _expected_terms(second, reference_state)["root_pos_error_mean"]
        ) / 2.0
        self.assertAlmostEqual(
            float(metrics["root_pos_error_mean"]),
            expected_root,
            places=6,
        )
        self.assertAlmostEqual(float(metrics["score"]), -expected_root, places=6)

    def test_accumulator_tracks_contact_diagnostic_maxima(self) -> None:
        reference_state = _reference_state()
        accumulator = init_score_accumulator((2,), jnp=_NumpyJnp)
        base_step = {
            name: np.stack([value, value], axis=0)
            for name, value in _step_state().items()
        }
        batched_reference = {
            name: np.stack([value, value], axis=0)
            for name, value in reference_state.items()
        }
        first = {
            **base_step,
            "active_contact_count": np.array([1.0, 4.0], dtype=np.float32),
            "contact_pair_count": np.array([3.0, 2.0], dtype=np.float32),
        }
        second = {
            **base_step,
            "active_contact_count": np.array([5.0, 2.0], dtype=np.float32),
            "contact_pair_count": np.array([4.0, 8.0], dtype=np.float32),
        }

        accumulator = score_step(
            accumulator,
            first,
            batched_reference,
            JaxScoreWeights({"root_pos": 1.0}),
            jnp=_NumpyJnp,
        )
        accumulator = score_step(
            accumulator,
            second,
            batched_reference,
            JaxScoreWeights({"root_pos": 1.0}),
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        np.testing.assert_allclose(metrics["active_contact_count"], [5.0, 4.0])
        np.testing.assert_allclose(metrics["contact_pair_count"], [4.0, 8.0])

    def test_batched_score_keeps_sample_axis(self) -> None:
        first = _step_state(offset=0.0)
        second = _step_state(offset=2.0)
        batched_step = {
            name: np.stack([first[name], second[name]], axis=0)
            for name in first
        }
        ref = _reference_state()
        batched_ref = {
            name: np.stack([ref[name], ref[name]], axis=0)
            for name in ref
        }
        accumulator = init_score_accumulator((2,), jnp=_NumpyJnp)

        accumulator = score_step(
            accumulator,
            batched_step,
            batched_ref,
            JaxScoreWeights({"root_pos": 1.0}),
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        self.assertEqual(metrics["score"].shape, (2,))
        expected = np.array(
            [
                _expected_terms(first, ref)["root_pos_error_mean"],
                _expected_terms(second, ref)["root_pos_error_mean"],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(metrics["root_pos_error_mean"], expected)
        np.testing.assert_allclose(metrics["score"], -expected)

    def test_score_accumulator_has_stable_keys(self) -> None:
        accumulator = init_score_accumulator((2,), jnp=_NumpyJnp)

        self.assertEqual(tuple(accumulator), ACCUMULATOR_KEYS)
        for value in accumulator.values():
            self.assertEqual(value.shape, (2,))

    def test_contact_force_peak_source_tracks_max_peak(self) -> None:
        accumulator = init_score_accumulator((2,), jnp=_NumpyJnp)
        ref = _reference_state()
        batched_ref = {
            name: np.stack([ref[name], ref[name]], axis=0)
            for name in ref
        }
        first = _step_state()
        second = _step_state()
        first.update(
            {
                "contact": np.array([1.0, 1.0], dtype=np.float32),
                "contact_force": np.array([100.0, 50.0], dtype=np.float32),
                "floor_contact_force_peak_source": np.array(
                    [
                        [10.0, 0.0, 17.0, 100.0, 10.0, 100.0, 3.0, 0.0],
                        [11.0, 0.0, 34.0, 50.0, 5.0, 50.0, 3.0, 4.0],
                        [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                    ],
                    dtype=np.float32,
                ),
            }
        )
        second.update(
            {
                "contact": np.array([1.0, 1.0], dtype=np.float32),
                "contact_force": np.array([150.0, 300.0], dtype=np.float32),
                "floor_contact_force_peak_source": np.array(
                    [
                        [20.0, 0.0, 17.0, 150.0, 15.0, 150.0, 3.0, 8.0],
                        [21.0, 0.0, 34.0, 300.0, 30.0, 300.0, 3.0, 12.0],
                        [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                    ],
                    dtype=np.float32,
                ),
            }
        )
        batched_first = {
            name: np.stack([first[name], second[name]], axis=0)
            for name in first
        }
        third = dict(first)
        fourth = dict(second)
        third["contact_force"] = np.array([400.0, 10.0], dtype=np.float32)
        third["floor_contact_force_peak_source"] = np.array(
            [
                [30.0, 0.0, 20.0, 400.0, 40.0, 400.0, 3.0, 16.0],
                [31.0, 0.0, 37.0, 10.0, 1.0, 10.0, 3.0, 20.0],
                [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        fourth["contact_force"] = np.array([200.0, 250.0], dtype=np.float32)
        fourth["floor_contact_force_peak_source"] = np.array(
            [
                [40.0, 0.0, 20.0, 200.0, 20.0, 200.0, 3.0, 24.0],
                [41.0, 0.0, 37.0, 250.0, 25.0, 250.0, 3.0, 28.0],
                [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        batched_second = {
            name: np.stack([third[name], fourth[name]], axis=0)
            for name in third
        }

        accumulator = score_step(
            accumulator,
            batched_first,
            batched_ref,
            JaxScoreWeights({"root_pos": 1.0}),
            jnp=_NumpyJnp,
        )
        accumulator = score_step(
            accumulator,
            batched_second,
            batched_ref,
            JaxScoreWeights({"root_pos": 1.0}),
            jnp=_NumpyJnp,
        )

        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        np.testing.assert_allclose(
            metrics["contact_force_peak"],
            np.array([400.0, 300.0], dtype=np.float32),
        )
        np.testing.assert_allclose(
            metrics["contact_force_peak_source"],
            np.array(
                [
                    [30.0, 0.0, 20.0, 400.0, 40.0, 400.0, 3.0, 16.0],
                    [21.0, 0.0, 34.0, 300.0, 30.0, 300.0, 3.0, 12.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_score_step_rejects_missing_accumulator_key(self) -> None:
        accumulator = init_score_accumulator((), jnp=_NumpyJnp)
        del accumulator["joint_acc_sum"]

        with self.assertRaisesRegex(KeyError, "joint_acc_sum"):
            score_step(
                accumulator,
                _step_state(),
                _reference_state(),
                JaxScoreWeights({"root_pos": 1.0}),
                jnp=_NumpyJnp,
            )

    def test_larger_errors_lower_total_score(self) -> None:
        reference_state = _reference_state()
        weights = JaxScoreWeights(
            {
                "root_pos": 1.0,
                "body_global_pos": 1.0,
                "ee_global_pos": 1.0,
                "contact": 1.0,
                "control_delta": 1.0,
                "joint_acc": 1.0,
            }
        )

        small = finalize_score(
            score_step(
                init_score_accumulator((), jnp=_NumpyJnp),
                _step_state(offset=0.0),
                reference_state,
                weights,
                jnp=_NumpyJnp,
            ),
            jnp=_NumpyJnp,
        )
        large = finalize_score(
            score_step(
                init_score_accumulator((), jnp=_NumpyJnp),
                _step_state(offset=2.0),
                reference_state,
                weights,
                jnp=_NumpyJnp,
            ),
            jnp=_NumpyJnp,
        )

        self.assertLess(float(large["score"]), float(small["score"]))

    def test_score_step_rejects_empty_accumulator(self) -> None:
        with self.assertRaisesRegex(KeyError, "init_score_accumulator"):
            score_step(
                {},
                _step_state(),
                _reference_state(),
                JaxScoreWeights({"root_pos": 1.0}),
                jnp=_NumpyJnp,
            )

    def test_finalize_score_handles_empty_accumulator(self) -> None:
        metrics = finalize_score({}, jnp=_NumpyJnp)

        self.assertEqual(float(metrics["score"]), 0.0)
        self.assertEqual(float(metrics["root_pos_error_mean"]), 0.0)

    def test_score_weights_are_static_jax_pytree_node(self) -> None:
        try:
            import jax
        except Exception as exc:
            self.skipTest(f"JAX is not available: {exc}")

        weights = JaxScoreWeights({"root_pos": 1.0, "control_delta": 0.5})

        leaves, treedef = jax.tree_util.tree_flatten(weights)

        self.assertEqual(leaves, [])
        self.assertIn("JaxScoreWeights", str(treedef))


if __name__ == "__main__":
    unittest.main()
