from __future__ import annotations

import torch

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    MUJOCO_BODY_NAMES,
    QPOS_DIM,
    QVEL_DIM,
)
from spider.tasks.g1_wbc.metrics import compute_rollout_metrics
from spider.tasks.g1_wbc.motion import G1Motion
from spider.tasks.g1_wbc.rollout import RolloutResult


def test_compute_rollout_metrics_reports_contact_force_first_row_diagnostics() -> None:
    frames = 4
    steps = frames - 1
    bodies = len(MUJOCO_BODY_NAMES)
    qpos = torch.zeros(frames, QPOS_DIM)
    qpos[:, 3] = 1.0
    body_quat = torch.zeros(frames, bodies, 4)
    body_quat[..., 0] = 1.0
    motion = G1Motion(
        path=None,
        motion_type="mujoco",
        fps=50.0,
        joint_pos=qpos[:, 7:].clone(),
        joint_vel=torch.zeros(frames, ACTION_DIM),
        body_pos_w=torch.zeros(frames, bodies, 3),
        body_quat_w=body_quat,
        body_lin_vel_w=torch.zeros(frames, bodies, 3),
        body_ang_vel_w=torch.zeros(frames, bodies, 3),
        contact=torch.ones(frames, 2),
    )
    contact_force = torch.tensor(
        [[0.0, 0.0], [4.0, 8.0], [6.0, 10.0], [8.0, 12.0]],
        dtype=torch.float32,
    ).view(frames, 1, 2)
    contact_force_first_row = torch.tensor(
        [[0.0, 0.0], [2.0, 4.0], [3.0, 5.0], [4.0, 6.0]],
        dtype=torch.float32,
    ).view(frames, 1, 2)
    rollout = RolloutResult(
        qpos=qpos[:, None, :],
        qvel=torch.zeros(frames, 1, QVEL_DIM),
        body_pos_w=torch.zeros(frames, 1, bodies, 3),
        body_quat_w=body_quat[:, None, :, :],
        body_lin_vel_w=torch.zeros(frames, 1, bodies, 3),
        body_ang_vel_w=torch.zeros(frames, 1, bodies, 3),
        actions=torch.zeros(steps, 1, ACTION_DIM),
        controls=torch.zeros(steps, 1, ACTION_DIM),
        contact_indicator=torch.ones(frames, 1, 2),
        contact_force=contact_force,
        contact_force_first_row=contact_force_first_row,
        ref_indices=torch.arange(frames).view(frames, 1),
    )

    metrics = compute_rollout_metrics(motion, rollout)

    assert metrics["contact_force_active_mean"] == 8.0
    assert metrics["contact_force_first_row_active_mean"] == 4.0
    assert metrics["contact_force_peak"] == 12.0
    assert metrics["contact_force_first_row_peak"] == 6.0
    assert metrics["contact_force_sum_to_first_row_active_ratio"] == 2.0
    assert metrics["contact_force_sum_to_first_row_peak_ratio"] == 2.0
