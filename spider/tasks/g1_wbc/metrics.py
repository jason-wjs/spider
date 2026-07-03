"""Evaluation metrics for G1 WBC rollout traces."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from spider.tasks.g1_wbc.constants import (
    ANCHOR_BODY_NAME,
    HAND_EE_BODY_NAMES,
    MUJOCO_BODY_NAMES,
    TASK_EE_BODY_NAMES,
)
from spider.tasks.g1_wbc.math_utils import (
    quat_error_magnitude,
    subtract_frame_transforms,
)
from spider.tasks.g1_wbc.motion import G1Motion
from spider.tasks.g1_wbc.rollout import RolloutResult


@dataclass(frozen=True)
class MetricThresholds:
    """Default success thresholds for G1 WBC benchmark rollouts."""

    root_pos_mean: float = 1.0
    root_rot_mean: float = 0.6
    ee_global_pos_mean: float = 1.0
    ee_local_pos_mean: float = 0.20
    contact_mismatch_rate: float = 0.40


def compute_rollout_metrics(
    motion: G1Motion,
    rollout: RolloutResult,
    *,
    thresholds: MetricThresholds = MetricThresholds(),
) -> dict[str, float | bool]:
    """Compare a rollout against the reference frames used as policy commands."""

    device = rollout.qpos.device
    motion = motion.to(device)
    ref_idx = rollout.ref_indices.to(device)

    ref_qpos = motion.qpos()[ref_idx]
    ref_qvel = motion.qvel()[ref_idx]
    ref_body_pos = motion.body_pos_w[ref_idx]
    ref_body_quat = motion.body_quat_w[ref_idx]
    ref_contact = motion.contact[ref_idx]

    root_pos_err = torch.linalg.norm(rollout.qpos[..., :3] - ref_qpos[..., :3], dim=-1)
    root_rot_err = quat_error_magnitude(rollout.qpos[..., 3:7], ref_qpos[..., 3:7])
    joint_pos_err = torch.linalg.norm(rollout.qpos[..., 7:] - ref_qpos[..., 7:], dim=-1)
    joint_vel_err = torch.linalg.norm(rollout.qvel[..., 6:] - ref_qvel[..., 6:], dim=-1)

    body_pos_err = torch.linalg.norm(rollout.body_pos_w - ref_body_pos, dim=-1)
    body_rot_err = quat_error_magnitude(rollout.body_quat_w, ref_body_quat)

    ee_indices = _body_indices(TASK_EE_BODY_NAMES, device)
    hand_indices = _body_indices(HAND_EE_BODY_NAMES, device)
    local_body_indices = _body_indices(
        tuple(name for name in MUJOCO_BODY_NAMES if name != ANCHOR_BODY_NAME),
        device,
    )
    ee_pos_err = body_pos_err.index_select(-1, ee_indices)
    ee_rot_err = body_rot_err.index_select(-1, ee_indices)
    local_pos_err, local_rot_err = _local_body_errors(
        rollout.body_pos_w,
        rollout.body_quat_w,
        ref_body_pos,
        ref_body_quat,
        ee_indices,
    )
    hand_pos_err = body_pos_err.index_select(-1, hand_indices)
    hand_rot_err = body_rot_err.index_select(-1, hand_indices)
    hand_local_pos_err, hand_local_rot_err = _local_body_errors(
        rollout.body_pos_w,
        rollout.body_quat_w,
        ref_body_pos,
        ref_body_quat,
        hand_indices,
    )
    body_local_pos_err, body_local_rot_err = _local_body_errors(
        rollout.body_pos_w,
        rollout.body_quat_w,
        ref_body_pos,
        ref_body_quat,
        local_body_indices,
    )

    sim_contact_eval = rollout.contact_indicator[1:]
    ref_contact_eval = ref_contact[1:]
    contact_err = (sim_contact_eval - ref_contact_eval).abs()
    false_positive = ((sim_contact_eval > 0.5) & (ref_contact_eval <= 0.5)).float()
    false_negative = ((sim_contact_eval <= 0.5) & (ref_contact_eval > 0.5)).float()
    contact_switch = _switch_rate(sim_contact_eval)
    ref_contact_switch = _switch_rate(ref_contact_eval)
    contact_force = rollout.contact_force[1:]
    contact_force_excess = _contact_force_excess(contact_force)
    contact_force_delta = _diff_norm(contact_force) / _contact_force_scale()
    floor_contact = _floor_contact_indicator(rollout)[1:]
    floor_force = _floor_contact_force(rollout)[1:]
    bad_floor_contact = floor_contact[..., 2:]
    bad_floor_force = floor_force[..., 2:]
    bad_floor_force_excess = bad_floor_force / _contact_force_scale()

    action_delta = _diff_norm(rollout.actions)
    ctrl_delta = _diff_norm(rollout.controls)
    joint_acc = _diff_norm(rollout.qvel[..., 6:]) / max(float(rollout.dt), 1.0e-6)
    joint_jerk = _diff_norm(torch.diff(rollout.qvel[..., 6:], dim=0)) / max(
        float(rollout.dt), 1.0e-6
    )

    metrics: dict[str, float | bool] = {
        "num_steps": float(rollout.num_steps),
        "control_dt_sec": float(rollout.dt),
        "evaluated_motion_duration_sec": float(rollout.num_steps) * float(rollout.dt),
        "root_pos_error_mean": _mean(root_pos_err),
        "root_pos_error_max": _max(root_pos_err),
        "root_rot_error_mean": _mean(root_rot_err),
        "joint_pos_error_mean": _mean(joint_pos_err),
        "joint_vel_error_mean": _mean(joint_vel_err),
        "body_global_pos_error_mean": _mean(body_pos_err),
        "body_global_rot_error_mean": _mean(body_rot_err),
        "ee_global_pos_error_mean": _mean(ee_pos_err),
        "ee_global_rot_error_mean": _mean(ee_rot_err),
        "ee_local_pos_error_mean": _mean(local_pos_err),
        "ee_local_rot_error_mean": _mean(local_rot_err),
        "hand_global_pos_error_mean": _mean(hand_pos_err),
        "hand_global_rot_error_mean": _mean(hand_rot_err),
        "hand_local_pos_error_mean": _mean(hand_local_pos_err),
        "hand_local_rot_error_mean": _mean(hand_local_rot_err),
        "body_local_pos_error_mean": _mean(body_local_pos_err),
        "body_local_rot_error_mean": _mean(body_local_rot_err),
        "contact_mismatch_rate": _mean(contact_err),
        "contact_false_positive_rate": _mean(false_positive),
        "contact_false_negative_rate": _mean(false_negative),
        "contact_switch_rate": _mean(contact_switch),
        "reference_contact_switch_rate": _mean(ref_contact_switch),
        "contact_force_active_mean": _active_force_mean(
            contact_force, sim_contact_eval
        ),
        "contact_force_peak": _max(contact_force),
        "contact_force_excess_mean": _mean(contact_force_excess),
        "contact_force_delta_mean": _mean(contact_force_delta),
        "bad_floor_contact_rate": _mean(bad_floor_contact),
        "bad_floor_force_mean": _mean(bad_floor_force),
        "bad_floor_force_excess_mean": _mean(bad_floor_force_excess),
        "action_delta_mean": _mean(action_delta),
        "control_delta_mean": _mean(ctrl_delta),
        "joint_acc_mean": _mean(joint_acc),
        "joint_jerk_mean": _mean(joint_jerk),
    }

    score = -(
        4.0 * float(metrics["contact_mismatch_rate"])
        + 2.0 * float(metrics["contact_switch_rate"])
        + 3.0 * float(metrics["ee_global_pos_error_mean"])
        + 2.0 * float(metrics["ee_local_pos_error_mean"])
        + 1.5 * float(metrics["root_pos_error_mean"])
        + 0.5 * float(metrics["root_rot_error_mean"])
        + 0.25 * float(metrics["joint_pos_error_mean"])
        + 0.05 * float(metrics["control_delta_mean"])
    )
    metrics["score"] = score
    metrics["success"] = (
        float(metrics["root_pos_error_mean"]) < thresholds.root_pos_mean
        and float(metrics["root_rot_error_mean"]) < thresholds.root_rot_mean
        and float(metrics["ee_global_pos_error_mean"]) < thresholds.ee_global_pos_mean
        and float(metrics["ee_local_pos_error_mean"]) < thresholds.ee_local_pos_mean
        and float(metrics["contact_mismatch_rate"]) < thresholds.contact_mismatch_rate
    )
    return metrics


def compute_rollout_scores(
    motion: G1Motion,
    rollout: RolloutResult,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return one scalar score per rollout world/sample plus per-sample terms."""

    device = rollout.qpos.device
    motion = motion.to(device)
    ref_idx = rollout.ref_indices.to(device)

    ref_qpos = motion.qpos()[ref_idx]
    ref_body_pos = motion.body_pos_w[ref_idx]
    ref_body_quat = motion.body_quat_w[ref_idx]
    ref_contact = motion.contact[ref_idx]

    root_pos_err = torch.linalg.norm(rollout.qpos[..., :3] - ref_qpos[..., :3], dim=-1)
    root_rot_err = quat_error_magnitude(rollout.qpos[..., 3:7], ref_qpos[..., 3:7])
    joint_pos_err = torch.linalg.norm(rollout.qpos[..., 7:] - ref_qpos[..., 7:], dim=-1)

    body_pos_err = torch.linalg.norm(rollout.body_pos_w - ref_body_pos, dim=-1)
    body_rot_err = quat_error_magnitude(rollout.body_quat_w, ref_body_quat)
    ee_indices = _body_indices(TASK_EE_BODY_NAMES, device)
    hand_indices = _body_indices(HAND_EE_BODY_NAMES, device)
    local_body_indices = _body_indices(
        tuple(name for name in MUJOCO_BODY_NAMES if name != ANCHOR_BODY_NAME),
        device,
    )
    ee_pos_err = body_pos_err.index_select(-1, ee_indices)
    ee_rot_err = body_rot_err.index_select(-1, ee_indices)
    local_pos_err, local_rot_err = _local_body_errors(
        rollout.body_pos_w,
        rollout.body_quat_w,
        ref_body_pos,
        ref_body_quat,
        ee_indices,
    )
    hand_pos_err = body_pos_err.index_select(-1, hand_indices)
    hand_rot_err = body_rot_err.index_select(-1, hand_indices)
    hand_local_pos_err, hand_local_rot_err = _local_body_errors(
        rollout.body_pos_w,
        rollout.body_quat_w,
        ref_body_pos,
        ref_body_quat,
        hand_indices,
    )
    body_local_pos_err, body_local_rot_err = _local_body_errors(
        rollout.body_pos_w,
        rollout.body_quat_w,
        ref_body_pos,
        ref_body_quat,
        local_body_indices,
    )

    sim_contact_eval = rollout.contact_indicator[1:]
    ref_contact_eval = ref_contact[1:]
    contact_err = (sim_contact_eval - ref_contact_eval).abs()
    false_positive = ((sim_contact_eval > 0.5) & (ref_contact_eval <= 0.5)).float()
    false_negative = ((sim_contact_eval <= 0.5) & (ref_contact_eval > 0.5)).float()
    contact_switch = _switch_rate(sim_contact_eval)
    contact_force = rollout.contact_force[1:]
    contact_force_excess = _contact_force_excess(contact_force)
    contact_force_delta = _diff_norm(contact_force) / _contact_force_scale()
    floor_contact = _floor_contact_indicator(rollout)[1:]
    floor_force = _floor_contact_force(rollout)[1:]
    bad_floor_contact = floor_contact[..., 2:]
    bad_floor_force = floor_force[..., 2:]
    bad_floor_force_excess = bad_floor_force / _contact_force_scale()
    action_delta = _diff_norm(rollout.actions)
    ctrl_delta = _diff_norm(rollout.controls)
    joint_acc = _diff_norm(rollout.qvel[..., 6:]) / max(float(rollout.dt), 1.0e-6)
    if rollout.qvel.shape[0] > 2:
        joint_jerk = _diff_norm(torch.diff(rollout.qvel[..., 6:], dim=0)) / max(
            float(rollout.dt), 1.0e-6
        )
    else:
        joint_jerk = torch.zeros_like(joint_acc)

    terms = {
        "root_pos_error": _per_env_mean(root_pos_err),
        "root_rot_error": _per_env_mean(root_rot_err),
        "joint_pos_error": _per_env_mean(joint_pos_err),
        "body_global_pos_error": _per_env_mean(body_pos_err),
        "body_global_rot_error": _per_env_mean(body_rot_err),
        "ee_global_pos_error": _per_env_mean(ee_pos_err),
        "ee_global_rot_error": _per_env_mean(ee_rot_err),
        "ee_local_pos_error": _per_env_mean(local_pos_err),
        "ee_local_rot_error": _per_env_mean(local_rot_err),
        "hand_global_pos_error": _per_env_mean(hand_pos_err),
        "hand_global_rot_error": _per_env_mean(hand_rot_err),
        "hand_local_pos_error": _per_env_mean(hand_local_pos_err),
        "hand_local_rot_error": _per_env_mean(hand_local_rot_err),
        "body_local_pos_error": _per_env_mean(body_local_pos_err),
        "body_local_rot_error": _per_env_mean(body_local_rot_err),
        "contact_mismatch": _per_env_mean(contact_err),
        "contact_false_positive": _per_env_mean(false_positive),
        "contact_false_negative": _per_env_mean(false_negative),
        "contact_switch": _per_env_mean(contact_switch),
        "contact_force_excess": _per_env_mean(contact_force_excess),
        "contact_force_delta": _per_env_mean(contact_force_delta),
        "bad_floor_contact": _per_env_mean(bad_floor_contact),
        "bad_floor_force_excess": _per_env_mean(bad_floor_force_excess),
        "action_delta": _per_env_mean(action_delta),
        "control_delta": _per_env_mean(ctrl_delta),
        "joint_acc": _per_env_mean(joint_acc),
        "joint_jerk": _per_env_mean(joint_jerk),
    }
    score = -(
        4.0 * terms["contact_mismatch"]
        + 2.0 * terms["contact_switch"]
        + 3.0 * terms["ee_global_pos_error"]
        + 2.0 * terms["ee_local_pos_error"]
        + 1.5 * terms["root_pos_error"]
        + 0.5 * terms["root_rot_error"]
        + 0.25 * terms["joint_pos_error"]
        + 0.05 * terms["control_delta"]
    )
    return score, terms


def _body_indices(names: tuple[str, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [MUJOCO_BODY_NAMES.index(name) for name in names],
        dtype=torch.long,
        device=device,
    )


def _floor_contact_indicator(rollout: RolloutResult) -> torch.Tensor:
    if rollout.floor_contact_indicator is not None:
        return rollout.floor_contact_indicator
    other = torch.zeros(
        *rollout.contact_indicator.shape[:-1],
        1,
        dtype=rollout.contact_indicator.dtype,
        device=rollout.contact_indicator.device,
    )
    return torch.cat([rollout.contact_indicator, other], dim=-1)


def _floor_contact_force(rollout: RolloutResult) -> torch.Tensor:
    if rollout.floor_contact_force is not None:
        return rollout.floor_contact_force
    other = torch.zeros(
        *rollout.contact_force.shape[:-1],
        1,
        dtype=rollout.contact_force.dtype,
        device=rollout.contact_force.device,
    )
    return torch.cat([rollout.contact_force, other], dim=-1)


def _local_body_errors(
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    ref_body_pos: torch.Tensor,
    ref_body_quat: torch.Tensor,
    body_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    anchor_idx = MUJOCO_BODY_NAMES.index(ANCHOR_BODY_NAME)
    pos_b, quat_b = _body_pose_in_anchor(body_pos, body_quat, anchor_idx, body_indices)
    ref_pos_b, ref_quat_b = _body_pose_in_anchor(
        ref_body_pos, ref_body_quat, anchor_idx, body_indices
    )
    pos_err = torch.linalg.norm(pos_b - ref_pos_b, dim=-1)
    rot_err = quat_error_magnitude(quat_b, ref_quat_b)
    return pos_err, rot_err


def _body_pose_in_anchor(
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    anchor_idx: int,
    body_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_pos = body_pos.index_select(-2, body_indices)
    target_quat = body_quat.index_select(-2, body_indices)
    anchor_pos = body_pos[..., anchor_idx : anchor_idx + 1, :].expand_as(target_pos)
    anchor_quat = body_quat[..., anchor_idx : anchor_idx + 1, :].expand_as(target_quat)
    return subtract_frame_transforms(anchor_pos, anchor_quat, target_pos, target_quat)


def _diff_norm(value: torch.Tensor) -> torch.Tensor:
    if value.shape[0] <= 1:
        return torch.zeros(value.shape[1:-1], dtype=value.dtype, device=value.device)
    return torch.linalg.norm(torch.diff(value, dim=0), dim=-1)


def _switch_rate(contact: torch.Tensor) -> torch.Tensor:
    if contact.shape[0] <= 1:
        return torch.zeros(contact.shape[1:], dtype=contact.dtype, device=contact.device)
    return (torch.diff((contact > 0.5).float(), dim=0).abs() > 0.5).float()


def _active_force_mean(force: torch.Tensor, contact: torch.Tensor) -> float:
    mask = contact > 0.5
    if not torch.any(mask):
        return 0.0
    return _mean(force[mask])


def _contact_force_scale() -> float:
    return 300.0


def _contact_force_excess(force: torch.Tensor) -> torch.Tensor:
    return torch.relu(force - _contact_force_scale()) / _contact_force_scale()


def _mean(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(torch.nan_to_num(value.float()).mean().detach().cpu().item())


def _max(value: torch.Tensor) -> float:
    if value.numel() == 0:
        return 0.0
    return float(torch.nan_to_num(value.float()).max().detach().cpu().item())


def _per_env_mean(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        env_count = value.shape[1] if value.ndim >= 2 else 0
        return torch.zeros(env_count, dtype=torch.float32, device=value.device)
    if value.ndim < 2:
        return torch.nan_to_num(value.float())
    env_count = value.shape[1]
    return torch.nan_to_num(value.float()).reshape(value.shape[0], env_count, -1).mean(
        dim=(0, 2)
    )
