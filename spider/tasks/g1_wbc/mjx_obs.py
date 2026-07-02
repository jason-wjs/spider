"""JAX-shaped observation helpers for the G1 WBC MJX backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    LIMB_EE_BODY_NAMES,
    OBS_DIM,
    OBS_HISTORY_LENGTH,
)


@dataclass(frozen=True)
class JaxObsState:
    history: object
    last_action: object


@dataclass(frozen=True)
class JaxObsIndices:
    command_body_indices: tuple[int, ...] | list[int]
    limb_indices: tuple[int, ...] | list[int]
    anchor_index: int
    tracking_anchor_index: int


LIMB_POSE_DIM = len(LIMB_EE_BODY_NAMES) * 9


OBS_FIELD_SPECS: dict[str, tuple[int, ...]] = {
    "command": (ACTION_DIM * 2,),
    "ref_limb_ee_pose_b": (OBS_HISTORY_LENGTH, LIMB_POSE_DIM),
    "motion_ref_ang_vel": (3,),
    "robot_limb_ee_pose_b": (OBS_HISTORY_LENGTH, LIMB_POSE_DIM),
    "projected_gravity": (OBS_HISTORY_LENGTH, 3),
    "base_ang_vel": (OBS_HISTORY_LENGTH, 3),
    "joint_pos": (OBS_HISTORY_LENGTH, ACTION_DIM),
    "joint_vel": (OBS_HISTORY_LENGTH, ACTION_DIM),
    "actions": (OBS_HISTORY_LENGTH, ACTION_DIM),
}


OBS_FIELD_ORDER = tuple(OBS_FIELD_SPECS)


def build_wbc_observation(fields: Mapping[str, object], *, jnp):
    """Build a WBC actor observation in the same slice order as ``obs.py``."""

    missing = [name for name in OBS_FIELD_ORDER if name not in fields]
    if missing:
        raise KeyError(f"Missing WBC observation fields: {missing}")

    parts = []
    batch_shape: tuple[int, ...] | None = None
    for name in OBS_FIELD_ORDER:
        part = _flatten_field(name, fields[name], jnp=jnp)
        leading = tuple(int(dim) for dim in part.shape[:-1])
        if batch_shape is None:
            batch_shape = leading
        elif leading != batch_shape:
            raise ValueError(
                f"Observation field {name!r} batch shape {leading} does not match "
                f"{batch_shape}"
            )
        parts.append(part)

    obs = jnp.concatenate(parts, axis=-1)
    obs_size = int(obs.shape[-1])
    if obs_size != OBS_DIM:
        raise ValueError(f"Observation has {obs_size} values, expected {OBS_DIM}")
    return obs


def build_wbc_observation_from_state(
    *,
    robot_state: Mapping[str, object],
    reference_state: Mapping[str, object],
    obs_state: JaxObsState,
    indices: JaxObsIndices,
    default_joint_pos,
    initialized: object,
    jnp,
):
    """Build the WBC observation from MJX-shaped robot and reference state."""

    robot_qpos = jnp.asarray(robot_state["qpos"])
    robot_qvel = jnp.asarray(robot_state["qvel"])
    robot_body_pos = _take_body(
        robot_state["body_pos_w"],
        indices.command_body_indices,
        jnp=jnp,
    )
    robot_body_quat = _take_body(
        robot_state["body_quat_w"],
        indices.command_body_indices,
        jnp=jnp,
    )
    robot_body_ang_vel = jnp.asarray(robot_state["body_ang_vel_w"])

    ref_joint_pos = jnp.asarray(reference_state["joint_pos"])
    ref_joint_vel = jnp.asarray(reference_state["joint_vel"])
    ref_body_pos = jnp.asarray(reference_state["body_pos_w"])
    ref_body_quat = jnp.asarray(reference_state["body_quat_w"])
    ref_body_ang_vel = jnp.asarray(reference_state["body_ang_vel_w"])

    command = jnp.concatenate([ref_joint_pos, ref_joint_vel], axis=-1)
    ref_limb = _limb_pose_in_anchor_frame(
        ref_body_pos,
        ref_body_quat,
        indices=indices,
        jnp=jnp,
    )
    robot_limb = _limb_pose_in_anchor_frame(
        robot_body_pos,
        robot_body_quat,
        indices=indices,
        jnp=jnp,
    )
    root_quat = robot_qpos[:, 3:7]
    gravity_w = jnp.repeat(
        jnp.asarray([[0.0, 0.0, -1.0]]),
        int(robot_qpos.shape[0]),
        axis=0,
    )
    projected_gravity = _quat_apply_inverse(root_quat, gravity_w, jnp=jnp)
    if "base_ang_vel_b" in robot_state and robot_state["base_ang_vel_b"] is not None:
        base_ang_vel_b = jnp.asarray(robot_state["base_ang_vel_b"])
    else:
        base_ang_vel_b = _quat_apply_inverse(
            root_quat,
            robot_body_ang_vel[:, 0],
            jnp=jnp,
        )
    joint_pos_rel = robot_qpos[:, 7:] - jnp.asarray(default_joint_pos).reshape(
        1,
        ACTION_DIM,
    )
    joint_vel_rel = robot_qvel[:, 6:]
    motion_ref_ang_vel = ref_body_ang_vel[:, int(indices.tracking_anchor_index)]

    previous_history = obs_state.history or {}
    history_fields = {
        "ref_limb_ee_pose_b": ref_limb,
        "robot_limb_ee_pose_b": robot_limb,
        "projected_gravity": projected_gravity,
        "base_ang_vel": base_ang_vel_b,
        "joint_pos": joint_pos_rel,
        "joint_vel": joint_vel_rel,
        "actions": obs_state.last_action,
    }
    next_history = {
        name: update_obs_history(
            previous_history.get(name),
            value,
            initialized=initialized,
            jnp=jnp,
        )
        for name, value in history_fields.items()
    }
    obs = build_wbc_observation(
        {
            "command": command,
            "ref_limb_ee_pose_b": next_history["ref_limb_ee_pose_b"],
            "motion_ref_ang_vel": motion_ref_ang_vel,
            "robot_limb_ee_pose_b": next_history["robot_limb_ee_pose_b"],
            "projected_gravity": next_history["projected_gravity"],
            "base_ang_vel": next_history["base_ang_vel"],
            "joint_pos": next_history["joint_pos"],
            "joint_vel": next_history["joint_vel"],
            "actions": next_history["actions"],
        },
        jnp=jnp,
    )
    return obs, JaxObsState(history=next_history, last_action=obs_state.last_action)


def update_obs_history(history, obs, *, initialized: object, jnp):
    """Update per-term observation history with first-frame backfill semantics."""

    obs = jnp.asarray(obs)
    obs_width = int(obs.shape[-1])
    backfilled = jnp.repeat(
        _insert_history_axis(obs, jnp=jnp),
        OBS_HISTORY_LENGTH,
        axis=-2,
    )
    if isinstance(initialized, bool) and not initialized:
        return backfilled

    _validate_history_shape(history, obs_width)
    shifted = jnp.concatenate(
        [history[..., 1:, :], _insert_history_axis(obs, jnp=jnp)],
        axis=-2,
    )
    if isinstance(initialized, bool):
        return shifted

    mask = jnp.asarray(initialized)
    while len(mask.shape) < len(shifted.shape):
        mask = jnp.expand_dims(mask, axis=-1)
    return jnp.where(mask, shifted, backfilled)


def _flatten_field(name: str, value, *, jnp):
    spec = OBS_FIELD_SPECS[name]
    arr = jnp.asarray(value)
    width = _prod(spec)
    if _tail_shape(arr, len(spec)) == spec:
        return arr.reshape((*arr.shape[: -len(spec)], width))
    if int(arr.shape[-1]) == width:
        return arr
    raise ValueError(
        f"Observation field {name!r} has shape {tuple(int(dim) for dim in arr.shape)}, "
        f"expected tail {spec} or flattened width {width}"
    )


def _take_body(value, indices, *, jnp):
    return jnp.take(jnp.asarray(value), tuple(int(index) for index in indices), axis=1)


def _limb_pose_in_anchor_frame(body_pos_w, body_quat_w, *, indices: JaxObsIndices, jnp):
    limb_indices = tuple(int(index) for index in indices.limb_indices)
    limb_pos_w = jnp.take(body_pos_w, limb_indices, axis=1)
    limb_quat_w = jnp.take(body_quat_w, limb_indices, axis=1)
    anchor = int(indices.anchor_index)
    anchor_pos_w = jnp.repeat(
        body_pos_w[:, anchor : anchor + 1],
        len(limb_indices),
        axis=1,
    )
    anchor_quat_w = jnp.repeat(
        body_quat_w[:, anchor : anchor + 1],
        len(limb_indices),
        axis=1,
    )
    pos_b, quat_b = _subtract_frame_transforms(
        anchor_pos_w,
        anchor_quat_w,
        limb_pos_w,
        limb_quat_w,
        jnp=jnp,
    )
    rot6d = _matrix_from_quat(quat_b, jnp=jnp)[..., :2].reshape(
        body_pos_w.shape[0],
        len(limb_indices),
        6,
    )
    return jnp.concatenate([pos_b, rot6d], axis=-1).reshape(body_pos_w.shape[0], -1)


def _subtract_frame_transforms(t01, q01, t02, q02, *, jnp):
    q10 = _quat_inv(q01, jnp=jnp)
    return _quat_apply(q10, t02 - t01, jnp=jnp), _quat_mul(q10, q02, jnp=jnp)


def _quat_inv(quat, *, jnp):
    return _quat_conjugate(quat, jnp=jnp) / jnp.sum(quat * quat, axis=-1, keepdims=True)


def _quat_conjugate(quat, *, jnp):
    return jnp.concatenate([quat[..., 0:1], -quat[..., 1:]], axis=-1)


def _quat_mul(q1, q2, *, jnp):
    w1, x1, y1, z1 = (q1[..., i] for i in range(4))
    w2, x2, y2, z2 = (q2[..., i] for i in range(4))
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return jnp.stack([w, x, y, z], axis=-1)


def _quat_apply_inverse(quat, vec, *, jnp):
    xyz = quat[..., 1:]
    t = _cross(xyz, vec, jnp=jnp) * 2.0
    return vec - quat[..., 0:1] * t + _cross(xyz, t, jnp=jnp)


def _quat_apply(quat, vec, *, jnp):
    xyz = quat[..., 1:]
    t = _cross(xyz, vec, jnp=jnp) * 2.0
    return vec + quat[..., 0:1] * t + _cross(xyz, t, jnp=jnp)


def _cross(a, b, *, jnp):
    return jnp.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def _matrix_from_quat(quat, *, jnp):
    r, i, j, k = (quat[..., index] for index in range(4))
    two_s = 2.0 / jnp.sum(quat * quat, axis=-1)
    out = jnp.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        axis=-1,
    )
    return out.reshape((*quat.shape[:-1], 3, 3))


def _insert_history_axis(value, *, jnp):
    return jnp.expand_dims(value, axis=-2)


def _validate_history_shape(history, obs_width: int) -> None:
    if history is None:
        raise ValueError("History is required when initialized is true")
    history_shape = tuple(int(dim) for dim in history.shape[-2:])
    expected_shape = (OBS_HISTORY_LENGTH, obs_width)
    if history_shape != expected_shape:
        raise ValueError(f"Expected history shape {expected_shape}, got {history_shape}")


def _tail_shape(value, ndim: int) -> tuple[int, ...]:
    if len(value.shape) < ndim:
        return ()
    return tuple(int(dim) for dim in value.shape[-ndim:])


def _prod(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


__all__ = [
    "ACTION_DIM",
    "JaxObsIndices",
    "JaxObsState",
    "LIMB_POSE_DIM",
    "OBS_FIELD_ORDER",
    "OBS_FIELD_SPECS",
    "build_wbc_observation",
    "build_wbc_observation_from_state",
    "update_obs_history",
]
