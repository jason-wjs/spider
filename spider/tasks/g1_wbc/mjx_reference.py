"""Reference-window builders for G1 WBC MJX rollout scoring."""

from __future__ import annotations

import math
from collections.abc import Mapping

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    ANCHOR_BODY_NAME,
    COMMAND_BODY_NAMES,
    KNEES_BENT_JOINT_POS,
    LIMB_EE_BODY_NAMES,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    QPOS_DIM,
    TASK_EE_BODY_NAMES,
    TRACKING_ANCHOR_BODY_NAME,
)
from spider.tasks.g1_wbc.mjx_obs import JaxObsIndices
from spider.tasks.g1_wbc.mjx_scoring import JaxScoreWeights


def build_mjx_rollout_reference(
    *,
    start: int,
    motion,
    controls,
    actor_params,
    model_bundle,
    runtime,
    score_weights: JaxScoreWeights | Mapping[str, float] | None = None,
    initial_robot_state: Mapping[str, object] | None = None,
    obs_state=None,
    obs_initialized=None,
    prev_control=None,
    prev_joint_acc=None,
    prev_contact=None,
    prev_contact_valid=None,
    prev_contact_force=None,
    prev_contact_force_valid=None,
) -> dict[str, object]:
    """Build the reference payload consumed by ``score_candidate_controls``."""

    del actor_params
    jnp = runtime.jnp
    start = int(start)
    if start < 0:
        raise ValueError("start must be non-negative")
    controls = _to_jnp(controls, jnp=jnp)
    if len(controls.shape) != 2 or int(controls.shape[-1]) != QPOS_DIM - 1:
        raise ValueError(
            f"Expected controls shape (horizon, {QPOS_DIM - 1}), got {controls.shape}"
        )
    horizon = int(controls.shape[0])
    if horizon < 1:
        raise ValueError("controls must contain at least one horizon step")

    indices = _window_indices(start, horizon, motion.num_frames)
    score_indices = _window_indices(start + 1, horizon, motion.num_frames)
    initial_index = min(start, motion.num_frames - 1)
    qpos = motion.qpos()
    qvel = motion.qvel()
    command_body_indices = _body_indices(COMMAND_BODY_NAMES)
    ee_body_indices = _body_indices(TASK_EE_BODY_NAMES)

    if initial_robot_state is None:
        initial_robot_state = _motion_robot_state(
            motion,
            initial_index=initial_index,
            jnp=jnp,
        )
    else:
        initial_robot_state = _robot_state_to_jnp(initial_robot_state, jnp=jnp)
    obs_reference = {
        "joint_pos": _to_jnp(_slice_window(motion.joint_pos, indices), jnp=jnp),
        "joint_vel": _to_jnp(_slice_window(motion.joint_vel, indices), jnp=jnp),
        "body_pos_w": _to_jnp(
            _slice_window(motion.body_pos_w, indices)[:, command_body_indices],
            jnp=jnp,
        ),
        "body_quat_w": _to_jnp(
            _slice_window(motion.body_quat_w, indices)[:, command_body_indices],
            jnp=jnp,
        ),
        "body_ang_vel_w": _to_jnp(
            _slice_window(motion.body_ang_vel_w, indices)[:, command_body_indices],
            jnp=jnp,
        ),
    }
    score_reference = {
        "root_pos": _to_jnp(_slice_window(qpos, score_indices)[:, :3], jnp=jnp),
        "root_quat": _to_jnp(_slice_window(qpos, score_indices)[:, 3:7], jnp=jnp),
        "joint_pos": _to_jnp(_slice_window(qpos, score_indices)[:, 7:], jnp=jnp),
        "body_pos": _to_jnp(_slice_window(motion.body_pos_w, score_indices), jnp=jnp),
        "body_quat": _to_jnp(_slice_window(motion.body_quat_w, score_indices), jnp=jnp),
        "ee_pos": _to_jnp(
            _slice_window(motion.body_pos_w, score_indices)[:, ee_body_indices],
            jnp=jnp,
        ),
        "ee_quat": _to_jnp(
            _slice_window(motion.body_quat_w, score_indices)[:, ee_body_indices],
            jnp=jnp,
        ),
        "contact": _to_jnp(_slice_window(motion.contact, score_indices), jnp=jnp),
    }
    return {
        "initial_robot_state": initial_robot_state,
        "base_qpos": _to_jnp(_slice_window(qpos, indices), jnp=jnp),
        "obs_reference": obs_reference,
        "score_reference": score_reference,
        "obs_state": obs_state,
        "obs_initialized": False if obs_initialized is None else obs_initialized,
        "obs_indices": _obs_indices(),
        "default_joint_pos": default_joint_pos(jnp=jnp),
        "joint_low": _joint_limit_array(model_bundle, high=False, jnp=jnp),
        "joint_high": _joint_limit_array(model_bundle, high=True, jnp=jnp),
        "prev_control": (
            _to_jnp([0.0] * ACTION_DIM, jnp=jnp)
            if prev_control is None
            else _to_jnp(prev_control, jnp=jnp)
        ),
        "prev_joint_acc": (
            _to_jnp([0.0] * ACTION_DIM, jnp=jnp)
            if prev_joint_acc is None
            else _to_jnp(prev_joint_acc, jnp=jnp)
        ),
        "prev_contact": (
            _to_jnp([0.0, 0.0], jnp=jnp)
            if prev_contact is None
            else _to_jnp(prev_contact, jnp=jnp)
        ),
        "prev_contact_valid": (
            prev_contact is not None
            if prev_contact_valid is None
            else _to_jnp(prev_contact_valid, jnp=jnp)
        ),
        "prev_contact_force": (
            _to_jnp([0.0, 0.0], jnp=jnp)
            if prev_contact_force is None
            else _to_jnp(prev_contact_force, jnp=jnp)
        ),
        "prev_contact_force_valid": (
            prev_contact_force is not None
            if prev_contact_force_valid is None
            else _to_jnp(prev_contact_force_valid, jnp=jnp)
        ),
        "score_weights": _score_weights(score_weights),
    }


def default_joint_pos(*, jnp):
    """Return the WXY default G1 joint pose in MuJoCo joint order."""

    values = [0.0] * ACTION_DIM
    for joint_name, value in KNEES_BENT_JOINT_POS.items():
        values[MUJOCO_JOINT_NAMES.index(joint_name)] = float(value)
    return jnp.asarray(values)


def _window_indices(start: int, horizon: int, frame_count: int) -> tuple[int, ...]:
    if frame_count < 1:
        raise ValueError("motion must contain at least one frame")
    last = int(frame_count) - 1
    return tuple(min(int(start) + offset, last) for offset in range(int(horizon)))


def _slice_index(value, index: int):
    return value[int(index)]


def _slice_window(value, indices: tuple[int, ...]):
    return value[list(indices)]


def _motion_robot_state(motion, *, initial_index: int, jnp) -> dict[str, object]:
    return {
        "qpos": _to_jnp(_slice_index(motion.qpos(), initial_index), jnp=jnp),
        "qvel": _to_jnp(_slice_index(motion.qvel(), initial_index), jnp=jnp),
        "body_pos_w": _to_jnp(_slice_index(motion.body_pos_w, initial_index), jnp=jnp),
        "body_quat_w": _to_jnp(_slice_index(motion.body_quat_w, initial_index), jnp=jnp),
        "body_lin_vel_w": _to_jnp(
            _slice_index(motion.body_lin_vel_w, initial_index),
            jnp=jnp,
        ),
        "body_ang_vel_w": _to_jnp(
            _slice_index(motion.body_ang_vel_w, initial_index),
            jnp=jnp,
        ),
    }


def _robot_state_to_jnp(state: Mapping[str, object], *, jnp) -> dict[str, object]:
    fields = (
        "qpos",
        "qvel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )
    missing = [name for name in fields if name not in state]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"initial_robot_state is missing fields: {names}")
    values = {name: _to_jnp(state[name], jnp=jnp) for name in fields}
    if "mjx_data" in state and state["mjx_data"] is not None:
        values["mjx_data"] = state["mjx_data"]
    return values


def _to_jnp(value, *, jnp):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return jnp.asarray(value)


def _body_indices(body_names: tuple[str, ...]) -> tuple[int, ...]:
    body_index = {name: index for index, name in enumerate(MUJOCO_BODY_NAMES)}
    return tuple(body_index[name] for name in body_names)


def _obs_indices() -> JaxObsIndices:
    return JaxObsIndices(
        command_body_indices=_body_indices(COMMAND_BODY_NAMES),
        limb_indices=tuple(COMMAND_BODY_NAMES.index(name) for name in LIMB_EE_BODY_NAMES),
        anchor_index=COMMAND_BODY_NAMES.index(ANCHOR_BODY_NAME),
        tracking_anchor_index=COMMAND_BODY_NAMES.index(TRACKING_ANCHOR_BODY_NAME),
    )


def _joint_limit_array(model_bundle, *, high: bool, jnp):
    cpu_model = getattr(model_bundle, "cpu_model", None)
    joint_name_to_id = getattr(model_bundle, "joint_name_to_id", None)
    if cpu_model is None or joint_name_to_id is None:
        raise ValueError("MJX model bundle must expose cpu_model and joint_name_to_id")

    values: list[float] = []
    unlimited = math.inf if high else -math.inf
    range_column = 1 if high else 0
    for joint_name in MUJOCO_JOINT_NAMES:
        joint_id = joint_name_to_id.get(f"robot/{joint_name}")
        if joint_id is None:
            joint_id = joint_name_to_id.get(joint_name)
        if joint_id is None:
            raise ValueError(f"MJX model bundle is missing joint {joint_name!r}")
        joint_id = int(joint_id)
        if int(cpu_model.jnt_limited[joint_id]):
            values.append(float(cpu_model.jnt_range[joint_id, range_column]))
        else:
            values.append(unlimited)
    return jnp.asarray(values)


def _score_weights(value) -> JaxScoreWeights:
    if value is None:
        return JaxScoreWeights({})
    if isinstance(value, JaxScoreWeights):
        return value
    return JaxScoreWeights(dict(value))


__all__ = [
    "build_mjx_rollout_reference",
    "default_joint_pos",
]
