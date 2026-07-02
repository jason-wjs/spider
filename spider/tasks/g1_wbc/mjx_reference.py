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
    initial_index = min(start, motion.num_frames - 1)
    qpos = motion.qpos()
    qvel = motion.qvel()
    command_body_indices = _body_indices(COMMAND_BODY_NAMES)
    ee_body_indices = _body_indices(TASK_EE_BODY_NAMES)

    initial_robot_state = {
        "qpos": _to_jnp(_slice_index(qpos, initial_index), jnp=jnp),
        "qvel": _to_jnp(_slice_index(qvel, initial_index), jnp=jnp),
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
        "root_pos": _to_jnp(_slice_window(qpos, indices)[:, :3], jnp=jnp),
        "body_pos": _to_jnp(_slice_window(motion.body_pos_w, indices), jnp=jnp),
        "ee_pos": _to_jnp(
            _slice_window(motion.body_pos_w, indices)[:, ee_body_indices],
            jnp=jnp,
        ),
        "contact": _to_jnp(_slice_window(motion.contact, indices), jnp=jnp),
    }
    return {
        "initial_robot_state": initial_robot_state,
        "obs_reference": obs_reference,
        "score_reference": score_reference,
        "obs_state": None,
        "obs_initialized": False,
        "obs_indices": _obs_indices(),
        "default_joint_pos": default_joint_pos(jnp=jnp),
        "joint_low": _joint_limit_array(model_bundle, high=False, jnp=jnp),
        "joint_high": _joint_limit_array(model_bundle, high=True, jnp=jnp),
        "prev_control": controls[0],
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
