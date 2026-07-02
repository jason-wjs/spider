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
    "JaxObsState",
    "LIMB_POSE_DIM",
    "OBS_FIELD_ORDER",
    "OBS_FIELD_SPECS",
    "build_wbc_observation",
    "update_obs_history",
]
