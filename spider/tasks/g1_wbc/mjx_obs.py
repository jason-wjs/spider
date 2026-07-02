"""JAX-shaped observation helpers for the G1 WBC MJX backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from spider.tasks.g1_wbc.constants import ACTION_DIM, OBS_DIM, OBS_HISTORY_LENGTH


@dataclass(frozen=True)
class JaxObsState:
    history: object
    last_action: object


OBS_FIELD_ORDER = (
    "projected_gravity",
    "base_ang_vel",
    "joint_pos_error",
    "joint_vel",
    "command",
    "last_action",
)


def build_wbc_observation(fields: Mapping[str, object], *, jnp):
    """Build a fixed-width WBC actor observation from ordered JAX arrays."""

    missing = [name for name in OBS_FIELD_ORDER if name not in fields]
    if missing:
        raise KeyError(f"Missing WBC observation fields: {missing}")

    parts = [jnp.ravel(jnp.asarray(fields[name])) for name in OBS_FIELD_ORDER]
    obs = jnp.concatenate(parts, axis=0)
    obs_size = int(obs.shape[0])
    if obs_size > OBS_DIM:
        raise ValueError(f"Observation has {obs_size} values, expected <= {OBS_DIM}")
    if obs_size < OBS_DIM:
        obs = jnp.pad(obs, (0, OBS_DIM - obs_size))
    return obs


def update_obs_history(history, obs, *, initialized: bool, jnp):
    """Update observation history with first-frame backfill semantics."""

    obs = jnp.asarray(obs)
    if int(obs.shape[-1]) != OBS_DIM:
        raise ValueError(f"Expected observation dim {OBS_DIM}, got {obs.shape[-1]}")
    if initialized:
        history_shape = tuple(int(dim) for dim in history.shape[-2:])
        expected_shape = (OBS_HISTORY_LENGTH, OBS_DIM)
        if history_shape != expected_shape:
            raise ValueError(f"Expected history shape {expected_shape}, got {history_shape}")
        return jnp.concatenate([history[1:], obs[None, :]], axis=0)
    return jnp.repeat(obs[None, :], OBS_HISTORY_LENGTH, axis=0)


__all__ = [
    "ACTION_DIM",
    "JaxObsState",
    "OBS_FIELD_ORDER",
    "build_wbc_observation",
    "update_obs_history",
]
