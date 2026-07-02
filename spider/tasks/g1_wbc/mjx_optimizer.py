"""One-window JAX-shaped optimizer skeleton for the G1 WBC MJX backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JaxWindowOptimizerConfig:
    samples: int
    horizon_steps: int
    control_steps: int
    knot_count: int
    temperature: float
    root_pos_sigma: float
    root_rot_sigma: float
    joint_sigma: float


@dataclass(frozen=True)
class JaxWindowResult:
    updated_controls: object
    execute_chunk: object
    info: dict[str, float]


def sample_residual_controls(config: JaxWindowOptimizerConfig, controls, key, *, runtime):
    """Sample residual control candidates around a control mean."""

    _validate_config(config)
    jnp = runtime.jnp
    controls = jnp.asarray(controls)
    if len(controls.shape) != 2:
        raise ValueError(f"Expected 2D controls, got shape {controls.shape}")
    if int(controls.shape[0]) != int(config.horizon_steps):
        raise ValueError(
            f"Expected controls horizon {config.horizon_steps}, got {controls.shape[0]}"
        )
    sigma = _control_sigma(config, controls.shape[-1], jnp=jnp)
    noise = runtime.jax.random.normal(
        key,
        (int(config.samples), int(config.horizon_steps), int(controls.shape[-1])),
    )
    samples = controls[None, :, :] + noise * sigma[None, :, :]
    if hasattr(samples, "at"):
        return samples.at[0].set(controls)
    return _set_first(samples, controls)


def optimize_window(
    config: JaxWindowOptimizerConfig,
    state,
    controls,
    reference,
    actor_params,
    model_bundle,
    key,
    *,
    runtime,
):
    """Optimize one receding MPC window using an injected rollout scorer."""

    _validate_config(config)
    jnp = runtime.jnp
    rollout_fn = state["rollout_fn"]
    samples = sample_residual_controls(config, controls, key, runtime=runtime)
    scores = jnp.asarray(rollout_fn(samples, reference, actor_params, model_bundle))
    best_index = jnp.argmax(scores)
    temperature = max(float(config.temperature), 1.0e-6)
    weights = runtime.jax.nn.softmax(scores / temperature)
    updated_controls = jnp.sum(samples * weights[:, None, None], axis=0)
    execute_chunk = updated_controls[: int(config.control_steps) + 1]
    return JaxWindowResult(
        updated_controls=updated_controls,
        execute_chunk=execute_chunk,
        info={
            "best_index": _as_float(best_index),
            "best_score": _as_float(scores[best_index]),
            "mean_score": _as_float(jnp.mean(scores)),
        },
    )


def _control_sigma(config: JaxWindowOptimizerConfig, width: int, *, jnp):
    root_pos_width = min(3, int(width))
    root_rot_width = min(3, max(0, int(width) - root_pos_width))
    joint_width = max(0, int(width) - root_pos_width - root_rot_width)
    return jnp.concatenate(
        [
            jnp.full((int(config.horizon_steps), root_pos_width), config.root_pos_sigma),
            jnp.full((int(config.horizon_steps), root_rot_width), config.root_rot_sigma),
            jnp.full((int(config.horizon_steps), joint_width), config.joint_sigma),
        ],
        axis=-1,
    )


def _validate_config(config: JaxWindowOptimizerConfig) -> None:
    if int(config.samples) < 1:
        raise ValueError("JAX window optimizer requires at least one sample")
    if int(config.horizon_steps) < 1:
        raise ValueError("JAX window optimizer horizon_steps must be positive")
    if int(config.control_steps) < 0:
        raise ValueError("JAX window optimizer control_steps must be non-negative")
    if int(config.control_steps) >= int(config.horizon_steps):
        raise ValueError("control_steps must be smaller than horizon_steps")


def _set_first(samples, controls):
    out = samples.copy()
    out[0] = controls
    return out


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


__all__ = [
    "JaxWindowOptimizerConfig",
    "JaxWindowResult",
    "optimize_window",
    "sample_residual_controls",
]
