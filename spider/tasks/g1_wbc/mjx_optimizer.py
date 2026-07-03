"""One-window JAX-shaped optimizer skeleton for the G1 WBC MJX backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


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
    iterations: int = 1


@dataclass(frozen=True)
class JaxWindowResult:
    updated_controls: object
    execute_chunk: object
    info: dict[str, object]


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
    parameter_steps = _sample_parameter_steps(config)
    sigma = _control_sigma(
        config,
        controls.shape[-1],
        steps=parameter_steps,
        jnp=jnp,
    )
    noise = runtime.jax.random.normal(
        _prng_key(key, runtime=runtime),
        (int(config.samples), int(parameter_steps), int(controls.shape[-1])),
    )
    delta = noise * sigma[None, :, :]
    if int(parameter_steps) != int(config.horizon_steps):
        delta = _interpolate_knot_samples(
            delta,
            target_steps=int(config.horizon_steps),
            jnp=jnp,
        )
    samples = controls[None, :, :] + delta
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
    updated_controls = controls
    info: dict[str, object] = {}
    for iteration in range(int(config.iterations)):
        samples = sample_residual_controls(
            config,
            updated_controls,
            _iteration_key(key, iteration, config=config),
            runtime=runtime,
        )
        rollout_result = rollout_fn(samples, reference, actor_params, model_bundle)
        scores = _rollout_scores(rollout_result, jnp=jnp)
        _validate_scores(scores, config)
        best_index = jnp.argmax(scores)
        temperature = max(float(config.temperature), 1.0e-6)
        weights = runtime.jax.nn.softmax(scores / temperature)
        updated_controls = jnp.sum(samples * weights[:, None, None], axis=0)
        info = {
            "best_index": best_index,
            "best_score": scores[best_index],
            "mean_score": jnp.mean(scores),
            "iteration": int(iteration),
            "iterations": int(config.iterations),
        }
        info.update(_rollout_diagnostics(rollout_result, jnp=jnp))
    execute_chunk = updated_controls[: int(config.control_steps) + 1]
    return JaxWindowResult(
        updated_controls=updated_controls,
        execute_chunk=execute_chunk,
        info=info,
    )


def _rollout_scores(rollout_result, *, jnp):
    if isinstance(rollout_result, Mapping):
        return jnp.asarray(rollout_result["score"])
    return jnp.asarray(rollout_result)


def _rollout_diagnostics(rollout_result, *, jnp) -> dict[str, object]:
    if not isinstance(rollout_result, Mapping):
        return {}
    diagnostics: dict[str, object] = {}
    for name in ("active_contact_count", "contact_pair_count"):
        if name in rollout_result:
            diagnostics[name] = _jnp_max(jnp.asarray(rollout_result[name]), jnp=jnp)
    return diagnostics


def _jnp_max(value, *, jnp):
    max_fn = getattr(jnp, "max", None)
    if callable(max_fn):
        return max_fn(value)
    return value.max()


def _sample_parameter_steps(config: JaxWindowOptimizerConfig) -> int:
    horizon = int(config.horizon_steps)
    if horizon <= 1:
        return 1
    return max(2, min(int(config.knot_count), horizon))


def _control_sigma(
    config: JaxWindowOptimizerConfig,
    width: int,
    *,
    steps: int | None = None,
    jnp,
):
    steps = int(config.horizon_steps) if steps is None else int(steps)
    root_pos_width = min(3, int(width))
    root_rot_width = min(3, max(0, int(width) - root_pos_width))
    joint_width = max(0, int(width) - root_pos_width - root_rot_width)
    return jnp.concatenate(
        [
            jnp.full((steps, root_pos_width), config.root_pos_sigma),
            jnp.full((steps, root_rot_width), config.root_rot_sigma),
            jnp.full((steps, joint_width), config.joint_sigma),
        ],
        axis=-1,
    )


def _interpolate_knot_samples(knot_samples, *, target_steps: int, jnp):
    target_steps = int(target_steps)
    if target_steps < 1:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    knot_count = int(knot_samples.shape[1])
    if target_steps == 1:
        return knot_samples[:, :1, :]
    if knot_count <= 1:
        return jnp.repeat(knot_samples[:, :1, :], target_steps, axis=1)
    positions = jnp.linspace(0.0, float(knot_count - 1), target_steps)
    left = jnp.floor(positions).astype("int32")
    right = jnp.minimum(left + 1, knot_count - 1)
    blend = (positions - left.astype(positions.dtype))[None, :, None]
    left_values = knot_samples[:, left, :]
    right_values = knot_samples[:, right, :]
    return left_values * (1.0 - blend) + right_values * blend


def _validate_config(config: JaxWindowOptimizerConfig) -> None:
    if int(config.samples) < 1:
        raise ValueError("JAX window optimizer requires at least one sample")
    if int(config.horizon_steps) < 1:
        raise ValueError("JAX window optimizer horizon_steps must be positive")
    if int(config.control_steps) < 0:
        raise ValueError("JAX window optimizer control_steps must be non-negative")
    if int(config.control_steps) >= int(config.horizon_steps):
        raise ValueError("control_steps must be smaller than horizon_steps")
    if int(config.knot_count) < 2 and int(config.horizon_steps) > 1:
        raise ValueError("JAX window optimizer knot_count must be at least 2")
    if int(config.iterations) < 1:
        raise ValueError("JAX window optimizer iterations must be positive")


def _validate_scores(scores, config: JaxWindowOptimizerConfig) -> None:
    expected = (int(config.samples),)
    actual = tuple(int(dim) for dim in scores.shape)
    if actual != expected:
        raise ValueError(f"Expected rollout scores shape {expected}, got {actual}")


def _prng_key(key, *, runtime):
    random = runtime.jax.random
    if not hasattr(random, "PRNGKey"):
        return key
    if isinstance(key, tuple):
        if not key:
            raise ValueError("JAX optimizer key tuple must not be empty")
        prng = random.PRNGKey(int(key[0]))
        fold_in = getattr(random, "fold_in", None)
        if fold_in is None:
            return prng
        for value in key[1:]:
            prng = fold_in(prng, int(value))
        return prng
    if isinstance(key, int):
        return random.PRNGKey(int(key))
    return key


def _iteration_key(key, iteration: int, *, config: JaxWindowOptimizerConfig):
    if int(config.iterations) == 1:
        return key
    if isinstance(key, tuple):
        return (*key, int(iteration))
    if isinstance(key, int):
        return (int(key), int(iteration))
    return key


def _set_first(samples, controls):
    out = samples.copy()
    out[0] = controls
    return out


__all__ = [
    "JaxWindowOptimizerConfig",
    "JaxWindowResult",
    "optimize_window",
    "sample_residual_controls",
]
