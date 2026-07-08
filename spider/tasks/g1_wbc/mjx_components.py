"""Explicit opt-in component wiring for G1 WBC MJX rollout scoring."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from spider.tasks.g1_wbc.constants import QPOS_DIM
from spider.tasks.g1_wbc.mjx_guided import guided_controls_from_trace
from spider.tasks.g1_wbc.mjx_physics import (
    default_action_scale,
    make_mjx_command_reference_fn,
    make_mjx_physics_step_fn,
)
from spider.tasks.g1_wbc.mjx_reference import (
    build_mjx_rollout_reference,
    default_joint_pos,
)
from spider.tasks.g1_wbc.mjx_scoring import JaxScoreWeights


@dataclass(frozen=True)
class MjxRolloutComponents:
    """Scorer and reference factory bundle for explicit MJX backend injection."""

    rollout_scorer: Callable[..., Any]
    score_only_rollout_scorer: Callable[..., Any] | None
    score_only_output_rollout_scorer: Callable[..., Any] | None
    rollout_reference_factory: Callable[..., dict[str, object]]
    rollout_tracer: Callable[..., dict[str, object]]
    rollout_state_advancer: Callable[..., dict[str, object]]
    physics_step_fn: Callable[..., Any]
    command_reference_fn: Callable[..., Any]
    default_joint_pos: Any
    action_scale: Any


def build_mjx_rollout_components(
    *,
    runtime,
    physics_step_fn: Callable[..., Any] | None = None,
    score_weights: JaxScoreWeights | Mapping[str, float] | None = None,
    default_joint_pos_override=None,
    action_scale_override=None,
    command_reference_fn: Callable[..., Any] | None = None,
    use_guided_candidate: bool = False,
    guided_root_pos_gain: float = 0.5,
    guided_root_rot_gain: float = 0.5,
    guided_joint_gain: float = 0.5,
    guided_root_pos_clip: float = 0.05,
    guided_root_rot_clip: float = 0.12,
    guided_joint_clip: float = 0.35,
    trace_prefix_steps: int | None = None,
    score_only_optimizer: bool = False,
    score_only_rescore_diagnostics: bool = False,
    score_only_output_rescore_diagnostics: bool = False,
    contact_force_mode: str = "sum_rows",
    contact_force_first_row_diagnostics: bool = False,
) -> MjxRolloutComponents:
    """Build explicit MJX rollout scorer/reference components."""

    jnp = runtime.jnp
    default_pos = (
        default_joint_pos(jnp=jnp)
        if default_joint_pos_override is None
        else jnp.asarray(default_joint_pos_override)
    )
    scale = (
        default_action_scale(jnp=jnp)
        if action_scale_override is None
        else jnp.asarray(action_scale_override)
    )
    trace_physics_step_fn = physics_step_fn
    if physics_step_fn is None:
        physics_step_fn = make_mjx_physics_step_fn(
            default_joint_pos=default_pos,
            action_scale=scale,
            contact_force_mode=contact_force_mode,
        )
        trace_physics_step_fn = (
            make_mjx_physics_step_fn(
                default_joint_pos=default_pos,
                action_scale=scale,
                contact_force_mode=contact_force_mode,
                contact_force_first_row_diagnostics=True,
            )
            if bool(contact_force_first_row_diagnostics)
            else physics_step_fn
        )
    if command_reference_fn is None:
        command_reference_fn = make_mjx_command_reference_fn()
    from spider.tasks.g1_wbc.mjx_rollout import make_rollout_scorer, make_rollout_tracer

    rollout_scorer = make_rollout_scorer(
        runtime=runtime,
        physics_step_fn=physics_step_fn,
        command_reference_fn=command_reference_fn,
        trace_prefix_steps=trace_prefix_steps,
        score_only=bool(score_only_optimizer),
        score_only_output=False,
    )
    score_only_rollout_scorer = None
    if bool(score_only_rescore_diagnostics):
        score_only_rollout_scorer = make_rollout_scorer(
            runtime=runtime,
            physics_step_fn=physics_step_fn,
            command_reference_fn=command_reference_fn,
            trace_prefix_steps=trace_prefix_steps,
            score_only=True,
            score_only_output=False,
        )
    score_only_output_rollout_scorer = None
    if bool(score_only_output_rescore_diagnostics):
        score_only_output_rollout_scorer = make_rollout_scorer(
            runtime=runtime,
            physics_step_fn=trace_physics_step_fn,
            command_reference_fn=command_reference_fn,
            trace_prefix_steps=trace_prefix_steps,
            score_only=False,
            score_only_output=True,
            score_only_output_full_metrics=True,
        )

    rollout_tracer = make_rollout_tracer(
        runtime=runtime,
        physics_step_fn=trace_physics_step_fn,
        command_reference_fn=command_reference_fn,
    )
    rollout_state_advancer = make_rollout_tracer(
        runtime=runtime,
        physics_step_fn=physics_step_fn,
        command_reference_fn=command_reference_fn,
        record_trace=False,
    )
    guided_rollout_tracer = make_rollout_tracer(
        runtime=runtime,
        physics_step_fn=physics_step_fn,
        command_reference_fn=None,
    )

    def rollout_reference_factory(**kwargs):
        reference_kwargs = dict(kwargs)
        include_guided_candidate = bool(reference_kwargs.pop("include_guided_candidate", True))
        reference_runtime = reference_kwargs.pop("runtime", runtime)
        reference = build_mjx_rollout_reference(
            runtime=reference_runtime,
            score_weights=score_weights,
            **reference_kwargs,
        )
        if bool(use_guided_candidate) and include_guided_candidate:
            horizon = int(reference["base_qpos"].shape[0])
            zero_controls = reference_runtime.jnp.zeros((1, horizon, QPOS_DIM - 1))
            trace_reference = dict(reference)
            if getattr(trace_reference.get("obs_state"), "history", None) is None:
                trace_reference["obs_initialized"] = False
            trace = guided_rollout_tracer(
                zero_controls,
                trace_reference,
                reference_kwargs["actor_params"],
                reference_kwargs["model_bundle"],
            )
            reference = dict(reference)
            reference["guided_controls"] = guided_controls_from_trace(
                reference["base_qpos"],
                trace["qpos"],
                guided_root_pos_gain=guided_root_pos_gain,
                guided_root_rot_gain=guided_root_rot_gain,
                guided_joint_gain=guided_joint_gain,
                guided_root_pos_clip=guided_root_pos_clip,
                guided_root_rot_clip=guided_root_rot_clip,
                guided_joint_clip=guided_joint_clip,
                freeze_first_frame=True,
                jnp=reference_runtime.jnp,
            )
        return reference

    return MjxRolloutComponents(
        rollout_scorer=rollout_scorer,
        score_only_rollout_scorer=score_only_rollout_scorer,
        score_only_output_rollout_scorer=score_only_output_rollout_scorer,
        rollout_reference_factory=rollout_reference_factory,
        rollout_tracer=rollout_tracer,
        rollout_state_advancer=rollout_state_advancer,
        physics_step_fn=physics_step_fn,
        command_reference_fn=command_reference_fn,
        default_joint_pos=default_pos,
        action_scale=scale,
    )


__all__ = [
    "MjxRolloutComponents",
    "build_mjx_rollout_components",
]
