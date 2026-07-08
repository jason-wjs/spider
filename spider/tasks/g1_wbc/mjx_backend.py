"""MJX backend entry point for G1 WBC sampled MPC."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any, Callable

import numpy as np
import torch

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    OBS_HISTORY_LENGTH,
    QPOS_DIM,
    QVEL_DIM,
)
from spider.tasks.g1_wbc.math_utils import normalize, quat_from_axis_angle, quat_mul
from spider.tasks.g1_wbc.mjx_contacts import get_contact_profile
from spider.tasks.g1_wbc.mjx_optimizer import (
    JaxWindowOptimizerConfig,
    _SCORE_COMPONENT_DIAGNOSTIC_FIELDS,
    make_jitted_window_optimizer,
    optimize_window,
)
from spider.tasks.g1_wbc.mjx_policy import convert_wbc_actor_to_jax
from spider.tasks.g1_wbc.mjx_runtime import require_mjx_runtime
from spider.tasks.g1_wbc.motion import G1CommandBatch, G1Motion
from spider.tasks.g1_wbc.result_types import (
    G1WbcExecutedCommandChunk,
    G1WbcMpcRun,
    G1WbcSpiderResult,
    G1WbcWindowReplayState,
)


_JIT_WARMUP_KEY_FOLD = 2_147_483_647


def run_g1_wbc_mjx_mpc(
    *,
    spider_config,
    motion: G1Motion,
    actor,
    rollout_config,
    execute_rollout_config,
    method: str,
    reward_weights: dict[str, float] | None,
    total_steps: int,
    seed: int,
    runtime=None,
    model_factory: Callable[..., Any] | None = None,
    policy_converter: Callable[..., Any] = convert_wbc_actor_to_jax,
    optimizer: Callable[..., Any] = optimize_window,
    rollout_factory: Callable[..., Any] | None = None,
    rollout_scorer: Callable[..., Any] | None = None,
    score_only_rollout_scorer: Callable[..., Any] | None = None,
    score_only_output_rollout_scorer: Callable[..., Any] | None = None,
    rollout_reference_factory: Callable[..., Any] | None = None,
    rollout_tracer: Callable[..., Any] | None = None,
    rollout_state_advancer: Callable[..., Any] | None = None,
    command_builder: Callable[..., G1CommandBatch] | None = None,
    enable_physics_scan: bool = False,
    mjx_impl: str | None = "jax",
    mjx_warp_naconmax: int | None = None,
    mjx_warp_njmax: int | None = None,
    mjx_model_options: dict[str, int] | None = None,
    trace_prefix_steps: int | None = None,
    strip_live_mjx_data_between_windows: bool = False,
    score_only_optimizer: bool = False,
    score_only_rescore_diagnostics: bool = False,
    score_only_output_rescore_diagnostics: bool = False,
    contact_force_mode: str = "sum_rows",
    contact_force_first_row_diagnostics: bool = False,
):
    """Run the MJX full-rollout backend or fail before touching Warp state."""

    compile_start = time.perf_counter()
    if runtime is None:
        runtime = require_mjx_runtime()
    device = torch.device(rollout_config.device)
    _validate_single_gpu_runtime(runtime, device=device)
    using_jax_window_optimizer = optimizer is optimize_window
    total_steps = int(total_steps)
    horizon = int(spider_config.horizon_steps)
    control_steps = int(spider_config.ctrl_steps)
    use_warm_start = bool(getattr(spider_config, "use_warm_start", True))

    auto_enable_physics_scan = rollout_factory is None or (
        using_jax_window_optimizer and rollout_scorer is None
    )
    physics_scan_enabled = bool(enable_physics_scan or auto_enable_physics_scan)
    if physics_scan_enabled:
        if rollout_scorer is None or rollout_reference_factory is None:
            components = _default_rollout_components(
                runtime=runtime,
                method=method,
                reward_weights=reward_weights,
                trace_prefix_steps=trace_prefix_steps,
                score_only_optimizer=score_only_optimizer,
                score_only_rescore_diagnostics=score_only_rescore_diagnostics,
                score_only_output_rescore_diagnostics=(
                    score_only_output_rescore_diagnostics
                ),
                contact_force_mode=contact_force_mode,
                contact_force_first_row_diagnostics=(
                    contact_force_first_row_diagnostics
                ),
                **_guided_component_kwargs(spider_config),
            )
            if rollout_scorer is None:
                rollout_scorer = components.rollout_scorer
            if score_only_rollout_scorer is None:
                score_only_rollout_scorer = getattr(
                    components,
                    "score_only_rollout_scorer",
                    None,
                )
            if score_only_output_rollout_scorer is None:
                score_only_output_rollout_scorer = getattr(
                    components,
                    "score_only_output_rollout_scorer",
                    None,
                )
            if rollout_reference_factory is None:
                rollout_reference_factory = components.rollout_reference_factory
            if rollout_tracer is None:
                rollout_tracer = getattr(components, "rollout_tracer", None)
        if rollout_factory is None:
            rollout_factory = _default_static_rollout_factory(execute_rollout_config)

    if rollout_factory is None or (
        using_jax_window_optimizer and rollout_scorer is None
    ):
        raise NotImplementedError(
            "MJX runtime is available, but the production MJX physics scan has not "
            "been wired yet. Keep using --mpc-backend mujoco_warp for production."
        )

    if model_factory is None:
        model_factory = _default_model_factory
    model_bundle = model_factory(
        profile_name=getattr(rollout_config, "collision_profile", "wxy_parity"),
        require_runtime=True,
        mjx_impl=mjx_impl,
        mjx_warp_naconmax=mjx_warp_naconmax,
        mjx_warp_njmax=mjx_warp_njmax,
        mjx_model_options=mjx_model_options,
    )
    actor_params = policy_converter(actor, jnp=runtime.jnp)

    window_config = _window_config_from_spider(spider_config)
    if using_jax_window_optimizer:
        optimizer = make_jitted_window_optimizer(runtime=runtime)
    use_jax_controls = using_jax_window_optimizer and _supports_jax_controls(runtime)
    controls = (
        _initial_jax_controls(horizon, runtime=runtime)
        if use_jax_controls
        else torch.zeros(horizon, QPOS_DIM - 1, dtype=torch.float32, device=device)
    )
    baseline_qpos = motion.qpos()[: total_steps + 1].to(device).detach().clone()
    refined_qpos = baseline_qpos.clone()
    joint_low, joint_high = _joint_limits_from_model_bundle(model_bundle, device=device)
    infos: list[dict[str, Any]] = []
    best_scores: list[Any] = []
    execute_traces: list[dict[str, Any]] = []
    executed_command_specs: list[dict[str, Any]] = []
    deferred_execute_trace_specs: list[dict[str, Any]] = []
    sim_step = 0
    accepted_windows = 0
    current_robot_state = None
    current_obs_state = None
    current_obs_initialized = False
    current_prev_control = None
    current_prev_joint_acc = None
    current_prev_contact = None
    current_prev_contact_valid = None
    current_prev_contact_force = None
    current_prev_contact_force_valid = None
    execute_state_advancer = rollout_state_advancer or rollout_tracer
    optimizer_rollout_state = _optimizer_rollout_state(
        rollout_scorer=rollout_scorer,
        score_only_rollout_scorer=score_only_rollout_scorer,
        score_only_output_rollout_scorer=score_only_output_rollout_scorer,
        score_only_rescore_diagnostics=score_only_rescore_diagnostics,
        score_only_output_rescore_diagnostics=(
            score_only_output_rescore_diagnostics
        ),
    )
    guided_candidate_enabled = _use_guided_candidate(spider_config)
    guided_candidate_period = _guided_candidate_period(spider_config)
    jit_warmup_enabled = False
    jit_warmup_wall_time_sec = 0.0
    if use_jax_controls:
        warmup_start = time.perf_counter()
        warmup_reference = _window_reference(
            rollout_reference_factory,
            start=0,
            motion=motion,
            controls=controls,
            actor_params=actor_params,
            model_bundle=model_bundle,
            runtime=runtime,
        )
        warmup_result = optimizer(
            config=window_config,
            state=optimizer_rollout_state,
            controls=controls,
            reference=warmup_reference,
            actor_params=actor_params,
            model_bundle=model_bundle,
            key=(int(seed), _JIT_WARMUP_KEY_FOLD),
            runtime=runtime,
        )
        _block_window_result_until_ready(warmup_result)
        if execute_state_advancer is not None:
            warmup_execute_steps = min(control_steps, total_steps)
            warmup_updated_controls = _validated_jax_controls(
                warmup_result.updated_controls,
                horizon=horizon,
                runtime=runtime,
            )
            warmup_execute_controls = warmup_updated_controls[:warmup_execute_steps]
            warmup_execute_reference = _window_reference(
                rollout_reference_factory,
                start=0,
                motion=motion,
                controls=warmup_execute_controls,
                actor_params=actor_params,
                model_bundle=model_bundle,
                runtime=runtime,
                include_guided_candidate=False,
            )
            warmup_execute_trace = execute_state_advancer(
                warmup_execute_controls[None, :, :],
                warmup_execute_reference,
                actor_params,
                model_bundle,
            )
            _block_trace_until_ready(warmup_execute_trace)
            if warmup_execute_steps < total_steps:
                live_warmup_controls = _shift_jax_controls(
                    warmup_updated_controls,
                    execute_steps=warmup_execute_steps,
                    horizon=horizon,
                    use_warm_start=use_warm_start,
                    runtime=runtime,
                )
                live_warmup_reference = _window_reference(
                    rollout_reference_factory,
                    start=warmup_execute_steps,
                    motion=motion,
                    controls=live_warmup_controls,
                    actor_params=actor_params,
                    model_bundle=model_bundle,
                    runtime=runtime,
                    initial_robot_state=_required_trace_value(
                        warmup_execute_trace,
                        "final_robot_state",
                    ),
                    obs_state=warmup_execute_trace.get("final_obs_state"),
                    obs_initialized=True,
                    prev_control=warmup_execute_trace.get(
                        "final_prev_control",
                        warmup_execute_controls[-1:],
                    ),
                    prev_joint_acc=warmup_execute_trace.get("final_prev_joint_acc"),
                    prev_contact=warmup_execute_trace.get("final_prev_contact"),
                    prev_contact_valid=warmup_execute_trace.get(
                        "final_prev_contact_valid"
                    ),
                    prev_contact_force=warmup_execute_trace.get(
                        "final_prev_contact_force"
                    ),
                    prev_contact_force_valid=warmup_execute_trace.get(
                        "final_prev_contact_force_valid"
                    ),
                )
                live_warmup_result = optimizer(
                    config=window_config,
                    state=optimizer_rollout_state,
                    controls=live_warmup_controls,
                    reference=live_warmup_reference,
                    actor_params=actor_params,
                    model_bundle=model_bundle,
                    key=(int(seed), _JIT_WARMUP_KEY_FOLD + 1),
                    runtime=runtime,
                )
                _block_window_result_until_ready(live_warmup_result)
                live_warmup_execute_steps = min(
                    control_steps,
                    total_steps - warmup_execute_steps,
                )
                live_warmup_updated_controls = _validated_jax_controls(
                    live_warmup_result.updated_controls,
                    horizon=horizon,
                    runtime=runtime,
                )
                live_warmup_execute_controls = live_warmup_updated_controls[
                    :live_warmup_execute_steps
                ]
                live_warmup_execute_reference = _window_reference(
                    rollout_reference_factory,
                    start=warmup_execute_steps,
                    motion=motion,
                    controls=live_warmup_execute_controls,
                    actor_params=actor_params,
                    model_bundle=model_bundle,
                    runtime=runtime,
                    initial_robot_state=_required_trace_value(
                        warmup_execute_trace,
                        "final_robot_state",
                    ),
                    obs_state=warmup_execute_trace.get("final_obs_state"),
                    obs_initialized=True,
                    prev_control=warmup_execute_trace.get(
                        "final_prev_control",
                        warmup_execute_controls[-1:],
                    ),
                    prev_joint_acc=warmup_execute_trace.get("final_prev_joint_acc"),
                    prev_contact=warmup_execute_trace.get("final_prev_contact"),
                    prev_contact_valid=warmup_execute_trace.get(
                        "final_prev_contact_valid"
                    ),
                    prev_contact_force=warmup_execute_trace.get(
                        "final_prev_contact_force"
                    ),
                    prev_contact_force_valid=warmup_execute_trace.get(
                        "final_prev_contact_force_valid"
                    ),
                    include_guided_candidate=False,
                )
                live_warmup_trace = execute_state_advancer(
                    live_warmup_execute_controls[None, :, :],
                    live_warmup_execute_reference,
                    actor_params,
                    model_bundle,
                )
                _block_trace_until_ready(live_warmup_trace)
                if _needs_live_unguided_guided_period_warmup(
                    guided_candidate_enabled=guided_candidate_enabled,
                    guided_candidate_period=guided_candidate_period,
                ):
                    live_unguided_warmup_reference = _window_reference(
                        rollout_reference_factory,
                        start=warmup_execute_steps,
                        motion=motion,
                        controls=live_warmup_controls,
                        actor_params=actor_params,
                        model_bundle=model_bundle,
                        runtime=runtime,
                        initial_robot_state=_required_trace_value(
                            warmup_execute_trace,
                            "final_robot_state",
                        ),
                        obs_state=warmup_execute_trace.get("final_obs_state"),
                        obs_initialized=True,
                        prev_control=warmup_execute_trace.get(
                            "final_prev_control",
                            warmup_execute_controls[-1:],
                        ),
                        prev_joint_acc=warmup_execute_trace.get(
                            "final_prev_joint_acc"
                        ),
                        prev_contact=warmup_execute_trace.get("final_prev_contact"),
                        prev_contact_valid=warmup_execute_trace.get(
                            "final_prev_contact_valid"
                        ),
                        prev_contact_force=warmup_execute_trace.get(
                            "final_prev_contact_force"
                        ),
                        prev_contact_force_valid=warmup_execute_trace.get(
                            "final_prev_contact_force_valid"
                        ),
                        include_guided_candidate=False,
                    )
                    live_unguided_warmup_result = optimizer(
                        config=window_config,
                        state=optimizer_rollout_state,
                        controls=live_warmup_controls,
                        reference=live_unguided_warmup_reference,
                        actor_params=actor_params,
                        model_bundle=model_bundle,
                        key=(int(seed), _JIT_WARMUP_KEY_FOLD + 2),
                        runtime=runtime,
                    )
                    _block_window_result_until_ready(live_unguided_warmup_result)
        jit_warmup_wall_time_sec = time.perf_counter() - warmup_start
        jit_warmup_enabled = True
        del warmup_result, warmup_reference
    compile_init_wall_time_sec = time.perf_counter() - compile_start
    steady_start = time.perf_counter()
    while sim_step < total_steps:
        window_start = time.perf_counter()
        live_robot_state = _live_robot_state_for_reference(
            current_robot_state,
            strip_mjx_data=strip_live_mjx_data_between_windows,
        )
        replay_state_spec = _window_replay_state_spec(
            start=sim_step,
            current_robot_state=live_robot_state,
            current_obs_state=current_obs_state,
        )
        reference_start = time.perf_counter()
        window_reference = _window_reference(
            rollout_reference_factory,
            start=sim_step,
            motion=motion,
            controls=controls,
            actor_params=actor_params,
            model_bundle=model_bundle,
            runtime=runtime,
            initial_robot_state=live_robot_state,
            obs_state=current_obs_state,
            obs_initialized=current_obs_initialized if current_obs_state is not None else None,
            prev_control=current_prev_control,
            prev_joint_acc=current_prev_joint_acc,
            prev_contact=current_prev_contact,
            prev_contact_valid=current_prev_contact_valid,
            prev_contact_force=current_prev_contact_force,
            prev_contact_force_valid=current_prev_contact_force_valid,
            include_guided_candidate=_include_guided_candidate_for_window(
                len(infos),
                guided_candidate_enabled=guided_candidate_enabled,
                guided_candidate_period=guided_candidate_period,
            ),
        )
        reference_wall_time_sec = time.perf_counter() - reference_start
        optimizer_start = time.perf_counter()
        window_result = optimizer(
            config=window_config,
            state=optimizer_rollout_state,
            controls=controls,
            reference=window_reference,
            actor_params=actor_params,
            model_bundle=model_bundle,
            key=(int(seed), len(infos)),
            runtime=runtime,
        )
        optimizer_wall_time_sec = time.perf_counter() - optimizer_start
        execute_chunk_start = time.perf_counter()
        execute_steps = min(control_steps, total_steps - sim_step)
        updated_controls = (
            _validated_jax_controls(
                window_result.updated_controls,
                horizon=horizon,
                runtime=runtime,
            )
            if use_jax_controls
            else _validated_controls(
                window_result.updated_controls,
                horizon=horizon,
                device=device,
            )
        )
        execute_chunk = (
            _validated_jax_execute_chunk(
                window_result.execute_chunk,
                execute_steps=execute_steps,
                runtime=runtime,
            )
            if use_jax_controls
            else _validated_execute_chunk(
                window_result.execute_chunk,
                execute_steps=execute_steps,
                device=device,
            )
        )
        execute_chunk_wall_time_sec = time.perf_counter() - execute_chunk_start
        optimizer_result_sync_start = time.perf_counter()
        info = dict(window_result.info)
        _require_optimizer_scores_finite(info)
        window_accepted = _window_accepted(info)
        if window_accepted:
            _require_physics_scan_evidence(
                info,
                enabled=physics_scan_enabled,
                horizon=horizon,
            )
        optimizer_result_sync_wall_time_sec = (
            time.perf_counter() - optimizer_result_sync_start
        )
        execute_trace_wall_time_sec = 0.0
        execute_trace_source = "none"
        if window_accepted:
            controls = updated_controls
            executed_command_specs.append(
                {
                    "start": sim_step,
                    "horizon": horizon,
                    "execute_steps": execute_steps,
                    "updated_controls": updated_controls,
                    "execute_chunk": execute_chunk,
                    "replay_state": replay_state_spec,
                }
            )
            optimizer_execute_trace = _optimizer_execute_trace(
                window_result,
                execute_steps=execute_steps,
            )
            if optimizer_execute_trace is not None:
                execute_trace = optimizer_execute_trace
                execute_trace_source = "optimizer_selected_prefix"
                _record_execute_trace(
                    execute_traces,
                    execute_trace,
                    start=sim_step,
                    execute_steps=execute_steps,
                )
                current_robot_state = _required_trace_value(
                    execute_trace,
                    "final_robot_state",
                )
                current_obs_state = execute_trace.get("final_obs_state")
                current_obs_initialized = True
                current_prev_control = execute_trace.get(
                    "final_prev_control",
                    execute_chunk[-1:],
                )
                current_prev_joint_acc = execute_trace.get(
                    "final_prev_joint_acc",
                    current_prev_joint_acc,
                )
                current_prev_contact = execute_trace.get(
                    "final_prev_contact",
                    current_prev_contact,
                )
                current_prev_contact_valid = execute_trace.get(
                    "final_prev_contact_valid",
                    current_prev_contact_valid,
                )
                current_prev_contact_force = execute_trace.get(
                    "final_prev_contact_force",
                    current_prev_contact_force,
                )
                current_prev_contact_force_valid = execute_trace.get(
                    "final_prev_contact_force_valid",
                    current_prev_contact_force_valid,
                )
            elif execute_state_advancer is not None:
                execute_controls = updated_controls[:execute_steps]
                execute_reference = _window_reference(
                    rollout_reference_factory,
                    start=sim_step,
                    motion=motion,
                    controls=execute_controls,
                    actor_params=actor_params,
                    model_bundle=model_bundle,
                    runtime=runtime,
                    initial_robot_state=live_robot_state,
                    obs_state=current_obs_state,
                    obs_initialized=(
                        current_obs_initialized if current_obs_state is not None else None
                    ),
                    prev_control=current_prev_control,
                    prev_joint_acc=current_prev_joint_acc,
                    prev_contact=current_prev_contact,
                    prev_contact_valid=current_prev_contact_valid,
                    prev_contact_force=current_prev_contact_force,
                    prev_contact_force_valid=current_prev_contact_force_valid,
                    include_guided_candidate=False,
                )
                deferred_execute_trace_spec = {
                    "start": sim_step,
                    "execute_steps": execute_steps,
                    "execute_controls": execute_controls,
                    "initial_robot_state": live_robot_state,
                    "obs_state": current_obs_state,
                    "obs_initialized": (
                        current_obs_initialized if current_obs_state is not None else None
                    ),
                    "prev_control": current_prev_control,
                    "prev_joint_acc": current_prev_joint_acc,
                    "prev_contact": current_prev_contact,
                    "prev_contact_valid": current_prev_contact_valid,
                    "prev_contact_force": current_prev_contact_force,
                    "prev_contact_force_valid": current_prev_contact_force_valid,
                }
                execute_trace_start = time.perf_counter()
                execute_trace = execute_state_advancer(
                    execute_controls[None, :, :],
                    execute_reference,
                    actor_params,
                    model_bundle,
                )
                execute_trace_source = "rollout_tracer"
                execute_trace_wall_time_sec = time.perf_counter() - execute_trace_start
                if _complete_execute_trace(execute_trace, execute_steps=execute_steps):
                    _record_execute_trace(
                        execute_traces,
                        execute_trace,
                        start=sim_step,
                        execute_steps=execute_steps,
                    )
                elif rollout_tracer is not None:
                    deferred_execute_trace_specs.append(deferred_execute_trace_spec)
                current_robot_state = _required_trace_value(
                    execute_trace,
                    "final_robot_state",
                )
                current_obs_state = execute_trace.get("final_obs_state")
                current_obs_initialized = True
                current_prev_control = execute_trace.get(
                    "final_prev_control",
                    execute_controls[-1:],
                )
                current_prev_joint_acc = execute_trace.get(
                    "final_prev_joint_acc",
                    current_prev_joint_acc,
                )
                current_prev_contact = execute_trace.get(
                    "final_prev_contact",
                    current_prev_contact,
                )
                current_prev_contact_valid = execute_trace.get(
                    "final_prev_contact_valid",
                    current_prev_contact_valid,
                )
                current_prev_contact_force = execute_trace.get(
                    "final_prev_contact_force",
                    current_prev_contact_force,
                )
                current_prev_contact_force_valid = execute_trace.get(
                    "final_prev_contact_force_valid",
                    current_prev_contact_force_valid,
                )
        sim_step += execute_steps
        shift_start = time.perf_counter()
        controls = (
            _shift_jax_controls(
                controls,
                execute_steps=execute_steps,
                horizon=horizon,
                use_warm_start=use_warm_start,
                runtime=runtime,
            )
            if use_jax_controls
            else _shift_controls(
                controls,
                execute_steps=execute_steps,
                horizon=horizon,
                use_warm_start=use_warm_start,
                device=device,
            )
        )
        shift_wall_time_sec = time.perf_counter() - shift_start
        info.update(
            {
                "backend": "mjx",
                "sim_step": sim_step - execute_steps,
                "execute_steps": execute_steps,
                "accepted": window_accepted,
                "reference_wall_time_sec": reference_wall_time_sec,
                "optimizer_wall_time_sec": optimizer_wall_time_sec,
                "optimizer_result_sync_wall_time_sec": (
                    optimizer_result_sync_wall_time_sec
                ),
                "execute_chunk_wall_time_sec": execute_chunk_wall_time_sec,
                "execute_trace_wall_time_sec": execute_trace_wall_time_sec,
                "execute_trace_source": execute_trace_source,
                "shift_wall_time_sec": shift_wall_time_sec,
                "window_wall_time_sec": time.perf_counter() - window_start,
                "guided_candidate_included": _include_guided_candidate_for_window(
                    len(infos),
                    guided_candidate_enabled=guided_candidate_enabled,
                    guided_candidate_period=guided_candidate_period,
                ),
            }
        )
        infos.append(info)
        best_scores.append(
            _raw_info_scalar(info, "best_score")
            if use_jax_controls
            else _scalar_info(info, "best_score")
        )
        accepted_windows += int(window_accepted)
    steady_state_wall_time_sec = time.perf_counter() - steady_start

    artifact_refined_qpos_start = time.perf_counter()
    _apply_executed_command_specs_to_refined_qpos(
        refined_qpos,
        executed_command_specs,
        baseline_qpos=baseline_qpos,
        joint_low=joint_low,
        joint_high=joint_high,
    )
    artifact_refined_qpos_wall_time_sec = (
        time.perf_counter() - artifact_refined_qpos_start
    )

    artifact_execute_trace_wall_time_sec = 0.0
    if deferred_execute_trace_specs and rollout_tracer is not None:
        artifact_execute_trace_start = time.perf_counter()
        _materialize_deferred_execute_traces(
            execute_traces,
            deferred_execute_trace_specs,
            rollout_reference_factory=rollout_reference_factory,
            rollout_tracer=rollout_tracer,
            motion=motion,
            actor_params=actor_params,
            model_bundle=model_bundle,
            runtime=runtime,
        )
        artifact_execute_trace_wall_time_sec = (
            time.perf_counter() - artifact_execute_trace_start
        )

    rollout = _rollout_from_execute_traces(
        execute_traces,
        total_steps=total_steps,
        device=device,
    )
    rollout_source = "dynamic_execute_trace" if rollout is not None else "static_qpos_fallback"
    rollout_matches_command = rollout is None
    if rollout is None:
        rollout = rollout_factory(
            motion,
            total_steps,
            device=device,
            refined_qpos=refined_qpos.detach().clone(),
        )
    _validate_rollout_shape(
        rollout,
        total_steps=total_steps,
        refined_qpos=refined_qpos,
        require_qpos_match=rollout_matches_command,
    )
    command = _command_from_refined_qpos(
        motion,
        refined_qpos,
        rollout,
        command_builder=command_builder,
        rollout_config=execute_rollout_config,
    )
    _validate_command_shape(command, total_steps=total_steps)
    executed_command_chunks = _executed_command_chunks_from_specs(
        executed_command_specs,
        motion=motion,
        baseline_qpos=baseline_qpos,
        joint_low=joint_low,
        joint_high=joint_high,
        command_builder=command_builder,
        rollout_config=execute_rollout_config,
        device=device,
    )
    contact_metadata = _contact_metadata(
        model_bundle=model_bundle,
        infos=infos,
        rollout=rollout,
    )
    physics_step_metadata = _physics_step_count_metadata(infos)
    optimizer_window_metadata = _optimizer_window_metadata(infos)
    from spider.optimizers.receding import RecedingHorizonResult
    final_controls = _validated_controls(controls, horizon=horizon, device=device)
    receding = RecedingHorizonResult(
        controls=final_controls.detach().clone(),
        infos=infos,
        executed_steps=total_steps,
    )
    result = G1WbcSpiderResult(
        command=command,
        rollout=rollout,
        refined_qpos=refined_qpos,
        controls=final_controls.detach().clone(),
        infos=infos,
        scores=_scores_to_torch(best_scores, device=device),
        num_windows=len(infos),
        executed_command_chunks=executed_command_chunks,
    )
    return G1WbcMpcRun(
        receding=receding,
        result=result,
        metadata={
            "backend": "mjx",
            "mpc_backend": "mjx",
            "method": method,
            "reward_weights": reward_weights,
            "accepted": accepted_windows == len(infos),
            "physics_scan_enabled": physics_scan_enabled,
            "mjx_impl": getattr(model_bundle, "mjx_impl", mjx_impl),
            "mjx_model_impl": _mjx_impl_name(
                getattr(getattr(model_bundle, "mjx_model", None), "impl", None)
            ),
            "mjx_warp_naconmax": getattr(model_bundle, "mjx_warp_naconmax", None),
            "mjx_warp_njmax": getattr(model_bundle, "mjx_warp_njmax", None),
            "mjx_model_options": dict(
                getattr(model_bundle, "mjx_model_options", {}) or {}
            ),
            "collision_profile": getattr(
                getattr(model_bundle, "profile", None),
                "name",
                getattr(rollout_config, "collision_profile", "wxy_parity"),
            ),
            "rollout_source": rollout_source,
            "rollout_dynamic_execute_trace": rollout_source == "dynamic_execute_trace",
            "execute_trace_chunks": len(execute_traces),
            "strip_live_mjx_data_between_windows": bool(
                strip_live_mjx_data_between_windows
            ),
            "score_only_optimizer": bool(score_only_optimizer),
            "score_only_rescore_diagnostics": bool(
                score_only_rescore_diagnostics
            ),
            "score_only_output_rescore_diagnostics": bool(
                score_only_output_rescore_diagnostics
            ),
            "contact_force_mode": str(contact_force_mode),
            "contact_force_first_row_diagnostics": bool(
                contact_force_first_row_diagnostics
            ),
            "used_baseline_fallback": False,
            "accepted_windows": accepted_windows,
            "num_windows": len(infos),
            "sample_count": int(spider_config.num_samples),
            "optimizer_iterations": int(spider_config.max_num_iterations),
            "planning_horizon_steps": horizon,
            "control_steps": control_steps,
            "knot_count": int(spider_config.num_knot_points),
            "temperature": float(spider_config.temperature),
            "elite_frac": float(getattr(spider_config, "elite_frac", 1.0)),
            "root_pos_sigma": float(spider_config.pos_noise_scale),
            "root_rot_sigma": float(spider_config.rot_noise_scale),
            "joint_sigma": float(spider_config.joint_noise_scale),
            "first_ctrl_noise_scale": float(
                getattr(spider_config, "first_ctrl_noise_scale", 1.0)
            ),
            "last_ctrl_noise_scale": float(
                getattr(spider_config, "last_ctrl_noise_scale", 1.0)
            ),
            "use_warm_start": use_warm_start,
            "final_noise_scale": float(getattr(spider_config, "final_noise_scale", 1.0)),
            "sigma_decay": getattr(spider_config, "sigma_decay", None),
            "mjx_min_score_improvement": float(
                getattr(spider_config, "mjx_min_score_improvement", 1.0e-9)
            ),
            "mjx_min_top_score_gap": float(
                getattr(spider_config, "mjx_min_top_score_gap", 0.0)
            ),
            "mjx_cem_update_min_top_score_gap": float(
                getattr(spider_config, "mjx_cem_update_min_top_score_gap", 0.0)
            ),
            "mjx_max_control_delta": getattr(
                spider_config,
                "mjx_max_control_delta",
                None,
            ),
            "mjx_candidate_rank_diagnostics_top_k": int(
                getattr(spider_config, "mjx_candidate_rank_diagnostics_top_k", 0)
            ),
            "mjx_candidate_rescore_diagnostics": bool(
                getattr(spider_config, "mjx_candidate_rescore_diagnostics", False)
            ),
            "mjx_candidate_rescore_selection_top_k": int(
                getattr(spider_config, "mjx_candidate_rescore_selection_top_k", 0)
            ),
            "mjx_score_only_rescore_diagnostics": bool(
                getattr(spider_config, "mjx_score_only_rescore_diagnostics", False)
            ),
            "mjx_score_only_output_rescore_diagnostics": bool(
                getattr(
                    spider_config,
                    "mjx_score_only_output_rescore_diagnostics",
                    False,
                )
            ),
            "mjx_candidate_score_component_diagnostics_top_k": int(
                getattr(
                    spider_config,
                    "mjx_candidate_score_component_diagnostics_top_k",
                    0,
                )
            ),
            "mjx_candidate_score_component_diagnostics_fields": tuple(
                _SCORE_COMPONENT_DIAGNOSTIC_FIELDS
            ),
            "min_root_pos_sigma": float(
                getattr(spider_config, "min_root_pos_sigma", 0.0)
            ),
            "min_root_rot_sigma": float(
                getattr(spider_config, "min_root_rot_sigma", 0.0)
            ),
            "min_joint_sigma": float(getattr(spider_config, "min_joint_sigma", 0.0)),
            "use_guided_candidate": guided_candidate_enabled,
            "guided_candidate_period": guided_candidate_period,
            "guided_candidate_windows": sum(
                1 for info in infos if bool(info.get("guided_candidate_included"))
            ),
            "compile_init_wall_time_sec": compile_init_wall_time_sec,
            "jit_warmup_enabled": jit_warmup_enabled,
            "jit_warmup_wall_time_sec": jit_warmup_wall_time_sec,
            "steady_state_wall_time_sec": steady_state_wall_time_sec,
            "optimizer_result_sync_wall_time_sec": _info_timing_sum(
                infos,
                "optimizer_result_sync_wall_time_sec",
            ),
            "artifact_refined_qpos_wall_time_sec": artifact_refined_qpos_wall_time_sec,
            "artifact_execute_trace_wall_time_sec": artifact_execute_trace_wall_time_sec,
            "deferred_execute_trace_chunks": len(deferred_execute_trace_specs),
            "runtime_visible_devices": tuple(
                getattr(getattr(runtime, "status", None), "visible_devices", ())
            ),
            "runtime_gpu_name": _runtime_gpu_name(device),
            **physics_step_metadata,
            **optimizer_window_metadata,
            **contact_metadata,
        },
    )


def _window_config_from_spider(spider_config) -> JaxWindowOptimizerConfig:
    return JaxWindowOptimizerConfig(
        samples=int(spider_config.num_samples),
        horizon_steps=int(spider_config.horizon_steps),
        control_steps=int(spider_config.ctrl_steps),
        knot_count=int(spider_config.num_knot_points),
        temperature=float(spider_config.temperature),
        root_pos_sigma=float(spider_config.pos_noise_scale),
        root_rot_sigma=float(spider_config.rot_noise_scale),
        joint_sigma=float(spider_config.joint_noise_scale),
        first_control_noise_scale=float(
            getattr(spider_config, "first_ctrl_noise_scale", 1.0)
        ),
        last_control_noise_scale=float(
            getattr(spider_config, "last_ctrl_noise_scale", 1.0)
        ),
        elite_fraction=float(getattr(spider_config, "elite_frac", 1.0)),
        iterations=int(spider_config.max_num_iterations),
        final_noise_scale=float(getattr(spider_config, "final_noise_scale", 1.0)),
        sigma_decay=getattr(spider_config, "sigma_decay", None),
        min_root_pos_sigma=float(getattr(spider_config, "min_root_pos_sigma", 0.0)),
        min_root_rot_sigma=float(getattr(spider_config, "min_root_rot_sigma", 0.0)),
        min_joint_sigma=float(getattr(spider_config, "min_joint_sigma", 0.0)),
        use_guided_candidate=_use_guided_candidate(spider_config),
        min_score_improvement=float(
            getattr(spider_config, "mjx_min_score_improvement", 1.0e-9)
        ),
        min_top_score_gap=float(getattr(spider_config, "mjx_min_top_score_gap", 0.0)),
        cem_update_min_top_score_gap=float(
            getattr(spider_config, "mjx_cem_update_min_top_score_gap", 0.0)
        ),
        max_control_delta=getattr(spider_config, "mjx_max_control_delta", None),
        candidate_rank_diagnostics_top_k=int(
            getattr(spider_config, "mjx_candidate_rank_diagnostics_top_k", 0)
        ),
        candidate_rescore_diagnostics=bool(
            getattr(spider_config, "mjx_candidate_rescore_diagnostics", False)
        ),
        candidate_rescore_selection_top_k=int(
            getattr(spider_config, "mjx_candidate_rescore_selection_top_k", 0)
        ),
        score_only_rescore_diagnostics=bool(
            getattr(spider_config, "mjx_score_only_rescore_diagnostics", False)
        ),
        score_only_output_rescore_diagnostics=bool(
            getattr(
                spider_config,
                "mjx_score_only_output_rescore_diagnostics",
                False,
            )
        ),
        candidate_score_component_diagnostics_top_k=int(
            getattr(
                spider_config,
                "mjx_candidate_score_component_diagnostics_top_k",
                0,
            )
        ),
    )


def _placeholder_rollout_scores(samples, reference, actor_params, model_bundle):
    del reference, actor_params, model_bundle
    return samples[..., 0, 0]


def _optimizer_rollout_state(
    *,
    rollout_scorer,
    score_only_rollout_scorer,
    score_only_output_rollout_scorer,
    score_only_rescore_diagnostics: bool,
    score_only_output_rescore_diagnostics: bool,
) -> dict[str, Any]:
    state: dict[str, Any] = {
        "rollout_fn": rollout_scorer or _placeholder_rollout_scores,
    }
    if bool(score_only_rescore_diagnostics):
        if score_only_rollout_scorer is None:
            raise ValueError(
                "score_only_rescore_diagnostics requires a score-only rollout scorer"
        )
        state["score_only_rollout_fn"] = score_only_rollout_scorer
    if bool(score_only_output_rescore_diagnostics):
        if score_only_output_rollout_scorer is None:
            raise ValueError(
                "score_only_output_rescore_diagnostics requires a "
                "score-only-output rollout scorer"
            )
        state["score_only_output_rollout_fn"] = score_only_output_rollout_scorer
    return state


def _block_window_result_until_ready(window_result) -> None:
    _block_until_ready(getattr(window_result, "updated_controls", None))
    _block_until_ready(getattr(window_result, "execute_chunk", None))
    info = getattr(window_result, "info", {})
    for value in info.values():
        _block_info_value_until_ready(value)
    _require_optimizer_scores_finite(info)


def _block_info_value_until_ready(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _block_info_value_until_ready(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _block_info_value_until_ready(item)
        return
    _block_until_ready(value)


def _block_trace_until_ready(trace) -> None:
    if isinstance(trace, dict):
        for value in trace.values():
            _block_trace_until_ready(value)
        return
    if isinstance(trace, (list, tuple)):
        for value in trace:
            _block_trace_until_ready(value)
        return
    if is_dataclass(trace):
        for field in fields(trace):
            _block_trace_until_ready(getattr(trace, field.name))
        return
    _block_until_ready(trace)


def _block_until_ready(value) -> None:
    block = getattr(value, "block_until_ready", None)
    if callable(block):
        block()


def _default_model_factory(**kwargs):
    from spider.tasks.g1_wbc.mjx_model import build_mjx_model_bundle

    return build_mjx_model_bundle(**kwargs)


def _default_rollout_components(
    *,
    runtime,
    method: str,
    reward_weights: dict[str, float] | None,
    use_guided_candidate: bool,
    guided_root_pos_gain: float,
    guided_root_rot_gain: float,
    guided_joint_gain: float,
    guided_root_pos_clip: float,
    guided_root_rot_clip: float,
    guided_joint_clip: float,
    trace_prefix_steps: int | None = None,
    score_only_optimizer: bool = False,
    score_only_rescore_diagnostics: bool = False,
    score_only_output_rescore_diagnostics: bool = False,
    contact_force_mode: str = "sum_rows",
    contact_force_first_row_diagnostics: bool = False,
):
    from spider.tasks.g1_wbc.mjx_components import build_mjx_rollout_components

    return build_mjx_rollout_components(
        runtime=runtime,
        score_weights=_mjx_score_weights(method, reward_weights),
        use_guided_candidate=use_guided_candidate,
        guided_root_pos_gain=guided_root_pos_gain,
        guided_root_rot_gain=guided_root_rot_gain,
        guided_joint_gain=guided_joint_gain,
        guided_root_pos_clip=guided_root_pos_clip,
        guided_root_rot_clip=guided_root_rot_clip,
        guided_joint_clip=guided_joint_clip,
        trace_prefix_steps=trace_prefix_steps,
        score_only_optimizer=score_only_optimizer,
        score_only_rescore_diagnostics=score_only_rescore_diagnostics,
        score_only_output_rescore_diagnostics=(
            score_only_output_rescore_diagnostics
        ),
        contact_force_mode=contact_force_mode,
        contact_force_first_row_diagnostics=contact_force_first_row_diagnostics,
    )


def _guided_component_kwargs(spider_config) -> dict[str, float | bool]:
    return {
        "use_guided_candidate": _use_guided_candidate(spider_config),
        "guided_root_pos_gain": float(getattr(spider_config, "guided_root_pos_gain", 0.5)),
        "guided_root_rot_gain": float(getattr(spider_config, "guided_root_rot_gain", 0.5)),
        "guided_joint_gain": float(getattr(spider_config, "guided_joint_gain", 0.5)),
        "guided_root_pos_clip": float(getattr(spider_config, "guided_root_pos_clip", 0.05)),
        "guided_root_rot_clip": float(getattr(spider_config, "guided_root_rot_clip", 0.12)),
        "guided_joint_clip": float(getattr(spider_config, "guided_joint_clip", 0.35)),
    }


def _use_guided_candidate(spider_config) -> bool:
    return bool(getattr(spider_config, "use_guided_candidate", False))


def _guided_candidate_period(spider_config) -> int | None:
    period = getattr(spider_config, "guided_candidate_period", None)
    if period is None:
        return None
    period = int(period)
    if period <= 0:
        raise ValueError("guided_candidate_period must be positive")
    return period


def _include_guided_candidate_for_window(
    window_index: int,
    *,
    guided_candidate_enabled: bool,
    guided_candidate_period: int | None,
) -> bool:
    if not bool(guided_candidate_enabled):
        return False
    if guided_candidate_period is None:
        return True
    return int(window_index) % int(guided_candidate_period) == 0


def _needs_live_unguided_guided_period_warmup(
    *,
    guided_candidate_enabled: bool,
    guided_candidate_period: int | None,
) -> bool:
    return bool(guided_candidate_enabled) and guided_candidate_period is not None and (
        int(guided_candidate_period) > 1
    )


def _default_static_rollout_factory(rollout_config):
    def rollout_factory(motion, total_steps, *, device: torch.device, refined_qpos):
        del motion, device
        from spider.tasks.g1_wbc.rollout import run_static_qpos_rollout

        return run_static_qpos_rollout(
            refined_qpos,
            rollout_config,
            max_steps=int(total_steps),
        )

    return rollout_factory


def _mjx_score_weights(
    method: str,
    reward_weights: dict[str, float] | None,
) -> dict[str, float]:
    del method
    if not reward_weights:
        return {}
    supported_terms = {
        "root_pos_error",
        "root_rot_error",
        "joint_pos_error",
        "body_global_pos_error",
        "body_global_rot_error",
        "body_local_pos_error",
        "body_local_rot_error",
        "ee_global_pos_error",
        "ee_global_rot_error",
        "ee_local_pos_error",
        "ee_local_rot_error",
        "hand_global_pos_error",
        "hand_global_rot_error",
        "hand_local_pos_error",
        "hand_local_rot_error",
        "bad_floor_contact",
        "bad_floor_force_excess",
        "contact_force_active",
        "contact_force_peak_excess",
        "contact_force_delta",
        "contact_mismatch",
        "contact_false_positive",
        "contact_false_negative",
        "contact_switch",
        "control_delta",
        "action_delta",
        "joint_acc",
        "joint_jerk",
    }
    unsupported = sorted(
        name
        for name, value in reward_weights.items()
        if name not in supported_terms and float(value) != 0.0
    )
    if unsupported:
        raise ValueError(
            "MJX scorer does not support nonzero reward weight terms: "
            + ", ".join(unsupported)
        )
    mapping = {
        "root_pos": float(reward_weights.get("root_pos_error", 0.0)),
        "root_rot": float(reward_weights.get("root_rot_error", 0.0)),
        "joint_pos": float(reward_weights.get("joint_pos_error", 0.0)),
        "body_global_pos": float(reward_weights.get("body_global_pos_error", 0.0)),
        "body_global_rot": float(reward_weights.get("body_global_rot_error", 0.0)),
        "body_local_pos": float(reward_weights.get("body_local_pos_error", 0.0)),
        "body_local_rot": float(reward_weights.get("body_local_rot_error", 0.0)),
        "ee_global_pos": float(reward_weights.get("ee_global_pos_error", 0.0)),
        "ee_global_rot": float(reward_weights.get("ee_global_rot_error", 0.0)),
        "ee_local_pos": float(reward_weights.get("ee_local_pos_error", 0.0)),
        "ee_local_rot": float(reward_weights.get("ee_local_rot_error", 0.0)),
        "hand_global_pos": float(reward_weights.get("hand_global_pos_error", 0.0)),
        "hand_global_rot": float(reward_weights.get("hand_global_rot_error", 0.0)),
        "hand_local_pos": float(reward_weights.get("hand_local_pos_error", 0.0)),
        "hand_local_rot": float(reward_weights.get("hand_local_rot_error", 0.0)),
        "bad_floor_contact": float(reward_weights.get("bad_floor_contact", 0.0)),
        "bad_floor_force_excess": float(
            reward_weights.get("bad_floor_force_excess", 0.0)
        ),
        "contact_force_active": float(reward_weights.get("contact_force_active", 0.0)),
        "contact_force_peak_excess": float(
            reward_weights.get("contact_force_peak_excess", 0.0)
        ),
        "contact_force_delta": float(reward_weights.get("contact_force_delta", 0.0)),
        "contact": float(reward_weights.get("contact_mismatch", 0.0)),
        "contact_false_positive": float(
            reward_weights.get("contact_false_positive", 0.0)
        ),
        "contact_false_negative": float(
            reward_weights.get("contact_false_negative", 0.0)
        ),
        "contact_switch": float(reward_weights.get("contact_switch", 0.0)),
        "control_delta": float(reward_weights.get("control_delta", 0.0)),
        "action_delta": float(reward_weights.get("action_delta", 0.0)),
        "joint_acc": float(reward_weights.get("joint_acc", 0.0)),
        "joint_jerk": float(reward_weights.get("joint_jerk", 0.0)),
    }
    return {name: value for name, value in mapping.items() if value != 0.0}


def _contact_metadata(
    *,
    model_bundle,
    infos: list[dict[str, Any]],
    rollout,
) -> dict[str, int | bool]:
    profile = _resolved_contact_profile(model_bundle)
    max_contact_points = _positive_profile_int(profile, "max_contact_points")
    max_geom_pairs = _positive_profile_int(profile, "max_geom_pairs")
    profile_pair_count = len(tuple(getattr(profile, "explicit_pair_names", ()) or ()))
    model_pair_count = _optional_nonnegative_int(
        getattr(getattr(model_bundle, "cpu_model", None), "npair", None)
    )
    info_pair_count = _max_info_count(
        infos,
        "contact_pair_count",
        "mjx_contact_pair_count",
    )
    contact_pair_count = max(
        value
        for value in (profile_pair_count, model_pair_count, info_pair_count)
        if value is not None
    )
    active_contact_count = max(
        value
        for value in (
            _max_info_count(infos, "active_contact_count", "mjx_active_contact_count"),
            _max_info_count(infos, "ncon", "contact_count", "mjx_contact_count"),
            _rollout_active_contact_count(rollout),
        )
        if value is not None
    )
    max_contact_points_saturated = (
        _any_info_flag(infos, "max_contact_points_saturated")
        or active_contact_count >= max_contact_points
    )
    max_geom_pairs_saturated = (
        _any_info_flag(infos, "max_geom_pairs_saturated")
        or contact_pair_count >= max_geom_pairs
    )
    contact_saturated = (
        _any_info_flag(infos, "contact_saturated")
        or max_contact_points_saturated
        or max_geom_pairs_saturated
    )
    return {
        "contact_saturated": bool(contact_saturated),
        "max_contact_points_saturated": bool(max_contact_points_saturated),
        "max_geom_pairs_saturated": bool(max_geom_pairs_saturated),
        "max_contact_points": int(max_contact_points),
        "max_geom_pairs": int(max_geom_pairs),
        "contact_pair_count": int(contact_pair_count),
        "active_contact_count": int(active_contact_count),
    }


def _resolved_contact_profile(model_bundle):
    profile = getattr(model_bundle, "profile", None)
    if (
        _optional_nonnegative_int(getattr(profile, "max_contact_points", None))
        is not None
        and _optional_nonnegative_int(getattr(profile, "max_geom_pairs", None))
        is not None
    ):
        return profile
    profile_name = str(getattr(profile, "name", "wxy_parity"))
    return get_contact_profile(profile_name)


def _positive_profile_int(profile, name: str) -> int:
    value = _optional_nonnegative_int(getattr(profile, name, None))
    if value is None or value <= 0:
        raise ValueError(f"MJX contact profile must define positive {name}")
    return int(value)


def _max_info_count(infos: list[dict[str, Any]], *names: str) -> int | None:
    values: list[int] = []
    for info in infos:
        for name in names:
            value = _optional_nonnegative_int(info.get(name))
            if value is not None:
                values.append(value)
    return max(values) if values else None


def _any_info_flag(infos: list[dict[str, Any]], name: str) -> bool:
    for info in infos:
        value = _python_scalar(info.get(name))
        if isinstance(value, bool):
            if value:
                return True
        elif isinstance(value, (int, float)):
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(numeric) and numeric != 0.0:
                return True
    return False


def _info_truthy(value) -> bool:
    scalar = _python_scalar(value)
    if isinstance(scalar, bool):
        return scalar
    if isinstance(scalar, (int, float)):
        try:
            numeric = float(scalar)
        except (TypeError, ValueError, OverflowError):
            return False
        return math.isfinite(numeric) and numeric != 0.0
    return False


def _window_accepted(info: dict[str, Any]) -> bool:
    if "accepted" not in info:
        raise ValueError("MJX optimizer info must include boolean accepted metadata")
    accepted = _python_scalar(info.get("accepted"))
    if isinstance(accepted, bool):
        return accepted
    if isinstance(accepted, (int, float)) and math.isfinite(float(accepted)):
        return bool(accepted)
    raise ValueError("MJX optimizer info accepted metadata must be boolean-like")


def _require_optimizer_scores_finite(info: dict[str, Any]) -> None:
    if "scores_finite" not in info:
        return
    if not _info_truthy(info.get("scores_finite")):
        raise ValueError("MJX optimizer finite scores check failed")


def _require_physics_scan_evidence(
    info: dict[str, Any],
    *,
    enabled: bool,
    horizon: int,
) -> None:
    if not enabled:
        return
    count = _optional_nonnegative_int(info.get("physics_step_count"))
    if count is None or count < int(horizon):
        raise ValueError(
            "Accepted MJX windows require physics scan evidence covering the "
            f"{int(horizon)}-step planning horizon"
        )


def _physics_step_count_metadata(infos: list[dict[str, Any]]) -> dict[str, int]:
    counts: list[int] = []
    for info in infos:
        if _window_accepted(info):
            count = _optional_nonnegative_int(info.get("physics_step_count"))
            if count is not None:
                counts.append(int(count))
    if not counts:
        return {
            "physics_step_count_min": 0,
            "physics_step_count_max": 0,
            "physics_step_count_windows": 0,
        }
    return {
        "physics_step_count_min": min(counts),
        "physics_step_count_max": max(counts),
        "physics_step_count_windows": len(counts),
    }


def _optimizer_window_metadata(infos: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_iteration_sum = 0
    current_selected_windows = 0
    noop_accepted_windows = 0
    zero_delta_noop_iteration_sum = 0
    zero_delta_noop_windows = 0
    zero_delta_noop_accepted_windows = 0
    score_threshold_noop_iteration_sum = 0
    score_threshold_noop_windows = 0
    score_threshold_noop_accepted_windows = 0
    top_score_gap_noop_iteration_sum = 0
    top_score_gap_noop_windows = 0
    top_score_gap_noop_accepted_windows = 0
    control_delta_guard_noop_iteration_sum = 0
    control_delta_guard_noop_windows = 0
    control_delta_guard_noop_accepted_windows = 0
    noop_candidate_iteration_sum = 0
    noop_candidate_windows = 0
    noop_candidate_accepted_windows = 0
    candidate_rank_diagnostics_top_k_values: list[int] = []
    candidate_rank_diagnostics_windows = 0
    candidate_rescore_diagnostics_windows = 0
    candidate_rescore_delta_max_values: list[float] = []
    candidate_rescore_delta_mean_values: list[float] = []
    candidate_rescore_top1_changed_iteration_sum = 0
    candidate_rescore_selection_top_k_values: list[int] = []
    candidate_rescore_selection_windows = 0
    candidate_rescore_selection_delta_max_values: list[float] = []
    candidate_rescore_selection_delta_mean_values: list[float] = []
    candidate_rescore_selection_changed_iteration_sum = 0
    score_only_rescore_diagnostics_windows = 0
    score_only_rescore_delta_max_values: list[float] = []
    score_only_rescore_delta_mean_values: list[float] = []
    score_only_rescore_top1_changed_iteration_sum = 0
    score_only_output_rescore_diagnostics_windows = 0
    score_only_output_rescore_delta_max_values: list[float] = []
    score_only_output_rescore_delta_mean_values: list[float] = []
    score_only_output_rescore_top1_changed_iteration_sum = 0
    top_score_gaps: list[float] = []
    source_counts: dict[str, int] = {}
    iteration_accepted_counts: list[int] = []
    iteration_current_selected_counts: list[int] = []
    iteration_noop_counts: list[int] = []
    iteration_zero_delta_noop_counts: list[int] = []
    iteration_score_threshold_noop_counts: list[int] = []
    iteration_top_score_gap_noop_counts: list[int] = []
    iteration_control_delta_guard_noop_counts: list[int] = []
    iteration_noop_candidate_counts: list[int] = []
    iteration_top_score_gap_values: list[list[float]] = []
    for info in infos:
        top_score_gap = _optional_finite_float(info.get("top_score_gap"))
        if top_score_gap is not None:
            top_score_gaps.append(float(top_score_gap))
        candidate_rank_top_k = _optional_nonnegative_int(
            info.get("candidate_rank_diagnostics_top_k")
        )
        if candidate_rank_top_k is not None:
            candidate_rank_diagnostics_top_k_values.append(int(candidate_rank_top_k))
        if _info_sequence(info.get("iteration_candidate_top_indices")):
            candidate_rank_diagnostics_windows += 1
        if _info_truthy(info.get("candidate_rescore_diagnostics")):
            candidate_rescore_diagnostics_windows += 1
        rescore_delta_maxes = _info_sequence(
            info.get("iteration_rescore_score_delta_maxes")
        )
        for value in rescore_delta_maxes:
            parsed = _optional_finite_float(value)
            if parsed is not None:
                candidate_rescore_delta_max_values.append(float(parsed))
        rescore_delta_means = _info_sequence(
            info.get("iteration_rescore_score_delta_means")
        )
        for value in rescore_delta_means:
            parsed = _optional_finite_float(value)
            if parsed is not None:
                candidate_rescore_delta_mean_values.append(float(parsed))
        rescore_top1_flags = _info_sequence(
            info.get("iteration_rescore_top1_changed_flags")
        )
        for value in rescore_top1_flags:
            if _info_truthy(value):
                candidate_rescore_top1_changed_iteration_sum += 1
        rescore_selection_top_k = _optional_nonnegative_int(
            info.get("candidate_rescore_selection_top_k")
        )
        if rescore_selection_top_k is not None:
            candidate_rescore_selection_top_k_values.append(
                int(rescore_selection_top_k)
            )
        if _info_sequence(info.get("iteration_rescore_selection_best_indices")):
            candidate_rescore_selection_windows += 1
        rescore_selection_delta_maxes = _info_sequence(
            info.get("iteration_rescore_selection_score_delta_maxes")
        )
        for value in rescore_selection_delta_maxes:
            parsed = _optional_finite_float(value)
            if parsed is not None:
                candidate_rescore_selection_delta_max_values.append(float(parsed))
        rescore_selection_delta_means = _info_sequence(
            info.get("iteration_rescore_selection_score_delta_means")
        )
        for value in rescore_selection_delta_means:
            parsed = _optional_finite_float(value)
            if parsed is not None:
                candidate_rescore_selection_delta_mean_values.append(float(parsed))
        rescore_selection_changed_flags = _info_sequence(
            info.get("iteration_rescore_selection_changed_flags")
        )
        for value in rescore_selection_changed_flags:
            if _info_truthy(value):
                candidate_rescore_selection_changed_iteration_sum += 1
        if _info_truthy(info.get("score_only_rescore_diagnostics")):
            score_only_rescore_diagnostics_windows += 1
        score_only_rescore_delta_maxes = _info_sequence(
            info.get("iteration_score_only_rescore_score_delta_maxes")
        )
        for value in score_only_rescore_delta_maxes:
            parsed = _optional_finite_float(value)
            if parsed is not None:
                score_only_rescore_delta_max_values.append(float(parsed))
        score_only_rescore_delta_means = _info_sequence(
            info.get("iteration_score_only_rescore_score_delta_means")
        )
        for value in score_only_rescore_delta_means:
            parsed = _optional_finite_float(value)
            if parsed is not None:
                score_only_rescore_delta_mean_values.append(float(parsed))
        score_only_rescore_top1_flags = _info_sequence(
            info.get("iteration_score_only_rescore_top1_changed_flags")
        )
        for value in score_only_rescore_top1_flags:
            if _info_truthy(value):
                score_only_rescore_top1_changed_iteration_sum += 1
        if _info_truthy(info.get("score_only_output_rescore_diagnostics")):
            score_only_output_rescore_diagnostics_windows += 1
        score_only_output_rescore_delta_maxes = _info_sequence(
            info.get("iteration_score_only_output_rescore_score_delta_maxes")
        )
        for value in score_only_output_rescore_delta_maxes:
            parsed = _optional_finite_float(value)
            if parsed is not None:
                score_only_output_rescore_delta_max_values.append(float(parsed))
        score_only_output_rescore_delta_means = _info_sequence(
            info.get("iteration_score_only_output_rescore_score_delta_means")
        )
        for value in score_only_output_rescore_delta_means:
            parsed = _optional_finite_float(value)
            if parsed is not None:
                score_only_output_rescore_delta_mean_values.append(float(parsed))
        score_only_output_rescore_top1_flags = _info_sequence(
            info.get("iteration_score_only_output_rescore_top1_changed_flags")
        )
        for value in score_only_output_rescore_top1_flags:
            if _info_truthy(value):
                score_only_output_rescore_top1_changed_iteration_sum += 1
        accepted_iterations = _optional_nonnegative_int(
            info.get("accepted_iterations"),
        )
        if accepted_iterations is None:
            accepted_iterations = 0
        accepted_iteration_sum += int(accepted_iterations)
        current_selected = _info_truthy(info.get("current_controls_selected"))
        if current_selected:
            current_selected_windows += 1
        if _window_accepted(info) and current_selected and accepted_iterations == 0:
            noop_accepted_windows += 1
        zero_delta_noop_iterations = _optional_nonnegative_int(
            info.get("zero_delta_noop_iterations"),
        )
        if zero_delta_noop_iterations is None:
            zero_delta_noop_iterations = 0
        zero_delta_noop_iteration_sum += int(zero_delta_noop_iterations)
        zero_delta_noop_selected = _info_truthy(info.get("zero_delta_noop_selected"))
        if zero_delta_noop_selected:
            zero_delta_noop_windows += 1
        if (
            _window_accepted(info)
            and accepted_iterations == 0
            and zero_delta_noop_iterations > 0
        ):
            zero_delta_noop_accepted_windows += 1
        threshold_noop_iterations = _optional_nonnegative_int(
            info.get("score_threshold_noop_iterations"),
        )
        if threshold_noop_iterations is None:
            threshold_noop_iterations = 0
        score_threshold_noop_iteration_sum += int(threshold_noop_iterations)
        threshold_noop_selected = _info_truthy(
            info.get("score_threshold_noop_selected")
        )
        if threshold_noop_selected:
            score_threshold_noop_windows += 1
        if (
            _window_accepted(info)
            and accepted_iterations == 0
            and threshold_noop_iterations > 0
        ):
            score_threshold_noop_accepted_windows += 1
        top_gap_noop_iterations = _optional_nonnegative_int(
            info.get("top_score_gap_noop_iterations"),
        )
        if top_gap_noop_iterations is None:
            top_gap_noop_iterations = 0
        top_score_gap_noop_iteration_sum += int(top_gap_noop_iterations)
        top_gap_noop_selected = _info_truthy(
            info.get("top_score_gap_noop_selected")
        )
        if top_gap_noop_selected:
            top_score_gap_noop_windows += 1
        if (
            _window_accepted(info)
            and accepted_iterations == 0
            and top_gap_noop_iterations > 0
        ):
            top_score_gap_noop_accepted_windows += 1
        guard_noop_iterations = _optional_nonnegative_int(
            info.get("control_delta_guard_noop_iterations"),
        )
        if guard_noop_iterations is None:
            guard_noop_iterations = 0
        control_delta_guard_noop_iteration_sum += int(guard_noop_iterations)
        guard_noop_selected = _info_truthy(
            info.get("control_delta_guard_noop_selected")
        )
        if guard_noop_selected:
            control_delta_guard_noop_windows += 1
        if (
            _window_accepted(info)
            and accepted_iterations == 0
            and guard_noop_iterations > 0
        ):
            control_delta_guard_noop_accepted_windows += 1
        noop_candidate_iterations = _optional_nonnegative_int(
            info.get("noop_candidate_iterations"),
        )
        if noop_candidate_iterations is None:
            noop_candidate_iterations = 0
        noop_candidate_iteration_sum += int(noop_candidate_iterations)
        noop_candidate_selected = _info_truthy(info.get("noop_candidate_selected"))
        if noop_candidate_selected:
            noop_candidate_windows += 1
        if (
            _window_accepted(info)
            and accepted_iterations == 0
            and noop_candidate_iterations > 0
        ):
            noop_candidate_accepted_windows += 1
        accepted_flags = _info_sequence(info.get("iteration_accepted_flags"))
        current_flags = _info_sequence(
            info.get("iteration_current_controls_selected_flags")
        )
        zero_delta_flags = _info_sequence(info.get("iteration_zero_delta_noop_flags"))
        threshold_flags = _info_sequence(
            info.get("iteration_score_threshold_noop_flags")
        )
        top_gap_flags = _info_sequence(info.get("iteration_top_score_gap_noop_flags"))
        guard_flags = _info_sequence(
            info.get("iteration_control_delta_guard_noop_flags")
        )
        noop_candidate_flags = _info_sequence(
            info.get("iteration_noop_candidate_flags")
        )
        top_score_gap_values = _info_sequence(info.get("iteration_top_score_gaps"))
        max_iterations = max(
            len(accepted_flags),
            len(current_flags),
            len(zero_delta_flags),
            len(threshold_flags),
            len(top_gap_flags),
            len(guard_flags),
            len(noop_candidate_flags),
            len(top_score_gap_values),
        )
        _extend_count_list(iteration_accepted_counts, max_iterations)
        _extend_count_list(iteration_current_selected_counts, max_iterations)
        _extend_count_list(iteration_noop_counts, max_iterations)
        _extend_count_list(iteration_zero_delta_noop_counts, max_iterations)
        _extend_count_list(iteration_score_threshold_noop_counts, max_iterations)
        _extend_count_list(iteration_top_score_gap_noop_counts, max_iterations)
        _extend_count_list(iteration_control_delta_guard_noop_counts, max_iterations)
        _extend_count_list(iteration_noop_candidate_counts, max_iterations)
        _extend_nested_float_list(iteration_top_score_gap_values, max_iterations)
        for index in range(max_iterations):
            accepted_flag = (
                _info_truthy(accepted_flags[index])
                if index < len(accepted_flags)
                else False
            )
            current_flag = (
                _info_truthy(current_flags[index])
                if index < len(current_flags)
                else False
            )
            zero_delta_flag = (
                _info_truthy(zero_delta_flags[index])
                if index < len(zero_delta_flags)
                else False
            )
            threshold_flag = (
                _info_truthy(threshold_flags[index])
                if index < len(threshold_flags)
                else False
            )
            top_gap_flag = (
                _info_truthy(top_gap_flags[index])
                if index < len(top_gap_flags)
                else False
            )
            guard_flag = (
                _info_truthy(guard_flags[index])
                if index < len(guard_flags)
                else False
            )
            noop_candidate_flag = (
                _info_truthy(noop_candidate_flags[index])
                if index < len(noop_candidate_flags)
                else False
            )
            if accepted_flag:
                iteration_accepted_counts[index] += 1
            if current_flag:
                iteration_current_selected_counts[index] += 1
            if current_flag and not accepted_flag:
                iteration_noop_counts[index] += 1
            if zero_delta_flag:
                iteration_zero_delta_noop_counts[index] += 1
            if threshold_flag:
                iteration_score_threshold_noop_counts[index] += 1
            if top_gap_flag:
                iteration_top_score_gap_noop_counts[index] += 1
            if guard_flag:
                iteration_control_delta_guard_noop_counts[index] += 1
            if noop_candidate_flag:
                iteration_noop_candidate_counts[index] += 1
            if index < len(top_score_gap_values):
                iteration_top_score_gap = _optional_finite_float(
                    top_score_gap_values[index]
                )
                if iteration_top_score_gap is not None:
                    iteration_top_score_gap_values[index].append(
                        float(iteration_top_score_gap)
                    )
        source = info.get("execute_trace_source")
        if source is not None:
            source_name = str(source)
            source_counts[source_name] = source_counts.get(source_name, 0) + 1
    return {
        "accepted_iteration_sum": int(accepted_iteration_sum),
        "current_controls_selected_windows": int(current_selected_windows),
        "noop_accepted_windows": int(noop_accepted_windows),
        "zero_delta_noop_iteration_sum": int(zero_delta_noop_iteration_sum),
        "zero_delta_noop_windows": int(zero_delta_noop_windows),
        "zero_delta_noop_accepted_windows": int(zero_delta_noop_accepted_windows),
        "score_threshold_noop_iteration_sum": int(
            score_threshold_noop_iteration_sum
        ),
        "score_threshold_noop_windows": int(score_threshold_noop_windows),
        "score_threshold_noop_accepted_windows": int(
            score_threshold_noop_accepted_windows
        ),
        "top_score_gap_noop_iteration_sum": int(
            top_score_gap_noop_iteration_sum
        ),
        "top_score_gap_noop_windows": int(top_score_gap_noop_windows),
        "top_score_gap_noop_accepted_windows": int(
            top_score_gap_noop_accepted_windows
        ),
        "control_delta_guard_noop_iteration_sum": int(
            control_delta_guard_noop_iteration_sum
        ),
        "control_delta_guard_noop_windows": int(control_delta_guard_noop_windows),
        "control_delta_guard_noop_accepted_windows": int(
            control_delta_guard_noop_accepted_windows
        ),
        "noop_candidate_iteration_sum": int(noop_candidate_iteration_sum),
        "noop_candidate_windows": int(noop_candidate_windows),
        "noop_candidate_accepted_windows": int(noop_candidate_accepted_windows),
        "candidate_rank_diagnostics_top_k": (
            max(candidate_rank_diagnostics_top_k_values)
            if candidate_rank_diagnostics_top_k_values
            else 0
        ),
        "candidate_rank_diagnostics_windows": int(
            candidate_rank_diagnostics_windows
        ),
        "candidate_rescore_diagnostics_windows": int(
            candidate_rescore_diagnostics_windows
        ),
        "candidate_rescore_score_delta_max": _float_max_or_none(
            candidate_rescore_delta_max_values
        ),
        "candidate_rescore_score_delta_mean": _float_mean_or_none(
            candidate_rescore_delta_mean_values
        ),
        "candidate_rescore_top1_changed_iteration_sum": int(
            candidate_rescore_top1_changed_iteration_sum
        ),
        "candidate_rescore_selection_top_k": (
            max(candidate_rescore_selection_top_k_values)
            if candidate_rescore_selection_top_k_values
            else 0
        ),
        "candidate_rescore_selection_windows": int(
            candidate_rescore_selection_windows
        ),
        "candidate_rescore_selection_score_delta_max": _float_max_or_none(
            candidate_rescore_selection_delta_max_values
        ),
        "candidate_rescore_selection_score_delta_mean": _float_mean_or_none(
            candidate_rescore_selection_delta_mean_values
        ),
        "candidate_rescore_selection_changed_iteration_sum": int(
            candidate_rescore_selection_changed_iteration_sum
        ),
        "score_only_rescore_diagnostics_windows": int(
            score_only_rescore_diagnostics_windows
        ),
        "score_only_rescore_score_delta_max": _float_max_or_none(
            score_only_rescore_delta_max_values
        ),
        "score_only_rescore_score_delta_mean": _float_mean_or_none(
            score_only_rescore_delta_mean_values
        ),
        "score_only_rescore_top1_changed_iteration_sum": int(
            score_only_rescore_top1_changed_iteration_sum
        ),
        "score_only_output_rescore_diagnostics_windows": int(
            score_only_output_rescore_diagnostics_windows
        ),
        "score_only_output_rescore_score_delta_max": _float_max_or_none(
            score_only_output_rescore_delta_max_values
        ),
        "score_only_output_rescore_score_delta_mean": _float_mean_or_none(
            score_only_output_rescore_delta_mean_values
        ),
        "score_only_output_rescore_top1_changed_iteration_sum": int(
            score_only_output_rescore_top1_changed_iteration_sum
        ),
        "iteration_accepted_window_counts": tuple(iteration_accepted_counts),
        "iteration_current_selected_window_counts": tuple(
            iteration_current_selected_counts
        ),
        "iteration_noop_window_counts": tuple(iteration_noop_counts),
        "iteration_zero_delta_noop_window_counts": tuple(
            iteration_zero_delta_noop_counts
        ),
        "iteration_score_threshold_noop_window_counts": tuple(
            iteration_score_threshold_noop_counts
        ),
        "iteration_top_score_gap_noop_window_counts": tuple(
            iteration_top_score_gap_noop_counts
        ),
        "iteration_control_delta_guard_noop_window_counts": tuple(
            iteration_control_delta_guard_noop_counts
        ),
        "iteration_noop_candidate_window_counts": tuple(
            iteration_noop_candidate_counts
        ),
        "top_score_gap_min": _float_min_or_none(top_score_gaps),
        "top_score_gap_mean": _float_mean_or_none(top_score_gaps),
        "top_score_gap_max": _float_max_or_none(top_score_gaps),
        "top_score_gap_windows": len(top_score_gaps),
        "iteration_top_score_gap_mins": tuple(
            _float_min_or_none(values) for values in iteration_top_score_gap_values
        ),
        "iteration_top_score_gap_means": tuple(
            _float_mean_or_none(values) for values in iteration_top_score_gap_values
        ),
        "execute_trace_source_counts": dict(sorted(source_counts.items())),
    }


def _info_timing_sum(infos: list[dict[str, Any]], name: str) -> float:
    total = 0.0
    for info in infos:
        value = info.get(name)
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def _info_sequence(value) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _extend_count_list(values: list[int], size: int) -> None:
    while len(values) < int(size):
        values.append(0)


def _extend_nested_float_list(values: list[list[float]], size: int) -> None:
    while len(values) < int(size):
        values.append([])


def _optional_finite_float(value) -> float | None:
    scalar = _python_scalar(value)
    if isinstance(scalar, bool) or scalar is None:
        return None
    try:
        parsed = float(scalar)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _float_min_or_none(values: list[float]) -> float | None:
    return min(values) if values else None


def _float_mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _float_max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def _rollout_active_contact_count(rollout) -> int:
    indicator = getattr(rollout, "contact_indicator", None)
    if indicator is None:
        return 0
    contact = _to_torch(indicator, device=torch.device("cpu"))
    if contact.numel() == 0:
        return 0
    if contact.ndim == 0:
        return int(float(contact.item()) > 0.5)
    return int(torch.count_nonzero(contact > 0.5, dim=-1).max().item())


def _optional_nonnegative_int(value) -> int | None:
    scalar = _python_scalar(value)
    if isinstance(scalar, bool) or scalar is None:
        return None
    try:
        parsed = int(scalar)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(numeric) or parsed < 0 or float(parsed) != numeric:
        return None
    return parsed


def _mjx_impl_name(value) -> str | None:
    if value is None:
        return None
    name = str(value).strip().lower()
    if name.startswith("impl."):
        name = name.split(".", 1)[1]
    return name or None


def _python_scalar(value):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() != 1:
            return None
        return tensor.reshape(()).item()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.shape != ():
        return None
    return array.item()


def _validate_single_gpu_runtime(runtime, *, device: torch.device) -> None:
    status = getattr(runtime, "status", None)
    visible_devices = tuple(getattr(status, "visible_devices", ()) or ())
    if len(visible_devices) > 1:
        raise RuntimeError(
            "MJX backend requires single GPU visibility per motion; "
            f"got CUDA_VISIBLE_DEVICES={visible_devices}."
        )
    if visible_devices and device.type == "cuda" and device.index not in (None, 0):
        raise RuntimeError(
            "MJX backend requires cuda:0 after narrowing CUDA_VISIBLE_DEVICES "
            f"to one GPU; got device={device}."
        )

    jax_devices_fn = getattr(getattr(runtime, "jax", None), "devices", None)
    if not visible_devices and device.type == "cuda" and callable(jax_devices_fn):
        try:
            devices = tuple(jax_devices_fn())
        except Exception:
            return
        accelerator_count = sum(
            1
            for item in devices
            if str(getattr(item, "platform", "")).lower() in {"gpu", "cuda", "tpu"}
        )
        if accelerator_count > 1:
            raise RuntimeError(
                "MJX backend requires single GPU visibility per motion; "
                f"jax.devices() reports {accelerator_count} accelerators."
            )


def _runtime_gpu_name(device: torch.device) -> str | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(device))


def _window_reference(
    rollout_reference_factory: Callable[..., Any] | None,
    *,
    start: int,
    motion: G1Motion,
    controls: torch.Tensor,
    actor_params,
    model_bundle,
    runtime,
    initial_robot_state=None,
    obs_state=None,
    obs_initialized=None,
    prev_control=None,
    prev_joint_acc=None,
    prev_contact=None,
    prev_contact_valid=None,
    prev_contact_force=None,
    prev_contact_force_valid=None,
    include_guided_candidate: bool = True,
):
    if rollout_reference_factory is None:
        return {"start": int(start)}
    kwargs = {
        "start": int(start),
        "motion": motion,
        "controls": controls,
        "actor_params": actor_params,
        "model_bundle": model_bundle,
        "runtime": runtime,
        "include_guided_candidate": bool(include_guided_candidate),
    }
    if initial_robot_state is not None:
        kwargs["initial_robot_state"] = initial_robot_state
    if obs_state is not None:
        kwargs["obs_state"] = obs_state
    if obs_initialized is not None:
        kwargs["obs_initialized"] = obs_initialized
    if prev_control is not None:
        kwargs["prev_control"] = prev_control
    if prev_joint_acc is not None:
        kwargs["prev_joint_acc"] = prev_joint_acc
    if prev_contact is not None:
        kwargs["prev_contact"] = prev_contact
    if prev_contact_valid is not None:
        kwargs["prev_contact_valid"] = prev_contact_valid
    if prev_contact_force is not None:
        kwargs["prev_contact_force"] = prev_contact_force
    if prev_contact_force_valid is not None:
        kwargs["prev_contact_force_valid"] = prev_contact_force_valid
    return rollout_reference_factory(**kwargs)


def _live_robot_state_for_reference(robot_state, *, strip_mjx_data: bool):
    if not strip_mjx_data or not isinstance(robot_state, Mapping):
        return robot_state
    if "mjx_data" not in robot_state:
        return robot_state
    return {name: value for name, value in robot_state.items() if name != "mjx_data"}


def _required_trace_value(trace: Any, name: str) -> Any:
    if not isinstance(trace, dict) or name not in trace:
        raise ValueError(f"MJX rollout trace is missing required field {name}")
    return trace[name]


def _optimizer_execute_trace(window_result: Any, *, execute_steps: int):
    trace = getattr(window_result, "execute_trace", None)
    if not _complete_execute_trace(trace, execute_steps=execute_steps):
        return None
    return trace


def _complete_execute_trace(trace: Any, *, execute_steps: int) -> bool:
    if not isinstance(trace, dict):
        return False
    for name in _FINAL_EXECUTE_TRACE_FIELDS:
        if name not in trace:
            return False
    for name in _ROLLOUT_FRAME_TRACE_FIELDS:
        if name not in trace or not _trace_has_leading_steps(
            trace[name],
            int(execute_steps) + 1,
        ):
            return False
    for name in _ROLLOUT_STEP_TRACE_FIELDS:
        if name not in trace or not _trace_has_leading_steps(
            trace[name],
            int(execute_steps),
        ):
            return False
    return True


def _trace_has_leading_steps(value: Any, steps: int) -> bool:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 1:
        return False
    return int(shape[0]) >= int(steps)


def _supports_jax_controls(runtime) -> bool:
    jnp = getattr(runtime, "jnp", None)
    return all(
        hasattr(jnp, name)
        for name in (
            "asarray",
            "concatenate",
            "zeros",
        )
    )


def _initial_jax_controls(horizon: int, *, runtime):
    return runtime.jnp.asarray(
        np.zeros((int(horizon), QPOS_DIM - 1), dtype=np.float32)
    )


def _joint_limits_from_model_bundle(
    model_bundle,
    *,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    cpu_model = getattr(model_bundle, "cpu_model", None)
    joint_name_to_id = getattr(model_bundle, "joint_name_to_id", None)
    if cpu_model is None and joint_name_to_id is None:
        return None, None
    if cpu_model is None or joint_name_to_id is None:
        raise ValueError("MJX model bundle must expose cpu_model and joint_name_to_id")

    low = []
    high = []
    for joint_name in MUJOCO_JOINT_NAMES:
        joint_id = joint_name_to_id.get(f"robot/{joint_name}")
        if joint_id is None:
            joint_id = joint_name_to_id.get(joint_name)
        if joint_id is None:
            raise ValueError(f"MJX model bundle is missing joint {joint_name!r}")
        if int(cpu_model.jnt_limited[int(joint_id)]):
            low.append(float(cpu_model.jnt_range[int(joint_id), 0]))
            high.append(float(cpu_model.jnt_range[int(joint_id), 1]))
        else:
            low.append(-float("inf"))
            high.append(float("inf"))
    return (
        torch.tensor(low, dtype=torch.float32, device=device),
        torch.tensor(high, dtype=torch.float32, device=device),
    )


def _validated_controls(value, *, horizon: int, device: torch.device) -> torch.Tensor:
    controls = _to_torch(value, device=device)
    expected = (int(horizon), QPOS_DIM - 1)
    if tuple(controls.shape) != expected:
        raise ValueError(
            f"Expected updated_controls shape {expected}, got {tuple(controls.shape)}"
        )
    return controls


def _validated_jax_controls(value, *, horizon: int, runtime):
    controls = runtime.jnp.asarray(value)
    expected = (int(horizon), QPOS_DIM - 1)
    if tuple(int(dim) for dim in controls.shape) != expected:
        raise ValueError(
            f"Expected updated_controls shape {expected}, got {tuple(controls.shape)}"
        )
    return controls


def _validated_execute_chunk(
    value,
    *,
    execute_steps: int,
    device: torch.device,
) -> torch.Tensor:
    execute_chunk = _to_torch(value, device=device)
    if execute_chunk.ndim != 2 or execute_chunk.shape[1] != QPOS_DIM - 1:
        raise ValueError(
            "Expected execute_chunk shape "
            f"(steps, {QPOS_DIM - 1}), got {tuple(execute_chunk.shape)}"
        )
    required_steps = int(execute_steps) + 1
    if int(execute_chunk.shape[0]) < required_steps:
        raise ValueError(
            f"execute_chunk has {execute_chunk.shape[0]} steps, need {required_steps}"
        )
    return execute_chunk


def _validated_jax_execute_chunk(value, *, execute_steps: int, runtime):
    execute_chunk = runtime.jnp.asarray(value)
    if execute_chunk.ndim != 2 or int(execute_chunk.shape[1]) != QPOS_DIM - 1:
        raise ValueError(
            "Expected execute_chunk shape "
            f"(steps, {QPOS_DIM - 1}), got {tuple(execute_chunk.shape)}"
        )
    required_steps = int(execute_steps) + 1
    if int(execute_chunk.shape[0]) < required_steps:
        raise ValueError(
            f"execute_chunk has {execute_chunk.shape[0]} steps, need {required_steps}"
        )
    return execute_chunk


def _apply_execute_chunk_to_refined_qpos(
    refined_qpos: torch.Tensor,
    chunk,
    *,
    baseline_qpos: torch.Tensor,
    start: int,
    execute_steps: int,
    joint_low: torch.Tensor | None,
    joint_high: torch.Tensor | None,
) -> None:
    end = int(start) + int(execute_steps)
    chunk = _artifact_execute_chunk_to_torch(
        chunk,
        execute_steps=execute_steps,
        device=refined_qpos.device,
    )
    base = baseline_qpos[int(start) : end + 1].to(device=chunk.device)
    if tuple(base.shape) != (int(execute_steps) + 1, QPOS_DIM):
        raise ValueError(
            "baseline_qpos slice has shape "
            f"{tuple(base.shape)}, expected {(int(execute_steps) + 1, QPOS_DIM)}"
        )
    qpos_chunk = _controls_to_qpos_torch(
        chunk[: int(execute_steps) + 1],
        base,
        joint_low=joint_low,
        joint_high=joint_high,
    )
    refined_qpos[int(start) : end] = qpos_chunk[:execute_steps]
    refined_qpos[end] = qpos_chunk[int(execute_steps)]


def _apply_executed_command_specs_to_refined_qpos(
    refined_qpos: torch.Tensor,
    specs: list[dict[str, Any]],
    *,
    baseline_qpos: torch.Tensor,
    joint_low: torch.Tensor | None,
    joint_high: torch.Tensor | None,
) -> None:
    for spec in specs:
        _apply_execute_chunk_to_refined_qpos(
            refined_qpos,
            spec["execute_chunk"],
            baseline_qpos=baseline_qpos,
            start=int(spec["start"]),
            execute_steps=int(spec["execute_steps"]),
            joint_low=joint_low,
            joint_high=joint_high,
        )


def _executed_command_chunks_from_specs(
    specs: list[dict[str, Any]],
    *,
    motion: G1Motion,
    baseline_qpos: torch.Tensor,
    joint_low: torch.Tensor | None,
    joint_high: torch.Tensor | None,
    command_builder: Callable[..., G1CommandBatch] | None,
    rollout_config,
    device: torch.device,
) -> list[G1WbcExecutedCommandChunk]:
    return [
        _executed_command_chunk_from_spec(
            spec,
            motion=motion,
            baseline_qpos=baseline_qpos,
            joint_low=joint_low,
            joint_high=joint_high,
            command_builder=command_builder,
            rollout_config=rollout_config,
            device=device,
        )
        for spec in specs
    ]


def _executed_command_chunk_from_spec(
    spec: dict[str, Any],
    *,
    motion: G1Motion,
    baseline_qpos: torch.Tensor,
    joint_low: torch.Tensor | None,
    joint_high: torch.Tensor | None,
    command_builder: Callable[..., G1CommandBatch] | None,
    rollout_config,
    device: torch.device,
) -> G1WbcExecutedCommandChunk:
    start = int(spec["start"])
    horizon = int(spec["horizon"])
    execute_steps = int(spec["execute_steps"])
    controls = _artifact_controls_to_torch(
        spec["updated_controls"],
        horizon=horizon,
        device=device,
    )
    execute_chunk = _artifact_execute_chunk_to_torch(
        spec["execute_chunk"],
        execute_steps=execute_steps,
        device=device,
    )
    prefix_steps = min(int(execute_chunk.shape[0]), int(controls.shape[0]))
    if prefix_steps > 0:
        controls[:prefix_steps] = execute_chunk[:prefix_steps]
    base_qpos = _slice_qpos_padded(
        baseline_qpos,
        start,
        horizon,
    ).to(device=device)
    qpos_chunk = _controls_to_qpos_torch(
        controls,
        base_qpos,
        joint_low=joint_low,
        joint_high=joint_high,
    )
    window_motion = _slice_motion_padded(motion, int(start), int(horizon))
    command = _command_from_refined_qpos(
        window_motion,
        qpos_chunk,
        None,
        command_builder=command_builder,
        rollout_config=rollout_config,
    )
    return G1WbcExecutedCommandChunk(
        start=start,
        execute_steps=execute_steps,
        horizon_steps=horizon,
        command=command,
        replay_state=_replay_state_from_spec(
            spec.get("replay_state"),
            motion=motion,
            device=device,
        ),
    )


def _window_replay_state_spec(
    *,
    start: int,
    current_robot_state: Any,
    current_obs_state: Any,
) -> dict[str, Any]:
    return {
        "start": int(start),
        "robot_state": current_robot_state,
        "obs_state": current_obs_state,
    }


def _replay_state_from_spec(
    spec: dict[str, Any] | None,
    *,
    motion: G1Motion,
    device: torch.device,
) -> G1WbcWindowReplayState | None:
    if spec is None:
        return None
    start = int(spec["start"])
    robot_state = spec.get("robot_state")
    if robot_state is None:
        initial_qpos = (
            motion.qpos()[start].to(device=device, dtype=torch.float32).detach().clone()
        )
        initial_qvel = (
            motion.qvel()[start].to(device=device, dtype=torch.float32).detach().clone()
        )
    else:
        initial_qpos = _state_vector_to_torch(
            _state_value(robot_state, "qpos"),
            dim=QPOS_DIM,
            device=device,
            name="qpos",
        )
        initial_qvel = _state_vector_to_torch(
            _state_value(robot_state, "qvel"),
            dim=QVEL_DIM,
            device=device,
            name="qvel",
        )
    obs_state = spec.get("obs_state")
    return G1WbcWindowReplayState(
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        initial_last_action=_obs_last_action_to_torch(obs_state, device=device),
        initial_history_state=_obs_history_state_to_torch(obs_state, device=device),
    )


def _state_value(state: Any, name: str) -> Any:
    if isinstance(state, dict):
        return state[name]
    return getattr(state, name)


def _obs_last_action_to_torch(
    obs_state: Any,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if obs_state is None or not hasattr(obs_state, "last_action"):
        return None
    value = getattr(obs_state, "last_action")
    if value is None:
        return None
    return _state_vector_to_torch(
        value,
        dim=ACTION_DIM,
        device=device,
        name="last_action",
    )


def _obs_history_state_to_torch(
    obs_state: Any,
    *,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor | int | None]] | None:
    if obs_state is None or not hasattr(obs_state, "history"):
        return None
    history = getattr(obs_state, "history")
    if not history:
        return None
    out: dict[str, dict[str, torch.Tensor | int | None]] = {}
    for name, value in history.items():
        if value is None:
            continue
        buffer = _obs_history_buffer_to_torch(value, device=device, name=str(name))
        out[str(name)] = {
            "buffer": buffer,
            "pointer": OBS_HISTORY_LENGTH - 1,
            "num_pushes": torch.full(
                (1,),
                OBS_HISTORY_LENGTH,
                dtype=torch.long,
                device=device,
            ),
        }
    return out or None


def _obs_history_buffer_to_torch(
    value: Any,
    *,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    tensor = _artifact_value_to_torch(value, device=device, dtype=torch.float32)
    if tensor.ndim >= 3 and int(tensor.shape[0]) == 1:
        if int(tensor.shape[1]) != OBS_HISTORY_LENGTH:
            raise ValueError(
                f"Expected obs history {name!r} length {OBS_HISTORY_LENGTH}, "
                f"got {tuple(tensor.shape)}."
            )
        order = [1, 0] + list(range(2, tensor.ndim))
        return tensor.permute(*order).contiguous()
    if tensor.ndim >= 2 and int(tensor.shape[0]) == OBS_HISTORY_LENGTH:
        return tensor.unsqueeze(1).contiguous()
    raise ValueError(
        f"Expected obs history {name!r} shape (1, {OBS_HISTORY_LENGTH}, ...) "
        f"or ({OBS_HISTORY_LENGTH}, ...), got {tuple(tensor.shape)}."
    )


def _state_vector_to_torch(
    value: Any,
    *,
    dim: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    tensor = _artifact_value_to_torch(value, device=device, dtype=torch.float32)
    if tensor.ndim == 2 and int(tensor.shape[0]) == 1:
        tensor = tensor[0]
    if tensor.ndim != 1 or int(tensor.shape[0]) != dim:
        raise ValueError(
            f"Expected {name} shape ({dim},) or (1, {dim}), "
            f"got {tuple(tensor.shape)}."
        )
    return tensor.contiguous()


def _artifact_value_to_torch(
    value: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype).detach().clone()
    if hasattr(value, "__dlpack__"):
        tensor = torch.utils.dlpack.from_dlpack(value)
        return tensor.to(device=device, dtype=dtype).detach().clone()
    return torch.as_tensor(
        np.array(value, dtype=np.float32, copy=True),
        dtype=dtype,
        device=device,
    )


def _artifact_controls_to_torch(
    value,
    *,
    horizon: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        controls = value.to(device=device, dtype=torch.float32).detach().clone()
    elif hasattr(value, "__dlpack__"):
        controls = torch.utils.dlpack.from_dlpack(value)
        controls = controls.to(device=device, dtype=torch.float32).detach().clone()
    else:
        controls = torch.as_tensor(
            np.array(value, dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=device,
        )
    expected = (int(horizon), QPOS_DIM - 1)
    if tuple(controls.shape) != expected:
        raise ValueError(
            f"Expected artifact controls shape {expected}, got {tuple(controls.shape)}"
        )
    return controls


def _artifact_execute_chunk_to_torch(
    value,
    *,
    execute_steps: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        chunk = value.to(device=device, dtype=torch.float32).detach().clone()
    elif hasattr(value, "__dlpack__"):
        chunk = torch.utils.dlpack.from_dlpack(value)
        chunk = chunk.to(device=device, dtype=torch.float32).detach().clone()
    else:
        chunk = torch.as_tensor(
            np.array(value, dtype=np.float32, copy=True),
            dtype=torch.float32,
            device=device,
        )
    if chunk.ndim != 2 or int(chunk.shape[1]) != QPOS_DIM - 1:
        raise ValueError(
            "Expected execute_chunk shape "
            f"(steps, {QPOS_DIM - 1}), got {tuple(chunk.shape)}"
        )
    required_steps = int(execute_steps) + 1
    if int(chunk.shape[0]) < required_steps:
        raise ValueError(
            f"execute_chunk has {chunk.shape[0]} steps, need {required_steps}"
        )
    return chunk


def _controls_to_qpos_torch(
    controls: torch.Tensor,
    base_qpos: torch.Tensor,
    *,
    joint_low: torch.Tensor | None,
    joint_high: torch.Tensor | None,
) -> torch.Tensor:
    qpos = base_qpos.clone()
    qpos[..., :3] = qpos[..., :3] + controls[..., :3]
    delta_quat = quat_from_axis_angle(controls[..., 3:6])
    qpos[..., 3:7] = normalize(quat_mul(delta_quat, qpos[..., 3:7]))
    joints = qpos[..., 7:] + controls[..., 6:]
    if joint_low is not None and joint_high is not None:
        joints = torch.clamp(joints, joint_low, joint_high)
    qpos[..., 7:] = joints
    return qpos.contiguous()


def _slice_qpos_padded(qpos: torch.Tensor, start: int, length: int) -> torch.Tensor:
    start = int(start)
    length = int(length)
    out = qpos[start : start + length]
    if int(out.shape[0]) < length:
        out = torch.cat([out, out[-1:].repeat(length - int(out.shape[0]), 1)], dim=0)
    return out.contiguous()


def _slice_motion_padded(motion: G1Motion, start: int, length: int) -> G1Motion:
    start = int(start)
    length = int(length)

    def sl(value: torch.Tensor) -> torch.Tensor:
        out = value[start : start + length]
        if int(out.shape[0]) < length:
            repeats = [length - int(out.shape[0])] + [1] * (out.ndim - 1)
            out = torch.cat([out, out[-1:].repeat(*repeats)], dim=0)
        return out.contiguous()

    return G1Motion(
        path=motion.path,
        motion_type=motion.motion_type,
        fps=motion.fps,
        joint_pos=sl(motion.joint_pos),
        joint_vel=sl(motion.joint_vel),
        body_pos_w=sl(motion.body_pos_w),
        body_quat_w=sl(motion.body_quat_w),
        body_lin_vel_w=sl(motion.body_lin_vel_w),
        body_ang_vel_w=sl(motion.body_ang_vel_w),
        contact=sl(motion.contact),
    )


def _shift_controls(
    controls: torch.Tensor,
    *,
    execute_steps: int,
    horizon: int,
    use_warm_start: bool,
    device: torch.device,
) -> torch.Tensor:
    if not use_warm_start:
        return torch.zeros(
            int(horizon),
            QPOS_DIM - 1,
            dtype=controls.dtype,
            device=device,
        )
    previous = controls[int(execute_steps) :]
    tail_steps = int(horizon) - int(previous.shape[0])
    if tail_steps > 0:
        tail = torch.zeros(
            tail_steps,
            QPOS_DIM - 1,
            dtype=controls.dtype,
            device=device,
        )
        controls = torch.cat([previous, tail], dim=0)
    else:
        controls = previous[: int(horizon)]
    return controls.contiguous()


def _shift_jax_controls(
    controls,
    *,
    execute_steps: int,
    horizon: int,
    use_warm_start: bool,
    runtime,
):
    jnp = runtime.jnp
    if not use_warm_start:
        return jnp.zeros((int(horizon), QPOS_DIM - 1))
    previous = controls[int(execute_steps) :]
    tail_steps = int(horizon) - int(previous.shape[0])
    if tail_steps > 0:
        tail = jnp.zeros((tail_steps, QPOS_DIM - 1))
        return jnp.concatenate([previous, tail], axis=0)
    return previous[: int(horizon)]


_ROLLOUT_FRAME_TRACE_FIELDS = (
    "qpos",
    "qvel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "contact_indicator",
    "contact_force",
    "floor_contact_indicator",
    "floor_contact_force",
)

_ROLLOUT_OPTIONAL_FRAME_TRACE_FIELDS = (
    "contact_force_first_row",
    "floor_contact_force_first_row",
    "floor_contact_force_peak_source",
)

_ROLLOUT_STEP_TRACE_FIELDS = ("actions", "controls")

_FINAL_EXECUTE_TRACE_FIELDS = (
    "final_robot_state",
    "final_obs_state",
    "final_prev_control",
    "final_prev_joint_acc",
    "final_prev_contact",
    "final_prev_contact_valid",
    "final_prev_contact_force",
    "final_prev_contact_force_valid",
)


def _materialize_deferred_execute_traces(
    traces: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    *,
    rollout_reference_factory: Callable[..., Any] | None,
    rollout_tracer: Callable[..., Any],
    motion: G1Motion,
    actor_params,
    model_bundle,
    runtime,
) -> None:
    for spec in specs:
        execute_controls = spec["execute_controls"]
        start = int(spec["start"])
        execute_steps = int(spec["execute_steps"])
        reference = _window_reference(
            rollout_reference_factory,
            start=start,
            motion=motion,
            controls=execute_controls,
            actor_params=actor_params,
            model_bundle=model_bundle,
            runtime=runtime,
            initial_robot_state=spec.get("initial_robot_state"),
            obs_state=spec.get("obs_state"),
            obs_initialized=spec.get("obs_initialized"),
            prev_control=spec.get("prev_control"),
            prev_joint_acc=spec.get("prev_joint_acc"),
            prev_contact=spec.get("prev_contact"),
            prev_contact_valid=spec.get("prev_contact_valid"),
            prev_contact_force=spec.get("prev_contact_force"),
            prev_contact_force_valid=spec.get("prev_contact_force_valid"),
            include_guided_candidate=False,
        )
        trace = rollout_tracer(
            execute_controls[None, :, :],
            reference,
            actor_params,
            model_bundle,
        )
        _block_trace_until_ready(trace)
        _record_execute_trace(
            traces,
            trace,
            start=start,
            execute_steps=execute_steps,
        )


def _record_execute_trace(
    traces: list[dict[str, Any]],
    trace: Any,
    *,
    start: int,
    execute_steps: int,
) -> None:
    if not isinstance(trace, dict):
        return
    if not all(name in trace for name in (*_ROLLOUT_FRAME_TRACE_FIELDS, *_ROLLOUT_STEP_TRACE_FIELDS)):
        return
    traces.append(
        {
            "start": int(start),
            "execute_steps": int(execute_steps),
            "trace": trace,
        }
    )


def _rollout_from_execute_traces(
    traces: list[dict[str, Any]],
    *,
    total_steps: int,
    device: torch.device,
):
    if not traces:
        return None
    expected_start = 0
    frame_values: dict[str, list[torch.Tensor]] = {
        name: [] for name in _ROLLOUT_FRAME_TRACE_FIELDS
    }
    optional_frame_values: dict[str, list[torch.Tensor]] = {
        name: [] for name in _ROLLOUT_OPTIONAL_FRAME_TRACE_FIELDS
    }
    optional_frame_available = {
        name: all(name in item["trace"] for item in traces)
        for name in _ROLLOUT_OPTIONAL_FRAME_TRACE_FIELDS
    }
    step_values: dict[str, list[torch.Tensor]] = {
        name: [] for name in _ROLLOUT_STEP_TRACE_FIELDS
    }
    ref_indices: list[torch.Tensor] = []
    for item in traces:
        start = int(item["start"])
        steps = int(item["execute_steps"])
        if start != expected_start or steps < 1:
            return None
        trace = item["trace"]
        for name in _ROLLOUT_FRAME_TRACE_FIELDS:
            value = _trace_tensor(trace[name], device=device)
            if int(value.shape[0]) < steps + 1:
                return None
            chunk = value[: steps + 1]
            if start > 0:
                chunk = chunk[1:]
            frame_values[name].append(chunk)
        for name in _ROLLOUT_OPTIONAL_FRAME_TRACE_FIELDS:
            if not optional_frame_available[name]:
                continue
            value = _trace_tensor(trace[name], device=device)
            if int(value.shape[0]) < steps + 1:
                return None
            chunk = value[: steps + 1]
            if start > 0:
                chunk = chunk[1:]
            optional_frame_values[name].append(chunk)
        for name in _ROLLOUT_STEP_TRACE_FIELDS:
            value = _trace_tensor(trace[name], device=device)
            if int(value.shape[0]) < steps:
                return None
            step_values[name].append(value[:steps])
        chunk_ref_indices = torch.cat(
            (
                torch.tensor([start], dtype=torch.long, device=device),
                torch.arange(
                    start,
                    start + steps,
                    dtype=torch.long,
                    device=device,
                ),
            ),
            dim=0,
        )
        if start > 0:
            chunk_ref_indices = chunk_ref_indices[1:]
        ref_indices.append(chunk_ref_indices.view(-1, 1))
        expected_start += steps
    if expected_start != int(total_steps):
        return None

    from spider.tasks.g1_wbc.rollout import RolloutResult

    kwargs: dict[str, Any] = {}
    if optional_frame_available["contact_force_first_row"]:
        kwargs["contact_force_first_row"] = torch.cat(
            optional_frame_values["contact_force_first_row"],
            dim=0,
        )
    if optional_frame_available["floor_contact_force_first_row"]:
        kwargs["floor_contact_force_first_row"] = torch.cat(
            optional_frame_values["floor_contact_force_first_row"],
            dim=0,
        )
    if optional_frame_available["floor_contact_force_peak_source"]:
        floor_contact_force_peak_source = torch.cat(
            optional_frame_values["floor_contact_force_peak_source"],
            dim=0,
        )
        if _has_valid_floor_contact_force_peak_source(
            floor_contact_force_peak_source
        ):
            kwargs["floor_contact_force_peak_source"] = (
                floor_contact_force_peak_source
            )
    return RolloutResult(
        qpos=torch.cat(frame_values["qpos"], dim=0),
        qvel=torch.cat(frame_values["qvel"], dim=0),
        body_pos_w=torch.cat(frame_values["body_pos_w"], dim=0),
        body_quat_w=torch.cat(frame_values["body_quat_w"], dim=0),
        body_lin_vel_w=torch.cat(frame_values["body_lin_vel_w"], dim=0),
        body_ang_vel_w=torch.cat(frame_values["body_ang_vel_w"], dim=0),
        actions=torch.cat(step_values["actions"], dim=0),
        controls=torch.cat(step_values["controls"], dim=0),
        contact_indicator=torch.cat(frame_values["contact_indicator"], dim=0),
        contact_force=torch.cat(frame_values["contact_force"], dim=0),
        floor_contact_indicator=torch.cat(
            frame_values["floor_contact_indicator"],
            dim=0,
        ),
        floor_contact_force=torch.cat(frame_values["floor_contact_force"], dim=0),
        ref_indices=torch.cat(ref_indices, dim=0),
        **kwargs,
    )


def _trace_tensor(value, *, device: torch.device) -> torch.Tensor:
    tensor = _to_torch(value, device=device)
    if tensor.ndim >= 2:
        return tensor.detach().clone()
    raise ValueError(f"Expected execute trace tensor with at least 2 dims, got {tensor.shape}")


def _has_valid_floor_contact_force_peak_source(value: torch.Tensor) -> bool:
    if value.ndim < 4 or int(value.shape[-1]) < 1:
        return False
    return bool(torch.any(value[..., 0] >= 0).item())


def _validate_rollout_shape(
    rollout,
    *,
    total_steps: int,
    refined_qpos: torch.Tensor,
    require_qpos_match: bool = True,
) -> None:
    frames = int(total_steps) + 1
    bodies = len(MUJOCO_BODY_NAMES)
    _require_shape("refined_qpos", refined_qpos, (frames, QPOS_DIM))
    _require_shape("rollout.qpos", rollout.qpos, (frames, 1, QPOS_DIM))
    if require_qpos_match and not torch.allclose(
        rollout.qpos[:, 0].to(device=refined_qpos.device, dtype=refined_qpos.dtype),
        refined_qpos,
        atol=1.0e-5,
        rtol=1.0e-5,
    ):
        raise ValueError("rollout.qpos must match refined_qpos for accepted MJX runs")
    _require_shape("rollout.qvel", rollout.qvel, (frames, 1, QVEL_DIM))
    _require_shape("rollout.body_pos_w", rollout.body_pos_w, (frames, 1, bodies, 3))
    _require_shape("rollout.body_quat_w", rollout.body_quat_w, (frames, 1, bodies, 4))
    _require_shape(
        "rollout.body_lin_vel_w",
        rollout.body_lin_vel_w,
        (frames, 1, bodies, 3),
    )
    _require_shape(
        "rollout.body_ang_vel_w",
        rollout.body_ang_vel_w,
        (frames, 1, bodies, 3),
    )
    _require_shape("rollout.actions", rollout.actions, (int(total_steps), 1, ACTION_DIM))
    _require_shape("rollout.controls", rollout.controls, (int(total_steps), 1, ACTION_DIM))
    _require_shape("rollout.contact_indicator", rollout.contact_indicator, (frames, 1, 2))
    _require_shape("rollout.contact_force", rollout.contact_force, (frames, 1, 2))
    _require_shape("rollout.ref_indices", rollout.ref_indices, (frames, 1))
    if rollout.floor_contact_indicator is not None:
        _require_shape(
            "rollout.floor_contact_indicator",
            rollout.floor_contact_indicator,
            (frames, 1, 3),
        )
    if rollout.floor_contact_force is not None:
        _require_shape(
            "rollout.floor_contact_force",
            rollout.floor_contact_force,
            (frames, 1, 3),
        )
    contact_force_first_row = getattr(rollout, "contact_force_first_row", None)
    if contact_force_first_row is not None:
        _require_shape(
            "rollout.contact_force_first_row",
            contact_force_first_row,
            (frames, 1, 2),
        )
    floor_contact_force_first_row = getattr(
        rollout,
        "floor_contact_force_first_row",
        None,
    )
    if floor_contact_force_first_row is not None:
        _require_shape(
            "rollout.floor_contact_force_first_row",
            floor_contact_force_first_row,
            (frames, 1, 3),
        )
    floor_contact_force_peak_source = getattr(
        rollout,
        "floor_contact_force_peak_source",
        None,
    )
    if floor_contact_force_peak_source is not None:
        _require_shape(
            "rollout.floor_contact_force_peak_source",
            floor_contact_force_peak_source,
            (frames, 1, 3, 8),
        )


def _validate_command_shape(command: G1CommandBatch, *, total_steps: int) -> None:
    frames = int(total_steps) + 1
    bodies = len(MUJOCO_BODY_NAMES)
    _require_shape("command.joint_pos", command.joint_pos, (frames, 1, ACTION_DIM))
    _require_shape("command.joint_vel", command.joint_vel, (frames, 1, ACTION_DIM))
    _require_shape("command.body_pos_w", command.body_pos_w, (frames, 1, bodies, 3))
    _require_shape("command.body_quat_w", command.body_quat_w, (frames, 1, bodies, 4))
    _require_shape(
        "command.body_lin_vel_w",
        command.body_lin_vel_w,
        (frames, 1, bodies, 3),
    )
    _require_shape(
        "command.body_ang_vel_w",
        command.body_ang_vel_w,
        (frames, 1, bodies, 3),
    )
    _require_shape(
        "command.qpos_trajectory",
        command.qpos_trajectory,
        (frames, 1, QPOS_DIM),
    )
    _require_shape(
        "command.qvel_trajectory",
        command.qvel_trajectory,
        (frames, 1, QVEL_DIM),
    )


def _require_shape(name: str, value, expected: tuple[int, ...]) -> None:
    if value is None or not hasattr(value, "shape"):
        raise ValueError(f"Expected {name} shape {expected}, got {value!r}")
    actual = tuple(value.shape)
    if actual != expected:
        raise ValueError(f"Expected {name} shape {expected}, got {actual}")


def _to_torch(value, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32).detach().clone()
    if hasattr(value, "__dlpack__"):
        try:
            tensor = torch.utils.dlpack.from_dlpack(value)
        except (BufferError, RuntimeError, TypeError):
            pass
        else:
            return tensor.to(device=device, dtype=torch.float32).detach().clone()
    array = np.array(value, dtype=np.float32, copy=True)
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _scalar_info(info: dict[str, Any], name: str) -> float:
    if name not in info:
        raise ValueError(f"Missing optimizer info field {name}")
    value = info[name]
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() != 1:
            raise ValueError(f"Optimizer info field {name} must be scalar")
        scalar = float(tensor.reshape(()).item())
    elif hasattr(value, "item"):
        try:
            scalar = float(value.item())
        except Exception as exc:
            raise ValueError(f"Optimizer info field {name} must be scalar") from exc
    else:
        try:
            scalar = float(value)
        except Exception as exc:
            raise ValueError(f"Optimizer info field {name} must be numeric") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"Optimizer info field {name} must be finite")
    return scalar


def _raw_info_scalar(info: dict[str, Any], name: str):
    if name not in info:
        raise ValueError(f"Missing optimizer info field {name}")
    value = info[name]
    shape = tuple(int(dim) for dim in getattr(value, "shape", ()))
    if shape:
        raise ValueError(f"Optimizer info field {name} must be scalar")
    return value


def _scores_to_torch(values: list[Any], *, device: torch.device) -> torch.Tensor:
    if not values:
        return torch.empty(0, dtype=torch.float32, device=device)
    tensor = _to_torch(values, device=device).reshape(-1)
    if not torch.isfinite(tensor).all():
        raise ValueError("Optimizer info field best_score must be finite")
    return tensor


def _command_from_refined_qpos(
    motion: G1Motion,
    refined_qpos: torch.Tensor,
    rollout,
    *,
    command_builder: Callable[..., G1CommandBatch] | None = None,
    rollout_config=None,
) -> G1CommandBatch:
    del rollout
    if command_builder is None:
        command_builder = _default_command_builder
    return command_builder(
        motion,
        refined_qpos[:, None, :].contiguous(),
        rollout_config,
    )


def _default_command_builder(
    motion: G1Motion,
    qpos_trajectory: torch.Tensor,
    rollout_config,
) -> G1CommandBatch:
    from spider.tasks.g1_wbc.rollout import command_batch_from_qpos_trajectory

    return command_batch_from_qpos_trajectory(
        motion,
        qpos_trajectory,
        rollout_config,
    )
