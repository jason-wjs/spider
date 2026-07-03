"""MJX backend entry point for G1 WBC sampled MPC."""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np
import torch

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    QPOS_DIM,
    QVEL_DIM,
)
from spider.tasks.g1_wbc.math_utils import normalize, quat_from_axis_angle, quat_mul
from spider.tasks.g1_wbc.mjx_contacts import get_contact_profile
from spider.tasks.g1_wbc.mjx_optimizer import JaxWindowOptimizerConfig, optimize_window
from spider.tasks.g1_wbc.mjx_policy import convert_wbc_actor_to_jax
from spider.tasks.g1_wbc.mjx_runtime import require_mjx_runtime
from spider.tasks.g1_wbc.motion import G1CommandBatch, G1Motion
from spider.tasks.g1_wbc.result_types import G1WbcMpcRun, G1WbcSpiderResult


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
    rollout_reference_factory: Callable[..., Any] | None = None,
    rollout_tracer: Callable[..., Any] | None = None,
    command_builder: Callable[..., G1CommandBatch] | None = None,
    enable_physics_scan: bool = False,
):
    """Run the MJX full-rollout backend or fail before touching Warp state."""

    compile_start = time.perf_counter()
    if runtime is None:
        runtime = require_mjx_runtime()
    device = torch.device(rollout_config.device)
    _validate_single_gpu_runtime(runtime, device=device)

    auto_enable_physics_scan = rollout_factory is None or (
        optimizer is optimize_window and rollout_scorer is None
    )
    physics_scan_enabled = bool(enable_physics_scan or auto_enable_physics_scan)
    if physics_scan_enabled:
        if rollout_scorer is None or rollout_reference_factory is None:
            components = _default_rollout_components(
                runtime=runtime,
                method=method,
                reward_weights=reward_weights,
                **_guided_component_kwargs(spider_config),
            )
            if rollout_scorer is None:
                rollout_scorer = components.rollout_scorer
            if rollout_reference_factory is None:
                rollout_reference_factory = components.rollout_reference_factory
            if rollout_tracer is None:
                rollout_tracer = getattr(components, "rollout_tracer", None)
        if rollout_factory is None:
            rollout_factory = _default_static_rollout_factory(execute_rollout_config)

    if rollout_factory is None or (
        optimizer is optimize_window and rollout_scorer is None
    ):
        raise NotImplementedError(
            "MJX runtime is available, but the production MJX physics scan has not "
            "been wired yet. Keep using --mpc-backend mujoco_warp for production."
        )

    if model_factory is None:
        model_factory = _default_model_factory
    model_bundle = model_factory(profile_name="wxy_parity", require_runtime=True)
    actor_params = policy_converter(actor, jnp=runtime.jnp)

    total_steps = int(total_steps)
    horizon = int(spider_config.horizon_steps)
    control_steps = int(spider_config.ctrl_steps)
    use_warm_start = bool(getattr(spider_config, "use_warm_start", True))
    window_config = _window_config_from_spider(spider_config)
    use_jax_controls = optimizer is optimize_window and _supports_jax_controls(runtime)
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
            state={"rollout_fn": rollout_scorer or _placeholder_rollout_scores},
            controls=controls,
            reference=warmup_reference,
            actor_params=actor_params,
            model_bundle=model_bundle,
            key=(int(seed), _JIT_WARMUP_KEY_FOLD),
            runtime=runtime,
        )
        _block_window_result_until_ready(warmup_result)
        jit_warmup_wall_time_sec = time.perf_counter() - warmup_start
        jit_warmup_enabled = True
        del warmup_result, warmup_reference
    compile_init_wall_time_sec = time.perf_counter() - compile_start
    steady_start = time.perf_counter()
    while sim_step < total_steps:
        window_reference = _window_reference(
            rollout_reference_factory,
            start=sim_step,
            motion=motion,
            controls=controls,
            actor_params=actor_params,
            model_bundle=model_bundle,
            runtime=runtime,
            initial_robot_state=current_robot_state,
            obs_state=current_obs_state,
            obs_initialized=current_obs_initialized if current_obs_state is not None else None,
            prev_control=current_prev_control,
            prev_joint_acc=current_prev_joint_acc,
            prev_contact=current_prev_contact,
            prev_contact_valid=current_prev_contact_valid,
            prev_contact_force=current_prev_contact_force,
            prev_contact_force_valid=current_prev_contact_force_valid,
        )
        window_result = optimizer(
            config=window_config,
            state={"rollout_fn": rollout_scorer or _placeholder_rollout_scores},
            controls=controls,
            reference=window_reference,
            actor_params=actor_params,
            model_bundle=model_bundle,
            key=(int(seed), len(infos)),
            runtime=runtime,
        )
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
        execute_chunk = _validated_execute_chunk(
            window_result.execute_chunk,
            execute_steps=execute_steps,
            device=device,
        )
        info = dict(window_result.info)
        window_accepted = _window_accepted(info)
        if window_accepted:
            _require_physics_scan_evidence(
                info,
                enabled=physics_scan_enabled,
                horizon=horizon,
            )
            controls = updated_controls
            _apply_execute_chunk_to_refined_qpos(
                refined_qpos,
                execute_chunk,
                baseline_qpos=baseline_qpos,
                start=sim_step,
                execute_steps=execute_steps,
                joint_low=joint_low,
                joint_high=joint_high,
            )
            if rollout_tracer is not None:
                execute_controls = updated_controls[:execute_steps]
                execute_reference = _window_reference(
                    rollout_reference_factory,
                    start=sim_step,
                    motion=motion,
                    controls=execute_controls,
                    actor_params=actor_params,
                    model_bundle=model_bundle,
                    runtime=runtime,
                    initial_robot_state=current_robot_state,
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
                )
                execute_trace = rollout_tracer(
                    execute_controls[None, :, :],
                    execute_reference,
                    actor_params,
                    model_bundle,
                )
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
        info.update(
            {
                "backend": "mjx",
                "sim_step": sim_step,
                "execute_steps": execute_steps,
                "accepted": window_accepted,
            }
        )
        infos.append(info)
        best_scores.append(
            _raw_info_scalar(info, "best_score")
            if use_jax_controls
            else _scalar_info(info, "best_score")
        )
        accepted_windows += int(window_accepted)
        sim_step += execute_steps
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
    steady_state_wall_time_sec = time.perf_counter() - steady_start

    rollout = rollout_factory(
        motion,
        total_steps,
        device=device,
        refined_qpos=refined_qpos.detach().clone(),
    )
    _validate_rollout_shape(rollout, total_steps=total_steps, refined_qpos=refined_qpos)
    command = _command_from_refined_qpos(
        motion,
        refined_qpos,
        rollout,
        command_builder=command_builder,
        rollout_config=execute_rollout_config,
    )
    _validate_command_shape(command, total_steps=total_steps)
    contact_metadata = _contact_metadata(
        model_bundle=model_bundle,
        infos=infos,
        rollout=rollout,
    )
    physics_step_metadata = _physics_step_count_metadata(infos)
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
            "used_baseline_fallback": False,
            "accepted_windows": accepted_windows,
            "num_windows": len(infos),
            "planning_horizon_steps": horizon,
            "control_steps": control_steps,
            "use_warm_start": use_warm_start,
            "final_noise_scale": float(getattr(spider_config, "final_noise_scale", 1.0)),
            "use_guided_candidate": _use_guided_candidate(spider_config),
            "compile_init_wall_time_sec": compile_init_wall_time_sec,
            "jit_warmup_enabled": jit_warmup_enabled,
            "jit_warmup_wall_time_sec": jit_warmup_wall_time_sec,
            "steady_state_wall_time_sec": steady_state_wall_time_sec,
            "runtime_visible_devices": tuple(
                getattr(getattr(runtime, "status", None), "visible_devices", ())
            ),
            "runtime_gpu_name": _runtime_gpu_name(device),
            **physics_step_metadata,
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
        iterations=int(spider_config.max_num_iterations),
        final_noise_scale=float(getattr(spider_config, "final_noise_scale", 1.0)),
        use_guided_candidate=_use_guided_candidate(spider_config),
    )


def _placeholder_rollout_scores(samples, reference, actor_params, model_bundle):
    del reference, actor_params, model_bundle
    return samples[..., 0, 0]


def _block_window_result_until_ready(window_result) -> None:
    _block_until_ready(getattr(window_result, "updated_controls", None))
    _block_until_ready(getattr(window_result, "execute_chunk", None))
    for value in getattr(window_result, "info", {}).values():
        _block_until_ready(value)


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
    mapping = {
        "root_pos": float(reward_weights.get("root_pos_error", 0.0)),
        "root_rot": float(reward_weights.get("root_rot_error", 0.0)),
        "body_global_pos": float(reward_weights.get("body_global_pos_error", 0.0)),
        "body_global_rot": float(reward_weights.get("body_global_rot_error", 0.0)),
        "ee_global_pos": float(reward_weights.get("ee_global_pos_error", 0.0)),
        "ee_global_rot": float(reward_weights.get("ee_global_rot_error", 0.0)),
        "bad_floor_contact": float(reward_weights.get("bad_floor_contact", 0.0)),
        "bad_floor_force_excess": float(
            reward_weights.get("bad_floor_force_excess", 0.0)
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


def _window_accepted(info: dict[str, Any]) -> bool:
    if "accepted" not in info:
        raise ValueError("MJX optimizer info must include boolean accepted metadata")
    accepted = _python_scalar(info.get("accepted"))
    if isinstance(accepted, bool):
        return accepted
    if isinstance(accepted, (int, float)) and math.isfinite(float(accepted)):
        return bool(accepted)
    raise ValueError("MJX optimizer info accepted metadata must be boolean-like")


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


def _required_trace_value(trace: Any, name: str) -> Any:
    if not isinstance(trace, dict) or name not in trace:
        raise ValueError(f"MJX rollout trace is missing required field {name}")
    return trace[name]


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


def _apply_execute_chunk_to_refined_qpos(
    refined_qpos: torch.Tensor,
    chunk: torch.Tensor,
    *,
    baseline_qpos: torch.Tensor,
    start: int,
    execute_steps: int,
    joint_low: torch.Tensor | None,
    joint_high: torch.Tensor | None,
) -> None:
    end = int(start) + int(execute_steps)
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


def _validate_rollout_shape(rollout, *, total_steps: int, refined_qpos: torch.Tensor) -> None:
    frames = int(total_steps) + 1
    bodies = len(MUJOCO_BODY_NAMES)
    _require_shape("refined_qpos", refined_qpos, (frames, QPOS_DIM))
    _require_shape("rollout.qpos", rollout.qpos, (frames, 1, QPOS_DIM))
    if not torch.allclose(
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
