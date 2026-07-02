"""MJX backend entry point for G1 WBC sampled MPC."""

from __future__ import annotations

import time
from typing import Any, Callable

import torch

from spider.tasks.g1_wbc.constants import QPOS_DIM
from spider.tasks.g1_wbc.mjx_optimizer import JaxWindowOptimizerConfig, optimize_window
from spider.tasks.g1_wbc.mjx_policy import convert_wbc_actor_to_jax
from spider.tasks.g1_wbc.mjx_runtime import require_mjx_runtime
from spider.tasks.g1_wbc.motion import G1CommandBatch, G1Motion


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
):
    """Run the MJX full-rollout backend or fail before touching Warp state."""

    del execute_rollout_config
    compile_start = time.perf_counter()
    if runtime is None:
        runtime = require_mjx_runtime()

    if rollout_factory is None or optimizer is optimize_window:
        raise NotImplementedError(
            "MJX runtime is available, but the production MJX physics scan has not "
            "been wired yet. Keep using --mpc-backend mujoco_warp for production."
        )

    if model_factory is None:
        model_factory = _default_model_factory
    model_bundle = model_factory(profile_name="wxy_parity", require_runtime=True)
    actor_params = policy_converter(actor, jnp=runtime.jnp)
    compile_init_wall_time_sec = time.perf_counter() - compile_start

    device = torch.device(rollout_config.device)
    total_steps = int(total_steps)
    horizon = int(spider_config.horizon_steps)
    control_steps = int(spider_config.ctrl_steps)
    controls = torch.zeros(horizon, QPOS_DIM - 1, dtype=torch.float32, device=device)
    refined_qpos = motion.qpos()[: total_steps + 1].to(device).detach().clone()
    infos: list[dict[str, Any]] = []
    best_scores: list[float] = []
    sim_step = 0
    accepted_windows = 0
    steady_start = time.perf_counter()
    while sim_step < total_steps:
        window_result = optimizer(
            config=_window_config_from_spider(spider_config),
            state={"rollout_fn": _placeholder_rollout_scores},
            controls=controls,
            reference={"start": sim_step},
            actor_params=actor_params,
            model_bundle=model_bundle,
            key=(int(seed), len(infos)),
            runtime=runtime,
        )
        execute_steps = min(control_steps, total_steps - sim_step)
        controls = _validated_controls(
            window_result.updated_controls,
            horizon=horizon,
            device=device,
        )
        _validated_execute_chunk(
            window_result.execute_chunk,
            execute_steps=execute_steps,
            device=device,
        )
        _apply_execute_chunk_to_refined_qpos(
            refined_qpos,
            window_result.execute_chunk,
            start=sim_step,
            execute_steps=execute_steps,
            device=device,
        )
        info = dict(window_result.info)
        info.update(
            {
                "backend": "mjx",
                "sim_step": sim_step,
                "execute_steps": execute_steps,
                "accepted": True,
            }
        )
        infos.append(info)
        best_scores.append(_scalar_info(info, "best_score"))
        accepted_windows += 1
        sim_step += execute_steps
    steady_state_wall_time_sec = time.perf_counter() - steady_start

    rollout = rollout_factory(motion, total_steps, device=device)
    _validate_rollout_shape(rollout, total_steps=total_steps, refined_qpos=refined_qpos)
    command = _command_from_refined_qpos(motion, refined_qpos, rollout)
    _validate_command_shape(command, total_steps=total_steps)
    from spider.optimizers.receding import RecedingHorizonResult
    from spider.tasks.g1_wbc.spider_task import G1WbcMpcRun, G1WbcSpiderResult

    receding = RecedingHorizonResult(
        controls=controls.detach().clone(),
        infos=infos,
        executed_steps=total_steps,
    )
    result = G1WbcSpiderResult(
        command=command,
        rollout=rollout,
        refined_qpos=refined_qpos,
        controls=controls.detach().clone(),
        infos=infos,
        scores=torch.tensor(best_scores, dtype=torch.float32, device=device),
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
            "accepted": True,
            "used_baseline_fallback": False,
            "accepted_windows": accepted_windows,
            "num_windows": len(infos),
            "compile_init_wall_time_sec": compile_init_wall_time_sec,
            "steady_state_wall_time_sec": steady_state_wall_time_sec,
            "runtime_visible_devices": tuple(
                getattr(getattr(runtime, "status", None), "visible_devices", ())
            ),
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
    )


def _placeholder_rollout_scores(samples, reference, actor_params, model_bundle):
    del reference, actor_params, model_bundle
    return samples[..., 0, 0]


def _default_model_factory(**kwargs):
    from spider.tasks.g1_wbc.mjx_model import build_mjx_model_bundle

    return build_mjx_model_bundle(**kwargs)


def _validated_controls(value, *, horizon: int, device: torch.device) -> torch.Tensor:
    controls = _to_torch(value, device=device)
    expected = (int(horizon), QPOS_DIM - 1)
    if tuple(controls.shape) != expected:
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
    if int(execute_chunk.shape[0]) < int(execute_steps):
        raise ValueError(
            f"execute_chunk has {execute_chunk.shape[0]} steps, need {execute_steps}"
        )
    return execute_chunk


def _apply_execute_chunk_to_refined_qpos(
    refined_qpos: torch.Tensor,
    execute_chunk,
    *,
    start: int,
    execute_steps: int,
    device: torch.device,
) -> None:
    chunk = _validated_execute_chunk(
        execute_chunk,
        execute_steps=execute_steps,
        device=device,
    )
    end = int(start) + int(execute_steps)
    refined_qpos[int(start) : end, 1:] = chunk[:execute_steps]


def _validate_rollout_shape(rollout, *, total_steps: int, refined_qpos: torch.Tensor) -> None:
    expected_qpos = (int(total_steps) + 1, 1, QPOS_DIM)
    if tuple(rollout.qpos.shape) != expected_qpos:
        raise ValueError(
            f"Expected rollout qpos shape {expected_qpos}, got {tuple(rollout.qpos.shape)}"
        )
    if tuple(refined_qpos.shape) != (int(total_steps) + 1, QPOS_DIM):
        expected_refined = (int(total_steps) + 1, QPOS_DIM)
        raise ValueError(
            f"Expected refined_qpos shape {expected_refined}, got {tuple(refined_qpos.shape)}"
        )
    for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        value = getattr(rollout, name)
        if int(value.shape[0]) != int(total_steps) + 1 or int(value.shape[1]) != 1:
            raise ValueError(f"Rollout {name} has inconsistent shape {tuple(value.shape)}")


def _validate_command_shape(command: G1CommandBatch, *, total_steps: int) -> None:
    expected_frames = int(total_steps) + 1
    if tuple(command.qpos_trajectory.shape) != (expected_frames, 1, QPOS_DIM):
        raise ValueError(
            "Expected command qpos trajectory shape "
            f"{(expected_frames, 1, QPOS_DIM)}, got {tuple(command.qpos_trajectory.shape)}"
        )


def _to_torch(value, *, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32).detach().clone()
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _scalar_info(info: dict[str, Any], name: str) -> float:
    value = info.get(name, 0.0)
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(()).item())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _command_from_refined_qpos(
    motion: G1Motion,
    refined_qpos: torch.Tensor,
    rollout,
) -> G1CommandBatch:
    return G1CommandBatch(
        path=motion.path,
        motion_type=motion.motion_type,
        fps=motion.fps,
        joint_pos=refined_qpos[:, None, 7:].contiguous(),
        joint_vel=torch.zeros_like(refined_qpos[:, None, 7:]),
        body_pos_w=rollout.body_pos_w.detach().clone(),
        body_quat_w=rollout.body_quat_w.detach().clone(),
        body_lin_vel_w=rollout.body_lin_vel_w.detach().clone(),
        body_ang_vel_w=rollout.body_ang_vel_w.detach().clone(),
        qpos_trajectory=refined_qpos[:, None, :].contiguous(),
        qvel_trajectory=torch.zeros(
            refined_qpos.shape[0],
            1,
            rollout.qvel.shape[-1],
            dtype=refined_qpos.dtype,
            device=refined_qpos.device,
        ),
    )
