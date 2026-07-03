"""Generic receding-horizon execution helper for SPIDER optimizers."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Callable, Any, Protocol

import numpy as np
import torch

from spider.config import Config


@dataclass
class RecedingHorizonResult:
    controls: torch.Tensor
    infos: list[dict[str, Any]]
    executed_steps: int


class SamplingMpcTask(Protocol):
    """Minimal task interface for SPIDER's sampled receding-horizon MPC."""

    def initial_controls(self, config: Config) -> torch.Tensor: ...

    def rollout(
        self,
        config: Config,
        env: Any,
        controls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
        env_param: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]: ...

    def ref_slice(self, start: int, horizon: int) -> tuple[torch.Tensor, ...]: ...

    def execute(self, controls: torch.Tensor, sim_step: int) -> dict[str, Any]: ...

    def tail_controls(self, sim_step: int, steps: int) -> torch.Tensor: ...


def run_sampling_receding_mpc(
    config: Config,
    task: SamplingMpcTask,
    *,
    total_steps: int,
    initial_controls: torch.Tensor | None = None,
) -> RecedingHorizonResult:
    """Run SPIDER's generic sampled-MPC optimizer on a task adapter."""

    from spider.optimizers.sampling import make_optimize_fn, make_optimize_once_fn

    optimize_once = make_optimize_once_fn(task.rollout)
    optimize = make_optimize_fn(optimize_once)
    controls = (
        task.initial_controls(config)
        if initial_controls is None
        else initial_controls
    )
    return run_receding_horizon(
        config,
        task,
        controls,
        total_steps=total_steps,
        optimize=optimize,
        get_ref_slice=task.ref_slice,
        execute_controls=task.execute,
        make_tail_controls=task.tail_controls,
    )


def sampling_mpc_metadata(config: Config) -> dict[str, Any]:
    """Return JSON-friendly metadata for SPIDER's generic sampled-MPC run."""

    return {
        "backend": "spider.optimizers.sampling",
        "receding_backend": "spider.optimizers.receding",
        "num_samples": int(config.num_samples),
        "rollout_batch_size": int(getattr(config, "rollout_batch_size", 0)),
        "num_iterations": int(config.max_num_iterations),
        "planning_horizon_steps": int(config.horizon_steps),
        "control_steps": int(config.ctrl_steps),
        "control_update_mode": str(getattr(config, "control_update_mode", "weighted_mean")),
        "sampling_mode": "knot",
        "knot_count": int(config.noise_scale.shape[1]),
        "temperature": float(config.temperature),
        "first_ctrl_noise_scale": float(config.first_ctrl_noise_scale),
        "last_ctrl_noise_scale": float(config.last_ctrl_noise_scale),
        "final_noise_scale": float(config.final_noise_scale),
        "use_warm_start": bool(getattr(config, "use_warm_start", True)),
        "beta_traj": float(config.beta_traj),
        "root_pos_sigma": float(config.pos_noise_scale),
        "root_rot_sigma": float(config.rot_noise_scale),
        "joint_sigma": float(config.joint_noise_scale),
        "exploit_ratio": float(config.exploit_ratio),
        "exploit_noise_scale": float(config.exploit_noise_scale),
        "use_torch_compile": bool(config.use_torch_compile),
    }


def run_receding_horizon(
    config: Config,
    env,
    controls: torch.Tensor,
    *,
    total_steps: int,
    optimize: Callable[[Config, Any, torch.Tensor, tuple[torch.Tensor, ...]], tuple[torch.Tensor, dict]],
    get_ref_slice: Callable[[int, int], tuple[torch.Tensor, ...]],
    execute_controls: Callable[[torch.Tensor, int], dict[str, Any]],
    make_tail_controls: Callable[[int, int], torch.Tensor],
) -> RecedingHorizonResult:
    """Run SPIDER's standard optimize-execute-shift MPC loop."""

    infos: list[dict[str, Any]] = []
    sim_step = 0
    total_steps = int(total_steps)
    while sim_step < total_steps:
        print(
            f"MPC_PROGRESS sim_step={sim_step} total_steps={total_steps}",
            file=sys.stderr,
            flush=True,
        )
        ref_slice = get_ref_slice(sim_step, int(config.horizon_steps))
        if int(config.max_num_iterations) > 0:
            controls, info = optimize(config, env, controls, ref_slice)
        else:
            info = {"opt_steps": np.array([0]), "improvement": np.array([0.0])}
        info["sim_step"] = np.array([sim_step])

        execute_steps = min(int(config.ctrl_steps), total_steps - sim_step)
        step_info = execute_controls(controls[:execute_steps], sim_step)
        info.update(step_info)
        infos.append(info)

        sim_step += execute_steps
        if bool(getattr(config, "use_warm_start", True)):
            prev_controls = controls[execute_steps:]
            tail_steps = int(config.horizon_steps) - int(prev_controls.shape[0])
            if tail_steps > 0:
                tail = make_tail_controls(sim_step, tail_steps)
                controls = torch.cat([prev_controls, tail], dim=0)
            else:
                controls = prev_controls[: int(config.horizon_steps)]
        else:
            controls = make_tail_controls(sim_step, int(config.horizon_steps))

    return RecedingHorizonResult(
        controls=controls,
        infos=infos,
        executed_steps=sim_step,
    )
