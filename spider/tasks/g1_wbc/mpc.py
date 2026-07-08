"""Batched policy-in-the-loop MPC for the G1 WBC task."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Literal

import mujoco
import torch
import torch.nn.functional as F

from spider.tasks.g1_wbc.constants import (
    MUJOCO_JOINT_NAMES,
    POLICY_DT,
    QPOS_DIM,
)
from spider.tasks.g1_wbc.math_utils import (
    axis_angle_from_quat,
    normalize,
    quat_from_axis_angle,
    quat_inv,
    quat_mul,
)
from spider.tasks.g1_wbc.metrics import compute_rollout_scores
from spider.tasks.g1_wbc.motion import G1CommandBatch, G1Motion
from spider.tasks.g1_wbc.policy import WbcActor
from spider.tasks.g1_wbc.result_types import (
    G1WbcExecutedCommandChunk,
    G1WbcWindowReplayState,
)
from spider.tasks.g1_wbc.rollout import (
    RolloutResult,
    WbcRolloutConfig,
    command_batch_from_qpos_trajectory,
    load_wbc_model,
    run_command_rollout,
)

MpcMode = Literal["g1_wbc_ee", "g1_wbc_joint", "g1_wbc_joint_global"]
MpcPreset = Literal["aggressive", "conservative", "explore", "rootrot", "wide"]
MpcSamplingMode = Literal["full", "knot"]
MpcWarmStartSource = Literal["best", "mean"]


REWARD_WEIGHT_PRESETS: dict[MpcMode, dict[str, float]] = {
    "g1_wbc_joint_global": {
        # Contact and smoothness first. Reference contact is intentionally weak
        # because it is estimated from motion, while bad floor contacts and
        # force/control smoothness are directly observed in simulation.
        "bad_floor_contact": 45.0,
        "bad_floor_force_excess": 10.0,
        "contact_switch": 12.0,
        "contact_force_delta": 2.5,
        "contact_false_positive": 1.5,
        "contact_false_negative": 0.4,
        "control_delta": 1.8,
        "action_delta": 0.6,
        "joint_acc": 0.006,
        "joint_jerk": 0.0012,
        "body_global_pos_error": 4.0,
        "body_global_rot_error": 0.8,
        "ee_global_pos_error": 1.5,
        "ee_global_rot_error": 0.3,
    },
    "g1_wbc_joint": {
        "bad_floor_contact": 35.0,
        "bad_floor_force_excess": 8.0,
        "contact_switch": 10.0,
        "contact_force_delta": 2.0,
        "contact_false_positive": 0.8,
        "contact_false_negative": 0.3,
        "control_delta": 1.6,
        "action_delta": 0.5,
        "joint_acc": 0.006,
        "joint_jerk": 0.0012,
        "body_local_pos_error": 26.0,
        "body_local_rot_error": 3.0,
        "joint_pos_error": 2.1,
        "ee_local_pos_error": 6.0,
        "ee_local_rot_error": 1.2,
        "body_global_pos_error": 0.8,
        "body_global_rot_error": 0.2,
        "ee_global_pos_error": 0.3,
    },
    "g1_wbc_ee": {
        "bad_floor_contact": 35.0,
        "bad_floor_force_excess": 8.0,
        "contact_switch": 10.0,
        "contact_force_delta": 2.0,
        "contact_false_positive": 0.5,
        "contact_false_negative": 0.2,
        "control_delta": 2.0,
        "action_delta": 0.6,
        "joint_acc": 0.006,
        "joint_jerk": 0.0015,
        "hand_global_pos_error": 35.0,
        "hand_global_rot_error": 3.0,
        "hand_local_pos_error": 8.0,
        "hand_local_rot_error": 1.5,
        "ee_global_pos_error": 2.0,
        "ee_global_rot_error": 0.4,
        "body_global_pos_error": 0.8,
        "body_local_pos_error": 0.8,
    },
}


@dataclass
class G1WbcMpcConfig:
    """Sampling MPC parameters for refined WBC command optimization."""

    mode: MpcMode = "g1_wbc_joint"
    num_samples: int = 64
    num_iterations: int = 4
    planning_horizon_steps: int = 80
    control_steps: int = 20
    sampling_mode: MpcSamplingMode = "full"
    knot_count: int = 4
    elite_frac: float = 0.125
    temperature: float = 0.7
    root_pos_sigma: float = 0.015
    root_rot_sigma: float = 0.035
    joint_sigma: float = 0.06
    min_root_pos_sigma: float = 0.002
    min_root_rot_sigma: float = 0.004
    min_joint_sigma: float = 0.008
    sigma_decay: float = 0.75
    smooth_passes: int = 2
    command_reg_weight: float = 0.02
    command_smooth_weight: float = 0.00005
    reward_weights: dict[str, float] | None = field(default=None)
    use_guided_candidate: bool = True
    guided_root_pos_gain: float = 0.5
    guided_root_rot_gain: float = 0.5
    guided_joint_gain: float = 0.4
    guided_root_pos_clip: float = 0.05
    guided_root_rot_clip: float = 0.12
    guided_joint_clip: float = 0.20
    acceptance_gate: bool = True
    seed: int | None = 0
    freeze_first_frame: bool = True
    use_warm_start: bool = False
    warm_start_source: MpcWarmStartSource = "best"
    warm_start_decay: float = 1.0


@dataclass
class MpcIterationInfo:
    iteration: int
    window_index: int
    window_start: int
    window_horizon: int
    window_execute_steps: int
    sampling_mode: str
    sample_parameter_steps: int
    best_score: float
    mean_score: float
    elite_score: float
    zero_delta_score: float
    best_index: int
    raw_best_score: float
    raw_zero_delta_score: float
    reg_best: float
    reg_zero_delta: float
    root_pos_delta_abs_max: float
    root_rot_delta_abs_max: float
    joint_delta_abs_max: float
    best_root_pos_delta_rms: float
    best_root_rot_delta_rms: float
    best_joint_delta_rms: float


@dataclass
class G1WbcMpcResult:
    command: G1CommandBatch
    rollout: RolloutResult
    refined_qpos: torch.Tensor
    scores: torch.Tensor
    history: list[MpcIterationInfo]
    accepted: bool = True
    used_baseline_fallback: bool = False
    final_candidate_score: float = 0.0
    final_baseline_score: float = 0.0
    num_windows: int = 0
    accepted_windows: int = 0
    executed_command_chunks: list[G1WbcExecutedCommandChunk] = field(default_factory=list)


@dataclass
class _MpcWindowOptimizeResult:
    best_qpos: torch.Tensor
    best_delta: torch.Tensor
    mean_delta: torch.Tensor
    scores: torch.Tensor
    history: list[MpcIterationInfo]
    best_is_template: bool
    best_score: float
    zero_delta_score: float


def mpc_config_from_preset(
    mode: MpcMode,
    preset: MpcPreset = "aggressive",
) -> G1WbcMpcConfig:
    """Return tuned MPC defaults for a motion class."""

    config = G1WbcMpcConfig(mode=mode)
    if preset == "aggressive":
        return config
    if preset == "conservative":
        return replace(
            config,
            num_iterations=3,
            root_pos_sigma=0.003,
            root_rot_sigma=0.008,
            joint_sigma=0.012,
            command_reg_weight=0.20,
            command_smooth_weight=0.0010,
            guided_root_pos_gain=0.25,
            guided_root_rot_gain=0.25,
            guided_joint_gain=0.20,
        )
    if preset == "explore":
        return replace(
            config,
            num_samples=128,
            num_iterations=5,
            root_pos_sigma=0.05,
            root_rot_sigma=0.10,
            joint_sigma=0.15,
            min_root_pos_sigma=0.01,
            min_root_rot_sigma=0.02,
            min_joint_sigma=0.03,
            sigma_decay=0.90,
            smooth_passes=0,
            command_reg_weight=0.0,
            command_smooth_weight=0.0,
            guided_root_pos_clip=0.15,
            guided_root_rot_clip=0.35,
            guided_joint_clip=0.45,
        )
    if preset == "rootrot":
        return replace(
            config,
            num_samples=128,
            num_iterations=5,
            root_pos_sigma=0.015,
            root_rot_sigma=0.12,
            joint_sigma=0.10,
            min_root_pos_sigma=0.002,
            min_root_rot_sigma=0.025,
            min_joint_sigma=0.02,
            sigma_decay=0.90,
            smooth_passes=0,
            command_reg_weight=0.0,
            command_smooth_weight=0.0,
            guided_root_pos_clip=0.05,
            guided_root_rot_clip=0.40,
            guided_joint_clip=0.35,
        )
    if preset == "wide":
        return replace(
            config,
            num_samples=512,
            num_iterations=4,
            root_pos_sigma=0.10,
            root_rot_sigma=0.24,
            joint_sigma=0.32,
            min_root_pos_sigma=0.025,
            min_root_rot_sigma=0.06,
            min_joint_sigma=0.08,
            sigma_decay=0.95,
            smooth_passes=0,
            command_reg_weight=0.0,
            command_smooth_weight=0.0,
            guided_root_pos_gain=0.75,
            guided_root_rot_gain=0.75,
            guided_joint_gain=0.0,
            guided_root_pos_clip=0.30,
            guided_root_rot_clip=0.75,
            guided_joint_clip=0.80,
            acceptance_gate=False,
        )
    raise ValueError(f"Unknown MPC preset: {preset}")


def load_reward_weights(path: str | Path, mode: MpcMode) -> dict[str, float]:
    """Load reward weights from JSON.

    Accepted forms:
    - a flat mapping from term name to scalar weight
    - a mapping keyed by method name, e.g. {"g1_wbc_ee": {...}}
    """

    raw = json.loads(Path(path).expanduser().read_text())
    if mode in raw and isinstance(raw[mode], dict):
        raw = raw[mode]
    if not isinstance(raw, dict):
        raise ValueError(f"Reward weight file must contain a JSON object: {path}")
    weights: dict[str, float] = {}
    for key, value in raw.items():
        if not isinstance(value, (int, float)):
            raise ValueError(f"Reward weight {key!r} must be numeric, got {value!r}.")
        weights[str(key)] = float(value)
    return weights


def optimize_mpc_command(
    motion: G1Motion,
    actor: WbcActor,
    rollout_config: WbcRolloutConfig,
    mpc_config: G1WbcMpcConfig,
    *,
    diagnostic_window_hook: Callable[[dict[str, Any]], None] | None = None,
) -> G1WbcMpcResult:
    """Optimize and execute a refined command with receding-horizon MPC."""

    if mpc_config.num_samples < 2:
        raise ValueError("MPC requires at least two samples.")
    if mpc_config.num_iterations < 1:
        raise ValueError("MPC requires at least one iteration.")
    if mpc_config.planning_horizon_steps < 1:
        raise ValueError("MPC planning_horizon_steps must be positive.")
    if mpc_config.control_steps < 1:
        raise ValueError("MPC control_steps must be positive.")
    if mpc_config.sampling_mode not in ("full", "knot"):
        raise ValueError(
            f"Unsupported MPC sampling_mode: {mpc_config.sampling_mode!r}."
        )
    if mpc_config.knot_count < 2:
        raise ValueError("MPC knot_count must be at least 2.")
    if mpc_config.warm_start_source not in ("best", "mean"):
        raise ValueError(
            f"Unsupported MPC warm_start_source: {mpc_config.warm_start_source!r}."
        )
    if not math.isfinite(float(mpc_config.warm_start_decay)):
        raise ValueError("MPC warm_start_decay must be finite.")

    device = torch.device(rollout_config.device)
    motion = motion.to(device)
    actor = actor.to(device).eval()
    total_steps = motion.num_frames - 1
    if rollout_config.max_steps is not None:
        total_steps = min(total_steps, int(rollout_config.max_steps))
    if total_steps < 1:
        raise ValueError("Need at least one MPC horizon step.")

    joint_low, joint_high = _joint_limits(rollout_config, device)

    if mpc_config.seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(mpc_config.seed))
    else:
        generator = None

    history: list[MpcIterationInfo] = []
    final_scores = torch.empty(0, dtype=torch.float32, device=device)
    refined_qpos = motion.qpos()[: total_steps + 1].detach().clone()

    qpos_trace: list[torch.Tensor] = []
    qvel_trace: list[torch.Tensor] = []
    body_pos_trace: list[torch.Tensor] = []
    body_quat_trace: list[torch.Tensor] = []
    body_lin_vel_trace: list[torch.Tensor] = []
    body_ang_vel_trace: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    controls: list[torch.Tensor] = []
    contact_indicator: list[torch.Tensor] = []
    contact_force: list[torch.Tensor] = []
    floor_contact_indicator: list[torch.Tensor] = []
    floor_contact_force: list[torch.Tensor] = []
    ref_indices: list[torch.Tensor] = []
    executed_command_chunks: list[G1WbcExecutedCommandChunk] = []

    current_qpos = motion.qpos()[0].detach().clone()
    current_qvel = motion.qvel()[0].detach().clone()
    current_last_action: torch.Tensor | None = None
    current_history_state = None
    sim_step = 0
    window_index = 0
    accepted_windows = 0
    warm_start_delta: torch.Tensor | None = None
    warm_start_execute_steps = 0

    while sim_step < total_steps:
        window_horizon = min(
            int(mpc_config.planning_horizon_steps),
            total_steps - sim_step,
        )
        execute_steps = min(int(mpc_config.control_steps), window_horizon)
        window_result = _optimize_mpc_window(
            motion,
            actor,
            rollout_config,
            mpc_config,
            start=sim_step,
            horizon=window_horizon,
            execute_steps=execute_steps,
            initial_qpos=current_qpos,
            initial_qvel=current_qvel,
            initial_last_action=current_last_action,
            initial_history_state=current_history_state,
            joint_low=joint_low,
            joint_high=joint_high,
            warm_start_delta=warm_start_delta if mpc_config.use_warm_start else None,
            warm_start_execute_steps=warm_start_execute_steps,
            generator=generator,
            window_index=window_index,
        )
        final_scores = window_result.scores.detach().clone()
        history.extend(window_result.history)

        final_config = replace(rollout_config, num_envs=1, max_steps=execute_steps)
        window_motion = _slice_motion(motion, sim_step, window_horizon)
        window_command = command_batch_from_qpos_trajectory(
            window_motion,
            window_result.best_qpos[:, None, :],
            final_config,
            preserve_template_first=window_result.best_is_template,
        )
        executed_command_chunks.append(
            G1WbcExecutedCommandChunk(
                start=sim_step,
                execute_steps=execute_steps,
                horizon_steps=window_horizon,
                command=window_command,
                replay_state=G1WbcWindowReplayState(
                    initial_qpos=current_qpos.detach().clone(),
                    initial_qvel=current_qvel.detach().clone(),
                    initial_last_action=_clone_diagnostic_value(current_last_action),
                    initial_history_state=_clone_diagnostic_value(
                        current_history_state
                    ),
                ),
            )
        )
        window_rollout = run_command_rollout(
            window_command,
            actor,
            final_config,
            initial_qpos=current_qpos,
            initial_qvel=current_qvel,
            initial_last_action=current_last_action,
            initial_history_state=current_history_state,
            ref_start=sim_step,
        )
        if diagnostic_window_hook is not None:
            diagnostic_window_hook(
                {
                    "window_index": int(window_index),
                    "start": int(sim_step),
                    "horizon": int(window_horizon),
                    "execute_steps": int(execute_steps),
                    "command": window_command,
                    "initial_qpos": current_qpos.detach().clone(),
                    "initial_qvel": current_qvel.detach().clone(),
                    "initial_last_action": _clone_diagnostic_value(
                        current_last_action
                    ),
                    "initial_history_state": _clone_diagnostic_value(
                        current_history_state
                    ),
                    "rollout": window_rollout,
                    "final_last_action": _clone_diagnostic_value(
                        window_rollout.final_last_action
                    ),
                    "final_history_state": _clone_diagnostic_value(
                        window_rollout.final_history_state
                    ),
                }
            )

        if sim_step == 0:
            qpos_trace.append(window_rollout.qpos[0, 0].detach().clone())
            qvel_trace.append(window_rollout.qvel[0, 0].detach().clone())
            body_pos_trace.append(window_rollout.body_pos_w[0, 0].detach().clone())
            body_quat_trace.append(window_rollout.body_quat_w[0, 0].detach().clone())
            body_lin_vel_trace.append(
                window_rollout.body_lin_vel_w[0, 0].detach().clone()
            )
            body_ang_vel_trace.append(
                window_rollout.body_ang_vel_w[0, 0].detach().clone()
            )
            contact_indicator.append(
                window_rollout.contact_indicator[0, 0].detach().clone()
            )
            contact_force.append(window_rollout.contact_force[0, 0].detach().clone())
            floor_contact_indicator.append(
                _floor_contact_indicator(window_rollout)[0, 0].detach().clone()
            )
            floor_contact_force.append(
                _floor_contact_force(window_rollout)[0, 0].detach().clone()
            )
            ref_indices.append(window_rollout.ref_indices[0, 0].detach().clone())

        qpos_trace.extend(t.detach().clone() for t in window_rollout.qpos[1:, 0])
        qvel_trace.extend(t.detach().clone() for t in window_rollout.qvel[1:, 0])
        body_pos_trace.extend(
            t.detach().clone() for t in window_rollout.body_pos_w[1:, 0]
        )
        body_quat_trace.extend(
            t.detach().clone() for t in window_rollout.body_quat_w[1:, 0]
        )
        body_lin_vel_trace.extend(
            t.detach().clone() for t in window_rollout.body_lin_vel_w[1:, 0]
        )
        body_ang_vel_trace.extend(
            t.detach().clone() for t in window_rollout.body_ang_vel_w[1:, 0]
        )
        contact_indicator.extend(
            t.detach().clone() for t in window_rollout.contact_indicator[1:, 0]
        )
        contact_force.extend(
            t.detach().clone() for t in window_rollout.contact_force[1:, 0]
        )
        floor_contact_indicator.extend(
            t.detach().clone() for t in _floor_contact_indicator(window_rollout)[1:, 0]
        )
        floor_contact_force.extend(
            t.detach().clone() for t in _floor_contact_force(window_rollout)[1:, 0]
        )
        ref_indices.extend(t.detach().clone() for t in window_rollout.ref_indices[1:, 0])
        actions.extend(t.detach().clone() for t in window_rollout.actions[:, 0])
        controls.extend(t.detach().clone() for t in window_rollout.controls[:, 0])

        end = sim_step + execute_steps
        refined_qpos[sim_step:end] = window_result.best_qpos[:execute_steps]
        if end <= total_steps:
            refined_qpos[end] = window_result.best_qpos[
                min(execute_steps, window_result.best_qpos.shape[0] - 1)
            ]
        current_qpos = window_rollout.qpos[-1, 0].detach().clone()
        current_qvel = window_rollout.qvel[-1, 0].detach().clone()
        current_last_action = (
            None
            if window_rollout.final_last_action is None
            else window_rollout.final_last_action[0].detach().clone()
        )
        current_history_state = window_rollout.final_history_state
        accepted_windows += int(window_result.best_score >= window_result.zero_delta_score)
        if mpc_config.use_warm_start:
            warm_start_delta = (
                window_result.mean_delta
                if mpc_config.warm_start_source == "mean"
                else window_result.best_delta
            ).detach().clone()
            warm_start_execute_steps = execute_steps
        sim_step = end
        window_index += 1

    rollout = RolloutResult(
        qpos=torch.stack(qpos_trace, dim=0)[:, None, :],
        qvel=torch.stack(qvel_trace, dim=0)[:, None, :],
        body_pos_w=torch.stack(body_pos_trace, dim=0)[:, None, :, :],
        body_quat_w=torch.stack(body_quat_trace, dim=0)[:, None, :, :],
        body_lin_vel_w=torch.stack(body_lin_vel_trace, dim=0)[:, None, :, :],
        body_ang_vel_w=torch.stack(body_ang_vel_trace, dim=0)[:, None, :, :],
        actions=torch.stack(actions, dim=0)[:, None, :],
        controls=torch.stack(controls, dim=0)[:, None, :],
        contact_indicator=torch.stack(contact_indicator, dim=0)[:, None, :],
        contact_force=torch.stack(contact_force, dim=0)[:, None, :],
        floor_contact_indicator=torch.stack(floor_contact_indicator, dim=0)[:, None, :],
        floor_contact_force=torch.stack(floor_contact_force, dim=0)[:, None, :],
        ref_indices=torch.stack(ref_indices, dim=0)[:, None],
    )
    final_score = _single_rollout_score(
        motion,
        rollout,
        mpc_config.mode,
        mpc_config.reward_weights,
    )

    base_qpos = motion.qpos()[: total_steps + 1].contiguous()
    baseline_config = replace(rollout_config, num_envs=1, max_steps=total_steps)
    baseline_command = command_batch_from_qpos_trajectory(
        motion,
        base_qpos[:, None, :],
        baseline_config,
        preserve_template_first=True,
    )
    baseline_rollout = run_command_rollout(
        baseline_command,
        actor,
        baseline_config,
        initial_qpos=motion.qpos()[0],
        initial_qvel=motion.qvel()[0],
    )
    baseline_score = _single_rollout_score(
        motion,
        baseline_rollout,
        mpc_config.mode,
        mpc_config.reward_weights,
    )
    accepted = final_score >= baseline_score
    used_baseline_fallback = bool(mpc_config.acceptance_gate and not accepted)
    if used_baseline_fallback:
        rollout = baseline_rollout
        final_command = baseline_command
        refined_qpos = base_qpos
        executed_command_chunks = [
            G1WbcExecutedCommandChunk(
                start=0,
                execute_steps=total_steps,
                horizon_steps=total_steps,
                command=baseline_command,
                replay_state=G1WbcWindowReplayState(
                    initial_qpos=motion.qpos()[0].detach().clone(),
                    initial_qvel=motion.qvel()[0].detach().clone(),
                ),
            )
        ]
    else:
        final_command = command_batch_from_qpos_trajectory(
            motion,
            refined_qpos[:, None, :],
            replace(rollout_config, num_envs=1, max_steps=None),
            preserve_template_first=False,
        )
    return G1WbcMpcResult(
        command=final_command,
        rollout=rollout,
        refined_qpos=refined_qpos,
        scores=final_scores,
        history=history,
        accepted=accepted,
        used_baseline_fallback=used_baseline_fallback,
        final_candidate_score=final_score,
        final_baseline_score=baseline_score,
        num_windows=window_index,
        accepted_windows=accepted_windows,
        executed_command_chunks=executed_command_chunks,
    )


def _clone_diagnostic_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_diagnostic_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_diagnostic_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_diagnostic_value(item) for item in value)
    return value


def _optimize_mpc_window(
    motion: G1Motion,
    actor: WbcActor,
    rollout_config: WbcRolloutConfig,
    mpc_config: G1WbcMpcConfig,
    *,
    start: int,
    horizon: int,
    execute_steps: int,
    initial_qpos: torch.Tensor,
    initial_qvel: torch.Tensor,
    initial_last_action: torch.Tensor | None,
    initial_history_state: dict | None,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
    warm_start_delta: torch.Tensor | None = None,
    warm_start_execute_steps: int = 0,
    generator: torch.Generator | None = None,
    window_index: int = 0,
) -> _MpcWindowOptimizeResult:
    device = torch.device(rollout_config.device)
    window_motion = _slice_motion(motion, start, horizon)
    base_qpos = window_motion.qpos()[:horizon].contiguous()
    sample_parameter_steps = _sample_parameter_steps(horizon, mpc_config)
    mean_delta = _warm_start_mean_delta(
        warm_start_delta,
        execute_steps=warm_start_execute_steps,
        horizon=horizon,
        config=mpc_config,
    )
    if mean_delta is None:
        mean_delta = torch.zeros(
            sample_parameter_steps, QPOS_DIM - 1, dtype=torch.float32, device=device
        )
    sigma = _initial_sigma(sample_parameter_steps, device, mpc_config)
    min_sigma = _min_sigma(sample_parameter_steps, device, mpc_config)
    guided_horizon_delta = _guided_delta_from_no_mpc(
        window_motion,
        actor,
        rollout_config,
        horizon,
        base_qpos,
        mpc_config,
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        initial_last_action=initial_last_action,
        initial_history_state=initial_history_state,
    )
    guided_delta = _delta_to_sampling_parameters(
        guided_horizon_delta,
        horizon,
        mpc_config,
    )

    best_score = torch.tensor(-float("inf"), dtype=torch.float32, device=device)
    best_qpos = base_qpos
    best_delta = torch.zeros(horizon, QPOS_DIM - 1, dtype=torch.float32, device=device)
    best_is_template = True
    final_scores = torch.empty(0, dtype=torch.float32, device=device)
    history: list[MpcIterationInfo] = []

    batch_config = replace(
        rollout_config,
        num_envs=mpc_config.num_samples,
        max_steps=horizon,
    )
    zero_delta_score = -float("inf")

    for iteration in range(mpc_config.num_iterations):
        candidates_param_delta = _sample_delta(
            mean_delta,
            sigma,
            mpc_config,
            generator,
            guided_delta=guided_delta,
        )
        candidates_delta = _expand_delta_to_horizon(
            candidates_param_delta,
            horizon,
            mpc_config,
        )
        candidates_qpos = _apply_delta_to_qpos(
            base_qpos,
            candidates_delta,
            joint_low=joint_low,
            joint_high=joint_high,
        )
        command = command_batch_from_qpos_trajectory(
            window_motion,
            candidates_qpos,
            batch_config,
            preserve_template_first=True,
        )
        rollout = run_command_rollout(
            command,
            actor,
            batch_config,
            initial_qpos=initial_qpos,
            initial_qvel=initial_qvel,
            initial_last_action=initial_last_action,
            initial_history_state=initial_history_state,
            ref_start=start,
        )
        _, terms = compute_rollout_scores(motion, rollout)
        raw_task_scores = _score_from_terms(
            terms,
            mpc_config.mode,
            mpc_config.reward_weights,
        )
        regularization = _command_regularization(
            candidates_delta,
            mpc_config.command_reg_weight,
            mpc_config.command_smooth_weight,
        )
        scores = raw_task_scores - regularization
        final_scores = scores.detach().clone()
        zero_delta_score = float(scores[0].detach().cpu().item())

        iteration_best = torch.argmax(scores)
        if scores[iteration_best] > best_score:
            best_score = scores[iteration_best].detach().clone()
            best_qpos = candidates_qpos[:, iteration_best].detach().clone()
            best_delta = candidates_delta[:, iteration_best].detach().clone()
            best_is_template = bool(best_delta.abs().max().detach().cpu().item() < 1.0e-8)

        elite_count = max(1, int(round(mpc_config.num_samples * mpc_config.elite_frac)))
        elite_scores, elite_indices = torch.topk(scores, k=elite_count, largest=True)
        elite_delta = candidates_param_delta[:, elite_indices]
        weights = torch.softmax(
            (elite_scores - elite_scores.mean())
            / max(float(mpc_config.temperature), 1.0e-6),
            dim=0,
        )
        mean_delta = (elite_delta * weights.view(1, -1, 1)).sum(dim=1)
        centered = elite_delta - mean_delta[:, None, :]
        elite_std = torch.sqrt(
            (centered.square() * weights.view(1, -1, 1)).sum(dim=1) + 1.0e-8
        )
        sigma = torch.maximum(
            min_sigma,
            (
                0.5 * sigma
                + 0.5 * elite_std
            )
            * float(mpc_config.sigma_decay),
        )
        if mpc_config.freeze_first_frame:
            mean_delta[0] = 0.0
            sigma[0] = 0.0

        history.append(
            MpcIterationInfo(
                iteration=iteration,
                window_index=window_index,
                window_start=start,
                window_horizon=horizon,
                window_execute_steps=execute_steps,
                sampling_mode=mpc_config.sampling_mode,
                sample_parameter_steps=sample_parameter_steps,
                best_score=float(scores.max().detach().cpu().item()),
                mean_score=float(scores.mean().detach().cpu().item()),
                elite_score=float(elite_scores.mean().detach().cpu().item()),
                zero_delta_score=float(scores[0].detach().cpu().item()),
                best_index=int(iteration_best.detach().cpu().item()),
                raw_best_score=float(raw_task_scores[iteration_best].detach().cpu().item()),
                raw_zero_delta_score=float(raw_task_scores[0].detach().cpu().item()),
                reg_best=float(regularization[iteration_best].detach().cpu().item()),
                reg_zero_delta=float(regularization[0].detach().cpu().item()),
                root_pos_delta_abs_max=float(
                    candidates_delta[..., :3].abs().max().detach().cpu().item()
                ),
                root_rot_delta_abs_max=float(
                    candidates_delta[..., 3:6].abs().max().detach().cpu().item()
                ),
                joint_delta_abs_max=float(
                    candidates_delta[..., 6:].abs().max().detach().cpu().item()
                ),
                best_root_pos_delta_rms=float(
                    best_delta[..., :3].square().mean().sqrt().detach().cpu().item()
                ),
                best_root_rot_delta_rms=float(
                    best_delta[..., 3:6].square().mean().sqrt().detach().cpu().item()
                ),
                best_joint_delta_rms=float(
                    best_delta[..., 6:].square().mean().sqrt().detach().cpu().item()
                ),
            )
        )
    mean_delta_horizon = _sampling_parameters_to_horizon_delta(
        mean_delta,
        horizon,
        mpc_config,
    )
    return _MpcWindowOptimizeResult(
        best_qpos=best_qpos,
        best_delta=best_delta.detach().clone(),
        mean_delta=mean_delta_horizon.detach().clone(),
        scores=final_scores,
        history=history,
        best_is_template=best_is_template,
        best_score=float(best_score.detach().cpu().item()),
        zero_delta_score=zero_delta_score,
    )


def _slice_motion(motion: G1Motion, start: int, length: int) -> G1Motion:
    end = min(motion.num_frames, int(start) + int(length))
    if end <= int(start):
        raise ValueError(f"Empty motion window start={start}, length={length}.")
    return G1Motion(
        path=motion.path,
        motion_type=motion.motion_type,
        fps=motion.fps,
        joint_pos=motion.joint_pos[start:end],
        joint_vel=motion.joint_vel[start:end],
        body_pos_w=motion.body_pos_w[start:end],
        body_quat_w=motion.body_quat_w[start:end],
        body_lin_vel_w=motion.body_lin_vel_w[start:end],
        body_ang_vel_w=motion.body_ang_vel_w[start:end],
        contact=motion.contact[start:end],
    )


def _initial_sigma(
    horizon: int,
    device: torch.device,
    config: G1WbcMpcConfig,
) -> torch.Tensor:
    sigma = torch.empty(horizon, QPOS_DIM - 1, dtype=torch.float32, device=device)
    sigma[:, :3] = float(config.root_pos_sigma)
    sigma[:, 3:6] = float(config.root_rot_sigma)
    sigma[:, 6:] = float(config.joint_sigma)
    if config.freeze_first_frame:
        sigma[0] = 0.0
    return sigma


def _min_sigma(
    horizon: int,
    device: torch.device,
    config: G1WbcMpcConfig,
) -> torch.Tensor:
    sigma = torch.empty(horizon, QPOS_DIM - 1, dtype=torch.float32, device=device)
    sigma[:, :3] = float(config.min_root_pos_sigma)
    sigma[:, 3:6] = float(config.min_root_rot_sigma)
    sigma[:, 6:] = float(config.min_joint_sigma)
    if config.freeze_first_frame:
        sigma[0] = 0.0
    return sigma


def _sample_parameter_steps(horizon: int, config: G1WbcMpcConfig) -> int:
    if config.sampling_mode == "full":
        return int(horizon)
    return max(2, min(int(config.knot_count), int(horizon)))


def _warm_start_mean_delta(
    previous_delta: torch.Tensor | None,
    *,
    execute_steps: int,
    horizon: int,
    config: G1WbcMpcConfig,
) -> torch.Tensor | None:
    if previous_delta is None:
        return None
    if previous_delta.ndim != 2 or previous_delta.shape[1] != QPOS_DIM - 1:
        raise ValueError(
            "MPC warm start delta must have shape "
            f"(steps, {QPOS_DIM - 1}), got {tuple(previous_delta.shape)}."
        )

    shifted = previous_delta[max(0, int(execute_steps)) :]
    warm_start = previous_delta.new_zeros((int(horizon), previous_delta.shape[1]))
    copy_steps = min(int(horizon), shifted.shape[0])
    if copy_steps > 0:
        warm_start[:copy_steps] = shifted[:copy_steps]
    warm_start = warm_start * float(config.warm_start_decay)
    if config.freeze_first_frame:
        warm_start[0] = 0.0
    return _delta_to_sampling_parameters(warm_start, horizon, config)


def _sample_delta(
    mean_delta: torch.Tensor,
    sigma: torch.Tensor,
    config: G1WbcMpcConfig,
    generator: torch.Generator | None,
    *,
    guided_delta: torch.Tensor | None,
) -> torch.Tensor:
    shape = (mean_delta.shape[0], config.num_samples, mean_delta.shape[1])
    noise = torch.randn(
        shape,
        dtype=mean_delta.dtype,
        device=mean_delta.device,
        generator=generator,
    )
    delta = mean_delta[:, None, :] + noise * sigma[:, None, :]
    delta[:, 0, :] = 0.0
    next_reserved = 1
    if (
        guided_delta is not None
        and config.num_samples > next_reserved
        and guided_delta.abs().max() > 1.0e-8
    ):
        delta[:, next_reserved, :] = guided_delta
        next_reserved += 1
    if config.num_samples > next_reserved and mean_delta.abs().max() > 1.0e-8:
        delta[:, next_reserved, :] = mean_delta
        next_reserved += 1
    if config.smooth_passes > 0:
        delta = _smooth_delta(delta, config.smooth_passes)
    if config.freeze_first_frame:
        delta[0] = 0.0
    return delta


def _delta_to_sampling_parameters(
    delta: torch.Tensor | None,
    horizon: int,
    config: G1WbcMpcConfig,
) -> torch.Tensor | None:
    if delta is None or config.sampling_mode == "full":
        return delta
    knot_count = _sample_parameter_steps(horizon, config)
    idx = _knot_indices(horizon, knot_count, delta.device)
    return delta.index_select(0, idx)


def _sampling_parameters_to_horizon_delta(
    delta: torch.Tensor,
    horizon: int,
    config: G1WbcMpcConfig,
) -> torch.Tensor:
    if delta.ndim != 2:
        raise ValueError(f"Expected 2D sampling delta, got {tuple(delta.shape)}.")
    if config.sampling_mode == "full" or delta.shape[0] == horizon:
        return delta
    expanded = _expand_delta_to_horizon(delta[:, None, :], horizon, config)
    return expanded[:, 0, :]


def _expand_delta_to_horizon(
    delta: torch.Tensor,
    horizon: int,
    config: G1WbcMpcConfig,
) -> torch.Tensor:
    if config.sampling_mode == "full" or delta.shape[0] == horizon:
        return delta
    if delta.shape[0] <= 1:
        return delta.expand(horizon, -1, -1)

    root_pos = _linear_interpolate_time_major(delta[..., :3], horizon)
    root_rot = _slerp_axis_angle_delta(delta[..., 3:6], horizon)
    joint = _linear_interpolate_time_major(delta[..., 6:], horizon)
    out = torch.cat([root_pos, root_rot, joint], dim=-1)
    if config.freeze_first_frame:
        out[0] = 0.0
    return out


def _linear_interpolate_time_major(value: torch.Tensor, target_steps: int) -> torch.Tensor:
    if value.shape[0] == target_steps:
        return value
    src = value.permute(1, 2, 0)
    dst = F.interpolate(src, size=int(target_steps), mode="linear", align_corners=True)
    return dst.permute(2, 0, 1).contiguous()


def _slerp_axis_angle_delta(delta_axis_angle: torch.Tensor, target_steps: int) -> torch.Tensor:
    if delta_axis_angle.shape[0] == target_steps:
        return delta_axis_angle

    knot_quat = quat_from_axis_angle(delta_axis_angle)
    knot_count = knot_quat.shape[0]
    if knot_count <= 1:
        return delta_axis_angle.expand(target_steps, -1, -1)

    positions = torch.linspace(
        0,
        knot_count - 1,
        target_steps,
        dtype=delta_axis_angle.dtype,
        device=delta_axis_angle.device,
    )
    left = torch.floor(positions).to(torch.long).clamp(max=knot_count - 1)
    right = (left + 1).clamp(max=knot_count - 1)
    blend = (positions - left.to(delta_axis_angle.dtype)).view(target_steps, 1, 1)
    q0 = knot_quat.index_select(0, left)
    q1 = knot_quat.index_select(0, right)
    q = _quat_slerp(q0, q1, blend)
    return axis_angle_from_quat(q)


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    q0 = normalize(q0)
    q1 = normalize(q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    linear = normalize((1.0 - blend) * q0 + blend * q1)
    s0 = torch.sin((1.0 - blend) * theta) / sin_theta.clamp(min=1.0e-8)
    s1 = torch.sin(blend * theta) / sin_theta.clamp(min=1.0e-8)
    spherical = s0 * q0 + s1 * q1
    return torch.where(sin_theta.abs() < 1.0e-6, linear, normalize(spherical))


def _knot_indices(horizon: int, knot_count: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(
        0,
        int(horizon) - 1,
        int(knot_count),
        device=device,
    ).round().to(torch.long)


def _guided_delta_from_no_mpc(
    motion: G1Motion,
    actor: WbcActor,
    rollout_config: WbcRolloutConfig,
    horizon: int,
    base_qpos: torch.Tensor,
    config: G1WbcMpcConfig,
    *,
    initial_qpos: torch.Tensor | None = None,
    initial_qvel: torch.Tensor | None = None,
    initial_last_action: torch.Tensor | None = None,
    initial_history_state: dict | None = None,
) -> torch.Tensor | None:
    if not config.use_guided_candidate or config.num_samples < 2:
        return None
    no_mpc_config = replace(rollout_config, num_envs=1, max_steps=horizon)
    no_mpc_rollout = run_command_rollout(
        motion,
        actor,
        no_mpc_config,
        initial_qpos=motion.qpos()[0] if initial_qpos is None else initial_qpos,
        initial_qvel=motion.qvel()[0] if initial_qvel is None else initial_qvel,
        initial_last_action=initial_last_action,
        initial_history_state=initial_history_state,
    )
    executed = no_mpc_rollout.qpos[1 : horizon + 1, 0]
    if executed.shape[0] != horizon:
        return None
    guided = torch.zeros(horizon, QPOS_DIM - 1, dtype=torch.float32, device=base_qpos.device)
    guided[:, :3] = (base_qpos[:, :3] - executed[:, :3]) * float(
        config.guided_root_pos_gain
    )
    quat_err = quat_mul(base_qpos[:, 3:7], quat_inv(executed[:, 3:7]))
    guided[:, 3:6] = axis_angle_from_quat(quat_err) * float(config.guided_root_rot_gain)
    joint_gain = float(config.guided_joint_gain)
    guided[:, 6:] = (base_qpos[:, 7:] - executed[:, 7:]) * float(
        joint_gain
    )
    guided[:, :3] = guided[:, :3].clamp(
        -float(config.guided_root_pos_clip), float(config.guided_root_pos_clip)
    )
    guided[:, 3:6] = guided[:, 3:6].clamp(
        -float(config.guided_root_rot_clip), float(config.guided_root_rot_clip)
    )
    guided[:, 6:] = guided[:, 6:].clamp(
        -float(config.guided_joint_clip), float(config.guided_joint_clip)
    )
    if config.smooth_passes > 0:
        guided = _smooth_delta(guided[:, None, :], config.smooth_passes)[:, 0]
    if config.freeze_first_frame:
        guided[0] = 0.0
    return guided


def _smooth_delta(delta: torch.Tensor, passes: int) -> torch.Tensor:
    out = delta
    for _ in range(int(passes)):
        if out.shape[0] <= 2:
            break
        smoothed = out.clone()
        smoothed[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
        out = smoothed
    return out


def _apply_delta_to_qpos(
    base_qpos: torch.Tensor,
    delta: torch.Tensor,
    *,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
) -> torch.Tensor:
    qpos = base_qpos[:, None, :].expand(-1, delta.shape[1], -1).clone()
    qpos[..., :3] = qpos[..., :3] + delta[..., :3]
    delta_quat = quat_from_axis_angle(delta[..., 3:6])
    qpos[..., 3:7] = normalize(quat_mul(delta_quat, qpos[..., 3:7]))
    qpos[..., 7:] = torch.clamp(qpos[..., 7:] + delta[..., 6:], joint_low, joint_high)
    return qpos.contiguous()


def _score_from_terms(
    terms: dict[str, torch.Tensor],
    mode: MpcMode,
    reward_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    weights = reward_weights or REWARD_WEIGHT_PRESETS[mode]
    missing = sorted(key for key in weights if key not in terms)
    if missing:
        raise KeyError(f"Reward weights reference missing terms: {missing}")
    loss = None
    for key, weight in weights.items():
        term = terms[key] * float(weight)
        loss = term if loss is None else loss + term
    if loss is None:
        first = next(iter(terms.values()))
        loss = torch.zeros_like(first)
    return -loss


def _single_rollout_score(
    motion: G1Motion,
    rollout: RolloutResult,
    mode: MpcMode,
    reward_weights: dict[str, float] | None = None,
) -> float:
    _, terms = compute_rollout_scores(motion, rollout)
    score = _score_from_terms(terms, mode, reward_weights)
    if score.numel() == 0:
        return -float("inf")
    return float(torch.nan_to_num(score.float()).max().detach().cpu().item())


def _floor_contact_indicator(rollout: RolloutResult) -> torch.Tensor:
    if rollout.floor_contact_indicator is not None:
        return rollout.floor_contact_indicator
    other = torch.zeros(
        *rollout.contact_indicator.shape[:-1],
        1,
        dtype=rollout.contact_indicator.dtype,
        device=rollout.contact_indicator.device,
    )
    return torch.cat([rollout.contact_indicator, other], dim=-1)


def _floor_contact_force(rollout: RolloutResult) -> torch.Tensor:
    if rollout.floor_contact_force is not None:
        return rollout.floor_contact_force
    other = torch.zeros(
        *rollout.contact_force.shape[:-1],
        1,
        dtype=rollout.contact_force.dtype,
        device=rollout.contact_force.device,
    )
    return torch.cat([rollout.contact_force, other], dim=-1)


def _command_regularization(
    delta: torch.Tensor,
    reg_weight: float,
    smooth_weight: float,
) -> torch.Tensor:
    reg = delta.square().mean(dim=(0, 2))
    if delta.shape[0] > 1:
        smooth = torch.diff(delta, dim=0).square().mean(dim=(0, 2)) / (POLICY_DT**2)
    else:
        smooth = torch.zeros(delta.shape[1], dtype=delta.dtype, device=delta.device)
    return float(reg_weight) * reg + float(smooth_weight) * smooth


def _joint_limits(
    rollout_config: WbcRolloutConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = load_wbc_model(rollout_config.model_path)
    low = []
    high = []
    for joint_name in MUJOCO_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{joint_name}"
            )
        if joint_id < 0:
            raise ValueError(f"G1 model is missing joint {joint_name}")
        if int(model.jnt_limited[joint_id]):
            low.append(float(model.jnt_range[joint_id, 0]))
            high.append(float(model.jnt_range[joint_id, 1]))
        else:
            low.append(-float("inf"))
            high.append(float("inf"))
    return (
        torch.tensor(low, dtype=torch.float32, device=device),
        torch.tensor(high, dtype=torch.float32, device=device),
    )
