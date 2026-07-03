"""G1 WBC task adapter for SPIDER's generic receding-horizon optimizer."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Literal

import mujoco
import torch

from spider.config import Config
from spider.optimizers.receding import (
    RecedingHorizonResult,
    run_receding_horizon,
    run_sampling_receding_mpc,
    sampling_mpc_metadata,
)
from spider.optimizers.sampling import make_rollout_fn
from spider.tasks.g1_wbc.constants import (
    MUJOCO_JOINT_NAMES,
    POLICY_DT,
    QPOS_DIM,
    QVEL_DIM,
)
from spider.tasks.g1_wbc.math_utils import normalize, quat_from_axis_angle, quat_mul
from spider.tasks.g1_wbc.motion import G1CommandBatch, G1Motion
from spider.tasks.g1_wbc.policy import WbcActor
from spider.tasks.g1_wbc.result_types import G1WbcMpcRun, G1WbcSpiderResult
from spider.tasks.g1_wbc.rollout import (
    RolloutResult,
    WbcRolloutConfig,
    command_batch_from_qpos_trajectory,
    load_wbc_model,
)
from spider.simulators.g1_wbc import (
    G1WbcBackend,
    copy_sample_state,
    get_reward,
    get_terminal_reward,
    get_terminate,
    get_trace,
    load_env_params,
    load_state,
    save_env_params,
    save_state,
    setup_env,
    step_env,
)

G1WbcObjective = Literal["g1_wbc_ee", "g1_wbc_joint", "g1_wbc_joint_global"]

REWARD_WEIGHT_PRESETS: dict[G1WbcObjective, dict[str, float]] = {
    "g1_wbc_joint_global": {
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

_WBC_ROLLOUT_FN = make_rollout_fn(
    step_env,
    save_state,
    load_state,
    get_reward,
    get_terminal_reward,
    get_terminate,
    get_trace,
    save_env_params,
    load_env_params,
    copy_sample_state,
)


def load_reward_weights(path: str | Path, mode: G1WbcObjective) -> dict[str, float]:
    raw = json.loads(Path(path).expanduser().read_text())
    if mode in raw and isinstance(raw[mode], dict):
        raw = raw[mode]
    if not isinstance(raw, dict):
        raise ValueError(f"Reward weight file must contain a JSON object: {path}")
    return {str(key): float(value) for key, value in raw.items()}


def reward_weights_for(
    mode: G1WbcObjective,
    weights: dict[str, float] | None,
) -> dict[str, float]:
    return weights or REWARD_WEIGHT_PRESETS[mode]


def build_g1_wbc_sampling_config(
    *,
    device: str,
    num_samples: int,
    rollout_batch_size: int,
    max_num_iterations: int,
    horizon_steps: int,
    ctrl_steps: int,
    knot_count: int,
    temperature: float,
    control_update_mode: str,
    pos_noise_scale: float,
    rot_noise_scale: float,
    joint_noise_scale: float,
    first_ctrl_noise_scale: float,
    last_ctrl_noise_scale: float,
    final_noise_scale: float,
    use_torch_compile: bool,
    seed: int,
) -> Config:
    """Build the SPIDER sampling config for the G1 WBC task adapter."""

    horizon_steps = int(horizon_steps)
    ctrl_steps = int(ctrl_steps)
    knot_count = int(knot_count)
    if horizon_steps < 1 or ctrl_steps < 1:
        raise ValueError("horizon_steps and ctrl_steps must be positive.")
    if knot_count < 2:
        raise ValueError("knot_count must be at least 2.")
    if control_update_mode not in {"weighted_mean", "best"}:
        raise ValueError("control_update_mode must be one of: weighted_mean, best.")

    config = Config(
        robot_type="g1",
        embodiment_type="humanoid",
        simulator="g1_wbc",
        device=device,
        sim_dt=POLICY_DT,
        ref_dt=POLICY_DT,
        render_dt=POLICY_DT,
        horizon=horizon_steps * POLICY_DT,
        ctrl_dt=ctrl_steps * POLICY_DT,
        knot_dt=max(1, int(round(horizon_steps / max(knot_count - 1, 1))))
        * POLICY_DT,
        num_samples=int(num_samples),
        max_num_iterations=int(max_num_iterations),
        temperature=float(temperature),
        pos_noise_scale=float(pos_noise_scale),
        rot_noise_scale=float(rot_noise_scale),
        joint_noise_scale=float(joint_noise_scale),
        first_ctrl_noise_scale=float(first_ctrl_noise_scale),
        last_ctrl_noise_scale=float(last_ctrl_noise_scale),
        final_noise_scale=float(final_noise_scale),
        use_torch_compile=bool(use_torch_compile),
        seed=int(seed),
        show_viewer=False,
        save_video=False,
    )
    config.horizon_steps = horizon_steps
    config.ctrl_steps = ctrl_steps
    config.knot_steps = max(1, int(round(config.knot_dt / POLICY_DT)))
    config.ref_steps = 1
    config.nq = QPOS_DIM
    config.nv = QVEL_DIM
    config.nu = QPOS_DIM - 1
    config.env_params_list = [[{}] for _ in range(int(max_num_iterations))]
    config.num_knot_points = knot_count
    config.rollout_batch_size = int(rollout_batch_size)
    config.control_update_mode = str(control_update_mode)
    config.noise_scale = _g1_wbc_noise_scale(config, knot_count)
    config.beta_traj = (
        config.final_noise_scale ** (1 / config.max_num_iterations)
        if config.max_num_iterations > 0
        else 1.0
    )
    return config


def _g1_wbc_noise_scale(config: Config, knot_count: int) -> torch.Tensor:
    noise_profile = torch.logspace(
        start=torch.log10(torch.tensor(config.first_ctrl_noise_scale)),
        end=torch.log10(torch.tensor(config.last_ctrl_noise_scale)),
        steps=int(knot_count),
        device=config.device,
        base=10,
    )
    noise_scale = noise_profile[None, :, None].repeat(1, 1, config.nu)
    noise_scale[:, :, :3] *= config.pos_noise_scale
    noise_scale[:, :, 3:6] *= config.rot_noise_scale
    noise_scale[:, :, 6:] *= config.joint_noise_scale
    noise_scale = noise_scale.repeat(config.num_samples, 1, 1)
    noise_scale[0] *= 0.0
    num_exploit_samples = int(config.num_samples * config.exploit_ratio)
    noise_scale[-num_exploit_samples:] *= config.exploit_noise_scale
    return noise_scale


def run_g1_wbc_sampling_mpc(
    config: Config,
    task: "G1WbcSamplingTask",
    *,
    total_steps: int,
) -> G1WbcMpcRun:
    """Run G1 WBC through SPIDER's generic sampled receding-horizon MPC."""

    _synchronize_torch_device(config)
    steady_start = time.perf_counter()
    receding = run_sampling_receding_mpc(
        config,
        task,
        total_steps=int(total_steps),
    )
    _synchronize_torch_device(config)
    steady_state_wall_time_sec = time.perf_counter() - steady_start
    result = task.build_result(
        receding.controls,
        receding.infos,
        total_steps=int(total_steps),
    )
    return G1WbcMpcRun(
        receding=receding,
        result=result,
        metadata={
            **sampling_mpc_metadata(config),
            "steady_state_wall_time_sec": steady_state_wall_time_sec,
            "runtime_visible_devices": _runtime_visible_devices(),
            "runtime_gpu_name": _runtime_gpu_name(config),
        },
    )


def _synchronize_torch_device(config: Config) -> None:
    device = getattr(config, "device", None)
    if device is None:
        return
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch_device)


def _runtime_visible_devices() -> tuple[str, ...]:
    return tuple(
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    )


def _runtime_gpu_name(config: Config) -> str | None:
    device = getattr(config, "device", None)
    if device is None:
        return None
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(torch_device))


class G1WbcSamplingTask:
    """Task adapter consumed by SPIDER's generic sampling optimizer."""

    def __init__(
        self,
        motion: G1Motion,
        actor: WbcActor,
        rollout_config: WbcRolloutConfig,
        *,
        mode: G1WbcObjective,
        reward_weights: dict[str, float] | None = None,
        execute_rollout_config: WbcRolloutConfig | None = None,
    ) -> None:
        self.device = torch.device(rollout_config.device)
        self.motion = motion.to(self.device)
        self.actor = actor.to(self.device).eval()
        self.rollout_config = rollout_config
        self.execute_rollout_config = execute_rollout_config or rollout_config
        self.mode = mode
        self.reward_weights = reward_weights_for(mode, reward_weights)
        self.joint_low, self.joint_high = _joint_limits(rollout_config, self.device)
        self._last_scores = torch.empty(0, dtype=torch.float32, device=self.device)
        self.rollout_backend: G1WbcBackend | None = None
        execute_setup_config = _minimal_backend_config(
            self.execute_rollout_config,
            num_envs=1,
        )
        self.execute_backend = setup_env(
            execute_setup_config,
            self._backend_ref_data(self.execute_rollout_config),
        )
        self.replay_qvel_trajectory: torch.Tensor | None = None
        self._clear_execution_trace()

    def _clear_execution_trace(self) -> None:
        self._qpos_trace: list[torch.Tensor] = []
        self._qvel_trace: list[torch.Tensor] = []
        self._body_pos_trace: list[torch.Tensor] = []
        self._body_quat_trace: list[torch.Tensor] = []
        self._body_lin_vel_trace: list[torch.Tensor] = []
        self._body_ang_vel_trace: list[torch.Tensor] = []
        self._actions: list[torch.Tensor] = []
        self._controls: list[torch.Tensor] = []
        self._contact_indicator: list[torch.Tensor] = []
        self._contact_force: list[torch.Tensor] = []
        self._floor_contact_indicator: list[torch.Tensor] = []
        self._floor_contact_force: list[torch.Tensor] = []
        self._ref_indices: list[torch.Tensor] = []
        self._executed_controls: list[torch.Tensor] = []

    def reset_execution_state(self) -> None:
        """Reset the physical execute backend and accumulated result trace."""

        self.execute_backend.reset_physical_state(
            self.motion.qpos()[0],
            self.motion.qvel()[0],
            ref_index=0,
        )
        self._clear_execution_trace()

    def initial_controls(self, config: Config) -> torch.Tensor:
        return torch.zeros(
            int(config.horizon_steps),
            int(config.nu),
            dtype=torch.float32,
            device=self.device,
        )

    def tail_controls(self, _sim_step: int, steps: int) -> torch.Tensor:
        return torch.zeros(
            int(steps),
            QPOS_DIM - 1,
            dtype=torch.float32,
            device=self.device,
        )

    def ref_slice(self, start: int, horizon: int) -> tuple[torch.Tensor, ...]:
        base_qpos = _slice_qpos_padded(self.motion.qpos(), start, horizon)
        return (
            torch.arange(
                int(start),
                int(start) + int(horizon),
                dtype=torch.long,
                device=self.device,
            ),
            base_qpos,
        )

    def rollout(
        self,
        config: Config,
        _env,
        controls: torch.Tensor,
        ref_slice: tuple[torch.Tensor, ...],
        _env_param: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        start = int(ref_slice[0][0].detach().cpu().item())
        base_qpos = ref_slice[1].to(self.device)
        self._active_rollout_start = start
        backend = self._rollout_backend_for(config)
        task_context = self._task_context()
        horizon = int(controls.shape[1])
        qpos = self.controls_to_qpos(
            controls.permute(1, 0, 2),
            base_qpos[:horizon],
        )
        command = command_batch_from_qpos_trajectory(
            self._window_motion(start, horizon),
            qpos,
            _replace_rollout_config(
                self.rollout_config,
                int(config.num_samples),
                horizon,
            ),
            preserve_template_first=True,
        )
        env_param = {
            "command": command,
            "ref_start": start,
            "initial_last_action": self.execute_backend.last_action,
            "initial_history_state": (
                None
                if self.execute_backend.obs_builder is None
                else self.execute_backend.obs_builder.history_state_dict()
            ),
            "task_context": task_context,
        }
        out_controls, scores, terminate, info = _WBC_ROLLOUT_FN(
            config,
            backend,
            controls,
            ref_slice,
            env_param,
        )
        self._last_scores = task_context["last_scores"].detach().clone()
        return out_controls, scores, terminate, info

    def execute(self, controls: torch.Tensor, sim_step: int) -> dict[str, Any]:
        execute_steps = int(controls.shape[0])
        base_qpos = _slice_qpos_padded(self.motion.qpos(), sim_step, execute_steps)
        qpos = self.controls_to_qpos(controls[:, None, :], base_qpos)
        info = self.execute_qpos_command(qpos, sim_step)
        self._executed_controls.extend(t.detach().clone() for t in controls)
        return {
            **info,
            "executed_controls_rms": torch.stack(
                [t.square().mean().sqrt() for t in controls]
            ).mean().detach().cpu().numpy(),
        }

    def execute_qpos_command(self, qpos: torch.Tensor, sim_step: int) -> dict[str, Any]:
        """Execute a precomputed qpos command chunk through the MPC rollout path."""

        qpos = qpos.to(self.device, dtype=torch.float32)
        if qpos.ndim == 3:
            if qpos.shape[1] != 1:
                raise ValueError(f"Expected single-env command qpos, got {qpos.shape}.")
            qpos = qpos[:, 0]
        if qpos.ndim != 2 or qpos.shape[-1] != QPOS_DIM:
            raise ValueError(f"Expected command qpos shape (T, {QPOS_DIM}), got {qpos.shape}.")

        execute_steps = int(qpos.shape[0])
        command = command_batch_from_qpos_trajectory(
            self._window_motion(sim_step, execute_steps),
            qpos[:, None, :],
            self.execute_rollout_config,
            qvel_trajectory=(
                None
                if self.replay_qvel_trajectory is None
                else self.replay_qvel_trajectory[
                    sim_step : sim_step + execute_steps, None, :
                ]
            ),
            preserve_template_first=False,
            kinematics_batch_size=1,
        )
        self._execute_command_batch(command, sim_step)
        return {}

    def _execute_command_batch(
        self,
        command: G1CommandBatch,
        sim_step: int,
    ) -> RolloutResult:
        """Execute one command chunk and update the receding-horizon task state."""

        self.execute_backend.set_command(
            command,
            ref_start=sim_step,
            initial_last_action=self.execute_backend.last_action,
            initial_history_state=(
                None
                if self.execute_backend.obs_builder is None
                else self.execute_backend.obs_builder.history_state_dict()
            ),
        )
        for _ in range(int(command.num_frames)):
            self.execute_backend.step()
        rollout_result = self.execute_backend.rollout_result()
        self._append_rollout(rollout_result)
        return rollout_result

    def replay_qpos_command_sequence(
        self,
        qpos_trajectory: torch.Tensor,
        *,
        qvel_trajectory: torch.Tensor | None = None,
        control_steps: int,
        total_steps: int | None = None,
    ) -> RolloutResult:
        """Replay a saved command sequence using the same chunked execute path as MPC."""

        qpos_trajectory = qpos_trajectory.to(self.device, dtype=torch.float32)
        if qpos_trajectory.ndim == 3:
            if qpos_trajectory.shape[1] != 1:
                raise ValueError(
                    f"Expected single-env command qpos trajectory, got {qpos_trajectory.shape}."
                )
            qpos_trajectory = qpos_trajectory[:, 0]
        if qpos_trajectory.ndim != 2 or qpos_trajectory.shape[-1] != QPOS_DIM:
            raise ValueError(
                f"Expected command qpos trajectory shape (T, {QPOS_DIM}), "
                f"got {qpos_trajectory.shape}."
            )
        if qvel_trajectory is not None:
            qvel_trajectory = qvel_trajectory.to(self.device, dtype=torch.float32)
            if qvel_trajectory.ndim == 3:
                if qvel_trajectory.shape[1] != 1:
                    raise ValueError(
                        f"Expected single-env command qvel trajectory, got {qvel_trajectory.shape}."
                    )
                qvel_trajectory = qvel_trajectory[:, 0]
            if qvel_trajectory.ndim != 2 or qvel_trajectory.shape[-1] != QVEL_DIM:
                raise ValueError(
                    f"Expected command qvel trajectory shape (T, {QVEL_DIM}), "
                    f"got {qvel_trajectory.shape}."
                )
        if total_steps is None:
            total_steps = int(qpos_trajectory.shape[0]) - 1
        total_steps = min(int(total_steps), int(qpos_trajectory.shape[0]) - 1)
        if total_steps < 1:
            raise ValueError("Need at least one replay step.")
        control_steps = int(control_steps)
        if control_steps < 1:
            raise ValueError("control_steps must be positive.")
        control_steps = min(control_steps, total_steps)

        self.reset_execution_state()
        previous_replay_qvel = self.replay_qvel_trajectory
        self.replay_qvel_trajectory = qvel_trajectory
        try:
            replay_config = _minimal_backend_config(
                self.execute_rollout_config,
                num_envs=1,
            )
            replay_config.horizon_steps = int(control_steps)
            replay_config.ctrl_steps = int(control_steps)
            replay_config.max_num_iterations = 0
            run_receding_horizon(
                replay_config,
                self,
                qpos_trajectory[:control_steps],
                total_steps=total_steps,
                optimize=_identity_replay_optimize,
                get_ref_slice=lambda start, horizon: (
                    qpos_trajectory[start : start + horizon],
                ),
                execute_controls=(
                    lambda controls, sim_step: self.execute_qpos_command(
                        controls,
                        sim_step,
                    )
                ),
                make_tail_controls=(
                    lambda sim_step, steps: qpos_trajectory[
                        sim_step : sim_step + steps
                    ]
                ),
            )
            return self._stack_rollout()
        finally:
            self.replay_qvel_trajectory = previous_replay_qvel

    def controls_to_qpos(
        self,
        controls_time_major: torch.Tensor,
        base_qpos: torch.Tensor,
    ) -> torch.Tensor:
        return _controls_to_qpos(
            controls_time_major,
            base_qpos,
            self.joint_low,
            self.joint_high,
        )

    def build_result(
        self,
        controls: torch.Tensor,
        infos: list[dict[str, Any]],
        *,
        total_steps: int,
    ) -> G1WbcSpiderResult:
        rollout_result = self._stack_rollout()
        refined_qpos = self.motion.qpos()[: total_steps + 1].detach().clone()
        if self._executed_controls:
            executed = torch.stack(self._executed_controls, dim=0)
            base = self.motion.qpos()[: executed.shape[0]]
            refined_qpos[: executed.shape[0]] = self.controls_to_qpos(
                executed[:, None, :],
                base,
            )[:, 0]
        command = command_batch_from_qpos_trajectory(
            self.motion,
            refined_qpos[:, None, :],
            _replace_rollout_config(self.execute_rollout_config, 1, total_steps),
            preserve_template_first=False,
        )
        return G1WbcSpiderResult(
            command=command,
            rollout=rollout_result,
            refined_qpos=refined_qpos,
            controls=controls.detach().clone(),
            infos=infos,
            scores=self._last_scores.detach().clone(),
            num_windows=len(infos),
        )

    def _task_context(self) -> dict[str, Any]:
        return {
            "motion": self.motion,
            "reward_weights": self.reward_weights,
            "last_scores": self._last_scores,
        }

    def _rollout_backend_for(self, config: Config) -> G1WbcBackend:
        num_samples = int(config.num_samples)
        current_state = self.execute_backend.last_robot_state
        if current_state is None:
            current_qpos = self.motion.qpos()[0]
            current_qvel = self.motion.qvel()[0]
        else:
            current_qpos = current_state.qpos[0]
            current_qvel = current_state.qvel[0]
        ref_index = int(getattr(self, "_active_rollout_start", len(self._actions)))
        if self.rollout_backend is not None and self.rollout_backend.num_envs == num_samples:
            self.rollout_backend.reset_physical_state(
                current_qpos,
                current_qvel,
                ref_index=ref_index,
            )
            return self.rollout_backend
        rollout_setup_config = _minimal_backend_config(
            self.rollout_config,
            num_envs=num_samples,
        )
        self.rollout_backend = setup_env(
            rollout_setup_config,
            self._backend_ref_data(self.rollout_config),
        )
        self.rollout_backend.reset_physical_state(
            current_qpos,
            current_qvel,
            ref_index=ref_index,
        )
        return self.rollout_backend

    def _backend_ref_data(self, rollout_config: WbcRolloutConfig) -> dict[str, Any]:
        return {
            "motion": self.motion,
            "actor": self.actor,
            "rollout_config": rollout_config,
        }

    def _window_motion(self, start: int, length: int) -> G1Motion:
        return _slice_motion_padded(self.motion, start, length)

    def _append_rollout(self, rollout_result: RolloutResult) -> None:
        if not self._qpos_trace:
            self._qpos_trace.append(rollout_result.qpos[0, 0].detach().clone())
            self._qvel_trace.append(rollout_result.qvel[0, 0].detach().clone())
            self._body_pos_trace.append(rollout_result.body_pos_w[0, 0].detach().clone())
            self._body_quat_trace.append(rollout_result.body_quat_w[0, 0].detach().clone())
            self._body_lin_vel_trace.append(rollout_result.body_lin_vel_w[0, 0].detach().clone())
            self._body_ang_vel_trace.append(rollout_result.body_ang_vel_w[0, 0].detach().clone())
            self._contact_indicator.append(rollout_result.contact_indicator[0, 0].detach().clone())
            self._contact_force.append(rollout_result.contact_force[0, 0].detach().clone())
            self._floor_contact_indicator.append(
                _floor_contact_indicator(rollout_result)[0, 0].detach().clone()
            )
            self._floor_contact_force.append(
                _floor_contact_force(rollout_result)[0, 0].detach().clone()
            )
            self._ref_indices.append(rollout_result.ref_indices[0, 0].detach().clone())

        self._qpos_trace.extend(t.detach().clone() for t in rollout_result.qpos[1:, 0])
        self._qvel_trace.extend(t.detach().clone() for t in rollout_result.qvel[1:, 0])
        self._body_pos_trace.extend(t.detach().clone() for t in rollout_result.body_pos_w[1:, 0])
        self._body_quat_trace.extend(t.detach().clone() for t in rollout_result.body_quat_w[1:, 0])
        self._body_lin_vel_trace.extend(
            t.detach().clone() for t in rollout_result.body_lin_vel_w[1:, 0]
        )
        self._body_ang_vel_trace.extend(
            t.detach().clone() for t in rollout_result.body_ang_vel_w[1:, 0]
        )
        self._contact_indicator.extend(
            t.detach().clone() for t in rollout_result.contact_indicator[1:, 0]
        )
        self._contact_force.extend(t.detach().clone() for t in rollout_result.contact_force[1:, 0])
        self._floor_contact_indicator.extend(
            t.detach().clone() for t in _floor_contact_indicator(rollout_result)[1:, 0]
        )
        self._floor_contact_force.extend(
            t.detach().clone() for t in _floor_contact_force(rollout_result)[1:, 0]
        )
        self._ref_indices.extend(t.detach().clone() for t in rollout_result.ref_indices[1:, 0])
        self._actions.extend(t.detach().clone() for t in rollout_result.actions[:, 0])
        self._controls.extend(t.detach().clone() for t in rollout_result.controls[:, 0])

    def _stack_rollout(self) -> RolloutResult:
        return RolloutResult(
            qpos=torch.stack(self._qpos_trace, dim=0)[:, None, :],
            qvel=torch.stack(self._qvel_trace, dim=0)[:, None, :],
            body_pos_w=torch.stack(self._body_pos_trace, dim=0)[:, None, :, :],
            body_quat_w=torch.stack(self._body_quat_trace, dim=0)[:, None, :, :],
            body_lin_vel_w=torch.stack(self._body_lin_vel_trace, dim=0)[:, None, :, :],
            body_ang_vel_w=torch.stack(self._body_ang_vel_trace, dim=0)[:, None, :, :],
            actions=torch.stack(self._actions, dim=0)[:, None, :],
            controls=torch.stack(self._controls, dim=0)[:, None, :],
            contact_indicator=torch.stack(self._contact_indicator, dim=0)[:, None, :],
            contact_force=torch.stack(self._contact_force, dim=0)[:, None, :],
            floor_contact_indicator=torch.stack(self._floor_contact_indicator, dim=0)[:, None, :],
            floor_contact_force=torch.stack(self._floor_contact_force, dim=0)[:, None, :],
            ref_indices=torch.stack(self._ref_indices, dim=0)[:, None],
        )


def _replace_rollout_config(
    config: WbcRolloutConfig,
    num_envs: int,
    max_steps: int | None,
) -> WbcRolloutConfig:
    from dataclasses import replace

    return replace(config, num_envs=int(num_envs), max_steps=max_steps)


def _identity_replay_optimize(
    _config: Config,
    _env,
    controls: torch.Tensor,
    _ref_slice: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, dict[str, Any]]:
    return controls, {"opt_steps": torch.zeros(1), "improvement": torch.zeros(1)}


def _controls_to_qpos(
    controls_time_major: torch.Tensor,
    base_qpos: torch.Tensor,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
) -> torch.Tensor:
    base = base_qpos[:, None, :].expand(
        -1,
        int(controls_time_major.shape[1]),
        -1,
    ).clone()
    base[..., :3] = base[..., :3] + controls_time_major[..., :3]
    delta_quat = quat_from_axis_angle(controls_time_major[..., 3:6])
    base[..., 3:7] = normalize(quat_mul(delta_quat, base[..., 3:7]))
    base[..., 7:] = torch.clamp(
        base[..., 7:] + controls_time_major[..., 6:],
        joint_low,
        joint_high,
    )
    return base.contiguous()


def _minimal_backend_config(
    rollout_config: WbcRolloutConfig,
    *,
    num_envs: int,
) -> Config:
    return Config(
        simulator="g1_wbc",
        device=str(rollout_config.device),
        num_samples=int(num_envs),
        num_dyn=1,
        max_num_iterations=1,
        horizon_steps=max(1, int(rollout_config.max_steps or 1)),
        ctrl_steps=1,
        nq=QPOS_DIM,
        nv=QVEL_DIM,
        nu=QPOS_DIM - 1,
    )


def _slice_qpos_padded(qpos: torch.Tensor, start: int, length: int) -> torch.Tensor:
    start = int(start)
    length = int(length)
    out = qpos[start : start + length]
    if out.shape[0] < length:
        out = torch.cat([out, out[-1:].repeat(length - out.shape[0], 1)], dim=0)
    return out.contiguous()


def _slice_motion_padded(motion: G1Motion, start: int, length: int) -> G1Motion:
    start = int(start)
    length = int(length)

    def sl(value: torch.Tensor) -> torch.Tensor:
        out = value[start : start + length]
        if out.shape[0] < length:
            repeats = [length - out.shape[0]] + [1] * (out.ndim - 1)
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


def _floor_contact_indicator(rollout_result: RolloutResult) -> torch.Tensor:
    if rollout_result.floor_contact_indicator is not None:
        return rollout_result.floor_contact_indicator
    other = torch.zeros(
        *rollout_result.contact_indicator.shape[:-1],
        1,
        dtype=rollout_result.contact_indicator.dtype,
        device=rollout_result.contact_indicator.device,
    )
    return torch.cat([rollout_result.contact_indicator, other], dim=-1)


def _floor_contact_force(rollout_result: RolloutResult) -> torch.Tensor:
    if rollout_result.floor_contact_force is not None:
        return rollout_result.floor_contact_force
    other = torch.zeros(
        *rollout_result.contact_force.shape[:-1],
        1,
        dtype=rollout_result.contact_force.dtype,
        device=rollout_result.contact_force.device,
    )
    return torch.cat([rollout_result.contact_force, other], dim=-1)


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


__all__ = [
    "G1WbcObjective",
    "G1WbcMpcRun",
    "G1WbcSamplingTask",
    "G1WbcSpiderResult",
    "REWARD_WEIGHT_PRESETS",
    "build_g1_wbc_sampling_config",
    "load_reward_weights",
    "reward_weights_for",
    "run_g1_wbc_sampling_mpc",
]
