"""CLI for evaluating G1 WBC policy rollouts on a single motion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spider.config import Config
from spider.tasks.g1_wbc.constants import QPOS_DIM, QVEL_DIM
from spider.tasks.g1_wbc.metrics import compute_rollout_metrics
from spider.tasks.g1_wbc.mpc import (
    G1WbcMpcConfig,
    mpc_config_from_preset,
    optimize_mpc_command,
)
from spider.tasks.g1_wbc.spider_task import (
    G1WbcSamplingTask,
    build_g1_wbc_sampling_config,
    load_reward_weights,
    reward_weights_for,
    run_g1_wbc_sampling_mpc,
)
from spider.tasks.g1_wbc.motion import (
    load_motion,
    validate_motion_dims,
)
from spider.tasks.g1_wbc.policy import load_wbc_actor, resolve_checkpoint_path
from spider.tasks.g1_wbc.rollout import (
    RolloutResult,
    WbcRolloutConfig,
    run_no_mpc_rollout,
    run_static_qpos_rollout,
)

MPC_METHODS = ("g1_wbc_ee", "g1_wbc_joint", "g1_wbc_joint_global")
LEGACY_ONLY_MPC_FLAGS = {
    "mpc_sampling_mode": "--mpc-sampling-mode",
    "mpc_elite_frac": "--mpc-elite-frac",
    "mpc_sigma_decay": "--mpc-sigma-decay",
    "mpc_smooth_passes": "--mpc-smooth-passes",
    "mpc_warm_start_source": "--mpc-warm-start-source",
    "mpc_warm_start_decay": "--mpc-warm-start-decay",
    "mpc_command_reg_weight": "--mpc-command-reg-weight",
    "mpc_command_smooth_weight": "--mpc-command-smooth-weight",
    "mpc_guided_candidate": "--mpc-guided-candidate/--no-mpc-guided-candidate",
    "mpc_acceptance_gate": "--mpc-acceptance-gate/--no-mpc-acceptance-gate",
    "mpc_guided_root_pos_gain": "--mpc-guided-root-pos-gain",
    "mpc_guided_root_rot_gain": "--mpc-guided-root-rot-gain",
    "mpc_guided_joint_gain": "--mpc-guided-joint-gain",
    "mpc_guided_root_pos_clip": "--mpc-guided-root-pos-clip",
    "mpc_guided_root_rot_clip": "--mpc-guided-root-rot-clip",
    "mpc_guided_joint_clip": "--mpc-guided-joint-clip",
}


def main() -> None:
    args = _parse_args()
    _validate_backend_args(args)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is not available.")

    motion = load_motion(args.motion, motion_type=args.motion_type, device=device)
    validate_motion_dims(motion)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    actor = None
    if args.method != "static_qpos":
        actor = load_wbc_actor(args.checkpoint, device=device)

    config = WbcRolloutConfig(
        model_path=args.model_path,
        device=device,
        num_envs=args.num_envs if args.method == "no_mpc" else 1,
        max_steps=args.max_steps,
        ref_offset=args.ref_offset,
        nconmax_per_env=args.nconmax_per_env,
        njmax_per_env=args.njmax_per_env,
        sync_after_step=args.sync_after_step,
        forward_after_step=args.forward_after_step,
        use_cuda_graph=args.use_cuda_graph,
        ls_parallel=args.ls_parallel,
        serial_warp_launches=args.serial_warp_launches,
    )
    execute_config = replace(
        config,
        use_cuda_graph=False,
        serial_warp_launches=True,
    ) if args.serial_execute_warp_launches else config
    mpc_payload = None
    mpc_result = None
    if args.method == "static_qpos":
        qpos_trajectory = _load_saved_qpos(args.saved_qpos, device=device)
        rollout = run_static_qpos_rollout(
            qpos_trajectory,
            config,
            max_steps=args.max_steps,
        )
    elif args.method == "replay_command":
        assert actor is not None
        if args.saved_command is None:
            raise ValueError("--method replay_command requires --saved-command.")
        saved_command_path = Path(args.saved_command).expanduser().resolve()
        qpos_trajectory, qvel_trajectory = _load_saved_command_trajectory(
            args.saved_command,
            device=device,
        )
        total_steps = qpos_trajectory.shape[0] - 1
        if args.max_steps is not None:
            total_steps = min(total_steps, int(args.max_steps))
        control_steps = _resolve_replay_control_steps(args)
        task = G1WbcSamplingTask(
            motion,
            actor,
            config,
            mode=args.replay_task_mode,
            execute_rollout_config=execute_config,
        )
        _synchronize_rollout_device(config)
        steady_start = time.perf_counter()
        rollout = task.replay_qpos_command_sequence(
            qpos_trajectory,
            qvel_trajectory=(
                qvel_trajectory if args.replay_use_saved_qvel else None
            ),
            control_steps=control_steps,
            total_steps=total_steps,
        )
        _synchronize_rollout_device(config)
        steady_state_wall_time_sec = time.perf_counter() - steady_start
        mpc_payload = {
            "backend": (
                "spider.tasks.g1_wbc.spider_task."
                "G1WbcSamplingTask.replay_qpos_command_sequence"
            ),
            "saved_command": str(saved_command_path),
            "saved_command_sha256": _file_sha256(saved_command_path),
            "replay_mode": "shared_execute_backend",
            "control_steps": control_steps,
            "use_saved_qvel": bool(args.replay_use_saved_qvel),
            "serial_execute_warp_launches": bool(args.serial_execute_warp_launches),
            "steady_state_wall_time_sec": steady_state_wall_time_sec,
            "num_command_frames": int(qpos_trajectory.shape[0]),
            "num_replay_steps": total_steps,
            "runtime_visible_devices": _runtime_visible_devices(),
            "runtime_gpu_name": _runtime_gpu_name(config),
        }
    elif args.method == "no_mpc":
        assert actor is not None
        rollout = run_no_mpc_rollout(motion, actor, config)
    else:
        assert actor is not None
        reward_weights = _load_method_reward_weights(args)
        total_steps = motion.num_frames - 1
        if args.max_steps is not None:
            total_steps = min(total_steps, int(args.max_steps))
        if args.mpc_backend == "mjx":
            spider_config = _build_sampling_config(args)
            from spider.tasks.g1_wbc.mjx_backend import run_g1_wbc_mjx_mpc

            effective_reward_weights = _effective_reward_weights(args.method, reward_weights)
            mpc_run = run_g1_wbc_mjx_mpc(
                spider_config=spider_config,
                motion=motion,
                actor=actor,
                rollout_config=config,
                execute_rollout_config=execute_config,
                method=args.method,
                reward_weights=effective_reward_weights,
                total_steps=total_steps,
                seed=int(args.seed),
                enable_physics_scan=_mjx_physics_scan_enabled(args),
            )
        elif args.mpc_optimizer == "legacy":
            legacy_config = _build_mpc_config(args)
            _synchronize_rollout_device(config)
            steady_start = time.perf_counter()
            legacy_result = optimize_mpc_command(
                motion,
                actor,
                config,
                legacy_config,
            )
            _synchronize_rollout_device(config)
            steady_state_wall_time_sec = time.perf_counter() - steady_start
            rollout = legacy_result.rollout
            mpc_result = legacy_result
            mpc_payload = {
                "backend": "spider.tasks.g1_wbc.mpc.optimize_mpc_command",
                "mpc_backend": args.mpc_backend,
                "mpc_optimizer": "legacy",
                "mpc_preset": args.mpc_preset,
                "reward_weight_source": (
                    str(Path(args.mpc_reward_weights).expanduser().resolve())
                    if args.mpc_reward_weights is not None
                    else "default"
                ),
                "reward_weights": _effective_reward_weights(
                    args.method,
                    legacy_config.reward_weights,
                ),
                "config": _legacy_mpc_config_payload(legacy_config),
                "history": [asdict(info) for info in legacy_result.history],
                "final_scores_mean": _safe_tensor_stat(legacy_result.scores, "mean"),
                "final_scores_max": _safe_tensor_stat(legacy_result.scores, "max"),
                "num_windows": int(legacy_result.num_windows),
                "accepted": bool(legacy_result.accepted),
                "accepted_windows": int(legacy_result.accepted_windows),
                "used_baseline_fallback": bool(legacy_result.used_baseline_fallback),
                "final_candidate_score": float(legacy_result.final_candidate_score),
                "final_baseline_score": float(legacy_result.final_baseline_score),
                "steady_state_wall_time_sec": steady_state_wall_time_sec,
                "runtime_visible_devices": _runtime_visible_devices(),
                "runtime_gpu_name": _runtime_gpu_name(config),
                "serial_execute_warp_launches": bool(args.serial_execute_warp_launches),
            }
        else:
            spider_config = _build_sampling_config(args)
            task = G1WbcSamplingTask(
                motion,
                actor,
                config,
                mode=args.method,
                reward_weights=reward_weights,
                execute_rollout_config=execute_config,
            )
            mpc_run = run_g1_wbc_sampling_mpc(
                spider_config,
                task,
                total_steps=total_steps,
            )
        if mpc_payload is None:
            receding_result = mpc_run.receding
            mpc_result = mpc_run.result
            rollout = mpc_result.rollout
            mpc_payload = {
                **mpc_run.metadata,
                "mpc_backend": args.mpc_backend,
                "mpc_optimizer": args.mpc_optimizer,
                "reward_weight_source": (
                    str(Path(args.mpc_reward_weights).expanduser().resolve())
                    if args.mpc_reward_weights is not None
                    else "default"
                ),
                "reward_weights": _effective_reward_weights(args.method, reward_weights),
                "history": _jsonable_infos(receding_result.infos),
                "final_scores_mean": _safe_tensor_stat(mpc_result.scores, "mean"),
                "final_scores_max": _safe_tensor_stat(mpc_result.scores, "max"),
                "num_windows": mpc_result.num_windows,
                "accepted": bool(
                    _required_mpc_metadata(mpc_run.metadata, "accepted")
                ),
                "accepted_windows": int(
                    _required_mpc_metadata(mpc_run.metadata, "accepted_windows")
                ),
                "used_baseline_fallback": bool(
                    _required_mpc_metadata(
                        mpc_run.metadata,
                        "used_baseline_fallback",
                    )
                ),
                "serial_execute_warp_launches": bool(args.serial_execute_warp_launches),
            }
    metrics = compute_rollout_metrics(motion, rollout)

    payload = {
        "method": args.method,
        "motion": str(Path(args.motion).expanduser().resolve()),
        "motion_type": motion.motion_type,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "device": device,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "ref_offset": args.ref_offset,
        "serial_warp_launches": bool(args.serial_warp_launches),
        "metrics": metrics,
    }
    if mpc_payload is not None:
        payload["mpc"] = mpc_payload
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        if args.save_rollout:
            _save_rollout(output_dir / "rollout.npz", rollout)
            if mpc_result is not None:
                _save_mpc_result(output_dir / "mpc_command.npz", mpc_result)
                _save_tracking_bfm_motion(
                    output_dir / "mpc_motion.npz",
                    mpc_result.command,
                )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", required=True, help="Reference motion npz path.")
    parser.add_argument(
        "--motion-type",
        default="auto",
        choices=("auto", "mujoco", "isaaclab"),
        help="Input npz semantic ordering.",
    )
    parser.add_argument(
        "--checkpoint",
        default="bc",
        help="WXY checkpoint alias ('bc'/'bcrl'), checkpoint directory, or .pt file.",
    )
    parser.add_argument(
        "--method",
        default="no_mpc",
        choices=(
            "no_mpc",
            "g1_wbc_ee",
            "g1_wbc_joint",
            "g1_wbc_joint_global",
            "replay_command",
            "static_qpos",
        ),
        help="Evaluation method to run.",
    )
    parser.add_argument(
        "--model-path",
        default=str(WbcRolloutConfig.model_path),
        help="MuJoCo XML path for G1 WBC simulation.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch/Warp device.")
    parser.add_argument("--num-envs", type=int, default=1, help="Batched worlds.")
    parser.add_argument("--max-steps", type=int, default=None, help="Policy steps.")
    parser.add_argument(
        "--ref-offset",
        type=int,
        default=0,
        help="Reference frame offset used when constructing policy command.",
    )
    parser.add_argument(
        "--nconmax-per-env",
        type=int,
        default=WbcRolloutConfig.nconmax_per_env,
        help="Per-world MuJoCo contact buffer size.",
    )
    parser.add_argument(
        "--njmax-per-env",
        type=int,
        default=WbcRolloutConfig.njmax_per_env,
        help="Per-world MuJoCo Jacobian buffer size.",
    )
    parser.add_argument(
        "--sync-after-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize Warp before reading tensors.",
    )
    parser.add_argument(
        "--forward-after-step",
        action=argparse.BooleanOptionalAction,
        default=WbcRolloutConfig.forward_after_step,
        help="Run mj_forward after each policy step before reading derived tensors.",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=WbcRolloutConfig.use_cuda_graph,
        help="Capture MuJoCo Warp step/forward/reset in CUDA graphs when available.",
    )
    parser.add_argument(
        "--ls-parallel",
        action=argparse.BooleanOptionalAction,
        default=WbcRolloutConfig.ls_parallel,
        help="Use MuJoCo Warp parallel line search in the Newton solver.",
    )
    parser.add_argument(
        "--serial-warp-launches",
        action="store_true",
        default=WbcRolloutConfig.serial_warp_launches,
        help=(
            "Diagnostic mode: serialize all regular Warp launches for every rollout. "
            "This is very slow if used during MPC sampling."
        ),
    )
    parser.add_argument(
        "--serial-execute-warp-launches",
        action="store_true",
        help=(
            "Diagnostic mode: serialize regular Warp launches only for MPC execute "
            "and chunked replay physical rollouts."
        ),
    )
    parser.add_argument("--output-dir", default=None, help="Optional result directory.")
    parser.add_argument(
        "--saved-qpos",
        default=None,
        help=(
            "Precomputed qpos trajectory for --method static_qpos. "
            "Accepts .npz keys qpos/refined_qpos/command_qpos_trajectory."
        ),
    )
    parser.add_argument(
        "--saved-command",
        default=None,
        help=(
            "Precomputed command NPZ for --method replay_command. "
            "Accepts .npz keys command_qpos_trajectory/refined_qpos/qpos."
        ),
    )
    parser.add_argument(
        "--replay-control-steps",
        type=int,
        default=None,
        help=(
            "Execute chunk size for --method replay_command. Defaults to the saved "
            "MPC control_steps from metrics.json when available."
        ),
    )
    parser.add_argument(
        "--replay-task-mode",
        default="g1_wbc_joint",
        choices=("g1_wbc_ee", "g1_wbc_joint", "g1_wbc_joint_global"),
        help="Task mode used to instantiate the shared MPC execute adapter.",
    )
    parser.add_argument(
        "--replay-use-saved-qvel",
        action="store_true",
        help=(
            "Diagnostic mode: use command_qvel_trajectory from the saved NPZ. "
            "Default recomputes chunk-local qvel to match MPC execute."
        ),
    )
    parser.add_argument(
        "--save-rollout",
        action="store_true",
        help="Save rollout tensors to rollout.npz when output-dir is set.",
    )
    parser.add_argument(
        "--mpc-backend",
        choices=("mujoco_warp", "mjx"),
        default="mujoco_warp",
        help="MPC rollout backend. MuJoCo-Warp remains the default.",
    )
    parser.add_argument(
        "--mjx-enable-scan",
        action="store_true",
        help=(
            "Compatibility flag for MJX/JAX physics scan. The scan is enabled "
            "whenever --mpc-backend mjx is selected."
        ),
    )
    parser.add_argument(
        "--mjx-guided-candidate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable MJX-generated guided candidate controls for the generic MJX backend.",
    )
    parser.add_argument(
        "--mpc-optimizer",
        choices=("generic", "legacy"),
        default="generic",
        help=(
            "MuJoCo-Warp MPC optimizer implementation. The default generic "
            "path is unchanged; Stage0 sweetpoint baselines use legacy."
        ),
    )
    parser.add_argument("--mpc-samples", type=int, default=None)
    parser.add_argument("--mpc-rollout-batch-size", type=int, default=0)
    parser.add_argument("--mpc-iterations", type=int, default=None)
    parser.add_argument("--mpc-planning-horizon-steps", type=int, default=None)
    parser.add_argument("--mpc-control-steps", type=int, default=None)
    parser.add_argument("--mpc-knot-count", type=int, default=None)
    parser.add_argument(
        "--mpc-preset",
        default="aggressive",
        choices=("aggressive", "conservative", "explore", "rootrot", "wide"),
        help="Legacy MPC preset accepted for compatibility with existing runners.",
    )
    parser.add_argument(
        "--mpc-sampling-mode",
        choices=("full", "knot"),
        default=None,
        help="Legacy MPC sampling mode accepted for compatibility with existing runners.",
    )
    parser.add_argument("--mpc-elite-frac", type=float, default=None)
    parser.add_argument("--mpc-temperature", type=float, default=None)
    parser.add_argument(
        "--mpc-control-update-mode",
        choices=("weighted_mean", "best"),
        default="weighted_mean",
        help="Generic sampled-MPC control update rule.",
    )
    parser.add_argument("--mpc-first-ctrl-noise-scale", type=float, default=None)
    parser.add_argument("--mpc-last-ctrl-noise-scale", type=float, default=None)
    parser.add_argument("--mpc-final-noise-scale", type=float, default=None)
    parser.add_argument(
        "--mpc-torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch.compile in SPIDER's generic sampling optimizer.",
    )
    parser.add_argument("--mpc-root-pos-sigma", type=float, default=None)
    parser.add_argument("--mpc-root-rot-sigma", type=float, default=None)
    parser.add_argument("--mpc-joint-sigma", type=float, default=None)
    parser.add_argument("--mpc-sigma-decay", type=float, default=None)
    parser.add_argument("--mpc-smooth-passes", type=int, default=None)
    parser.add_argument(
        "--mpc-warm-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Legacy warm-start flag accepted for compatibility with existing runners.",
    )
    parser.add_argument(
        "--mpc-warm-start-source",
        choices=("best", "mean"),
        default=None,
        help="Legacy warm-start source accepted for compatibility with existing runners.",
    )
    parser.add_argument(
        "--mpc-warm-start-decay",
        type=float,
        default=None,
        help="Legacy warm-start decay accepted for compatibility with existing runners.",
    )
    parser.add_argument("--mpc-command-reg-weight", type=float, default=None)
    parser.add_argument("--mpc-command-smooth-weight", type=float, default=None)
    parser.add_argument(
        "--mpc-guided-candidate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Legacy guided-candidate flag accepted for compatibility with existing runners.",
    )
    parser.add_argument(
        "--mpc-acceptance-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Legacy acceptance-gate flag accepted for compatibility with existing runners.",
    )
    parser.add_argument("--mpc-guided-root-pos-gain", type=float, default=None)
    parser.add_argument("--mpc-guided-root-rot-gain", type=float, default=None)
    parser.add_argument("--mpc-guided-joint-gain", type=float, default=None)
    parser.add_argument("--mpc-guided-root-pos-clip", type=float, default=None)
    parser.add_argument("--mpc-guided-root-rot-clip", type=float, default=None)
    parser.add_argument("--mpc-guided-joint-clip", type=float, default=None)
    parser.add_argument(
        "--mpc-reward-weights",
        default=None,
        help=(
            "Optional JSON reward weights. Accepts a flat term mapping or a mapping "
            "keyed by method name."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _validate_backend_args(args: argparse.Namespace) -> None:
    if args.mpc_backend == "mjx" and args.method not in MPC_METHODS:
        raise ValueError("--mpc-backend mjx requires an MPC method.")
    if args.mpc_backend == "mjx" and args.mpc_optimizer != "generic":
        raise ValueError("--mpc-backend mjx requires --mpc-optimizer generic.")
    if args.mjx_guided_candidate is not None and not (
        args.mpc_backend == "mjx" and args.mpc_optimizer == "generic"
    ):
        raise ValueError(
            "--mjx-guided-candidate requires --mpc-backend mjx "
            "and --mpc-optimizer generic."
        )
    if args.mpc_optimizer == "legacy" and args.method not in MPC_METHODS:
        raise ValueError("--mpc-optimizer legacy requires an MPC method.")
    if args.mpc_optimizer != "legacy":
        legacy_only_flags = _configured_legacy_only_mpc_flags(args)
        if legacy_only_flags:
            flags = ", ".join(legacy_only_flags)
            raise ValueError(
                "Legacy MPC options require the legacy optimizer "
                "(--mpc-optimizer legacy): "
                f"{flags}"
            )


def _configured_legacy_only_mpc_flags(args: argparse.Namespace) -> tuple[str, ...]:
    flags: list[str] = []
    if args.mpc_preset != "aggressive":
        flags.append("--mpc-preset")
    for name, flag in LEGACY_ONLY_MPC_FLAGS.items():
        if getattr(args, name) is not None:
            flags.append(flag)
    return tuple(flags)


def _mjx_physics_scan_enabled(args: argparse.Namespace) -> bool:
    return bool(args.mpc_backend == "mjx" or args.mjx_enable_scan)


def _required_mpc_metadata(metadata: dict[str, Any], name: str) -> Any:
    if name not in metadata:
        raise ValueError(f"MPC metadata is missing required field {name}")
    return metadata[name]


def _build_sampling_config(args: argparse.Namespace) -> Config:
    if args.mpc_samples is None:
        raise ValueError("--mpc-samples is required.")
    if args.mpc_iterations is None:
        raise ValueError("--mpc-iterations is required.")
    if args.mpc_planning_horizon_steps is None:
        raise ValueError("--mpc-planning-horizon-steps is required.")
    if args.mpc_control_steps is None:
        raise ValueError("--mpc-control-steps is required.")
    if args.mpc_knot_count is None:
        raise ValueError("--mpc-knot-count is required.")
    if args.mpc_temperature is None:
        raise ValueError("--mpc-temperature is required.")
    if args.mpc_root_pos_sigma is None:
        raise ValueError("--mpc-root-pos-sigma is required.")
    if args.mpc_root_rot_sigma is None:
        raise ValueError("--mpc-root-rot-sigma is required.")
    if args.mpc_joint_sigma is None:
        raise ValueError("--mpc-joint-sigma is required.")
    if args.mpc_final_noise_scale is None:
        final_noise_scale = 0.1
    else:
        final_noise_scale = float(args.mpc_final_noise_scale)
    first_ctrl_noise_scale = (
        0.5 if args.mpc_first_ctrl_noise_scale is None else float(args.mpc_first_ctrl_noise_scale)
    )
    last_ctrl_noise_scale = (
        1.0 if args.mpc_last_ctrl_noise_scale is None else float(args.mpc_last_ctrl_noise_scale)
    )
    use_warm_start = True if args.mpc_warm_start is None else bool(args.mpc_warm_start)
    use_guided_candidate = (
        False
        if args.mjx_guided_candidate is None
        else bool(args.mjx_guided_candidate)
    )
    return build_g1_wbc_sampling_config(
        device=args.device,
        num_samples=int(args.mpc_samples),
        rollout_batch_size=int(args.mpc_rollout_batch_size),
        max_num_iterations=int(args.mpc_iterations),
        horizon_steps=int(args.mpc_planning_horizon_steps),
        ctrl_steps=int(args.mpc_control_steps),
        knot_count=int(args.mpc_knot_count),
        temperature=float(args.mpc_temperature),
        control_update_mode=args.mpc_control_update_mode,
        pos_noise_scale=float(args.mpc_root_pos_sigma),
        rot_noise_scale=float(args.mpc_root_rot_sigma),
        joint_noise_scale=float(args.mpc_joint_sigma),
        first_ctrl_noise_scale=first_ctrl_noise_scale,
        last_ctrl_noise_scale=last_ctrl_noise_scale,
        final_noise_scale=final_noise_scale,
        use_torch_compile=bool(args.mpc_torch_compile),
        seed=int(args.seed),
        use_warm_start=use_warm_start,
        use_guided_candidate=use_guided_candidate,
    )


def _build_mpc_config(args: argparse.Namespace) -> G1WbcMpcConfig:
    """Build the legacy G1 WBC MPC config used by existing tests and runners."""

    config = mpc_config_from_preset(args.method, args.mpc_preset)
    overrides = {
        "num_samples": args.mpc_samples,
        "num_iterations": args.mpc_iterations,
        "planning_horizon_steps": args.mpc_planning_horizon_steps,
        "control_steps": args.mpc_control_steps,
        "sampling_mode": args.mpc_sampling_mode,
        "knot_count": args.mpc_knot_count,
        "elite_frac": args.mpc_elite_frac,
        "temperature": args.mpc_temperature,
        "root_pos_sigma": args.mpc_root_pos_sigma,
        "root_rot_sigma": args.mpc_root_rot_sigma,
        "joint_sigma": args.mpc_joint_sigma,
        "sigma_decay": args.mpc_sigma_decay,
        "smooth_passes": args.mpc_smooth_passes,
        "use_warm_start": args.mpc_warm_start,
        "warm_start_source": args.mpc_warm_start_source,
        "warm_start_decay": args.mpc_warm_start_decay,
        "command_reg_weight": args.mpc_command_reg_weight,
        "command_smooth_weight": args.mpc_command_smooth_weight,
        "acceptance_gate": args.mpc_acceptance_gate,
        "use_guided_candidate": args.mpc_guided_candidate,
        "guided_root_pos_gain": args.mpc_guided_root_pos_gain,
        "guided_root_rot_gain": args.mpc_guided_root_rot_gain,
        "guided_joint_gain": args.mpc_guided_joint_gain,
        "guided_root_pos_clip": args.mpc_guided_root_pos_clip,
        "guided_root_rot_clip": args.mpc_guided_root_rot_clip,
        "guided_joint_clip": args.mpc_guided_joint_clip,
        "seed": args.seed,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    if args.mpc_reward_weights is not None:
        config.reward_weights = load_reward_weights(args.mpc_reward_weights, args.method)
    return config


def _legacy_mpc_config_payload(config: G1WbcMpcConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("reward_weights", None)
    return payload


def _load_method_reward_weights(args: argparse.Namespace) -> dict[str, float] | None:
    if args.mpc_reward_weights is None:
        return None
    return load_reward_weights(args.mpc_reward_weights, args.method)


def _effective_reward_weights(method: str, weights: dict[str, float] | None) -> dict[str, float]:
    return {key: float(value) for key, value in reward_weights_for(method, weights).items()}


def _jsonable_infos(infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for info in infos:
        row: dict[str, Any] = {}
        for key, value in info.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    row[key] = float(value.detach().cpu().item())
                else:
                    row[key] = value.detach().cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                row[key] = value.tolist()
            elif isinstance(value, (int, float, bool, str)):
                row[key] = value
            else:
                row[key] = str(value)
        out.append(row)
    return out


def _save_rollout(path: Path, rollout: RolloutResult) -> None:
    arrays = {
        "qpos": _cpu_np(rollout.qpos),
        "qvel": _cpu_np(rollout.qvel),
        "body_pos_w": _cpu_np(rollout.body_pos_w),
        "body_quat_w": _cpu_np(rollout.body_quat_w),
        "body_lin_vel_w": _cpu_np(rollout.body_lin_vel_w),
        "body_ang_vel_w": _cpu_np(rollout.body_ang_vel_w),
        "actions": _cpu_np(rollout.actions),
        "controls": _cpu_np(rollout.controls),
        "contact_indicator": _cpu_np(rollout.contact_indicator),
        "contact_force": _cpu_np(rollout.contact_force),
        "ref_indices": _cpu_np(rollout.ref_indices),
        "dt": np.array(rollout.dt, dtype=np.float32),
    }
    if rollout.floor_contact_indicator is not None:
        arrays["floor_contact_indicator"] = _cpu_np(rollout.floor_contact_indicator)
    if rollout.floor_contact_force is not None:
        arrays["floor_contact_force"] = _cpu_np(rollout.floor_contact_force)
    np.savez_compressed(path, **arrays)


def _save_mpc_result(path: Path, result) -> None:
    arrays = {
        "refined_qpos": _cpu_np(result.refined_qpos),
        "candidate_scores": _cpu_np(result.scores),
        "command_joint_pos": _cpu_np(result.command.joint_pos),
        "command_joint_vel": _cpu_np(result.command.joint_vel),
        "command_body_pos_w": _cpu_np(result.command.body_pos_w),
        "command_body_quat_w": _cpu_np(result.command.body_quat_w),
        "command_body_lin_vel_w": _cpu_np(result.command.body_lin_vel_w),
        "command_body_ang_vel_w": _cpu_np(result.command.body_ang_vel_w),
        "command_qpos_trajectory": _cpu_np(result.command.qpos_trajectory),
        "command_qvel_trajectory": _cpu_np(result.command.qvel_trajectory),
    }
    np.savez_compressed(path, **arrays)


def _save_tracking_bfm_motion(path: Path, command) -> None:
    if int(command.num_envs) != 1:
        raise ValueError(
            "Expected single-env command for tracking-bfm export, "
            f"got {command.num_envs}."
        )
    arrays = {
        "fps": np.array(float(command.fps), dtype=np.float32),
        "joint_pos": _cpu_np(command.joint_pos[:, 0]).astype(np.float32, copy=False),
        "joint_vel": _cpu_np(command.joint_vel[:, 0]).astype(np.float32, copy=False),
        "body_pos_w": _cpu_np(command.body_pos_w[:, 0]).astype(np.float32, copy=False),
        "body_quat_w": _cpu_np(command.body_quat_w[:, 0]).astype(
            np.float32, copy=False
        ),
        "body_lin_vel_w": _cpu_np(command.body_lin_vel_w[:, 0]).astype(
            np.float32, copy=False
        ),
        "body_ang_vel_w": _cpu_np(command.body_ang_vel_w[:, 0]).astype(
            np.float32, copy=False
        ),
        "motion_type": np.array("mujoco"),
    }
    np.savez_compressed(path, **arrays)


def _load_saved_qpos(path: str | None, *, device: str) -> torch.Tensor:
    if path is None:
        raise ValueError("--method static_qpos requires --saved-qpos.")
    qpos_path = Path(path).expanduser().resolve()
    with np.load(qpos_path) as data:
        for key in ("qpos", "refined_qpos", "command_qpos_trajectory"):
            if key in data.files:
                qpos = data[key]
                break
        else:
            raise ValueError(
                f"{qpos_path} is missing qpos/refined_qpos/command_qpos_trajectory."
            )
    if qpos.ndim == 3 and qpos.shape[1] == 1:
        qpos = qpos[:, 0]
    if qpos.ndim != 2 or qpos.shape[-1] != 36:
        raise ValueError(f"Expected saved qpos shape (T,36) or (T,1,36), got {qpos.shape}.")
    return torch.tensor(qpos, dtype=torch.float32, device=device)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_saved_command_trajectory(
    path: str | None,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if path is None:
        raise ValueError("--method replay_command requires --saved-command.")
    command_path = Path(path).expanduser().resolve()
    with np.load(command_path) as data:
        for key in ("command_qpos_trajectory", "refined_qpos", "qpos"):
            if key in data.files:
                qpos = data[key]
                break
        else:
            raise ValueError(
                f"{command_path} is missing command_qpos_trajectory/refined_qpos/qpos."
            )
        qvel = (
            data["command_qvel_trajectory"]
            if "command_qvel_trajectory" in data.files
            else None
        )
    if qpos.ndim not in (2, 3) or qpos.shape[-1] != QPOS_DIM:
        raise ValueError(f"Expected saved command qpos shape (T,36), got {qpos.shape}.")
    if qvel is not None and (qvel.ndim not in (2, 3) or qvel.shape[-1] != QVEL_DIM):
        raise ValueError(f"Expected saved command qvel shape (T,35), got {qvel.shape}.")
    qpos_tensor = torch.tensor(qpos, dtype=torch.float32, device=device)
    qvel_tensor = (
        None
        if qvel is None
        else torch.tensor(qvel, dtype=torch.float32, device=device)
    )
    return qpos_tensor, qvel_tensor


def _resolve_replay_control_steps(args: argparse.Namespace) -> int:
    if args.replay_control_steps is not None:
        return int(args.replay_control_steps)
    if args.mpc_control_steps is not None:
        return int(args.mpc_control_steps)
    if args.saved_command is not None:
        metrics_path = Path(args.saved_command).expanduser().resolve().parent / "metrics.json"
        if metrics_path.exists():
            try:
                payload = json.loads(metrics_path.read_text())
                control_steps = payload.get("mpc", {}).get("control_steps")
                if control_steps is not None:
                    return int(control_steps)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
    return 20


def _synchronize_rollout_device(config: WbcRolloutConfig) -> None:
    torch_device = torch.device(config.device)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch_device)


def _runtime_visible_devices() -> tuple[str, ...]:
    return tuple(
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    )


def _runtime_gpu_name(config: WbcRolloutConfig) -> str | None:
    torch_device = torch.device(config.device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(torch_device))


def _cpu_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _safe_tensor_stat(value: torch.Tensor, stat: str) -> float:
    if value.numel() == 0:
        return 0.0
    value = torch.nan_to_num(value.float())
    if stat == "max":
        return float(value.max().detach().cpu().item())
    return float(value.mean().detach().cpu().item())


if __name__ == "__main__":
    main()
