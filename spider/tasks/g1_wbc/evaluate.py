"""CLI for evaluating G1 WBC policy rollouts on a single motion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from spider.tasks.g1_wbc.mjx_contacts import CONTACT_PROFILES
from spider.tasks.g1_wbc.spider_task import (
    G1WbcSamplingTask,
    build_g1_wbc_sampling_config,
    load_reward_weights,
    reward_weights_for,
    run_g1_wbc_sampling_mpc,
)
from spider.tasks.g1_wbc.motion import (
    G1CommandBatch,
    load_motion,
    validate_motion_dims,
)
from spider.tasks.g1_wbc.policy import load_wbc_actor, resolve_checkpoint_path
from spider.tasks.g1_wbc.result_types import (
    G1WbcExecutedCommandChunk,
    G1WbcWindowReplayState,
)
from spider.tasks.g1_wbc.rollout import (
    RolloutResult,
    WbcRolloutConfig,
    run_no_mpc_rollout,
    run_static_qpos_rollout,
)

MPC_METHODS = ("g1_wbc_ee", "g1_wbc_joint", "g1_wbc_joint_global")
LEGACY_ONLY_MPC_FLAGS = {
    "mpc_sampling_mode": "--mpc-sampling-mode",
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

    motion_path = Path(args.motion).expanduser().resolve()
    motion = load_motion(args.motion, motion_type=args.motion_type, device=device)
    validate_motion_dims(motion)
    metrics_reference_motion = motion
    metrics_reference_motion_path = motion_path
    if args.metrics_reference_motion is not None:
        metrics_reference_motion_path = (
            Path(args.metrics_reference_motion).expanduser().resolve()
        )
        metrics_reference_motion = load_motion(
            args.metrics_reference_motion,
            motion_type=args.metrics_reference_motion_type or args.motion_type,
            device=device,
        )
        validate_motion_dims(metrics_reference_motion)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    actor = None
    if args.method != "static_qpos":
        actor = load_wbc_actor(args.checkpoint, device=device)

    config = WbcRolloutConfig(
        model_path=args.model_path,
        collision_profile=args.collision_profile,
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
        command_chunks = _load_saved_command_chunks(
            args.saved_command,
            device=device,
        )
        qpos_trajectory = None
        qvel_trajectory = None
        if command_chunks:
            total_steps = _saved_command_chunk_step_count(command_chunks)
        else:
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
        if command_chunks:
            rollout = task.replay_command_chunks(
                command_chunks,
                total_steps=total_steps,
            )
        else:
            assert qpos_trajectory is not None
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
        replay_backend = (
            "G1WbcSamplingTask.replay_command_chunks"
            if command_chunks
            else "G1WbcSamplingTask.replay_qpos_command_sequence"
        )
        mpc_payload = {
            "backend": "spider.tasks.g1_wbc.spider_task." + replay_backend,
            "saved_command": str(saved_command_path),
            "saved_command_sha256": _file_sha256(saved_command_path),
            "replay_mode": "shared_execute_backend",
            "control_steps": control_steps,
            "use_saved_qvel": bool(args.replay_use_saved_qvel),
            "saved_command_replay_source": (
                "window_command_chunks"
                if command_chunks
                else "stitched_command_trajectory"
            ),
            "saved_command_replay_state_source": (
                _command_replay_state_source(command_chunks) if command_chunks else "none"
            ),
            "num_command_chunks": (
                len(command_chunks) if command_chunks else 0
            ),
            "num_command_replay_state_chunks": (
                _command_replay_state_chunk_count(command_chunks)
                if command_chunks
                else 0
            ),
            "serial_execute_warp_launches": bool(args.serial_execute_warp_launches),
            "steady_state_wall_time_sec": steady_state_wall_time_sec,
            "num_command_frames": int(_saved_command_frame_count(saved_command_path)),
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

            effective_reward_weights = _effective_reward_weights(
                args.method,
                reward_weights,
                mjx_contact_force_active_weight=(
                    args.mjx_contact_force_active_weight
                ),
                mjx_contact_force_delta_weight=(
                    args.mjx_contact_force_delta_weight
                ),
                mjx_contact_force_peak_excess_weight=(
                    args.mjx_contact_force_peak_excess_weight
                ),
                mjx_contact_false_positive_weight=(
                    args.mjx_contact_false_positive_weight
                ),
            )
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
                mjx_impl=args.mjx_impl,
                mjx_warp_naconmax=args.mjx_warp_naconmax,
                mjx_warp_njmax=args.mjx_warp_njmax,
                mjx_model_options=_mjx_model_options(args),
                trace_prefix_steps=args.mjx_trace_prefix_steps,
                strip_live_mjx_data_between_windows=bool(
                    args.mjx_strip_live_mjx_data
                ),
                score_only_optimizer=bool(args.mjx_score_only_optimizer),
                score_only_rescore_diagnostics=bool(
                    args.mjx_score_only_rescore_diagnostics
                ),
                score_only_output_rescore_diagnostics=bool(
                    args.mjx_score_only_output_rescore_diagnostics
                ),
                contact_force_mode=str(args.mjx_contact_force_mode),
                contact_force_first_row_diagnostics=bool(
                    args.mjx_contact_force_first_row_diagnostics
                ),
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
                "reward_weights": _mpc_payload_reward_weights(
                    args.method,
                    reward_weights,
                    mpc_run.metadata,
                ),
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
    metrics = compute_rollout_metrics(metrics_reference_motion, rollout)

    payload = {
        "method": args.method,
        "motion": str(motion_path),
        "motion_type": motion.motion_type,
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "device": device,
        "collision_profile": args.collision_profile,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "ref_offset": args.ref_offset,
        "serial_warp_launches": bool(args.serial_warp_launches),
        "metrics": metrics,
    }
    if args.metrics_reference_motion is not None:
        payload["metrics_reference_motion"] = str(metrics_reference_motion_path)
        payload["metrics_reference_motion_type"] = (
            metrics_reference_motion.motion_type
        )
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
        "--metrics-reference-motion",
        default=None,
        help=(
            "Optional motion npz used only for metrics. Rollout still consumes "
            "--motion. This is useful for scoring an optimized command motion "
            "against its original source motion."
        ),
    )
    parser.add_argument(
        "--metrics-reference-motion-type",
        default=None,
        choices=("auto", "mujoco", "isaaclab"),
        help=(
            "Semantic ordering for --metrics-reference-motion. Defaults to "
            "--motion-type."
        ),
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
    parser.add_argument(
        "--collision-profile",
        default=WbcRolloutConfig.collision_profile,
        choices=tuple(sorted(CONTACT_PROFILES)),
        help="WXY contact profile used when dynamically building the G1 WBC model.",
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
        "--mjx-impl",
        choices=("jax", "warp"),
        default="jax",
        help="MuJoCo MJX implementation used by --mpc-backend mjx.",
    )
    parser.add_argument(
        "--mjx-warp-naconmax",
        type=int,
        default=30000,
        help="Global MJX-Warp contact buffer passed to mujoco.mjx.make_data.",
    )
    parser.add_argument(
        "--mjx-warp-njmax",
        type=int,
        default=256,
        help="Global MJX-Warp constraint/Jacobian buffer passed to mujoco.mjx.make_data.",
    )
    parser.add_argument(
        "--mjx-model-iterations",
        type=int,
        default=None,
        help="Diagnostic MJX model opt.iterations override for solver speed studies.",
    )
    parser.add_argument(
        "--mjx-model-ls-iterations",
        type=int,
        default=None,
        help="Diagnostic MJX model opt.ls_iterations override for solver speed studies.",
    )
    parser.add_argument(
        "--mjx-trace-prefix-steps",
        type=int,
        default=None,
        help=(
            "Diagnostic: ask the MJX scorer to return an execute-prefix trace "
            "for optimizer-selected candidates. Non-default values are not formal "
            "acceptance evidence unless promoted in criteria."
        ),
    )
    parser.add_argument(
        "--mjx-strip-live-mjx-data",
        action="store_true",
        help=(
            "Diagnostic: strip live MJX data from window-to-window robot state "
            "while preserving qpos/qvel and explicit carry fields."
        ),
    )
    parser.add_argument(
        "--mjx-score-only-optimizer",
        action="store_true",
        help=(
            "Diagnostic: score optimizer candidates with compact score-only "
            "metrics. Formal acceptance must keep the default full metrics unless "
            "criteria promote this path."
        ),
    )
    parser.add_argument(
        "--mjx-score-only-rescore-diagnostics",
        action="store_true",
        help=(
            "Diagnostic: keep the primary MJX optimizer scorer unchanged, but "
            "shadow-rescore the same candidates with a score-only scorer and "
            "record score deltas. Shadow scores are not used for selection."
        ),
    )
    parser.add_argument(
        "--mjx-score-only-output-rescore-diagnostics",
        action="store_true",
        help=(
            "Diagnostic: keep the primary MJX optimizer scorer unchanged, but "
            "shadow-rescore the same candidates with a scorer that accumulates "
            "full metrics and returns only score. Shadow scores are not used "
            "for selection."
        ),
    )
    parser.add_argument(
        "--mjx-contact-force-first-row-diagnostics",
        action="store_true",
        help=(
            "Diagnostic: materialize first solver-row contact-force fields in MJX "
            "artifact traces while keeping the optimizer scorer on the default "
            "light physics path."
        ),
    )
    parser.add_argument(
        "--mjx-contact-force-mode",
        choices=("sum_rows", "first_row"),
        default="sum_rows",
        help=(
            "Diagnostic: choose which MJX solver-row contact force is exposed "
            "to the optimizer scorer and rollout metrics. Default preserves "
            "the existing sum-rows behavior."
        ),
    )
    parser.add_argument(
        "--mjx-contact-force-active-weight",
        type=float,
        default=0.0,
        help=(
            "Diagnostic MJX-only scorer weight for active foot contact-force "
            "magnitude. The term is normalized by 300N and defaults to 0/off."
        ),
    )
    parser.add_argument(
        "--mjx-contact-force-delta-weight",
        type=float,
        default=None,
        help=(
            "Diagnostic MJX-only override for the contact-force-delta scorer "
            "weight. Omit to preserve the method reward config."
        ),
    )
    parser.add_argument(
        "--mjx-contact-force-peak-excess-weight",
        type=float,
        default=0.0,
        help=(
            "Diagnostic MJX-only scorer weight for peak contact force above "
            "300N. The excess is normalized by 300N and defaults to 0/off."
        ),
    )
    parser.add_argument(
        "--mjx-contact-false-positive-weight",
        type=float,
        default=None,
        help=(
            "Diagnostic MJX-only override for the contact false-positive scorer "
            "weight. Omit to preserve the method reward config."
        ),
    )
    parser.add_argument(
        "--mjx-guided-candidate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable MJX-generated guided candidate controls for the generic MJX backend.",
    )
    parser.add_argument(
        "--mjx-guided-candidate-period",
        type=int,
        default=None,
        help=(
            "Diagnostic: when guided candidate is enabled, generate it only every "
            "N MPC windows to reduce guided no-MPC rollout overhead."
        ),
    )
    parser.add_argument(
        "--mjx-min-score-improvement",
        type=float,
        default=None,
        help=(
            "Diagnostic: require this score improvement before accepting a "
            "non-current MJX optimizer candidate. Defaults to the legacy 1e-9."
        ),
    )
    parser.add_argument(
        "--mjx-min-top-score-gap",
        type=float,
        default=None,
        help=(
            "Diagnostic: require this best-vs-second-best score gap before "
            "accepting a non-current MJX optimizer candidate. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--mjx-cem-update-min-top-score-gap",
        type=float,
        default=None,
        help=(
            "Diagnostic: require this best-vs-second-best score gap before "
            "updating the adaptive CEM sampling center/sigma. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--mjx-max-control-delta",
        type=float,
        default=None,
        help=(
            "Diagnostic: keep the current controls when the selected MJX "
            "candidate exceeds this max absolute control delta."
        ),
    )
    parser.add_argument(
        "--mjx-candidate-rank-diagnostics-top-k",
        type=int,
        default=0,
        help=(
            "Diagnostic: record per-iteration top-k MJX optimizer candidate "
            "indices and scores in MPC history. Defaults to 0/off."
        ),
    )
    parser.add_argument(
        "--mjx-candidate-rescore-diagnostics",
        action="store_true",
        help=(
            "Diagnostic: rescore the same sampled MJX optimizer candidates once "
            "per iteration and record score deltas. Defaults to off."
        ),
    )
    parser.add_argument(
        "--mjx-candidate-rescore-selection-top-k",
        type=int,
        default=0,
        help=(
            "Diagnostic: rescore the current top-k MJX optimizer candidates once "
            "per iteration and select using the average score. Defaults to 0/off."
        ),
    )
    parser.add_argument(
        "--mjx-candidate-score-component-diagnostics-top-k",
        type=int,
        default=0,
        help=(
            "Diagnostic: record score-component values for current, selected, "
            "and top-k MJX optimizer candidates. Defaults to 0/off."
        ),
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
    if int(args.mjx_warp_naconmax) <= 0:
        raise ValueError("--mjx-warp-naconmax must be positive.")
    if int(args.mjx_warp_njmax) <= 0:
        raise ValueError("--mjx-warp-njmax must be positive.")
    if args.mjx_model_iterations is not None and int(args.mjx_model_iterations) <= 0:
        raise ValueError("--mjx-model-iterations must be positive.")
    if (
        args.mjx_model_ls_iterations is not None
        and int(args.mjx_model_ls_iterations) <= 0
    ):
        raise ValueError("--mjx-model-ls-iterations must be positive.")
    if args.mjx_trace_prefix_steps is not None and int(args.mjx_trace_prefix_steps) <= 0:
        raise ValueError("--mjx-trace-prefix-steps must be positive.")
    if args.mjx_impl == "warp" and args.mpc_backend != "mjx":
        raise ValueError("--mjx-impl warp requires --mpc-backend mjx.")
    if args.mjx_strip_live_mjx_data and args.mpc_backend != "mjx":
        raise ValueError("--mjx-strip-live-mjx-data requires --mpc-backend mjx.")
    if args.mjx_score_only_optimizer and args.mpc_backend != "mjx":
        raise ValueError("--mjx-score-only-optimizer requires --mpc-backend mjx.")
    if args.mjx_contact_force_first_row_diagnostics and args.mpc_backend != "mjx":
        raise ValueError(
            "--mjx-contact-force-first-row-diagnostics requires --mpc-backend mjx."
        )
    if args.mjx_contact_force_mode != "sum_rows" and args.mpc_backend != "mjx":
        raise ValueError("--mjx-contact-force-mode requires --mpc-backend mjx.")
    contact_force_active_weight = float(args.mjx_contact_force_active_weight)
    if (
        not math.isfinite(contact_force_active_weight)
        or contact_force_active_weight < 0.0
    ):
        raise ValueError("--mjx-contact-force-active-weight must be non-negative.")
    if contact_force_active_weight > 0.0 and not (
        args.mpc_backend == "mjx" and args.mpc_optimizer == "generic"
    ):
        raise ValueError(
            "--mjx-contact-force-active-weight requires --mpc-backend mjx "
            "and --mpc-optimizer generic."
        )
    if args.mjx_contact_force_delta_weight is not None:
        contact_force_delta_weight = float(args.mjx_contact_force_delta_weight)
        if (
            not math.isfinite(contact_force_delta_weight)
            or contact_force_delta_weight < 0.0
        ):
            raise ValueError("--mjx-contact-force-delta-weight must be non-negative.")
        if not (args.mpc_backend == "mjx" and args.mpc_optimizer == "generic"):
            raise ValueError(
                "--mjx-contact-force-delta-weight requires --mpc-backend mjx "
                "and --mpc-optimizer generic."
            )
    contact_force_peak_excess_weight = float(
        args.mjx_contact_force_peak_excess_weight
    )
    if (
        not math.isfinite(contact_force_peak_excess_weight)
        or contact_force_peak_excess_weight < 0.0
    ):
        raise ValueError(
            "--mjx-contact-force-peak-excess-weight must be non-negative."
        )
    if contact_force_peak_excess_weight > 0.0 and not (
        args.mpc_backend == "mjx" and args.mpc_optimizer == "generic"
    ):
        raise ValueError(
            "--mjx-contact-force-peak-excess-weight requires --mpc-backend mjx "
            "and --mpc-optimizer generic."
        )
    if args.mjx_contact_false_positive_weight is not None:
        contact_false_positive_weight = float(
            args.mjx_contact_false_positive_weight
        )
        if (
            not math.isfinite(contact_false_positive_weight)
            or contact_false_positive_weight < 0.0
        ):
            raise ValueError(
                "--mjx-contact-false-positive-weight must be non-negative."
            )
        if not (args.mpc_backend == "mjx" and args.mpc_optimizer == "generic"):
            raise ValueError(
                "--mjx-contact-false-positive-weight requires --mpc-backend mjx "
                "and --mpc-optimizer generic."
            )
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
    if args.mjx_guided_candidate_period is not None:
        if int(args.mjx_guided_candidate_period) <= 0:
            raise ValueError("--mjx-guided-candidate-period must be positive.")
        if not (
            args.mpc_backend == "mjx"
            and args.mpc_optimizer == "generic"
            and args.mjx_guided_candidate is True
        ):
            raise ValueError(
                "--mjx-guided-candidate-period requires enabled "
                "--mjx-guided-candidate on the MJX generic backend."
            )
    if args.mjx_min_score_improvement is not None:
        min_score_improvement = float(args.mjx_min_score_improvement)
        if not math.isfinite(min_score_improvement) or min_score_improvement < 0.0:
            raise ValueError("--mjx-min-score-improvement must be non-negative.")
        if not (args.mpc_backend == "mjx" and args.mpc_optimizer == "generic"):
            raise ValueError(
                "--mjx-min-score-improvement requires --mpc-backend mjx "
                "and --mpc-optimizer generic."
            )
    for flag_name, value in (
        ("--mjx-min-top-score-gap", args.mjx_min_top_score_gap),
        (
            "--mjx-cem-update-min-top-score-gap",
            args.mjx_cem_update_min_top_score_gap,
        ),
    ):
        if value is None:
            continue
        parsed = float(value)
        if not math.isfinite(parsed) or parsed < 0.0:
            raise ValueError(f"{flag_name} must be non-negative.")
        if not (args.mpc_backend == "mjx" and args.mpc_optimizer == "generic"):
            raise ValueError(
                f"{flag_name} requires --mpc-backend mjx "
                "and --mpc-optimizer generic."
            )
    if args.mjx_max_control_delta is not None:
        max_control_delta = float(args.mjx_max_control_delta)
        if not math.isfinite(max_control_delta) or max_control_delta <= 0.0:
            raise ValueError("--mjx-max-control-delta must be positive.")
        if not (args.mpc_backend == "mjx" and args.mpc_optimizer == "generic"):
            raise ValueError(
                "--mjx-max-control-delta requires --mpc-backend mjx "
                "and --mpc-optimizer generic."
            )
    if int(args.mjx_candidate_rank_diagnostics_top_k) < 0:
        raise ValueError("--mjx-candidate-rank-diagnostics-top-k must be non-negative.")
    if int(args.mjx_candidate_rank_diagnostics_top_k) > 0 and not (
        args.mpc_backend == "mjx" and args.mpc_optimizer == "generic"
    ):
        raise ValueError(
            "--mjx-candidate-rank-diagnostics-top-k requires --mpc-backend mjx "
            "and --mpc-optimizer generic."
        )
    if args.mjx_candidate_rescore_diagnostics and not (
        args.mpc_backend == "mjx" and args.mpc_optimizer == "generic"
    ):
        raise ValueError(
            "--mjx-candidate-rescore-diagnostics requires --mpc-backend mjx "
            "and --mpc-optimizer generic."
        )
    if args.mjx_score_only_rescore_diagnostics and not (
        args.mpc_backend == "mjx" and args.mpc_optimizer == "generic"
    ):
        raise ValueError(
            "--mjx-score-only-rescore-diagnostics requires --mpc-backend mjx "
            "and --mpc-optimizer generic."
        )
    if args.mjx_score_only_output_rescore_diagnostics and not (
        args.mpc_backend == "mjx" and args.mpc_optimizer == "generic"
    ):
        raise ValueError(
            "--mjx-score-only-output-rescore-diagnostics requires "
            "--mpc-backend mjx and --mpc-optimizer generic."
        )
    if int(args.mjx_candidate_rescore_selection_top_k) < 0:
        raise ValueError("--mjx-candidate-rescore-selection-top-k must be non-negative.")
    if int(args.mjx_candidate_rescore_selection_top_k) > 0 and not (
        args.mpc_backend == "mjx" and args.mpc_optimizer == "generic"
    ):
        raise ValueError(
            "--mjx-candidate-rescore-selection-top-k requires --mpc-backend mjx "
            "and --mpc-optimizer generic."
        )
    if int(args.mjx_candidate_score_component_diagnostics_top_k) < 0:
        raise ValueError(
            "--mjx-candidate-score-component-diagnostics-top-k must be non-negative."
        )
    if int(args.mjx_candidate_score_component_diagnostics_top_k) > 0 and not (
        args.mpc_backend == "mjx" and args.mpc_optimizer == "generic"
    ):
        raise ValueError(
            "--mjx-candidate-score-component-diagnostics-top-k requires "
            "--mpc-backend mjx and --mpc-optimizer generic."
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


def _mjx_model_options(args: argparse.Namespace) -> dict[str, int] | None:
    options: dict[str, int] = {}
    if args.mjx_model_iterations is not None:
        options["iterations"] = int(args.mjx_model_iterations)
    if args.mjx_model_ls_iterations is not None:
        options["ls_iterations"] = int(args.mjx_model_ls_iterations)
    return options or None


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
    mjx_min_score_improvement = (
        1.0e-9
        if args.mjx_min_score_improvement is None
        else float(args.mjx_min_score_improvement)
    )
    mjx_min_top_score_gap = (
        0.0 if args.mjx_min_top_score_gap is None else float(args.mjx_min_top_score_gap)
    )
    mjx_cem_update_min_top_score_gap = (
        0.0
        if args.mjx_cem_update_min_top_score_gap is None
        else float(args.mjx_cem_update_min_top_score_gap)
    )
    mjx_max_control_delta = (
        None
        if args.mjx_max_control_delta is None
        else float(args.mjx_max_control_delta)
    )
    mjx_candidate_rank_diagnostics_top_k = int(
        args.mjx_candidate_rank_diagnostics_top_k
    )
    mjx_candidate_rescore_selection_top_k = int(
        args.mjx_candidate_rescore_selection_top_k
    )
    mjx_candidate_score_component_diagnostics_top_k = int(
        args.mjx_candidate_score_component_diagnostics_top_k
    )
    return build_g1_wbc_sampling_config(
        device=args.device,
        num_samples=int(args.mpc_samples),
        rollout_batch_size=int(args.mpc_rollout_batch_size),
        max_num_iterations=int(args.mpc_iterations),
        horizon_steps=int(args.mpc_planning_horizon_steps),
        ctrl_steps=int(args.mpc_control_steps),
        knot_count=int(args.mpc_knot_count),
        elite_frac=args.mpc_elite_frac,
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
        guided_candidate_period=args.mjx_guided_candidate_period,
        mjx_min_score_improvement=mjx_min_score_improvement,
        mjx_min_top_score_gap=mjx_min_top_score_gap,
        mjx_cem_update_min_top_score_gap=mjx_cem_update_min_top_score_gap,
        mjx_max_control_delta=mjx_max_control_delta,
        mjx_candidate_rank_diagnostics_top_k=(
            mjx_candidate_rank_diagnostics_top_k
        ),
        mjx_candidate_rescore_diagnostics=bool(
            args.mjx_candidate_rescore_diagnostics
        ),
        mjx_candidate_rescore_selection_top_k=(
            mjx_candidate_rescore_selection_top_k
        ),
        mjx_score_only_rescore_diagnostics=bool(
            args.mjx_score_only_rescore_diagnostics
        ),
        mjx_score_only_output_rescore_diagnostics=bool(
            args.mjx_score_only_output_rescore_diagnostics
        ),
        mjx_candidate_score_component_diagnostics_top_k=(
            mjx_candidate_score_component_diagnostics_top_k
        ),
        sigma_decay=args.mpc_sigma_decay,
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


def _effective_reward_weights(
    method: str,
    weights: dict[str, float] | None,
    *,
    mjx_contact_force_active_weight: float = 0.0,
    mjx_contact_force_delta_weight: float | None = None,
    mjx_contact_force_peak_excess_weight: float = 0.0,
    mjx_contact_false_positive_weight: float | None = None,
) -> dict[str, float]:
    effective = {
        key: float(value)
        for key, value in reward_weights_for(method, weights).items()
    }
    active_weight = float(mjx_contact_force_active_weight)
    if active_weight > 0.0:
        effective["contact_force_active"] = active_weight
    if mjx_contact_force_delta_weight is not None:
        effective["contact_force_delta"] = float(mjx_contact_force_delta_weight)
    peak_excess_weight = float(mjx_contact_force_peak_excess_weight)
    if peak_excess_weight > 0.0:
        effective["contact_force_peak_excess"] = peak_excess_weight
    if mjx_contact_false_positive_weight is not None:
        effective["contact_false_positive"] = float(
            mjx_contact_false_positive_weight
        )
    return effective


def _mpc_payload_reward_weights(
    method: str,
    weights: dict[str, float] | None,
    metadata: dict[str, Any] | None,
) -> dict[str, float]:
    metadata_weights = (metadata or {}).get("reward_weights")
    if isinstance(metadata_weights, dict):
        return {str(key): float(value) for key, value in metadata_weights.items()}
    return _effective_reward_weights(method, weights)


def _jsonable_infos(infos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for info in infos:
        row: dict[str, Any] = {}
        for key, value in info.items():
            row[key] = _jsonable_info_value(value)
        out.append(row)
    return out


def _jsonable_info_value(value):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.detach().cpu().item())
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable_info_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_info_value(item) for key, item in value.items()}
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable_info_value(item())
        except Exception:
            pass
    return str(value)


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
    contact_force_first_row = getattr(rollout, "contact_force_first_row", None)
    if contact_force_first_row is not None:
        arrays["contact_force_first_row"] = _cpu_np(contact_force_first_row)
    floor_contact_force_first_row = getattr(
        rollout,
        "floor_contact_force_first_row",
        None,
    )
    if floor_contact_force_first_row is not None:
        arrays["floor_contact_force_first_row"] = _cpu_np(
            floor_contact_force_first_row
        )
    floor_contact_force_peak_source = getattr(
        rollout,
        "floor_contact_force_peak_source",
        None,
    )
    if floor_contact_force_peak_source is not None:
        arrays["floor_contact_force_peak_source"] = _cpu_np(
            floor_contact_force_peak_source
        )
    np.savez_compressed(path, **arrays)


def _save_mpc_result(path: Path, result) -> None:
    arrays = {
        "refined_qpos": _cpu_np(result.refined_qpos),
        "candidate_scores": _cpu_np(result.scores),
        "command_fps": np.array(float(result.command.fps), dtype=np.float32),
        "command_motion_type": np.array(str(result.command.motion_type)),
        "command_joint_pos": _cpu_np(result.command.joint_pos),
        "command_joint_vel": _cpu_np(result.command.joint_vel),
        "command_body_pos_w": _cpu_np(result.command.body_pos_w),
        "command_body_quat_w": _cpu_np(result.command.body_quat_w),
        "command_body_lin_vel_w": _cpu_np(result.command.body_lin_vel_w),
        "command_body_ang_vel_w": _cpu_np(result.command.body_ang_vel_w),
        "command_qpos_trajectory": _cpu_np(result.command.qpos_trajectory),
        "command_qvel_trajectory": _cpu_np(result.command.qvel_trajectory),
    }
    chunks = getattr(result, "executed_command_chunks", None)
    if chunks:
        arrays.update(_executed_command_chunk_arrays(chunks))
    np.savez_compressed(path, **arrays)


def _executed_command_chunk_arrays(
    chunks: list[G1WbcExecutedCommandChunk],
) -> dict[str, np.ndarray]:
    chunk_list = list(chunks)
    if not chunk_list:
        return {}
    max_steps = max(int(chunk.command.num_frames) for chunk in chunk_list)
    if max_steps < 1:
        raise ValueError("Executed command chunks must contain at least one step.")
    first = chunk_list[0].command
    num_chunks = len(chunk_list)

    def padded(name: str) -> np.ndarray:
        value = _cpu_np(getattr(first, name))
        return np.zeros((num_chunks, max_steps) + value.shape[1:], dtype=value.dtype)

    arrays: dict[str, np.ndarray] = {
        "window_command_schema_version": np.array(1, dtype=np.int32),
        "window_starts": np.zeros(num_chunks, dtype=np.int32),
        "window_execute_steps": np.zeros(num_chunks, dtype=np.int32),
        "window_horizons": np.zeros(num_chunks, dtype=np.int32),
        "window_command_joint_pos": padded("joint_pos"),
        "window_command_joint_vel": padded("joint_vel"),
        "window_command_body_pos_w": padded("body_pos_w"),
        "window_command_body_quat_w": padded("body_quat_w"),
        "window_command_body_lin_vel_w": padded("body_lin_vel_w"),
        "window_command_body_ang_vel_w": padded("body_ang_vel_w"),
        "window_command_qpos_chunks": padded("qpos_trajectory"),
        "window_command_qvel_chunks": padded("qvel_trajectory"),
    }
    fields = (
        ("window_command_joint_pos", "joint_pos"),
        ("window_command_joint_vel", "joint_vel"),
        ("window_command_body_pos_w", "body_pos_w"),
        ("window_command_body_quat_w", "body_quat_w"),
        ("window_command_body_lin_vel_w", "body_lin_vel_w"),
        ("window_command_body_ang_vel_w", "body_ang_vel_w"),
        ("window_command_qpos_chunks", "qpos_trajectory"),
        ("window_command_qvel_chunks", "qvel_trajectory"),
    )
    for index, chunk in enumerate(chunk_list):
        command = chunk.command
        if int(command.num_envs) != 1:
            raise ValueError(
                "Executed command chunk export expects single-env commands, "
                f"got {command.num_envs}."
            )
        steps = int(command.num_frames)
        execute_steps = int(chunk.execute_steps)
        if execute_steps < 1:
            raise ValueError("Executed command chunks must have positive execute steps.")
        if execute_steps > steps:
            raise ValueError(
                "Executed command chunk execute_steps exceeds command horizon: "
                f"{execute_steps} > {steps}."
            )
        arrays["window_starts"][index] = int(chunk.start)
        arrays["window_execute_steps"][index] = execute_steps
        arrays["window_horizons"][index] = int(chunk.horizon_steps)
        for array_name, attr_name in fields:
            arrays[array_name][index, :steps] = _cpu_np(getattr(command, attr_name)[:steps])
    arrays.update(_executed_command_replay_state_arrays(chunk_list))
    return arrays


def _executed_command_replay_state_arrays(
    chunks: list[G1WbcExecutedCommandChunk],
) -> dict[str, np.ndarray]:
    states = [chunk.replay_state for chunk in chunks]
    if not any(state is not None for state in states):
        return {}
    num_chunks = len(chunks)
    arrays: dict[str, np.ndarray] = {
        "window_replay_state_schema_version": np.array(1, dtype=np.int32),
        "window_replay_state_valid": np.zeros(num_chunks, dtype=np.int8),
        "window_replay_initial_qpos": np.zeros((num_chunks, QPOS_DIM), dtype=np.float32),
        "window_replay_initial_qvel": np.zeros((num_chunks, QVEL_DIM), dtype=np.float32),
        "window_replay_initial_last_action_valid": np.zeros(num_chunks, dtype=np.int8),
        "window_replay_initial_last_action": np.zeros(
            (num_chunks, QPOS_DIM - 7),
            dtype=np.float32,
        ),
    }
    history_names = sorted(
        {
            str(name)
            for state in states
            if state is not None and state.initial_history_state
            for name in state.initial_history_state
        }
    )
    history_buffers: dict[str, tuple[int, ...]] = {}
    history_num_pushes: dict[str, tuple[int, ...]] = {}
    for name in history_names:
        for state in states:
            if state is None or not state.initial_history_state:
                continue
            item = state.initial_history_state.get(name)
            if not isinstance(item, dict):
                continue
            buffer = item.get("buffer")
            if isinstance(buffer, torch.Tensor):
                history_buffers[name] = tuple(int(dim) for dim in buffer.shape)
            num_pushes = item.get("num_pushes")
            if isinstance(num_pushes, torch.Tensor):
                history_num_pushes[name] = tuple(int(dim) for dim in num_pushes.shape)
            if name in history_buffers and name in history_num_pushes:
                break
    for name in history_names:
        key = _history_npz_key(name)
        buffer_shape = history_buffers.get(name)
        num_pushes_shape = history_num_pushes.get(name, (1,))
        arrays[f"window_replay_history__{key}__valid"] = np.zeros(
            num_chunks,
            dtype=np.int8,
        )
        arrays[f"window_replay_history__{key}__pointer"] = np.full(
            num_chunks,
            -1,
            dtype=np.int32,
        )
        arrays[f"window_replay_history__{key}__num_pushes"] = np.zeros(
            (num_chunks, *num_pushes_shape),
            dtype=np.int64,
        )
        if buffer_shape is not None:
            arrays[f"window_replay_history__{key}__buffer"] = np.zeros(
                (num_chunks, *buffer_shape),
                dtype=np.float32,
            )
    for index, state in enumerate(states):
        if state is None:
            continue
        arrays["window_replay_state_valid"][index] = 1
        arrays["window_replay_initial_qpos"][index] = _cpu_np(state.initial_qpos)
        arrays["window_replay_initial_qvel"][index] = _cpu_np(state.initial_qvel)
        if state.initial_last_action is not None:
            arrays["window_replay_initial_last_action_valid"][index] = 1
            arrays["window_replay_initial_last_action"][index] = _cpu_np(
                state.initial_last_action
            ).reshape(-1)
        if not state.initial_history_state:
            continue
        for name, item in state.initial_history_state.items():
            if not isinstance(item, dict):
                continue
            key = _history_npz_key(str(name))
            valid_name = f"window_replay_history__{key}__valid"
            if valid_name not in arrays:
                continue
            arrays[valid_name][index] = 1
            arrays[f"window_replay_history__{key}__pointer"][index] = int(
                item.get("pointer", -1)
            )
            if isinstance(item.get("num_pushes"), torch.Tensor):
                arrays[f"window_replay_history__{key}__num_pushes"][index] = _cpu_np(
                    item["num_pushes"]
                )
            if isinstance(item.get("buffer"), torch.Tensor):
                arrays[f"window_replay_history__{key}__buffer"][index] = _cpu_np(
                    item["buffer"]
                )
    return arrays


def _history_npz_key(name: str) -> str:
    if "__" in name:
        raise ValueError(f"History state names may not contain '__': {name!r}")
    return name


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


def _load_saved_command_chunks(
    path: str | None,
    *,
    device: str,
) -> list[G1WbcExecutedCommandChunk] | None:
    if path is None:
        raise ValueError("--method replay_command requires --saved-command.")
    command_path = Path(path).expanduser().resolve()
    with np.load(command_path) as data:
        if "window_starts" not in data.files:
            return None
        if "window_command_schema_version" not in data.files:
            raise ValueError(f"{command_path} is missing window_command_schema_version.")
        schema_version = _npz_scalar_int(data, "window_command_schema_version")
        if schema_version != 1:
            raise ValueError(
                f"Unsupported window command schema version {schema_version} "
                f"in {command_path}; expected 1."
            )
        required = (
            "window_execute_steps",
            "window_horizons",
            "window_command_joint_pos",
            "window_command_joint_vel",
            "window_command_body_pos_w",
            "window_command_body_quat_w",
            "window_command_body_lin_vel_w",
            "window_command_body_ang_vel_w",
            "window_command_qpos_chunks",
            "window_command_qvel_chunks",
        )
        missing = [name for name in required if name not in data.files]
        if missing:
            raise ValueError(
                f"{command_path} is missing window command chunk arrays: {missing}"
            )
        starts = np.asarray(data["window_starts"], dtype=np.int64)
        execute_steps = np.asarray(
            data["window_execute_steps"],
            dtype=np.int64,
        )
        horizon_steps = np.asarray(
            data["window_horizons"],
            dtype=np.int64,
        )
        if starts.ndim != 1 or execute_steps.shape != starts.shape:
            raise ValueError(f"Malformed executed command chunk metadata in {command_path}.")
        if horizon_steps.shape != starts.shape:
            raise ValueError(f"Malformed executed command chunk metadata in {command_path}.")
        motion_type = _npz_scalar_text(data, "command_motion_type", default="mujoco")
        fps = _npz_scalar_float(data, "command_fps", default=50.0)
        fields = {
            "joint_pos": np.asarray(data["window_command_joint_pos"]),
            "joint_vel": np.asarray(data["window_command_joint_vel"]),
            "body_pos_w": np.asarray(data["window_command_body_pos_w"]),
            "body_quat_w": np.asarray(data["window_command_body_quat_w"]),
            "body_lin_vel_w": np.asarray(data["window_command_body_lin_vel_w"]),
            "body_ang_vel_w": np.asarray(data["window_command_body_ang_vel_w"]),
            "qpos_trajectory": np.asarray(data["window_command_qpos_chunks"]),
            "qvel_trajectory": np.asarray(data["window_command_qvel_chunks"]),
        }
        replay_states = _load_window_replay_states(
            data,
            expected_chunks=int(starts.shape[0]),
            device=device,
        )
    chunks: list[G1WbcExecutedCommandChunk] = []
    for index, start in enumerate(starts.tolist()):
        steps = int(execute_steps[index])
        horizon = int(horizon_steps[index])
        if steps < 1 or horizon < 1:
            raise ValueError("Executed command chunk execute_steps must be positive.")
        if steps > horizon:
            raise ValueError(
                f"Executed command chunk execute_steps exceeds horizon: {steps} > {horizon}."
            )
        command = G1CommandBatch(
            path=command_path,
            motion_type=motion_type,
            fps=fps,
            joint_pos=_tensor_chunk(fields["joint_pos"], index, horizon, device=device),
            joint_vel=_tensor_chunk(fields["joint_vel"], index, horizon, device=device),
            body_pos_w=_tensor_chunk(fields["body_pos_w"], index, horizon, device=device),
            body_quat_w=_tensor_chunk(fields["body_quat_w"], index, horizon, device=device),
            body_lin_vel_w=_tensor_chunk(
                fields["body_lin_vel_w"],
                index,
                horizon,
                device=device,
            ),
            body_ang_vel_w=_tensor_chunk(
                fields["body_ang_vel_w"],
                index,
                horizon,
                device=device,
            ),
            qpos_trajectory=_tensor_chunk(
                fields["qpos_trajectory"],
                index,
                horizon,
                device=device,
            ),
            qvel_trajectory=_tensor_chunk(
                fields["qvel_trajectory"],
                index,
                horizon,
                device=device,
            ),
        )
        chunks.append(
            G1WbcExecutedCommandChunk(
                start=int(start),
                execute_steps=steps,
                horizon_steps=horizon,
                command=command,
                replay_state=replay_states[index] if replay_states else None,
            )
        )
    return chunks


def _load_window_replay_states(
    data: np.lib.npyio.NpzFile,
    *,
    expected_chunks: int,
    device: str,
) -> list[G1WbcWindowReplayState | None] | None:
    if "window_replay_state_schema_version" not in data.files:
        return None
    schema_version = _npz_scalar_int(data, "window_replay_state_schema_version")
    if schema_version != 1:
        raise ValueError(
            f"Unsupported window replay state schema version {schema_version}; expected 1."
        )
    required = (
        "window_replay_state_valid",
        "window_replay_initial_qpos",
        "window_replay_initial_qvel",
        "window_replay_initial_last_action_valid",
        "window_replay_initial_last_action",
    )
    missing = [name for name in required if name not in data.files]
    if missing:
        raise ValueError(f"Missing window replay state arrays: {missing}")
    valid = np.asarray(data["window_replay_state_valid"], dtype=np.int8)
    qpos = np.asarray(data["window_replay_initial_qpos"], dtype=np.float32)
    qvel = np.asarray(data["window_replay_initial_qvel"], dtype=np.float32)
    last_action_valid = np.asarray(
        data["window_replay_initial_last_action_valid"],
        dtype=np.int8,
    )
    last_action = np.asarray(
        data["window_replay_initial_last_action"],
        dtype=np.float32,
    )
    if valid.shape != (expected_chunks,):
        raise ValueError(
            "Malformed window replay state valid shape "
            f"{valid.shape}; expected ({expected_chunks},)."
        )
    if qpos.shape != (expected_chunks, QPOS_DIM):
        raise ValueError(
            "Malformed window replay state qpos shape "
            f"{qpos.shape}; expected ({expected_chunks}, {QPOS_DIM})."
        )
    if qvel.shape != (expected_chunks, QVEL_DIM):
        raise ValueError(
            "Malformed window replay state qvel shape "
            f"{qvel.shape}; expected ({expected_chunks}, {QVEL_DIM})."
        )
    if last_action_valid.shape != (expected_chunks,):
        raise ValueError(
            "Malformed window replay state last-action valid shape "
            f"{last_action_valid.shape}; expected ({expected_chunks},)."
        )
    if last_action.shape != (expected_chunks, QPOS_DIM - 7):
        raise ValueError(
            "Malformed window replay state last-action shape "
            f"{last_action.shape}; expected ({expected_chunks}, {QPOS_DIM - 7})."
        )
    history_names = _window_replay_history_names(data)
    _validate_window_history_state_shapes(data, history_names, expected_chunks=expected_chunks)
    states: list[G1WbcWindowReplayState | None] = []
    for index, is_valid in enumerate(valid.tolist()):
        if not is_valid:
            states.append(None)
            continue
        history_state = _load_window_history_state(
            data,
            history_names,
            index=index,
            device=device,
        )
        states.append(
            G1WbcWindowReplayState(
                initial_qpos=torch.tensor(qpos[index], dtype=torch.float32, device=device),
                initial_qvel=torch.tensor(qvel[index], dtype=torch.float32, device=device),
                initial_last_action=(
                    torch.tensor(
                        last_action[index],
                        dtype=torch.float32,
                        device=device,
                    )
                    if bool(last_action_valid[index])
                    else None
                ),
                initial_history_state=history_state,
            )
        )
    return states


def _window_replay_history_names(data: np.lib.npyio.NpzFile) -> list[str]:
    prefix = "window_replay_history__"
    suffix = "__valid"
    names = []
    for name in data.files:
        if name.startswith(prefix) and name.endswith(suffix):
            names.append(name[len(prefix) : -len(suffix)])
    return sorted(names)


def _validate_window_history_state_shapes(
    data: np.lib.npyio.NpzFile,
    names: list[str],
    *,
    expected_chunks: int,
) -> None:
    for name in names:
        required = (
            f"window_replay_history__{name}__valid",
            f"window_replay_history__{name}__pointer",
            f"window_replay_history__{name}__num_pushes",
        )
        missing = [field for field in required if field not in data.files]
        if missing:
            raise ValueError(f"Missing window replay history arrays: {missing}")
        for field in required:
            value = np.asarray(data[field])
            if value.shape[:1] != (expected_chunks,):
                raise ValueError(
                    "Malformed window replay history shape "
                    f"{field}={value.shape}; expected first dimension {expected_chunks}."
                )
        buffer_name = f"window_replay_history__{name}__buffer"
        if buffer_name in data.files:
            value = np.asarray(data[buffer_name])
            if value.shape[:1] != (expected_chunks,):
                raise ValueError(
                    "Malformed window replay history shape "
                    f"{buffer_name}={value.shape}; expected first dimension {expected_chunks}."
                )


def _load_window_history_state(
    data: np.lib.npyio.NpzFile,
    names: list[str],
    *,
    index: int,
    device: str,
) -> dict[str, dict[str, torch.Tensor | int | None]] | None:
    history: dict[str, dict[str, torch.Tensor | int | None]] = {}
    for name in names:
        valid_name = f"window_replay_history__{name}__valid"
        if not bool(np.asarray(data[valid_name])[index]):
            continue
        item: dict[str, torch.Tensor | int | None] = {
            "pointer": int(np.asarray(data[f"window_replay_history__{name}__pointer"])[index]),
            "num_pushes": torch.tensor(
                np.asarray(data[f"window_replay_history__{name}__num_pushes"])[index],
                dtype=torch.long,
                device=device,
            ),
            "buffer": None,
        }
        buffer_name = f"window_replay_history__{name}__buffer"
        if buffer_name in data.files:
            item["buffer"] = torch.tensor(
                np.asarray(data[buffer_name])[index],
                dtype=torch.float32,
                device=device,
            )
        history[name] = item
    return history or None


def _tensor_chunk(
    value: np.ndarray,
    index: int,
    steps: int,
    *,
    device: str,
) -> torch.Tensor:
    if value.ndim < 2 or index >= value.shape[0] or steps > value.shape[1]:
        raise ValueError(f"Malformed executed command chunk array shape {value.shape}.")
    return torch.tensor(value[index, :steps], dtype=torch.float32, device=device)


def _npz_scalar_text(
    data: np.lib.npyio.NpzFile,
    name: str,
    *,
    default: str,
) -> str:
    if name not in data.files:
        return default
    value = np.asarray(data[name])
    return str(value.item() if value.shape == () else value.tolist())


def _npz_scalar_float(
    data: np.lib.npyio.NpzFile,
    name: str,
    *,
    default: float,
) -> float:
    if name not in data.files:
        return float(default)
    value = np.asarray(data[name])
    return float(value.item() if value.shape == () else value.reshape(-1)[0])


def _npz_scalar_int(
    data: np.lib.npyio.NpzFile,
    name: str,
) -> int:
    value = np.asarray(data[name])
    if value.size != 1:
        raise ValueError(f"{name} must be a scalar integer, got shape {value.shape}.")
    return int(value.reshape(-1)[0])


def _saved_command_chunk_step_count(chunks: list[G1WbcExecutedCommandChunk]) -> int:
    return max(int(chunk.start) + int(chunk.execute_steps) for chunk in chunks)


def _command_replay_state_source(
    chunks: list[G1WbcExecutedCommandChunk],
) -> str:
    count = _command_replay_state_chunk_count(chunks)
    if count == 0:
        return "none"
    if count == len(chunks):
        return "window_replay_state_chunks"
    return "partial_window_replay_state_chunks"


def _command_replay_state_chunk_count(
    chunks: list[G1WbcExecutedCommandChunk],
) -> int:
    return sum(1 for chunk in chunks if chunk.replay_state is not None)


def _saved_command_frame_count(path: Path) -> int:
    with np.load(path) as data:
        for key in ("refined_qpos", "command_qpos_trajectory", "qpos"):
            if key in data.files:
                return int(np.asarray(data[key]).shape[0])
    raise ValueError(f"{path} is missing refined_qpos/command_qpos_trajectory/qpos.")


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
