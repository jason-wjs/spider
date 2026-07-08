"""Diagnose G1 WBC saved-command replay drift with window carry-state ablations."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from spider.tasks.g1_wbc.motion import load_motion, validate_motion_dims
from spider.tasks.g1_wbc.mpc import G1WbcMpcConfig, optimize_mpc_command
from spider.tasks.g1_wbc.policy import load_wbc_actor
from spider.tasks.g1_wbc.rollout import RolloutResult, WbcRolloutConfig, run_command_rollout

DEFAULT_FRAMES = (0, 1, 2, 3, 4, 20)
DEFAULT_THRESHOLDS = (1.0e-5, 1.0e-3, 1.0e-2, 1.0e-1)
DEFAULT_MODES = (
    "state_all",
    "drop_history",
    "drop_prev_action",
    "command_qvel",
    "command_only",
)


def main() -> None:
    args = _parse_args()
    source = _source_payload(args.source_run)
    device = str(args.device or source.get("device") or "cuda:0")
    motion_path = Path(args.motion or source["motion"]).expanduser().resolve()
    checkpoint = Path(args.checkpoint or source["checkpoint"]).expanduser().resolve()
    motion_type = str(args.motion_type or source.get("motion_type") or "isaaclab")
    method = str(args.method or source.get("method") or "g1_wbc_joint_global")

    mpc_config = _mpc_config_from_source(source, method=method)
    if args.seed is not None:
        mpc_config.seed = int(args.seed)
    capture_steps = int(args.window_start) + int(args.windows) * int(
        mpc_config.control_steps
    )
    rollout_config = _rollout_config_from_source(
        source,
        device=device,
        max_steps=capture_steps,
    )
    replay_config = WbcRolloutConfig(
        **{
            **rollout_config.__dict__,
            "num_envs": 1,
            "use_cuda_graph": bool(args.use_cuda_graph),
            "serial_warp_launches": bool(args.serial_warp_launches),
        }
    )

    motion = load_motion(motion_path, motion_type=motion_type, device=device)
    validate_motion_dims(motion)
    actor = load_wbc_actor(checkpoint, device=device)
    captured: list[dict[str, Any]] = []

    def hook(payload: dict[str, Any]) -> None:
        if int(payload["start"]) < int(args.window_start):
            return
        if len(captured) >= int(args.windows):
            return
        captured.append(payload)

    optimize_mpc_command(
        motion,
        actor,
        rollout_config,
        mpc_config,
        diagnostic_window_hook=hook,
    )

    reports = []
    sidecar_dir = args.sidecar_dir
    if sidecar_dir is not None:
        sidecar_dir.mkdir(parents=True, exist_ok=True)
    for item in captured:
        reports.append(
            _diagnose_window(
                item,
                actor=actor,
                replay_config=replay_config,
                modes=tuple(args.modes),
                frames=tuple(args.frames),
                thresholds=tuple(args.thresholds),
                sidecar_dir=sidecar_dir,
            )
        )

    output = {
        "source_run": str(args.source_run.expanduser().resolve()),
        "motion": str(motion_path),
        "checkpoint": str(checkpoint),
        "device": device,
        "capture_steps": capture_steps,
        "window_start": int(args.window_start),
        "requested_windows": int(args.windows),
        "captured_windows": len(captured),
        "modes": list(args.modes),
        "windows": reports,
    }
    payload = json.dumps(output, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)


def _diagnose_window(
    item: dict[str, Any],
    *,
    actor,
    replay_config: WbcRolloutConfig,
    modes: tuple[str, ...],
    frames: tuple[int, ...],
    thresholds: tuple[float, ...],
    sidecar_dir: Path | None,
) -> dict[str, Any]:
    original = item["rollout"]
    report: dict[str, Any] = {
        "window_index": int(item["window_index"]),
        "start": int(item["start"]),
        "horizon": int(item["horizon"]),
        "execute_steps": int(item["execute_steps"]),
        "modes": {},
    }
    if sidecar_dir is not None:
        _save_rollout_npz(sidecar_dir / f"window_{int(item['start']):04d}_original.npz", original)
    for mode in modes:
        replay = _run_mode(mode, item, actor=actor, replay_config=replay_config)
        if sidecar_dir is not None:
            _save_rollout_npz(
                sidecar_dir / f"window_{int(item['start']):04d}_{mode}.npz",
                replay,
            )
        report["modes"][mode] = _compare_rollouts(
            replay,
            original,
            frames=frames,
            thresholds=thresholds,
        )
    return report


def _run_mode(
    mode: str,
    item: dict[str, Any],
    *,
    actor,
    replay_config: WbcRolloutConfig,
) -> RolloutResult:
    command = item["command"]
    initial_qpos = item["initial_qpos"]
    initial_qvel = item["initial_qvel"]
    initial_last_action = item["initial_last_action"]
    initial_history_state = item["initial_history_state"]
    if mode == "state_all":
        pass
    elif mode == "drop_history":
        initial_history_state = None
    elif mode == "drop_prev_action":
        initial_last_action = torch.zeros_like(item["final_last_action"][0])
    elif mode == "command_qvel":
        initial_qvel = command.qvel_trajectory[0, 0].detach().clone()
    elif mode == "command_only":
        initial_qpos = command.qpos_trajectory[0, 0].detach().clone()
        initial_qvel = command.qvel_trajectory[0, 0].detach().clone()
        initial_last_action = None
        initial_history_state = None
    else:
        raise ValueError(f"Unknown replay mode: {mode}")
    return run_command_rollout(
        command,
        actor,
        replace(replay_config, max_steps=int(item["execute_steps"])),
        initial_qpos=initial_qpos,
        initial_qvel=initial_qvel,
        initial_last_action=initial_last_action,
        initial_history_state=initial_history_state,
        ref_start=int(item["start"]),
    )


def _compare_rollouts(
    lhs: RolloutResult,
    rhs: RolloutResult,
    *,
    frames: tuple[int, ...],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, lhs_value, rhs_value in _comparison_arrays(lhs, rhs):
        out[name] = _series_summary(
            _norm_series(lhs_value, rhs_value),
            frames=frames,
            thresholds=thresholds,
        )
    return out


def _comparison_arrays(lhs: RolloutResult, rhs: RolloutResult):
    yield "root_pos", _squeeze_env(lhs.qpos)[..., :3], _squeeze_env(rhs.qpos)[..., :3]
    for name in (
        "qpos",
        "qvel",
        "actions",
        "controls",
        "contact_indicator",
        "contact_force",
        "floor_contact_indicator",
        "floor_contact_force",
    ):
        lhs_value = getattr(lhs, name, None)
        rhs_value = getattr(rhs, name, None)
        if lhs_value is not None and rhs_value is not None:
            yield name, _squeeze_env(lhs_value), _squeeze_env(rhs_value)


def _squeeze_env(value) -> np.ndarray:
    array = _cpu_np(value)
    if array.ndim >= 3 and int(array.shape[1]) == 1:
        return array[:, 0]
    return array


def _norm_series(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    length = min(int(lhs.shape[0]), int(rhs.shape[0]))
    if length < 1:
        return np.zeros((0,), dtype=np.float64)
    lhs_flat = np.asarray(lhs[:length], dtype=np.float64).reshape(length, -1)
    rhs_flat = np.asarray(rhs[:length], dtype=np.float64).reshape(length, -1)
    return np.linalg.norm(lhs_flat - rhs_flat, axis=1)


def _series_summary(
    diff: np.ndarray,
    *,
    frames: tuple[int, ...],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    return {
        "length": int(diff.shape[0]),
        "mean": float(np.mean(diff)) if diff.size else 0.0,
        "max": float(np.max(diff)) if diff.size else 0.0,
        "first_exceed": {
            f"{float(threshold):.0e}": _first_exceed(diff, threshold)
            for threshold in thresholds
        },
        "frames": {
            str(frame): float(diff[int(frame)])
            for frame in frames
            if 0 <= int(frame) < int(diff.shape[0])
        },
    }


def _first_exceed(diff: np.ndarray, threshold: float) -> int | None:
    hits = np.flatnonzero(diff > float(threshold))
    if hits.size < 1:
        return None
    return int(hits[0])


def _save_rollout_npz(path: Path, rollout: RolloutResult) -> None:
    np.savez_compressed(
        path,
        qpos=_cpu_np(rollout.qpos),
        qvel=_cpu_np(rollout.qvel),
        actions=_cpu_np(rollout.actions),
        controls=_cpu_np(rollout.controls),
        contact_indicator=_cpu_np(rollout.contact_indicator),
        contact_force=_cpu_np(rollout.contact_force),
        floor_contact_indicator=_cpu_np(rollout.floor_contact_indicator),
        floor_contact_force=_cpu_np(rollout.floor_contact_force),
        ref_indices=_cpu_np(rollout.ref_indices),
    )


def _cpu_np(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _source_payload(source_run: Path) -> dict[str, Any]:
    metrics_path = source_run.expanduser().resolve() / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json under {source_run}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _mpc_config_from_source(source: dict[str, Any], *, method: str) -> G1WbcMpcConfig:
    raw = dict(source.get("mpc", {}).get("config", {}))
    names = {field.name for field in fields(G1WbcMpcConfig)}
    kwargs = {name: raw[name] for name in names if name in raw}
    config = G1WbcMpcConfig(**kwargs)
    config.mode = method
    reward_weights = source.get("mpc", {}).get("reward_weights")
    if isinstance(reward_weights, dict):
        config.reward_weights = {str(key): float(value) for key, value in reward_weights.items()}
    return config


def _rollout_config_from_source(
    source: dict[str, Any],
    *,
    device: str,
    max_steps: int,
) -> WbcRolloutConfig:
    return WbcRolloutConfig(
        device=device,
        num_envs=1,
        max_steps=int(max_steps),
        ref_offset=int(source.get("ref_offset", 0)),
        sync_after_step=True,
        forward_after_step=True,
        use_cuda_graph=True,
        serial_warp_launches=bool(source.get("serial_warp_launches", False)),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--motion", type=Path)
    parser.add_argument("--motion-type")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--window-start", type=int, required=True)
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sidecar-dir", type=Path)
    parser.add_argument("--use-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--serial-warp-launches", action="store_true")
    parser.add_argument("--modes", nargs="*", default=list(DEFAULT_MODES))
    parser.add_argument("--frames", type=int, nargs="*", default=list(DEFAULT_FRAMES))
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=list(DEFAULT_THRESHOLDS),
    )
    args = parser.parse_args()
    if int(args.windows) < 1:
        raise ValueError("--windows must be positive")
    if int(args.window_start) < 0:
        raise ValueError("--window-start must be non-negative")
    return args


if __name__ == "__main__":
    main()
