#!/usr/bin/env python3
"""Run the frozen Stage 0 G1 WBC MuJoCo-Warp baseline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

SPIDER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SPIDER_ROOT.parent
TESTBED_PACKAGE_ROOT = WORKSPACE_ROOT / "g1_wbc_testbed_motion_package_20260617"
DEFAULT_JUMP_MOTION = TESTBED_PACKAGE_ROOT / "input_motions" / "jump" / "motion.npz"
DEFAULT_WALK_MOTION = TESTBED_PACKAGE_ROOT / "input_motions" / "walk" / "motion.npz"
DEFAULT_REWARD_WEIGHTS = (
    TESTBED_PACKAGE_ROOT
    / "metadata"
    / "g1_wbc_reward_weights_method_specific_v14_20260612.json"
)
DEFAULT_PYTHON_EXECUTABLE = (
    SPIDER_ROOT / ".venv" / "bin" / "python"
    if (SPIDER_ROOT / ".venv" / "bin" / "python").exists()
    else Path(sys.executable)
)
BASELINE_NAME = "g1_wbc_stage0_mujoco_warp_sweetpoint"
MOTIONS = ("jump", "walk")
SEEDS = (0, 1, 2)
SWEETPOINT_ARGS = (
    "--method",
    "g1_wbc_joint_global",
    "--mpc-backend",
    "mujoco_warp",
    "--save-rollout",
    "--max-steps",
    "800",
    "--mpc-samples",
    "512",
    "--mpc-iterations",
    "2",
    "--mpc-planning-horizon-steps",
    "40",
    "--mpc-control-steps",
    "20",
    "--mpc-sampling-mode",
    "knot",
    "--mpc-knot-count",
    "8",
    "--mpc-temperature",
    "0.7",
    "--mpc-root-pos-sigma",
    "0.04",
    "--mpc-root-rot-sigma",
    "0.10",
    "--mpc-joint-sigma",
    "0.18",
    "--mpc-smooth-passes",
    "0",
    "--mpc-command-reg-weight",
    "0.0",
    "--mpc-command-smooth-weight",
    "0.0",
    "--mpc-guided-candidate",
    "--mpc-acceptance-gate",
)


@dataclass(frozen=True)
class Stage0Command:
    """One Stage 0 evaluate.py invocation."""

    motion_name: str
    motion: str
    seed: int
    output_dir: str
    argv: list[str]
    command_text: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse runner arguments without importing the SPIDER runtime."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jump-motion", type=Path, default=DEFAULT_JUMP_MOTION)
    parser.add_argument("--walk-motion", type=Path, default=DEFAULT_WALK_MOTION)
    parser.add_argument("--motion-type", default="isaaclab")
    parser.add_argument(
        "--checkpoint",
        default="bc",
        help="Checkpoint alias, directory, or .pt file passed to evaluate.py.",
    )
    parser.add_argument(
        "--reward-weights",
        type=Path,
        default=DEFAULT_REWARD_WEIGHTS,
        help="Reward-weight JSON passed to evaluate.py as --mpc-reward-weights.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-executable", default=str(DEFAULT_PYTHON_EXECUTABLE))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest without running evaluate.py.",
    )
    return parser.parse_args(argv)


def build_stage0_commands(args: argparse.Namespace) -> list[Stage0Command]:
    """Build the six frozen Stage 0 evaluate.py commands."""

    output_root = args.output_dir.expanduser().resolve()
    reward_weights = args.reward_weights.expanduser().resolve()
    motions = {
        "jump": args.jump_motion.expanduser().resolve(),
        "walk": args.walk_motion.expanduser().resolve(),
    }
    commands: list[Stage0Command] = []
    for motion_name in MOTIONS:
        for seed in SEEDS:
            run_output_dir = output_root / motion_name / f"seed_{seed}"
            command = [
                args.python_executable,
                "-m",
                "spider.tasks.g1_wbc.evaluate",
                "--motion",
                str(motions[motion_name]),
                "--motion-type",
                str(args.motion_type),
                "--checkpoint",
                str(args.checkpoint),
                "--device",
                args.device,
                "--output-dir",
                str(run_output_dir),
                "--seed",
                str(seed),
                *SWEETPOINT_ARGS,
                "--mpc-reward-weights",
                str(reward_weights),
            ]
            commands.append(
                Stage0Command(
                    motion_name=motion_name,
                    motion=str(motions[motion_name]),
                    seed=seed,
                    output_dir=str(run_output_dir),
                    argv=command,
                    command_text=shlex.join(command),
                )
            )
    return commands


def validate_input_paths(args: argparse.Namespace) -> tuple[str, ...]:
    """Return missing formal Stage 0 inputs before writing any manifest."""

    missing: list[str] = []
    for label, path in (
        ("jump motion", args.jump_motion),
        ("walk motion", args.walk_motion),
        ("reward weights", args.reward_weights),
    ):
        if not Path(path).expanduser().is_file():
            missing.append(f"{label}: {Path(path).expanduser()}")
    checkpoint = str(args.checkpoint)
    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.is_absolute() or checkpoint_path.exists():
        if not checkpoint_path.is_file() and not checkpoint_path.is_dir():
            missing.append(f"checkpoint: {checkpoint_path}")
    return tuple(missing)


def validate_runtime_environment(args: argparse.Namespace) -> tuple[str, ...]:
    """Return runtime environment errors that would make formal Stage 0 invalid."""

    if args.dry_run or not str(args.device).startswith("cuda"):
        return ()
    visible = tuple(
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    )
    errors: list[str] = []
    if len(visible) != 1:
        errors.append(
            "single GPU visibility: CUDA_VISIBLE_DEVICES must contain exactly "
            f"one GPU for real Stage0 CUDA runs; got {visible or '<unset>'}"
        )
    if str(args.device) not in {"cuda", "cuda:0"}:
        errors.append(
            "single GPU visibility: use --device cuda:0 after narrowing "
            f"CUDA_VISIBLE_DEVICES to one GPU; got {args.device}"
        )
    return tuple(errors)


def validate_checkpoint_format(args: argparse.Namespace) -> tuple[str, ...]:
    """Return errors for explicit checkpoints incompatible with the WBC actor."""

    checkpoint = Path(str(args.checkpoint)).expanduser()
    if checkpoint.is_dir():
        candidates = sorted(checkpoint.glob("model_*.pt"))
        if not candidates:
            return (f"checkpoint format: no model_*.pt checkpoint found under {checkpoint}",)
        checkpoint = candidates[-1]
    elif not checkpoint.exists():
        return ()
    if checkpoint.suffix != ".pt":
        return ()

    try:
        import torch

        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
    except Exception as exc:  # pragma: no cover - defensive around third-party I/O
        return (f"checkpoint format: unable to inspect {checkpoint}: {exc}",)

    if not isinstance(payload, dict):
        return (f"checkpoint format: {checkpoint} is not a dict checkpoint",)
    state_dict = payload.get("actor_state_dict", payload)
    if not isinstance(state_dict, dict):
        return (f"checkpoint format: {checkpoint} has no actor state dict",)
    keys = {str(key) for key in state_dict}
    has_obs_normalizer = {
        "obs_normalizer._mean",
        "obs_normalizer._std",
    }.issubset(keys)
    has_mlp = any(key.startswith("mlp.") for key in keys)
    if not has_obs_normalizer or not has_mlp:
        return (
            "checkpoint format: expected a WBC MLP actor checkpoint with "
            f"obs_normalizer.* and mlp.* weights; got {checkpoint}",
        )
    return ()


def attach_artifact_paths(row: dict[str, Any]) -> dict[str, Any]:
    """Attach known evaluate.py artifact paths, using None for missing files."""

    output_dir = Path(row["output_dir"]).expanduser()
    artifacts = {
        "metrics_json": output_dir / "metrics.json",
        "rollout_npz": output_dir / "rollout.npz",
        "mpc_command_npz": output_dir / "mpc_command.npz",
    }
    row = dict(row)
    row["artifacts"] = {
        key: str(path.resolve()) if path.exists() else None
        for key, path in artifacts.items()
    }
    row["artifact_mtime_ns"] = {
        key: int(path.stat().st_mtime_ns)
        for key, path in artifacts.items()
        if path.exists()
    }
    return row


def run_command(command: Stage0Command) -> dict[str, Any]:
    """Run one Stage 0 command and return captured subprocess metadata."""

    command_start_time_ns = time.time_ns()
    result = subprocess.run(
        command.argv,
        cwd=SPIDER_ROOT,
        env=None,
        capture_output=True,
        text=True,
        check=False,
    )
    row: dict[str, Any] = {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command_start_time_ns": int(command_start_time_ns),
    }
    metrics_path = Path(command.output_dir) / "metrics.json"
    if metrics_path.is_file():
        payload = json.loads(metrics_path.read_text())
        metrics = payload.get("metrics", {})
        mpc = payload.get("mpc", {})
        row["metrics"] = metrics
        row["mpc_accepted"] = bool(mpc.get("accepted", True))
        row["accepted_windows"] = int(mpc.get("accepted_windows", mpc.get("num_windows", -1)))
        row["mpc_used_baseline_fallback"] = bool(mpc.get("used_baseline_fallback", False))
        row["num_steps"] = int(metrics.get("num_steps", -1))
        if isinstance(mpc.get("steady_state_wall_time_sec"), (int, float)):
            row["steady_state_wall_time_sec"] = float(mpc["steady_state_wall_time_sec"])
        if isinstance(mpc.get("runtime_visible_devices"), list):
            row["runtime_visible_devices"] = [
                str(value)
                for value in mpc["runtime_visible_devices"]
            ]
        if isinstance(mpc.get("runtime_gpu_name"), str):
            row["runtime_gpu_name"] = mpc["runtime_gpu_name"]
    return row


def write_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Write baseline_manifest.json and return its path."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "baseline_name": BASELINE_NAME,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "motions": list(MOTIONS),
        "seeds": list(SEEDS),
        "rows": rows,
    }
    manifest_path = output_dir / "baseline_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    """Run the Stage 0 baseline or emit its dry-run manifest."""

    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    missing_inputs = validate_input_paths(args)
    if missing_inputs:
        for missing in missing_inputs:
            print(f"missing input: {missing}", file=sys.stderr)
        return 2
    runtime_errors = validate_runtime_environment(args)
    if runtime_errors:
        for error in runtime_errors:
            print(f"invalid runtime: {error}", file=sys.stderr)
        return 2
    checkpoint_errors = validate_checkpoint_format(args)
    if checkpoint_errors:
        for error in checkpoint_errors:
            print(f"invalid input: {error}", file=sys.stderr)
        return 2
    commands = build_stage0_commands(args)
    rows: list[dict[str, Any]] = []
    worst_returncode = 0
    for command in commands:
        row = asdict(command)
        row["dry_run"] = args.dry_run
        if args.dry_run:
            row.update({"status": "dry_run", "returncode": None, "stdout": None, "stderr": None})
        else:
            Path(command.output_dir).mkdir(parents=True, exist_ok=True)
            execution = run_command(command)
            row.update(execution)
            row["status"] = "ok" if execution["returncode"] == 0 else "failed"
            if execution["returncode"] != 0 and worst_returncode == 0:
                worst_returncode = int(execution["returncode"])
        rows.append(attach_artifact_paths(row))

    manifest_path = write_manifest(output_dir, rows)
    print(str(manifest_path))
    return worst_returncode


if __name__ == "__main__":
    raise SystemExit(main())
