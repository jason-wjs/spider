#!/usr/bin/env python3
"""Run the frozen Stage 0 G1 WBC MuJoCo-Warp baseline."""

from __future__ import annotations

import argparse
import hashlib
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
MODEL_BASED_ROOT = next(
    (
        candidate
        for candidate in (SPIDER_ROOT, *SPIDER_ROOT.parents)
        if (candidate / "wbc_results" / "assets").is_dir()
    ),
    SPIDER_ROOT.parent,
)
WBC_RESULTS_ROOT = MODEL_BASED_ROOT / "wbc_results"
DEFAULT_JUMP_MOTION = WBC_RESULTS_ROOT / "assets" / "motion_data" / "jump" / "motion.npz"
DEFAULT_WALK_MOTION = WBC_RESULTS_ROOT / "assets" / "motion_data" / "walk" / "motion.npz"
DEFAULT_CHECKPOINT = WBC_RESULTS_ROOT / "assets" / "checkpoints" / "model_8000.pt"
DEFAULT_REWARD_WEIGHTS = (
    WBC_RESULTS_ROOT
    / "g1_body_tracking_wbc"
    / "spider"
    / "2026-06-23-mechanism-quality-speed-wjs"
    / "configs"
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
    "--mpc-optimizer",
    "legacy",
    "--mpc-preset",
    "aggressive",
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
    "--mpc-guided-root-pos-gain",
    "0.50",
    "--mpc-guided-root-rot-gain",
    "0.50",
    "--mpc-guided-joint-gain",
    "0.50",
    "--mpc-guided-root-pos-clip",
    "0.05",
    "--mpc-guided-root-rot-clip",
    "0.12",
    "--mpc-guided-joint-clip",
    "0.35",
    "--mpc-guided-candidate",
    "--mpc-acceptance-gate",
    "--nconmax-per-env",
    "512",
    "--njmax-per-env",
    "2048",
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
        default=str(DEFAULT_CHECKPOINT),
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
    parser.add_argument(
        "--reuse-existing-ok",
        action="store_true",
        help=(
            "Reuse existing completed row artifacts in output-dir instead of "
            "rerunning that row. Incomplete or failed rows are rerun."
        ),
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
    if resolve_checkpoint_file(args.checkpoint) is None:
        missing.append(
            "checkpoint: "
            f"{Path(str(args.checkpoint)).expanduser()} "
            "(expected an existing .pt file or directory with model_*.pt)"
        )
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
    if len(visible) == 1:
        contention_errors = validate_visible_gpu_is_idle(visible[0])
        errors.extend(contention_errors)
    return tuple(errors)


def validate_visible_gpu_is_idle(visible_gpu: str) -> tuple[str, ...]:
    """Return errors when the only visible GPU already has compute processes."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--id",
                str(visible_gpu),
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return (f"GPU contention: unable to inspect GPU {visible_gpu}: {exc}",)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        return (f"GPU contention: unable to inspect GPU {visible_gpu}: {detail}",)
    processes = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    if not processes:
        return ()
    process_list = "; ".join(processes[:4])
    extra = "" if len(processes) <= 4 else f"; +{len(processes) - 4} more"
    return (
        "GPU contention: formal Stage0 requires the single visible GPU to have "
        f"no existing compute processes; GPU {visible_gpu} has {process_list}{extra}",
    )


def validate_checkpoint_format(args: argparse.Namespace) -> tuple[str, ...]:
    """Return errors for explicit checkpoints incompatible with the WBC actor."""

    checkpoint = resolve_checkpoint_file(args.checkpoint)
    if checkpoint is None:
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
    row["artifact_sha256"] = {
        key: file_sha256(path)
        for key, path in artifacts.items()
        if path.exists()
    }
    return row


def file_sha256(path: Path) -> str:
    """Return a SHA256 hex digest for a local artifact file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Build reproducibility metadata for the Stage 0 baseline manifest."""

    jump_motion = args.jump_motion.expanduser().resolve()
    walk_motion = args.walk_motion.expanduser().resolve()
    reward_weights = args.reward_weights.expanduser().resolve()
    checkpoint = resolve_checkpoint_file(args.checkpoint)
    input_paths = {
        "jump_motion": str(jump_motion),
        "walk_motion": str(walk_motion),
        "checkpoint": str(checkpoint) if checkpoint is not None else str(args.checkpoint),
        "reward_weights": str(reward_weights),
    }
    input_hashes = {
        "jump_motion": file_sha256(jump_motion),
        "walk_motion": file_sha256(walk_motion),
        "checkpoint": file_sha256(checkpoint) if checkpoint is not None else None,
        "reward_weights": file_sha256(reward_weights),
    }
    return {
        "provenance": {
            "worktree_path": str(SPIDER_ROOT),
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_status_short": _git_output("status", "--short"),
            "python_executable": str(Path(args.python_executable).expanduser()),
            "device": str(args.device),
        },
        "input_paths": input_paths,
        "input_sha256": input_hashes,
        "versions": {
            "python": sys.version.split()[0],
        },
    }


def resolve_checkpoint_file(checkpoint: str | Path) -> Path | None:
    path = Path(str(checkpoint)).expanduser()
    if path.is_dir():
        candidates = sorted(path.glob("model_*.pt"))
        return candidates[-1].resolve() if candidates else None
    if path.is_file():
        return path.resolve()
    return None


def _git_output(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=SPIDER_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


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


def load_existing_ok_row(command: Stage0Command) -> dict[str, Any] | None:
    """Return reusable row execution metadata when all row artifacts are complete."""

    output_dir = Path(command.output_dir)
    metrics_path = output_dir / "metrics.json"
    rollout_path = output_dir / "rollout.npz"
    command_path = output_dir / "mpc_command.npz"
    if not (
        metrics_path.is_file()
        and rollout_path.is_file()
        and command_path.is_file()
    ):
        return None
    try:
        payload = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    metrics = payload.get("metrics")
    mpc = payload.get("mpc")
    if not isinstance(metrics, dict) or not isinstance(mpc, dict):
        return None
    if not _existing_row_provenance_matches(command, payload):
        return None
    try:
        num_steps = int(metrics.get("num_steps", -1))
        accepted_windows = int(mpc.get("accepted_windows", mpc.get("num_windows", -1)))
    except (TypeError, ValueError):
        return None
    if num_steps != 800:
        return None
    if not bool(mpc.get("accepted", True)):
        return None
    if accepted_windows != 40:
        return None
    if bool(mpc.get("used_baseline_fallback", False)):
        return None

    row: dict[str, Any] = {
        "returncode": 0,
        "stdout": None,
        "stderr": None,
        "metrics": metrics,
        "mpc_accepted": True,
        "accepted_windows": accepted_windows,
        "mpc_used_baseline_fallback": False,
        "num_steps": num_steps,
        "reused_existing": True,
    }
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


def _existing_row_provenance_matches(
    command: Stage0Command,
    payload: dict[str, Any],
) -> bool:
    """Return whether an existing metrics payload belongs to this command."""

    expected = _command_expected_provenance(command)
    for key in ("motion", "checkpoint", "method"):
        value = expected.get(key)
        if value is None:
            continue
        actual = payload.get(key)
        if actual is None:
            return False
        if str(actual) != str(value):
            return False
    mpc = payload.get("mpc", {})
    if not isinstance(mpc, dict):
        return False
    for key in ("mpc_backend", "mpc_optimizer"):
        value = expected.get(key)
        if value is None:
            continue
        actual = mpc.get(key)
        if actual is None or str(actual) != str(value):
            return False
    return True


def _command_expected_provenance(command: Stage0Command) -> dict[str, str]:
    """Extract metrics provenance fields expected from an evaluate.py command."""

    values = {
        "motion": command.motion,
        "checkpoint": _argv_value(command.argv, "--checkpoint"),
        "method": _argv_value(command.argv, "--method"),
        "mpc_backend": _argv_value(command.argv, "--mpc-backend"),
        "mpc_optimizer": _argv_value(command.argv, "--mpc-optimizer"),
    }
    return {
        key: str(value)
        for key, value in values.items()
        if value is not None
    }


def _argv_value(argv: list[str], flag: str) -> str | None:
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    value_index = index + 1
    if value_index >= len(argv):
        return None
    return str(argv[value_index])


def write_manifest(
    output_dir: Path,
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
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
    if metadata:
        payload.update(metadata)
    baseline_summary = build_baseline_summary(rows)
    if baseline_summary:
        payload.update(baseline_summary)
    manifest_path = output_dir / "baseline_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return manifest_path


def build_baseline_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return frozen baseline envelopes for a passing formal Stage 0 run."""

    from spider.tasks.g1_wbc.acceptance import evaluate_baseline_group

    baseline_envelopes: dict[str, dict[str, dict[str, float]]] = {}
    promoted_seeds: dict[str, int | None] = {}
    gate_failures: dict[str, list[str]] = {}
    for motion in MOTIONS:
        group = [row for row in rows if row.get("motion_name") == motion]
        gate = evaluate_baseline_group(motion, group)
        if gate.passed:
            baseline_envelopes[motion] = gate.envelope
            promoted_seeds[motion] = gate.promoted_seed
        else:
            gate_failures[motion] = list(gate.failures)
    if gate_failures:
        return {"baseline_gate_failures": gate_failures}
    return {
        "baseline_envelopes": baseline_envelopes,
        "promoted_seeds": promoted_seeds,
    }


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
    checkpoint = resolve_checkpoint_file(args.checkpoint)
    if checkpoint is not None:
        args = argparse.Namespace(**vars(args))
        args.checkpoint = str(checkpoint)
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
            execution = (
                load_existing_ok_row(command)
                if args.reuse_existing_ok
                else None
            )
            if execution is None:
                execution = run_command(command)
            row.update(execution)
            row["status"] = "ok" if execution["returncode"] == 0 else "failed"
            if execution["returncode"] != 0 and worst_returncode == 0:
                worst_returncode = int(execution["returncode"])
        rows.append(attach_artifact_paths(row))

    manifest_path = write_manifest(
        output_dir,
        rows,
        metadata=build_manifest_metadata(args),
    )
    print(str(manifest_path))
    if not args.dry_run and worst_returncode == 0:
        baseline_summary = build_baseline_summary(rows)
        if "baseline_gate_failures" in baseline_summary:
            return 1
    return worst_returncode


if __name__ == "__main__":
    raise SystemExit(main())
