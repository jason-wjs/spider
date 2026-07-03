#!/usr/bin/env python3
"""Profile one G1 WBC evaluate.py run from outside the MPC internals."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

SPIDER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = SPIDER_ROOT.parent
TESTBED_PACKAGE_ROOT = WORKSPACE_ROOT / "g1_wbc_testbed_motion_package_20260617"
DEFAULT_MOTION = TESTBED_PACKAGE_ROOT / "input_motions" / "jump" / "motion.npz"
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

MPC_METHODS = ("g1_wbc_joint_global", "g1_wbc_joint", "g1_wbc_ee")
METHODS = ("no_mpc", *MPC_METHODS)
MPC_PRESETS = ("aggressive", "conservative", "explore", "rootrot", "wide")


def main(argv: list[str] | None = None) -> int:
    """Run the profiler CLI."""

    args = parse_args(argv)
    missing_inputs = validate_input_paths(args)
    if missing_inputs:
        for missing in missing_inputs:
            print(f"missing input: {missing}", file=sys.stderr)
        return 2
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluate_output_dir = output_dir / "evaluate"
    log_path = output_dir / "evaluate.log"
    command = build_evaluate_command(args, evaluate_output_dir)
    profile_path = profile_json_path(output_dir, args.motion, args.method)

    profile = build_base_profile(args, command, evaluate_output_dir, log_path)
    if args.execute:
        execution = run_profiled_command(
            command,
            cwd=SPIDER_ROOT,
            env_delta=pythonpath_env_delta(SPIDER_ROOT),
            log_path=log_path,
            device=args.device,
            poll_interval_sec=args.poll_interval_sec,
            timeout_sec=args.timeout_sec,
        )
        log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
        evaluate_payload = load_evaluate_payload(evaluate_output_dir, log_text)
        profile.update(execution)
        profile["evaluate_payload"] = evaluate_payload
        profile["metrics"] = extract_metrics(evaluate_payload)
        profile["mpc"] = extract_mpc(evaluate_payload, fallback=profile["mpc"])
    else:
        profile.update(
            {
                "status": "dry_run",
                "returncode": None,
                "timed_out": False,
                "wall_time_sec": 0.0,
                "steady_state_wall_time_sec": 0.0,
                "compile_init_wall_time_sec": None,
                "started_at": None,
                "finished_at": None,
                "cuda_memory": cuda_memory_unavailable(args.device, "dry-run"),
                "evaluate_payload": None,
            }
        )

    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n")
    print(str(profile_path))
    return int(profile.get("returncode") or 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments without importing SPIDER runtime modules."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion",
        type=Path,
        default=DEFAULT_MOTION,
        help="Reference motion npz.",
    )
    parser.add_argument(
        "--motion-type",
        default="auto",
        choices=("auto", "mujoco", "isaaclab"),
        help="Input npz semantic ordering passed through to evaluate.py.",
    )
    parser.add_argument("--method", choices=METHODS, default="g1_wbc_joint_global")
    parser.add_argument(
        "--checkpoint",
        default="bc",
        help="Checkpoint alias, directory, or .pt file passed to evaluate.py.",
    )
    parser.add_argument("--samples", type=positive_int, default=8192)
    parser.add_argument("--iterations", type=positive_int, default=2)
    parser.add_argument("--horizon", type=positive_int, default=80)
    parser.add_argument("--control", "--control-steps", type=positive_int, default=20)
    parser.add_argument("--knot-count", type=positive_int, default=8)
    parser.add_argument("--root-pos-sigma", type=positive_float, default=0.08)
    parser.add_argument("--root-rot-sigma", type=positive_float, default=0.18)
    parser.add_argument("--joint-sigma", type=positive_float, default=0.28)
    parser.add_argument(
        "--mpc-warm-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable shifted receding-plan warm start in evaluate.py.",
    )
    parser.add_argument(
        "--mpc-warm-start-source",
        choices=("best", "mean"),
        default="best",
    )
    parser.add_argument("--mpc-warm-start-decay", type=positive_float, default=1.0)
    parser.add_argument("--mpc-command-reg-weight", type=nonnegative_float, default=0.0)
    parser.add_argument("--mpc-command-smooth-weight", type=nonnegative_float, default=0.0)
    parser.add_argument(
        "--sampling-mode",
        choices=("full", "knot"),
        default="knot",
        help="MPC sampling mode passed as --mpc-sampling-mode.",
    )
    parser.add_argument(
        "--mpc-preset",
        choices=MPC_PRESETS,
        default="aggressive",
        help="MPC preset passed through before explicit overrides.",
    )
    parser.add_argument(
        "--reward-weights",
        type=Path,
        default=DEFAULT_REWARD_WEIGHTS,
        help="Optional method-keyed or flat reward-weight JSON passed to evaluate.py.",
    )
    parser.add_argument("--max-steps", type=positive_int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mpc-backend",
        choices=("mujoco_warp", "mjx"),
        default="mujoco_warp",
    )
    parser.add_argument("--save-rollout", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-executable", default=str(DEFAULT_PYTHON_EXECUTABLE))
    parser.add_argument("--timeout-sec", type=positive_int, default=None)
    parser.add_argument(
        "--poll-interval-sec",
        type=positive_float,
        default=0.25,
        help="nvidia-smi polling interval while the evaluate.py process is alive.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="execute",
        action="store_false",
        help="Write the profile JSON with the command but do not run evaluate.py.",
    )
    mode.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        help="Run evaluate.py and collect wall time, metrics, and memory samples.",
    )
    parser.set_defaults(execute=False)
    return parser.parse_args(argv)


def validate_input_paths(args: argparse.Namespace) -> tuple[str, ...]:
    """Return missing evaluate.py inputs before writing a profile."""

    missing: list[str] = []
    if not args.motion.expanduser().is_file():
        missing.append(f"motion: {args.motion.expanduser()}")
    if args.method != "no_mpc" and args.reward_weights is not None:
        if not args.reward_weights.expanduser().is_file():
            missing.append(f"reward weights: {args.reward_weights.expanduser()}")
    checkpoint = str(args.checkpoint)
    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.is_absolute() or checkpoint_path.exists():
        if not checkpoint_path.is_file() and not checkpoint_path.is_dir():
            missing.append(f"checkpoint: {checkpoint_path}")
    return tuple(missing)


def build_evaluate_command(args: argparse.Namespace, evaluate_output_dir: Path) -> list[str]:
    """Build the evaluate.py subprocess argv."""

    command = [
        args.python_executable,
        "-m",
        "spider.tasks.g1_wbc.evaluate",
        "--motion",
        str(args.motion.expanduser().resolve()),
        "--motion-type",
        args.motion_type,
        "--checkpoint",
        str(args.checkpoint),
        "--method",
        args.method,
        "--device",
        args.device,
        "--output-dir",
        str(evaluate_output_dir),
        "--seed",
        str(args.seed),
        "--mpc-backend",
        args.mpc_backend,
    ]
    if args.save_rollout:
        command.append("--save-rollout")
    if args.max_steps is not None:
        command.extend(["--max-steps", str(args.max_steps)])
    if args.method == "no_mpc":
        command.extend(["--num-envs", "1"])
    else:
        command.extend(
            [
                "--mpc-preset",
                args.mpc_preset,
                "--mpc-samples",
                str(args.samples),
                "--mpc-iterations",
                str(args.iterations),
                "--mpc-planning-horizon-steps",
                str(args.horizon),
                "--mpc-control-steps",
                str(args.control),
                "--mpc-sampling-mode",
                args.sampling_mode,
                "--mpc-knot-count",
                str(args.knot_count),
                "--mpc-root-pos-sigma",
                str(args.root_pos_sigma),
                "--mpc-root-rot-sigma",
                str(args.root_rot_sigma),
                "--mpc-joint-sigma",
                str(args.joint_sigma),
                "--mpc-smooth-passes",
                "0",
                "--mpc-command-reg-weight",
                str(args.mpc_command_reg_weight),
                "--mpc-command-smooth-weight",
                str(args.mpc_command_smooth_weight),
                "--mpc-warm-start" if args.mpc_warm_start else "--no-mpc-warm-start",
                "--mpc-warm-start-source",
                args.mpc_warm_start_source,
                "--mpc-warm-start-decay",
                str(args.mpc_warm_start_decay),
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
            ]
        )
        if args.reward_weights is not None:
            command.extend(
                [
                    "--mpc-reward-weights",
                    str(args.reward_weights.expanduser().resolve()),
                ]
            )
    return command


def build_base_profile(
    args: argparse.Namespace,
    command: list[str],
    evaluate_output_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Create the profile keys that are known before execution."""

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    mpc_config = None
    if args.method != "no_mpc":
        mpc_config = {
            "preset": args.mpc_preset,
            "num_samples": args.samples,
            "num_iterations": args.iterations,
            "planning_horizon_steps": args.horizon,
            "control_steps": args.control,
            "sampling_mode": args.sampling_mode,
            "knot_count": args.knot_count,
            "use_warm_start": args.mpc_warm_start,
            "warm_start_source": args.mpc_warm_start_source,
            "warm_start_decay": args.mpc_warm_start_decay,
            "command_reg_weight": args.mpc_command_reg_weight,
            "command_smooth_weight": args.mpc_command_smooth_weight,
            "root_pos_sigma": args.root_pos_sigma,
            "root_rot_sigma": args.root_rot_sigma,
            "joint_sigma": args.joint_sigma,
            "reward_weights": (
                str(args.reward_weights.expanduser().resolve())
                if args.reward_weights is not None
                else None
            ),
        }
    return {
        "schema_version": 1,
        "created_at": now,
        "dry_run": not args.execute,
        "motion": str(args.motion.expanduser().resolve()),
        "motion_type": args.motion_type,
        "method": args.method,
        "seed": args.seed,
        "mpc_backend": args.mpc_backend,
        "checkpoint": str(args.checkpoint),
        "device": args.device,
        "max_steps": args.max_steps,
        "python_executable": args.python_executable,
        "cwd": str(SPIDER_ROOT),
        "command": command,
        "command_text": shlex.join(command),
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "evaluate_output_dir": str(evaluate_output_dir),
        "log_path": str(log_path),
        "metrics": {},
        "mpc": mpc_config,
    }


def run_profiled_command(
    command: list[str],
    *,
    cwd: Path,
    env_delta: dict[str, str],
    log_path: Path,
    device: str,
    poll_interval_sec: float,
    timeout_sec: int | None,
) -> dict[str, Any]:
    """Run a command while measuring wall time and approximate CUDA peak memory."""

    env = os.environ.copy()
    env.update(env_delta)
    tracker = CudaMemoryTracker(device, poll_interval_sec)
    started_at = datetime.now().astimezone()
    start_perf = time.perf_counter()
    timed_out = False
    deadline = None if timeout_sec is None else start_perf + timeout_sec

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        log_file.write(f"START {started_at.isoformat(timespec='seconds')}\n")
        log_file.write(shlex.join(command) + "\n\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            tracker.sample(process.pid)
            if deadline is not None and time.perf_counter() >= deadline:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(poll_interval_sec)
        tracker.sample(process.pid)
        returncode = process.wait()
        finished_at = datetime.now().astimezone()
        log_file.write(f"\nDONE {finished_at.isoformat(timespec='seconds')}\n")
        log_file.write(f"RETURNCODE {returncode}\n")

    wall_time_sec = time.perf_counter() - start_perf
    if timed_out:
        status = "timed_out"
    elif returncode == 0:
        status = "succeeded"
    else:
        status = "failed"
    return {
        "status": status,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_time_sec": wall_time_sec,
        "steady_state_wall_time_sec": wall_time_sec,
        "compile_init_wall_time_sec": None,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "cuda_memory": tracker.to_payload(),
    }


class CudaMemoryTracker:
    """Poll nvidia-smi for approximate process CUDA memory."""

    def __init__(self, device: str, poll_interval_sec: float) -> None:
        self.device = device
        self.poll_interval_sec = poll_interval_sec
        self.peak_mib: int | None = None
        self.samples = 0
        self.error: str | None = None
        self.enabled = device.lower().startswith("cuda")
        self.nvidia_smi = shutil.which("nvidia-smi") if self.enabled else None
        if self.enabled and self.nvidia_smi is None:
            self.error = "nvidia-smi not found"

    def sample(self, pid: int) -> None:
        """Record one memory sample for the evaluate.py PID."""

        if not self.enabled or self.nvidia_smi is None:
            return
        try:
            result = subprocess.run(
                [
                    self.nvidia_smi,
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.error = str(exc)
            return
        if result.returncode != 0:
            self.error = result.stderr.strip() or f"nvidia-smi exited {result.returncode}"
            return
        self.samples += 1
        current_mib = 0
        matched_pid = False
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 2:
                continue
            try:
                sample_pid = int(parts[0])
                used_mib = int(parts[1].split()[0])
            except ValueError:
                continue
            if sample_pid == pid:
                matched_pid = True
                current_mib += used_mib
        if matched_pid:
            self.peak_mib = max(self.peak_mib or 0, current_mib)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable memory summary."""

        if not self.enabled:
            return cuda_memory_unavailable(self.device, "device is not cuda")
        return {
            "available": self.peak_mib is not None,
            "device": self.device,
            "source": "nvidia-smi --query-compute-apps=pid,used_memory",
            "peak_mib": self.peak_mib,
            "peak_bytes": None
            if self.peak_mib is None
            else int(self.peak_mib * 1024 * 1024),
            "samples": self.samples,
            "poll_interval_sec": self.poll_interval_sec,
            "error": self.error,
        }


def cuda_memory_unavailable(device: str, reason: str) -> dict[str, Any]:
    """Return the memory payload shape when no CUDA peak can be sampled."""

    return {
        "available": False,
        "device": device,
        "source": "nvidia-smi --query-compute-apps=pid,used_memory",
        "peak_mib": None,
        "peak_bytes": None,
        "samples": 0,
        "poll_interval_sec": None,
        "error": reason,
    }


def load_evaluate_payload(evaluate_output_dir: Path, stdout_text: str) -> dict[str, Any] | None:
    """Load evaluate.py JSON from metrics.json, falling back to stdout parsing."""

    metrics_path = evaluate_output_dir / "metrics.json"
    if metrics_path.exists():
        try:
            return json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            pass
    return extract_last_json_payload(stdout_text)


def extract_last_json_payload(text: str) -> dict[str, Any] | None:
    """Extract the last complete top-level JSON object printed in stdout."""

    blocks: list[str] = []
    current: list[str] = []
    brace_depth = 0
    in_json = False
    for line in text.splitlines():
        stripped = line.strip()
        if not in_json:
            if not stripped.startswith("{"):
                continue
            in_json = True
            current = [line]
            brace_depth = stripped.count("{") - stripped.count("}")
        else:
            current.append(line)
            brace_depth += stripped.count("{") - stripped.count("}")
        if in_json and brace_depth <= 0:
            blocks.append("\n".join(current))
            current = []
            in_json = False
            brace_depth = 0
    for block in reversed(blocks):
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def extract_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Return the metrics mapping from an evaluate.py payload."""

    if not isinstance(payload, dict):
        return {}
    metrics = payload.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def extract_mpc(
    payload: dict[str, Any] | None,
    *,
    fallback: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return evaluate.py MPC metadata when available."""

    if isinstance(payload, dict) and isinstance(payload.get("mpc"), dict):
        return payload["mpc"]
    return fallback


def pythonpath_env_delta(spider_root: Path) -> dict[str, str]:
    """Build a PYTHONPATH override that makes the local spider package importable."""

    existing = os.environ.get("PYTHONPATH")
    value = str(spider_root)
    if existing:
        value = value + os.pathsep + existing
    return {"PYTHONPATH": value}


def profile_json_path(output_dir: Path, motion: Path, method: str) -> Path:
    """Return a timestamped profile JSON path."""

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    motion_stem = slugify(motion.expanduser().stem)
    return output_dir / f"profile_{stamp}_{method}_{motion_stem}.json"


def slugify(value: str) -> str:
    """Make a short filesystem-safe token."""

    chars = [ch.lower() if ch.isalnum() else "_" for ch in value]
    slug = "_".join(part for part in "".join(chars).split("_") if part)
    return slug or "motion"


def positive_int(value: str) -> int:
    """argparse type for positive integers."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def positive_float(value: str) -> float:
    """argparse type for positive floats."""

    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("expected a positive float")
    return parsed


def nonnegative_float(value: str) -> float:
    """argparse type for non-negative floats."""

    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("expected a non-negative float")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
