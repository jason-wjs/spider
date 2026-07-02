#!/usr/bin/env python3
"""Run the frozen Stage 0 G1 WBC MuJoCo-Warp baseline."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
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
                "auto",
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
    return row


def run_command(command: Stage0Command) -> dict[str, Any]:
    """Run one Stage 0 command and return captured subprocess metadata."""

    result = subprocess.run(
        command.argv,
        cwd=SPIDER_ROOT,
        env=None,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


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
            row["status"] = "succeeded" if execution["returncode"] == 0 else "failed"
            if execution["returncode"] != 0 and worst_returncode == 0:
                worst_returncode = int(execution["returncode"])
        rows.append(attach_artifact_paths(row))

    manifest_path = write_manifest(output_dir, rows)
    print(str(manifest_path))
    return worst_returncode


if __name__ == "__main__":
    raise SystemExit(main())
