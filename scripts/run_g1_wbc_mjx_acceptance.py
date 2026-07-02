#!/usr/bin/env python3
"""Run formal G1 WBC MJX acceptance against a frozen baseline manifest."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from spider.tasks.g1_wbc.acceptance import (
    MjxQualityPolicy,
    PRIMARY_ERROR_METRICS,
    REQUIRED_ARTIFACT_FIELDS,
    SpeedGateResult,
    evaluate_baseline_group,
    evaluate_mjx_group,
    evaluate_speed_gate,
)

SPIDER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON_EXECUTABLE = (
    SPIDER_ROOT / ".venv" / "bin" / "python"
    if (SPIDER_ROOT / ".venv" / "bin" / "python").exists()
    else Path(sys.executable)
)
MOTIONS = ("jump", "walk")
SEEDS = (0, 1, 2)
MIN_SPEEDUP = 12.0


@dataclass(frozen=True)
class PlannedAcceptanceRun:
    motion: str
    seed: int
    backend: str
    replay_backend: str
    output_dir: str
    replay_output_dir: str
    mjx_argv: list[str]
    replay_argv: list[str]
    mjx_command_text: str
    replay_command_text: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python-executable", default=str(DEFAULT_PYTHON_EXECUTABLE))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-speedup", type=float, default=MIN_SPEEDUP)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_acceptance_plan(
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> list[PlannedAcceptanceRun]:
    output_root = args.output_dir.expanduser().resolve()
    rows = manifest.get("rows", [])
    matrix = _baseline_run_matrix(rows)
    plan: list[PlannedAcceptanceRun] = []
    for motion in MOTIONS:
        for seed in SEEDS:
            row = matrix[(motion, seed)]
            output_dir = output_root / motion / f"seed_{seed}" / "mjx"
            replay_output_dir = output_root / motion / f"seed_{seed}" / "replay"
            mjx_argv = _mjx_argv_from_baseline_row(
                row,
                args.python_executable,
                output_dir,
                args.device,
            )
            replay_argv = _replay_argv_from_mjx(
                row,
                mjx_argv,
                output_dir,
                replay_output_dir,
            )
            plan.append(
                PlannedAcceptanceRun(
                    motion=motion,
                    seed=seed,
                    backend="mjx",
                    replay_backend="mujoco_warp",
                    output_dir=str(output_dir),
                    replay_output_dir=str(replay_output_dir),
                    mjx_argv=mjx_argv,
                    replay_argv=replay_argv,
                    mjx_command_text=shlex.join(mjx_argv),
                    replay_command_text=shlex.join(replay_argv),
                )
            )
    return plan


def run_command(argv: list[str], *, cwd: Path) -> dict[str, Any]:
    start = time.perf_counter()
    result = subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    wall_time_sec = time.perf_counter() - start
    row = {
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "status": "ok" if result.returncode == 0 else "failed",
        "command_wall_time_sec": float(wall_time_sec),
    }
    metrics_path = _output_dir_from_argv(argv) / "metrics.json"
    if metrics_path.is_file():
        row.update(_row_from_metrics(metrics_path))
    return row


def write_report(output_dir: Path, report: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "acceptance_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    baseline_manifest = args.baseline_manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = json.loads(baseline_manifest.read_text())
    plan = build_acceptance_plan(args, manifest)

    mjx_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    for planned in plan:
        if args.dry_run:
            mjx_row = {
                **asdict(planned),
                "status": "dry_run",
                "returncode": None,
                "metrics": {},
            }
            replay_row = {
                **asdict(planned),
                "status": "dry_run",
                "returncode": None,
                "metrics": {},
            }
        else:
            Path(planned.output_dir).mkdir(parents=True, exist_ok=True)
            Path(planned.replay_output_dir).mkdir(parents=True, exist_ok=True)
            mjx_row = {
                **asdict(planned),
                **run_command(planned.mjx_argv, cwd=SPIDER_ROOT),
            }
            replay_row = {
                **asdict(planned),
                **run_command(planned.replay_argv, cwd=SPIDER_ROOT),
            }
        mjx_rows.append(_attach_artifacts(mjx_row, planned.output_dir))
        replay_rows.append(_attach_artifacts(replay_row, planned.replay_output_dir))

    report = _build_report(
        baseline_manifest=baseline_manifest,
        baseline_rows=list(manifest.get("rows", [])),
        mjx_rows=mjx_rows,
        replay_rows=replay_rows,
        min_speedup=float(args.min_speedup),
    )
    report["planned_runs"] = [asdict(item) for item in plan]
    report_path = write_report(output_dir, report)
    print(str(report_path))
    return 0 if bool(report["passed"]) else 1


def _baseline_run_matrix(rows: Any) -> dict[tuple[str, int], dict[str, Any]]:
    expected = {(motion, seed) for motion in MOTIONS for seed in SEEDS}
    matrix: dict[tuple[str, int], dict[str, Any]] = {}
    duplicates: list[tuple[str, int]] = []
    extras: list[tuple[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        motion = str(row.get("motion_name"))
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError):
            extras.append((motion, row.get("seed")))
            continue
        key = (motion, seed)
        if key not in expected:
            extras.append(key)
            continue
        if key in matrix:
            duplicates.append(key)
        matrix[key] = row
    missing = sorted(expected - set(matrix))
    if missing or duplicates or extras:
        raise ValueError(
            "Baseline manifest run matrix must contain exactly one row for "
            f"{MOTIONS} x seeds {SEEDS}; "
            f"missing={missing}, duplicates={duplicates}, extras={extras}."
        )
    return matrix


def _mjx_argv_from_baseline_row(
    row: dict[str, Any],
    python_executable: str,
    output_dir: Path,
    device: str,
) -> list[str]:
    argv = list(row["argv"])
    argv[0] = str(python_executable)
    argv = _set_arg(argv, "--mpc-backend", "mjx")
    argv = _set_arg(argv, "--output-dir", str(output_dir))
    argv = _set_arg(argv, "--device", str(device))
    if "--save-rollout" not in argv:
        argv.append("--save-rollout")
    return argv


def _replay_argv_from_mjx(
    baseline_row: dict[str, Any],
    mjx_argv: list[str],
    mjx_output_dir: Path,
    replay_output_dir: Path,
) -> list[str]:
    argv = list(mjx_argv)
    argv = _set_arg(argv, "--method", "replay_command")
    argv = _set_arg(argv, "--mpc-backend", "mujoco_warp")
    argv = _set_arg(argv, "--output-dir", str(replay_output_dir))
    argv = _drop_arg_with_value(argv, "--mpc-reward-weights")
    argv.extend(["--saved-command", str(mjx_output_dir / "mpc_command.npz")])
    argv.extend(["--replay-control-steps", "20"])
    argv.extend(["--replay-task-mode", "g1_wbc_joint_global"])
    argv = _set_arg(argv, "--seed", str(int(baseline_row["seed"])))
    return argv


def _build_report(
    *,
    baseline_manifest: Path,
    baseline_rows: list[dict[str, Any]],
    mjx_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    min_speedup: float,
) -> dict[str, Any]:
    motion_results = {}
    replay_results = {}
    speed_results = {}
    passed = True
    for motion in MOTIONS:
        baseline_group = [row for row in baseline_rows if row.get("motion_name") == motion]
        mjx_group = [row for row in mjx_rows if row.get("motion") == motion]
        replay_group = [row for row in replay_rows if row.get("motion") == motion]
        baseline_gate = evaluate_baseline_group(motion, baseline_group)
        mjx_gate = evaluate_mjx_group(
            motion,
            mjx_group,
            baseline_gate.envelope,
            MjxQualityPolicy.for_motion(motion),
        )
        replay_failures = _row_evidence_failures(
            replay_group,
            timing_failure="replay_steady_state_wall_time",
            timing_fields=("command_wall_time_sec", "steady_state_wall_time_sec"),
            artifact_fields=("metrics_json", "rollout_npz"),
            require_mpc_fields=False,
        )
        replay_passed = not replay_failures
        speed_gate = _speed_gate_for_motion(
            baseline_group,
            mjx_group,
            min_speedup=min_speedup,
        )
        motion_results[motion] = {
            "baseline_passed": baseline_gate.passed,
            "baseline_failures": baseline_gate.failures,
            "mjx_passed": mjx_gate.passed,
            "mjx_failures": mjx_gate.failures,
        }
        replay_results[motion] = {
            "passed": replay_passed,
            "failures": replay_failures,
        }
        speed_results[motion] = {
            "passed": speed_gate.passed,
            "speedup": speed_gate.speedup,
            "failures": speed_gate.failures,
        }
        passed = (
            passed
            and baseline_gate.passed
            and mjx_gate.passed
            and replay_passed
            and speed_gate.passed
        )
    return {
        "schema_version": 1,
        "backend": "mjx_canonical",
        "baseline_manifest": str(baseline_manifest),
        "min_speedup": float(min_speedup),
        "motion_results": motion_results,
        "replay_results": replay_results,
        "speed_results": speed_results,
        "baseline_rows": baseline_rows,
        "mjx_rows": mjx_rows,
        "replay_rows": replay_rows,
        "passed": passed,
    }


def _speed_gate_for_motion(
    baseline_rows: list[dict[str, Any]],
    mjx_rows: list[dict[str, Any]],
    *,
    min_speedup: float,
):
    baseline_time, baseline_failures = _mean_timing_strict(
        baseline_rows,
        "steady_state_wall_time_sec",
        expected_count=len(SEEDS),
        failure="baseline_wall_time",
    )
    mjx_time, mjx_failures = _mean_timing_strict(
        mjx_rows,
        "steady_state_wall_time_sec",
        expected_count=len(SEEDS),
        failure="mjx_steady_state_wall_time",
    )
    gate = evaluate_speed_gate(
        baseline_wall_time_sec=baseline_time,
        mjx_steady_state_wall_time_sec=mjx_time,
        min_speedup=min_speedup,
    )
    failures = _unique((*baseline_failures, *mjx_failures, *gate.failures))
    if failures:
        return SpeedGateResult(False, gate.speedup, failures)
    return gate


def _mean_timing_strict(
    rows: list[dict[str, Any]],
    name: str,
    *,
    expected_count: int,
    failure: str,
) -> tuple[float, tuple[str, ...]]:
    values = []
    failures: list[str] = []
    for row in rows:
        value = _timing_value(row, name)
        if (
            isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 0.0
        ):
            values.append(float(value))
        else:
            failures.append(failure)
    if len(values) != int(expected_count):
        failures.append(failure)
    if failures:
        return float("nan"), _unique(failures)
    return sum(values) / len(values), ()


def _row_from_metrics(metrics_path: Path) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text())
    metrics = payload.get("metrics", {})
    mpc = payload.get("mpc", {})
    return {
        "metrics": metrics,
        "mpc": mpc,
        "mpc_accepted": mpc.get("accepted") is True,
        "accepted_windows": _safe_int(
            mpc.get("accepted_windows", mpc.get("num_windows", -1))
        ),
        "mpc_used_baseline_fallback": mpc.get("used_baseline_fallback") is not False,
        "num_steps": _safe_int(metrics.get("num_steps", -1)),
        "steady_state_wall_time_sec": mpc.get("steady_state_wall_time_sec"),
    }


def _row_evidence_failures(
    rows: list[dict[str, Any]],
    *,
    timing_failure: str | None = None,
    timing_fields: tuple[str, ...] = ("steady_state_wall_time_sec",),
    artifact_fields: tuple[str, ...] = REQUIRED_ARTIFACT_FIELDS,
    require_mpc_fields: bool = True,
) -> tuple[str, ...]:
    failures: list[str] = []
    if len(rows) != len(SEEDS):
        failures.append("repeat_count")
    seen: set[int] = set()
    for row in rows:
        seed = _safe_int(row.get("seed"))
        if seed not in SEEDS:
            failures.append("seed")
        elif seed in seen:
            failures.append("seed")
        else:
            seen.add(seed)
        if row.get("status") != "ok":
            failures.append("status")
        if row.get("returncode") != 0:
            failures.append("returncode")
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            failures.append("metrics")
            metrics = {}
        for metric in ("success", "score", *PRIMARY_ERROR_METRICS):
            if not _has_valid_metric(metrics, metric):
                failures.append(f"{metric}_missing")
        if require_mpc_fields:
            if row.get("mpc_accepted") is not True:
                failures.append("mpc_accepted")
            if _safe_int(row.get("accepted_windows")) != 40:
                failures.append("accepted_windows")
            if row.get("mpc_used_baseline_fallback") is not False:
                failures.append("baseline_fallback")
        artifacts = row.get("artifacts", {})
        for key in artifact_fields:
            if not isinstance(artifacts, dict) or not artifacts.get(key):
                failures.append(key)
        if timing_failure is not None:
            if not any(_valid_timing(_timing_value(row, field)) for field in timing_fields):
                failures.append(timing_failure)
    missing = set(SEEDS) - seen
    if missing:
        failures.append("seed")
    return _unique(failures)


def _timing_value(row: dict[str, Any], name: str):
    mpc = row.get("mpc", {})
    return row.get(name, mpc.get(name) if isinstance(mpc, dict) else None)


def _valid_timing(value) -> bool:
    return (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _has_valid_metric(metrics: dict[str, Any], name: str) -> bool:
    value = metrics.get(name)
    if isinstance(value, bool):
        return True
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _safe_int(value) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if float(parsed) == float(value) else None


def _unique(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _attach_artifacts(row: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser()
    row = dict(row)
    row["artifacts"] = {
        "metrics_json": _existing_path(output_dir / "metrics.json"),
        "rollout_npz": _existing_path(output_dir / "rollout.npz"),
        "mpc_command_npz": _existing_path(output_dir / "mpc_command.npz"),
    }
    return row


def _existing_path(path: Path) -> str | None:
    return str(path.resolve()) if path.exists() else None


def _output_dir_from_argv(argv: list[str]) -> Path:
    try:
        return Path(argv[argv.index("--output-dir") + 1]).expanduser()
    except (ValueError, IndexError) as exc:
        raise ValueError("Command argv is missing --output-dir") from exc


def _set_arg(argv: list[str], flag: str, value: str) -> list[str]:
    out = list(argv)
    try:
        index = out.index(flag)
    except ValueError:
        out.extend([flag, value])
    else:
        if index + 1 >= len(out):
            raise ValueError(f"{flag} has no value")
        out[index + 1] = value
    return out


def _drop_arg_with_value(argv: list[str], flag: str) -> list[str]:
    out = []
    skip_next = False
    for index, value in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if value == flag:
            if index + 1 >= len(argv):
                raise ValueError(f"{flag} has no value")
            skip_next = True
            continue
        out.append(value)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
