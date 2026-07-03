#!/usr/bin/env python3
"""Run formal G1 WBC MJX acceptance against a frozen baseline manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import numpy as np

from spider.tasks.g1_wbc.acceptance import (
    MjxQualityPolicy,
    PRIMARY_ERROR_METRICS,
    REQUIRED_ARTIFACT_FIELDS,
    SpeedGateResult,
    evaluate_baseline_group,
    evaluate_mjx_group,
    evaluate_speed_gate,
)
from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    MUJOCO_BODY_NAMES,
    POLICY_DT,
    QPOS_DIM,
    QVEL_DIM,
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
MIN_REALTIME_FACTOR = 1.0
DEFAULT_REQUIRED_GPU_NAME_FRAGMENT = "H100"
ARTIFACT_FRESHNESS_TOLERANCE_NS = 2_000_000_000
FORMAL_BASELINE_NAME = "g1_wbc_stage0_mujoco_warp_sweetpoint"
TARGET_H100_SPEEDUP = "h100_speedup"
TARGET_4090_REALTIME = "4090_realtime"
REQUIRED_INPUT_SHA256_FIELDS = (
    "jump_motion",
    "walk_motion",
    "checkpoint",
    "reward_weights",
)
FORMAL_STAGE0_ARG_VALUES = {
    "--motion-type": "isaaclab",
    "--method": "g1_wbc_joint_global",
    "--mpc-backend": "mujoco_warp",
    "--max-steps": "800",
    "--mpc-samples": "512",
    "--mpc-iterations": "2",
    "--mpc-planning-horizon-steps": "40",
    "--mpc-control-steps": "20",
    "--mpc-sampling-mode": "knot",
    "--mpc-knot-count": "8",
    "--mpc-temperature": "0.7",
    "--mpc-root-pos-sigma": "0.04",
    "--mpc-root-rot-sigma": "0.10",
    "--mpc-joint-sigma": "0.18",
    "--mpc-smooth-passes": "0",
    "--mpc-command-reg-weight": "0.0",
    "--mpc-command-smooth-weight": "0.0",
}
FORMAL_STAGE0_FLAGS = (
    "--save-rollout",
    "--mpc-guided-candidate",
    "--mpc-acceptance-gate",
)
MJX_CONTACT_SATURATION_FIELDS = (
    "contact_saturated",
    "max_contact_points_saturated",
    "max_geom_pairs_saturated",
)
MJX_CONTACT_COUNT_FIELDS = (
    "max_contact_points",
    "max_geom_pairs",
    "contact_pair_count",
    "active_contact_count",
)


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
    parser.add_argument(
        "--target",
        choices=(TARGET_H100_SPEEDUP, TARGET_4090_REALTIME),
        default=TARGET_H100_SPEEDUP,
        help=(
            "Formal acceptance target. The H100 target gates on baseline-relative "
            "speedup; the 4090 target gates on real-time factor."
        ),
    )
    parser.add_argument("--min-speedup", type=float, default=MIN_SPEEDUP)
    parser.add_argument(
        "--min-realtime-factor",
        type=float,
        default=MIN_REALTIME_FACTOR,
        help="Minimum motion-duration / MJX steady-state wall-time for the 4090 target.",
    )
    parser.add_argument(
        "--required-gpu-name-fragment",
        default=DEFAULT_REQUIRED_GPU_NAME_FRAGMENT,
        help=(
            "Substring required in baseline and MJX runtime GPU names for this "
            "acceptance milestone. Use an empty value to disable the model-name check."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_acceptance_plan(
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> list[PlannedAcceptanceRun]:
    output_root = args.output_dir.expanduser().resolve()
    rows = manifest.get("rows", [])
    matrix = _baseline_run_matrix(rows)
    _validate_formal_baseline_manifest(manifest, matrix)
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
    command_start_time_ns = time.time_ns()
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
        "command_start_time_ns": int(command_start_time_ns),
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
        baseline_envelopes=_baseline_envelopes_from_manifest(manifest),
        mjx_rows=mjx_rows,
        replay_rows=replay_rows,
        min_speedup=float(args.min_speedup),
        target=str(args.target),
        min_realtime_factor=float(args.min_realtime_factor),
        required_gpu_name_fragment=str(args.required_gpu_name_fragment),
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


def _validate_formal_baseline_manifest(
    manifest: dict[str, Any],
    matrix: dict[tuple[str, int], dict[str, Any]],
) -> None:
    failures: list[str] = []
    if manifest.get("schema_version") != 1:
        failures.append("schema_version")
    if manifest.get("baseline_name") != FORMAL_BASELINE_NAME:
        failures.append("baseline_name")
    if tuple(str(value) for value in manifest.get("motions", ())) != MOTIONS:
        failures.append("motions")
    try:
        seeds = tuple(int(value) for value in manifest.get("seeds", ()))
    except (TypeError, ValueError):
        seeds = ()
    if seeds != SEEDS:
        failures.append("seeds")
    failures.extend(_formal_manifest_provenance_failures(manifest, matrix))
    failures.extend(_formal_manifest_baseline_envelope_failures(manifest, matrix))

    for (motion, seed), row in matrix.items():
        failures.extend(_formal_stage0_row_failures(row, motion=motion, seed=seed))

    unique_failures = _unique(failures)
    if unique_failures:
        raise ValueError(
            "Baseline manifest is not a formal Stage 0 sweetpoint: "
            + ", ".join(unique_failures)
        )


def _formal_manifest_provenance_failures(
    manifest: dict[str, Any],
    matrix: dict[tuple[str, int], dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        failures.append("manifest_provenance")
    else:
        for field in ("worktree_path", "git_commit"):
            value = provenance.get(field)
            if not isinstance(value, str) or not value.strip():
                failures.append("manifest_provenance")
        status = provenance.get("git_status_short")
        if status is not None and not isinstance(status, str):
            failures.append("manifest_provenance")
    input_hashes = manifest.get("input_sha256")
    if not isinstance(input_hashes, dict):
        failures.append("manifest_input_sha256")
    else:
        for field in REQUIRED_INPUT_SHA256_FIELDS:
            if not _is_sha256_hex(input_hashes.get(field)):
                failures.append("manifest_input_sha256")
        if not _manifest_input_hashes_match(input_hashes, matrix):
            failures.append("manifest_input_sha256")
    return failures


def _formal_manifest_baseline_envelope_failures(
    manifest: dict[str, Any],
    matrix: dict[tuple[str, int], dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    baseline_envelopes = _baseline_envelopes_from_manifest(manifest)
    promoted_seeds = manifest.get("promoted_seeds")
    if set(baseline_envelopes) != set(MOTIONS):
        failures.append("baseline_envelopes")
    if not isinstance(promoted_seeds, dict):
        failures.append("promoted_seeds")
        promoted_seeds = {}

    for motion in MOTIONS:
        group = [matrix[(motion, seed)] for seed in SEEDS]
        gate = evaluate_baseline_group(motion, group)
        if not gate.passed:
            failures.append("baseline_envelopes")
            continue
        frozen = baseline_envelopes.get(motion)
        if frozen is None or not _envelopes_equivalent(frozen, gate.envelope):
            failures.append("baseline_envelopes")
        if _safe_int(promoted_seeds.get(motion)) != gate.promoted_seed:
            failures.append("promoted_seeds")
    return failures


def _baseline_envelopes_from_manifest(
    manifest: dict[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    raw = manifest.get("baseline_envelopes")
    if not isinstance(raw, dict):
        return {}
    baseline_envelopes: dict[str, dict[str, dict[str, float]]] = {}
    for motion in MOTIONS:
        envelope = _normalise_metric_envelope(raw.get(motion))
        if envelope is not None:
            baseline_envelopes[motion] = envelope
    return baseline_envelopes


def _normalise_metric_envelope(
    value: Any,
) -> dict[str, dict[str, float]] | None:
    if not isinstance(value, dict):
        return None
    envelope: dict[str, dict[str, float]] = {}
    for metric, stats in value.items():
        if not isinstance(metric, str) or not isinstance(stats, dict):
            return None
        envelope[metric] = {}
        for stat, stat_value in stats.items():
            if (
                not isinstance(stat, str)
                or isinstance(stat_value, bool)
                or not isinstance(stat_value, (int, float))
            ):
                return None
            stat_float = float(stat_value)
            if not math.isfinite(stat_float):
                return None
            envelope[metric][stat] = stat_float
    return envelope


def _envelopes_equivalent(
    observed: dict[str, dict[str, float]],
    expected: dict[str, dict[str, float]],
) -> bool:
    if set(observed) != set(expected):
        return False
    for metric, expected_stats in expected.items():
        observed_stats = observed.get(metric)
        if observed_stats is None or set(observed_stats) != set(expected_stats):
            return False
        for stat, expected_value in expected_stats.items():
            observed_value = observed_stats[stat]
            if not math.isclose(
                float(observed_value),
                float(expected_value),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                return False
    return True


def _manifest_input_hashes_match(
    input_hashes: dict[str, Any],
    matrix: dict[tuple[str, int], dict[str, Any]],
) -> bool:
    observed: dict[str, set[str]] = {
        field: set()
        for field in REQUIRED_INPUT_SHA256_FIELDS
    }
    for (motion, _seed), row in matrix.items():
        argv = row.get("argv")
        if not isinstance(argv, list):
            continue
        _record_file_hash(observed, f"{motion}_motion", _argv_value(argv, "--motion"))
        _record_file_hash(observed, "checkpoint", _argv_value(argv, "--checkpoint"))
        _record_file_hash(
            observed,
            "reward_weights",
            _argv_value(argv, "--mpc-reward-weights"),
        )
    for field in REQUIRED_INPUT_SHA256_FIELDS:
        if observed[field] != {input_hashes.get(field)}:
            return False
    return True


def _record_file_hash(
    observed: dict[str, set[str]],
    field: str,
    path: str | None,
) -> None:
    if field not in observed or not _is_existing_file(path):
        return
    observed[field].add(_file_sha256(Path(str(path)).expanduser()))


def _formal_stage0_row_failures(
    row: dict[str, Any],
    *,
    motion: str,
    seed: int,
) -> list[str]:
    failures: list[str] = []
    argv = row.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        return ["argv"]

    for flag, expected in FORMAL_STAGE0_ARG_VALUES.items():
        value = _argv_value(argv, flag)
        if value != expected:
            failures.append(flag)
    for flag in FORMAL_STAGE0_FLAGS:
        if flag not in argv:
            failures.append(flag)

    if _argv_value(argv, "--seed") != str(seed):
        failures.append("--seed")
    if not _same_path(_argv_value(argv, "--motion"), row.get("motion")):
        failures.append("--motion")
    if not _same_path(_argv_value(argv, "--output-dir"), row.get("output_dir")):
        failures.append("--output-dir")

    motion_path = _argv_value(argv, "--motion")
    if not _is_existing_file(motion_path):
        failures.append("motion_file")
    checkpoint_path = _argv_value(argv, "--checkpoint")
    if not _is_existing_file(checkpoint_path):
        failures.append("checkpoint")
    reward_path = _argv_value(argv, "--mpc-reward-weights")
    if not _is_existing_file(reward_path):
        failures.append("--mpc-reward-weights")
    if row.get("motion_name") != motion:
        failures.append("motion_name")
    return failures


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
    if "--mjx-enable-scan" not in argv:
        argv.append("--mjx-enable-scan")
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
    argv = _drop_flag(argv, "--mjx-enable-scan")
    argv.extend(["--saved-command", str(mjx_output_dir / "mpc_command.npz")])
    argv.extend(["--replay-control-steps", "20"])
    argv.extend(["--replay-task-mode", "g1_wbc_joint_global"])
    argv = _set_arg(argv, "--seed", str(int(baseline_row["seed"])))
    return argv


def _build_report(
    *,
    baseline_manifest: Path,
    baseline_rows: list[dict[str, Any]],
    baseline_envelopes: dict[str, dict[str, dict[str, float]]],
    mjx_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    min_speedup: float,
    target: str,
    min_realtime_factor: float,
    required_gpu_name_fragment: str,
) -> dict[str, Any]:
    motion_results = {}
    replay_results = {}
    speed_results = {}
    realtime_results = {}
    passed = True
    for motion in MOTIONS:
        baseline_group = [row for row in baseline_rows if row.get("motion_name") == motion]
        mjx_group = [row for row in mjx_rows if row.get("motion") == motion]
        replay_group = [row for row in replay_rows if row.get("motion") == motion]
        baseline_gate = evaluate_baseline_group(motion, baseline_group)
        baseline_envelope = baseline_envelopes[motion]
        baseline_failures = _unique(
            (
                *baseline_gate.failures,
                *_baseline_artifact_evidence_failures(baseline_group),
                *_artifact_freshness_failures(
                    baseline_group,
                    artifact_fields=REQUIRED_ARTIFACT_FIELDS,
                ),
                *_artifact_hash_failures(
                    baseline_group,
                    artifact_fields=REQUIRED_ARTIFACT_FIELDS,
                ),
                *_artifact_npz_schema_failures(
                    baseline_group,
                    require_command=True,
                ),
                *_baseline_runtime_evidence_failures(
                    baseline_group,
                    required_gpu_name_fragment=required_gpu_name_fragment,
                ),
            )
        )
        mjx_gate = evaluate_mjx_group(
            motion,
            mjx_group,
            baseline_envelope,
            MjxQualityPolicy.for_motion(motion),
        )
        mjx_failures = _unique(
            (
                *mjx_gate.failures,
                *_mjx_timing_evidence_failures(mjx_group),
                *_mjx_runtime_evidence_failures(
                    mjx_group,
                    required_gpu_name_fragment=required_gpu_name_fragment,
                ),
                *_mjx_contact_evidence_failures(mjx_group),
                *_artifact_freshness_failures(
                    mjx_group,
                    artifact_fields=REQUIRED_ARTIFACT_FIELDS,
                ),
                *_artifact_hash_failures(
                    mjx_group,
                    artifact_fields=REQUIRED_ARTIFACT_FIELDS,
                ),
                *_artifact_npz_schema_failures(
                    mjx_group,
                    require_command=True,
                ),
            )
        )
        replay_failures = _row_evidence_failures(
            replay_group,
            timing_failure="replay_steady_state_wall_time",
            timing_fields=("command_wall_time_sec", "steady_state_wall_time_sec"),
            artifact_fields=("metrics_json", "rollout_npz"),
            require_mpc_fields=False,
        )
        replay_failures = _unique(
            (
                *replay_failures,
                *_replay_provenance_failures(replay_group, mjx_group),
                *_artifact_freshness_failures(
                    replay_group,
                    artifact_fields=("metrics_json", "rollout_npz"),
                ),
                *_artifact_hash_failures(
                    replay_group,
                    artifact_fields=("metrics_json", "rollout_npz"),
                ),
                *_artifact_npz_schema_failures(
                    replay_group,
                    require_command=False,
                ),
                *_replay_quality_failures(
                    motion,
                    replay_group,
                    baseline_envelope,
                ),
            )
        )
        replay_passed = not replay_failures
        speed_gate = _speed_gate_for_motion(
            baseline_group,
            mjx_group,
            min_speedup=min_speedup,
        )
        motion_results[motion] = {
            "baseline_passed": not baseline_failures,
            "baseline_failures": baseline_failures,
            "mjx_passed": not mjx_failures,
            "mjx_failures": mjx_failures,
        }
        replay_results[motion] = {
            "passed": replay_passed,
            "failures": replay_failures,
        }
        speed_results[motion] = {
            "passed": speed_gate.passed,
            "speedup": speed_gate.speedup,
            "worst_speedup": speed_gate.worst_speedup,
            "failures": speed_gate.failures,
        }
        realtime_gate = _realtime_gate_for_motion(
            mjx_group,
            min_realtime_factor=min_realtime_factor,
            require_explicit_duration=target == TARGET_4090_REALTIME,
        )
        realtime_results[motion] = realtime_gate
        target_speed_passed = (
            bool(realtime_gate["passed"])
            if target == TARGET_4090_REALTIME
            else speed_gate.passed
        )
        passed = (
            passed
            and not baseline_failures
            and not mjx_failures
            and replay_passed
            and target_speed_passed
        )
    classification = _classify_report(
        motion_results=motion_results,
        replay_results=replay_results,
        speed_results=speed_results,
        realtime_results=realtime_results,
        target=target,
        passed=passed,
    )
    return {
        "schema_version": 1,
        "backend": "mjx_canonical",
        "baseline_manifest": str(baseline_manifest),
        "baseline_envelopes": baseline_envelopes,
        "classification": classification,
        "target": target,
        "min_speedup": float(min_speedup),
        "min_realtime_factor": float(min_realtime_factor),
        "required_gpu_name_fragment": required_gpu_name_fragment,
        "motion_results": motion_results,
        "replay_results": replay_results,
        "speed_results": speed_results,
        "realtime_results": realtime_results,
        "timing_summary": _timing_summary(
            baseline_rows=baseline_rows,
            mjx_rows=mjx_rows,
            replay_rows=replay_rows,
        ),
        "contact_summary": _contact_summary(mjx_rows=mjx_rows),
        "baseline_rows": baseline_rows,
        "mjx_rows": mjx_rows,
        "replay_rows": replay_rows,
        "passed": passed,
    }


def _timing_summary(
    *,
    baseline_rows: list[dict[str, Any]],
    mjx_rows: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for motion in MOTIONS:
        baseline_group = [row for row in baseline_rows if row.get("motion_name") == motion]
        mjx_group = [row for row in mjx_rows if row.get("motion") == motion]
        replay_group = [row for row in replay_rows if row.get("motion") == motion]
        summary[motion] = {
            "baseline": {
                "steady_state_wall_time_sec": _timing_stats(
                    _timing_values(baseline_group, "steady_state_wall_time_sec")
                ),
            },
            "mjx": {
                "steady_state_wall_time_sec": _timing_stats(
                    _timing_values(mjx_group, "steady_state_wall_time_sec")
                ),
                "compile_init_wall_time_sec": _timing_stats(
                    _timing_values(mjx_group, "compile_init_wall_time_sec")
                ),
                "jit_warmup_wall_time_sec": _timing_stats(
                    _timing_values(mjx_group, "jit_warmup_wall_time_sec")
                ),
                "per_window_steady_state_wall_time_sec": _timing_stats(
                    _per_window_timing_values(mjx_group)
                ),
                "num_windows": _timing_stats(_window_count_values(mjx_group)),
                "runtime_visible_devices": [
                    [str(value) for value in row.get("runtime_visible_devices", ())]
                    for row in mjx_group
                ],
                "runtime_gpu_names": [
                    row.get("runtime_gpu_name")
                    if isinstance(row.get("runtime_gpu_name"), str)
                    else None
                    for row in mjx_group
                ],
            },
            "replay": {
                "command_wall_time_sec": _timing_stats(
                    _timing_values(replay_group, "command_wall_time_sec")
                ),
                "steady_state_wall_time_sec": _timing_stats(
                    _timing_values(replay_group, "steady_state_wall_time_sec")
                ),
            },
        }
    return summary


def _contact_summary(*, mjx_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for motion in MOTIONS:
        mjx_group = [row for row in mjx_rows if row.get("motion") == motion]
        summary[motion] = {
            "mjx": {
                "max_contact_points": _timing_stats(
                    _contact_count_values(mjx_group, "max_contact_points")
                ),
                "max_geom_pairs": _timing_stats(
                    _contact_count_values(mjx_group, "max_geom_pairs")
                ),
                "contact_pair_count": _timing_stats(
                    _contact_count_values(mjx_group, "contact_pair_count")
                ),
                "active_contact_count": _timing_stats(
                    _contact_count_values(mjx_group, "active_contact_count")
                ),
                "saturation_flags": {
                    field: [_contact_flag_value(row, field) for row in mjx_group]
                    for field in MJX_CONTACT_SATURATION_FIELDS
                },
            },
        }
    return summary


def _timing_values(rows: list[dict[str, Any]], name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _timing_value(row, name)
        if _valid_timing(value):
            values.append(float(value))
    return values


def _window_count_values(rows: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _safe_int(row.get("num_windows", row.get("accepted_windows")))
        if value is not None and value > 0:
            values.append(float(value))
    return values


def _per_window_timing_values(rows: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        steady = _timing_value(row, "steady_state_wall_time_sec")
        windows = _safe_int(row.get("num_windows", row.get("accepted_windows")))
        if _valid_timing(steady) and windows is not None and windows > 0:
            values.append(float(steady) / float(windows))
    return values


def _timing_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "values": [],
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
        }
    return {
        "values": values,
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def _classify_report(
    *,
    motion_results: dict[str, dict[str, Any]],
    replay_results: dict[str, dict[str, Any]],
    speed_results: dict[str, dict[str, Any]],
    realtime_results: dict[str, dict[str, Any]],
    target: str,
    passed: bool,
) -> str:
    if passed:
        return (
            "pass_4090_realtime"
            if target == TARGET_4090_REALTIME
            else "pass_h100_milestone"
        )
    if _has_invalid_benchmark_failure(
        motion_results,
        replay_results,
        speed_results,
        realtime_results,
    ):
        return "invalid_benchmark"
    if any(not result.get("passed") for result in replay_results.values()):
        return "parity_failure"
    if any(
        not result.get("baseline_passed") or not result.get("mjx_passed")
        for result in motion_results.values()
    ):
        return "quality_regression"
    target_results = realtime_results if target == TARGET_4090_REALTIME else speed_results
    if any(not result.get("passed") for result in target_results.values()):
        return "speed_regression"
    return "invalid_benchmark"


def _replay_quality_failures(
    motion: str,
    rows: list[dict[str, Any]],
    baseline_envelope: dict[str, dict[str, float]],
) -> tuple[str, ...]:
    if len(rows) != len(SEEDS):
        return ()
    quality_rows = [_replay_quality_row(row) for row in rows]
    gate = evaluate_mjx_group(
        motion,
        quality_rows,
        baseline_envelope,
        MjxQualityPolicy.for_motion(motion),
    )
    return tuple(
        failure
        for failure in gate.failures
        if failure
        not in {
            "accepted_windows",
            "baseline_fallback",
            "mpc_accepted",
            "mpc_command_npz",
        }
    )


def _replay_quality_row(row: dict[str, Any]) -> dict[str, Any]:
    quality_row = dict(row)
    artifacts = dict(quality_row.get("artifacts", {}) or {})
    artifacts.setdefault("mpc_command_npz", "replay_uses_mjx_command")
    quality_row.update(
        {
            "mpc_accepted": True,
            "accepted_windows": 40,
            "mpc_used_baseline_fallback": False,
            "artifacts": artifacts,
        }
    )
    return quality_row


def _replay_provenance_failures(
    rows: list[dict[str, Any]],
    mjx_rows: list[dict[str, Any]],
) -> tuple[str, ...]:
    failures: list[str] = []
    command_by_seed = _mjx_command_path_by_seed(mjx_rows)
    for row in rows:
        mpc = row.get("mpc")
        if not isinstance(mpc, dict):
            failures.append("replay_mode")
            failures.append("replay_saved_command")
            continue

        if mpc.get("replay_mode") != "shared_execute_backend":
            failures.append("replay_mode")

        expected_saved = _argv_value(row.get("replay_argv", []), "--saved-command")
        saved = mpc.get("saved_command")
        if not _same_path(saved if isinstance(saved, str) else None, expected_saved):
            failures.append("replay_saved_command")
        elif not _is_existing_file(saved):
            failures.append("replay_saved_command")
        seed = _safe_int(row.get("seed"))
        source_command = command_by_seed.get(seed)
        if not _same_path(saved if isinstance(saved, str) else None, source_command):
            failures.append("replay_saved_command_source")

        expected_control = _safe_int(
            _argv_value(row.get("replay_argv", []), "--replay-control-steps")
        )
        if _safe_int(mpc.get("control_steps")) != expected_control:
            failures.append("replay_control_steps")

        replay_steps = _safe_int(mpc.get("num_replay_steps"))
        if replay_steps != _safe_int(row.get("num_steps")):
            failures.append("replay_num_replay_steps")
        command_frames = _safe_int(mpc.get("num_command_frames"))
        if (
            command_frames is None
            or replay_steps is None
            or command_frames < replay_steps + 1
        ):
            failures.append("replay_num_command_frames")
        command_npz_frames = (
            _npz_frame_count(Path(saved).expanduser(), keys=("refined_qpos",))
            if isinstance(saved, str)
            else None
        )
        if command_frames is not None and command_npz_frames != command_frames:
            failures.append("replay_command_npz_frames")
    return _unique(failures)


def _mjx_command_path_by_seed(rows: list[dict[str, Any]]) -> dict[int, str]:
    command_by_seed: dict[int, str] = {}
    for row in rows:
        seed = _safe_int(row.get("seed"))
        artifacts = row.get("artifacts", {})
        if seed is None or not isinstance(artifacts, dict):
            continue
        command_path = artifacts.get("mpc_command_npz")
        if isinstance(command_path, str) and command_path.strip():
            command_by_seed[int(seed)] = command_path
    return command_by_seed


def _npz_frame_count(path: Path, *, keys: tuple[str, ...]) -> int | None:
    try:
        with np.load(path) as data:
            for key in keys:
                if key in data.files:
                    shape = tuple(np.asarray(data[key]).shape)
                    if shape:
                        return int(shape[0])
    except Exception:
        return None
    return None


def _has_invalid_benchmark_failure(
    motion_results: dict[str, dict[str, Any]],
    replay_results: dict[str, dict[str, Any]],
    speed_results: dict[str, dict[str, Any]],
    realtime_results: dict[str, dict[str, Any]],
) -> bool:
    invalid_markers = {
        "accepted_windows",
        "baseline_fallback",
        "baseline_required_gpu",
        "baseline_runtime_gpu_name",
        "baseline_runtime_visible_devices",
        "baseline_single_visible_gpu",
        "baseline_wall_time",
        "compile_init_wall_time",
        "contact_saturation",
        "fallback",
        "metrics_json",
        "metrics_json_hash",
        "metrics_json_stale",
        "max_contact_points_saturation",
        "max_geom_pairs_saturation",
        "mjx_compile_init_wall_time",
        "mjx_contact_diagnostics",
        "mjx_jit_warmup_enabled",
        "mjx_jit_warmup_wall_time",
        "mjx_runtime_visible_devices",
        "mjx_runtime_gpu_name",
        "mjx_required_gpu",
        "mjx_single_visible_gpu",
        "mjx_steady_state_wall_time",
        "control_dt_sec",
        "evaluated_motion_duration_sec",
        "motion_duration_sec",
        "mpc_accepted",
        "mpc_command_npz",
        "mpc_command_npz_hash",
        "mpc_command_npz_schema",
        "mpc_command_npz_stale",
        "mpc_rollout_qpos_mismatch",
        "num_steps",
        "repeat_count",
        "replay_control_steps",
        "replay_command_npz_frames",
        "replay_mode",
        "replay_num_command_frames",
        "replay_num_replay_steps",
        "replay_saved_command",
        "replay_saved_command_source",
        "returncode",
        "rollout_npz",
        "rollout_npz_hash",
        "rollout_npz_schema",
        "rollout_npz_stale",
        "seed",
        "status",
    }
    for result in motion_results.values():
        failures = (*result.get("baseline_failures", ()), *result.get("mjx_failures", ()))
        if any(failure in invalid_markers for failure in failures):
            return True
    for result in replay_results.values():
        if any(failure in invalid_markers for failure in result.get("failures", ())):
            return True
    for result in speed_results.values():
        failures = result.get("failures", ())
        if any(failure in invalid_markers for failure in failures):
            return True
    for result in realtime_results.values():
        failures = result.get("failures", ())
        if any(failure in invalid_markers for failure in failures):
            return True
    return False


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
    worst_speedup, worst_failures = _worst_seed_speedup(
        baseline_rows,
        mjx_rows,
        min_speedup=min_speedup,
    )
    failures = _unique(
        (*baseline_failures, *mjx_failures, *gate.failures, *worst_failures)
    )
    if failures:
        return SpeedGateResult(False, gate.speedup, failures, worst_speedup)
    return SpeedGateResult(gate.passed, gate.speedup, gate.failures, worst_speedup)


def _worst_seed_speedup(
    baseline_rows: list[dict[str, Any]],
    mjx_rows: list[dict[str, Any]],
    *,
    min_speedup: float,
) -> tuple[float | None, tuple[str, ...]]:
    baseline_times = _timing_by_seed(baseline_rows, "steady_state_wall_time_sec")
    mjx_times = _timing_by_seed(mjx_rows, "steady_state_wall_time_sec")
    if set(baseline_times) != set(SEEDS) or set(mjx_times) != set(SEEDS):
        return None, ()
    speedups = [
        baseline_times[seed] / mjx_times[seed]
        for seed in SEEDS
    ]
    worst_speedup = min(speedups)
    failures = ("speedup_worst",) if worst_speedup < float(min_speedup) else ()
    return worst_speedup, failures


def _timing_by_seed(
    rows: list[dict[str, Any]],
    name: str,
) -> dict[int, float]:
    values: dict[int, float] = {}
    for row in rows:
        seed = _safe_int(row.get("seed"))
        value = _timing_value(row, name)
        if seed in SEEDS and _valid_timing(value) and float(value) > 0.0:
            values[int(seed)] = float(value)
    return values


def _realtime_gate_for_motion(
    mjx_rows: list[dict[str, Any]],
    *,
    min_realtime_factor: float,
    require_explicit_duration: bool = False,
) -> dict[str, Any]:
    duration, duration_failures = _mean_motion_duration_strict(
        mjx_rows,
        expected_count=len(SEEDS),
        require_explicit_duration=require_explicit_duration,
    )
    mjx_time, mjx_failures = _mean_timing_strict(
        mjx_rows,
        "steady_state_wall_time_sec",
        expected_count=len(SEEDS),
        failure="mjx_steady_state_wall_time",
    )
    failures = list(_unique((*duration_failures, *mjx_failures)))
    real_time_factor = float("nan")
    if not failures:
        real_time_factor = duration / mjx_time
        if real_time_factor < float(min_realtime_factor):
            failures.append("real_time_factor")
    return {
        "passed": not failures,
        "motion_duration_sec": duration,
        "mjx_steady_state_wall_time_sec": mjx_time,
        "real_time_factor": real_time_factor,
        "min_realtime_factor": float(min_realtime_factor),
        "failures": _unique(failures),
    }


def _mean_motion_duration_strict(
    rows: list[dict[str, Any]],
    *,
    expected_count: int,
    require_explicit_duration: bool,
) -> tuple[float, tuple[str, ...]]:
    values = []
    failures: list[str] = []
    for row in rows:
        value, row_failures = _motion_duration_sec(
            row,
            require_explicit_duration=require_explicit_duration,
        )
        failures.extend(row_failures)
        if (
            not row_failures
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 0.0
        ):
            values.append(float(value))
        else:
            failures.append("motion_duration_sec")
    if len(values) != int(expected_count):
        failures.append("motion_duration_sec")
    if failures:
        return float("nan"), _unique(failures)
    return sum(values) / len(values), ()


def _motion_duration_sec(
    row: dict[str, Any],
    *,
    require_explicit_duration: bool = False,
) -> tuple[float | None, tuple[str, ...]]:
    metrics = row.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    control_dt = _positive_float(
        row.get("control_dt_sec", metrics.get("control_dt_sec"))
    )
    evaluated_duration = _positive_float(
        row.get(
            "evaluated_motion_duration_sec",
            metrics.get("evaluated_motion_duration_sec"),
        )
    )
    steps = _safe_int(row.get("num_steps", metrics.get("num_steps")))

    failures: list[str] = []
    if require_explicit_duration:
        if control_dt is None:
            failures.append("control_dt_sec")
        if evaluated_duration is None:
            failures.append("evaluated_motion_duration_sec")
        if steps is None or steps <= 0:
            failures.append("num_steps")
        if (
            control_dt is not None
            and evaluated_duration is not None
            and steps is not None
            and steps > 0
        ):
            expected = float(steps) * float(control_dt)
            if not math.isclose(
                evaluated_duration,
                expected,
                rel_tol=1.0e-6,
                abs_tol=1.0e-6,
            ):
                failures.append("evaluated_motion_duration_sec")
        if failures:
            return None, _unique(failures)
        return evaluated_duration, ()

    if evaluated_duration is not None:
        return evaluated_duration, ()
    for candidate in (
        row.get("motion_duration_sec"),
        metrics.get("motion_duration_sec"),
        metrics.get("duration_sec"),
    ):
        value = _positive_float(candidate)
        if value is not None:
            return value, ()
    steps = _safe_int(row.get("num_steps", metrics.get("num_steps")))
    if steps is None or steps <= 0:
        return None, ("motion_duration_sec",)
    return float(steps) * float(POLICY_DT), ()


def _positive_float(value: Any) -> float | None:
    if (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    ):
        return float(value)
    return None


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
        "compile_init_wall_time_sec": mpc.get("compile_init_wall_time_sec"),
        "jit_warmup_enabled": mpc.get("jit_warmup_enabled"),
        "jit_warmup_wall_time_sec": mpc.get("jit_warmup_wall_time_sec"),
        "runtime_visible_devices": mpc.get("runtime_visible_devices"),
        "runtime_gpu_name": mpc.get("runtime_gpu_name"),
        "steady_state_wall_time_sec": mpc.get("steady_state_wall_time_sec"),
        "control_dt_sec": metrics.get("control_dt_sec"),
        "evaluated_motion_duration_sec": metrics.get("evaluated_motion_duration_sec"),
        "contact_saturated": mpc.get("contact_saturated"),
        "max_contact_points_saturated": mpc.get("max_contact_points_saturated"),
        "max_geom_pairs_saturated": mpc.get("max_geom_pairs_saturated"),
        "max_contact_points": mpc.get("max_contact_points"),
        "max_geom_pairs": mpc.get("max_geom_pairs"),
        "contact_pair_count": mpc.get("contact_pair_count"),
        "active_contact_count": mpc.get("active_contact_count"),
    }


def _mjx_timing_evidence_failures(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    failures: list[str] = []
    for row in rows:
        if not _valid_timing(_timing_value(row, "compile_init_wall_time_sec")):
            failures.append("mjx_compile_init_wall_time")
        if row.get("jit_warmup_enabled") is not True:
            failures.append("mjx_jit_warmup_enabled")
        if not _valid_timing(_timing_value(row, "jit_warmup_wall_time_sec")):
            failures.append("mjx_jit_warmup_wall_time")
    return _unique(failures)


def _mjx_runtime_evidence_failures(
    rows: list[dict[str, Any]],
    *,
    required_gpu_name_fragment: str,
) -> tuple[str, ...]:
    failures: list[str] = []
    required = str(required_gpu_name_fragment)
    for row in rows:
        devices = row.get("runtime_visible_devices")
        if not isinstance(devices, (list, tuple)) or not devices:
            failures.append("mjx_runtime_visible_devices")
            continue
        visible = tuple(str(value) for value in devices if str(value))
        if len(visible) != 1:
            failures.append("mjx_single_visible_gpu")
        gpu_name = row.get("runtime_gpu_name")
        if not isinstance(gpu_name, str) or not gpu_name.strip():
            failures.append("mjx_runtime_gpu_name")
        elif required and required not in gpu_name:
            failures.append("mjx_required_gpu")
    return _unique(failures)


def _mjx_contact_evidence_failures(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    failures: list[str] = []
    for row in rows:
        for field in MJX_CONTACT_SATURATION_FIELDS:
            if not isinstance(_row_value(row, field), bool):
                failures.append("mjx_contact_diagnostics")
        counts = {field: _row_value(row, field) for field in MJX_CONTACT_COUNT_FIELDS}
        for field, value in counts.items():
            if not _valid_contact_count(
                value,
                positive=field in {"max_contact_points", "max_geom_pairs"},
            ):
                failures.append("mjx_contact_diagnostics")
        if (
            _valid_contact_count(counts["active_contact_count"])
            and _valid_contact_count(counts["max_contact_points"], positive=True)
            and float(counts["active_contact_count"])
            > float(counts["max_contact_points"])
        ):
            failures.append("mjx_contact_diagnostics")
        if (
            _valid_contact_count(counts["contact_pair_count"])
            and _valid_contact_count(counts["max_geom_pairs"], positive=True)
            and float(counts["contact_pair_count"])
            > float(counts["max_geom_pairs"])
        ):
            failures.append("mjx_contact_diagnostics")
    return _unique(failures)


def _baseline_artifact_evidence_failures(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    failures: list[str] = []
    for row in rows:
        artifacts = row.get("artifacts", {})
        for key in REQUIRED_ARTIFACT_FIELDS:
            if not isinstance(artifacts, dict):
                failures.append(key)
                continue
            value = artifacts.get(key)
            if not isinstance(value, str) or not Path(value).expanduser().is_file():
                failures.append(key)
    return _unique(failures)


def _baseline_runtime_evidence_failures(
    rows: list[dict[str, Any]],
    *,
    required_gpu_name_fragment: str,
) -> tuple[str, ...]:
    failures: list[str] = []
    required = str(required_gpu_name_fragment)
    for row in rows:
        devices = row.get("runtime_visible_devices")
        if not isinstance(devices, (list, tuple)) or not devices:
            failures.append("baseline_runtime_visible_devices")
        else:
            visible = tuple(str(value) for value in devices if str(value))
            if len(visible) != 1:
                failures.append("baseline_single_visible_gpu")
        gpu_name = row.get("runtime_gpu_name")
        if not isinstance(gpu_name, str) or not gpu_name.strip():
            failures.append("baseline_runtime_gpu_name")
        elif required and required not in gpu_name:
            failures.append("baseline_required_gpu")
    return _unique(failures)


def _artifact_freshness_failures(
    rows: list[dict[str, Any]],
    *,
    artifact_fields: tuple[str, ...],
) -> tuple[str, ...]:
    failures: list[str] = []
    for row in rows:
        start_ns = _safe_int(row.get("command_start_time_ns"))
        if start_ns is None:
            continue
        mtimes = row.get("artifact_mtime_ns", {})
        if not isinstance(mtimes, dict):
            mtimes = {}
        for key in artifact_fields:
            mtime_ns = _safe_int(mtimes.get(key))
            if mtime_ns is None:
                continue
            if int(mtime_ns) + ARTIFACT_FRESHNESS_TOLERANCE_NS < int(start_ns):
                failures.append(f"{key}_stale")
    return _unique(failures)


def _artifact_hash_failures(
    rows: list[dict[str, Any]],
    *,
    artifact_fields: tuple[str, ...],
) -> tuple[str, ...]:
    failures: list[str] = []
    for row in rows:
        artifacts = row.get("artifacts", {})
        expected_hashes = row.get("artifact_sha256", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        if not isinstance(expected_hashes, dict):
            expected_hashes = {}
        for key in artifact_fields:
            expected = expected_hashes.get(key)
            if not isinstance(expected, str) or not expected.strip():
                failures.append(f"{key}_hash")
                continue
            artifact = artifacts.get(key)
            if not isinstance(artifact, str) or not Path(artifact).expanduser().is_file():
                continue
            actual = _file_sha256(Path(artifact).expanduser())
            if actual != expected:
                failures.append(f"{key}_hash")
    return _unique(failures)


def _artifact_npz_schema_failures(
    rows: list[dict[str, Any]],
    *,
    require_command: bool,
) -> tuple[str, ...]:
    failures: list[str] = []
    for row in rows:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        num_steps = _safe_int(row.get("num_steps", metrics.get("num_steps")))
        artifacts = row.get("artifacts", {})
        if not isinstance(artifacts, dict) or num_steps is None:
            continue
        rollout_path = artifacts.get("rollout_npz")
        rollout_valid = False
        if isinstance(rollout_path, str) and Path(rollout_path).expanduser().is_file():
            rollout_valid = _valid_rollout_npz(
                Path(rollout_path).expanduser(),
                num_steps=num_steps,
            )
            if not rollout_valid:
                failures.append("rollout_npz_schema")
        if require_command:
            command_path = artifacts.get("mpc_command_npz")
            command_valid = False
            if isinstance(command_path, str) and Path(command_path).expanduser().is_file():
                command_valid = _valid_command_npz(
                    Path(command_path).expanduser(),
                    num_steps=num_steps,
                )
                if not command_valid:
                    failures.append("mpc_command_npz_schema")
            if (
                rollout_valid
                and command_valid
                and isinstance(rollout_path, str)
                and isinstance(command_path, str)
                and not _rollout_matches_command_npz(
                    Path(rollout_path).expanduser(),
                    Path(command_path).expanduser(),
                )
            ):
                failures.append("mpc_rollout_qpos_mismatch")
    return _unique(failures)


def _valid_rollout_npz(path: Path, *, num_steps: int) -> bool:
    frames = int(num_steps) + 1
    bodies = len(MUJOCO_BODY_NAMES)
    required_shapes = {
        "qpos": (frames, 1, QPOS_DIM),
        "qvel": (frames, 1, QVEL_DIM),
        "body_pos_w": (frames, 1, bodies, 3),
        "body_quat_w": (frames, 1, bodies, 4),
        "body_lin_vel_w": (frames, 1, bodies, 3),
        "body_ang_vel_w": (frames, 1, bodies, 3),
        "actions": (int(num_steps), 1, ACTION_DIM),
        "controls": (int(num_steps), 1, ACTION_DIM),
        "contact_indicator": (frames, 1, 2),
        "contact_force": (frames, 1, 2),
        "floor_contact_indicator": (frames, 1, 3),
        "floor_contact_force": (frames, 1, 3),
        "ref_indices": (frames, 1),
    }
    try:
        with np.load(path) as data:
            if "dt" not in data.files or np.asarray(data["dt"]).shape not in {(), (1,)}:
                return False
            return _npz_has_shapes(data, required_shapes)
    except Exception:
        return False


def _valid_command_npz(path: Path, *, num_steps: int) -> bool:
    frames = int(num_steps) + 1
    bodies = len(MUJOCO_BODY_NAMES)
    required_shapes = {
        "refined_qpos": ((frames, QPOS_DIM), (frames, 1, QPOS_DIM)),
        "candidate_scores": None,
        "command_joint_pos": (frames, 1, ACTION_DIM),
        "command_joint_vel": (frames, 1, ACTION_DIM),
        "command_body_pos_w": (frames, 1, bodies, 3),
        "command_body_quat_w": (frames, 1, bodies, 4),
        "command_body_lin_vel_w": (frames, 1, bodies, 3),
        "command_body_ang_vel_w": (frames, 1, bodies, 3),
        "command_qpos_trajectory": ((frames, QPOS_DIM), (frames, 1, QPOS_DIM)),
        "command_qvel_trajectory": ((frames, QVEL_DIM), (frames, 1, QVEL_DIM)),
    }
    try:
        with np.load(path) as data:
            return _npz_has_shapes(data, required_shapes)
    except Exception:
        return False


def _rollout_matches_command_npz(rollout_path: Path, command_path: Path) -> bool:
    try:
        with np.load(rollout_path) as rollout, np.load(command_path) as command:
            rollout_qpos = np.asarray(rollout["qpos"])[:, 0]
            refined_qpos = np.asarray(command["refined_qpos"])
            if refined_qpos.ndim == 3 and refined_qpos.shape[1] == 1:
                refined_qpos = refined_qpos[:, 0]
            if rollout_qpos.shape != refined_qpos.shape:
                return False
            return bool(
                np.allclose(
                    rollout_qpos,
                    refined_qpos,
                    atol=1.0e-5,
                    rtol=1.0e-5,
                )
            )
    except Exception:
        return False


def _npz_has_shapes(
    data,
    required_shapes: dict[str, tuple[int, ...] | tuple[tuple[int, ...], ...] | None],
) -> bool:
    for key, expected in required_shapes.items():
        if key not in data.files:
            return False
        if expected is None:
            if np.asarray(data[key]).size <= 0:
                return False
            continue
        observed = tuple(np.asarray(data[key]).shape)
        allowed = expected if expected and isinstance(expected[0], tuple) else (expected,)
        if observed not in allowed:
            return False
    return True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return _row_value(row, name)


def _row_value(row: dict[str, Any], name: str):
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


def _contact_count_values(rows: list[dict[str, Any]], name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _row_value(row, name)
        if _valid_contact_count(value):
            values.append(float(value))
    return values


def _contact_flag_value(row: dict[str, Any], name: str) -> bool | None:
    value = _row_value(row, name)
    return value if isinstance(value, bool) else None


def _valid_contact_count(value, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if not math.isfinite(float(value)):
        return False
    lower_bound = 1.0 if positive else 0.0
    if float(value) < lower_bound:
        return False
    return float(value).is_integer()


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
    artifact_paths = {
        "metrics_json": output_dir / "metrics.json",
        "rollout_npz": output_dir / "rollout.npz",
        "mpc_command_npz": output_dir / "mpc_command.npz",
    }
    row["artifacts"] = {
        key: _existing_path(path)
        for key, path in artifact_paths.items()
    }
    row["artifact_mtime_ns"] = {
        key: path.stat().st_mtime_ns if path.exists() else None
        for key, path in artifact_paths.items()
    }
    row["artifact_sha256"] = {
        key: _file_sha256(path)
        for key, path in artifact_paths.items()
        if path.exists()
    }
    return row


def _existing_path(path: Path) -> str | None:
    return str(path.resolve()) if path.exists() else None


def _output_dir_from_argv(argv: list[str]) -> Path:
    try:
        return Path(argv[argv.index("--output-dir") + 1]).expanduser()
    except (ValueError, IndexError) as exc:
        raise ValueError("Command argv is missing --output-dir") from exc


def _argv_value(argv: Any, flag: str) -> str | None:
    if not isinstance(argv, list):
        return None
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def _same_path(left: str | None, right: Any) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _is_existing_file(path: str | None) -> bool:
    return isinstance(path, str) and Path(path).expanduser().is_file()


def _is_sha256_hex(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


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


def _drop_flag(argv: list[str], flag: str) -> list[str]:
    return [value for value in argv if value != flag]


if __name__ == "__main__":
    raise SystemExit(main())
