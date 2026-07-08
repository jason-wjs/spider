#!/usr/bin/env python3
"""Check same-configuration repeat stability for saved G1 WBC MJX rows."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.diagnose_g1_wbc_mjx_command_divergence import (
    DEFAULT_THRESHOLDS,
    PRIMARY_METRICS,
    analyze_mjx_command_divergence,
)


DEFAULT_METRIC_RANGE_THRESHOLDS = {
    "score": 0.02,
    "root_pos_error_mean": 0.005,
    "body_global_pos_error_mean": 0.005,
    "ee_global_pos_error_mean": 0.005,
    "ee_local_pos_error_mean": 0.003,
    "contact_mismatch_rate": 0.01,
    "control_delta_mean": 0.02,
    "joint_acc_mean": 10.0,
    "contact_force_active_mean": 10.0,
    "contact_force_peak": 100.0,
}


def main() -> None:
    args = _parse_args()
    report = analyze_mjx_repeat_stability(
        args.run,
        target=args.target,
        expected_steps=args.expected_steps,
        expected_windows=args.expected_windows,
        command_thresholds=tuple(args.thresholds),
        max_command_qpos_delta=args.max_command_qpos_delta,
        metric_range_thresholds=_metric_range_thresholds_from_args(args),
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)


def analyze_mjx_repeat_stability(
    runs: Sequence[str | Path],
    *,
    target: str = "diagnostic_repeat_stability",
    expected_steps: int = 800,
    expected_windows: int = 40,
    command_thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
    max_command_qpos_delta: float = 1.0e-3,
    metric_range_thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Return no-GPU diagnostics for repeated MJX rows with the same argv."""

    thresholds = dict(DEFAULT_METRIC_RANGE_THRESHOLDS)
    if metric_range_thresholds is not None:
        thresholds.update(
            {str(key): float(value) for key, value in metric_range_thresholds.items()}
        )

    resolved = [_resolve_repeat_run(path) for path in runs]
    metric_ranges = _metric_ranges(resolved)
    pairwise = _pairwise_reports(resolved, thresholds=command_thresholds)
    failures = _stability_failures(
        resolved,
        metric_ranges,
        pairwise,
        expected_steps=expected_steps,
        expected_windows=expected_windows,
        max_command_qpos_delta=max_command_qpos_delta,
        metric_range_thresholds=thresholds,
    )
    classification = _classification(failures)
    command_content_hashes = [
        run["artifacts"].get("mpc_command_npz_content_sha256") for run in resolved
    ]
    return {
        "schema_version": 1,
        "gate": "offline_repeat_stability",
        "target": str(target),
        "passed": not failures,
        "classification": classification,
        "failures": failures,
        "run_count": len(resolved),
        "expected_steps": int(expected_steps),
        "expected_windows": int(expected_windows),
        "thresholds": {
            "command": {
                "reported_norm_first_exceed": [float(value) for value in command_thresholds],
                "max_command_qpos_delta": float(max_command_qpos_delta),
            },
            "metric_ranges": thresholds,
        },
        "all_command_content_hashes_equal": len(set(command_content_hashes)) <= 1,
        "runs": resolved,
        "metric_ranges": metric_ranges,
        "pairwise": pairwise,
    }


def _resolve_repeat_run(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    root = source
    divergence_input = source
    row: dict[str, Any] = {}
    if root.is_file() and root.name == "acceptance_row.json":
        row = _read_json(root)
        artifacts = row.get("artifacts", {}) if isinstance(row, dict) else {}
        metrics_path = (
            Path(str(artifacts.get("metrics_json", ""))).expanduser().resolve()
        )
        command_path = (
            Path(str(artifacts.get("mpc_command_npz", ""))).expanduser().resolve()
        )
        divergence_input = root
    else:
        if root.is_file() and root.name == "metrics.json":
            root = root.parent
            divergence_input = source
        elif root.is_dir() and not (root / "metrics.json").is_file():
            matches = sorted(root.glob("*/*/mjx/metrics.json"))
            if len(matches) == 1:
                root = matches[0].parent
                divergence_input = root
        metrics_path = root / "metrics.json"
        command_path = root / "mpc_command.npz"
    payload = _read_json(metrics_path) if metrics_path.is_file() else {}
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    mpc = payload.get("mpc", {}) if isinstance(payload, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(mpc, dict):
        mpc = {}
    artifacts = {
        "metrics_json": str(metrics_path),
        "metrics_json_exists": metrics_path.is_file(),
        "mpc_command_npz": str(command_path),
        "mpc_command_npz_exists": command_path.is_file(),
        "mpc_command_npz_sha256": (
            _file_sha256(command_path) if command_path.is_file() else None
        ),
        "mpc_command_npz_content_sha256": (
            _npz_content_sha256(command_path) if command_path.is_file() else None
        ),
    }
    return {
        "input": str(root),
        "divergence_input": str(divergence_input),
        "status": row.get("status") if isinstance(row, dict) else None,
        "returncode": row.get("returncode") if isinstance(row, dict) else None,
        "artifacts": artifacts,
        "metrics": {
            name: _finite_float_or_none(metrics.get(name)) for name in PRIMARY_METRICS
        },
        "basic_gate": _basic_gate(
            row=row,
            metrics=metrics,
            mpc=mpc,
            artifacts=artifacts,
        ),
        "mpc": {
            "accepted": mpc.get("accepted"),
            "accepted_windows": _int_or_none(mpc.get("accepted_windows")),
            "used_baseline_fallback": mpc.get("used_baseline_fallback"),
            "contact_saturated": mpc.get("contact_saturated"),
            "max_contact_points_saturated": mpc.get("max_contact_points_saturated"),
            "max_geom_pairs_saturated": mpc.get("max_geom_pairs_saturated"),
            "steady_state_wall_time_sec": _finite_float_or_none(
                mpc.get("steady_state_wall_time_sec")
            ),
            "mjx_impl": mpc.get("mjx_impl"),
            "mjx_model_impl": mpc.get("mjx_model_impl"),
            "mjx_warp_naconmax": mpc.get("mjx_warp_naconmax"),
            "mjx_warp_njmax": mpc.get("mjx_warp_njmax"),
        },
        "num_steps": _int_or_none(metrics.get("num_steps")),
    }


def _basic_gate(
    *,
    row: Mapping[str, Any],
    metrics: Mapping[str, Any],
    mpc: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    if row:
        if row.get("status") != "ok":
            failures.append("status")
        if row.get("returncode") != 0:
            failures.append("returncode")
    if not artifacts.get("metrics_json_exists"):
        failures.append("metrics_json")
    if not artifacts.get("mpc_command_npz_exists"):
        failures.append("mpc_command_npz")
    for name in PRIMARY_METRICS:
        if _finite_float_or_none(metrics.get(name)) is None:
            failures.append(f"{name}_missing")
    if mpc.get("accepted") is not True:
        failures.append("mpc_accepted")
    if _int_or_none(mpc.get("accepted_windows")) is None:
        failures.append("accepted_windows")
    if mpc.get("used_baseline_fallback") is not False:
        failures.append("baseline_fallback")
    for name in (
        "contact_saturated",
        "max_contact_points_saturated",
        "max_geom_pairs_saturated",
    ):
        if mpc.get(name) is not False:
            failures.append(name)
    return {"passed": not failures, "failures": _unique(failures)}


def _metric_ranges(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in PRIMARY_METRICS:
        values = [
            _finite_float_or_none(run.get("metrics", {}).get(name))
            for run in runs
            if isinstance(run.get("metrics"), dict)
        ]
        finite = [value for value in values if value is not None]
        if not finite:
            summary[name] = {"values": values, "min": None, "max": None, "range": None}
            continue
        minimum = min(finite)
        maximum = max(finite)
        summary[name] = {
            "values": values,
            "min": minimum,
            "max": maximum,
            "range": float(maximum - minimum),
            "mean": float(sum(finite) / len(finite)),
        }
    return summary


def _pairwise_reports(
    runs: Sequence[Mapping[str, Any]],
    *,
    thresholds: tuple[float, ...],
) -> list[dict[str, Any]]:
    reports = []
    for left_index, right_index in itertools.combinations(range(len(runs)), 2):
        left = runs[left_index]
        right = runs[right_index]
        left_input = Path(str(left["divergence_input"]))
        right_input = Path(str(right["divergence_input"]))
        if not left_input.exists() or not right_input.exists():
            reports.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "error": "missing_divergence_input",
                }
            )
            continue
        try:
            report = analyze_mjx_command_divergence(
                left_input,
                right_input,
                thresholds=thresholds,
            )
        except (OSError, ValueError, KeyError) as exc:
            reports.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "error": str(exc),
                }
            )
            continue
        reports.append(
            {
                "left_index": left_index,
                "right_index": right_index,
                "first_best_index_divergence": report["history"].get(
                    "first_best_index_divergence"
                ),
                "best_index_divergence_count": report["history"].get(
                    "best_index_divergence_count"
                ),
                "first_current_selected_divergence": report["history"].get(
                    "first_current_selected_divergence"
                ),
                "current_selected_divergence_count": report["history"].get(
                    "current_selected_divergence_count"
                ),
                "command_qpos_trajectory": report["command"].get(
                    "command_qpos_trajectory"
                ),
                "metrics": report["metrics"],
            }
        )
    return reports


def _stability_failures(
    runs: Sequence[Mapping[str, Any]],
    metric_ranges: Mapping[str, Any],
    pairwise: Sequence[Mapping[str, Any]],
    *,
    expected_steps: int,
    expected_windows: int,
    max_command_qpos_delta: float,
    metric_range_thresholds: Mapping[str, float],
) -> list[str]:
    failures: list[str] = []
    if len(runs) < 2:
        failures.append("run_count")
    for index, run in enumerate(runs):
        basic = run.get("basic_gate", {})
        for failure in basic.get("failures", []) if isinstance(basic, dict) else []:
            failures.append(f"run_{index}:{failure}")
        if _int_or_none(run.get("num_steps")) != int(expected_steps):
            failures.append(f"run_{index}:num_steps")
        mpc = run.get("mpc", {})
        if not isinstance(mpc, dict):
            mpc = {}
        if _int_or_none(mpc.get("accepted_windows")) != int(expected_windows):
            failures.append(f"run_{index}:accepted_windows")

    for name, threshold in metric_range_thresholds.items():
        row = metric_ranges.get(name)
        if not isinstance(row, dict):
            continue
        value_range = _finite_float_or_none(row.get("range"))
        if value_range is not None and value_range > float(threshold):
            failures.append(f"metric_range:{name}")

    for pair in pairwise:
        left = pair.get("left_index")
        right = pair.get("right_index")
        prefix = f"pair_{left}_{right}"
        if "error" in pair:
            failures.append(f"{prefix}:pairwise_error")
            continue
        if pair.get("first_best_index_divergence") is not None:
            failures.append(f"{prefix}:best_index_divergence")
        if pair.get("first_current_selected_divergence") is not None:
            failures.append(f"{prefix}:current_selected_divergence")
        qpos = pair.get("command_qpos_trajectory")
        if isinstance(qpos, dict):
            qpos_max = _finite_float_or_none(qpos.get("max"))
            if qpos_max is not None and qpos_max > float(max_command_qpos_delta):
                failures.append(f"{prefix}:command_qpos_delta")
        else:
            failures.append(f"{prefix}:command_qpos_trajectory")
    return _unique(failures)


def _classification(failures: Sequence[str]) -> str:
    if not failures:
        return "pass"
    if any(
        failure.endswith(
            (
                "metrics_json",
                "mpc_command_npz",
                "num_steps",
                "accepted_windows",
                "status",
                "returncode",
            )
        )
        or "pairwise_error" in failure
        for failure in failures
    ):
        return "invalid_artifacts"
    if any(
        "best_index_divergence" in failure
        or "current_selected_divergence" in failure
        or "command_qpos_delta" in failure
        for failure in failures
    ):
        return "repeat_instability"
    if any(failure.startswith("metric_range:") for failure in failures):
        return "metric_instability"
    return "invalid_artifacts"


def _metric_range_thresholds_from_args(args: argparse.Namespace) -> dict[str, float]:
    thresholds = dict(DEFAULT_METRIC_RANGE_THRESHOLDS)
    thresholds.update(
        {
            "score": float(args.max_score_range),
            "root_pos_error_mean": float(args.max_root_range),
            "body_global_pos_error_mean": float(args.max_body_range),
            "ee_global_pos_error_mean": float(args.max_ee_global_range),
            "ee_local_pos_error_mean": float(args.max_ee_local_range),
            "contact_mismatch_rate": float(args.max_contact_mismatch_range),
            "contact_force_active_mean": float(args.max_contact_force_active_range),
            "contact_force_peak": float(args.max_contact_force_peak_range),
        }
    )
    return thresholds


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for name in sorted(data.files):
            array = np.asarray(data[name])
            digest.update(name.encode("utf-8"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _finite_float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values))


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=Path,
        required=True,
        help="MJX row dir, metrics.json, or acceptance_row.json. Repeat at least twice.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--target",
        default="diagnostic_repeat_stability",
        help="Reader-facing target label recorded in the JSON report.",
    )
    parser.add_argument("--expected-steps", type=int, default=800)
    parser.add_argument("--expected-windows", type=int, default=40)
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        action="append",
        type=_non_negative_float,
        default=[],
        help="Command norm threshold for first-exceed reporting. Repeatable.",
    )
    parser.add_argument(
        "--max-command-qpos-delta",
        type=_non_negative_float,
        default=1.0e-3,
    )
    parser.add_argument("--max-score-range", type=_non_negative_float, default=0.02)
    parser.add_argument("--max-root-range", type=_non_negative_float, default=0.005)
    parser.add_argument("--max-body-range", type=_non_negative_float, default=0.005)
    parser.add_argument(
        "--max-ee-global-range",
        type=_non_negative_float,
        default=0.005,
    )
    parser.add_argument(
        "--max-ee-local-range",
        type=_non_negative_float,
        default=0.003,
    )
    parser.add_argument(
        "--max-contact-mismatch-range",
        type=_non_negative_float,
        default=0.01,
    )
    parser.add_argument(
        "--max-contact-force-active-range",
        type=_non_negative_float,
        default=10.0,
    )
    parser.add_argument(
        "--max-contact-force-peak-range",
        type=_non_negative_float,
        default=100.0,
    )
    args = parser.parse_args()
    if not args.thresholds:
        args.thresholds = list(DEFAULT_THRESHOLDS)
    return args


if __name__ == "__main__":
    main()
