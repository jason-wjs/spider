#!/usr/bin/env python3
"""Compare repeated G1 WBC MJX acceptance reports row by row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.check_g1_wbc_mjx_repeat_stability import analyze_mjx_repeat_stability


PATH_VALUE_ARGS = {
    "--baseline-manifest",
    "--output-dir",
}


def main() -> None:
    args = _parse_args()
    report = analyze_acceptance_repeat_stability(
        args.report,
        motions=args.motion,
        seeds=args.seed,
        target=args.target,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)


def analyze_acceptance_repeat_stability(
    reports: Sequence[str | Path],
    *,
    motions: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    target: str = "acceptance_repeat_stability",
) -> dict[str, Any]:
    """Return no-GPU repeat diagnostics for matching MJX rows in reports."""

    loaded = [_load_report(path) for path in reports]
    selected_keys = _selected_keys(loaded, motions=motions, seeds=seeds)
    pairs: list[dict[str, Any]] = []
    for motion, seed in selected_keys:
        rows = [_row_by_key(report["mjx_rows"], motion=motion, seed=seed) for report in loaded]
        missing = [index for index, row in enumerate(rows) if row is None]
        if missing:
            pairs.append(
                {
                    "motion": motion,
                    "seed": seed,
                    "status": "missing_rows",
                    "missing_report_indices": missing,
                    "passed": False,
                }
            )
            continue
        concrete_rows = [row for row in rows if row is not None]
        normalized_argvs = [
            normalize_mjx_argv_for_repeat(_row_argv(row)) for row in concrete_rows
        ]
        argv_equivalent = len({tuple(argv) for argv in normalized_argvs}) <= 1
        pair: dict[str, Any] = {
            "motion": motion,
            "seed": seed,
            "status": "pending",
            "passed": False,
            "argv_equivalent": argv_equivalent,
            "normalized_argv": normalized_argvs[0] if normalized_argvs else [],
        }
        if not argv_equivalent:
            pair.update(
                {
                    "status": "configuration_mismatch",
                    "normalized_argvs": normalized_argvs,
                }
            )
            pairs.append(pair)
            continue
        repeat_report = analyze_mjx_repeat_stability(
            [_row_repeat_input(row) for row in concrete_rows],
            target=f"{target}:{motion}:seed_{seed}",
        )
        pair.update(
            {
                "status": repeat_report["classification"],
                "passed": bool(repeat_report["passed"]),
                "repeat_report": repeat_report,
            }
        )
        pairs.append(pair)
    failures = [pair["status"] for pair in pairs if not pair.get("passed")]
    return {
        "schema_version": 1,
        "gate": "acceptance_report_repeat_stability",
        "target": target,
        "report_paths": [str(Path(path).expanduser().resolve()) for path in reports],
        "report_count": len(loaded),
        "pair_count": len(pairs),
        "passed": not failures,
        "classification": _classification(failures),
        "pairs": pairs,
    }


def normalize_mjx_argv_for_repeat(argv: Sequence[Any]) -> list[str]:
    normalized: list[str] = []
    iterator = iter(str(item) for item in argv)
    for item in iterator:
        if item in PATH_VALUE_ARGS:
            next(iterator, None)
            continue
        normalized.append(item)
    return normalized


def _load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"acceptance report is not a JSON object: {report_path}")
    rows = payload.get("mjx_rows")
    if not isinstance(rows, list):
        raise ValueError(f"acceptance report lacks mjx_rows: {report_path}")
    return {
        "path": report_path,
        "mjx_rows": [row for row in rows if isinstance(row, dict)],
    }


def _selected_keys(
    reports: Sequence[Mapping[str, Any]],
    *,
    motions: Sequence[str] | None,
    seeds: Sequence[int] | None,
) -> list[tuple[str, int]]:
    motion_filter = {str(motion) for motion in motions} if motions else None
    seed_filter = {int(seed) for seed in seeds} if seeds else None
    keys: set[tuple[str, int]] = set()
    for report in reports:
        rows = report.get("mjx_rows", [])
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            motion = row.get("motion")
            seed = _int_or_none(row.get("seed"))
            if not isinstance(motion, str) or seed is None:
                continue
            if motion_filter is not None and motion not in motion_filter:
                continue
            if seed_filter is not None and seed not in seed_filter:
                continue
            keys.add((motion, seed))
    return sorted(keys)


def _row_by_key(
    rows: Sequence[Mapping[str, Any]],
    *,
    motion: str,
    seed: int,
) -> Mapping[str, Any] | None:
    for row in rows:
        if row.get("motion") == motion and _int_or_none(row.get("seed")) == seed:
            return row
    return None


def _row_argv(row: Mapping[str, Any]) -> Sequence[Any]:
    argv = row.get("mjx_argv")
    if isinstance(argv, list):
        return argv
    argv = row.get("argv")
    return argv if isinstance(argv, list) else []


def _row_repeat_input(row: Mapping[str, Any]) -> Path:
    artifacts = row.get("artifacts")
    if isinstance(artifacts, dict):
        metrics = artifacts.get("metrics_json")
        if isinstance(metrics, str):
            return Path(metrics).expanduser().resolve()
    output_dir = row.get("output_dir")
    if isinstance(output_dir, str):
        return Path(output_dir).expanduser().resolve()
    raise ValueError(
        f"MJX row lacks artifacts.metrics_json and output_dir: "
        f"{row.get('motion')} seed={row.get('seed')}"
    )


def _classification(failures: Sequence[str]) -> str:
    if not failures:
        return "pass"
    if "missing_rows" in failures:
        return "missing_rows"
    if "configuration_mismatch" in failures:
        return "configuration_mismatch"
    if "repeat_instability" in failures:
        return "repeat_instability"
    if "metric_instability" in failures:
        return "metric_instability"
    return "invalid_artifacts"


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        type=Path,
        required=True,
        help="Acceptance report JSON. Repeat at least twice.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--target",
        default="acceptance_repeat_stability",
        help="Reader-facing target label recorded in the JSON report.",
    )
    parser.add_argument(
        "--motion",
        action="append",
        default=None,
        help="Restrict comparison to one motion. Repeatable.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        default=None,
        help="Restrict comparison to one seed. Repeatable.",
    )
    args = parser.parse_args()
    if len(args.report) < 2:
        parser.error("--report must be provided at least twice.")
    return args


if __name__ == "__main__":
    main()
