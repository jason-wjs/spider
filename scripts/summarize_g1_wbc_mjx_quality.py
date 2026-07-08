#!/usr/bin/env python3
"""Summarize G1 WBC MJX quality, force, contact, and speed evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from spider.tasks.g1_wbc.acceptance import MjxQualityPolicy

MOTIONS = ("jump", "walk")
SEEDS = (0, 1, 2)
CONTACT_FORCE_SEMANTICS = "pyramidal_contact_normal_v1"
PRIMARY_QUALITY_FIELDS = (
    "score",
    "root_pos_error_mean",
    "body_global_pos_error_mean",
    "ee_global_pos_error_mean",
    "ee_local_pos_error_mean",
    "contact_mismatch_rate",
    "control_delta_mean",
    "joint_acc_mean",
)
FORCE_RATIO_HIGH = "force_ratio_high"
FORCE_RATIO_NOT_ISOLATED = "force_ratio_not_isolated"
FORCE_RATIO_MISSING = "force_ratio_missing"
ISOLATED_RAW_MJX_FORCE_HIGH = "isolated_raw_mjx_force_high"
SHARED_FORCE_HIGH = "shared_force_high"
REPLAY_FORCE_HIGH = "replay_force_high"
FORCE_RATIO_NOMINAL = "force_ratio_nominal"
FORCE_MJX_HIGH_RATIO_MIN = 3.0
FORCE_REPLAY_NEAR_BASELINE_RATIO_MAX = 1.5


def summarize_quality_runs(
    *,
    baseline_manifest: Path,
    acceptance_root: Path,
    min_speedup: float = 11.0,
) -> dict[str, Any]:
    """Return a compact quality-first summary for matrix-shaped MJX results."""

    manifest = json.loads(Path(baseline_manifest).read_text())
    contact_force_semantics = manifest.get("contact_force_semantics")
    if contact_force_semantics != CONTACT_FORCE_SEMANTICS:
        raise ValueError(
            "baseline manifest contact_force_semantics must be "
            f"{CONTACT_FORCE_SEMANTICS}"
        )
    acceptance_root = Path(acceptance_root)
    baseline_rows = {
        (str(row["motion_name"]), int(row["seed"])): row
        for row in manifest.get("rows", [])
    }
    rows = []
    for motion in MOTIONS:
        for seed in SEEDS:
            key = (motion, seed)
            if key not in baseline_rows:
                continue
            mjx_metrics = acceptance_root / motion / f"seed_{seed}" / "mjx" / "metrics.json"
            if not mjx_metrics.is_file():
                continue
            rows.append(_summarize_row(manifest, baseline_rows[key], mjx_metrics))
    return {
        "schema_version": 1,
        "baseline_manifest": str(Path(baseline_manifest).expanduser().resolve()),
        "acceptance_root": str(acceptance_root.expanduser().resolve()),
        "contact_force_semantics": contact_force_semantics,
        "min_speedup": float(min_speedup),
        "rows": rows,
        "passed_speed_floor": all(
            row["same_seed_speedup"] >= float(min_speedup) for row in rows
        ),
        "quality_warning_count": sum(len(row["quality_warnings"]) for row in rows),
        "replay_quality_warning_count": sum(
            len(row["replay_quality_warnings"]) for row in rows
        ),
        "force_ratio_high_count": sum(
            _force_ratio_high_count(row.get("force_semantics")) for row in rows
        ),
        "force_semantics_summary": _force_semantics_summary(rows),
    }


def _summarize_row(
    manifest: dict[str, Any],
    baseline_row: dict[str, Any],
    mjx_metrics_path: Path,
) -> dict[str, Any]:
    motion = str(baseline_row["motion_name"])
    seed = int(baseline_row["seed"])
    baseline_payload = json.loads(
        Path(baseline_row["artifacts"]["metrics_json"]).read_text()
    )
    mjx_payload = json.loads(mjx_metrics_path.read_text())
    replay_metrics_path = mjx_metrics_path.parent.parent / "replay" / "metrics.json"
    replay_payload = (
        json.loads(replay_metrics_path.read_text())
        if replay_metrics_path.is_file()
        else None
    )
    baseline_metrics = baseline_payload.get("metrics", {})
    mjx_metrics = mjx_payload.get("metrics", {})
    replay_metrics = replay_payload.get("metrics", {}) if replay_payload else {}
    mjx_mpc = mjx_payload.get("mpc", {})
    baseline_steady = float(baseline_row["steady_state_wall_time_sec"])
    mjx_steady = float(mjx_mpc["steady_state_wall_time_sec"])
    replay_quality_warnings = (
        _quality_warnings(
            motion,
            manifest.get("baseline_envelopes", {}).get(motion, {}),
            replay_metrics,
        )
        if replay_payload
        else []
    )
    force_semantics = _force_semantics(
        baseline_metrics=baseline_metrics,
        mjx_metrics=mjx_metrics,
        replay_metrics=replay_metrics if replay_payload else None,
    )
    return {
        "motion": motion,
        "seed": seed,
        "contact_force_semantics": manifest.get("contact_force_semantics"),
        "baseline_steady_state_wall_time_sec": baseline_steady,
        "mjx_steady_state_wall_time_sec": mjx_steady,
        "same_seed_speedup": baseline_steady / mjx_steady,
        "realtime_factor": 16.0 / mjx_steady,
        "quality": {
            field: mjx_metrics.get(field)
            for field in PRIMARY_QUALITY_FIELDS
            if field in mjx_metrics
        },
        "quality_warnings": _quality_warnings(
            motion,
            manifest.get("baseline_envelopes", {}).get(motion, {}),
            mjx_metrics,
        ),
        "replay_quality": (
            {
                field: replay_metrics.get(field)
                for field in PRIMARY_QUALITY_FIELDS
                if field in replay_metrics
            }
            if replay_payload
            else None
        ),
        "replay_quality_warnings": replay_quality_warnings,
        "baseline_force_peak": baseline_metrics.get("contact_force_peak"),
        "mjx_force_peak": mjx_metrics.get("contact_force_peak"),
        "replay_force_peak": replay_metrics.get("contact_force_peak")
        if replay_payload
        else None,
        "force_peak_ratio": _ratio(
            mjx_metrics.get("contact_force_peak"),
            baseline_metrics.get("contact_force_peak"),
        ),
        "replay_force_peak_ratio": _ratio(
            replay_metrics.get("contact_force_peak"),
            baseline_metrics.get("contact_force_peak"),
        )
        if replay_payload
        else None,
        "baseline_force_active_mean": baseline_metrics.get(
            "contact_force_active_mean"
        ),
        "mjx_force_active_mean": mjx_metrics.get("contact_force_active_mean"),
        "replay_force_active_mean": replay_metrics.get(
            "contact_force_active_mean"
        )
        if replay_payload
        else None,
        "force_active_ratio": _ratio(
            mjx_metrics.get("contact_force_active_mean"),
            baseline_metrics.get("contact_force_active_mean"),
        ),
        "replay_force_active_ratio": _ratio(
            replay_metrics.get("contact_force_active_mean"),
            baseline_metrics.get("contact_force_active_mean"),
        )
        if replay_payload
        else None,
        "force_semantics": force_semantics,
        "contact_saturated": bool(mjx_mpc.get("contact_saturated")),
        "max_contact_points_saturated": bool(
            mjx_mpc.get("max_contact_points_saturated")
        ),
        "max_geom_pairs_saturated": bool(mjx_mpc.get("max_geom_pairs_saturated")),
        "contact_pair_count": mjx_mpc.get("contact_pair_count"),
        "active_contact_count": mjx_mpc.get("active_contact_count"),
    }


def _quality_warnings(
    motion: str,
    envelope: dict[str, Any],
    metrics: dict[str, Any],
) -> list[str]:
    policy = MjxQualityPolicy.for_motion(motion)
    warnings = []
    if _metric_present("score", envelope, metrics):
        value = float(metrics["score"])
        score_mean_floor = float(envelope["score"]["mean"]) + policy.score_mean_allowance
        score_worst_floor = float(envelope["score"]["min"]) + policy.score_worst_allowance
        if value < score_mean_floor:
            warnings.append("score_mean")
        if value < score_worst_floor:
            warnings.append("score_worst")

    for metric in (
        "root_pos_error_mean",
        "body_global_pos_error_mean",
        "ee_global_pos_error_mean",
        "ee_local_pos_error_mean",
    ):
        if not _metric_present(metric, envelope, metrics):
            continue
        value = float(metrics[metric])
        if value > float(envelope[metric]["mean"]) * policy.error_ratio_max:
            warnings.append(metric)
        absolute_cap = policy.absolute_caps.get(metric)
        if absolute_cap is not None and value > absolute_cap:
            warnings.append(f"{metric}_absolute")

    for metric in (
        "contact_mismatch_rate",
        "contact_false_positive_rate",
        "contact_false_negative_rate",
    ):
        if not _metric_present(metric, envelope, metrics):
            continue
        value = float(metrics[metric])
        if value > float(envelope[metric]["mean"]) + policy.contact_abs_allowance:
            warnings.append(metric)

    if _metric_present("bad_floor_contact_rate", envelope, metrics):
        value = float(metrics["bad_floor_contact_rate"])
        if (
            value
            > float(envelope["bad_floor_contact_rate"]["mean"])
            + policy.bad_floor_contact_abs_allowance
        ):
            warnings.append("bad_floor_contact_rate")

    if _metric_present("contact_mismatch_rate", envelope, metrics):
        contact_cap = policy.absolute_caps["contact_mismatch_rate"]
        if float(metrics["contact_mismatch_rate"]) > contact_cap:
            warnings.append("contact_mismatch_rate_absolute")

    for metric in ("control_delta_mean", "joint_acc_mean"):
        if _metric_present(metric, envelope, metrics):
            value = float(metrics[metric])
            if value > float(envelope[metric]["mean"]) * policy.control_ratio_max:
                warnings.append(metric)

    if _metric_present("joint_jerk_mean", envelope, metrics):
        value = float(metrics["joint_jerk_mean"])
        if value > float(envelope["joint_jerk_mean"]["mean"]) * policy.jerk_ratio_max:
            warnings.append("joint_jerk_mean")
    return warnings


def _metric_present(metric: str, envelope: dict[str, Any], metrics: dict[str, Any]) -> bool:
    stats = envelope.get(metric)
    return isinstance(stats, dict) and "mean" in stats and metric in metrics


def _force_semantics(
    *,
    baseline_metrics: dict[str, Any],
    mjx_metrics: dict[str, Any],
    replay_metrics: dict[str, Any] | None,
) -> dict[str, dict[str, float | str | None]] | None:
    if replay_metrics is None:
        return None
    return {
        "active": _classify_force_ratios(
            baseline=baseline_metrics.get("contact_force_active_mean"),
            mjx=mjx_metrics.get("contact_force_active_mean"),
            replay=replay_metrics.get("contact_force_active_mean"),
            name="active",
        ),
        "peak": _classify_force_ratios(
            baseline=baseline_metrics.get("contact_force_peak"),
            mjx=mjx_metrics.get("contact_force_peak"),
            replay=replay_metrics.get("contact_force_peak"),
            name="peak",
        ),
    }


def _classify_force_ratios(
    *,
    baseline: Any,
    mjx: Any,
    replay: Any,
    name: str,
) -> dict[str, float | str | None]:
    mjx_ratio = _ratio(mjx, baseline)
    replay_ratio = _ratio(replay, baseline)
    classification = (
        FORCE_RATIO_HIGH
        if mjx_ratio is not None
        and replay_ratio is not None
        and mjx_ratio >= FORCE_MJX_HIGH_RATIO_MIN
        and replay_ratio <= FORCE_REPLAY_NEAR_BASELINE_RATIO_MAX
        else FORCE_RATIO_NOT_ISOLATED
    )
    return {
        f"mjx_{name}_force_ratio": mjx_ratio,
        f"replay_{name}_force_ratio": replay_ratio,
        "classification": classification,
        "semantic_classification": _force_semantic_classification(
            mjx_ratio=mjx_ratio,
            replay_ratio=replay_ratio,
        ),
    }


def _force_ratio_high_count(force_semantics: Any) -> int:
    if not isinstance(force_semantics, dict):
        return 0
    return sum(
        1
        for value in force_semantics.values()
        if isinstance(value, dict)
        and value.get("classification") == FORCE_RATIO_HIGH
    )


def _force_semantics_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "active": _force_semantics_metric_summary(rows, "active"),
        "peak": _force_semantics_metric_summary(rows, "peak"),
    }


def _force_semantics_metric_summary(
    rows: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    semantic_counts: dict[str, int] = {}
    classification_counts: dict[str, int] = {}
    row_summaries = []
    for row in rows:
        force_semantics = row.get("force_semantics")
        metric_semantics = (
            force_semantics.get(metric) if isinstance(force_semantics, dict) else None
        )
        if not isinstance(metric_semantics, dict):
            continue
        classification = str(
            metric_semantics.get("classification", FORCE_RATIO_MISSING)
        )
        semantic = str(
            metric_semantics.get("semantic_classification", FORCE_RATIO_MISSING)
        )
        classification_counts[classification] = (
            classification_counts.get(classification, 0) + 1
        )
        semantic_counts[semantic] = semantic_counts.get(semantic, 0) + 1
        row_summaries.append(
            {
                "motion": row.get("motion"),
                "seed": row.get("seed"),
                "classification": classification,
                "semantic_classification": semantic,
                "mjx_to_baseline": metric_semantics.get(
                    f"mjx_{metric}_force_ratio"
                ),
                "replay_to_baseline": metric_semantics.get(
                    f"replay_{metric}_force_ratio"
                ),
            }
        )
    return {
        "rows": row_summaries,
        "classification_counts": classification_counts,
        "semantic_classification_counts": semantic_counts,
        "force_ratio_high_count": classification_counts.get(FORCE_RATIO_HIGH, 0),
        "isolated_raw_mjx_force_high_count": semantic_counts.get(
            ISOLATED_RAW_MJX_FORCE_HIGH,
            0,
        ),
        "shared_force_high_count": semantic_counts.get(SHARED_FORCE_HIGH, 0),
        "replay_force_high_count": semantic_counts.get(REPLAY_FORCE_HIGH, 0),
    }


def _force_semantic_classification(
    *,
    mjx_ratio: float | None,
    replay_ratio: float | None,
) -> str:
    if mjx_ratio is None or replay_ratio is None:
        return FORCE_RATIO_MISSING
    if (
        mjx_ratio >= FORCE_MJX_HIGH_RATIO_MIN
        and replay_ratio <= FORCE_REPLAY_NEAR_BASELINE_RATIO_MAX
    ):
        return ISOLATED_RAW_MJX_FORCE_HIGH
    if (
        mjx_ratio >= FORCE_MJX_HIGH_RATIO_MIN
        and replay_ratio > FORCE_REPLAY_NEAR_BASELINE_RATIO_MAX
    ):
        return SHARED_FORCE_HIGH
    if replay_ratio > FORCE_REPLAY_NEAR_BASELINE_RATIO_MAX:
        return REPLAY_FORCE_HIGH
    return FORCE_RATIO_NOMINAL


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-root", type=Path, required=True)
    parser.add_argument("--min-speedup", type=float, default=11.0)
    args = parser.parse_args()
    payload = summarize_quality_runs(
        baseline_manifest=args.baseline_manifest,
        acceptance_root=args.acceptance_root,
        min_speedup=args.min_speedup,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
