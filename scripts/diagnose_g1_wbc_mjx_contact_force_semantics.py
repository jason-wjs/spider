"""Classify MJX contact-force semantics against baseline and replay metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


FORCE_RATIO_HIGH = "force_ratio_high"
FORCE_RATIO_NOT_ISOLATED = "force_ratio_not_isolated"
ISOLATED_RAW_MJX_FORCE_HIGH = "isolated_raw_mjx_force_high"
SHARED_FORCE_HIGH = "shared_force_high"
REPLAY_FORCE_HIGH = "replay_force_high"
FORCE_RATIO_NOMINAL = "force_ratio_nominal"
MJX_HIGH_RATIO_MIN = 3.0
REPLAY_NEAR_BASELINE_RATIO_MAX = 1.5


def classify_force_semantics(
    *,
    baseline: float,
    mjx: float,
    replay: float,
) -> dict[str, float | str]:
    """Return whether MJX force is high while replay stays near baseline."""

    mjx_ratio = _ratio(mjx, baseline)
    replay_ratio = _ratio(replay, baseline)
    return {
        "mjx_active_force_ratio": mjx_ratio,
        "replay_active_force_ratio": replay_ratio,
        "classification": _classify_ratios(mjx_ratio, replay_ratio),
        "semantic_classification": _semantic_classification(
            mjx_ratio,
            replay_ratio,
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    baseline_metrics = _read_metric_values(args.baseline_metrics)
    mjx_metrics = _read_metric_values(args.mjx_metrics)
    replay_metrics = _read_metric_values(args.replay_metrics)

    active = classify_force_semantics(
        baseline=baseline_metrics["contact_force_active_mean"],
        mjx=mjx_metrics["contact_force_active_mean"],
        replay=replay_metrics["contact_force_active_mean"],
    )
    peak = _classify_named_force(
        baseline=baseline_metrics["contact_force_peak"],
        mjx=mjx_metrics["contact_force_peak"],
        replay=replay_metrics["contact_force_peak"],
        name="peak",
    )
    print(json.dumps({"active": active, "peak": peak}, indent=2, sort_keys=True))


def _classify_named_force(
    *,
    baseline: float,
    mjx: float,
    replay: float,
    name: str,
) -> dict[str, float | str]:
    mjx_ratio = _ratio(mjx, baseline)
    replay_ratio = _ratio(replay, baseline)
    return {
        f"mjx_{name}_force_ratio": mjx_ratio,
        f"replay_{name}_force_ratio": replay_ratio,
        "classification": _classify_ratios(mjx_ratio, replay_ratio),
        "semantic_classification": _semantic_classification(
            mjx_ratio,
            replay_ratio,
        ),
    }


def _classify_ratios(mjx_ratio: float, replay_ratio: float) -> str:
    if (
        mjx_ratio >= MJX_HIGH_RATIO_MIN
        and replay_ratio <= REPLAY_NEAR_BASELINE_RATIO_MAX
    ):
        return FORCE_RATIO_HIGH
    return FORCE_RATIO_NOT_ISOLATED


def _semantic_classification(mjx_ratio: float, replay_ratio: float) -> str:
    if (
        mjx_ratio >= MJX_HIGH_RATIO_MIN
        and replay_ratio <= REPLAY_NEAR_BASELINE_RATIO_MAX
    ):
        return ISOLATED_RAW_MJX_FORCE_HIGH
    if (
        mjx_ratio >= MJX_HIGH_RATIO_MIN
        and replay_ratio > REPLAY_NEAR_BASELINE_RATIO_MAX
    ):
        return SHARED_FORCE_HIGH
    if replay_ratio > REPLAY_NEAR_BASELINE_RATIO_MAX:
        return REPLAY_FORCE_HIGH
    return FORCE_RATIO_NOMINAL


def _ratio(value: float, baseline: float) -> float:
    if baseline == 0.0:
        if value == 0.0:
            return 1.0
        return math.inf
    return float(value) / float(baseline)


def _read_metric_values(path: Path) -> dict[str, float]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    return {
        "contact_force_active_mean": _required_float(
            metrics,
            "contact_force_active_mean",
            path,
        ),
        "contact_force_peak": _required_float(metrics, "contact_force_peak", path),
    }


def _required_float(metrics: Any, key: str, path: Path) -> float:
    if not isinstance(metrics, dict) or key not in metrics:
        raise KeyError(f"{path} is missing metrics.{key}")
    return float(metrics[key])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--mjx-metrics", type=Path, required=True)
    parser.add_argument("--replay-metrics", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
