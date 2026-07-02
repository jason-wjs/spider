"""Pure acceptance gate logic for G1 WBC MJX rollout migration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Any, Mapping, Sequence


PRIMARY_ERROR_METRICS = (
    "root_pos_error_mean",
    "body_global_pos_error_mean",
    "ee_global_pos_error_mean",
    "ee_local_pos_error_mean",
    "contact_mismatch_rate",
    "contact_false_positive_rate",
    "contact_false_negative_rate",
    "bad_floor_contact_rate",
    "control_delta_mean",
    "joint_acc_mean",
    "joint_jerk_mean",
)

REQUIRED_ARTIFACT_FIELDS = ("metrics_json", "rollout_npz", "mpc_command_npz")


@dataclass(frozen=True)
class BaselineThreshold:
    success_count_min: int
    score_mean_min: float
    metric_mean_max: dict[str, float]


BASELINE_THRESHOLDS = {
    "jump": BaselineThreshold(
        success_count_min=2,
        score_mean_min=-2.12,
        metric_mean_max={
            "root_pos_error_mean": 0.065,
            "body_global_pos_error_mean": 0.075,
            "ee_global_pos_error_mean": 0.085,
            "contact_mismatch_rate": 0.36,
            "control_delta_mean": 0.46,
            "joint_acc_mean": 225.0,
        },
    ),
    "walk": BaselineThreshold(
        success_count_min=3,
        score_mean_min=-1.27,
        metric_mean_max={
            "root_pos_error_mean": 0.075,
            "body_global_pos_error_mean": 0.080,
            "ee_global_pos_error_mean": 0.082,
            "ee_local_pos_error_mean": 0.038,
            "contact_mismatch_rate": 0.155,
            "control_delta_mean": 0.25,
            "joint_acc_mean": 120.0,
        },
    ),
}


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: tuple[str, ...]


@dataclass(frozen=True)
class BaselineGroupResult(GateResult):
    envelope: dict[str, dict[str, float]]
    promoted_seed: int | None
    success_count: int


@dataclass(frozen=True)
class MjxQualityPolicy:
    score_mean_allowance: float
    score_worst_allowance: float
    error_ratio_max: float
    contact_abs_allowance: float
    bad_floor_contact_abs_allowance: float
    control_ratio_max: float
    jerk_ratio_max: float
    absolute_caps: dict[str, float]

    @staticmethod
    def for_motion(motion: str) -> "MjxQualityPolicy":
        if motion == "jump":
            return MjxQualityPolicy(
                score_mean_allowance=-0.08,
                score_worst_allowance=-0.08,
                error_ratio_max=1.10,
                contact_abs_allowance=0.02,
                bad_floor_contact_abs_allowance=0.01,
                control_ratio_max=1.15,
                jerk_ratio_max=1.20,
                absolute_caps={
                    "root_pos_error_mean": 0.075,
                    "body_global_pos_error_mean": 0.085,
                    "ee_global_pos_error_mean": 0.095,
                    "ee_local_pos_error_mean": 0.050,
                    "contact_mismatch_rate": 0.375,
                },
            )
        if motion == "walk":
            return MjxQualityPolicy(
                score_mean_allowance=-0.05,
                score_worst_allowance=-0.05,
                error_ratio_max=1.10,
                contact_abs_allowance=0.02,
                bad_floor_contact_abs_allowance=0.01,
                control_ratio_max=1.15,
                jerk_ratio_max=1.20,
                absolute_caps={
                    "root_pos_error_mean": 0.082,
                    "body_global_pos_error_mean": 0.088,
                    "ee_global_pos_error_mean": 0.090,
                    "ee_local_pos_error_mean": 0.042,
                    "contact_mismatch_rate": 0.175,
                },
            )
        raise ValueError(f"Unsupported motion for MJX quality policy: {motion}")


@dataclass(frozen=True)
class SpeedGateResult:
    passed: bool
    speedup: float
    failures: tuple[str, ...]


def evaluate_baseline_group(
    motion: str,
    repeats: Sequence[Mapping[str, Any]],
) -> BaselineGroupResult:
    threshold = _baseline_threshold(motion)
    failures = _global_repeat_failures(repeats)
    required_metrics = ("success", "score", *PRIMARY_ERROR_METRICS)

    if len(repeats) < 3:
        failures.append("repeat_count")
    failures.extend(_missing_metric_failures(repeats, required_metrics))

    success_count = _success_count(repeats)
    if success_count < threshold.success_count_min:
        failures.append("success_count")
    if motion == "walk" and success_count != len(repeats):
        failures.append("success_count")

    envelope = _build_envelope(repeats)
    score = envelope.get("score")
    if score is None or score["mean"] < threshold.score_mean_min:
        failures.append("score_mean")

    for metric, upper in threshold.metric_mean_max.items():
        stats = envelope.get(metric)
        if stats is None or stats["mean"] > upper:
            failures.append(f"{metric}_mean")

    failures.extend(_timing_outlier_failures(repeats))
    unique_failures = _unique(failures)
    return BaselineGroupResult(
        passed=not unique_failures,
        failures=unique_failures,
        envelope=envelope,
        promoted_seed=_promoted_seed(repeats) if not unique_failures else None,
        success_count=success_count,
    )


def evaluate_mjx_group(
    motion: str,
    repeats: Sequence[Mapping[str, Any]],
    baseline_envelope: Mapping[str, Mapping[str, float]],
    policy: MjxQualityPolicy,
) -> GateResult:
    _baseline_threshold(motion)
    failures = _global_repeat_failures(repeats)
    required_metrics = ("score", *baseline_envelope.keys())
    failures.extend(_missing_metric_failures(repeats, required_metrics))
    baseline_success_count = _baseline_success_count(baseline_envelope)
    if baseline_success_count is None:
        failures.append("baseline_success_count")
    elif _success_count(repeats) < baseline_success_count:
        failures.append("success_count")

    if any(bool(row.get("mpc_used_baseline_fallback")) for row in repeats):
        failures.append("fallback")
    if any(bool(row.get("contact_saturated")) for row in repeats):
        failures.append("contact_saturation")
    if any(bool(row.get("max_contact_points_saturated")) for row in repeats):
        failures.append("max_contact_points_saturation")
    if any(bool(row.get("max_geom_pairs_saturated")) for row in repeats):
        failures.append("max_geom_pairs_saturation")

    envelope = _build_envelope(repeats)
    if "score" in envelope and "score" in baseline_envelope:
        score_mean_floor = baseline_envelope["score"]["mean"] + policy.score_mean_allowance
        score_worst_floor = baseline_envelope["score"]["min"] + policy.score_worst_allowance
        if envelope["score"]["mean"] < score_mean_floor:
            failures.append("score_mean")
        if envelope["score"]["min"] < score_worst_floor:
            failures.append("score_worst")

    for metric in (
        "root_pos_error_mean",
        "body_global_pos_error_mean",
        "ee_global_pos_error_mean",
        "ee_local_pos_error_mean",
    ):
        if not _metric_present(metric, envelope, baseline_envelope):
            continue
        if envelope[metric]["mean"] > baseline_envelope[metric]["mean"] * policy.error_ratio_max:
            failures.append(metric)
        absolute_cap = policy.absolute_caps.get(metric)
        if absolute_cap is not None and envelope[metric]["mean"] > absolute_cap:
            failures.append(f"{metric}_absolute")

    for metric in (
        "contact_mismatch_rate",
        "contact_false_positive_rate",
        "contact_false_negative_rate",
    ):
        if not _metric_present(metric, envelope, baseline_envelope):
            continue
        if envelope[metric]["mean"] > baseline_envelope[metric]["mean"] + policy.contact_abs_allowance:
            failures.append(metric)

    if _metric_present("bad_floor_contact_rate", envelope, baseline_envelope):
        if (
            envelope["bad_floor_contact_rate"]["mean"]
            > baseline_envelope["bad_floor_contact_rate"]["mean"]
            + policy.bad_floor_contact_abs_allowance
        ):
            failures.append("bad_floor_contact_rate")

    if _metric_present("contact_mismatch_rate", envelope, baseline_envelope):
        contact_cap = policy.absolute_caps["contact_mismatch_rate"]
        if envelope["contact_mismatch_rate"]["mean"] > contact_cap:
            failures.append("contact_mismatch_rate_absolute")

    for metric in ("control_delta_mean", "joint_acc_mean"):
        if _metric_present(metric, envelope, baseline_envelope):
            if envelope[metric]["mean"] > baseline_envelope[metric]["mean"] * policy.control_ratio_max:
                failures.append(metric)

    if _metric_present("joint_jerk_mean", envelope, baseline_envelope):
        if envelope["joint_jerk_mean"]["mean"] > baseline_envelope["joint_jerk_mean"]["mean"] * policy.jerk_ratio_max:
            failures.append("joint_jerk_mean")

    return GateResult(passed=not failures, failures=_unique(failures))


def evaluate_speed_gate(
    *,
    baseline_wall_time_sec: float,
    mjx_steady_state_wall_time_sec: float,
    min_speedup: float,
) -> SpeedGateResult:
    failures: list[str] = []
    if not math.isfinite(baseline_wall_time_sec) or baseline_wall_time_sec <= 0.0:
        failures.append("baseline_wall_time")
    if not math.isfinite(mjx_steady_state_wall_time_sec) or mjx_steady_state_wall_time_sec <= 0.0:
        failures.append("mjx_steady_state_wall_time")
    if not math.isfinite(min_speedup) or min_speedup <= 0.0:
        failures.append("min_speedup")
    if failures:
        return SpeedGateResult(False, 0.0, tuple(failures))

    speedup = baseline_wall_time_sec / mjx_steady_state_wall_time_sec
    if speedup < min_speedup:
        failures.append("speedup")
    return SpeedGateResult(not failures, speedup, tuple(failures))


def _baseline_threshold(motion: str) -> BaselineThreshold:
    try:
        return BASELINE_THRESHOLDS[motion]
    except KeyError as exc:
        raise ValueError(f"Unsupported motion for baseline gate: {motion}") from exc


def _global_repeat_failures(repeats: Sequence[Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    for row in repeats:
        metrics = _metrics(row)
        if row.get("status") != "ok":
            failures.append("status")
        if int(row.get("num_steps", metrics.get("num_steps", -1))) != 800:
            failures.append("num_steps")
        if not bool(row.get("mpc_accepted")):
            failures.append("mpc_accepted")
        if int(row.get("accepted_windows", -1)) != 40:
            failures.append("accepted_windows")
        if bool(row.get("mpc_used_baseline_fallback")):
            failures.append("baseline_fallback")
        artifacts = row.get("artifacts", {})
        for key in REQUIRED_ARTIFACT_FIELDS:
            if not isinstance(artifacts, Mapping) or not artifacts.get(key):
                failures.append(key)
        for value in metrics.values():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                failures.append("finite_metrics")
    return failures


def _missing_metric_failures(
    repeats: Sequence[Mapping[str, Any]],
    required_metrics: Sequence[str],
) -> list[str]:
    failures: list[str] = []
    for metric in required_metrics:
        if any(metric not in _metrics(row) for row in repeats):
            failures.append(f"{metric}_missing")
    return failures


def _metrics(row: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = row.get("metrics", {})
    if not isinstance(metrics, Mapping):
        raise TypeError("repeat metrics must be a mapping")
    return metrics


def _build_envelope(repeats: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    envelope: dict[str, dict[str, float]] = {}
    success_values = [1.0 if bool(_metrics(row).get("success")) else 0.0 for row in repeats]
    if success_values:
        envelope["success"] = {
            "mean": mean(success_values),
            "std": pstdev(success_values) if len(success_values) > 1 else 0.0,
            "min": min(success_values),
            "max": max(success_values),
            "median": median(success_values),
            "count": float(sum(success_values)),
        }
    for metric in ("score", *PRIMARY_ERROR_METRICS):
        values = _metric_values(repeats, metric)
        if not values:
            continue
        envelope[metric] = {
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "median": median(values),
        }
    return envelope


def _metric_values(repeats: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for row in repeats:
        metrics = _metrics(row)
        if metric not in metrics:
            continue
        value = metrics[metric]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values.append(float(value))
    return values


def _metric_present(
    metric: str,
    envelope: Mapping[str, Mapping[str, float]],
    baseline_envelope: Mapping[str, Mapping[str, float]],
) -> bool:
    return metric in envelope and metric in baseline_envelope


def _success_count(repeats: Sequence[Mapping[str, Any]]) -> int:
    return sum(bool(_metrics(row).get("success")) for row in repeats)


def _baseline_success_count(
    baseline_envelope: Mapping[str, Mapping[str, float]],
) -> int | None:
    success = baseline_envelope.get("success")
    if success is None or "count" not in success:
        return None
    return int(round(float(success["count"])))


def _promoted_seed(repeats: Sequence[Mapping[str, Any]]) -> int | None:
    if not repeats:
        return None
    best = max(repeats, key=lambda row: float(_metrics(row).get("score", float("-inf"))))
    seed = best.get("seed")
    return int(seed) if seed is not None else None


def _timing_outlier_failures(repeats: Sequence[Mapping[str, Any]]) -> list[str]:
    wall_times = [
        float(row["wall_time_sec"])
        for row in repeats
        if isinstance(row.get("wall_time_sec"), (int, float)) and math.isfinite(float(row["wall_time_sec"]))
    ]
    if not wall_times:
        return ["wall_time_sec"]
    median_time = median(wall_times)
    if median_time <= 0.0:
        return ["wall_time_sec"]
    if any(wall_time > median_time * 1.25 for wall_time in wall_times):
        return ["timing_outlier"]
    return []


def _unique(failures: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(failures))
