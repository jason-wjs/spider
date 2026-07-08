#!/usr/bin/env python3
"""Compare MJX optimizer candidate rank diagnostics across repeated rows."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Sequence


_SCORE_COMPONENT_DIAGNOSTIC_FIELDS = (
    "root_pos_error_mean",
    "root_rot_error_mean",
    "joint_pos_error_mean",
    "body_global_pos_error_mean",
    "body_global_rot_error_mean",
    "body_local_pos_error_mean",
    "body_local_rot_error_mean",
    "ee_global_pos_error_mean",
    "ee_global_rot_error_mean",
    "ee_local_pos_error_mean",
    "ee_local_rot_error_mean",
    "hand_global_pos_error_mean",
    "hand_global_rot_error_mean",
    "hand_local_pos_error_mean",
    "hand_local_rot_error_mean",
    "contact_mismatch_rate",
    "contact_false_positive_rate",
    "contact_false_negative_rate",
    "contact_switch_rate",
    "bad_floor_contact_rate",
    "bad_floor_force_excess_mean",
    "contact_force_active",
    "contact_force_delta_mean",
    "contact_force_peak",
    "contact_force_peak_excess_mean",
    "control_delta_mean",
    "action_delta_mean",
    "joint_acc_mean",
    "joint_jerk_mean",
)


def main() -> None:
    args = _parse_args()
    report = analyze_candidate_rank_divergence(args.run, target=args.target)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)


def analyze_candidate_rank_divergence(
    runs: Sequence[str | Path],
    *,
    target: str = "diagnostic_candidate_rank_divergence",
) -> dict[str, Any]:
    """Return no-GPU diagnostics for per-iteration candidate top-k histories."""

    resolved = [_resolve_rank_run(path) for path in runs]
    pairwise = [
        _pairwise_rank_report(left, right, left_index=left_index, right_index=right_index)
        for (left_index, left), (right_index, right) in itertools.combinations(
            enumerate(resolved),
            2,
        )
    ]
    failures = _rank_failures(resolved, pairwise)
    return {
        "schema_version": 1,
        "gate": "offline_candidate_rank_divergence",
        "target": str(target),
        "passed": not failures,
        "classification": _classification(failures),
        "failures": failures,
        "run_count": len(resolved),
        "runs": resolved,
        "pairwise": pairwise,
    }


def _resolve_rank_run(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    metrics_path = _metrics_path_from_input(source)
    payload = _read_json(metrics_path) if metrics_path.is_file() else {}
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    mpc = payload.get("mpc", {}) if isinstance(payload, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(mpc, dict):
        mpc = {}
    history = mpc.get("history")
    if not isinstance(history, list):
        history = []
    windows = [_rank_window(info) for info in history if isinstance(info, dict)]
    top_k = _int_or_none(
        mpc.get(
            "candidate_rank_diagnostics_top_k",
            mpc.get("mjx_candidate_rank_diagnostics_top_k"),
        )
    )
    rank_window_count = sum(1 for window in windows if window["candidate_top_indices"])
    return {
        "input": str(source),
        "metrics_json": str(metrics_path),
        "metrics_json_exists": metrics_path.is_file(),
        "score": _finite_float_or_none(metrics.get("score")),
        "root_pos_error_mean": _finite_float_or_none(
            metrics.get("root_pos_error_mean")
        ),
        "ee_global_pos_error_mean": _finite_float_or_none(
            metrics.get("ee_global_pos_error_mean")
        ),
        "contact_mismatch_rate": _finite_float_or_none(
            metrics.get("contact_mismatch_rate")
        ),
        "candidate_rank_diagnostics_top_k": 0 if top_k is None else int(top_k),
        "candidate_rank_diagnostics_windows": _int_or_none(
            mpc.get("candidate_rank_diagnostics_windows")
        ),
        "rank_window_count": int(rank_window_count),
        "history_window_count": len(windows),
        "windows": windows,
    }


def _metrics_path_from_input(source: Path) -> Path:
    if source.is_file() and source.name == "metrics.json":
        return source
    if source.is_file() and source.name == "acceptance_row.json":
        row = _read_json(source)
        artifacts = row.get("artifacts", {}) if isinstance(row, dict) else {}
        return Path(str(artifacts.get("metrics_json", ""))).expanduser().resolve()
    if source.is_dir() and (source / "metrics.json").is_file():
        return source / "metrics.json"
    if source.is_dir():
        matches = sorted(source.glob("*/*/mjx/metrics.json"))
        if len(matches) == 1:
            return matches[0]
    return source / "metrics.json"


def _rank_window(info: dict[str, Any]) -> dict[str, Any]:
    return {
        "sim_step": _int_or_none(info.get("sim_step")),
        "best_index": _int_or_none(info.get("best_index")),
        "current_controls_selected": _bool_or_none(
            info.get("current_controls_selected")
        ),
        "top_score_gap": _finite_float_or_none(info.get("top_score_gap")),
        "score_improvement": _finite_float_or_none(info.get("score_improvement")),
        "iteration_best_indices": _int_sequence(
            info.get("iteration_best_indices")
        ),
        "iteration_current_controls_selected_flags": _bool_sequence(
            info.get("iteration_current_controls_selected_flags")
        ),
        "iteration_top_score_gaps": _float_sequence(
            info.get("iteration_top_score_gaps")
        ),
        "iteration_score_improvements": _float_sequence(
            info.get("iteration_score_improvements")
        ),
        "iteration_sample_sums": _float_sequence(info.get("iteration_sample_sums")),
        "iteration_sample_squared_sums": _float_sequence(
            info.get("iteration_sample_squared_sums")
        ),
        "iteration_sample_abs_maxes": _float_sequence(
            info.get("iteration_sample_abs_maxes")
        ),
        "iteration_sample_checksums": _nested_float_sequence(
            info.get("iteration_sample_checksums")
        ),
        "sample_checksum_slots": _int_sequence(info.get("sample_checksum_slots")),
        "iteration_sample_slot_checksums": _nested_nested_float_sequence(
            info.get("iteration_sample_slot_checksums")
        ),
        "sample_diagnostics_version": _int_or_none(
            info.get("sample_diagnostics_version")
        ),
        "sample_shape": _int_sequence(info.get("sample_shape")),
        "sample_probe_indices": _nested_int_sequence(
            info.get("sample_probe_indices")
        ),
        "iteration_sample_probe_values": _nested_float_sequence(
            info.get("iteration_sample_probe_values")
        ),
        "candidate_top_indices": _nested_int_sequence(
            info.get("iteration_candidate_top_indices")
        ),
        "candidate_top_scores": _nested_float_sequence(
            info.get("iteration_candidate_top_scores")
        ),
        "score_component_source": _score_component_source(info),
        "score_component_field_indices": _int_sequence(
            info.get("candidate_score_component_diagnostics_field_indices")
        ),
        "iteration_score_component_field_indices": _nested_int_sequence(
            info.get("iteration_score_component_field_indices")
        ),
        "iteration_score_component_top_indices": _nested_int_sequence(
            info.get("iteration_score_component_top_indices")
        ),
        "iteration_score_component_top_values": _nested_nested_float_sequence(
            info.get("iteration_score_component_top_values")
        ),
        "iteration_score_component_peak_source_top_values": (
            _nested_nested_float_sequence(
                info.get("iteration_score_component_peak_source_top_values")
            )
        ),
    }


def _pairwise_rank_report(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_index: int,
    right_index: int,
) -> dict[str, Any]:
    left_windows = left["windows"]
    right_windows = right["windows"]
    window_count = min(len(left_windows), len(right_windows))
    best_divergence = _first_sequence_divergence(
        left_windows,
        right_windows,
        "iteration_best_indices",
    )
    current_divergence = _first_sequence_divergence(
        left_windows,
        right_windows,
        "iteration_current_controls_selected_flags",
    )
    rank_stats = _candidate_rank_pair_stats(left_windows, right_windows)
    sample_stats = _candidate_sample_signature_pair_stats(
        left_windows,
        right_windows,
    )
    sample_probe_stats = _candidate_sample_probe_pair_stats(
        left_windows,
        right_windows,
    )
    sample_checksum_stats = _candidate_sample_checksum_pair_stats(
        left_windows,
        right_windows,
    )
    sample_slot_checksum_stats = _candidate_sample_slot_checksum_pair_stats(
        left_windows,
        right_windows,
    )
    score_component_stats = _candidate_score_component_pair_stats(
        left_windows,
        right_windows,
    )
    return {
        "left_index": int(left_index),
        "right_index": int(right_index),
        "compared_windows": int(window_count),
        "first_best_index_divergence": best_divergence,
        "first_current_selected_divergence": current_divergence,
        **rank_stats,
        **sample_stats,
        **sample_probe_stats,
        **sample_checksum_stats,
        **sample_slot_checksum_stats,
        **score_component_stats,
    }


def _candidate_rank_pair_stats(
    left_windows: list[dict[str, Any]],
    right_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    top1_count = 0
    top_order_count = 0
    top_set_count = 0
    compared_iterations = 0
    first_top1 = None
    first_top_order = None
    first_top_set = None
    min_overlap = None
    min_overlap_location = None
    first_same_index_score_delta = None
    same_index_score_delta_max = 0.0
    same_index_score_delta_max_location = None
    compared_same_index_score_values = 0
    for window_index, (left, right) in enumerate(zip(left_windows, right_windows)):
        left_tops = left["candidate_top_indices"]
        right_tops = right["candidate_top_indices"]
        left_scores = left["candidate_top_scores"]
        right_scores = right["candidate_top_scores"]
        for iteration, (left_top, right_top) in enumerate(zip(left_tops, right_tops)):
            compared_iterations += 1
            left_score_row = left_scores[iteration] if iteration < len(left_scores) else []
            right_score_row = (
                right_scores[iteration] if iteration < len(right_scores) else []
            )
            score_delta = _same_index_score_delta_payload(
                window_index,
                iteration,
                left_top,
                right_top,
                left_score_row,
                right_score_row,
            )
            compared_same_index_score_values += int(
                score_delta["compared_common_count"]
            )
            if score_delta["max_abs_delta"] > same_index_score_delta_max:
                same_index_score_delta_max = float(score_delta["max_abs_delta"])
                same_index_score_delta_max_location = {
                    "window": int(window_index),
                    "iteration": int(iteration),
                    "candidate_index": score_delta["candidate_index"],
                }
            if (
                score_delta["max_abs_delta"] > 1.0e-6
                and first_same_index_score_delta is None
            ):
                first_same_index_score_delta = score_delta
            left_set = set(left_top)
            right_set = set(right_top)
            overlap = len(left_set.intersection(right_set))
            denominator = max(len(left_set), len(right_set), 1)
            if min_overlap is None or overlap < min_overlap["count"]:
                min_overlap = {"count": overlap, "denominator": denominator}
                min_overlap_location = {
                    "window": int(window_index),
                    "iteration": int(iteration),
                }
            if left_top[:1] != right_top[:1]:
                top1_count += 1
                if first_top1 is None:
                    first_top1 = _rank_divergence_payload(
                        window_index,
                        iteration,
                        left_top,
                        right_top,
                        left_scores,
                        right_scores,
                    )
            if left_top != right_top:
                top_order_count += 1
                if first_top_order is None:
                    first_top_order = _rank_divergence_payload(
                        window_index,
                        iteration,
                        left_top,
                        right_top,
                        left_scores,
                        right_scores,
                    )
            if left_set != right_set:
                top_set_count += 1
                if first_top_set is None:
                    first_top_set = _rank_divergence_payload(
                        window_index,
                        iteration,
                        left_top,
                        right_top,
                        left_scores,
                        right_scores,
                    )
    return {
        "compared_rank_iterations": int(compared_iterations),
        "candidate_top1_divergence_count": int(top1_count),
        "candidate_top_order_divergence_count": int(top_order_count),
        "candidate_top_set_divergence_count": int(top_set_count),
        "first_candidate_top1_divergence": first_top1,
        "first_candidate_top_order_divergence": first_top_order,
        "first_candidate_top_set_divergence": first_top_set,
        "min_candidate_top_overlap": min_overlap,
        "min_candidate_top_overlap_location": min_overlap_location,
        "compared_same_index_score_values": int(compared_same_index_score_values),
        "same_index_score_delta_max": float(same_index_score_delta_max),
        "same_index_score_delta_max_location": same_index_score_delta_max_location,
        "first_same_index_score_delta": first_same_index_score_delta,
    }


def _same_index_score_delta_payload(
    window_index: int,
    iteration: int,
    left_top: list[int],
    right_top: list[int],
    left_scores: list[float],
    right_scores: list[float],
) -> dict[str, Any]:
    left_by_index = {
        int(candidate): float(left_scores[index])
        for index, candidate in enumerate(left_top)
        if index < len(left_scores)
    }
    right_by_index = {
        int(candidate): float(right_scores[index])
        for index, candidate in enumerate(right_top)
        if index < len(right_scores)
    }
    deltas = [
        {
            "candidate_index": int(candidate),
            "left_score": float(left_by_index[candidate]),
            "right_score": float(right_by_index[candidate]),
            "delta_right_minus_left": float(
                right_by_index[candidate] - left_by_index[candidate]
            ),
        }
        for candidate in sorted(set(left_by_index).intersection(right_by_index))
    ]
    deltas.sort(key=lambda item: abs(item["delta_right_minus_left"]), reverse=True)
    if not deltas:
        return {
            "window": int(window_index),
            "iteration": int(iteration),
            "candidate_index": None,
            "left_score": None,
            "right_score": None,
            "delta_right_minus_left": None,
            "max_abs_delta": 0.0,
            "compared_common_count": 0,
            "largest_deltas": [],
        }
    largest = deltas[0]
    nonzero_deltas = [
        item for item in deltas if abs(item["delta_right_minus_left"]) > 1.0e-12
    ]
    return {
        "window": int(window_index),
        "iteration": int(iteration),
        "candidate_index": int(largest["candidate_index"]),
        "left_score": float(largest["left_score"]),
        "right_score": float(largest["right_score"]),
        "delta_right_minus_left": float(largest["delta_right_minus_left"]),
        "max_abs_delta": float(abs(largest["delta_right_minus_left"])),
        "compared_common_count": int(len(deltas)),
        "largest_deltas": nonzero_deltas[:5],
    }


def _candidate_sample_signature_pair_stats(
    left_windows: list[dict[str, Any]],
    right_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    first_divergence = None
    max_abs_delta = 0.0
    compared_iterations = 0
    for window_index, (left, right) in enumerate(zip(left_windows, right_windows)):
        left_values = _sample_signature_values(left)
        right_values = _sample_signature_values(right)
        for iteration, (left_sig, right_sig) in enumerate(zip(left_values, right_values)):
            compared_iterations += 1
            deltas = [
                abs(float(left_sig[name]) - float(right_sig[name]))
                for name in ("sum", "squared_sum", "abs_max")
            ]
            iteration_delta = max(deltas)
            max_abs_delta = max(max_abs_delta, iteration_delta)
            if iteration_delta > 1.0e-6 and first_divergence is None:
                first_divergence = {
                    "window": int(window_index),
                    "iteration": int(iteration),
                    "left": dict(left_sig),
                    "right": dict(right_sig),
                    "max_abs_delta": float(iteration_delta),
                }
    return {
        "compared_sample_signature_iterations": int(compared_iterations),
        "first_sample_signature_divergence": first_divergence,
        "sample_signature_max_abs_delta": float(max_abs_delta),
    }


def _sample_signature_values(window: dict[str, Any]) -> list[dict[str, float]]:
    sums = window.get("iteration_sample_sums", [])
    squared_sums = window.get("iteration_sample_squared_sums", [])
    abs_maxes = window.get("iteration_sample_abs_maxes", [])
    values = []
    for sample_sum, squared_sum, abs_max in zip(sums, squared_sums, abs_maxes):
        values.append(
            {
                "sum": float(sample_sum),
                "squared_sum": float(squared_sum),
                "abs_max": float(abs_max),
            }
        )
    return values


def _candidate_sample_probe_pair_stats(
    left_windows: list[dict[str, Any]],
    right_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    first_divergence = None
    max_abs_delta = 0.0
    compared_iterations = 0
    for window_index, (left, right) in enumerate(zip(left_windows, right_windows)):
        left_values = left.get("iteration_sample_probe_values", [])
        right_values = right.get("iteration_sample_probe_values", [])
        if not left_values or not right_values:
            continue
        left_shape = left.get("sample_shape", [])
        right_shape = right.get("sample_shape", [])
        left_indices = left.get("sample_probe_indices", [])
        right_indices = right.get("sample_probe_indices", [])
        for iteration, (left_probe, right_probe) in enumerate(
            zip(left_values, right_values)
        ):
            compared_iterations += 1
            probe_count = min(len(left_probe), len(right_probe))
            deltas = [
                abs(float(left_probe[index]) - float(right_probe[index]))
                for index in range(probe_count)
            ]
            iteration_delta = max(deltas) if deltas else 0.0
            index_mismatch = left_indices != right_indices
            shape_mismatch = left_shape != right_shape
            value_length_mismatch = len(left_probe) != len(right_probe)
            diverged = (
                index_mismatch
                or shape_mismatch
                or value_length_mismatch
                or iteration_delta > 1.0e-6
            )
            max_abs_delta = max(max_abs_delta, iteration_delta)
            if diverged and first_divergence is None:
                first_divergence = {
                    "window": int(window_index),
                    "iteration": int(iteration),
                    "left_indices": list(left_indices),
                    "right_indices": list(right_indices),
                    "left_shape": list(left_shape),
                    "right_shape": list(right_shape),
                    "left_values": list(left_probe),
                    "right_values": list(right_probe),
                    "max_abs_delta": float(iteration_delta),
                    "index_mismatch": bool(index_mismatch),
                    "shape_mismatch": bool(shape_mismatch),
                    "value_length_mismatch": bool(value_length_mismatch),
                }
    return {
        "compared_sample_probe_iterations": int(compared_iterations),
        "first_sample_probe_divergence": first_divergence,
        "sample_probe_max_abs_delta": float(max_abs_delta),
    }


def _candidate_sample_checksum_pair_stats(
    left_windows: list[dict[str, Any]],
    right_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    first_divergence = None
    max_abs_delta = 0.0
    compared_iterations = 0
    for window_index, (left, right) in enumerate(zip(left_windows, right_windows)):
        left_values = left.get("iteration_sample_checksums", [])
        right_values = right.get("iteration_sample_checksums", [])
        if not left_values or not right_values:
            continue
        for iteration, (left_checksum, right_checksum) in enumerate(
            zip(left_values, right_values)
        ):
            compared_iterations += 1
            checksum_count = min(len(left_checksum), len(right_checksum))
            deltas = [
                abs(float(left_checksum[index]) - float(right_checksum[index]))
                for index in range(checksum_count)
            ]
            iteration_delta = max(deltas) if deltas else 0.0
            length_mismatch = len(left_checksum) != len(right_checksum)
            max_abs_delta = max(max_abs_delta, iteration_delta)
            if (
                length_mismatch or iteration_delta > 1.0e-6
            ) and first_divergence is None:
                first_divergence = {
                    "window": int(window_index),
                    "iteration": int(iteration),
                    "left": list(left_checksum),
                    "right": list(right_checksum),
                    "max_abs_delta": float(iteration_delta),
                    "length_mismatch": bool(length_mismatch),
                }
    return {
        "compared_sample_checksum_iterations": int(compared_iterations),
        "first_sample_checksum_divergence": first_divergence,
        "sample_checksum_max_abs_delta": float(max_abs_delta),
    }


def _candidate_sample_slot_checksum_pair_stats(
    left_windows: list[dict[str, Any]],
    right_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    first_divergence = None
    max_abs_delta = 0.0
    compared_slots = 0
    for window_index, (left, right) in enumerate(zip(left_windows, right_windows)):
        left_slots = left.get("sample_checksum_slots", [])
        right_slots = right.get("sample_checksum_slots", [])
        left_values = left.get("iteration_sample_slot_checksums", [])
        right_values = right.get("iteration_sample_slot_checksums", [])
        if not left_values or not right_values:
            continue
        for iteration, (left_iteration, right_iteration) in enumerate(
            zip(left_values, right_values)
        ):
            slot_count = min(
                len(left_iteration),
                len(right_iteration),
                len(left_slots),
                len(right_slots),
            )
            for slot_offset in range(slot_count):
                compared_slots += 1
                left_checksum = left_iteration[slot_offset]
                right_checksum = right_iteration[slot_offset]
                checksum_count = min(len(left_checksum), len(right_checksum))
                deltas = [
                    abs(float(left_checksum[index]) - float(right_checksum[index]))
                    for index in range(checksum_count)
                ]
                iteration_delta = max(deltas) if deltas else 0.0
                slot_mismatch = left_slots[slot_offset] != right_slots[slot_offset]
                length_mismatch = len(left_checksum) != len(right_checksum)
                max_abs_delta = max(max_abs_delta, iteration_delta)
                if (
                    slot_mismatch or length_mismatch or iteration_delta > 1.0e-6
                ) and first_divergence is None:
                    first_divergence = {
                        "window": int(window_index),
                        "iteration": int(iteration),
                        "slot": int(left_slots[slot_offset]),
                        "left": list(left_checksum),
                        "right": list(right_checksum),
                        "max_abs_delta": float(iteration_delta),
                        "slot_mismatch": bool(slot_mismatch),
                        "length_mismatch": bool(length_mismatch),
                    }
    return {
        "compared_sample_slot_checksum_values": int(compared_slots),
        "first_sample_slot_checksum_divergence": first_divergence,
        "sample_slot_checksum_max_abs_delta": float(max_abs_delta),
    }


def _candidate_score_component_pair_stats(
    left_windows: list[dict[str, Any]],
    right_windows: list[dict[str, Any]],
) -> dict[str, Any]:
    first_delta = None
    max_abs_delta = 0.0
    max_location = None
    max_detail = None
    compared_values = 0
    for window_index, (left, right) in enumerate(zip(left_windows, right_windows)):
        left_top_indices = left.get("iteration_score_component_top_indices", [])
        right_top_indices = right.get("iteration_score_component_top_indices", [])
        left_top_values = left.get("iteration_score_component_top_values", [])
        right_top_values = right.get("iteration_score_component_top_values", [])
        left_peak_sources = left.get(
            "iteration_score_component_peak_source_top_values",
            [],
        )
        right_peak_sources = right.get(
            "iteration_score_component_peak_source_top_values",
            [],
        )
        for iteration, (left_top, right_top) in enumerate(
            zip(left_top_indices, right_top_indices)
        ):
            left_fields = _score_component_iteration_fields(left, iteration)
            right_fields = _score_component_iteration_fields(right, iteration)
            left_values = (
                left_top_values[iteration]
                if iteration < len(left_top_values)
                else []
            )
            right_values = (
                right_top_values[iteration]
                if iteration < len(right_top_values)
                else []
            )
            left_source_values = (
                left_peak_sources[iteration]
                if iteration < len(left_peak_sources)
                else []
            )
            right_source_values = (
                right_peak_sources[iteration]
                if iteration < len(right_peak_sources)
                else []
            )
            delta = _score_component_delta_payload(
                window_index,
                iteration,
                left_fields=left_fields,
                right_fields=right_fields,
                left_top=left_top,
                right_top=right_top,
                left_values=left_values,
                right_values=right_values,
                left_peak_sources=left_source_values,
                right_peak_sources=right_source_values,
            )
            compared_values += int(delta["compared_common_count"])
            if delta["max_abs_delta"] > max_abs_delta:
                max_abs_delta = float(delta["max_abs_delta"])
                max_location = {
                    "window": int(window_index),
                    "iteration": int(iteration),
                    "candidate_index": delta["candidate_index"],
                    "field_index": delta["field_index"],
                    "field_name": delta["field_name"],
                }
                max_detail = {
                    key: value
                    for key, value in delta.items()
                    if key != "compared_common_count"
                }
            if delta["max_abs_delta"] > 1.0e-6 and first_delta is None:
                first_delta = {
                    key: value
                    for key, value in delta.items()
                    if key != "compared_common_count"
                }
    return {
        "score_component_sources": [
            _first_score_component_source(left_windows),
            _first_score_component_source(right_windows),
        ],
        "compared_score_component_values": int(compared_values),
        "first_score_component_delta": first_delta,
        "score_component_delta_max": float(max_abs_delta),
        "score_component_delta_max_location": max_location,
        "score_component_delta_max_detail": max_detail,
    }


def _score_component_delta_payload(
    window_index: int,
    iteration: int,
    *,
    left_fields: list[int],
    right_fields: list[int],
    left_top: list[int],
    right_top: list[int],
    left_values: list[list[float]],
    right_values: list[list[float]],
    left_peak_sources: list[list[float]],
    right_peak_sources: list[list[float]],
) -> dict[str, Any]:
    left_by_key = _score_component_values_by_key(left_fields, left_top, left_values)
    right_by_key = _score_component_values_by_key(right_fields, right_top, right_values)
    left_source_by_candidate = _score_component_peak_sources_by_candidate(
        left_top,
        left_peak_sources,
    )
    right_source_by_candidate = _score_component_peak_sources_by_candidate(
        right_top,
        right_peak_sources,
    )
    deltas = []
    for field_index, candidate_index in sorted(
        set(left_by_key).intersection(right_by_key)
    ):
        left_value = float(left_by_key[(field_index, candidate_index)])
        right_value = float(right_by_key[(field_index, candidate_index)])
        deltas.append(
            {
                "candidate_index": int(candidate_index),
                "field_index": int(field_index),
                "field_name": _score_component_field_name(field_index),
                "left_value": left_value,
                "right_value": right_value,
                "delta_right_minus_left": float(right_value - left_value),
            }
        )
    deltas.sort(key=lambda item: abs(item["delta_right_minus_left"]), reverse=True)
    if not deltas:
        return {
            "window": int(window_index),
            "iteration": int(iteration),
            "candidate_index": None,
            "field_index": None,
            "field_name": None,
            "left_value": None,
            "right_value": None,
            "delta_right_minus_left": None,
            "max_abs_delta": 0.0,
            "compared_common_count": 0,
        }
    largest = deltas[0]
    payload = {
        "window": int(window_index),
        "iteration": int(iteration),
        "candidate_index": int(largest["candidate_index"]),
        "field_index": int(largest["field_index"]),
        "field_name": largest["field_name"],
        "left_value": float(largest["left_value"]),
        "right_value": float(largest["right_value"]),
        "delta_right_minus_left": float(largest["delta_right_minus_left"]),
        "max_abs_delta": float(abs(largest["delta_right_minus_left"])),
        "compared_common_count": int(len(deltas)),
    }
    candidate_index = int(largest["candidate_index"])
    left_source = left_source_by_candidate.get(candidate_index)
    right_source = right_source_by_candidate.get(candidate_index)
    if left_source is not None:
        payload["left_peak_source"] = left_source
    if right_source is not None:
        payload["right_peak_source"] = right_source
    return payload


def _score_component_values_by_key(
    fields: list[int],
    top_indices: list[int],
    top_values: list[list[float]],
) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], float] = {}
    for field_offset, field_index in enumerate(fields):
        if field_offset >= len(top_values):
            continue
        field_values = top_values[field_offset]
        for top_offset, candidate_index in enumerate(top_indices):
            if top_offset >= len(field_values):
                continue
            values[(int(field_index), int(candidate_index))] = float(
                field_values[top_offset]
            )
    return values


def _score_component_peak_sources_by_candidate(
    top_indices: list[int],
    top_sources: list[list[float]],
) -> dict[int, list[float]]:
    sources: dict[int, list[float]] = {}
    for top_offset, candidate_index in enumerate(top_indices):
        if top_offset >= len(top_sources):
            continue
        source = top_sources[top_offset]
        if source:
            sources[int(candidate_index)] = [float(value) for value in source]
    return sources


def _score_component_iteration_fields(
    window: dict[str, Any],
    iteration: int,
) -> list[int]:
    iteration_fields = window.get("iteration_score_component_field_indices", [])
    if iteration < len(iteration_fields) and iteration_fields[iteration]:
        return list(iteration_fields[iteration])
    return list(window.get("score_component_field_indices", []))


def _first_score_component_source(windows: list[dict[str, Any]]) -> str | None:
    for window in windows:
        source = window.get("score_component_source")
        if source is not None:
            return str(source)
    return None


def _score_component_field_name(field_index: int | None) -> str | None:
    if field_index is None:
        return None
    if 0 <= int(field_index) < len(_SCORE_COMPONENT_DIAGNOSTIC_FIELDS):
        return _SCORE_COMPONENT_DIAGNOSTIC_FIELDS[int(field_index)]
    return f"field_{int(field_index)}"


def _score_component_source(info: dict[str, Any]) -> str | None:
    legacy = _string_or_none(
        info.get("candidate_score_component_diagnostics_source")
    )
    if legacy is not None:
        return legacy
    source_code = _int_or_none(
        info.get("candidate_score_component_diagnostics_source_code")
    )
    if source_code == 0:
        return "rollout"
    if source_code == 1:
        return "score_only_output_rescore"
    return None


def _rank_divergence_payload(
    window_index: int,
    iteration: int,
    left_top: list[int],
    right_top: list[int],
    left_scores: list[list[float]],
    right_scores: list[list[float]],
) -> dict[str, Any]:
    return {
        "window": int(window_index),
        "iteration": int(iteration),
        "left_top_indices": list(left_top),
        "right_top_indices": list(right_top),
        "left_top_scores": (
            list(left_scores[iteration]) if iteration < len(left_scores) else []
        ),
        "right_top_scores": (
            list(right_scores[iteration]) if iteration < len(right_scores) else []
        ),
        "overlap": len(set(left_top).intersection(right_top)),
        "denominator": max(len(left_top), len(right_top), 1),
    }


def _first_sequence_divergence(
    left_windows: list[dict[str, Any]],
    right_windows: list[dict[str, Any]],
    key: str,
) -> int | None:
    for index, (left, right) in enumerate(zip(left_windows, right_windows)):
        if left.get(key) != right.get(key):
            return int(index)
    return None


def _rank_failures(
    runs: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    for index, run in enumerate(runs):
        if not run["metrics_json_exists"]:
            failures.append(f"run_{index}:metrics_json")
        if int(run["candidate_rank_diagnostics_top_k"]) <= 0:
            failures.append(f"run_{index}:candidate_rank_diagnostics_top_k")
        if int(run["rank_window_count"]) <= 0:
            failures.append(f"run_{index}:candidate_rank_history")
    for item in pairwise:
        prefix = f"pair_{item['left_index']}_{item['right_index']}"
        if item["first_best_index_divergence"] is not None:
            failures.append(f"{prefix}:best_index_divergence")
        if item["first_current_selected_divergence"] is not None:
            failures.append(f"{prefix}:current_selected_divergence")
        if item["first_candidate_top1_divergence"] is not None:
            failures.append(f"{prefix}:candidate_top1_divergence")
        if item["first_candidate_top_order_divergence"] is not None:
            failures.append(f"{prefix}:candidate_top_order_divergence")
        if item["first_candidate_top_set_divergence"] is not None:
            failures.append(f"{prefix}:candidate_top_set_divergence")
        if item.get("first_same_index_score_delta") is not None:
            failures.append(f"{prefix}:candidate_same_index_score_delta")
        if item["first_sample_signature_divergence"] is not None:
            failures.append(f"{prefix}:candidate_sample_signature_divergence")
        if item["first_sample_probe_divergence"] is not None:
            failures.append(f"{prefix}:candidate_sample_probe_divergence")
        if item.get("first_sample_checksum_divergence") is not None:
            failures.append(f"{prefix}:candidate_sample_checksum_divergence")
        if item.get("first_sample_slot_checksum_divergence") is not None:
            failures.append(f"{prefix}:candidate_sample_slot_checksum_divergence")
        if item.get("first_score_component_delta") is not None:
            failures.append(f"{prefix}:candidate_score_component_delta")
    return failures


def _classification(failures: list[str]) -> str:
    if not failures:
        return "rank_consistent"
    if any("candidate_rank" in item or "metrics_json" in item for item in failures):
        return "missing_rank_diagnostics"
    return "rank_divergence"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _int_sequence(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        parsed = _int_or_none(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _float_sequence(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for item in value:
        parsed = _finite_float_or_none(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _bool_sequence(value: Any) -> list[bool]:
    if not isinstance(value, list):
        return []
    out: list[bool] = []
    for item in value:
        parsed = _bool_or_none(item)
        if parsed is not None:
            out.append(parsed)
    return out


def _nested_int_sequence(value: Any) -> list[list[int]]:
    if not isinstance(value, list):
        return []
    return [_int_sequence(item) for item in value if isinstance(item, list)]


def _nested_float_sequence(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    return [_float_sequence(item) for item in value if isinstance(item, list)]


def _nested_nested_float_sequence(value: Any) -> list[list[list[float]]]:
    if not isinstance(value, list):
        return []
    return [
        _nested_float_sequence(item)
        for item in value
        if isinstance(item, list)
    ]


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not numeric == float(parsed):
        return None
    return parsed


def _finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        return None
    return parsed


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    return None


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="MJX row dir, metrics.json, or acceptance_row.json. Repeat at least twice.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--target",
        default="diagnostic_candidate_rank_divergence",
        help="Reader-facing target label recorded in the JSON report.",
    )
    args = parser.parse_args()
    if len(args.run) < 2:
        parser.error("--run must be provided at least twice.")
    return args


if __name__ == "__main__":
    main()
