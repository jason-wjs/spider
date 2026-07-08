from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_g1_wbc_mjx_candidate_rank_divergence import (
    analyze_candidate_rank_divergence,
)


def test_candidate_rank_divergence_passes_matching_histories(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left)
    _write_run(right)

    report = analyze_candidate_rank_divergence([left, right])

    assert report["passed"] is True
    assert report["classification"] == "rank_consistent"
    assert report["failures"] == []
    assert report["pairwise"][0]["first_best_index_divergence"] is None
    assert report["pairwise"][0]["first_candidate_top1_divergence"] is None
    assert report["pairwise"][0]["first_sample_signature_divergence"] is None
    assert report["pairwise"][0]["min_candidate_top_overlap"] == {
        "count": 3,
        "denominator": 3,
    }


def test_candidate_rank_divergence_flags_topk_and_selection_drift(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left)
    _write_run(
        right,
        best_indices=([0, 2], [0, 0]),
        current_flags=([True, True], [True, True]),
        top_indices=(
            ([1, 0, 2], [2, 1, 0]),
            ([0, 1, 2], [0, 2, 1]),
        ),
        sample_sums=([9.0, 11.0], [20.0, 21.0]),
    )

    report = analyze_candidate_rank_divergence([left, right])

    assert report["passed"] is False
    assert report["classification"] == "rank_divergence"
    assert "pair_0_1:best_index_divergence" in report["failures"]
    assert "pair_0_1:current_selected_divergence" in report["failures"]
    assert "pair_0_1:candidate_top1_divergence" in report["failures"]
    assert "pair_0_1:candidate_top_order_divergence" in report["failures"]
    pair = report["pairwise"][0]
    assert pair["first_best_index_divergence"] == 0
    assert pair["first_current_selected_divergence"] == 0
    assert pair["first_candidate_top1_divergence"]["window"] == 0
    assert pair["first_candidate_top1_divergence"]["iteration"] == 1
    assert pair["first_candidate_top1_divergence"]["left_top_indices"] == [1, 0, 2]
    assert pair["first_candidate_top1_divergence"]["right_top_indices"] == [2, 1, 0]
    assert pair["candidate_top_order_divergence_count"] == 2
    assert pair["candidate_top_set_divergence_count"] == 0
    assert pair["first_sample_signature_divergence"]["window"] == 0
    assert pair["first_sample_signature_divergence"]["iteration"] == 0
    assert "pair_0_1:candidate_sample_signature_divergence" in report["failures"]


def test_candidate_rank_divergence_flags_sample_probe_drift(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left)
    _write_run(
        right,
        sample_probe_values=([2.0, 3.0, 4.0], [4.0, 5.0, 6.0]),
    )

    report = analyze_candidate_rank_divergence([left, right])

    pair = report["pairwise"][0]
    assert report["passed"] is False
    assert pair["first_sample_signature_divergence"] is None
    assert pair["first_sample_probe_divergence"]["window"] == 0
    assert pair["first_sample_probe_divergence"]["iteration"] == 0
    assert pair["first_sample_probe_divergence"]["max_abs_delta"] == 1.0
    assert "pair_0_1:candidate_sample_probe_divergence" in report["failures"]


def test_candidate_rank_divergence_flags_sample_checksum_drift(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(
        left,
        sample_checksums=(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ),
    )
    _write_run(
        right,
        sample_checksums=(
            [[1.0, 2.0, 3.5], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ),
    )

    report = analyze_candidate_rank_divergence([left, right])

    pair = report["pairwise"][0]
    assert report["passed"] is False
    assert pair["first_sample_signature_divergence"] is None
    assert pair["first_sample_probe_divergence"] is None
    assert pair["first_sample_checksum_divergence"] == {
        "window": 0,
        "iteration": 0,
        "left": [1.0, 2.0, 3.0],
        "right": [1.0, 2.0, 3.5],
        "max_abs_delta": 0.5,
        "length_mismatch": False,
    }
    assert pair["sample_checksum_max_abs_delta"] == 0.5
    assert "pair_0_1:candidate_sample_checksum_divergence" in report["failures"]


def test_candidate_rank_divergence_flags_sample_slot_checksum_drift(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(
        left,
        sample_slot_checksums=(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            [
                [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]],
            ],
        ),
    )
    _write_run(
        right,
        sample_slot_checksums=(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.5]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            [
                [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]],
                [[19.0, 20.0, 21.0], [22.0, 23.0, 24.0]],
            ],
        ),
    )

    report = analyze_candidate_rank_divergence([left, right])

    pair = report["pairwise"][0]
    assert report["passed"] is False
    assert pair["first_sample_slot_checksum_divergence"] == {
        "window": 0,
        "iteration": 0,
        "slot": 1,
        "left": [4.0, 5.0, 6.0],
        "right": [4.0, 5.0, 6.5],
        "max_abs_delta": 0.5,
        "slot_mismatch": False,
        "length_mismatch": False,
    }
    assert pair["sample_slot_checksum_max_abs_delta"] == 0.5
    assert "pair_0_1:candidate_sample_slot_checksum_divergence" in report["failures"]


def test_candidate_rank_divergence_reports_same_index_score_drift(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(
        left,
        top_indices=([1, 0, 2], [3, 4, 5]),
        top_scores=([3.0, 2.0, 1.0], [6.0, 5.0, 4.0]),
    )
    _write_run(
        right,
        top_indices=([1, 0, 2], [3, 4, 5]),
        top_scores=([3.5, 2.0, 0.75], [6.0, 5.0, 4.0]),
    )

    report = analyze_candidate_rank_divergence([left, right])

    pair = report["pairwise"][0]
    assert report["passed"] is False
    assert pair["same_index_score_delta_max"] == 0.5
    assert pair["same_index_score_delta_max_location"] == {
        "window": 0,
        "iteration": 0,
        "candidate_index": 1,
    }
    assert pair["first_same_index_score_delta"] == {
        "window": 0,
        "iteration": 0,
        "candidate_index": 1,
        "left_score": 3.0,
        "right_score": 3.5,
        "delta_right_minus_left": 0.5,
        "max_abs_delta": 0.5,
        "compared_common_count": 3,
        "largest_deltas": [
            {
                "candidate_index": 1,
                "left_score": 3.0,
                "right_score": 3.5,
                "delta_right_minus_left": 0.5,
            },
            {
                "candidate_index": 2,
                "left_score": 1.0,
                "right_score": 0.75,
                "delta_right_minus_left": -0.25,
            },
        ],
    }
    assert "pair_0_1:candidate_same_index_score_delta" in report["failures"]


def test_candidate_rank_divergence_reports_score_component_drift(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left)
    _write_run(right)
    _inject_score_components(
        left,
        field_indices=[0, 22],
        top_indices=[1, 2],
        top_values=[
            [0.10, 0.20],
            [10.0, 20.0],
        ],
        source="score_only_output_rescore",
    )
    _inject_score_components(
        right,
        field_indices=[0, 22],
        top_indices=[1, 2],
        top_values=[
            [0.10, 0.25],
            [10.0, 30.0],
        ],
        source="score_only_output_rescore",
    )

    report = analyze_candidate_rank_divergence([left, right])

    pair = report["pairwise"][0]
    assert report["passed"] is False
    assert pair["score_component_sources"] == [
        "score_only_output_rescore",
        "score_only_output_rescore",
    ]
    assert pair["score_component_delta_max"] == 10.0
    assert pair["score_component_delta_max_location"] == {
        "window": 0,
        "iteration": 0,
        "candidate_index": 2,
        "field_index": 22,
        "field_name": "contact_force_delta_mean",
    }
    expected_delta = {
        "window": 0,
        "iteration": 0,
        "candidate_index": 2,
        "field_index": 22,
        "field_name": "contact_force_delta_mean",
        "left_value": 20.0,
        "right_value": 30.0,
        "delta_right_minus_left": 10.0,
        "max_abs_delta": 10.0,
    }
    assert pair["first_score_component_delta"] == expected_delta
    assert pair["score_component_delta_max_detail"] == expected_delta
    assert "pair_0_1:candidate_score_component_delta" in report["failures"]


def test_candidate_rank_divergence_reports_score_component_peak_sources(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left)
    _write_run(right)
    _inject_score_components(
        left,
        field_indices=[23],
        top_indices=[1, 2],
        top_values=[[100.0, 200.0]],
        source="score_only_output_rescore",
        peak_sources=[
            [10.0, 0.0, 17.0, 100.0, 10.0, 100.0, 3.0, 0.0],
            [20.0, 0.0, 34.0, 200.0, 20.0, 200.0, 3.0, 4.0],
        ],
    )
    _inject_score_components(
        right,
        field_indices=[23],
        top_indices=[1, 2],
        top_values=[[100.0, 350.0]],
        source="score_only_output_rescore",
        peak_sources=[
            [10.0, 0.0, 17.0, 100.0, 10.0, 100.0, 3.0, 0.0],
            [30.0, 0.0, 37.0, 350.0, 35.0, 350.0, 3.0, 8.0],
        ],
    )

    report = analyze_candidate_rank_divergence([left, right])

    delta = report["pairwise"][0]["first_score_component_delta"]
    max_delta = report["pairwise"][0]["score_component_delta_max_detail"]
    assert delta["candidate_index"] == 2
    assert delta["field_name"] == "contact_force_peak"
    assert max_delta["candidate_index"] == 2
    assert max_delta["field_name"] == "contact_force_peak"
    assert delta["left_peak_source"] == [
        20.0,
        0.0,
        34.0,
        200.0,
        20.0,
        200.0,
        3.0,
        4.0,
    ]
    assert max_delta["left_peak_source"] == delta["left_peak_source"]
    assert delta["right_peak_source"] == [
        30.0,
        0.0,
        37.0,
        350.0,
        35.0,
        350.0,
        3.0,
        8.0,
    ]
    assert max_delta["right_peak_source"] == delta["right_peak_source"]


def test_candidate_rank_divergence_allows_legacy_histories_without_sample_probes(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left, include_sample_probe=False, include_sample_checksums=False)
    _write_run(right, include_sample_probe=False, include_sample_checksums=False)

    report = analyze_candidate_rank_divergence([left, right])

    pair = report["pairwise"][0]
    assert report["passed"] is True
    assert pair["compared_sample_probe_iterations"] == 0
    assert pair["first_sample_probe_divergence"] is None
    assert pair["compared_sample_checksum_iterations"] == 0
    assert pair["first_sample_checksum_divergence"] is None


def test_candidate_rank_divergence_flags_missing_rank_history(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left)
    _write_run(right, top_k=0, top_indices=None)

    report = analyze_candidate_rank_divergence([left, right])

    assert report["passed"] is False
    assert report["classification"] == "missing_rank_diagnostics"
    assert "run_1:candidate_rank_diagnostics_top_k" in report["failures"]
    assert "run_1:candidate_rank_history" in report["failures"]


def _write_run(
    root: Path,
    *,
    top_k: int = 3,
    best_indices: tuple[list[int], ...] = ([0, 1], [0, 0]),
    current_flags: tuple[list[bool], ...] = ([True, False], [True, True]),
    top_indices: tuple[object, ...] | None = (
        [1, 0, 2],
        [0, 1, 2],
    ),
    top_scores: tuple[list[float], ...] = (
        [3.0, 2.0, 1.0],
        [3.0, 2.0, 1.0],
    ),
    sample_sums: tuple[list[float], ...] = ([10.0, 11.0], [12.0, 13.0]),
    sample_checksums: tuple[list[list[float]], ...] = (
        [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
        [[12.0, 22.0, 32.0], [13.0, 23.0, 33.0]],
    ),
    sample_checksum_slots: list[int] = [0, 1],
    sample_slot_checksums: tuple[list[list[list[float]]], ...] = (
        [
            [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]],
            [[110.0, 120.0, 130.0], [111.0, 121.0, 131.0]],
        ],
        [
            [[12.0, 22.0, 32.0], [13.0, 23.0, 33.0]],
            [[112.0, 122.0, 132.0], [113.0, 123.0, 133.0]],
        ],
    ),
    sample_probe_values: tuple[list[float], ...] = (
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ),
    include_sample_probe: bool = True,
    include_sample_checksums: bool = True,
) -> None:
    root.mkdir(parents=True)
    history = []
    for index, indices in enumerate(best_indices):
        info = {
            "sim_step": index * 20,
            "best_index": indices[-1],
            "current_controls_selected": current_flags[index][-1],
            "top_score_gap": 0.01,
            "score_improvement": 0.02,
            "iteration_best_indices": indices,
            "iteration_current_controls_selected_flags": current_flags[index],
            "iteration_top_score_gaps": [0.1, 0.01],
            "iteration_score_improvements": [0.0, 0.02],
            "iteration_sample_sums": sample_sums[index],
            "iteration_sample_squared_sums": [
                value * value for value in sample_sums[index]
            ],
            "iteration_sample_abs_maxes": [1.0, 1.5],
        }
        if include_sample_checksums:
            info["iteration_sample_checksums"] = [
                list(values) for values in sample_checksums[index]
            ]
            info["sample_checksum_slots"] = list(sample_checksum_slots)
            info["iteration_sample_slot_checksums"] = [
                [list(checksum) for checksum in iteration_slot_checksums]
                for iteration_slot_checksums in sample_slot_checksums[index]
            ]
        if include_sample_probe:
            info.update(
                {
                    "sample_diagnostics_version": 1,
                    "sample_shape": [3, 5, 8],
                    "sample_probe_indices": [[0, 0, 0], [1, 2, 3], [2, 4, 5]],
                    "iteration_sample_probe_values": [
                        list(sample_probe_values[index]),
                        list(sample_probe_values[index]),
                    ],
                }
            )
        if top_indices is not None:
            window_top = top_indices[index]
            if window_top and isinstance(window_top[0], list):
                iteration_top_indices = [list(values) for values in window_top]
            else:
                iteration_top_indices = [list(window_top), list(window_top)]
            info["iteration_candidate_top_indices"] = iteration_top_indices
            info["iteration_candidate_top_scores"] = [
                list(top_scores[index]),
                list(top_scores[index]),
            ]
        history.append(info)
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "score": -1.0,
                    "root_pos_error_mean": 0.05,
                    "ee_global_pos_error_mean": 0.07,
                    "contact_mismatch_rate": 0.1,
                },
                "mpc": {
                    "candidate_rank_diagnostics_top_k": top_k,
                    "candidate_rank_diagnostics_windows": (
                        len(history) if top_indices is not None else 0
                    ),
                    "history": history,
                },
            }
        ),
        encoding="utf-8",
    )


def _inject_score_components(
    root: Path,
    *,
    field_indices: list[int],
    top_indices: list[int],
    top_values: list[list[float]],
    source: str,
    peak_sources: list[list[float]] | None = None,
) -> None:
    metrics_path = root / "metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    first_window = payload["mpc"]["history"][0]
    if source == "rollout":
        first_window["candidate_score_component_diagnostics_source_code"] = 0
    elif source == "score_only_output_rescore":
        first_window["candidate_score_component_diagnostics_source_code"] = 1
    else:
        first_window["candidate_score_component_diagnostics_source"] = source
    first_window["candidate_score_component_diagnostics_field_indices"] = list(
        field_indices
    )
    first_window["iteration_score_component_field_indices"] = [list(field_indices)]
    first_window["iteration_score_component_top_indices"] = [list(top_indices)]
    first_window["iteration_score_component_top_values"] = [
        [list(values) for values in top_values]
    ]
    if peak_sources is not None:
        first_window["candidate_score_component_peak_source_available"] = True
        first_window["candidate_score_component_peak_source_width"] = 8
        first_window["iteration_score_component_peak_source_top_values"] = [
            [list(values) for values in peak_sources]
        ]
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
