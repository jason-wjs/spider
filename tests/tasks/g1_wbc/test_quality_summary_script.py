from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_g1_wbc_mjx_quality import summarize_quality_runs


def _metrics_payload(
    *,
    score: float = -1.0,
    root: float = 0.05,
    body: float = 0.06,
    ee: float = 0.07,
    ee_local: float = 0.04,
    mismatch: float = 0.1,
    force_peak: float = 400.0,
    force_active: float = 60.0,
) -> dict[str, object]:
    return {
        "metrics": {
            "num_steps": 800,
            "success": True,
            "score": score,
            "root_pos_error_mean": root,
            "body_global_pos_error_mean": body,
            "ee_global_pos_error_mean": ee,
            "ee_local_pos_error_mean": ee_local,
            "contact_mismatch_rate": mismatch,
            "contact_force_peak": force_peak,
            "contact_force_active_mean": force_active,
        },
        "mpc": {
            "steady_state_wall_time_sec": 20.0,
            "accepted": True,
            "accepted_windows": 40,
            "used_baseline_fallback": False,
            "contact_saturated": False,
            "max_contact_points_saturated": False,
            "max_geom_pairs_saturated": False,
            "contact_pair_count": 65,
            "active_contact_count": 28,
            "runtime_gpu_name": "NVIDIA GeForce RTX 4090",
        },
    }


def test_summarize_quality_runs_reports_speed_force_and_failures(
    tmp_path: Path,
) -> None:
    stage0 = tmp_path / "stage0"
    mjx = tmp_path / "mjx"
    stage0_row = stage0 / "jump" / "seed_0"
    mjx_row = mjx / "jump" / "seed_0" / "mjx"
    replay_row = mjx / "jump" / "seed_0" / "replay"
    stage0_row.mkdir(parents=True)
    mjx_row.mkdir(parents=True)
    replay_row.mkdir(parents=True)
    (stage0_row / "metrics.json").write_text(
        json.dumps(_metrics_payload(force_peak=400.0, force_active=60.0))
    )
    (mjx_row / "metrics.json").write_text(
        json.dumps(
            _metrics_payload(
                score=-2.1,
                ee_local=0.055,
                force_peak=2400.0,
                force_active=250.0,
            )
        )
    )
    (replay_row / "metrics.json").write_text(
        json.dumps(
            _metrics_payload(
                score=-0.95,
                ee_local=0.039,
                force_peak=450.0,
                force_active=65.0,
            )
        )
    )
    manifest = {
        "contact_force_semantics": "pyramidal_contact_normal_v1",
        "baseline_envelopes": {
            "jump": {
                "ee_local_pos_error_mean": {
                    "mean": 0.04,
                    "std": 0.0,
                    "min": 0.04,
                    "max": 0.04,
                    "median": 0.04,
                },
                "score": {
                    "mean": -1.0,
                    "std": 0.0,
                    "min": -1.0,
                    "max": -1.0,
                    "median": -1.0,
                },
            }
        },
        "rows": [
            {
                "motion_name": "jump",
                "seed": 0,
                "steady_state_wall_time_sec": 220.0,
                "artifacts": {"metrics_json": str(stage0_row / "metrics.json")},
            }
        ],
    }
    manifest_path = stage0 / "baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    summary = summarize_quality_runs(
        baseline_manifest=manifest_path,
        acceptance_root=mjx,
        min_speedup=11.0,
    )

    row = summary["rows"][0]
    assert row["motion"] == "jump"
    assert row["seed"] == 0
    assert summary["contact_force_semantics"] == "pyramidal_contact_normal_v1"
    assert row["contact_force_semantics"] == "pyramidal_contact_normal_v1"
    assert row["same_seed_speedup"] == 11.0
    assert row["force_active_ratio"] > 4.0
    assert row["force_peak_ratio"] == 6.0
    assert row["replay_force_active_ratio"] == 65.0 / 60.0
    assert row["replay_quality_warnings"] == []
    assert row["force_semantics"]["active"]["classification"] == "force_ratio_high"
    assert (
        row["force_semantics"]["active"]["semantic_classification"]
        == "isolated_raw_mjx_force_high"
    )
    assert row["force_semantics"]["peak"]["classification"] == "force_ratio_high"
    assert (
        row["force_semantics"]["peak"]["semantic_classification"]
        == "isolated_raw_mjx_force_high"
    )
    assert row["quality_warnings"] == [
        "score_mean",
        "score_worst",
        "ee_local_pos_error_mean",
        "ee_local_pos_error_mean_absolute",
    ]
    assert summary["passed_speed_floor"] is True
    assert summary["quality_warning_count"] == 4
    assert summary["replay_quality_warning_count"] == 0
    assert summary["force_ratio_high_count"] == 2
    assert summary["force_semantics_summary"]["active"][
        "semantic_classification_counts"
    ] == {"isolated_raw_mjx_force_high": 1}
    assert summary["force_semantics_summary"]["active"][
        "isolated_raw_mjx_force_high_count"
    ] == 1
    assert summary["force_semantics_summary"]["active"]["replay_force_high_count"] == 0
    assert summary["force_semantics_summary"]["peak"][
        "semantic_classification_counts"
    ] == {"isolated_raw_mjx_force_high": 1}


@pytest.mark.parametrize("value", [None, "first_solver_row_v0"])
def test_summarize_quality_runs_rejects_noncanonical_force_semantics(
    tmp_path: Path,
    value: str | None,
) -> None:
    stage0 = tmp_path / "stage0"
    mjx = tmp_path / "mjx"
    stage0_row = stage0 / "jump" / "seed_0"
    mjx_row = mjx / "jump" / "seed_0" / "mjx"
    stage0_row.mkdir(parents=True)
    mjx_row.mkdir(parents=True)
    (stage0_row / "metrics.json").write_text(json.dumps(_metrics_payload()))
    (mjx_row / "metrics.json").write_text(json.dumps(_metrics_payload()))
    manifest = {
        "baseline_envelopes": {"jump": {}},
        "rows": [
            {
                "motion_name": "jump",
                "seed": 0,
                "steady_state_wall_time_sec": 220.0,
                "artifacts": {"metrics_json": str(stage0_row / "metrics.json")},
            }
        ],
    }
    if value is not None:
        manifest["contact_force_semantics"] = value
    manifest_path = stage0 / "baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="contact_force_semantics"):
        summarize_quality_runs(
            baseline_manifest=manifest_path,
            acceptance_root=mjx,
            min_speedup=11.0,
        )
