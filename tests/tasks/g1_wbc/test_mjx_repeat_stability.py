from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.check_g1_wbc_mjx_repeat_stability import (
    analyze_mjx_repeat_stability,
)


def test_repeat_stability_passes_identical_repeats(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left)
    _write_run(right)

    report = analyze_mjx_repeat_stability([left, right])

    assert report["passed"] is True
    assert report["classification"] == "pass"
    assert report["failures"] == []
    assert report["all_command_content_hashes_equal"] is True
    assert report["pairwise"][0]["first_best_index_divergence"] is None
    assert report["pairwise"][0]["command_qpos_trajectory"]["max"] == 0.0


def test_repeat_stability_flags_history_command_and_metric_drift(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_run(left, score=-1.0)
    _write_run(
        right,
        score=-1.25,
        best_indices=((0, 1), (1, 2), (2, 3)),
        command_delta=0.02,
    )

    report = analyze_mjx_repeat_stability(
        [left, right],
        max_command_qpos_delta=1.0e-3,
    )

    assert report["passed"] is False
    assert report["classification"] == "repeat_instability"
    assert "metric_range:score" in report["failures"]
    assert "pair_0_1:best_index_divergence" in report["failures"]
    assert "pair_0_1:command_qpos_delta" in report["failures"]
    assert report["pairwise"][0]["first_best_index_divergence"] == 1
    assert report["metric_ranges"]["score"]["range"] == 0.25


def test_repeat_stability_fails_missing_artifact_and_basic_gate(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    _write_run(good)
    _write_run(bad, accepted=False)
    (bad / "mpc_command.npz").unlink()

    report = analyze_mjx_repeat_stability([good, bad])

    assert report["passed"] is False
    assert report["classification"] == "invalid_artifacts"
    assert "run_1:mpc_command_npz" in report["failures"]
    assert "run_1:mpc_accepted" in report["failures"]
    assert "pair_0_1:pairwise_error" in report["failures"]


def test_repeat_stability_resolves_acceptance_row(tmp_path: Path) -> None:
    run = tmp_path / "run"
    repeat = tmp_path / "repeat"
    _write_run(run)
    _write_run(repeat)
    row = tmp_path / "acceptance_row.json"
    row.write_text(
        json.dumps(
            {
                "status": "ok",
                "returncode": 0,
                "artifacts": {
                    "metrics_json": str(run / "metrics.json"),
                    "mpc_command_npz": str(run / "mpc_command.npz"),
                },
            }
        ),
        encoding="utf-8",
    )

    report = analyze_mjx_repeat_stability([row, repeat])

    assert report["passed"] is True
    assert report["runs"][0]["status"] == "ok"
    assert report["runs"][0]["returncode"] == 0


def _write_run(
    root: Path,
    *,
    score: float = -1.0,
    accepted: bool = True,
    best_indices: tuple[tuple[int, ...], ...] = ((0, 1), (0, 1), (2, 3)),
    command_delta: float = 0.0,
) -> None:
    root.mkdir(parents=True)
    metrics = {
        "score": score,
        "root_pos_error_mean": 0.05,
        "body_global_pos_error_mean": 0.06,
        "ee_global_pos_error_mean": 0.07,
        "ee_local_pos_error_mean": 0.04,
        "contact_mismatch_rate": 0.1,
        "control_delta_mean": 0.2,
        "joint_acc_mean": 120.0,
        "contact_force_active_mean": 250.0,
        "contact_force_peak": 1000.0,
        "num_steps": 800,
    }
    payload = {
        "metrics": metrics,
        "mpc": {
            "accepted": accepted,
            "accepted_windows": 40,
            "used_baseline_fallback": False,
            "contact_saturated": False,
            "max_contact_points_saturated": False,
            "max_geom_pairs_saturated": False,
            "steady_state_wall_time_sec": 18.0,
            "history": [
                {
                    "sim_step": index * 20,
                    "best_index": indices[-1],
                    "score_improvement": 0.1,
                    "current_controls_selected": indices[-1] == 0,
                    "iteration_best_indices": list(indices),
                    "iteration_score_improvements": [0.0, 0.1],
                }
                for index, indices in enumerate(best_indices)
            ],
        },
    }
    (root / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    qpos = np.zeros((4, 36), dtype=np.float32)
    qpos[2, 0] = command_delta
    np.savez(
        root / "mpc_command.npz",
        command_qpos_trajectory=qpos,
        candidate_scores=np.zeros((3,), dtype=np.float32),
    )
