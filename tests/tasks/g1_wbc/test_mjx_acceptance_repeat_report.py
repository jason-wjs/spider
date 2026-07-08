from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.compare_g1_wbc_mjx_acceptance_repeats import (
    analyze_acceptance_repeat_stability,
    normalize_mjx_argv_for_repeat,
)


def test_acceptance_repeat_report_compares_same_configuration_rows(
    tmp_path: Path,
) -> None:
    left_run = _write_run(tmp_path / "left_run", score=-1.0)
    right_run = _write_run(
        tmp_path / "right_run",
        score=-1.05,
        command_delta=0.02,
        best_indices=((0, 1), (2, 3)),
    )
    left_report = _write_report(
        tmp_path / "left_report.json",
        _row(left_run, output_dir="/tmp/left", naconmax="20000"),
    )
    right_report = _write_report(
        tmp_path / "right_report.json",
        _row(right_run, output_dir="/tmp/right", naconmax="20000"),
    )

    report = analyze_acceptance_repeat_stability([left_report, right_report])

    assert report["passed"] is False
    assert report["classification"] == "repeat_instability"
    assert report["pair_count"] == 1
    pair = report["pairs"][0]
    assert pair["motion"] == "jump"
    assert pair["seed"] == 0
    assert pair["status"] == "repeat_instability"
    assert pair["argv_equivalent"] is True
    assert pair["repeat_report"]["classification"] == "repeat_instability"
    assert "pair_0_1:command_qpos_delta" in pair["repeat_report"]["failures"]


def test_acceptance_repeat_report_rejects_configuration_mismatch(
    tmp_path: Path,
) -> None:
    left_run = _write_run(tmp_path / "left_run")
    right_run = _write_run(tmp_path / "right_run")
    left_report = _write_report(
        tmp_path / "left_report.json",
        _row(left_run, output_dir="/tmp/left", naconmax="20000"),
    )
    right_report = _write_report(
        tmp_path / "right_report.json",
        _row(right_run, output_dir="/tmp/right", naconmax="30000"),
    )

    report = analyze_acceptance_repeat_stability([left_report, right_report])

    assert report["passed"] is False
    assert report["classification"] == "configuration_mismatch"
    assert report["pairs"][0]["status"] == "configuration_mismatch"
    assert report["pairs"][0]["argv_equivalent"] is False
    assert "repeat_report" not in report["pairs"][0]


def test_normalized_argv_ignores_output_paths_but_keeps_contact_capacity() -> None:
    left = [
        "python",
        "-m",
        "spider.tasks.g1_wbc.evaluate",
        "--output-dir",
        "/tmp/left",
        "--baseline-manifest",
        "/tmp/left_manifest.json",
        "--mjx-warp-naconmax",
        "20000",
    ]
    right = [
        "python",
        "-m",
        "spider.tasks.g1_wbc.evaluate",
        "--output-dir",
        "/tmp/right",
        "--baseline-manifest",
        "/tmp/right_manifest.json",
        "--mjx-warp-naconmax",
        "30000",
    ]

    assert normalize_mjx_argv_for_repeat(left) != normalize_mjx_argv_for_repeat(right)


def _write_report(path: Path, row: dict[str, object]) -> Path:
    path.write_text(json.dumps({"mjx_rows": [row]}), encoding="utf-8")
    return path


def _row(root: Path, *, output_dir: str, naconmax: str) -> dict[str, object]:
    return {
        "motion": "jump",
        "seed": 0,
        "output_dir": output_dir,
        "mjx_argv": [
            "python",
            "-m",
            "spider.tasks.g1_wbc.evaluate",
            "--motion",
            "jump.npz",
            "--output-dir",
            output_dir,
            "--seed",
            "0",
            "--mjx-warp-naconmax",
            naconmax,
        ],
        "artifacts": {
            "metrics_json": str(root / "metrics.json"),
            "mpc_command_npz": str(root / "mpc_command.npz"),
        },
    }


def _write_run(
    root: Path,
    *,
    score: float = -1.0,
    command_delta: float = 0.0,
    best_indices: tuple[tuple[int, ...], ...] = ((0, 1), (0, 1)),
) -> Path:
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
            "accepted": True,
            "accepted_windows": 40,
            "used_baseline_fallback": False,
            "contact_saturated": False,
            "max_contact_points_saturated": False,
            "max_geom_pairs_saturated": False,
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
    return root
