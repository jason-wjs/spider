from __future__ import annotations

import json
from pathlib import Path

from scripts.diagnose_g1_wbc_mjx_contact_force_semantics import (
    classify_force_semantics,
    main,
)


def _write_metrics(path: Path, *, active: float, peak: float) -> None:
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "contact_force_active_mean": active,
                    "contact_force_peak": peak,
                }
            }
        ),
        encoding="utf-8",
    )


def test_classify_force_semantics_identifies_isolated_high_mjx_force() -> None:
    result = classify_force_semantics(baseline=50.0, mjx=180.0, replay=70.0)

    assert result == {
        "mjx_active_force_ratio": 3.6,
        "replay_active_force_ratio": 1.4,
        "classification": "force_ratio_high",
        "semantic_classification": "isolated_raw_mjx_force_high",
    }


def test_classify_force_semantics_requires_replay_near_baseline() -> None:
    result = classify_force_semantics(baseline=50.0, mjx=180.0, replay=90.0)

    assert result["mjx_active_force_ratio"] == 3.6
    assert result["replay_active_force_ratio"] == 1.8
    assert result["classification"] == "force_ratio_not_isolated"
    assert result["semantic_classification"] == "shared_force_high"


def test_cli_reports_active_and_peak_contact_force_classifications(
    tmp_path: Path,
    capsys,
) -> None:
    baseline = tmp_path / "baseline" / "metrics.json"
    mjx = tmp_path / "mjx" / "metrics.json"
    replay = tmp_path / "replay" / "metrics.json"
    baseline.parent.mkdir()
    mjx.parent.mkdir()
    replay.parent.mkdir()
    _write_metrics(baseline, active=50.0, peak=400.0)
    _write_metrics(mjx, active=180.0, peak=1300.0)
    _write_metrics(replay, active=70.0, peak=720.0)

    main(
        [
            "--baseline-metrics",
            str(baseline),
            "--mjx-metrics",
            str(mjx),
            "--replay-metrics",
            str(replay),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["active"] == {
        "mjx_active_force_ratio": 3.6,
        "replay_active_force_ratio": 1.4,
        "classification": "force_ratio_high",
        "semantic_classification": "isolated_raw_mjx_force_high",
    }
    assert payload["peak"] == {
        "mjx_peak_force_ratio": 3.25,
        "replay_peak_force_ratio": 1.8,
        "classification": "force_ratio_not_isolated",
        "semantic_classification": "shared_force_high",
    }
