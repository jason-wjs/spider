from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.diagnose_g1_wbc_mjx_command_divergence import (
    analyze_mjx_command_divergence,
)


class MjxCommandDivergenceDiagnosticTest(unittest.TestCase):
    def test_reports_history_and_command_first_divergence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            _write_metrics(
                left / "metrics.json",
                score=-1.0,
                best_indices=((0, 1), (0, 0), (2, 3)),
                improvements=(0.1, 0.0, 0.2),
            )
            _write_metrics(
                right / "metrics.json",
                score=-1.5,
                best_indices=((0, 1), (0, 4), (2, 3)),
                improvements=(0.1, 0.3, -0.2),
            )
            left_qpos = np.zeros((4, 36), dtype=np.float32)
            right_qpos = left_qpos.copy()
            right_qpos[2, 0] = 0.02
            right_qpos[3, 0] = 0.2
            np.savez(
                left / "mpc_command.npz",
                command_qpos_trajectory=left_qpos,
                candidate_scores=np.zeros((3,), dtype=np.float32),
            )
            np.savez(
                right / "mpc_command.npz",
                command_qpos_trajectory=right_qpos,
                candidate_scores=np.ones((3,), dtype=np.float32),
            )

            report = analyze_mjx_command_divergence(
                left,
                right,
                thresholds=(1.0e-3, 1.0e-1),
            )

        self.assertEqual(
            report["history"]["first_best_index_divergence"],
            1,
        )
        self.assertEqual(
            report["history"]["first_current_selected_divergence"],
            1,
        )
        self.assertEqual(
            report["history"]["largest_score_improvement_delta"]["window"],
            2,
        )
        qpos = report["command"]["command_qpos_trajectory"]
        self.assertEqual(qpos["first_exceed"]["1e-03"], 2)
        self.assertEqual(qpos["first_exceed"]["1e-01"], 3)
        self.assertAlmostEqual(
            report["metrics"]["score"]["delta_right_minus_left"],
            -0.5,
        )


def _write_metrics(
    path: Path,
    *,
    score: float,
    best_indices: tuple[tuple[int, ...], ...],
    improvements: tuple[float, ...],
) -> None:
    path.write_text(
        json.dumps(
            {
                "metrics": {
                    "score": score,
                    "root_pos_error_mean": 0.1,
                    "body_global_pos_error_mean": 0.1,
                    "ee_global_pos_error_mean": 0.1,
                    "ee_local_pos_error_mean": 0.1,
                    "contact_mismatch_rate": 0.1,
                },
                "mpc": {
                    "history": [
                        {
                            "sim_step": index * 20,
                            "best_index": indices[-1],
                            "score_improvement": improvements[index],
                            "current_controls_selected": indices[-1] == 0,
                            "iteration_best_indices": list(indices),
                            "iteration_score_improvements": [
                                0.0,
                                improvements[index],
                            ],
                        }
                        for index, indices in enumerate(best_indices)
                    ]
                },
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
