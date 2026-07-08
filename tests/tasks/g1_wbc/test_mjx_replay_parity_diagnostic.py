from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.diagnose_g1_wbc_mjx_replay_parity import analyze_replay_parity


class MjxReplayParityDiagnosticTest(unittest.TestCase):
    def test_analyze_replay_parity_reports_first_divergence_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mjx_rollout = root / "mjx_rollout.npz"
            replay_rollout = root / "replay_rollout.npz"
            command_npz = root / "mpc_command.npz"

            mjx_qpos = np.zeros((5, 1, 36), dtype=np.float32)
            replay_qpos = mjx_qpos.copy()
            replay_qpos[2:, 0, 0] = np.array([0.0002, 0.02, 0.2], dtype=np.float32)
            command_qpos = mjx_qpos.copy()
            command_qpos[1:, 0, 0] = 0.01

            mjx_actions = np.zeros((4, 1, 29), dtype=np.float32)
            replay_actions = mjx_actions.copy()
            replay_actions[1, 0, 0] = 0.5

            np.savez(
                mjx_rollout,
                qpos=mjx_qpos,
                qvel=np.zeros((5, 1, 35), dtype=np.float32),
                actions=mjx_actions,
            )
            np.savez(
                replay_rollout,
                qpos=replay_qpos,
                qvel=np.zeros((5, 1, 35), dtype=np.float32),
                actions=replay_actions,
            )
            np.savez(
                command_npz,
                command_qpos_trajectory=command_qpos,
                command_qvel_trajectory=np.zeros((5, 1, 35), dtype=np.float32),
            )

            report = analyze_replay_parity(
                mjx_rollout,
                replay_rollout,
                command_npz=command_npz,
                frames=(0, 1, 2, 3, 4),
                thresholds=(1.0e-5, 1.0e-3, 1.0e-2, 1.0e-1),
            )

        root_pos = report["comparisons"]["replay_vs_mjx"]["root_pos"]
        self.assertEqual(root_pos["first_exceed"]["1e-05"], 2)
        self.assertEqual(root_pos["first_exceed"]["1e-03"], 3)
        self.assertEqual(root_pos["first_exceed"]["1e-02"], 3)
        self.assertEqual(root_pos["first_exceed"]["1e-01"], 4)
        self.assertAlmostEqual(root_pos["frames"]["4"], 0.2)

        actions = report["comparisons"]["replay_vs_mjx"]["actions"]
        self.assertEqual(actions["first_exceed"]["1e-01"], 1)

        command_root = report["comparisons"]["mjx_vs_command"]["root_pos"]
        self.assertEqual(command_root["first_exceed"]["1e-03"], 1)

    def test_acceptance_report_selects_matching_motion_and_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mjx_rollout = root / "mjx_rollout.npz"
            replay_rollout = root / "replay_rollout.npz"
            command_npz = root / "mpc_command.npz"
            qpos = np.zeros((2, 1, 36), dtype=np.float32)
            qvel = np.zeros((2, 1, 35), dtype=np.float32)
            np.savez(mjx_rollout, qpos=qpos, qvel=qvel)
            np.savez(replay_rollout, qpos=qpos, qvel=qvel)
            np.savez(command_npz, command_qpos_trajectory=qpos)
            report_path = root / "acceptance_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "mjx_rows": [
                            {
                                "motion": "jump",
                                "seed": 0,
                                "artifacts": {
                                    "rollout_npz": str(mjx_rollout),
                                    "mpc_command_npz": str(command_npz),
                                },
                            }
                        ],
                        "replay_rows": [
                            {
                                "motion": "jump",
                                "seed": 0,
                                "artifacts": {"rollout_npz": str(replay_rollout)},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            from scripts.diagnose_g1_wbc_mjx_replay_parity import (
                _paths_from_acceptance_report,
            )

            selected = _paths_from_acceptance_report(
                report_path,
                motion="jump",
                seed=0,
            )

        self.assertEqual(selected, (mjx_rollout, replay_rollout, command_npz))


if __name__ == "__main__":
    unittest.main()
