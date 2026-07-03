from __future__ import annotations

import unittest
from pathlib import Path


DOC_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "tasks"
    / "g1_wbc_mjx_full_rollout"
    / "criteria.md"
)


class MjxFullRolloutCriteriaDocTest(unittest.TestCase):
    def test_final_criteria_doc_exists_with_required_acceptance_terms(self) -> None:
        body = DOC_PATH.read_text()

        required_terms = (
            "# G1 WBC MJX Full-Rollout Final Acceptance Criteria",
            "g1_wbc_stage0_mujoco_warp_sweetpoint",
            "baseline_envelopes",
            "promoted_seeds",
            "single GPU",
            "H100",
            ">= 12x",
            "RTX 4090",
            "MuJoCo-Warp replay",
            "worst_speedup",
            "rollout_npz_schema",
            "mpc_command_npz_schema",
            "mpc_rollout_qpos_mismatch",
            "mpc_command_qpos_mismatch",
            "replay_saved_command_source",
            "replay_command_npz_frames",
            "pass_h100_milestone",
            "invalid_benchmark",
        )
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, body)


if __name__ == "__main__":
    unittest.main()
