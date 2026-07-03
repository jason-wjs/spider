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
PLAN_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "superpowers"
    / "plans"
    / "2026-07-02-g1-wbc-mjx-full-rollout-plan.md"
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
            "mpc_command_qvel_mismatch",
            "replay_saved_command_source",
            "replay_saved_command_hash",
            "replay_command_npz_frames",
            "replay_rollout_ref_indices",
            "replay_metrics_provenance",
            "pass_h100_milestone",
            "invalid_benchmark",
        )
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, body)

    def test_gpu_verification_plan_uses_wbc_mlp_stage0_checkpoint(self) -> None:
        body = PLAN_PATH.read_text()
        task_11 = body.split("## Task 11: GPU Verification Sequence", 1)[1]

        self.assertIn("model_8000.pt", task_11)
        self.assertNotIn("model_11800.pt", task_11)


if __name__ == "__main__":
    unittest.main()
