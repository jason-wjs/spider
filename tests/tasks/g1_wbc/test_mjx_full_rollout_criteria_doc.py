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
    def test_final_criteria_doc_defines_rtx4090_quality_first_contract(self) -> None:
        body = DOC_PATH.read_text()

        self.assertLess(len(body.splitlines()), 1000)
        required_terms = (
            "# G1 WBC MJX RTX 4090 Acceptance Criteria",
            "g1_wbc_stage0_mujoco_warp_sweetpoint",
            "Current No-MPC Reference",
            "capture(x)",
            "capture(score)",
            "80%",
            "100%",
            "12 MJX rows",
            "12 replay rows",
            "relative_pair_gap",
            "paired_gap",
            "realtime_factor",
            "min(realtime_factor)",
            ">= 0.8",
            "NVIDIA GeForce RTX 4090",
            "clean Git commit",
            "invalid_evidence",
            "functional_or_resource_failure",
            "no_mpc_dominance_failure",
            "sweetpoint_quality_failure",
            "guardrail_failure",
            "repeat_stability_failure",
            "replay_parity_failure",
            "speed_failure",
            "promotion_candidate",
            "n8192_parity_candidate",
        )
        for term in required_terms:
            with self.subTest(term=term):
                self.assertIn(term, body)

        forbidden_terms = (
            "## H100 Speed Gate",
            "pass_h100_milestone",
            "pass_4090_realtime",
            ">= 12x",
            "/data_team/",
            "/tmp/",
            "criteria_legacy",
        )
        for term in forbidden_terms:
            with self.subTest(term=term):
                self.assertNotIn(term, body)

        task_files = sorted(path.name for path in DOC_PATH.parent.iterdir())
        self.assertEqual(task_files, ["criteria.md"])


if __name__ == "__main__":
    unittest.main()
