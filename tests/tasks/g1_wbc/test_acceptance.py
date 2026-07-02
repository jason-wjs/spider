import unittest

from spider.tasks.g1_wbc.acceptance import (
    BASELINE_THRESHOLDS,
    MjxQualityPolicy,
    evaluate_baseline_group,
    evaluate_mjx_group,
    evaluate_speed_gate,
)


def _baseline_repeat(motion, seed, score, root, body, ee, contact, control, acc):
    return {
        "motion": motion,
        "seed": seed,
        "status": "ok",
        "num_steps": 800,
        "mpc_accepted": True,
        "accepted_windows": 40,
        "mpc_used_baseline_fallback": False,
        "wall_time_sec": 100.0 + seed,
        "metrics": {
            "success": True,
            "score": score,
            "root_pos_error_mean": root,
            "body_global_pos_error_mean": body,
            "ee_global_pos_error_mean": ee,
            "ee_local_pos_error_mean": 0.035,
            "contact_mismatch_rate": contact,
            "contact_false_positive_rate": contact * 0.5,
            "contact_false_negative_rate": contact * 0.5,
            "bad_floor_contact_rate": 0.0,
            "control_delta_mean": control,
            "joint_acc_mean": acc,
            "joint_jerk_mean": acc * 2.0,
            "num_steps": 800,
        },
        "artifacts": {
            "metrics_json": "metrics.json",
            "rollout_npz": "rollout.npz",
            "mpc_command_npz": "mpc_command.npz",
        },
    }


class AcceptanceTest(unittest.TestCase):
    def test_jump_baseline_group_passes_thresholds_with_three_repeats(self):
        rows = [
            _baseline_repeat("jump", 0, -2.05, 0.060, 0.070, 0.080, 0.34, 0.42, 210.0),
            _baseline_repeat("jump", 1, -2.08, 0.061, 0.071, 0.081, 0.35, 0.43, 211.0),
            _baseline_repeat("jump", 2, -2.10, 0.062, 0.072, 0.082, 0.35, 0.44, 212.0),
        ]

        result = evaluate_baseline_group("jump", rows)

        self.assertTrue(result.passed)
        self.assertGreaterEqual(
            result.envelope["score"]["mean"],
            BASELINE_THRESHOLDS["jump"].score_mean_min,
        )
        self.assertEqual(result.envelope["root_pos_error_mean"]["max"], 0.062)
        self.assertEqual(result.promoted_seed, 0)

    def test_baseline_group_fails_when_primary_metric_is_missing(self):
        rows = [
            _baseline_repeat("jump", 0, -2.05, 0.060, 0.070, 0.080, 0.34, 0.42, 210.0),
            _baseline_repeat("jump", 1, -2.08, 0.061, 0.071, 0.081, 0.35, 0.43, 211.0),
            _baseline_repeat("jump", 2, -2.10, 0.062, 0.072, 0.082, 0.35, 0.44, 212.0),
        ]
        del rows[1]["metrics"]["joint_jerk_mean"]

        result = evaluate_baseline_group("jump", rows)

        self.assertFalse(result.passed)
        self.assertIn("joint_jerk_mean_missing", result.failures)

    def test_baseline_group_fails_when_primary_metric_is_non_numeric(self):
        rows = [
            _baseline_repeat("jump", 0, -2.05, 0.060, 0.070, 0.080, 0.34, 0.42, 210.0),
            _baseline_repeat("jump", 1, -2.08, 0.061, 0.071, 0.081, 0.35, 0.43, 211.0),
            _baseline_repeat("jump", 2, -2.10, 0.062, 0.072, 0.082, 0.35, 0.44, 212.0),
        ]
        rows[1]["metrics"]["joint_jerk_mean"] = "bad"

        result = evaluate_baseline_group("jump", rows)

        self.assertFalse(result.passed)
        self.assertIn("joint_jerk_mean_missing", result.failures)

    def test_baseline_group_fails_closed_when_metadata_is_malformed(self):
        rows = [
            _baseline_repeat("jump", 0, -2.05, 0.060, 0.070, 0.080, 0.34, 0.42, 210.0),
            _baseline_repeat("jump", 1, -2.08, 0.061, 0.071, 0.081, 0.35, 0.43, 211.0),
            _baseline_repeat("jump", 2, -2.10, 0.062, 0.072, 0.082, 0.35, 0.44, 212.0),
        ]
        rows[0]["num_steps"] = "bad"
        rows[1]["accepted_windows"] = None

        result = evaluate_baseline_group("jump", rows)

        self.assertFalse(result.passed)
        self.assertIn("num_steps", result.failures)
        self.assertIn("accepted_windows", result.failures)

    def test_baseline_group_does_not_hard_fail_on_wall_time_outlier(self):
        rows = [
            _baseline_repeat("jump", 0, -2.05, 0.060, 0.070, 0.080, 0.34, 0.42, 210.0),
            _baseline_repeat("jump", 1, -2.08, 0.061, 0.071, 0.081, 0.35, 0.43, 211.0),
            _baseline_repeat("jump", 2, -2.10, 0.062, 0.072, 0.082, 0.35, 0.44, 212.0),
        ]
        rows[2]["wall_time_sec"] = 9999.0

        result = evaluate_baseline_group("jump", rows)

        self.assertTrue(result.passed)
        self.assertNotIn("timing_outlier", result.failures)

    def test_walk_baseline_fails_when_contact_mismatch_rate_mean_exceeds_threshold(self):
        rows = [
            _baseline_repeat("walk", seed, -1.20, 0.070, 0.075, 0.078, 0.18, 0.23, 110.0)
            for seed in (0, 1, 2)
        ]

        result = evaluate_baseline_group("walk", rows)

        self.assertFalse(result.passed)
        self.assertIn("contact_mismatch_rate_mean", result.failures)

    def test_walk_baseline_requires_all_evaluated_repeats_to_succeed(self):
        rows = [
            _baseline_repeat("walk", seed, -1.20, 0.070, 0.075, 0.078, 0.14, 0.23, 110.0)
            for seed in (0, 1, 2, 3)
        ]
        rows[3]["metrics"]["success"] = False

        result = evaluate_baseline_group("walk", rows)

        self.assertFalse(result.passed)
        self.assertIn("success_count", result.failures)

    def test_mjx_group_fails_when_any_repeat_uses_baseline_fallback_even_with_good_metrics(self):
        baseline = evaluate_baseline_group(
            "walk",
            [
                _baseline_repeat("walk", seed, -1.20, 0.070, 0.075, 0.078, 0.14, 0.23, 110.0)
                for seed in (0, 1, 2)
            ],
        )
        mjx_rows = [
            {
                **_baseline_repeat("walk", seed, -1.19, 0.071, 0.076, 0.079, 0.14, 0.23, 111.0),
                "mpc_used_baseline_fallback": seed == 1,
                "contact_saturated": False,
            }
            for seed in (0, 1, 2)
        ]

        result = evaluate_mjx_group(
            "walk",
            mjx_rows,
            baseline.envelope,
            MjxQualityPolicy.for_motion("walk"),
        )

        self.assertFalse(result.passed)
        self.assertIn("fallback", result.failures)
        self.assertNotIn("baseline_fallback", result.failures)

    def test_mjx_group_requires_success_count_at_least_baseline_success_count(self):
        baseline = evaluate_baseline_group(
            "walk",
            [
                _baseline_repeat("walk", seed, -1.20, 0.070, 0.075, 0.078, 0.14, 0.23, 110.0)
                for seed in (0, 1, 2)
            ],
        )
        mjx_rows = [
            _baseline_repeat("walk", seed, -1.19, 0.071, 0.076, 0.079, 0.14, 0.23, 111.0)
            for seed in (0, 1, 2)
        ]
        mjx_rows[2]["metrics"]["success"] = False

        result = evaluate_mjx_group(
            "walk",
            mjx_rows,
            baseline.envelope,
            MjxQualityPolicy.for_motion("walk"),
        )

        self.assertFalse(result.passed)
        self.assertIn("success_count", result.failures)

    def test_speed_gate_computes_steady_state_wall_time_ratio_and_passes_at_twelve_x(self):
        result = evaluate_speed_gate(
            baseline_wall_time_sec=120.0,
            mjx_steady_state_wall_time_sec=10.0,
            min_speedup=12.0,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.speedup, 12.0)


if __name__ == "__main__":
    unittest.main()
