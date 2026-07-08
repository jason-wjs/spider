import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from spider.tasks.g1_wbc import evaluate


class MjxBackendCliTest(unittest.TestCase):
    def test_parse_args_defaults_to_mujoco_warp_backend(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-samples",
            "512",
            "--mpc-iterations",
            "2",
            "--mpc-planning-horizon-steps",
            "40",
            "--mpc-control-steps",
            "20",
            "--mpc-knot-count",
            "8",
            "--mpc-temperature",
            "0.7",
            "--mpc-root-pos-sigma",
            "0.04",
            "--mpc-root-rot-sigma",
            "0.10",
            "--mpc-joint-sigma",
            "0.18",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        self.assertEqual(args.mpc_backend, "mujoco_warp")
        self.assertFalse(args.mjx_enable_scan)
        self.assertEqual(args.mjx_impl, "jax")
        self.assertEqual(args.mjx_warp_naconmax, 30000)
        self.assertEqual(args.mjx_warp_njmax, 256)

    def test_parse_args_accepts_explicit_mjx_scan_enable(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mjx-enable-scan",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        self.assertTrue(args.mjx_enable_scan)

    def test_parse_args_accepts_explicit_mjx_warp_impl(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mjx-impl",
            "warp",
            "--mjx-warp-naconmax",
            "4096",
            "--mjx-warp-njmax",
            "512",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        self.assertEqual(args.mjx_impl, "warp")
        self.assertEqual(args.mjx_warp_naconmax, 4096)
        self.assertEqual(args.mjx_warp_njmax, 512)

    def test_parse_args_accepts_mjx_model_solver_overrides(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mjx-model-iterations",
            "6",
            "--mjx-model-ls-iterations",
            "10",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        self.assertEqual(args.mjx_model_iterations, 6)
        self.assertEqual(args.mjx_model_ls_iterations, 10)

    def test_parse_args_accepts_collision_profile(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--collision-profile",
            "wxy_explicit_pairs_7caps",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        self.assertEqual(args.collision_profile, "wxy_explicit_pairs_7caps")

    def test_parse_args_accepts_mjx_trace_prefix_diagnostic(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mjx-trace-prefix-steps",
            "20",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        self.assertEqual(args.mjx_trace_prefix_steps, 20)

    def test_parse_args_accepts_mjx_strip_live_data_diagnostic(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mjx-strip-live-mjx-data",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        self.assertTrue(args.mjx_strip_live_mjx_data)

    def test_parse_args_accepts_mjx_score_only_optimizer_diagnostic(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mjx-score-only-optimizer",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        self.assertTrue(args.mjx_score_only_optimizer)

    def test_parse_args_accepts_mjx_score_only_rescore_diagnostic(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mjx-score-only-rescore-diagnostics",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        self.assertTrue(args.mjx_score_only_rescore_diagnostics)

    def test_parse_args_accepts_mjx_score_only_output_rescore_diagnostic(
        self,
    ) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mjx-score-only-output-rescore-diagnostics",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        self.assertTrue(args.mjx_score_only_output_rescore_diagnostics)

    def test_parse_args_accepts_mjx_first_row_force_diagnostic(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mjx-contact-force-first-row-diagnostics",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        self.assertTrue(args.mjx_contact_force_first_row_diagnostics)

    def test_parse_args_accepts_mjx_first_row_contact_force_mode(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mjx-contact-force-mode",
            "first_row",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        self.assertEqual(args.mjx_contact_force_mode, "first_row")

    def test_mjx_first_row_force_diagnostic_requires_mjx_backend(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mjx-contact-force-first-row-diagnostics",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(
            ValueError,
            "--mjx-contact-force-first-row-diagnostics",
        ):
            evaluate._validate_backend_args(args)

    def test_mjx_contact_force_mode_requires_mjx_backend(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mjx-contact-force-mode",
            "first_row",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "--mjx-contact-force-mode"):
            evaluate._validate_backend_args(args)

    def test_mjx_trace_prefix_must_be_positive(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mjx-trace-prefix-steps",
            "0",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "--mjx-trace-prefix-steps"):
            evaluate._validate_backend_args(args)

    def test_mjx_warp_impl_requires_mjx_backend(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mjx-impl",
            "warp",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "--mjx-impl warp"):
            evaluate._validate_backend_args(args)

    def test_mjx_model_solver_overrides_must_be_positive(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mjx-model-iterations",
            "0",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "--mjx-model-iterations"):
            evaluate._validate_backend_args(args)

    def test_mjx_backend_enables_physics_scan_without_extra_flag(self) -> None:
        self.assertTrue(
            evaluate._mjx_physics_scan_enabled(
                SimpleNamespace(mpc_backend="mjx", mjx_enable_scan=False)
            )
        )
        self.assertFalse(
            evaluate._mjx_physics_scan_enabled(
                SimpleNamespace(mpc_backend="mujoco_warp", mjx_enable_scan=False)
            )
        )

    def test_mpc_payload_requires_explicit_safety_metadata(self) -> None:
        metadata = {
            "accepted": True,
            "accepted_windows": 40,
            "used_baseline_fallback": False,
        }

        self.assertIs(evaluate._required_mpc_metadata(metadata, "accepted"), True)
        with self.assertRaisesRegex(ValueError, "accepted"):
            evaluate._required_mpc_metadata({}, "accepted")

    def test_mjx_backend_rejects_non_mpc_method(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "no_mpc",
            "--mpc-backend",
            "mjx",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "requires an MPC method"):
            evaluate._validate_backend_args(args)

    def test_generic_optimizer_rejects_legacy_only_mpc_knobs(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-optimizer",
            "generic",
            "--mpc-guided-candidate",
            "--mpc-acceptance-gate",
            "--mpc-guided-joint-gain",
            "0.50",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "legacy optimizer"):
            evaluate._validate_backend_args(args)

    def test_generic_optimizer_accepts_elite_fraction(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mpc-elite-frac",
            "0.125",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)

    def test_generic_optimizer_accepts_sigma_decay(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mpc-sigma-decay",
            "0.75",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)

    def test_generic_optimizer_accepts_explicit_disabled_warm_start(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--device",
            "cpu",
            "--mpc-samples",
            "512",
            "--mpc-iterations",
            "2",
            "--mpc-planning-horizon-steps",
            "40",
            "--mpc-control-steps",
            "20",
            "--mpc-knot-count",
            "8",
            "--mpc-temperature",
            "0.7",
            "--mpc-root-pos-sigma",
            "0.04",
            "--mpc-root-rot-sigma",
            "0.10",
            "--mpc-joint-sigma",
            "0.18",
            "--no-mpc-warm-start",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        config = evaluate._build_sampling_config(args)
        self.assertFalse(config.use_warm_start)

    def test_mjx_generic_accepts_mjx_guided_candidate_toggle(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--device",
            "cpu",
            "--mpc-samples",
            "512",
            "--mpc-iterations",
            "2",
            "--mpc-planning-horizon-steps",
            "40",
            "--mpc-control-steps",
            "20",
            "--mpc-knot-count",
            "8",
            "--mpc-temperature",
            "0.7",
            "--mpc-root-pos-sigma",
            "0.04",
            "--mpc-root-rot-sigma",
            "0.10",
            "--mpc-joint-sigma",
            "0.18",
            "--mjx-guided-candidate",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        config = evaluate._build_sampling_config(args)
        self.assertTrue(config.use_guided_candidate)

    def test_mjx_generic_accepts_guided_candidate_period(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--device",
            "cpu",
            "--mpc-samples",
            "512",
            "--mpc-iterations",
            "2",
            "--mpc-planning-horizon-steps",
            "40",
            "--mpc-control-steps",
            "20",
            "--mpc-knot-count",
            "8",
            "--mpc-temperature",
            "0.7",
            "--mpc-root-pos-sigma",
            "0.04",
            "--mpc-root-rot-sigma",
            "0.10",
            "--mpc-joint-sigma",
            "0.18",
            "--mjx-guided-candidate",
            "--mjx-guided-candidate-period",
            "5",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        config = evaluate._build_sampling_config(args)
        self.assertTrue(config.use_guided_candidate)
        self.assertEqual(config.guided_candidate_period, 5)

    def test_mjx_guided_candidate_toggle_requires_mjx_generic(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-optimizer",
            "generic",
            "--mjx-guided-candidate",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "--mjx-guided-candidate"):
            evaluate._validate_backend_args(args)

    def test_mjx_guided_candidate_period_requires_enabled_guided_candidate(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mjx-guided-candidate-period",
            "5",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "--mjx-guided-candidate-period"):
            evaluate._validate_backend_args(args)

    def test_mjx_guided_candidate_period_must_be_positive(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mjx-guided-candidate",
            "--mjx-guided-candidate-period",
            "0",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "positive"):
            evaluate._validate_backend_args(args)

    def test_mjx_generic_accepts_min_score_improvement(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--device",
            "cpu",
            "--mpc-samples",
            "512",
            "--mpc-iterations",
            "2",
            "--mpc-planning-horizon-steps",
            "40",
            "--mpc-control-steps",
            "20",
            "--mpc-knot-count",
            "8",
            "--mpc-temperature",
            "0.7",
            "--mpc-root-pos-sigma",
            "0.04",
            "--mpc-root-rot-sigma",
            "0.10",
            "--mpc-joint-sigma",
            "0.18",
            "--mjx-min-score-improvement",
            "0.01",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        config = evaluate._build_sampling_config(args)
        self.assertEqual(config.mjx_min_score_improvement, 0.01)

    def test_mjx_generic_accepts_top_score_gap_guards(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--device",
            "cpu",
            "--mpc-samples",
            "512",
            "--mpc-iterations",
            "2",
            "--mpc-planning-horizon-steps",
            "40",
            "--mpc-control-steps",
            "20",
            "--mpc-knot-count",
            "8",
            "--mpc-temperature",
            "0.7",
            "--mpc-root-pos-sigma",
            "0.04",
            "--mpc-root-rot-sigma",
            "0.10",
            "--mpc-joint-sigma",
            "0.18",
            "--mjx-min-top-score-gap",
            "0.02",
            "--mjx-cem-update-min-top-score-gap",
            "0.015",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        config = evaluate._build_sampling_config(args)
        self.assertEqual(config.mjx_min_top_score_gap, 0.02)
        self.assertEqual(config.mjx_cem_update_min_top_score_gap, 0.015)

    def test_mjx_generic_accepts_max_control_delta(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--device",
            "cpu",
            "--mpc-samples",
            "512",
            "--mpc-iterations",
            "2",
            "--mpc-planning-horizon-steps",
            "40",
            "--mpc-control-steps",
            "20",
            "--mpc-knot-count",
            "8",
            "--mpc-temperature",
            "0.7",
            "--mpc-root-pos-sigma",
            "0.04",
            "--mpc-root-rot-sigma",
            "0.10",
            "--mpc-joint-sigma",
            "0.18",
            "--mjx-max-control-delta",
            "0.25",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        config = evaluate._build_sampling_config(args)
        self.assertEqual(config.mjx_max_control_delta, 0.25)

    def test_mjx_generic_accepts_contact_force_active_weight(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mjx-contact-force-active-weight",
            "0.75",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        weights = evaluate._effective_reward_weights(
            args.method,
            None,
            mjx_contact_force_active_weight=args.mjx_contact_force_active_weight,
        )
        self.assertEqual(weights["contact_force_active"], 0.75)

    def test_mjx_contact_force_active_weight_requires_mjx_generic(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-optimizer",
            "generic",
            "--mjx-contact-force-active-weight",
            "0.75",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "--mjx-contact-force-active-weight"):
            evaluate._validate_backend_args(args)

    def test_mjx_contact_force_active_weight_must_be_non_negative(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mjx-contact-force-active-weight",
            "-0.1",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "non-negative"):
            evaluate._validate_backend_args(args)

    def test_mjx_generic_accepts_contact_force_peak_excess_weight(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mjx-contact-force-peak-excess-weight",
            "1.25",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        weights = evaluate._effective_reward_weights(
            args.method,
            None,
            mjx_contact_force_peak_excess_weight=(
                args.mjx_contact_force_peak_excess_weight
            ),
        )
        self.assertEqual(weights["contact_force_peak_excess"], 1.25)

    def test_mjx_contact_force_peak_excess_weight_requires_mjx_generic(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-optimizer",
            "generic",
            "--mjx-contact-force-peak-excess-weight",
            "1.25",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(
            ValueError,
            "--mjx-contact-force-peak-excess-weight",
        ):
            evaluate._validate_backend_args(args)

    def test_mjx_contact_force_peak_excess_weight_must_be_non_negative(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mjx-contact-force-peak-excess-weight",
            "-0.1",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "non-negative"):
            evaluate._validate_backend_args(args)

    def test_mjx_generic_can_override_contact_force_delta_weight(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mjx-contact-force-delta-weight",
            "0.0",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        weights = evaluate._effective_reward_weights(
            args.method,
            {"contact_force_delta": 1.0},
            mjx_contact_force_delta_weight=args.mjx_contact_force_delta_weight,
        )
        self.assertEqual(weights["contact_force_delta"], 0.0)

    def test_mjx_contact_force_delta_weight_requires_mjx_generic(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-optimizer",
            "generic",
            "--mjx-contact-force-delta-weight",
            "0.0",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "--mjx-contact-force-delta-weight"):
            evaluate._validate_backend_args(args)

    def test_mjx_contact_force_delta_weight_must_be_non_negative(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mjx-contact-force-delta-weight",
            "-0.1",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "non-negative"):
            evaluate._validate_backend_args(args)

    def test_mjx_generic_can_override_contact_false_positive_weight(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mjx-contact-false-positive-weight",
            "1.2",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        weights = evaluate._effective_reward_weights(
            args.method,
            {"contact_false_positive": 0.6},
            mjx_contact_false_positive_weight=(
                args.mjx_contact_false_positive_weight
            ),
        )
        self.assertEqual(weights["contact_false_positive"], 1.2)

    def test_mjx_contact_false_positive_weight_requires_mjx_generic(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-optimizer",
            "generic",
            "--mjx-contact-false-positive-weight",
            "1.2",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(
            ValueError,
            "--mjx-contact-false-positive-weight",
        ):
            evaluate._validate_backend_args(args)

    def test_mjx_contact_false_positive_weight_must_be_non_negative(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mjx-contact-false-positive-weight",
            "-0.1",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "non-negative"):
            evaluate._validate_backend_args(args)

    def test_mpc_payload_reward_weights_prefers_mjx_run_metadata(self) -> None:
        weights = evaluate._mpc_payload_reward_weights(
            "g1_wbc_joint_global",
            None,
            {
                "reward_weights": {
                    "body_global_pos_error": 58.0,
                    "contact_force_active": 0.5,
                }
            },
        )

        self.assertEqual(weights["body_global_pos_error"], 58.0)
        self.assertEqual(weights["contact_force_active"], 0.5)

    def test_mjx_generic_accepts_candidate_rank_diagnostics(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--device",
            "cpu",
            "--mpc-samples",
            "512",
            "--mpc-iterations",
            "2",
            "--mpc-planning-horizon-steps",
            "40",
            "--mpc-control-steps",
            "20",
            "--mpc-knot-count",
            "8",
            "--mpc-temperature",
            "0.7",
            "--mpc-root-pos-sigma",
            "0.04",
            "--mpc-root-rot-sigma",
            "0.10",
            "--mpc-joint-sigma",
            "0.18",
            "--mjx-candidate-rank-diagnostics-top-k",
            "8",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        config = evaluate._build_sampling_config(args)
        self.assertEqual(config.mjx_candidate_rank_diagnostics_top_k, 8)

    def test_mjx_generic_accepts_candidate_rescore_diagnostics(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--device",
            "cpu",
            "--mpc-samples",
            "512",
            "--mpc-iterations",
            "2",
            "--mpc-planning-horizon-steps",
            "40",
            "--mpc-control-steps",
            "20",
            "--mpc-knot-count",
            "8",
            "--mpc-temperature",
            "0.7",
            "--mpc-root-pos-sigma",
            "0.04",
            "--mpc-root-rot-sigma",
            "0.10",
            "--mpc-joint-sigma",
            "0.18",
            "--mjx-candidate-rescore-diagnostics",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        config = evaluate._build_sampling_config(args)
        self.assertTrue(config.mjx_candidate_rescore_diagnostics)

    def test_mjx_generic_accepts_candidate_score_component_diagnostics(
        self,
    ) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--device",
            "cpu",
            "--mpc-samples",
            "512",
            "--mpc-iterations",
            "2",
            "--mpc-planning-horizon-steps",
            "40",
            "--mpc-control-steps",
            "20",
            "--mpc-knot-count",
            "8",
            "--mpc-temperature",
            "0.7",
            "--mpc-root-pos-sigma",
            "0.04",
            "--mpc-root-rot-sigma",
            "0.10",
            "--mpc-joint-sigma",
            "0.18",
            "--mjx-candidate-score-component-diagnostics-top-k",
            "6",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        evaluate._validate_backend_args(args)
        config = evaluate._build_sampling_config(args)
        self.assertEqual(config.mjx_candidate_score_component_diagnostics_top_k, 6)

    def test_candidate_score_component_diagnostics_requires_mjx_generic(
        self,
    ) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-optimizer",
            "generic",
            "--mjx-candidate-score-component-diagnostics-top-k",
            "2",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(
            ValueError,
            "--mjx-candidate-score-component-diagnostics-top-k",
        ):
            evaluate._validate_backend_args(args)

    def test_candidate_score_component_diagnostics_must_be_non_negative(
        self,
    ) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mjx-candidate-score-component-diagnostics-top-k",
            "-1",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "score-component"):
            evaluate._validate_backend_args(args)

    def test_mjx_max_control_delta_requires_mjx_generic(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-optimizer",
            "generic",
            "--mjx-max-control-delta",
            "0.25",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "--mjx-max-control-delta"):
            evaluate._validate_backend_args(args)

    def test_mjx_max_control_delta_must_be_positive(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mjx-max-control-delta",
            "0",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "positive"):
            evaluate._validate_backend_args(args)

    def test_candidate_rank_diagnostics_requires_mjx_generic(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-optimizer",
            "generic",
            "--mjx-candidate-rank-diagnostics-top-k",
            "8",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(
            ValueError,
            "--mjx-candidate-rank-diagnostics-top-k",
        ):
            evaluate._validate_backend_args(args)

    def test_candidate_rank_diagnostics_must_be_non_negative(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mjx-candidate-rank-diagnostics-top-k",
            "-1",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "non-negative"):
            evaluate._validate_backend_args(args)

    def test_candidate_rescore_diagnostics_requires_mjx_generic(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-optimizer",
            "generic",
            "--mjx-candidate-rescore-diagnostics",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(
            ValueError,
            "--mjx-candidate-rescore-diagnostics",
        ):
            evaluate._validate_backend_args(args)

    def test_mjx_min_score_improvement_requires_mjx_generic(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-optimizer",
            "generic",
            "--mjx-min-score-improvement",
            "0.01",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "--mjx-min-score-improvement"):
            evaluate._validate_backend_args(args)

    def test_mjx_min_score_improvement_must_be_non_negative(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mjx",
            "--mpc-optimizer",
            "generic",
            "--mjx-min-score-improvement",
            "-0.01",
        ]

        with mock.patch("sys.argv", argv):
            args = evaluate._parse_args()

        with self.assertRaisesRegex(ValueError, "non-negative"):
            evaluate._validate_backend_args(args)

    def test_mjx_top_score_gap_guards_require_mjx_generic(self) -> None:
        for flag in (
            "--mjx-min-top-score-gap",
            "--mjx-cem-update-min-top-score-gap",
        ):
            with self.subTest(flag=flag):
                argv = [
                    "evaluate.py",
                    "--motion",
                    "/tmp/motion.npz",
                    "--method",
                    "g1_wbc_joint_global",
                    "--mpc-optimizer",
                    "generic",
                    flag,
                    "0.01",
                ]

                with mock.patch("sys.argv", argv):
                    args = evaluate._parse_args()

                with self.assertRaisesRegex(ValueError, flag):
                    evaluate._validate_backend_args(args)

    def test_mjx_top_score_gap_guards_must_be_non_negative(self) -> None:
        for flag in (
            "--mjx-min-top-score-gap",
            "--mjx-cem-update-min-top-score-gap",
        ):
            with self.subTest(flag=flag):
                argv = [
                    "evaluate.py",
                    "--motion",
                    "/tmp/motion.npz",
                    "--method",
                    "g1_wbc_joint_global",
                    "--mpc-backend",
                    "mjx",
                    "--mpc-optimizer",
                    "generic",
                    flag,
                    "-0.01",
                ]

                with mock.patch("sys.argv", argv):
                    args = evaluate._parse_args()

                with self.assertRaisesRegex(ValueError, "non-negative"):
                    evaluate._validate_backend_args(args)

    def test_file_sha256_hashes_saved_command_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            command = Path(tmp_dir) / "mpc_command.npz"
            command.write_bytes(b"saved command bytes")

            digest = evaluate._file_sha256(command)

        self.assertEqual(
            digest,
            "368e0095ec392ef10cec152a60864c463232a071f6266307cb5924a2e2ad487e",
        )


if __name__ == "__main__":
    unittest.main()
