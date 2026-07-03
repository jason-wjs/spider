from __future__ import annotations

# ruff: noqa: D101,D102
import io
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import torch

from spider.tasks.g1_wbc import evaluate, mpc


class G1WbcMpcWarmStartTest(unittest.TestCase):
    def test_parse_args_accepts_legacy_mpc_optimizer(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-optimizer",
            "legacy",
        ]

        with patch.object(sys, "argv", argv):
            args = evaluate._parse_args()

        self.assertEqual(args.mpc_optimizer, "legacy")

    def test_main_routes_legacy_mujoco_warp_to_legacy_optimizer(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--checkpoint",
            "/tmp/model.pt",
            "--device",
            "cpu",
            "--method",
            "g1_wbc_joint_global",
            "--mpc-backend",
            "mujoco_warp",
            "--mpc-optimizer",
            "legacy",
            "--mpc-samples",
            "512",
            "--mpc-iterations",
            "2",
            "--mpc-planning-horizon-steps",
            "40",
            "--mpc-control-steps",
            "20",
            "--mpc-sampling-mode",
            "knot",
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
        legacy_result = SimpleNamespace(
            command=SimpleNamespace(controls=torch.zeros(1, 1)),
            rollout=SimpleNamespace(),
            refined_qpos=torch.zeros(1, 36),
            scores=torch.zeros(1),
            history=[],
            accepted=True,
            used_baseline_fallback=False,
            final_candidate_score=1.0,
            final_baseline_score=0.5,
            num_windows=2,
            accepted_windows=2,
        )

        with patch.object(sys, "argv", argv), \
            patch.object(
                evaluate,
                "load_motion",
                return_value=SimpleNamespace(num_frames=41, motion_type="isaaclab"),
            ), \
            patch.object(evaluate, "validate_motion_dims"), \
            patch.object(evaluate, "resolve_checkpoint_path", return_value="/tmp/model.pt"), \
            patch.object(evaluate, "load_wbc_actor", return_value=SimpleNamespace()), \
            patch.object(evaluate, "compute_rollout_metrics", return_value={"num_steps": 40}), \
            patch.object(evaluate, "optimize_mpc_command", return_value=legacy_result) as legacy, \
            patch.object(evaluate, "run_g1_wbc_sampling_mpc") as generic, \
            redirect_stdout(io.StringIO()):
            evaluate.main()

        legacy.assert_called_once()
        generic.assert_not_called()
        mpc_config = legacy.call_args.args[3]
        self.assertEqual(mpc_config.num_samples, 512)
        self.assertEqual(mpc_config.sampling_mode, "knot")

    def test_config_defaults_preserve_disabled_warm_start(self) -> None:
        config = mpc.G1WbcMpcConfig()

        self.assertFalse(config.use_warm_start)
        self.assertEqual(config.warm_start_source, "best")
        self.assertEqual(config.warm_start_decay, 1.0)

    def test_parse_args_builds_warm_start_config(self) -> None:
        argv = [
            "evaluate.py",
            "--motion",
            "/tmp/motion.npz",
            "--method",
            "g1_wbc_joint",
            "--mpc-warm-start",
            "--mpc-warm-start-source",
            "mean",
            "--mpc-warm-start-decay",
            "0.25",
        ]

        with patch.object(sys, "argv", argv):
            args = evaluate._parse_args()
        config = evaluate._build_mpc_config(args)

        self.assertTrue(config.use_warm_start)
        self.assertEqual(config.warm_start_source, "mean")
        self.assertEqual(config.warm_start_decay, 0.25)

    def test_warm_start_mean_delta_shifts_decays_and_zero_pads_full(self) -> None:
        config = mpc.G1WbcMpcConfig(
            sampling_mode="full",
            freeze_first_frame=False,
            warm_start_decay=0.5,
        )
        dim = mpc.QPOS_DIM - 1
        previous_delta = torch.arange(6 * dim, dtype=torch.float32).reshape(6, dim)

        warm_start = mpc._warm_start_mean_delta(
            previous_delta,
            execute_steps=2,
            horizon=5,
            config=config,
        )

        assert warm_start is not None
        expected = torch.cat(
            [
                previous_delta[2:] * 0.5,
                torch.zeros(1, dim, dtype=torch.float32),
            ],
            dim=0,
        )
        torch.testing.assert_close(warm_start, expected)

    def test_warm_start_mean_delta_converts_shifted_horizon_to_knots(self) -> None:
        config = mpc.G1WbcMpcConfig(
            sampling_mode="knot",
            knot_count=3,
            freeze_first_frame=False,
        )
        dim = mpc.QPOS_DIM - 1
        previous_delta = torch.arange(8, dtype=torch.float32)[:, None].expand(8, dim)

        warm_start = mpc._warm_start_mean_delta(
            previous_delta,
            execute_steps=2,
            horizon=5,
            config=config,
        )

        assert warm_start is not None
        expected = torch.tensor([2.0, 4.0, 6.0])[:, None].expand(3, dim)
        torch.testing.assert_close(warm_start, expected)


if __name__ == "__main__":
    unittest.main()
