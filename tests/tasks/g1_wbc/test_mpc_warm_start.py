from __future__ import annotations

# ruff: noqa: D101,D102
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import torch

from spider.tasks.g1_wbc import evaluate, mpc
from spider.tasks.g1_wbc.constants import ACTION_DIM, MUJOCO_BODY_NAMES, QPOS_DIM, QVEL_DIM
from spider.tasks.g1_wbc.motion import G1Motion
from spider.tasks.g1_wbc.rollout import RolloutResult


class _FakeActor:
    def to(self, device):
        del device
        return self

    def eval(self):
        return self


def _fake_motion(frames: int = 5) -> G1Motion:
    bodies = len(MUJOCO_BODY_NAMES)
    qpos = torch.zeros(frames, QPOS_DIM)
    qpos[:, 3] = 1.0
    body_quat = torch.zeros(frames, bodies, 4)
    body_quat[..., 0] = 1.0
    return G1Motion(
        path=None,
        motion_type="mujoco",
        fps=50.0,
        joint_pos=torch.zeros(frames, ACTION_DIM),
        joint_vel=torch.zeros(frames, ACTION_DIM),
        body_pos_w=torch.zeros(frames, bodies, 3),
        body_quat_w=body_quat,
        body_lin_vel_w=torch.zeros(frames, bodies, 3),
        body_ang_vel_w=torch.zeros(frames, bodies, 3),
        contact=torch.zeros(frames, 2),
    )


def _fake_rollout_result(steps: int, *, initial_qpos: torch.Tensor) -> RolloutResult:
    bodies = len(MUJOCO_BODY_NAMES)
    frames = int(steps) + 1
    qpos = initial_qpos.detach().clone().view(1, 1, QPOS_DIM).repeat(frames, 1, 1)
    qpos[:, :, 3] = 1.0
    qpos[-1, 0, 0] = qpos[0, 0, 0] + float(steps)
    body_quat = torch.zeros(frames, 1, bodies, 4)
    body_quat[..., 0] = 1.0
    final_action = torch.full((1, ACTION_DIM), float(steps), dtype=torch.float32)
    return RolloutResult(
        qpos=qpos,
        qvel=torch.zeros(frames, 1, QVEL_DIM),
        body_pos_w=torch.zeros(frames, 1, bodies, 3),
        body_quat_w=body_quat,
        body_lin_vel_w=torch.zeros(frames, 1, bodies, 3),
        body_ang_vel_w=torch.zeros(frames, 1, bodies, 3),
        actions=torch.zeros(steps, 1, ACTION_DIM),
        controls=torch.zeros(steps, 1, ACTION_DIM),
        contact_indicator=torch.zeros(frames, 1, 2),
        contact_force=torch.zeros(frames, 1, 2),
        floor_contact_indicator=torch.zeros(frames, 1, 3),
        floor_contact_force=torch.zeros(frames, 1, 3),
        ref_indices=torch.arange(frames).view(frames, 1),
        final_last_action=final_action,
        final_history_state={
            "actions": {
                "data": torch.full((1, 2), float(steps), dtype=torch.float32),
                "index": 1,
            }
        },
    )


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

        stdout = io.StringIO()
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
            redirect_stdout(stdout):
            evaluate.main()

        legacy.assert_called_once()
        generic.assert_not_called()
        mpc_config = legacy.call_args.args[3]
        self.assertEqual(mpc_config.num_samples, 512)
        self.assertEqual(mpc_config.sampling_mode, "knot")
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["mpc"]["config"]["num_samples"], 512)
        self.assertEqual(payload["mpc"]["config"]["sampling_mode"], "knot")
        self.assertEqual(payload["mpc"]["config"]["planning_horizon_steps"], 40)
        self.assertEqual(payload["mpc"]["config"]["control_steps"], 20)
        self.assertEqual(payload["mpc"]["config"]["root_pos_sigma"], 0.04)
        self.assertNotIn("reward_weights", payload["mpc"]["config"])

    def test_config_defaults_preserve_disabled_warm_start(self) -> None:
        config = mpc.G1WbcMpcConfig()

        self.assertFalse(config.use_warm_start)
        self.assertEqual(config.warm_start_source, "best")
        self.assertEqual(config.warm_start_decay, 1.0)

    def test_diagnostic_window_hook_receives_closed_loop_carry_state(self) -> None:
        def fake_optimize(*args, **kwargs):
            del args
            horizon = int(kwargs["horizon"])
            best_qpos = torch.zeros(horizon, QPOS_DIM)
            best_qpos[:, 3] = 1.0
            return mpc._MpcWindowOptimizeResult(
                best_qpos=best_qpos,
                best_delta=torch.zeros(horizon, QPOS_DIM - 1),
                mean_delta=torch.zeros(horizon, QPOS_DIM - 1),
                scores=torch.tensor([1.0, 0.0]),
                history=[],
                best_is_template=False,
                best_score=1.0,
                zero_delta_score=0.0,
            )

        def fake_command_builder(template_motion, qpos_trajectory, config, **kwargs):
            del template_motion, config, kwargs
            return SimpleNamespace(num_frames=int(qpos_trajectory.shape[0]))

        def fake_rollout(command, actor, config, **kwargs):
            del command, actor
            return _fake_rollout_result(
                int(config.max_steps),
                initial_qpos=kwargs["initial_qpos"],
            )

        hooks: list[dict] = []
        config = mpc.G1WbcMpcConfig(
            num_samples=2,
            num_iterations=1,
            planning_horizon_steps=2,
            control_steps=2,
            acceptance_gate=False,
        )
        rollout_config = mpc.WbcRolloutConfig(device="cpu", num_envs=1, max_steps=4)

        with patch.object(mpc, "_optimize_mpc_window", side_effect=fake_optimize), \
            patch.object(
                mpc,
                "command_batch_from_qpos_trajectory",
                side_effect=fake_command_builder,
            ), \
            patch.object(mpc, "run_command_rollout", side_effect=fake_rollout), \
            patch.object(mpc, "_single_rollout_score", return_value=1.0):
            result = mpc.optimize_mpc_command(
                _fake_motion(frames=5),
                _FakeActor(),
                rollout_config,
                config,
                diagnostic_window_hook=hooks.append,
            )

        self.assertEqual(result.num_windows, 2)
        self.assertEqual([item["start"] for item in hooks], [0, 2])
        self.assertEqual([item["execute_steps"] for item in hooks], [2, 2])
        self.assertIsNone(hooks[0]["initial_last_action"])
        self.assertIsNone(hooks[0]["initial_history_state"])
        torch.testing.assert_close(
            hooks[1]["initial_last_action"],
            torch.full((ACTION_DIM,), 2.0),
        )
        self.assertIn("actions", hooks[1]["initial_history_state"])
        self.assertEqual(float(hooks[1]["initial_qpos"][0]), 2.0)
        hooks[1]["initial_qpos"][0] = 999.0
        self.assertEqual(float(result.rollout.qpos[2, 0, 0]), 2.0)

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
