from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

RUNNER_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "run_g1_wbc_stage0_baseline.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("stage0_baseline_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Stage0BaselineRunnerTest(unittest.TestCase):
    def test_parse_args_accepts_stage0_inputs(self) -> None:
        runner = load_runner()

        args = runner.parse_args(
            [
                "--jump-motion",
                "/tmp/missing/jump.npz",
                "--walk-motion",
                "/tmp/missing/walk.npz",
                "--checkpoint",
                "/tmp/missing/checkpoint.pt",
                "--reward-weights",
                "/tmp/missing/reward.json",
                "--output-dir",
                "/tmp/stage0",
                "--dry-run",
            ]
        )

        self.assertEqual(args.jump_motion, Path("/tmp/missing/jump.npz"))
        self.assertEqual(args.walk_motion, Path("/tmp/missing/walk.npz"))
        self.assertEqual(args.checkpoint, "/tmp/missing/checkpoint.pt")
        self.assertEqual(args.reward_weights, Path("/tmp/missing/reward.json"))
        self.assertEqual(args.output_dir, Path("/tmp/stage0"))
        self.assertTrue(args.dry_run)

    def test_build_stage0_commands_covers_motions_and_seeds(self) -> None:
        runner = load_runner()
        args = runner.parse_args(
            [
                "--jump-motion",
                "/tmp/missing/jump.npz",
                "--walk-motion",
                "/tmp/missing/walk.npz",
                "--checkpoint",
                "model.pt",
                "--reward-weights",
                "/tmp/missing/reward.json",
                "--output-dir",
                "/tmp/stage0",
                "--dry-run",
            ]
        )

        commands = runner.build_stage0_commands(args)

        self.assertEqual(len(commands), 6)
        self.assertEqual({command.motion_name for command in commands}, {"jump", "walk"})
        self.assertEqual({command.seed for command in commands}, {0, 1, 2})
        for command in commands:
            with self.subTest(motion=command.motion_name, seed=command.seed):
                self.assertIn("--mpc-backend", command.argv)
                backend_index = command.argv.index("--mpc-backend")
                self.assertEqual(command.argv[backend_index + 1], "mujoco_warp")
                self.assertIn("--save-rollout", command.argv)

    def test_attach_artifact_paths_uses_none_for_missing_files(self) -> None:
        runner = load_runner()
        row = {
            "output_dir": "/tmp/stage0/missing_eval_output",
        }

        updated = runner.attach_artifact_paths(row)

        self.assertIsNone(updated["artifacts"]["metrics_json"])
        self.assertIsNone(updated["artifacts"]["rollout_npz"])
        self.assertIsNone(updated["artifacts"]["mpc_command_npz"])


if __name__ == "__main__":
    unittest.main()
