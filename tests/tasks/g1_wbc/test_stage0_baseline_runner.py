from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
        self.assertEqual(args.motion_type, "isaaclab")
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
                motion_type_index = command.argv.index("--motion-type")
                self.assertEqual(command.argv[motion_type_index + 1], "isaaclab")

    def test_build_stage0_commands_forwards_motion_type_override(self) -> None:
        runner = load_runner()
        args = runner.parse_args(
            [
                "--jump-motion",
                "/tmp/missing/jump.npz",
                "--walk-motion",
                "/tmp/missing/walk.npz",
                "--motion-type",
                "mujoco",
                "--checkpoint",
                "model.pt",
                "--reward-weights",
                "/tmp/missing/reward.json",
                "--output-dir",
                "/tmp/stage0",
                "--dry-run",
            ]
        )

        command = runner.build_stage0_commands(args)[0]

        motion_type_index = command.argv.index("--motion-type")
        self.assertEqual(command.argv[motion_type_index + 1], "mujoco")

    def test_attach_artifact_paths_uses_none_for_missing_files(self) -> None:
        runner = load_runner()
        row = {
            "output_dir": "/tmp/stage0/missing_eval_output",
        }

        updated = runner.attach_artifact_paths(row)

        self.assertIsNone(updated["artifacts"]["metrics_json"])
        self.assertIsNone(updated["artifacts"]["rollout_npz"])
        self.assertIsNone(updated["artifacts"]["mpc_command_npz"])

    def test_run_command_extracts_metrics_and_mpc_payload(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "jump" / "seed_0"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "metrics": {"num_steps": 800, "success": True, "score": -2.0},
                        "mpc": {
                            "accepted": True,
                            "accepted_windows": 40,
                            "used_baseline_fallback": False,
                        },
                    }
                )
            )
            command = runner.Stage0Command(
                motion_name="jump",
                motion="/tmp/missing/jump.npz",
                seed=0,
                output_dir=str(output_dir),
                argv=["python", "-m", "spider.tasks.g1_wbc.evaluate"],
                command_text="python -m spider.tasks.g1_wbc.evaluate",
            )
            completed = mock.Mock(returncode=0, stdout="{}", stderr="")

            with mock.patch.object(runner.subprocess, "run", return_value=completed):
                row = runner.run_command(command)

        self.assertEqual(row["metrics"]["score"], -2.0)
        self.assertTrue(row["mpc_accepted"])
        self.assertEqual(row["accepted_windows"], 40)
        self.assertFalse(row["mpc_used_baseline_fallback"])
        self.assertEqual(row["num_steps"], 800)


if __name__ == "__main__":
    unittest.main()
