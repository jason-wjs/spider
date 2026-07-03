from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
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

    def test_main_fails_fast_when_stage0_input_paths_are_missing(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "stage0"

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = runner.main(
                    [
                        "--jump-motion",
                        str(Path(tmp_dir) / "missing_jump.npz"),
                        "--walk-motion",
                        str(Path(tmp_dir) / "missing_walk.npz"),
                        "--checkpoint",
                        str(Path(tmp_dir) / "missing_model.pt"),
                        "--reward-weights",
                        str(Path(tmp_dir) / "missing_rewards.json"),
                        "--output-dir",
                        str(output_root),
                        "--dry-run",
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("missing input: jump motion", stderr.getvalue())
        self.assertFalse((output_root / "baseline_manifest.json").exists())

    def test_main_fails_fast_when_checkpoint_alias_is_unresolved(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = root / "stage0"
            jump_motion = root / "jump.npz"
            walk_motion = root / "walk.npz"
            reward_weights = root / "reward.json"
            for path in (jump_motion, walk_motion, reward_weights):
                path.write_text("{}")

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = runner.main(
                    [
                        "--jump-motion",
                        str(jump_motion),
                        "--walk-motion",
                        str(walk_motion),
                        "--checkpoint",
                        "bc",
                        "--reward-weights",
                        str(reward_weights),
                        "--output-dir",
                        str(output_root),
                        "--dry-run",
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("missing input: checkpoint", stderr.getvalue())
        self.assertFalse((output_root / "baseline_manifest.json").exists())

    def test_main_fails_fast_when_real_cuda_run_has_multiple_visible_gpus(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = root / "stage0"
            jump_motion = root / "jump.npz"
            walk_motion = root / "walk.npz"
            checkpoint = root / "model.pt"
            reward_weights = root / "reward.json"
            for path in (jump_motion, walk_motion, checkpoint, reward_weights):
                path.write_text("{}")

            stderr = io.StringIO()
            with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}):
                with redirect_stderr(stderr):
                    with mock.patch.object(
                        runner,
                        "run_command",
                        side_effect=AssertionError("run_command should not be called"),
                    ):
                        exit_code = runner.main(
                            [
                                "--jump-motion",
                                str(jump_motion),
                                "--walk-motion",
                                str(walk_motion),
                                "--checkpoint",
                                str(checkpoint),
                                "--reward-weights",
                                str(reward_weights),
                                "--output-dir",
                                str(output_root),
                                "--device",
                                "cuda:0",
                            ]
                        )

        self.assertEqual(exit_code, 2)
        self.assertIn("single GPU visibility", stderr.getvalue())
        self.assertFalse((output_root / "baseline_manifest.json").exists())

    def test_main_fails_fast_when_checkpoint_is_not_wbc_actor_format(self) -> None:
        import torch

        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = root / "stage0"
            jump_motion = root / "jump.npz"
            walk_motion = root / "walk.npz"
            checkpoint = root / "transformer.pt"
            reward_weights = root / "reward.json"
            for path in (jump_motion, walk_motion, reward_weights):
                path.write_text("{}")
            torch.save(
                {"model_state_dict": {"actor.projection_head.weight": torch.zeros(1, 1)}},
                checkpoint,
            )

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = runner.main(
                    [
                        "--jump-motion",
                        str(jump_motion),
                        "--walk-motion",
                        str(walk_motion),
                        "--checkpoint",
                        str(checkpoint),
                        "--reward-weights",
                        str(reward_weights),
                        "--output-dir",
                        str(output_root),
                        "--dry-run",
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("checkpoint format", stderr.getvalue())
        self.assertFalse((output_root / "baseline_manifest.json").exists())

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

    def test_attach_artifact_paths_records_mtime_for_existing_files(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            for name in ("metrics.json", "rollout.npz", "mpc_command.npz"):
                (output_dir / name).write_text("{}")
            row = {"output_dir": str(output_dir)}

            updated = runner.attach_artifact_paths(row)

        self.assertEqual(
            set(updated["artifact_mtime_ns"]),
            {"metrics_json", "rollout_npz", "mpc_command_npz"},
        )
        self.assertTrue(
            all(isinstance(value, int) for value in updated["artifact_mtime_ns"].values())
        )
        self.assertEqual(
            set(updated["artifact_sha256"]),
            {"metrics_json", "rollout_npz", "mpc_command_npz"},
        )
        self.assertTrue(
            all(
                isinstance(value, str) and len(value) == 64
                for value in updated["artifact_sha256"].values()
            )
        )

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
                            "steady_state_wall_time_sec": 123.45,
                            "runtime_visible_devices": ["0"],
                            "runtime_gpu_name": "NVIDIA H100 80GB HBM3",
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
        self.assertEqual(row["steady_state_wall_time_sec"], 123.45)
        self.assertEqual(row["runtime_visible_devices"], ["0"])
        self.assertEqual(row["runtime_gpu_name"], "NVIDIA H100 80GB HBM3")
        self.assertIsInstance(row["command_start_time_ns"], int)

    def test_main_writes_ok_status_for_successful_real_run(self) -> None:
        import torch

        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = Path(tmp_dir) / "stage0"
            jump_motion = root / "jump.npz"
            walk_motion = root / "walk.npz"
            checkpoint = root / "model.pt"
            reward_weights = root / "reward.json"
            for path in (jump_motion, walk_motion, reward_weights):
                path.write_text("{}")
            torch.save(
                {
                    "actor_state_dict": {
                        "obs_normalizer._mean": torch.zeros(1, 886),
                        "obs_normalizer._std": torch.ones(1, 886),
                        "mlp.0.weight": torch.zeros(1, 1),
                    }
                },
                checkpoint,
            )

            def fake_run_command(command):
                output_dir = Path(command.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                for name in ("metrics.json", "rollout.npz", "mpc_command.npz"):
                    (output_dir / name).write_text("{}")
                return {
                    "returncode": 0,
                    "stdout": "{}",
                    "stderr": "",
                    "metrics": {"num_steps": 800, "success": True, "score": -2.0},
                    "mpc_accepted": True,
                    "accepted_windows": 40,
                    "mpc_used_baseline_fallback": False,
                    "num_steps": 800,
                    "steady_state_wall_time_sec": 120.0,
                }

            argv = [
                "--jump-motion",
                str(jump_motion),
                "--walk-motion",
                str(walk_motion),
                "--checkpoint",
                str(checkpoint),
                "--reward-weights",
                str(reward_weights),
                "--output-dir",
                str(output_root),
            ]
            with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
                with mock.patch.object(runner, "run_command", side_effect=fake_run_command):
                    exit_code = runner.main(argv)

            manifest = json.loads((output_root / "baseline_manifest.json").read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual({row["status"] for row in manifest["rows"]}, {"ok"})
        self.assertTrue(all(row["artifacts"]["metrics_json"] for row in manifest["rows"]))
        self.assertEqual(
            {row["steady_state_wall_time_sec"] for row in manifest["rows"]},
            {120.0},
        )

    def test_main_resolves_checkpoint_directory_to_file_in_manifest(self) -> None:
        import torch

        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = root / "stage0"
            jump_motion = root / "jump.npz"
            walk_motion = root / "walk.npz"
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            checkpoint = checkpoint_dir / "model_20.pt"
            reward_weights = root / "reward.json"
            jump_motion.write_text("jump")
            walk_motion.write_text("walk")
            reward_weights.write_text("{}")
            torch.save(
                {
                    "actor_state_dict": {
                        "obs_normalizer._mean": torch.zeros(1, 886),
                        "obs_normalizer._std": torch.ones(1, 886),
                        "mlp.0.weight": torch.zeros(1, 1),
                    }
                },
                checkpoint,
            )

            exit_code = runner.main(
                [
                    "--jump-motion",
                    str(jump_motion),
                    "--walk-motion",
                    str(walk_motion),
                    "--checkpoint",
                    str(checkpoint_dir),
                    "--reward-weights",
                    str(reward_weights),
                    "--output-dir",
                    str(output_root),
                    "--dry-run",
                ]
            )
            manifest = json.loads((output_root / "baseline_manifest.json").read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["input_paths"]["checkpoint"], str(checkpoint.resolve()))
        self.assertTrue(
            all(
                row["argv"][row["argv"].index("--checkpoint") + 1]
                == str(checkpoint.resolve())
                for row in manifest["rows"]
            )
        )

    def test_main_dry_run_records_manifest_provenance_and_input_hashes(self) -> None:
        import torch

        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = root / "stage0"
            jump_motion = root / "jump.npz"
            walk_motion = root / "walk.npz"
            checkpoint = root / "model.pt"
            reward_weights = root / "reward.json"
            jump_motion.write_text("jump")
            walk_motion.write_text("walk")
            reward_weights.write_text("{}")
            torch.save(
                {
                    "actor_state_dict": {
                        "obs_normalizer._mean": torch.zeros(1, 886),
                        "obs_normalizer._std": torch.ones(1, 886),
                        "mlp.0.weight": torch.zeros(1, 1),
                    }
                },
                checkpoint,
            )

            exit_code = runner.main(
                [
                    "--jump-motion",
                    str(jump_motion),
                    "--walk-motion",
                    str(walk_motion),
                    "--checkpoint",
                    str(checkpoint),
                    "--reward-weights",
                    str(reward_weights),
                    "--output-dir",
                    str(output_root),
                    "--dry-run",
                ]
            )
            manifest = json.loads((output_root / "baseline_manifest.json").read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(manifest["provenance"]["worktree_path"], str(runner.SPIDER_ROOT))
        self.assertIsInstance(manifest["provenance"]["git_commit"], str)
        self.assertTrue(manifest["provenance"]["git_commit"])
        self.assertEqual(
            set(manifest["input_sha256"]),
            {"jump_motion", "walk_motion", "checkpoint", "reward_weights"},
        )
        self.assertTrue(
            all(len(value) == 64 for value in manifest["input_sha256"].values())
        )


if __name__ == "__main__":
    unittest.main()
