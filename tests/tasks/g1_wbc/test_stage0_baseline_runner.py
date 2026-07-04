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
    def test_parse_args_defaults_to_versioned_wbc_results_assets(self) -> None:
        runner = load_runner()

        args = runner.parse_args(["--output-dir", "/tmp/stage0", "--dry-run"])

        self.assertEqual(
            args.jump_motion,
            runner.WBC_RESULTS_ROOT / "assets" / "motion_data" / "jump" / "motion.npz",
        )
        self.assertEqual(
            args.walk_motion,
            runner.WBC_RESULTS_ROOT / "assets" / "motion_data" / "walk" / "motion.npz",
        )
        self.assertEqual(
            args.reward_weights,
            runner.WBC_RESULTS_ROOT
            / "g1_body_tracking_wbc"
            / "spider"
            / "2026-06-23-mechanism-quality-speed-wjs"
            / "configs"
            / "g1_wbc_reward_weights_method_specific_v14_20260612.json",
        )
        self.assertEqual(
            args.checkpoint,
            str(runner.WBC_RESULTS_ROOT / "assets" / "checkpoints" / "model_8000.pt"),
        )

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
                self.assertIn("--mpc-preset", command.argv)
                preset_index = command.argv.index("--mpc-preset")
                self.assertEqual(command.argv[preset_index + 1], "aggressive")
                self.assertIn("--mpc-optimizer", command.argv)
                optimizer_index = command.argv.index("--mpc-optimizer")
                self.assertEqual(command.argv[optimizer_index + 1], "legacy")
                for flag, value in (
                    ("--mpc-samples", "512"),
                    ("--mpc-iterations", "2"),
                    ("--mpc-planning-horizon-steps", "40"),
                    ("--mpc-control-steps", "20"),
                    ("--mpc-sampling-mode", "knot"),
                    ("--mpc-knot-count", "8"),
                    ("--mpc-elite-frac", "0.125"),
                    ("--mpc-temperature", "0.7"),
                    ("--mpc-root-pos-sigma", "0.04"),
                    ("--mpc-root-rot-sigma", "0.10"),
                    ("--mpc-joint-sigma", "0.18"),
                    ("--mpc-sigma-decay", "0.75"),
                    ("--mpc-smooth-passes", "0"),
                    ("--mpc-command-reg-weight", "0.0"),
                    ("--mpc-command-smooth-weight", "0.0"),
                    ("--mpc-guided-root-pos-gain", "0.50"),
                    ("--mpc-guided-root-rot-gain", "0.50"),
                    ("--mpc-guided-joint-gain", "0.50"),
                    ("--mpc-guided-root-pos-clip", "0.05"),
                    ("--mpc-guided-root-rot-clip", "0.12"),
                    ("--mpc-guided-joint-clip", "0.35"),
                    ("--mpc-warm-start-source", "best"),
                    ("--mpc-warm-start-decay", "1.0"),
                    ("--nconmax-per-env", "512"),
                    ("--njmax-per-env", "2048"),
                ):
                    self.assertIn(flag, command.argv)
                    flag_index = command.argv.index(flag)
                    self.assertEqual(command.argv[flag_index + 1], value)
                for flag in (
                    "--mpc-guided-candidate",
                    "--mpc-acceptance-gate",
                    "--no-mpc-warm-start",
                ):
                    self.assertIn(flag, command.argv)

    def test_build_stage0_commands_can_select_single_parallel_shard_row(self) -> None:
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
                "--only-motion",
                "walk",
                "--only-seed",
                "1",
            ]
        )

        commands = runner.build_stage0_commands(args)

        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0].motion_name, "walk")
        self.assertEqual(commands[0].seed, 1)
        self.assertTrue(commands[0].output_dir.endswith("/walk/seed_1"))

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

    def test_main_fails_fast_when_visible_gpu_has_compute_processes(self) -> None:
        import subprocess
        import torch

        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = root / "stage0"
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

            busy_gpu = subprocess.CompletedProcess(
                args=["nvidia-smi"],
                returncode=0,
                stdout="12345, python, 13269 MiB\n",
                stderr="",
            )
            stderr = io.StringIO()
            with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2"}):
                with mock.patch.object(runner.subprocess, "run", return_value=busy_gpu):
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
        self.assertIn("GPU contention", stderr.getvalue())
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

    def test_run_command_missing_or_invalid_mpc_accepted_fails_closed(self) -> None:
        runner = load_runner()
        invalid_values = {
            "missing": None,
            "none": None,
            "string": "false",
        }
        for name, accepted in invalid_values.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    output_dir = Path(tmp_dir) / "jump" / "seed_0"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    mpc = {
                        "accepted_windows": 40,
                        "used_baseline_fallback": False,
                    }
                    if name != "missing":
                        mpc["accepted"] = accepted
                    (output_dir / "metrics.json").write_text(
                        json.dumps(
                            {
                                "metrics": {
                                    "num_steps": 800,
                                    "success": True,
                                    "score": -2.0,
                                },
                                "mpc": mpc,
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

                    with mock.patch.object(
                        runner.subprocess,
                        "run",
                        return_value=completed,
                    ):
                        row = runner.run_command(command)

                self.assertFalse(row["mpc_accepted"])

    def test_run_command_missing_or_invalid_mpc_fallback_fails_closed(self) -> None:
        runner = load_runner()
        invalid_values = {
            "missing": None,
            "none": None,
            "string": "false",
        }
        for name, fallback in invalid_values.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    output_dir = Path(tmp_dir) / "jump" / "seed_0"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    mpc = {
                        "accepted": True,
                        "accepted_windows": 40,
                    }
                    if name != "missing":
                        mpc["used_baseline_fallback"] = fallback
                    (output_dir / "metrics.json").write_text(
                        json.dumps(
                            {
                                "metrics": {
                                    "num_steps": 800,
                                    "success": True,
                                    "score": -2.0,
                                },
                                "mpc": mpc,
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

                    with mock.patch.object(
                        runner.subprocess,
                        "run",
                        return_value=completed,
                    ):
                        row = runner.run_command(command)

                self.assertIsNot(row["mpc_used_baseline_fallback"], False)

    def test_main_reuses_existing_ok_rows_when_requested(self) -> None:
        import torch

        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = root / "stage0"
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
                "--reuse-existing-ok",
            ]
            parsed = runner.parse_args(argv)
            for command in runner.build_stage0_commands(parsed):
                output_dir = Path(command.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "motion": command.motion,
                            "motion_type": "isaaclab",
                            "checkpoint": command.argv[command.argv.index("--checkpoint") + 1],
                            "device": "cuda:0",
                            "method": "g1_wbc_joint_global",
                            "max_steps": 800,
                            "metrics": _passing_metrics(),
                            "mpc": {
                                "mpc_backend": "mujoco_warp",
                                "mpc_optimizer": "legacy",
                                "reward_weight_source": str(reward_weights.resolve()),
                                "accepted": True,
                                "accepted_windows": 40,
                                "used_baseline_fallback": False,
                                "config": _stage0_mpc_config(command),
                                "steady_state_wall_time_sec": 120.0,
                                "runtime_visible_devices": ["0"],
                                "runtime_gpu_name": "NVIDIA H100 80GB HBM3",
                            },
                        }
                    )
                )
                (output_dir / "rollout.npz").write_text("{}")
                (output_dir / "mpc_command.npz").write_text("{}")
                runner.write_stage0_runner_provenance(command)

            with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
                with mock.patch.object(runner, "validate_visible_gpu_is_idle", return_value=()):
                    with mock.patch.object(
                        runner,
                        "run_command",
                        side_effect=AssertionError("run_command should not be called"),
                    ):
                        exit_code = runner.main(argv)

            manifest = json.loads((output_root / "baseline_manifest.json").read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual({row["status"] for row in manifest["rows"]}, {"ok"})
        self.assertTrue(all(row["reused_existing"] for row in manifest["rows"]))
        self.assertEqual(set(manifest["baseline_envelopes"]), {"jump", "walk"})
        self.assertEqual(manifest["promoted_seeds"], {"jump": 0, "walk": 0})

    def test_existing_row_reuse_rejects_missing_stage0_runner_provenance(self) -> None:
        import torch

        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_root = root / "stage0"
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
            args = runner.parse_args(
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
                    "--reuse-existing-ok",
                ]
            )
            command = next(
                command
                for command in runner.build_stage0_commands(args)
                if command.motion_name == "jump" and command.seed == 0
            )
            output_dir = Path(command.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "motion": command.motion,
                        "motion_type": "isaaclab",
                        "checkpoint": command.argv[command.argv.index("--checkpoint") + 1],
                        "device": "cuda:0",
                        "method": "g1_wbc_joint_global",
                        "max_steps": 800,
                        "metrics": _passing_metrics(),
                        "mpc": {
                            "mpc_backend": "mujoco_warp",
                            "mpc_optimizer": "legacy",
                            "reward_weight_source": str(reward_weights.resolve()),
                            "accepted": True,
                            "accepted_windows": 40,
                            "used_baseline_fallback": False,
                        },
                    }
                )
            )
            (output_dir / "rollout.npz").write_text("{}")
            (output_dir / "mpc_command.npz").write_text("{}")

            row = runner.load_existing_ok_row(command)

        self.assertIsNone(row)

    def test_existing_row_reuse_rejects_missing_or_mismatched_mpc_config(self) -> None:
        runner = load_runner()
        invalid_cases = ("missing", "mismatched")
        for name in invalid_cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    root = Path(tmp_dir)
                    args = runner.parse_args(
                        [
                            "--jump-motion",
                            str(root / "jump.npz"),
                            "--walk-motion",
                            str(root / "walk.npz"),
                            "--checkpoint",
                            str(root / "model.pt"),
                            "--reward-weights",
                            str(root / "reward.json"),
                            "--output-dir",
                            str(root / "stage0"),
                            "--reuse-existing-ok",
                        ]
                    )
                    command = next(
                        command
                        for command in runner.build_stage0_commands(args)
                        if command.motion_name == "jump" and command.seed == 0
                    )
                    output_dir = Path(command.output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    mpc = {
                        "mpc_backend": "mujoco_warp",
                        "mpc_optimizer": "legacy",
                        "reward_weight_source": str((root / "reward.json").resolve()),
                        "accepted": True,
                        "accepted_windows": 40,
                        "used_baseline_fallback": False,
                    }
                    if name == "mismatched":
                        mpc["config"] = {
                            **_stage0_mpc_config(command),
                            "num_samples": 256,
                        }
                    (output_dir / "metrics.json").write_text(
                        json.dumps(
                            {
                                "motion": command.motion,
                                "motion_type": "isaaclab",
                                "checkpoint": str(root / "model.pt"),
                                "device": "cuda:0",
                                "method": "g1_wbc_joint_global",
                                "max_steps": 800,
                                "metrics": _passing_metrics(),
                                "mpc": mpc,
                            }
                        )
                    )
                    (output_dir / "rollout.npz").write_text("{}")
                    (output_dir / "mpc_command.npz").write_text("{}")
                    runner.write_stage0_runner_provenance(command)

                    row = runner.load_existing_ok_row(command)

                self.assertIsNone(row)

    def test_existing_row_reuse_rejects_stale_stage0_runner_argv_provenance(self) -> None:
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
            args = runner.parse_args(
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
                    "--reuse-existing-ok",
                ]
            )
            command = next(
                command
                for command in runner.build_stage0_commands(args)
                if command.motion_name == "jump" and command.seed == 0
            )
            output_dir = Path(command.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "motion": command.motion,
                        "motion_type": "isaaclab",
                        "checkpoint": command.argv[command.argv.index("--checkpoint") + 1],
                        "device": "cuda:0",
                        "method": "g1_wbc_joint_global",
                        "max_steps": 800,
                        "metrics": _passing_metrics(),
                        "mpc": {
                            "mpc_backend": "mujoco_warp",
                            "mpc_optimizer": "legacy",
                            "reward_weight_source": str(reward_weights.resolve()),
                            "accepted": True,
                            "accepted_windows": 40,
                            "used_baseline_fallback": False,
                        },
                    }
                )
            )
            (output_dir / "rollout.npz").write_text("{}")
            (output_dir / "mpc_command.npz").write_text("{}")
            provenance_path = runner.write_stage0_runner_provenance(command)
            provenance = json.loads(provenance_path.read_text())
            samples_index = provenance["argv"].index("--mpc-samples") + 1
            provenance["argv"][samples_index] = "256"
            provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")

            row = runner.load_existing_ok_row(command)

        self.assertIsNone(row)

    def test_existing_row_reuse_rejects_mismatched_metrics_provenance(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            command = runner.Stage0Command(
                motion_name="jump",
                motion="/tmp/formal_jump.npz",
                seed=0,
                output_dir=str(output_dir),
                argv=[
                    "python",
                    "-m",
                    "spider.tasks.g1_wbc.evaluate",
                    "--motion",
                    "/tmp/formal_jump.npz",
                    "--checkpoint",
                    "/tmp/model.pt",
                    "--method",
                    "g1_wbc_joint_global",
                    "--mpc-backend",
                    "mujoco_warp",
                    "--mpc-optimizer",
                    "legacy",
                ],
                command_text="python -m spider.tasks.g1_wbc.evaluate",
            )
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "motion": "/tmp/stale_jump.npz",
                        "checkpoint": "/tmp/model.pt",
                        "method": "g1_wbc_joint_global",
                        "metrics": _passing_metrics(),
                        "mpc": {
                            "accepted": True,
                            "accepted_windows": 40,
                            "used_baseline_fallback": False,
                        },
                    }
                )
            )
            (output_dir / "rollout.npz").write_text("{}")
            (output_dir / "mpc_command.npz").write_text("{}")
            runner.write_stage0_runner_provenance(command)

            row = runner.load_existing_ok_row(command)

        self.assertIsNone(row)

    def test_existing_row_reuse_rejects_mismatched_reward_weights(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            formal_reward = output_dir / "formal_reward.json"
            stale_reward = output_dir / "stale_reward.json"
            formal_reward.write_text("{}")
            stale_reward.write_text("{}")
            command = runner.Stage0Command(
                motion_name="jump",
                motion="/tmp/formal_jump.npz",
                seed=0,
                output_dir=str(output_dir),
                argv=[
                    "python",
                    "-m",
                    "spider.tasks.g1_wbc.evaluate",
                    "--motion",
                    "/tmp/formal_jump.npz",
                    "--motion-type",
                    "isaaclab",
                    "--checkpoint",
                    "/tmp/model.pt",
                    "--device",
                    "cuda:0",
                    "--method",
                    "g1_wbc_joint_global",
                    "--max-steps",
                    "800",
                    "--mpc-backend",
                    "mujoco_warp",
                    "--mpc-optimizer",
                    "legacy",
                    "--mpc-reward-weights",
                    str(formal_reward),
                ],
                command_text="python -m spider.tasks.g1_wbc.evaluate",
            )
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "motion": "/tmp/formal_jump.npz",
                        "motion_type": "isaaclab",
                        "checkpoint": "/tmp/model.pt",
                        "device": "cuda:0",
                        "method": "g1_wbc_joint_global",
                        "max_steps": 800,
                        "metrics": _passing_metrics(),
                        "mpc": {
                            "mpc_backend": "mujoco_warp",
                            "mpc_optimizer": "legacy",
                            "reward_weight_source": str(stale_reward),
                            "accepted": True,
                            "accepted_windows": 40,
                            "used_baseline_fallback": False,
                        },
                    }
                )
            )
            (output_dir / "rollout.npz").write_text("{}")
            (output_dir / "mpc_command.npz").write_text("{}")
            runner.write_stage0_runner_provenance(command)

            row = runner.load_existing_ok_row(command)

        self.assertIsNone(row)

    def test_existing_row_reuse_rejects_missing_or_invalid_mpc_accepted(self) -> None:
        runner = load_runner()
        invalid_values = {
            "missing": None,
            "none": None,
            "string": "false",
        }
        for name, accepted in invalid_values.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    output_dir = Path(tmp_dir)
                    command = runner.Stage0Command(
                        motion_name="jump",
                        motion="/tmp/formal_jump.npz",
                        seed=0,
                        output_dir=str(output_dir),
                        argv=[
                            "python",
                            "-m",
                            "spider.tasks.g1_wbc.evaluate",
                            "--motion",
                            "/tmp/formal_jump.npz",
                            "--checkpoint",
                            "/tmp/model.pt",
                            "--method",
                            "g1_wbc_joint_global",
                            "--mpc-backend",
                            "mujoco_warp",
                            "--mpc-optimizer",
                            "legacy",
                        ],
                        command_text="python -m spider.tasks.g1_wbc.evaluate",
                    )
                    mpc = {
                        "mpc_backend": "mujoco_warp",
                        "mpc_optimizer": "legacy",
                        "accepted_windows": 40,
                        "used_baseline_fallback": False,
                    }
                    if name != "missing":
                        mpc["accepted"] = accepted
                    (output_dir / "metrics.json").write_text(
                        json.dumps(
                            {
                                "motion": "/tmp/formal_jump.npz",
                                "checkpoint": "/tmp/model.pt",
                                "method": "g1_wbc_joint_global",
                                "metrics": _passing_metrics(),
                                "mpc": mpc,
                            }
                        )
                    )
                    (output_dir / "rollout.npz").write_text("{}")
                    (output_dir / "mpc_command.npz").write_text("{}")
                    runner.write_stage0_runner_provenance(command)

                    row = runner.load_existing_ok_row(command)

                self.assertIsNone(row)

    def test_existing_row_reuse_rejects_missing_or_invalid_mpc_safety_fields(
        self,
    ) -> None:
        runner = load_runner()
        invalid_cases = (
            ("accepted_windows_missing", "accepted_windows", None),
            ("accepted_windows_num_windows_fallback", "accepted_windows", None),
            ("accepted_windows_string", "accepted_windows", "40"),
            ("fallback_missing", "used_baseline_fallback", None),
            ("fallback_none", "used_baseline_fallback", None),
            ("fallback_string", "used_baseline_fallback", "false"),
        )
        for name, field, value in invalid_cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    output_dir = Path(tmp_dir)
                    command = runner.Stage0Command(
                        motion_name="jump",
                        motion="/tmp/formal_jump.npz",
                        seed=0,
                        output_dir=str(output_dir),
                        argv=[
                            "python",
                            "-m",
                            "spider.tasks.g1_wbc.evaluate",
                            "--motion",
                            "/tmp/formal_jump.npz",
                            "--checkpoint",
                            "/tmp/model.pt",
                            "--method",
                            "g1_wbc_joint_global",
                            "--mpc-backend",
                            "mujoco_warp",
                            "--mpc-optimizer",
                            "legacy",
                        ],
                        command_text="python -m spider.tasks.g1_wbc.evaluate",
                    )
                    mpc = {
                        "mpc_backend": "mujoco_warp",
                        "mpc_optimizer": "legacy",
                        "accepted": True,
                        "accepted_windows": 40,
                        "used_baseline_fallback": False,
                    }
                    if field == "accepted_windows":
                        mpc.pop("accepted_windows")
                        if name == "accepted_windows_num_windows_fallback":
                            mpc["num_windows"] = 40
                        elif value is not None:
                            mpc["accepted_windows"] = value
                    else:
                        mpc.pop("used_baseline_fallback")
                        if name != "fallback_missing":
                            mpc["used_baseline_fallback"] = value
                    (output_dir / "metrics.json").write_text(
                        json.dumps(
                            {
                                "motion": "/tmp/formal_jump.npz",
                                "checkpoint": "/tmp/model.pt",
                                "method": "g1_wbc_joint_global",
                                "metrics": _passing_metrics(),
                                "mpc": mpc,
                            }
                        )
                    )
                    (output_dir / "rollout.npz").write_text("{}")
                    (output_dir / "mpc_command.npz").write_text("{}")
                    runner.write_stage0_runner_provenance(command)

                    row = runner.load_existing_ok_row(command)

                self.assertIsNone(row)

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
                    "metrics": _passing_metrics(),
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
                with mock.patch.object(runner, "validate_visible_gpu_is_idle", return_value=()):
                    with mock.patch.object(runner, "run_command", side_effect=fake_run_command):
                        exit_code = runner.main(argv)

            manifest = json.loads((output_root / "baseline_manifest.json").read_text())
            provenance_files_exist = [
                (Path(row["output_dir"]) / runner.RUNNER_PROVENANCE_FILENAME).is_file()
                for row in manifest["rows"]
            ]

        self.assertEqual(exit_code, 0)
        self.assertEqual({row["status"] for row in manifest["rows"]}, {"ok"})
        self.assertTrue(all(row["artifacts"]["metrics_json"] for row in manifest["rows"]))
        self.assertTrue(all(provenance_files_exist))
        self.assertEqual(
            {row["steady_state_wall_time_sec"] for row in manifest["rows"]},
            {120.0},
        )

    def test_main_writes_frozen_baseline_envelopes_for_passing_real_run(self) -> None:
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
                    "metrics": _passing_metrics(),
                    "mpc_accepted": True,
                    "accepted_windows": 40,
                    "mpc_used_baseline_fallback": False,
                    "num_steps": 800,
                    "steady_state_wall_time_sec": 120.0,
                    "runtime_visible_devices": ["0"],
                    "runtime_gpu_name": "NVIDIA H100 80GB HBM3",
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
                with mock.patch.object(runner, "validate_visible_gpu_is_idle", return_value=()):
                    with mock.patch.object(runner, "run_command", side_effect=fake_run_command):
                        exit_code = runner.main(argv)

            manifest = json.loads((output_root / "baseline_manifest.json").read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(set(manifest["baseline_envelopes"]), {"jump", "walk"})
        self.assertEqual(manifest["baseline_envelopes"]["jump"]["success"]["count"], 3.0)
        self.assertEqual(manifest["baseline_envelopes"]["walk"]["score"]["mean"], -1.0)
        self.assertEqual(manifest["promoted_seeds"], {"jump": 0, "walk": 0})

    def test_main_returns_nonzero_when_baseline_gate_fails(self) -> None:
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
                metrics = _passing_metrics()
                metrics["success"] = False
                metrics["score"] = -99.0
                return {
                    "returncode": 0,
                    "stdout": "{}",
                    "stderr": "",
                    "metrics": metrics,
                    "mpc_accepted": True,
                    "accepted_windows": 40,
                    "mpc_used_baseline_fallback": False,
                    "num_steps": 800,
                    "steady_state_wall_time_sec": 120.0,
                    "runtime_visible_devices": ["0"],
                    "runtime_gpu_name": "NVIDIA H100 80GB HBM3",
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
                with mock.patch.object(runner, "validate_visible_gpu_is_idle", return_value=()):
                    with mock.patch.object(runner, "run_command", side_effect=fake_run_command):
                        exit_code = runner.main(argv)

            manifest = json.loads((output_root / "baseline_manifest.json").read_text())

        self.assertEqual(exit_code, 1)
        self.assertIn("baseline_gate_failures", manifest)
        self.assertNotIn("baseline_envelopes", manifest)

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

    def test_main_skip_manifest_does_not_write_partial_shard_manifest(self) -> None:
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
                    "--skip-manifest",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "2",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertFalse((output_root / "baseline_manifest.json").exists())


def _passing_metrics() -> dict[str, float | bool]:
    return {
        "num_steps": 800,
        "success": True,
        "score": -1.0,
        "root_pos_error_mean": 0.01,
        "body_global_pos_error_mean": 0.01,
        "ee_global_pos_error_mean": 0.01,
        "ee_local_pos_error_mean": 0.01,
        "contact_mismatch_rate": 0.01,
        "contact_false_positive_rate": 0.0,
        "contact_false_negative_rate": 0.0,
        "bad_floor_contact_rate": 0.0,
        "control_delta_mean": 0.01,
        "joint_acc_mean": 1.0,
        "joint_jerk_mean": 1.0,
    }


def _stage0_mpc_config(command) -> dict[str, float | int | str | bool | None]:
    return {
        "mode": "g1_wbc_joint_global",
        "num_samples": 512,
        "num_iterations": 2,
        "planning_horizon_steps": 40,
        "control_steps": 20,
        "sampling_mode": "knot",
        "knot_count": 8,
        "elite_frac": 0.125,
        "temperature": 0.7,
        "root_pos_sigma": 0.04,
        "root_rot_sigma": 0.10,
        "joint_sigma": 0.18,
        "min_root_pos_sigma": 0.002,
        "min_root_rot_sigma": 0.004,
        "min_joint_sigma": 0.008,
        "sigma_decay": 0.75,
        "smooth_passes": 0,
        "command_reg_weight": 0.0,
        "command_smooth_weight": 0.0,
        "use_guided_candidate": True,
        "guided_root_pos_gain": 0.50,
        "guided_root_rot_gain": 0.50,
        "guided_joint_gain": 0.50,
        "guided_root_pos_clip": 0.05,
        "guided_root_rot_clip": 0.12,
        "guided_joint_clip": 0.35,
        "acceptance_gate": True,
        "seed": int(command.seed),
        "freeze_first_frame": True,
        "use_warm_start": False,
        "warm_start_source": "best",
        "warm_start_decay": 1.0,
    }


if __name__ == "__main__":
    unittest.main()
