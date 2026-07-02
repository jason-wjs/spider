from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RUNNER_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "run_g1_wbc_mjx_acceptance.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("mjx_acceptance_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _baseline_manifest(tmp_path: Path) -> Path:
    rows = []
    for motion in ("jump", "walk"):
        for seed in (0, 1, 2):
            output_dir = tmp_path / "baseline" / motion / f"seed_{seed}"
            rows.append(
                {
                    "motion_name": motion,
                    "motion": f"/tmp/{motion}.npz",
                    "seed": seed,
                    "output_dir": str(output_dir),
                    "argv": [
                        "python",
                        "-m",
                        "spider.tasks.g1_wbc.evaluate",
                        "--motion",
                        f"/tmp/{motion}.npz",
                        "--motion-type",
                        "isaaclab",
                        "--checkpoint",
                        "model.pt",
                        "--device",
                        "cuda:0",
                        "--output-dir",
                        str(output_dir),
                        "--seed",
                        str(seed),
                        "--method",
                        "g1_wbc_joint_global",
                        "--mpc-backend",
                        "mujoco_warp",
                        "--max-steps",
                        "800",
                        "--mpc-control-steps",
                        "20",
                        "--save-rollout",
                    ],
                    "status": "ok",
                    "returncode": 0,
                    "metrics": _metrics(success=True),
                    "mpc_accepted": True,
                    "accepted_windows": 40,
                    "mpc_used_baseline_fallback": False,
                    "num_steps": 800,
                    "artifacts": {
                        "metrics_json": str(output_dir / "metrics.json"),
                        "rollout_npz": str(output_dir / "rollout.npz"),
                        "mpc_command_npz": str(output_dir / "mpc_command.npz"),
                    },
                    "steady_state_wall_time_sec": 120.0,
                }
            )
    manifest = {
        "schema_version": 1,
        "baseline_name": "test",
        "motions": ["jump", "walk"],
        "seeds": [0, 1, 2],
        "rows": rows,
    }
    path = tmp_path / "baseline_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _metrics(*, success: bool) -> dict[str, float | bool]:
    return {
        "num_steps": 800,
        "success": success,
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


class MjxAcceptanceRunnerTest(unittest.TestCase):
    def test_build_acceptance_plan_covers_mjx_and_replay_backends(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                    "--device",
                    "cuda:0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        self.assertEqual(len(plan), 6)
        self.assertEqual({item.backend for item in plan}, {"mjx"})
        self.assertEqual({item.replay_backend for item in plan}, {"mujoco_warp"})
        self.assertEqual({item.motion for item in plan}, {"jump", "walk"})
        self.assertEqual({item.seed for item in plan}, {0, 1, 2})
        for item in plan:
            with self.subTest(motion=item.motion, seed=item.seed):
                backend_idx = item.mjx_argv.index("--mpc-backend")
                self.assertEqual(item.mjx_argv[backend_idx + 1], "mjx")
                replay_backend_idx = item.replay_argv.index("--mpc-backend")
                self.assertEqual(item.replay_argv[replay_backend_idx + 1], "mujoco_warp")
                method_idx = item.replay_argv.index("--method")
                self.assertEqual(item.replay_argv[method_idx + 1], "replay_command")

    def test_main_returns_one_when_any_replay_fails(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                output.mkdir(parents=True, exist_ok=True)
                is_replay = "replay_command" in argv
                return {
                    "returncode": 2 if is_replay else 0,
                    "stdout": "",
                    "stderr": "replay failed" if is_replay else "",
                    "status": "failed" if is_replay else "ok",
                    "metrics": _metrics(success=not is_replay),
                    "mpc_accepted": not is_replay,
                    "accepted_windows": 40,
                    "mpc_used_baseline_fallback": False,
                    "num_steps": 800,
                    "steady_state_wall_time_sec": 1.0,
                }

            with mock.patch.object(runner, "run_command", side_effect=fake_run_command):
                exit_code = runner.main(
                    [
                        "--baseline-manifest",
                        str(manifest_path),
                        "--output-dir",
                        str(output_dir),
                        "--device",
                        "cuda:0",
                    ]
                )

            report = json.loads((output_dir / "acceptance_report.json").read_text())

        self.assertEqual(exit_code, 1)
        self.assertFalse(report["passed"])
        self.assertFalse(report["replay_results"]["jump"]["passed"])
        self.assertFalse(report["replay_results"]["walk"]["passed"])

    def test_dry_run_writes_plan_but_fails_gates(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            exit_code = runner.main(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cuda:0",
                    "--dry-run",
                ]
            )

            report = json.loads((output_dir / "acceptance_report.json").read_text())

        self.assertEqual(exit_code, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["planned_runs"]), 6)
        self.assertFalse(report["motion_results"]["jump"]["mjx_passed"])

    def test_missing_mjx_timing_or_artifacts_fail_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                is_replay = "replay_command" in argv
                return {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "mpc_accepted": True,
                    "accepted_windows": 40,
                    "mpc_used_baseline_fallback": False,
                    "num_steps": 800,
                    "steady_state_wall_time_sec": 1.0 if is_replay else None,
                }

            with mock.patch.object(runner, "run_command", side_effect=fake_run_command):
                exit_code = runner.main(
                    [
                        "--baseline-manifest",
                        str(manifest_path),
                        "--output-dir",
                        str(output_dir),
                        "--device",
                        "cuda:0",
                    ]
                )

            report = json.loads((output_dir / "acceptance_report.json").read_text())

        self.assertEqual(exit_code, 1)
        self.assertFalse(report["passed"])
        self.assertIn("mjx_steady_state_wall_time", report["speed_results"]["jump"]["failures"])
        self.assertIn("metrics_json", report["motion_results"]["jump"]["mjx_failures"])
        self.assertIn("rollout_npz", report["motion_results"]["jump"]["mjx_failures"])
        self.assertIn("mpc_command_npz", report["motion_results"]["jump"]["mjx_failures"])


if __name__ == "__main__":
    unittest.main()
