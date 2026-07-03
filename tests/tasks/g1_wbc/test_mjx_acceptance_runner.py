from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np

RUNNER_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "run_g1_wbc_mjx_acceptance.py"
)


def load_runner(*, assume_idle_gpu: bool = True):
    spec = importlib.util.spec_from_file_location("mjx_acceptance_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if assume_idle_gpu:
        module._visible_gpu_has_compute_processes = lambda visible_gpu: False
    return module


def _baseline_manifest(tmp_path: Path) -> Path:
    rows = []
    checkpoint = tmp_path / "model.pt"
    reward_weights = tmp_path / "reward.json"
    checkpoint.write_text("checkpoint")
    reward_weights.write_text("{}")
    for motion in ("jump", "walk"):
        motion_path = tmp_path / f"{motion}.npz"
        motion_path.write_text("motion")
        for seed in (0, 1, 2):
            output_dir = tmp_path / "baseline" / motion / f"seed_{seed}"
            _write_artifacts(output_dir)
            metrics = _metrics(success=True)
            (output_dir / "metrics.json").write_text(
                json.dumps(
                    _baseline_metrics_payload(
                        motion=str(motion_path),
                        checkpoint=str(checkpoint),
                        metrics=metrics,
                        seed=seed,
                    )
                )
            )
            artifact_mtime_ns = {
                "metrics_json": (output_dir / "metrics.json").stat().st_mtime_ns,
                "rollout_npz": (output_dir / "rollout.npz").stat().st_mtime_ns,
                "mpc_command_npz": (output_dir / "mpc_command.npz").stat().st_mtime_ns,
            }
            rows.append(
                {
                    "motion_name": motion,
                    "motion": str(motion_path),
                    "seed": seed,
                    "output_dir": str(output_dir),
                    "argv": [
                        "python",
                        "-m",
                        "spider.tasks.g1_wbc.evaluate",
                        "--motion",
                        str(motion_path),
                        "--motion-type",
                        "isaaclab",
                        "--checkpoint",
                        str(checkpoint),
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
                        "--mpc-optimizer",
                        "legacy",
                        "--mpc-preset",
                        "aggressive",
                        "--max-steps",
                        "800",
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
                        "--mpc-elite-frac",
                        "0.125",
                        "--mpc-temperature",
                        "0.7",
                        "--mpc-root-pos-sigma",
                        "0.04",
                        "--mpc-root-rot-sigma",
                        "0.10",
                        "--mpc-joint-sigma",
                        "0.18",
                        "--mpc-sigma-decay",
                        "0.75",
                        "--mpc-smooth-passes",
                        "0",
                        "--mpc-command-reg-weight",
                        "0.0",
                        "--mpc-command-smooth-weight",
                        "0.0",
                        "--mpc-guided-root-pos-gain",
                        "0.50",
                        "--mpc-guided-root-rot-gain",
                        "0.50",
                        "--mpc-guided-joint-gain",
                        "0.50",
                        "--mpc-guided-root-pos-clip",
                        "0.05",
                        "--mpc-guided-root-rot-clip",
                        "0.12",
                        "--mpc-guided-joint-clip",
                        "0.35",
                        "--mpc-guided-candidate",
                        "--mpc-acceptance-gate",
                        "--no-mpc-warm-start",
                        "--mpc-warm-start-source",
                        "best",
                        "--mpc-warm-start-decay",
                        "1.0",
                        "--nconmax-per-env",
                        "512",
                        "--njmax-per-env",
                        "2048",
                        "--save-rollout",
                        "--mpc-reward-weights",
                        str(reward_weights),
                    ],
                    "status": "ok",
                    "returncode": 0,
                    "metrics": metrics,
                    "mpc_accepted": True,
                    "accepted_windows": 40,
                    "mpc_used_baseline_fallback": False,
                    "num_steps": 800,
                    "runtime_visible_devices": ["0"],
                    "runtime_gpu_name": "NVIDIA H100 80GB HBM3",
                    "command_start_time_ns": min(artifact_mtime_ns.values()) - 1_000_000,
                    "artifacts": {
                        "metrics_json": str(output_dir / "metrics.json"),
                        "rollout_npz": str(output_dir / "rollout.npz"),
                        "mpc_command_npz": str(output_dir / "mpc_command.npz"),
                    },
                    "artifact_mtime_ns": artifact_mtime_ns,
                    "artifact_sha256": {
                        "metrics_json": _file_sha256(output_dir / "metrics.json"),
                        "rollout_npz": _file_sha256(output_dir / "rollout.npz"),
                        "mpc_command_npz": _file_sha256(output_dir / "mpc_command.npz"),
                    },
                    "steady_state_wall_time_sec": 120.0,
                }
            )
    manifest = {
        "schema_version": 1,
        "baseline_name": "g1_wbc_stage0_mujoco_warp_sweetpoint",
        "motions": ["jump", "walk"],
        "seeds": [0, 1, 2],
        "provenance": {
            "worktree_path": str(Path(__file__).resolve().parents[3]),
            "git_commit": "0123456789abcdef",
            "git_status_short": "",
        },
        "input_sha256": {
            "jump_motion": _file_sha256(tmp_path / "jump.npz"),
            "walk_motion": _file_sha256(tmp_path / "walk.npz"),
            "checkpoint": _file_sha256(checkpoint),
            "reward_weights": _file_sha256(reward_weights),
        },
        "baseline_envelopes": _baseline_envelopes(),
        "promoted_seeds": {"jump": 0, "walk": 0},
        "rows": rows,
    }
    path = tmp_path / "baseline_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def _baseline_envelopes() -> dict[str, dict[str, dict[str, float]]]:
    metric_stats = {
        "score": {"mean": -1.0, "std": 0.0, "min": -1.0, "max": -1.0, "median": -1.0},
        "root_pos_error_mean": {
            "mean": 0.01,
            "std": 0.0,
            "min": 0.01,
            "max": 0.01,
            "median": 0.01,
        },
        "body_global_pos_error_mean": {
            "mean": 0.01,
            "std": 0.0,
            "min": 0.01,
            "max": 0.01,
            "median": 0.01,
        },
        "ee_global_pos_error_mean": {
            "mean": 0.01,
            "std": 0.0,
            "min": 0.01,
            "max": 0.01,
            "median": 0.01,
        },
        "ee_local_pos_error_mean": {
            "mean": 0.01,
            "std": 0.0,
            "min": 0.01,
            "max": 0.01,
            "median": 0.01,
        },
        "contact_mismatch_rate": {
            "mean": 0.01,
            "std": 0.0,
            "min": 0.01,
            "max": 0.01,
            "median": 0.01,
        },
        "contact_false_positive_rate": {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        },
        "contact_false_negative_rate": {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        },
        "bad_floor_contact_rate": {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        },
        "control_delta_mean": {
            "mean": 0.01,
            "std": 0.0,
            "min": 0.01,
            "max": 0.01,
            "median": 0.01,
        },
        "joint_acc_mean": {
            "mean": 1.0,
            "std": 0.0,
            "min": 1.0,
            "max": 1.0,
            "median": 1.0,
        },
        "joint_jerk_mean": {
            "mean": 1.0,
            "std": 0.0,
            "min": 1.0,
            "max": 1.0,
            "median": 1.0,
        },
    }
    envelope = {
        "success": {
            "mean": 1.0,
            "std": 0.0,
            "min": 1.0,
            "max": 1.0,
            "median": 1.0,
            "count": 3.0,
        },
        **metric_stats,
    }
    return {"jump": dict(envelope), "walk": dict(envelope)}


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


def _baseline_metrics_payload(
    *,
    motion: str,
    checkpoint: str,
    metrics: dict[str, float | bool],
    accepted: bool = True,
    seed: int = 0,
) -> dict[str, object]:
    return {
        "method": "g1_wbc_joint_global",
        "motion": motion,
        "device": "cuda:0",
        "checkpoint": checkpoint,
        "max_steps": 800,
        "metrics": metrics,
        "mpc": {
            "mpc_backend": "mujoco_warp",
            "mpc_optimizer": "legacy",
            "accepted": accepted,
            "accepted_windows": 40,
            "num_windows": 40,
            "used_baseline_fallback": False,
            "config": _baseline_mpc_config(seed),
            "steady_state_wall_time_sec": 120.0,
            "runtime_visible_devices": ["0"],
            "runtime_gpu_name": "NVIDIA H100 80GB HBM3",
        },
    }


def _baseline_mpc_config(seed: int) -> dict[str, float | int | str | bool | None]:
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
        "seed": int(seed),
        "freeze_first_frame": True,
        "use_warm_start": False,
        "warm_start_source": "best",
        "warm_start_decay": 1.0,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _rewrite_baseline_gpu(manifest_path: Path, gpu_name: str) -> None:
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["rows"]:
        row["runtime_gpu_name"] = gpu_name
        row["runtime_visible_devices"] = ["0"]
    manifest_path.write_text(json.dumps(manifest))


class MjxAcceptanceRunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._env_patcher = mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"})
        self._env_patcher.start()
        self.addCleanup(self._env_patcher.stop)

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
                optimizer_idx = item.mjx_argv.index("--mpc-optimizer")
                self.assertEqual(item.mjx_argv[optimizer_idx + 1], "generic")
                self.assertIn("--mjx-enable-scan", item.mjx_argv)
                for flag in (
                    "--mpc-preset",
                    "--mpc-sampling-mode",
                    "--mpc-elite-frac",
                    "--mpc-sigma-decay",
                    "--mpc-smooth-passes",
                    "--mpc-command-reg-weight",
                    "--mpc-command-smooth-weight",
                    "--mpc-guided-candidate",
                    "--mpc-acceptance-gate",
                    "--no-mpc-warm-start",
                    "--mpc-warm-start-source",
                    "--mpc-warm-start-decay",
                ):
                    self.assertNotIn(flag, item.mjx_argv)
                    self.assertNotIn(flag, item.replay_argv)
                replay_backend_idx = item.replay_argv.index("--mpc-backend")
                self.assertEqual(item.replay_argv[replay_backend_idx + 1], "mujoco_warp")
                self.assertNotIn("--mjx-enable-scan", item.replay_argv)
                method_idx = item.replay_argv.index("--method")
                self.assertEqual(item.replay_argv[method_idx + 1], "replay_command")

    def test_build_acceptance_plan_rejects_missing_or_duplicate_matrix_rows(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            manifest["rows"] = [
                row
                for row in manifest["rows"]
                if not (row["motion_name"] == "walk" and row["seed"] == 2)
            ]
            duplicate = dict(manifest["rows"][0])
            duplicate["output_dir"] = str(root / "duplicate")
            manifest["rows"].append(duplicate)
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "run matrix"):
                runner.build_acceptance_plan(args, manifest)

    def test_build_acceptance_plan_rejects_missing_manifest_provenance(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            manifest.pop("provenance")
            manifest.pop("input_sha256")
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "provenance"):
                runner.build_acceptance_plan(args, manifest)

    def test_formal_baseline_manifest_requires_legacy_optimizer(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            for row in manifest["rows"]:
                argv = row["argv"]
                optimizer_idx = argv.index("--mpc-optimizer")
                del argv[optimizer_idx:optimizer_idx + 2]
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "--mpc-optimizer"):
                runner.build_acceptance_plan(args, manifest)

    def test_formal_baseline_manifest_requires_guided_stage0_settings(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            for row in manifest["rows"]:
                argv = row["argv"]
                gain_idx = argv.index("--mpc-guided-root-pos-gain")
                del argv[gain_idx:gain_idx + 2]
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "--mpc-guided-root-pos-gain"):
                runner.build_acceptance_plan(args, manifest)

    def test_build_acceptance_plan_rejects_missing_manifest_input_hashes(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            manifest.pop("input_sha256")
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "manifest_input_sha256"):
                runner.build_acceptance_plan(args, manifest)

    def test_build_acceptance_plan_rejects_missing_frozen_baseline_envelopes(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            manifest.pop("baseline_envelopes")
            manifest.pop("promoted_seeds")
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "baseline_envelopes"):
                runner.build_acceptance_plan(args, manifest)

    def test_build_acceptance_plan_rejects_mismatched_frozen_baseline_envelope(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            manifest["baseline_envelopes"]["jump"]["score"]["mean"] = -9.0
            manifest["promoted_seeds"]["walk"] = 2
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "baseline_envelopes"):
                runner.build_acceptance_plan(args, manifest)

    def test_build_acceptance_plan_rejects_noncanonical_manifest_input_hashes(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            manifest["input_sha256"]["checkpoint"] = "A" * 64
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "manifest_input_sha256"):
                runner.build_acceptance_plan(args, manifest)

    def test_build_acceptance_plan_rejects_manifest_input_hash_mismatch(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            manifest["input_sha256"]["checkpoint"] = "0" * 64
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "manifest_input_sha256"):
                runner.build_acceptance_plan(args, manifest)

    def test_build_acceptance_plan_rejects_extra_matrix_rows(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            extra = dict(manifest["rows"][0])
            extra["motion_name"] = "turn"
            extra["seed"] = 0
            manifest["rows"].append(extra)
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "run matrix"):
                runner.build_acceptance_plan(args, manifest)

    def test_build_acceptance_plan_rejects_non_sweetpoint_baseline_argv(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            argv = manifest["rows"][0]["argv"]
            argv[argv.index("--mpc-samples") + 1] = "128"
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "formal Stage 0 sweetpoint"):
                runner.build_acceptance_plan(args, manifest)

    def test_build_acceptance_plan_rejects_baseline_metrics_artifact_mismatch(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            row = manifest["rows"][0]
            metrics_path = Path(row["artifacts"]["metrics_json"])
            bad_metrics = dict(row["metrics"])
            bad_metrics["success"] = False
            metrics_path.write_text(
                json.dumps(
                    _baseline_metrics_payload(
                        motion=row["motion"],
                        checkpoint=row["argv"][row["argv"].index("--checkpoint") + 1],
                        metrics=bad_metrics,
                        accepted=False,
                    )
                )
            )
            row["artifact_mtime_ns"]["metrics_json"] = metrics_path.stat().st_mtime_ns
            row["artifact_sha256"]["metrics_json"] = _file_sha256(metrics_path)
            manifest_path.write_text(json.dumps(manifest))
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "baseline_metrics_artifact"):
                runner.build_acceptance_plan(args, manifest)

    def test_build_acceptance_plan_rejects_missing_or_mismatched_baseline_mpc_config(
        self,
    ) -> None:
        runner = load_runner()
        invalid_cases = ("missing", "mismatched")
        for name in invalid_cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    root = Path(tmp_dir)
                    manifest_path = _baseline_manifest(root)
                    manifest = json.loads(manifest_path.read_text())
                    row = manifest["rows"][0]
                    metrics_path = Path(row["artifacts"]["metrics_json"])
                    payload = json.loads(metrics_path.read_text())
                    if name == "missing":
                        payload["mpc"].pop("config")
                    else:
                        payload["mpc"]["config"]["num_samples"] = 256
                    metrics_path.write_text(json.dumps(payload))
                    row["artifact_mtime_ns"]["metrics_json"] = (
                        metrics_path.stat().st_mtime_ns
                    )
                    row["artifact_sha256"]["metrics_json"] = _file_sha256(metrics_path)
                    manifest_path.write_text(json.dumps(manifest))
                    args = runner.parse_args(
                        [
                            "--baseline-manifest",
                            str(manifest_path),
                            "--output-dir",
                            str(root / "acceptance"),
                        ]
                    )

                    with self.assertRaisesRegex(
                        ValueError,
                        "baseline_metrics_artifact",
                    ):
                        runner.build_acceptance_plan(args, manifest)

    def test_run_command_records_wall_time_for_replay_gate(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            output_dir = root / "run"
            argv = ["python", "--output-dir", str(output_dir)]

            def fake_run(argv, *, cwd, capture_output, text, check):
                del argv, cwd, capture_output, text, check
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with mock.patch.object(runner.subprocess, "run", side_effect=fake_run):
                row = runner.run_command(argv, cwd=root)

        self.assertEqual(row["returncode"], 0)
        self.assertIsInstance(row["command_wall_time_sec"], float)
        self.assertGreaterEqual(row["command_wall_time_sec"], 0.0)

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
        self.assertEqual(report["classification"], "invalid_benchmark")

    def test_replay_quality_regression_classifies_as_parity_failure(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                _write_artifacts(output, include_command=not is_replay)
                metrics = _metrics(success=True)
                if is_replay:
                    metrics["root_pos_error_mean"] = 1.0
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": metrics,
                    "num_steps": 800,
                    **_fresh_metrics_provenance(argv, is_replay=is_replay),
                }
                if is_replay:
                    row.update(_replay_evidence(argv))
                else:
                    row.update(
                        {
                            "mpc_accepted": True,
                            "accepted_windows": 40,
                            "mpc_used_baseline_fallback": False,
                            "compile_init_wall_time_sec": 2.0,
                            "jit_warmup_enabled": True,
                            "jit_warmup_wall_time_sec": 1.5,
                            "runtime_visible_devices": ("0",),
                            "steady_state_wall_time_sec": 1.0,
                            **_mjx_contact_evidence(),
                        }
                    )
                return row

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
        self.assertEqual(report["classification"], "parity_failure")
        self.assertIn("root_pos_error_mean", report["replay_results"]["jump"]["failures"])

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

    def test_main_reuses_existing_ok_rows_when_requested(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cuda:0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())
            plan = runner.build_acceptance_plan(args, manifest)
            _write_reusable_acceptance_outputs(runner, plan, output_dir=output_dir)

            with mock.patch.object(
                runner,
                "run_command",
                side_effect=AssertionError("run_command should not be called"),
            ):
                exit_code = runner.main(
                    [
                        "--baseline-manifest",
                        str(manifest_path),
                        "--output-dir",
                        str(output_dir),
                        "--device",
                        "cuda:0",
                        "--reuse-existing-ok",
                    ]
                )

            report = json.loads((output_dir / "acceptance_report.json").read_text())

        self.assertEqual(exit_code, 0)
        self.assertTrue(report["passed"])
        self.assertTrue(all(row["reused_existing"] for row in report["mjx_rows"]))
        self.assertTrue(all(row["reused_existing"] for row in report["replay_rows"]))

    def test_existing_acceptance_row_reuse_rejects_mismatched_provenance(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cuda:0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())
            plan = runner.build_acceptance_plan(args, manifest)
            mjx_rows, _replay_rows = _write_reusable_acceptance_outputs(
                runner,
                plan,
                output_dir=output_dir,
            )
            planned = plan[0]
            (Path(planned.output_dir) / "metrics.json").write_text(
                json.dumps(
                    _mjx_metrics_payload(planned, motion="/tmp/stale_motion.npz")
                )
            )

            row = runner.load_existing_acceptance_row(
                planned,
                kind="mjx",
                existing_rows=mjx_rows,
            )

        self.assertIsNone(row)

    def test_existing_acceptance_row_reuse_checks_later_duplicate_candidates(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cuda:0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())
            plan = runner.build_acceptance_plan(args, manifest)
            _write_reusable_acceptance_outputs(runner, plan, output_dir=output_dir)
            partial_path = output_dir / "acceptance_report.partial.json"
            partial = json.loads(partial_path.read_text())
            bad_duplicate = json.loads(json.dumps(partial["mjx_rows"][0]))
            bad_duplicate["mjx_argv"] = ["python", "--stale"]
            partial["mjx_rows"].insert(0, bad_duplicate)
            partial_path.write_text(json.dumps(partial))

            existing_rows = runner.load_existing_acceptance_rows(output_dir)
            row = runner.load_existing_acceptance_row(
                plan[0],
                kind="mjx",
                existing_rows=existing_rows["mjx"],
            )

        self.assertIsNotNone(row)
        self.assertTrue(row["reused_existing"])

    def test_existing_acceptance_row_reuse_rejects_malformed_metrics_shape(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cuda:0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())
            plan = runner.build_acceptance_plan(args, manifest)
            mjx_rows, _replay_rows = _write_reusable_acceptance_outputs(
                runner,
                plan,
                output_dir=output_dir,
            )
            planned = plan[0]
            metrics_path = Path(planned.output_dir) / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "method": "g1_wbc_joint_global",
                        "motion": planned.mjx_argv[
                            planned.mjx_argv.index("--motion") + 1
                        ],
                        "device": "cuda:0",
                        "checkpoint": planned.mjx_argv[
                            planned.mjx_argv.index("--checkpoint") + 1
                        ],
                        "max_steps": 800,
                        "metrics": [],
                        "mpc": [],
                    }
                )
            )
            mjx_rows[0]["artifact_sha256"]["metrics_json"] = _file_sha256(metrics_path)

            row = runner.load_existing_acceptance_row(
                planned,
                kind="mjx",
                existing_rows=mjx_rows,
            )

        self.assertIsNone(row)

    def test_existing_acceptance_row_reuse_rejects_malformed_mjx_npz_schema(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cuda:0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())
            plan = runner.build_acceptance_plan(args, manifest)
            mjx_rows, _replay_rows = _write_reusable_acceptance_outputs(
                runner,
                plan,
                output_dir=output_dir,
            )
            planned = plan[0]
            rollout_path = Path(planned.output_dir) / "rollout.npz"
            np.savez_compressed(rollout_path, qpos=np.zeros((801, 1, 36), dtype=np.float32))
            mjx_rows[0]["artifact_sha256"]["rollout_npz"] = _file_sha256(rollout_path)
            mjx_rows[0]["artifact_mtime_ns"]["rollout_npz"] = rollout_path.stat().st_mtime_ns

            row = runner.load_existing_acceptance_row(
                planned,
                kind="mjx",
                existing_rows=mjx_rows,
            )

        self.assertIsNone(row)

    def test_existing_acceptance_row_reuse_rejects_malformed_replay_npz_schema(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cuda:0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())
            plan = runner.build_acceptance_plan(args, manifest)
            _mjx_rows, replay_rows = _write_reusable_acceptance_outputs(
                runner,
                plan,
                output_dir=output_dir,
            )
            planned = plan[0]
            rollout_path = Path(planned.replay_output_dir) / "rollout.npz"
            np.savez_compressed(rollout_path, qpos=np.zeros((801, 1, 36), dtype=np.float32))
            replay_rows[0]["artifact_sha256"]["rollout_npz"] = _file_sha256(rollout_path)
            replay_rows[0]["artifact_mtime_ns"]["rollout_npz"] = rollout_path.stat().st_mtime_ns

            row = runner.load_existing_acceptance_row(
                planned,
                kind="replay",
                existing_rows=replay_rows,
            )

        self.assertIsNone(row)

    def test_existing_acceptance_rows_ignore_unpaired_partial_rows(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cuda:0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())
            plan = runner.build_acceptance_plan(args, manifest)
            _write_reusable_acceptance_outputs(runner, plan, output_dir=output_dir)
            partial_path = output_dir / "acceptance_report.partial.json"
            partial = json.loads(partial_path.read_text())
            partial["replay_rows"] = []
            partial_path.write_text(json.dumps(partial))

            existing_rows = runner.load_existing_acceptance_rows(output_dir)

        self.assertEqual(existing_rows["mjx"], [])
        self.assertEqual(existing_rows["replay"], [])

    def test_main_fails_fast_when_real_cuda_run_has_multiple_visible_gpus(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,1"}):
                with mock.patch.object(
                    runner,
                    "run_command",
                    side_effect=AssertionError("run_command should not be called"),
                ):
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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn("single_gpu_visibility", report["environment_failures"])

    def test_main_fails_fast_when_visible_gpu_has_compute_processes(self) -> None:
        runner = load_runner(assume_idle_gpu=False)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"
            busy_gpu = runner.subprocess.CompletedProcess(
                args=["nvidia-smi"],
                returncode=0,
                stdout="12345, python, 13269 MiB\n",
                stderr="",
            )

            with mock.patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2"}):
                with mock.patch.object(runner.subprocess, "run", return_value=busy_gpu):
                    with mock.patch.object(
                        runner,
                        "run_command",
                        side_effect=AssertionError("run_command should not be called"),
                    ):
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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn("gpu_contention", report["environment_failures"])

    def test_report_quality_gate_uses_frozen_baseline_envelope(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            frozen = _baseline_envelopes()
            frozen["jump"]["root_pos_error_mean"] = {
                "mean": 0.005,
                "std": 0.0,
                "min": 0.005,
                "max": 0.005,
                "median": 0.005,
            }
            mjx_rows = []
            replay_rows = []
            for motion in ("jump", "walk"):
                for seed in (0, 1, 2):
                    output_dir = root / "acceptance" / motion / f"seed_{seed}" / "mjx"
                    replay_output_dir = root / "acceptance" / motion / f"seed_{seed}" / "replay"
                    _write_artifacts(output_dir)
                    _write_artifacts(replay_output_dir, include_command=False)
                    metrics = _metrics(success=True)
                    if motion == "jump":
                        metrics["root_pos_error_mean"] = 0.006
                    mjx_rows.append(
                        {
                            "motion": motion,
                            "seed": seed,
                            "status": "ok",
                            "returncode": 0,
                            "metrics": metrics,
                            "mpc_accepted": True,
                            "accepted_windows": 40,
                            "mpc_used_baseline_fallback": False,
                            "num_steps": 800,
                            "compile_init_wall_time_sec": 2.0,
                            "jit_warmup_enabled": True,
                            "jit_warmup_wall_time_sec": 1.5,
                            "runtime_visible_devices": ("0",),
                            "steady_state_wall_time_sec": 1.0,
                            "artifacts": {
                                "metrics_json": str(output_dir / "metrics.json"),
                                "rollout_npz": str(output_dir / "rollout.npz"),
                                "mpc_command_npz": str(output_dir / "mpc_command.npz"),
                            },
                            **_mjx_contact_evidence(),
                        }
                    )
                    replay_rows.append(
                        {
                            "motion": motion,
                            "seed": seed,
                            "status": "ok",
                            "returncode": 0,
                            "metrics": metrics,
                            "num_steps": 800,
                            "artifacts": {
                                "metrics_json": str(replay_output_dir / "metrics.json"),
                                "rollout_npz": str(replay_output_dir / "rollout.npz"),
                            },
                            **_replay_evidence(
                                [
                                    "python",
                                    "--saved-command",
                                    str(output_dir / "mpc_command.npz"),
                                ]
                            ),
                        }
                    )

            report = runner._build_report(
                baseline_manifest=manifest_path,
                baseline_rows=list(manifest["rows"]),
                baseline_envelopes=frozen,
                mjx_rows=mjx_rows,
                replay_rows=replay_rows,
                min_speedup=12.0,
                target="h100_speedup",
                min_realtime_factor=1.0,
                required_gpu_name_fragment="H100",
            )

        self.assertIn("root_pos_error_mean", report["motion_results"]["jump"]["mjx_failures"])

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
        self.assertIn(
            "mjx_steady_state_wall_time",
            report["speed_results"]["jump"]["failures"],
        )
        self.assertIn("metrics_json", report["motion_results"]["jump"]["mjx_failures"])
        self.assertIn("rollout_npz", report["motion_results"]["jump"]["mjx_failures"])
        self.assertIn("mpc_command_npz", report["motion_results"]["jump"]["mjx_failures"])

    def test_stale_mjx_artifact_paths_fail_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                    **_fresh_metrics_provenance(argv, is_replay=is_replay),
                }
                if is_replay:
                    row.update(_replay_evidence(argv))
                    return row
                row.update(
                    {
                        "mpc_accepted": True,
                        "accepted_windows": 40,
                        "mpc_used_baseline_fallback": False,
                        "compile_init_wall_time_sec": 2.0,
                        "jit_warmup_enabled": True,
                        "jit_warmup_wall_time_sec": 1.5,
                        "runtime_visible_devices": ("0",),
                        "steady_state_wall_time_sec": 1.0,
                        **_mjx_contact_evidence(),
                    }
                )
                if argv[argv.index("--seed") + 1] == "1":
                    row["command_start_time_ns"] = time.time_ns() + 10_000_000_000
                return row

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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn("metrics_json_stale", report["motion_results"]["jump"]["mjx_failures"])

    def test_stale_baseline_artifact_paths_fail_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            stale_row = manifest["rows"][1]
            stale_row["command_start_time_ns"] = time.time_ns() + 10_000_000_000
            stale_row["artifact_mtime_ns"] = {
                "metrics_json": Path(stale_row["artifacts"]["metrics_json"]).stat().st_mtime_ns,
                "rollout_npz": Path(stale_row["artifacts"]["rollout_npz"]).stat().st_mtime_ns,
                "mpc_command_npz": Path(stale_row["artifacts"]["mpc_command_npz"]).stat().st_mtime_ns,
            }
            manifest_path.write_text(json.dumps(manifest))
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                    **_fresh_metrics_provenance(argv, is_replay=is_replay),
                }
                if is_replay:
                    row.update(_replay_evidence(argv))
                    return row
                row.update(
                    {
                        "mpc_accepted": True,
                        "accepted_windows": 40,
                        "mpc_used_baseline_fallback": False,
                        "compile_init_wall_time_sec": 2.0,
                        "jit_warmup_enabled": True,
                        "jit_warmup_wall_time_sec": 1.5,
                        "runtime_visible_devices": ("0",),
                        "steady_state_wall_time_sec": 1.0,
                        **_mjx_contact_evidence(),
                    }
                )
                return row

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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "metrics_json_stale",
            report["motion_results"]["jump"]["baseline_failures"],
        )

    def test_invalid_baseline_rollout_npz_fails_before_launching_mjx(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            row = manifest["rows"][0]
            rollout_path = Path(row["artifacts"]["rollout_npz"])
            rollout_path.write_text("not an npz")
            row["artifact_sha256"]["rollout_npz"] = _file_sha256(rollout_path)
            row["artifact_mtime_ns"]["rollout_npz"] = rollout_path.stat().st_mtime_ns
            manifest_path.write_text(json.dumps(manifest))
            output_dir = root / "acceptance"

            with mock.patch.object(
                runner,
                "run_command",
                side_effect=AssertionError("run_command should not be called"),
            ):
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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "rollout_npz_schema",
            report["motion_results"]["jump"]["baseline_failures"],
        )

    def test_missing_baseline_freshness_evidence_fails_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            manifest["rows"][1].pop("command_start_time_ns")
            manifest["rows"][1].pop("artifact_mtime_ns")
            manifest_path.write_text(json.dumps(manifest))
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                    **_fresh_metrics_provenance(argv, is_replay=is_replay),
                }
                if is_replay:
                    row.update(_replay_evidence(argv))
                    return row
                row.update(
                    {
                        "mpc_accepted": True,
                        "accepted_windows": 40,
                        "mpc_used_baseline_fallback": False,
                        "compile_init_wall_time_sec": 2.0,
                        "jit_warmup_enabled": True,
                        "jit_warmup_wall_time_sec": 1.5,
                        "runtime_visible_devices": ("0",),
                        "steady_state_wall_time_sec": 1.0,
                        **_mjx_contact_evidence(),
                    }
                )
                return row

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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "metrics_json_stale",
            report["motion_results"]["jump"]["baseline_failures"],
        )

    def test_mismatched_baseline_artifact_hash_fails_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            manifest["rows"][1]["artifact_sha256"]["metrics_json"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest))
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                    **_fresh_metrics_provenance(argv, is_replay=is_replay),
                }
                if is_replay:
                    row.update(_replay_evidence(argv))
                    return row
                row.update(
                    {
                        "mpc_accepted": True,
                        "accepted_windows": 40,
                        "mpc_used_baseline_fallback": False,
                        "compile_init_wall_time_sec": 2.0,
                        "jit_warmup_enabled": True,
                        "jit_warmup_wall_time_sec": 1.5,
                        "runtime_visible_devices": ("0",),
                        "steady_state_wall_time_sec": 1.0,
                        **_mjx_contact_evidence(),
                    }
                )
                return row

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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "metrics_json_hash",
            report["motion_results"]["jump"]["baseline_failures"],
        )

    def test_missing_replay_artifacts_fail_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                if not is_replay:
                    _write_artifacts(output)
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
        self.assertIn("metrics_json", report["replay_results"]["jump"]["failures"])

    def test_replay_saved_command_mismatch_fails_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"
            stale_command = root / "stale_mpc_command.npz"
            stale_command.write_text("stale")

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                seed = int(argv[argv.index("--seed") + 1])
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                    **_fresh_metrics_provenance(argv, is_replay=is_replay),
                }
                if is_replay:
                    row.update(
                        _replay_evidence(
                            argv,
                            saved_command=str(stale_command) if seed == 1 else None,
                        )
                    )
                    return row
                row.update(
                    {
                        "mpc_accepted": True,
                        "accepted_windows": 40,
                        "mpc_used_baseline_fallback": False,
                        "compile_init_wall_time_sec": 2.0,
                        "jit_warmup_enabled": True,
                        "jit_warmup_wall_time_sec": 1.5,
                        "runtime_visible_devices": ("0",),
                        "steady_state_wall_time_sec": 1.0,
                        **_mjx_contact_evidence(),
                    }
                )
                return row

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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "replay_saved_command",
            report["replay_results"]["jump"]["failures"],
        )

    def test_replay_saved_command_must_match_matching_mjx_row(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            mjx_rows, replay_rows = _acceptance_rows_with_artifacts(root)
            bad_row = next(
                row
                for row in replay_rows
                if row["motion"] == "jump" and row["seed"] == 1
            )
            other_command = next(
                row["artifacts"]["mpc_command_npz"]
                for row in mjx_rows
                if row["motion"] == "walk" and row["seed"] == 2
            )
            saved_idx = bad_row["replay_argv"].index("--saved-command") + 1
            bad_row["replay_argv"][saved_idx] = other_command
            bad_row.update(_replay_evidence(bad_row["replay_argv"]))

            report = runner._build_report(
                baseline_manifest=manifest_path,
                baseline_rows=list(manifest["rows"]),
                baseline_envelopes=manifest["baseline_envelopes"],
                mjx_rows=mjx_rows,
                replay_rows=replay_rows,
                min_speedup=12.0,
                target="h100_speedup",
                min_realtime_factor=1.0,
                required_gpu_name_fragment="H100",
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "replay_saved_command_source",
            report["replay_results"]["jump"]["failures"],
        )

    def test_replay_command_frame_count_must_match_saved_command_npz(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            mjx_rows, replay_rows = _acceptance_rows_with_artifacts(root)
            bad_row = next(
                row
                for row in replay_rows
                if row["motion"] == "jump" and row["seed"] == 1
            )
            bad_row["mpc"]["num_command_frames"] = 900

            report = runner._build_report(
                baseline_manifest=manifest_path,
                baseline_rows=list(manifest["rows"]),
                baseline_envelopes=manifest["baseline_envelopes"],
                mjx_rows=mjx_rows,
                replay_rows=replay_rows,
                min_speedup=12.0,
                target="h100_speedup",
                min_realtime_factor=1.0,
                required_gpu_name_fragment="H100",
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "replay_command_npz_frames",
            report["replay_results"]["jump"]["failures"],
        )

    def test_replay_saved_command_hash_must_match_matching_mjx_artifact(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            mjx_rows, replay_rows = _acceptance_rows_with_artifacts(root)
            bad_row = next(
                row
                for row in replay_rows
                if row["motion"] == "jump" and row["seed"] == 1
            )
            bad_row["mpc"]["saved_command_sha256"] = "0" * 64

            report = runner._build_report(
                baseline_manifest=manifest_path,
                baseline_rows=list(manifest["rows"]),
                baseline_envelopes=manifest["baseline_envelopes"],
                mjx_rows=mjx_rows,
                replay_rows=replay_rows,
                min_speedup=12.0,
                target="h100_speedup",
                min_realtime_factor=1.0,
                required_gpu_name_fragment="H100",
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "replay_saved_command_hash",
            report["replay_results"]["jump"]["failures"],
        )

    def test_replay_rollout_ref_indices_must_stay_within_command_frames(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            mjx_rows, replay_rows = _acceptance_rows_with_artifacts(root)
            bad_row = next(
                row
                for row in replay_rows
                if row["motion"] == "jump" and row["seed"] == 1
            )
            bad_rollout = Path(bad_row["artifacts"]["rollout_npz"])
            arrays = _valid_rollout_arrays()
            arrays["ref_indices"][-1, 0] = 9999
            np.savez_compressed(bad_rollout, **arrays)
            bad_row["artifact_sha256"]["rollout_npz"] = _file_sha256(bad_rollout)

            report = runner._build_report(
                baseline_manifest=manifest_path,
                baseline_rows=list(manifest["rows"]),
                baseline_envelopes=manifest["baseline_envelopes"],
                mjx_rows=mjx_rows,
                replay_rows=replay_rows,
                min_speedup=12.0,
                target="h100_speedup",
                min_realtime_factor=1.0,
                required_gpu_name_fragment="H100",
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "replay_rollout_ref_indices",
            report["replay_results"]["jump"]["failures"],
        )

    def test_replay_metrics_method_must_match_replay_command(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            mjx_rows, replay_rows = _acceptance_rows_with_artifacts(root)
            bad_row = next(
                row
                for row in replay_rows
                if row["motion"] == "jump" and row["seed"] == 1
            )
            bad_row["metrics_method"] = "g1_wbc_joint_global"

            report = runner._build_report(
                baseline_manifest=manifest_path,
                baseline_rows=list(manifest["rows"]),
                baseline_envelopes=manifest["baseline_envelopes"],
                mjx_rows=mjx_rows,
                replay_rows=replay_rows,
                min_speedup=12.0,
                target="h100_speedup",
                min_realtime_factor=1.0,
                required_gpu_name_fragment="H100",
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "replay_metrics_provenance",
            report["replay_results"]["jump"]["failures"],
        )

    def test_missing_baseline_artifact_paths_fail_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            missing_path = root / "missing_baseline_rollout.npz"
            manifest["rows"][0]["artifacts"]["rollout_npz"] = str(missing_path)
            manifest_path.write_text(json.dumps(manifest))
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                    **_fresh_metrics_provenance(argv, is_replay=is_replay),
                }
                if is_replay:
                    row.update(_replay_evidence(argv))
                else:
                    row.update(
                        {
                            "mpc_accepted": True,
                            "accepted_windows": 40,
                            "mpc_used_baseline_fallback": False,
                            "compile_init_wall_time_sec": 2.0,
                            "jit_warmup_enabled": True,
                            "jit_warmup_wall_time_sec": 1.5,
                            "runtime_visible_devices": ("0",),
                            "steady_state_wall_time_sec": 1.0,
                            **_mjx_contact_evidence(),
                        }
                    )
                return row

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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn("rollout_npz", report["motion_results"]["jump"]["baseline_failures"])

    def test_missing_one_mjx_timing_seed_fails_speed_gate(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                _write_artifacts(output)
                is_seed_2 = argv[argv.index("--seed") + 1] == "2"
                is_replay = "replay_command" in argv
                timing = None if is_seed_2 and not is_replay else 1.0
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
                    "steady_state_wall_time_sec": timing,
                    "command_wall_time_sec": 1.0,
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

    def test_missing_mjx_compile_or_warmup_timing_fails_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                    **_fresh_metrics_provenance(argv, is_replay=is_replay),
                }
                if is_replay:
                    row.update(_replay_evidence(argv))
                else:
                    row.update(
                        {
                            "mpc_accepted": True,
                            "accepted_windows": 40,
                            "mpc_used_baseline_fallback": False,
                            "steady_state_wall_time_sec": 1.0,
                        }
                    )
                    row["mpc"].pop("compile_init_wall_time_sec", None)
                    row["mpc"].pop("jit_warmup_wall_time_sec", None)
                return row

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
        self.assertIn(
            "mjx_compile_init_wall_time",
            report["motion_results"]["jump"]["mjx_failures"],
        )
        self.assertIn(
            "mjx_jit_warmup_enabled",
            report["motion_results"]["jump"]["mjx_failures"],
        )
        self.assertIn(
            "mjx_jit_warmup_wall_time",
            report["motion_results"]["jump"]["mjx_failures"],
        )
        self.assertEqual(report["classification"], "invalid_benchmark")

    def test_speed_shortfall_classifies_as_speed_regression(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                    **_fresh_metrics_provenance(argv, is_replay=is_replay),
                }
                if is_replay:
                    row.update(_replay_evidence(argv))
                else:
                    row.update(
                        {
                            "mpc_accepted": True,
                            "accepted_windows": 40,
                            "mpc_used_baseline_fallback": False,
                            "compile_init_wall_time_sec": 2.0,
                            "jit_warmup_enabled": True,
                            "jit_warmup_wall_time_sec": 1.5,
                            "runtime_visible_devices": ("0",),
                            "steady_state_wall_time_sec": 20.0,
                            **_mjx_contact_evidence(),
                        }
                    )
                return row

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
        self.assertEqual(report["classification"], "speed_regression")
        self.assertIn("speedup", report["speed_results"]["jump"]["failures"])

    def test_worst_seed_speed_shortfall_fails_even_when_mean_speedup_passes(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                seed = int(argv[argv.index("--seed") + 1])
                motion = Path(argv[argv.index("--motion") + 1]).stem
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                    **_fresh_metrics_provenance(argv, is_replay=is_replay),
                }
                if is_replay:
                    row.update(_replay_evidence(argv))
                    return row
                mjx_times = {
                    "jump": {0: 0.1, 1: 0.1, 2: 29.8},
                    "walk": {0: 1.0, 1: 1.0, 2: 1.0},
                }
                row.update(
                    {
                        "mpc_accepted": True,
                        "accepted_windows": 40,
                        "mpc_used_baseline_fallback": False,
                        "compile_init_wall_time_sec": 2.0,
                        "jit_warmup_enabled": True,
                        "jit_warmup_wall_time_sec": 1.5,
                        "runtime_visible_devices": ("0",),
                        "steady_state_wall_time_sec": mjx_times[motion][seed],
                        **_mjx_contact_evidence(),
                    }
                )
                return row

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
        self.assertEqual(report["classification"], "speed_regression")
        self.assertEqual(report["speed_results"]["jump"]["speedup"], 12.0)
        self.assertAlmostEqual(report["speed_results"]["jump"]["worst_speedup"], 120.0 / 29.8)
        self.assertIn("speedup_worst", report["speed_results"]["jump"]["failures"])
        self.assertNotIn("speedup", report["speed_results"]["jump"]["failures"])

    def test_missing_or_multiple_mjx_visible_devices_fail_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                }
                if is_replay:
                    row["command_wall_time_sec"] = 1.0
                    return row
                row.update(
                    {
                        "mpc_accepted": True,
                        "accepted_windows": 40,
                        "mpc_used_baseline_fallback": False,
                        "compile_init_wall_time_sec": 2.0,
                        "jit_warmup_enabled": True,
                        "jit_warmup_wall_time_sec": 1.5,
                        "steady_state_wall_time_sec": 1.0,
                        **_mjx_contact_evidence(),
                    }
                )
                if argv[argv.index("--seed") + 1] == "1":
                    row["runtime_visible_devices"] = ("0", "1")
                elif argv[argv.index("--seed") + 1] == "2":
                    row["runtime_visible_devices"] = ("0",)
                return row

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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "mjx_runtime_visible_devices",
            report["motion_results"]["jump"]["mjx_failures"],
        )
        self.assertIn(
            "mjx_single_visible_gpu",
            report["motion_results"]["jump"]["mjx_failures"],
        )

    def test_missing_or_non_h100_mjx_gpu_name_fails_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                }
                if is_replay:
                    row["command_wall_time_sec"] = 1.0
                    return row
                row.update(
                    {
                        "mpc_accepted": True,
                        "accepted_windows": 40,
                        "mpc_used_baseline_fallback": False,
                        "compile_init_wall_time_sec": 2.0,
                        "jit_warmup_enabled": True,
                        "jit_warmup_wall_time_sec": 1.5,
                        "runtime_visible_devices": ("0",),
                        "steady_state_wall_time_sec": 1.0,
                        **_mjx_contact_evidence(),
                    }
                )
                if argv[argv.index("--seed") + 1] == "1":
                    row["runtime_gpu_name"] = "NVIDIA GeForce RTX 4090"
                elif argv[argv.index("--seed") + 1] == "2":
                    row.pop("runtime_gpu_name")
                return row

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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "mjx_runtime_gpu_name",
            report["motion_results"]["jump"]["mjx_failures"],
        )
        self.assertIn(
            "mjx_required_gpu",
            report["motion_results"]["jump"]["mjx_failures"],
        )

    def test_missing_or_non_h100_baseline_gpu_name_fails_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            manifest["rows"][0]["runtime_gpu_name"] = "NVIDIA GeForce RTX 4090"
            manifest["rows"][1].pop("runtime_gpu_name")
            manifest["rows"][2].pop("runtime_visible_devices")
            manifest["rows"][3]["runtime_visible_devices"] = ["0", "1"]
            manifest_path.write_text(json.dumps(manifest))
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                }
                if is_replay:
                    row.update(_replay_evidence(argv))
                    return row
                row.update(
                    {
                        "mpc_accepted": True,
                        "accepted_windows": 40,
                        "mpc_used_baseline_fallback": False,
                        "compile_init_wall_time_sec": 2.0,
                        "jit_warmup_enabled": True,
                        "jit_warmup_wall_time_sec": 1.5,
                        "runtime_visible_devices": ("0",),
                        "steady_state_wall_time_sec": 1.0,
                        **_mjx_contact_evidence(),
                    }
                )
                return row

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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "baseline_runtime_gpu_name",
            report["motion_results"]["jump"]["baseline_failures"],
        )
        self.assertIn(
            "baseline_required_gpu",
            report["motion_results"]["jump"]["baseline_failures"],
        )
        self.assertIn(
            "baseline_runtime_visible_devices",
            report["motion_results"]["jump"]["baseline_failures"],
        )
        self.assertIn(
            "baseline_single_visible_gpu",
            report["motion_results"]["walk"]["baseline_failures"],
        )

    def test_missing_or_saturated_mjx_contact_diagnostics_fail_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                }
                if is_replay:
                    row["command_wall_time_sec"] = 1.0
                    return row
                row.update(
                    {
                        "mpc_accepted": True,
                        "accepted_windows": 40,
                        "mpc_used_baseline_fallback": False,
                        "compile_init_wall_time_sec": 2.0,
                        "jit_warmup_enabled": True,
                        "jit_warmup_wall_time_sec": 1.5,
                        "runtime_visible_devices": ("0",),
                        "steady_state_wall_time_sec": 1.0,
                    }
                )
                if argv[argv.index("--seed") + 1] == "1":
                    row.update(
                        _mjx_contact_evidence(max_contact_points_saturated=True)
                    )
                elif argv[argv.index("--seed") + 1] == "2":
                    row.update(_mjx_contact_evidence(active_contact_count=4))
                return row

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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "mjx_contact_diagnostics",
            report["motion_results"]["jump"]["mjx_failures"],
        )
        self.assertIn(
            "max_contact_points_saturation",
            report["motion_results"]["jump"]["mjx_failures"],
        )

    def test_main_returns_zero_when_all_gates_have_complete_evidence(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                seed = int(argv[argv.index("--seed") + 1])
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                    **_fresh_metrics_provenance(argv, is_replay=is_replay),
                }
                if is_replay:
                    row.update(_replay_evidence(argv))
                else:
                    row.update(
                        {
                            "mpc_accepted": True,
                            "accepted_windows": 40,
                            "mpc_used_baseline_fallback": False,
                            "compile_init_wall_time_sec": 2.0,
                            "jit_warmup_enabled": True,
                            "jit_warmup_wall_time_sec": 1.5,
                            "runtime_visible_devices": ("0",),
                            "steady_state_wall_time_sec": 1.0 + seed,
                            **_mjx_contact_evidence(active_contact_count=seed),
                        }
                    )
                return row

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

        self.assertEqual(exit_code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["min_speedup"], 12.0)
        self.assertEqual(len(report["planned_runs"]), 6)
        self.assertEqual(len(report["mjx_rows"]), 6)
        self.assertEqual(len(report["replay_rows"]), 6)
        self.assertEqual(report["classification"], "pass_h100_milestone")
        for row in report["mjx_rows"]:
            with self.subTest(kind="mjx_hashes", motion=row["motion"], seed=row["seed"]):
                self.assertEqual(
                    set(row["artifact_sha256"]),
                    {"metrics_json", "rollout_npz", "mpc_command_npz"},
                )
        for row in report["replay_rows"]:
            with self.subTest(kind="replay_hashes", motion=row["motion"], seed=row["seed"]):
                self.assertEqual(
                    set(row["artifact_sha256"]),
                    {"metrics_json", "rollout_npz"},
                )
        jump_timing = report["timing_summary"]["jump"]
        self.assertEqual(
            jump_timing["baseline"]["steady_state_wall_time_sec"]["values"],
            [120.0, 120.0, 120.0],
        )
        self.assertEqual(jump_timing["baseline"]["steady_state_wall_time_sec"]["mean"], 120.0)
        self.assertEqual(jump_timing["mjx"]["steady_state_wall_time_sec"]["values"], [1.0, 2.0, 3.0])
        self.assertEqual(jump_timing["mjx"]["steady_state_wall_time_sec"]["mean"], 2.0)
        self.assertEqual(jump_timing["mjx"]["steady_state_wall_time_sec"]["min"], 1.0)
        self.assertEqual(jump_timing["mjx"]["steady_state_wall_time_sec"]["max"], 3.0)
        self.assertEqual(jump_timing["mjx"]["compile_init_wall_time_sec"]["mean"], 2.0)
        self.assertEqual(jump_timing["mjx"]["jit_warmup_wall_time_sec"]["mean"], 1.5)
        self.assertEqual(jump_timing["mjx"]["per_window_steady_state_wall_time_sec"]["values"], [0.025, 0.05, 0.075])
        self.assertEqual(jump_timing["mjx"]["num_windows"]["values"], [40, 40, 40])
        self.assertEqual(jump_timing["mjx"]["runtime_visible_devices"], [["0"], ["0"], ["0"]])
        self.assertEqual(jump_timing["replay"]["command_wall_time_sec"]["values"], [1.0, 1.0, 1.0])
        jump_contact = report["contact_summary"]["jump"]["mjx"]
        self.assertEqual(jump_contact["max_contact_points"]["values"], [512, 512, 512])
        self.assertEqual(jump_contact["max_geom_pairs"]["values"], [1024, 1024, 1024])
        self.assertEqual(jump_contact["contact_pair_count"]["values"], [0, 0, 0])
        self.assertEqual(jump_contact["active_contact_count"]["values"], [0, 1, 2])
        self.assertEqual(
            jump_contact["saturation_flags"],
            {
                "contact_saturated": [False, False, False],
                "max_contact_points_saturated": [False, False, False],
                "max_geom_pairs_saturated": [False, False, False],
            },
        )

    def test_fresh_mjx_metrics_with_wrong_backend_fail_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cuda:0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())
            plan = runner.build_acceptance_plan(args, manifest)
            planned_by_mjx_output = {planned.output_dir: planned for planned in plan}
            planned_by_replay_output = {
                planned.replay_output_dir: planned for planned in plan
            }

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                planned = (
                    planned_by_replay_output[str(output)]
                    if is_replay
                    else planned_by_mjx_output[str(output)]
                )
                _write_artifacts(output, include_command=not is_replay)
                payload = (
                    _replay_metrics_payload(planned)
                    if is_replay
                    else _mjx_metrics_payload(planned)
                )
                if not is_replay and planned.motion == "jump" and planned.seed == 0:
                    payload["mpc"]["mpc_backend"] = "mujoco_warp"
                (output / "metrics.json").write_text(json.dumps(payload))
                row = runner._row_from_metrics(output / "metrics.json")
                return {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "command_wall_time_sec": 1.0,
                    "command_start_time_ns": min(
                        (output / name).stat().st_mtime_ns
                        for name in (
                            ("metrics.json", "rollout.npz")
                            if is_replay
                            else ("metrics.json", "rollout.npz", "mpc_command.npz")
                        )
                    )
                    - 1_000_000,
                    **row,
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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "mjx_metrics_provenance",
            report["motion_results"]["jump"]["mjx_failures"],
        )

    def test_fresh_mjx_metrics_without_physics_scan_fail_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cuda:0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())
            plan = runner.build_acceptance_plan(args, manifest)
            planned_by_mjx_output = {planned.output_dir: planned for planned in plan}
            planned_by_replay_output = {
                planned.replay_output_dir: planned for planned in plan
            }

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                planned = (
                    planned_by_replay_output[str(output)]
                    if is_replay
                    else planned_by_mjx_output[str(output)]
                )
                _write_artifacts(output, include_command=not is_replay)
                payload = (
                    _replay_metrics_payload(planned)
                    if is_replay
                    else _mjx_metrics_payload(planned)
                )
                if not is_replay and planned.motion == "jump" and planned.seed == 0:
                    payload["mpc"]["physics_scan_enabled"] = False
                (output / "metrics.json").write_text(json.dumps(payload))
                row = runner._row_from_metrics(output / "metrics.json")
                return {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "command_wall_time_sec": 1.0,
                    "command_start_time_ns": min(
                        (output / name).stat().st_mtime_ns
                        for name in (
                            ("metrics.json", "rollout.npz")
                            if is_replay
                            else ("metrics.json", "rollout.npz", "mpc_command.npz")
                        )
                    )
                    - 1_000_000,
                    **row,
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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "mjx_physics_scan_enabled",
            report["motion_results"]["jump"]["mjx_failures"],
        )

    def test_fresh_mjx_metrics_without_physics_step_count_fail_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cuda:0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())
            plan = runner.build_acceptance_plan(args, manifest)
            planned_by_mjx_output = {planned.output_dir: planned for planned in plan}
            planned_by_replay_output = {
                planned.replay_output_dir: planned for planned in plan
            }

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                planned = (
                    planned_by_replay_output[str(output)]
                    if is_replay
                    else planned_by_mjx_output[str(output)]
                )
                _write_artifacts(output, include_command=not is_replay)
                payload = (
                    _replay_metrics_payload(planned)
                    if is_replay
                    else _mjx_metrics_payload(planned)
                )
                if not is_replay and planned.motion == "jump" and planned.seed == 0:
                    payload["mpc"].pop("physics_step_count_min")
                (output / "metrics.json").write_text(json.dumps(payload))
                row = runner._row_from_metrics(output / "metrics.json")
                return {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "command_wall_time_sec": 1.0,
                    "command_start_time_ns": min(
                        (output / name).stat().st_mtime_ns
                        for name in (
                            ("metrics.json", "rollout.npz")
                            if is_replay
                            else ("metrics.json", "rollout.npz", "mpc_command.npz")
                        )
                    )
                    - 1_000_000,
                    **row,
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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "mjx_physics_step_count",
            report["motion_results"]["jump"]["mjx_failures"],
        )

    def test_fresh_replay_metrics_with_multiple_visible_devices_fail_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cuda:0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())
            plan = runner.build_acceptance_plan(args, manifest)
            planned_by_mjx_output = {planned.output_dir: planned for planned in plan}
            planned_by_replay_output = {
                planned.replay_output_dir: planned for planned in plan
            }

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                planned = (
                    planned_by_replay_output[str(output)]
                    if is_replay
                    else planned_by_mjx_output[str(output)]
                )
                _write_artifacts(output, include_command=not is_replay)
                payload = (
                    _replay_metrics_payload(planned)
                    if is_replay
                    else _mjx_metrics_payload(planned)
                )
                if is_replay and planned.motion == "jump" and planned.seed == 0:
                    payload["mpc"]["runtime_visible_devices"] = ["0", "1"]
                (output / "metrics.json").write_text(json.dumps(payload))
                row = runner._row_from_metrics(output / "metrics.json")
                return {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "command_wall_time_sec": 1.0,
                    "command_start_time_ns": min(
                        (output / name).stat().st_mtime_ns
                        for name in (
                            ("metrics.json", "rollout.npz")
                            if is_replay
                            else ("metrics.json", "rollout.npz", "mpc_command.npz")
                        )
                    )
                    - 1_000_000,
                    **row,
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
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "replay_single_visible_gpu",
            report["replay_results"]["jump"]["failures"],
        )

    def test_mismatched_mjx_artifact_hash_fails_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            mjx_rows = []
            replay_rows = []
            for motion in ("jump", "walk"):
                for seed in (0, 1, 2):
                    output_dir = root / "acceptance" / motion / f"seed_{seed}" / "mjx"
                    replay_output_dir = root / "acceptance" / motion / f"seed_{seed}" / "replay"
                    _write_artifacts(output_dir)
                    _write_artifacts(replay_output_dir, include_command=False)
                    mjx_artifacts = {
                        "metrics_json": str(output_dir / "metrics.json"),
                        "rollout_npz": str(output_dir / "rollout.npz"),
                        "mpc_command_npz": str(output_dir / "mpc_command.npz"),
                    }
                    replay_artifacts = {
                        "metrics_json": str(replay_output_dir / "metrics.json"),
                        "rollout_npz": str(replay_output_dir / "rollout.npz"),
                    }
                    mjx_hashes = {
                        key: _file_sha256(Path(path))
                        for key, path in mjx_artifacts.items()
                    }
                    replay_hashes = {
                        key: _file_sha256(Path(path))
                        for key, path in replay_artifacts.items()
                    }
                    if motion == "jump" and seed == 1:
                        mjx_hashes["metrics_json"] = "0" * 64
                    mjx_rows.append(
                        {
                            "motion": motion,
                            "seed": seed,
                            "status": "ok",
                            "returncode": 0,
                            "metrics": _metrics(success=True),
                            "mpc_accepted": True,
                            "accepted_windows": 40,
                            "mpc_used_baseline_fallback": False,
                            "num_steps": 800,
                            "compile_init_wall_time_sec": 2.0,
                            "jit_warmup_enabled": True,
                            "jit_warmup_wall_time_sec": 1.5,
                            "runtime_visible_devices": ("0",),
                            "steady_state_wall_time_sec": 1.0,
                            "artifacts": mjx_artifacts,
                            "artifact_sha256": mjx_hashes,
                            **_mjx_contact_evidence(),
                        }
                    )
                    replay_rows.append(
                        {
                            "motion": motion,
                            "seed": seed,
                            "status": "ok",
                            "returncode": 0,
                            "metrics": _metrics(success=True),
                            "num_steps": 800,
                            "artifacts": replay_artifacts,
                            "artifact_sha256": replay_hashes,
                            **_replay_evidence(
                                [
                                    "python",
                                    "--saved-command",
                                    str(output_dir / "mpc_command.npz"),
                                ]
                            ),
                        }
                    )

            report = runner._build_report(
                baseline_manifest=manifest_path,
                baseline_rows=list(manifest["rows"]),
                baseline_envelopes=manifest["baseline_envelopes"],
                mjx_rows=mjx_rows,
                replay_rows=replay_rows,
                min_speedup=12.0,
                target="h100_speedup",
                min_realtime_factor=1.0,
                required_gpu_name_fragment="H100",
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn("metrics_json_hash", report["motion_results"]["jump"]["mjx_failures"])

    def test_malformed_mjx_rollout_npz_fails_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            mjx_rows, replay_rows = _acceptance_rows_with_artifacts(root)
            bad_row = next(
                row
                for row in mjx_rows
                if row["motion"] == "jump" and row["seed"] == 1
            )
            bad_rollout = Path(bad_row["artifacts"]["rollout_npz"])
            arrays = _valid_rollout_arrays()
            arrays.pop("body_ang_vel_w")
            np.savez_compressed(bad_rollout, **arrays)
            bad_row["artifact_sha256"]["rollout_npz"] = _file_sha256(bad_rollout)

            report = runner._build_report(
                baseline_manifest=manifest_path,
                baseline_rows=list(manifest["rows"]),
                baseline_envelopes=manifest["baseline_envelopes"],
                mjx_rows=mjx_rows,
                replay_rows=replay_rows,
                min_speedup=12.0,
                target="h100_speedup",
                min_realtime_factor=1.0,
                required_gpu_name_fragment="H100",
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn("rollout_npz_schema", report["motion_results"]["jump"]["mjx_failures"])

    def test_malformed_mjx_command_npz_fails_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            mjx_rows, replay_rows = _acceptance_rows_with_artifacts(root)
            bad_row = next(
                row
                for row in mjx_rows
                if row["motion"] == "jump" and row["seed"] == 1
            )
            bad_command = Path(bad_row["artifacts"]["mpc_command_npz"])
            arrays = _valid_command_arrays()
            arrays.pop("command_qvel_trajectory")
            np.savez_compressed(bad_command, **arrays)
            bad_row["artifact_sha256"]["mpc_command_npz"] = _file_sha256(bad_command)

            report = runner._build_report(
                baseline_manifest=manifest_path,
                baseline_rows=list(manifest["rows"]),
                baseline_envelopes=manifest["baseline_envelopes"],
                mjx_rows=mjx_rows,
                replay_rows=replay_rows,
                min_speedup=12.0,
                target="h100_speedup",
                min_realtime_factor=1.0,
                required_gpu_name_fragment="H100",
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn("mpc_command_npz_schema", report["motion_results"]["jump"]["mjx_failures"])

    def test_mjx_command_refined_qpos_must_match_rollout_qpos(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            mjx_rows, replay_rows = _acceptance_rows_with_artifacts(root)
            bad_row = next(
                row
                for row in mjx_rows
                if row["motion"] == "jump" and row["seed"] == 1
            )
            bad_command = Path(bad_row["artifacts"]["mpc_command_npz"])
            arrays = _valid_command_arrays()
            arrays["refined_qpos"][10, 0] = 1.0
            np.savez_compressed(bad_command, **arrays)
            bad_row["artifact_sha256"]["mpc_command_npz"] = _file_sha256(bad_command)

            report = runner._build_report(
                baseline_manifest=manifest_path,
                baseline_rows=list(manifest["rows"]),
                baseline_envelopes=manifest["baseline_envelopes"],
                mjx_rows=mjx_rows,
                replay_rows=replay_rows,
                min_speedup=12.0,
                target="h100_speedup",
                min_realtime_factor=1.0,
                required_gpu_name_fragment="H100",
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "mpc_rollout_qpos_mismatch",
            report["motion_results"]["jump"]["mjx_failures"],
        )

    def test_mjx_command_qpos_trajectory_must_match_refined_qpos(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            mjx_rows, replay_rows = _acceptance_rows_with_artifacts(root)
            bad_row = next(
                row
                for row in mjx_rows
                if row["motion"] == "jump" and row["seed"] == 1
            )
            bad_command = Path(bad_row["artifacts"]["mpc_command_npz"])
            arrays = _valid_command_arrays()
            arrays["command_qpos_trajectory"][10, 0, 0] = 1.0
            np.savez_compressed(bad_command, **arrays)
            bad_row["artifact_sha256"]["mpc_command_npz"] = _file_sha256(bad_command)

            report = runner._build_report(
                baseline_manifest=manifest_path,
                baseline_rows=list(manifest["rows"]),
                baseline_envelopes=manifest["baseline_envelopes"],
                mjx_rows=mjx_rows,
                replay_rows=replay_rows,
                min_speedup=12.0,
                target="h100_speedup",
                min_realtime_factor=1.0,
                required_gpu_name_fragment="H100",
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "mpc_command_qpos_mismatch",
            report["motion_results"]["jump"]["mjx_failures"],
        )

    def test_mjx_command_qvel_trajectory_must_match_refined_qpos_derivative(
        self,
    ) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            mjx_rows, replay_rows = _acceptance_rows_with_artifacts(root)
            bad_row = next(
                row
                for row in mjx_rows
                if row["motion"] == "jump" and row["seed"] == 1
            )
            bad_command = Path(bad_row["artifacts"]["mpc_command_npz"])
            arrays = _valid_command_arrays()
            arrays["refined_qpos"][1:, 0] = 0.02
            arrays["command_qpos_trajectory"][:, 0] = arrays["refined_qpos"]
            arrays["command_qvel_trajectory"][:] = 0.0
            np.savez_compressed(bad_command, **arrays)
            bad_row["artifact_sha256"]["mpc_command_npz"] = _file_sha256(bad_command)

            report = runner._build_report(
                baseline_manifest=manifest_path,
                baseline_rows=list(manifest["rows"]),
                baseline_envelopes=manifest["baseline_envelopes"],
                mjx_rows=mjx_rows,
                replay_rows=replay_rows,
                min_speedup=12.0,
                target="h100_speedup",
                min_realtime_factor=1.0,
                required_gpu_name_fragment="H100",
            )

        self.assertFalse(report["passed"])
        self.assertEqual(report["classification"], "invalid_benchmark")
        self.assertIn(
            "mpc_command_qvel_mismatch",
            report["motion_results"]["jump"]["mjx_failures"],
        )

    def test_4090_target_reports_realtime_pass_classification(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            _rewrite_baseline_gpu(manifest_path, "NVIDIA GeForce RTX 4090")
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                    **_fresh_metrics_provenance(argv, is_replay=is_replay),
                }
                if is_replay:
                    row.update(
                        _replay_evidence(
                            argv,
                            runtime_gpu_name="NVIDIA GeForce RTX 4090",
                        )
                    )
                else:
                    row.update(
                        {
                            "mpc_accepted": True,
                            "accepted_windows": 40,
                            "mpc_used_baseline_fallback": False,
                            "compile_init_wall_time_sec": 2.0,
                            "jit_warmup_enabled": True,
                            "jit_warmup_wall_time_sec": 1.5,
                            "runtime_visible_devices": ("0",),
                            "steady_state_wall_time_sec": 8.0,
                            "control_dt_sec": 0.02,
                            "evaluated_motion_duration_sec": 16.0,
                            **_mjx_contact_evidence(),
                            "runtime_gpu_name": "NVIDIA GeForce RTX 4090",
                        }
                    )
                return row

            with mock.patch.object(runner, "run_command", side_effect=fake_run_command):
                exit_code = runner.main(
                    [
                        "--baseline-manifest",
                        str(manifest_path),
                        "--output-dir",
                        str(output_dir),
                        "--device",
                        "cuda:0",
                        "--target",
                        "4090_realtime",
                        "--required-gpu-name-fragment",
                        "4090",
                    ]
                )

            report = json.loads((output_dir / "acceptance_report.json").read_text())

        self.assertEqual(exit_code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["target"], "4090_realtime")
        self.assertEqual(report["classification"], "pass_4090_realtime")
        self.assertEqual(report["realtime_results"]["jump"]["motion_duration_sec"], 16.0)
        self.assertEqual(report["realtime_results"]["jump"]["real_time_factor"], 2.0)

    def test_4090_target_fails_when_realtime_factor_is_below_target(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            _rewrite_baseline_gpu(manifest_path, "NVIDIA GeForce RTX 4090")
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                output = Path(argv[argv.index("--output-dir") + 1])
                is_replay = "replay_command" in argv
                _write_artifacts(output, include_command=not is_replay)
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                    **_fresh_metrics_provenance(argv, is_replay=is_replay),
                }
                if is_replay:
                    row.update(
                        _replay_evidence(
                            argv,
                            runtime_gpu_name="NVIDIA GeForce RTX 4090",
                        )
                    )
                else:
                    row.update(
                        {
                            "mpc_accepted": True,
                            "accepted_windows": 40,
                            "mpc_used_baseline_fallback": False,
                            "compile_init_wall_time_sec": 2.0,
                            "jit_warmup_enabled": True,
                            "jit_warmup_wall_time_sec": 1.5,
                            "runtime_visible_devices": ("0",),
                            "steady_state_wall_time_sec": 20.0,
                            "control_dt_sec": 0.02,
                            "evaluated_motion_duration_sec": 16.0,
                            **_mjx_contact_evidence(),
                            "runtime_gpu_name": "NVIDIA GeForce RTX 4090",
                        }
                    )
                return row

            with mock.patch.object(runner, "run_command", side_effect=fake_run_command):
                exit_code = runner.main(
                    [
                        "--baseline-manifest",
                        str(manifest_path),
                        "--output-dir",
                        str(output_dir),
                        "--device",
                        "cuda:0",
                        "--target",
                        "4090_realtime",
                        "--required-gpu-name-fragment",
                        "4090",
                    ]
                )

            report = json.loads((output_dir / "acceptance_report.json").read_text())

        self.assertEqual(exit_code, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(report["classification"], "speed_regression")
        self.assertIn("real_time_factor", report["realtime_results"]["jump"]["failures"])

    def test_realtime_gate_requires_explicit_duration_evidence_when_strict(self) -> None:
        runner = load_runner()
        rows = [
            {
                "metrics": _metrics(success=True),
                "num_steps": 800,
                "steady_state_wall_time_sec": 8.0,
            }
            for _ in range(3)
        ]

        gate = runner._realtime_gate_for_motion(
            rows,
            min_realtime_factor=1.0,
            require_explicit_duration=True,
        )

        self.assertFalse(gate["passed"])
        self.assertIn("control_dt_sec", gate["failures"])
        self.assertIn("evaluated_motion_duration_sec", gate["failures"])

    def test_metrics_parser_extracts_compile_and_warmup_timing(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "metrics.json"
            path.write_text(
                json.dumps(
                    {
                        "metrics": {
                            **_metrics(success=True),
                            "control_dt_sec": 0.02,
                            "evaluated_motion_duration_sec": 16.0,
                        },
                        "mpc": {
                            "accepted": True,
                            "accepted_windows": 40,
                            "used_baseline_fallback": False,
                            "compile_init_wall_time_sec": 2.0,
                            "jit_warmup_enabled": True,
                            "jit_warmup_wall_time_sec": 1.5,
                            "runtime_visible_devices": ["0"],
                            "steady_state_wall_time_sec": 1.0,
                            **_mjx_contact_evidence(active_contact_count=3),
                        },
                    }
                )
            )

            row = runner._row_from_metrics(path)

        self.assertEqual(row["compile_init_wall_time_sec"], 2.0)
        self.assertTrue(row["jit_warmup_enabled"])
        self.assertEqual(row["jit_warmup_wall_time_sec"], 1.5)
        self.assertEqual(row["runtime_visible_devices"], ["0"])
        self.assertEqual(row["runtime_gpu_name"], "NVIDIA H100 80GB HBM3")
        self.assertEqual(row["control_dt_sec"], 0.02)
        self.assertEqual(row["evaluated_motion_duration_sec"], 16.0)
        self.assertFalse(row["contact_saturated"])
        self.assertEqual(row["max_contact_points"], 512)
        self.assertEqual(row["active_contact_count"], 3)

    def test_metrics_parser_preserves_top_level_provenance(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "metrics.json"
            path.write_text(
                json.dumps(
                    {
                        "method": "replay_command",
                        "motion": "/tmp/jump.npz",
                        "device": "cuda:0",
                        "metrics": _metrics(success=True),
                        "mpc": {},
                    }
                )
            )

            row = runner._row_from_metrics(path)

        self.assertEqual(row["metrics_method"], "replay_command")
        self.assertEqual(row["metrics_motion"], "/tmp/jump.npz")
        self.assertEqual(row["metrics_device"], "cuda:0")

    def test_metrics_parser_missing_mpc_safety_fields_fails_closed(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "metrics.json"
            path.write_text(
                json.dumps(
                    {
                        "metrics": _metrics(success=True),
                        "mpc": {"accepted_windows": 40},
                    }
                )
            )

            row = runner._row_from_metrics(path)

        self.assertFalse(row["mpc_accepted"])
        self.assertTrue(row["mpc_used_baseline_fallback"])


def _write_artifacts(output_dir: Path, *, include_command: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text("{}")
    np.savez_compressed(output_dir / "rollout.npz", **_valid_rollout_arrays())
    if include_command:
        np.savez_compressed(
            output_dir / "mpc_command.npz",
            **_valid_command_arrays(),
        )


def _write_reusable_acceptance_outputs(
    runner,
    plan,
    *,
    output_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    mjx_rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []
    for planned in plan:
        mjx_output = Path(planned.output_dir)
        replay_output = Path(planned.replay_output_dir)
        _write_artifacts(mjx_output)
        _write_artifacts(replay_output, include_command=False)

        mjx_payload = _mjx_metrics_payload(planned)
        (mjx_output / "metrics.json").write_text(json.dumps(mjx_payload))
        mjx_mpc = mjx_payload["mpc"]
        mjx_rows.append(
            runner._attach_artifacts(
                {
                    **runner.asdict(planned),
                    "status": "ok",
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "command_start_time_ns": min(
                        (mjx_output / name).stat().st_mtime_ns
                        for name in ("metrics.json", "rollout.npz", "mpc_command.npz")
                    ) - 1_000_000,
                    "command_wall_time_sec": 11.0,
                    "metrics": mjx_payload["metrics"],
                    "mpc": mjx_mpc,
                    "metrics_method": mjx_payload["method"],
                    "metrics_motion": mjx_payload["motion"],
                    "metrics_device": mjx_payload["device"],
                    "mpc_accepted": True,
                    "accepted_windows": 40,
                    "mpc_used_baseline_fallback": False,
                    "num_steps": 800,
                    "compile_init_wall_time_sec": mjx_mpc["compile_init_wall_time_sec"],
                    "jit_warmup_enabled": True,
                    "jit_warmup_wall_time_sec": mjx_mpc["jit_warmup_wall_time_sec"],
                    "runtime_visible_devices": ["0"],
                    "runtime_gpu_name": "NVIDIA H100 80GB HBM3",
                    "steady_state_wall_time_sec": mjx_mpc["steady_state_wall_time_sec"],
                    **_mjx_contact_evidence(),
                },
                planned.output_dir,
            )
        )

        replay_payload = _replay_metrics_payload(planned)
        (replay_output / "metrics.json").write_text(json.dumps(replay_payload))
        replay_rows.append(
            runner._attach_artifacts(
                {
                    **runner.asdict(planned),
                    "status": "ok",
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "command_start_time_ns": min(
                        (replay_output / name).stat().st_mtime_ns
                        for name in ("metrics.json", "rollout.npz")
                    ) - 1_000_000,
                    "command_wall_time_sec": 12.0,
                    "metrics": replay_payload["metrics"],
                    "mpc": replay_payload["mpc"],
                    "metrics_method": "replay_command",
                    "metrics_motion": replay_payload["motion"],
                    "metrics_device": replay_payload["device"],
                    "num_steps": 800,
                },
                planned.replay_output_dir,
            )
        )

    partial = {
        "schema_version": 1,
        "run_status": "interrupted",
        "planned_runs": [runner.asdict(item) for item in plan],
        "mjx_rows": mjx_rows,
        "replay_rows": replay_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acceptance_report.partial.json").write_text(json.dumps(partial))
    return mjx_rows, replay_rows


def _mjx_metrics_payload(planned, *, motion: str | None = None) -> dict[str, object]:
    if motion is None:
        motion = str(
            Path(planned.mjx_argv[planned.mjx_argv.index("--motion") + 1]).resolve()
        )
    return {
        "method": "g1_wbc_joint_global",
        "motion": motion,
        "device": "cuda:0",
        "checkpoint": planned.mjx_argv[planned.mjx_argv.index("--checkpoint") + 1],
        "max_steps": 800,
        "metrics": {
            **_metrics(success=True),
            "control_dt_sec": 0.02,
            "evaluated_motion_duration_sec": 16.0,
        },
        "mpc": {
            "backend": "mjx",
            "mpc_backend": "mjx",
            "mpc_optimizer": "generic",
            "accepted": True,
            "accepted_windows": 40,
            "num_windows": 40,
            "used_baseline_fallback": False,
            "compile_init_wall_time_sec": 2.0,
            "jit_warmup_enabled": True,
            "jit_warmup_wall_time_sec": 1.5,
            "physics_scan_enabled": True,
            "physics_step_count_min": 40,
            "physics_step_count_max": 40,
            "physics_step_count_windows": 40,
            "steady_state_wall_time_sec": 1.0,
            "runtime_visible_devices": ["0"],
            **_mjx_contact_evidence(),
        },
    }


def _replay_metrics_payload(planned) -> dict[str, object]:
    saved_command = Path(planned.output_dir) / "mpc_command.npz"
    return {
        "method": "replay_command",
        "motion": str(
            Path(
                planned.replay_argv[planned.replay_argv.index("--motion") + 1]
            ).resolve()
        ),
        "device": "cuda:0",
        "checkpoint": planned.replay_argv[planned.replay_argv.index("--checkpoint") + 1],
        "max_steps": 800,
        "metrics": _metrics(success=True),
        "mpc": {
            "backend": (
                "spider.tasks.g1_wbc.spider_task."
                "G1WbcSamplingTask.replay_qpos_command_sequence"
            ),
            "saved_command": str(saved_command.resolve()),
            "saved_command_sha256": _file_sha256(saved_command),
            "replay_mode": "shared_execute_backend",
            "control_steps": 20,
            "num_command_frames": 801,
            "num_replay_steps": 800,
            "runtime_visible_devices": ["0"],
            "runtime_gpu_name": "NVIDIA H100 80GB HBM3",
        },
    }


def _valid_rollout_arrays() -> dict[str, np.ndarray]:
    frames = 801
    steps = 800
    return {
        "qpos": np.zeros((frames, 1, 36), dtype=np.float32),
        "qvel": np.zeros((frames, 1, 35), dtype=np.float32),
        "body_pos_w": np.zeros((frames, 1, 30, 3), dtype=np.float32),
        "body_quat_w": np.zeros((frames, 1, 30, 4), dtype=np.float32),
        "body_lin_vel_w": np.zeros((frames, 1, 30, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((frames, 1, 30, 3), dtype=np.float32),
        "actions": np.zeros((steps, 1, 29), dtype=np.float32),
        "controls": np.zeros((steps, 1, 29), dtype=np.float32),
        "contact_indicator": np.zeros((frames, 1, 2), dtype=np.float32),
        "contact_force": np.zeros((frames, 1, 2), dtype=np.float32),
        "floor_contact_indicator": np.zeros((frames, 1, 3), dtype=np.float32),
        "floor_contact_force": np.zeros((frames, 1, 3), dtype=np.float32),
        "ref_indices": np.zeros((frames, 1), dtype=np.int64),
        "dt": np.array(0.02, dtype=np.float32),
    }


def _valid_command_arrays() -> dict[str, np.ndarray]:
    frames = 801
    return {
        "refined_qpos": np.zeros((frames, 36), dtype=np.float32),
        "candidate_scores": np.zeros((1,), dtype=np.float32),
        "command_joint_pos": np.zeros((frames, 1, 29), dtype=np.float32),
        "command_joint_vel": np.zeros((frames, 1, 29), dtype=np.float32),
        "command_body_pos_w": np.zeros((frames, 1, 30, 3), dtype=np.float32),
        "command_body_quat_w": np.zeros((frames, 1, 30, 4), dtype=np.float32),
        "command_body_lin_vel_w": np.zeros((frames, 1, 30, 3), dtype=np.float32),
        "command_body_ang_vel_w": np.zeros((frames, 1, 30, 3), dtype=np.float32),
        "command_qpos_trajectory": np.zeros((frames, 1, 36), dtype=np.float32),
        "command_qvel_trajectory": np.zeros((frames, 1, 35), dtype=np.float32),
    }


def _acceptance_rows_with_artifacts(
    root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    mjx_rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []
    for motion in ("jump", "walk"):
        for seed in (0, 1, 2):
            output_dir = root / "acceptance" / motion / f"seed_{seed}" / "mjx"
            replay_output_dir = root / "acceptance" / motion / f"seed_{seed}" / "replay"
            _write_artifacts(output_dir)
            _write_artifacts(replay_output_dir, include_command=False)
            mjx_artifacts = {
                "metrics_json": str(output_dir / "metrics.json"),
                "rollout_npz": str(output_dir / "rollout.npz"),
                "mpc_command_npz": str(output_dir / "mpc_command.npz"),
            }
            replay_artifacts = {
                "metrics_json": str(replay_output_dir / "metrics.json"),
                "rollout_npz": str(replay_output_dir / "rollout.npz"),
            }
            replay_argv = [
                "python",
                "--saved-command",
                str(output_dir / "mpc_command.npz"),
                "--replay-control-steps",
                "20",
            ]
            mjx_rows.append(
                {
                    "motion": motion,
                    "seed": seed,
                    "status": "ok",
                    "returncode": 0,
                    "metrics": _metrics(success=True),
                    "mpc_accepted": True,
                    "accepted_windows": 40,
                    "mpc_used_baseline_fallback": False,
                    "num_steps": 800,
                    "compile_init_wall_time_sec": 2.0,
                    "jit_warmup_enabled": True,
                    "jit_warmup_wall_time_sec": 1.5,
                    "runtime_visible_devices": ("0",),
                    "steady_state_wall_time_sec": 1.0,
                    "artifacts": mjx_artifacts,
                    "artifact_sha256": {
                        key: _file_sha256(Path(path))
                        for key, path in mjx_artifacts.items()
                    },
                    **_mjx_contact_evidence(),
                }
            )
            replay_rows.append(
                {
                    "motion": motion,
                    "seed": seed,
                    "status": "ok",
                    "returncode": 0,
                    "metrics": _metrics(success=True),
                    "num_steps": 800,
                    "artifacts": replay_artifacts,
                    "artifact_sha256": {
                        key: _file_sha256(Path(path))
                        for key, path in replay_artifacts.items()
                    },
                    "replay_argv": replay_argv,
                    **_replay_evidence(replay_argv),
                }
            )
    return mjx_rows, replay_rows


def _mjx_contact_evidence(
    *,
    contact_saturated: bool = False,
    max_contact_points_saturated: bool = False,
    max_geom_pairs_saturated: bool = False,
    max_contact_points: int = 512,
    max_geom_pairs: int = 1024,
    contact_pair_count: int = 0,
    active_contact_count: int = 0,
) -> dict[str, int | bool | str]:
    return {
        "runtime_gpu_name": "NVIDIA H100 80GB HBM3",
        "contact_saturated": contact_saturated,
        "max_contact_points_saturated": max_contact_points_saturated,
        "max_geom_pairs_saturated": max_geom_pairs_saturated,
        "max_contact_points": max_contact_points,
        "max_geom_pairs": max_geom_pairs,
        "contact_pair_count": contact_pair_count,
        "active_contact_count": active_contact_count,
    }


def _fresh_metrics_provenance(
    argv: list[str],
    *,
    is_replay: bool,
) -> dict[str, object]:
    payload = {
        "metrics_method": _test_argv_value(argv, "--method"),
        "metrics_motion": str(Path(_test_argv_value(argv, "--motion")).resolve()),
        "metrics_device": _test_argv_value(argv, "--device"),
        "metrics_checkpoint": _test_argv_value(argv, "--checkpoint"),
        "metrics_max_steps": int(_test_argv_value(argv, "--max-steps")),
    }
    if is_replay:
        return {**payload, "mpc": _replay_evidence(argv)["mpc"]}
    return {
        **payload,
        "mpc": {
            "mpc_backend": "mjx",
            "mpc_optimizer": "generic",
            "accepted": True,
            "accepted_windows": 40,
            "num_windows": 40,
            "used_baseline_fallback": False,
            "compile_init_wall_time_sec": 2.0,
            "jit_warmup_enabled": True,
            "jit_warmup_wall_time_sec": 1.5,
            "physics_scan_enabled": True,
            "physics_step_count_min": 40,
            "physics_step_count_max": 40,
            "physics_step_count_windows": 40,
            "steady_state_wall_time_sec": 1.0,
        },
    }


def _test_argv_value(argv: list[str], flag: str) -> str:
    return str(argv[argv.index(flag) + 1])


def _replay_evidence(
    argv: list[str],
    *,
    saved_command: str | None = None,
    control_steps: int = 20,
    num_command_frames: int = 801,
    num_replay_steps: int = 800,
    runtime_gpu_name: str = "NVIDIA H100 80GB HBM3",
) -> dict[str, object]:
    if saved_command is None:
        saved_command = argv[argv.index("--saved-command") + 1]
    return {
        "command_wall_time_sec": 1.0,
        "metrics_method": "replay_command",
        "mpc": {
            "backend": (
                "spider.tasks.g1_wbc.spider_task."
                "G1WbcSamplingTask.replay_qpos_command_sequence"
            ),
            "saved_command": str(Path(saved_command).expanduser().resolve()),
            "saved_command_sha256": _file_sha256(Path(saved_command)),
            "replay_mode": "shared_execute_backend",
            "control_steps": control_steps,
            "num_command_frames": num_command_frames,
            "num_replay_steps": num_replay_steps,
            "runtime_visible_devices": ["0"],
            "runtime_gpu_name": runtime_gpu_name,
        },
    }


if __name__ == "__main__":
    unittest.main()
