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
                        reward_weight_source=str(reward_weights),
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
            _write_stage0_runner_provenance_sidecar(
                output_dir,
                motion_name=motion,
                motion=str(motion_path),
                seed=seed,
            )
    manifest = {
        "schema_version": 1,
        "baseline_name": "g1_wbc_stage0_mujoco_warp_sweetpoint",
        "contact_force_semantics": "pyramidal_contact_normal_v1",
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


def _write_stage0_runner_provenance_sidecar(
    output_dir: Path,
    *,
    motion_name: str,
    motion: str,
    seed: int,
) -> None:
    (output_dir / "stage0_runner_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "g1_wbc_stage0_runner_provenance",
                "motion_name": motion_name,
                "motion": motion,
                "seed": seed,
                "output_dir": str(output_dir),
                "contact_force_semantics": "pyramidal_contact_normal_v1",
            }
        )
    )


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


def _metrics(
    *,
    success: bool,
    contact_force_active_mean: float = 60.0,
    contact_force_peak: float = 400.0,
) -> dict[str, float | bool]:
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
        "contact_force_active_mean": contact_force_active_mean,
        "contact_force_peak": contact_force_peak,
    }


def _baseline_metrics_payload(
    *,
    motion: str,
    checkpoint: str,
    reward_weight_source: str,
    metrics: dict[str, float | bool],
    accepted: bool = True,
    seed: int = 0,
) -> dict[str, object]:
    return {
        "method": "g1_wbc_joint_global",
        "motion": motion,
        "motion_type": "isaaclab",
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
            "reward_weight_source": reward_weight_source,
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
                self.assertEqual(
                    item.mjx_argv[item.mjx_argv.index("--mjx-impl") + 1],
                    "jax",
                )
                self.assertEqual(
                    item.mjx_argv[item.mjx_argv.index("--mjx-warp-naconmax") + 1],
                    "30000",
                )
                self.assertEqual(
                    item.mjx_argv[item.mjx_argv.index("--mjx-warp-njmax") + 1],
                    "256",
                )
                self.assertIn("--no-mjx-guided-candidate", item.mjx_argv)
                self.assertNotIn("--mjx-guided-candidate", item.mjx_argv)
                self.assertEqual(
                    item.mjx_argv[item.mjx_argv.index("--mpc-elite-frac") + 1],
                    "0.125",
                )
                self.assertEqual(
                    item.mjx_argv[
                        item.mjx_argv.index("--mpc-first-ctrl-noise-scale") + 1
                    ],
                    "1.0",
                )
                self.assertEqual(
                    item.mjx_argv[
                        item.mjx_argv.index("--mpc-last-ctrl-noise-scale") + 1
                    ],
                    "1.0",
                )
                self.assertEqual(
                    item.mjx_argv[item.mjx_argv.index("--mpc-final-noise-scale") + 1],
                    "1.0",
                )
                self.assertEqual(
                    item.mjx_argv[item.mjx_argv.index("--mpc-sigma-decay") + 1],
                    "0.75",
                )
                self.assertNotIn("--mjx-guided-candidate", item.replay_argv)
                self.assertNotIn("--no-mjx-guided-candidate", item.replay_argv)
                self.assertNotIn("--mjx-impl", item.replay_argv)
                self.assertNotIn("--mjx-warp-naconmax", item.replay_argv)
                self.assertNotIn("--mjx-warp-njmax", item.replay_argv)
                self.assertNotIn(
                    "--mjx-contact-force-first-row-diagnostics",
                    item.mjx_argv,
                )
                self.assertNotIn(
                    "--mjx-contact-force-first-row-diagnostics",
                    item.replay_argv,
                )
                self.assertNotIn("--mjx-strip-live-mjx-data", item.mjx_argv)
                self.assertNotIn("--mjx-strip-live-mjx-data", item.replay_argv)
                self.assertNotIn("--mjx-model-iterations", item.mjx_argv)
                self.assertNotIn("--mjx-model-ls-iterations", item.mjx_argv)
                self.assertNotIn("--mjx-model-iterations", item.replay_argv)
                self.assertNotIn("--mjx-model-ls-iterations", item.replay_argv)
                self.assertNotIn("--mpc-sigma-decay", item.replay_argv)
                self.assertEqual(
                    item.replay_argv[
                        item.replay_argv.index("--mpc-first-ctrl-noise-scale") + 1
                    ],
                    "1.0",
                )
                self.assertEqual(
                    item.replay_argv[
                        item.replay_argv.index("--mpc-last-ctrl-noise-scale") + 1
                    ],
                    "1.0",
                )
                self.assertEqual(
                    item.replay_argv[
                        item.replay_argv.index("--mpc-final-noise-scale") + 1
                    ],
                    "1.0",
                )
                for flag in (
                    "--mpc-preset",
                    "--mpc-sampling-mode",
                    "--mpc-smooth-passes",
                    "--mpc-command-reg-weight",
                    "--mpc-command-smooth-weight",
                    "--mpc-guided-candidate",
                    "--mpc-acceptance-gate",
                    "--mpc-warm-start-source",
                    "--mpc-warm-start-decay",
                ):
                    self.assertNotIn(flag, item.mjx_argv)
                    self.assertNotIn(flag, item.replay_argv)
                self.assertIn("--no-mpc-warm-start", item.mjx_argv)
                self.assertIn("--no-mpc-warm-start", item.replay_argv)
                replay_backend_idx = item.replay_argv.index("--mpc-backend")
                self.assertEqual(item.replay_argv[replay_backend_idx + 1], "mujoco_warp")
                self.assertNotIn("--mjx-enable-scan", item.replay_argv)
                method_idx = item.replay_argv.index("--method")
                self.assertEqual(item.replay_argv[method_idx + 1], "replay_command")

    def test_build_acceptance_plan_can_enable_first_row_force_diagnostic(self) -> None:
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
                    "--mjx-contact-force-first-row-diagnostics",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        for item in plan:
            self.assertIn(
                "--mjx-contact-force-first-row-diagnostics",
                item.mjx_argv,
            )
            self.assertNotIn(
                "--mjx-contact-force-first-row-diagnostics",
                item.replay_argv,
            )

    def test_build_acceptance_plan_can_set_contact_force_mode(self) -> None:
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
                    "--mjx-contact-force-mode",
                    "first_row",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        for item in plan:
            self.assertIn("--mjx-contact-force-mode", item.mjx_argv)
            mode_idx = item.mjx_argv.index("--mjx-contact-force-mode")
            self.assertEqual(item.mjx_argv[mode_idx + 1], "first_row")
            self.assertNotIn("--mjx-contact-force-mode", item.replay_argv)

    def test_build_acceptance_plan_omits_default_contact_force_mode(self) -> None:
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
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        for item in plan:
            self.assertNotIn("--mjx-contact-force-mode", item.mjx_argv)
            self.assertNotIn("--mjx-contact-force-mode", item.replay_argv)

    def test_build_acceptance_plan_can_set_contact_force_active_weight(self) -> None:
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
                    "--mjx-contact-force-active-weight",
                    "0.75",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        for item in plan:
            self.assertIn("--mjx-contact-force-active-weight", item.mjx_argv)
            weight_idx = item.mjx_argv.index("--mjx-contact-force-active-weight")
            self.assertEqual(item.mjx_argv[weight_idx + 1], "0.75")
            self.assertNotIn("--mjx-contact-force-active-weight", item.replay_argv)

    def test_build_acceptance_plan_can_set_contact_force_peak_excess_weight(
        self,
    ) -> None:
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
                    "--mjx-contact-force-peak-excess-weight",
                    "1.25",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        for item in plan:
            self.assertIn("--mjx-contact-force-peak-excess-weight", item.mjx_argv)
            weight_idx = item.mjx_argv.index(
                "--mjx-contact-force-peak-excess-weight"
            )
            self.assertEqual(item.mjx_argv[weight_idx + 1], "1.25")
            self.assertNotIn(
                "--mjx-contact-force-peak-excess-weight",
                item.replay_argv,
            )

    def test_build_acceptance_plan_can_override_contact_force_delta_weight(
        self,
    ) -> None:
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
                    "--mjx-contact-force-delta-weight",
                    "0.0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        for item in plan:
            self.assertIn("--mjx-contact-force-delta-weight", item.mjx_argv)
            weight_idx = item.mjx_argv.index("--mjx-contact-force-delta-weight")
            self.assertEqual(item.mjx_argv[weight_idx + 1], "0.0")
            self.assertNotIn("--mjx-contact-force-delta-weight", item.replay_argv)

    def test_build_acceptance_plan_can_override_contact_false_positive_weight(
        self,
    ) -> None:
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
                    "--mjx-contact-false-positive-weight",
                    "1.2",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        for item in plan:
            self.assertIn("--mjx-contact-false-positive-weight", item.mjx_argv)
            weight_idx = item.mjx_argv.index(
                "--mjx-contact-false-positive-weight"
            )
            self.assertEqual(item.mjx_argv[weight_idx + 1], "1.2")
            self.assertNotIn(
                "--mjx-contact-false-positive-weight",
                item.replay_argv,
            )

    def test_build_acceptance_plan_can_enable_strip_live_mjx_data(self) -> None:
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
                    "--mjx-strip-live-mjx-data",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        for item in plan:
            self.assertIn("--mjx-strip-live-mjx-data", item.mjx_argv)
            self.assertNotIn("--mjx-strip-live-mjx-data", item.replay_argv)

    def test_build_acceptance_plan_can_enable_score_only_optimizer(self) -> None:
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
                    "--mjx-score-only-optimizer",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        for item in plan:
            self.assertIn("--mjx-score-only-optimizer", item.mjx_argv)
            self.assertNotIn("--mjx-score-only-optimizer", item.replay_argv)

    def test_build_acceptance_plan_can_enable_score_only_rescore_diagnostics(
        self,
    ) -> None:
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
                    "--mjx-score-only-rescore-diagnostics",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        for item in plan:
            self.assertIn("--mjx-score-only-rescore-diagnostics", item.mjx_argv)
            self.assertNotIn(
                "--mjx-score-only-rescore-diagnostics",
                item.replay_argv,
            )

    def test_build_acceptance_plan_can_enable_score_only_output_rescore_diagnostics(
        self,
    ) -> None:
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
                    "--mjx-score-only-output-rescore-diagnostics",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        for item in plan:
            self.assertIn(
                "--mjx-score-only-output-rescore-diagnostics",
                item.mjx_argv,
            )
            self.assertNotIn(
                "--mjx-score-only-output-rescore-diagnostics",
                item.replay_argv,
            )

    def test_build_acceptance_plan_preserves_collision_profile(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            for row in manifest["rows"]:
                row["argv"].extend(
                    ["--collision-profile", "wxy_explicit_pairs_7caps"]
                )
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                    "--device",
                    "cuda:0",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )

            plan = runner.build_acceptance_plan(args, manifest)

        self.assertEqual(len(plan), 1)
        item = plan[0]
        self.assertEqual(
            item.mjx_argv[item.mjx_argv.index("--collision-profile") + 1],
            "wxy_explicit_pairs_7caps",
        )
        self.assertEqual(
            item.replay_argv[item.replay_argv.index("--collision-profile") + 1],
            "wxy_explicit_pairs_7caps",
        )

    def test_build_acceptance_plan_can_override_collision_profile(self) -> None:
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
                    "--collision-profile",
                    "wxy_explicit_floor_leg_pairs_7caps",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        self.assertEqual(len(plan), 1)
        item = plan[0]
        self.assertEqual(
            item.mjx_argv[item.mjx_argv.index("--collision-profile") + 1],
            "wxy_explicit_floor_leg_pairs_7caps",
        )
        self.assertEqual(
            item.replay_argv[item.replay_argv.index("--collision-profile") + 1],
            "wxy_explicit_floor_leg_pairs_7caps",
        )

    def test_build_acceptance_plan_can_enable_mjx_guided_candidate(self) -> None:
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
                    "--mjx-guided-candidate",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        self.assertEqual(len(plan), 6)
        for item in plan:
            with self.subTest(motion=item.motion, seed=item.seed):
                self.assertIn("--mjx-guided-candidate", item.mjx_argv)
                self.assertNotIn("--no-mjx-guided-candidate", item.mjx_argv)
                self.assertNotIn("--mjx-guided-candidate", item.replay_argv)
                self.assertNotIn("--no-mjx-guided-candidate", item.replay_argv)

    def test_build_acceptance_plan_can_enable_periodic_mjx_guided_candidate(
        self,
    ) -> None:
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
                    "--mjx-guided-candidate",
                    "--mjx-guided-candidate-period",
                    "5",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        self.assertEqual(len(plan), 6)
        for item in plan:
            with self.subTest(motion=item.motion, seed=item.seed):
                self.assertIn("--mjx-guided-candidate", item.mjx_argv)
                self.assertEqual(
                    item.mjx_argv[
                        item.mjx_argv.index("--mjx-guided-candidate-period") + 1
                    ],
                    "5",
                )
                self.assertNotIn("--mjx-guided-candidate", item.replay_argv)
                self.assertNotIn("--no-mjx-guided-candidate", item.replay_argv)
                self.assertNotIn("--mjx-guided-candidate-period", item.replay_argv)

    def test_build_acceptance_plan_can_override_guided_candidate_period_by_row(
        self,
    ) -> None:
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
                    "--mjx-guided-candidate-period-override",
                    "jump:0:4",
                    "--mjx-guided-candidate-period-override",
                    "walk:1:8",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        by_row = {(item.motion, item.seed): item for item in plan}
        self.assertIn("--mjx-guided-candidate", by_row[("jump", 0)].mjx_argv)
        self.assertEqual(
            by_row[("jump", 0)].mjx_argv[
                by_row[("jump", 0)].mjx_argv.index(
                    "--mjx-guided-candidate-period"
                )
                + 1
            ],
            "4",
        )
        self.assertIn("--mjx-guided-candidate", by_row[("walk", 1)].mjx_argv)
        self.assertEqual(
            by_row[("walk", 1)].mjx_argv[
                by_row[("walk", 1)].mjx_argv.index(
                    "--mjx-guided-candidate-period"
                )
                + 1
            ],
            "8",
        )
        self.assertIn("--no-mjx-guided-candidate", by_row[("jump", 1)].mjx_argv)
        self.assertNotIn("--mjx-guided-candidate-period", by_row[("jump", 1)].mjx_argv)
        for item in plan:
            self.assertNotIn("--mjx-guided-candidate", item.replay_argv)
            self.assertNotIn("--no-mjx-guided-candidate", item.replay_argv)
            self.assertNotIn("--mjx-guided-candidate-period", item.replay_argv)

    def test_build_acceptance_plan_can_set_mjx_min_score_improvement(self) -> None:
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
                    "--mjx-min-score-improvement",
                    "0.01",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        self.assertEqual(len(plan), 6)
        for item in plan:
            with self.subTest(motion=item.motion, seed=item.seed):
                self.assertEqual(
                    item.mjx_argv[
                        item.mjx_argv.index("--mjx-min-score-improvement") + 1
                    ],
                    "0.01",
                )
                self.assertNotIn("--mjx-min-score-improvement", item.replay_argv)

    def test_build_acceptance_plan_can_override_mjx_min_score_improvement_by_row(
        self,
    ) -> None:
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
                    "--mjx-min-score-improvement-override",
                    "jump:2:0.01",
                    "--mjx-min-score-improvement-override",
                    "walk:0:0.02",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        by_row = {(item.motion, item.seed): item for item in plan}
        self.assertEqual(
            by_row[("jump", 2)].mjx_argv[
                by_row[("jump", 2)].mjx_argv.index("--mjx-min-score-improvement")
                + 1
            ],
            "0.01",
        )
        self.assertEqual(
            by_row[("walk", 0)].mjx_argv[
                by_row[("walk", 0)].mjx_argv.index("--mjx-min-score-improvement")
                + 1
            ],
            "0.02",
        )
        self.assertNotIn("--mjx-min-score-improvement", by_row[("jump", 0)].mjx_argv)
        for item in plan:
            self.assertNotIn("--mjx-min-score-improvement", item.replay_argv)

    def test_build_acceptance_plan_can_set_mjx_top_score_gap_guards(self) -> None:
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
                    "--mjx-min-top-score-gap",
                    "0.02",
                    "--mjx-cem-update-min-top-score-gap",
                    "0.015",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        self.assertEqual(len(plan), 6)
        for item in plan:
            with self.subTest(motion=item.motion, seed=item.seed):
                self.assertEqual(
                    item.mjx_argv[item.mjx_argv.index("--mjx-min-top-score-gap") + 1],
                    "0.02",
                )
                self.assertEqual(
                    item.mjx_argv[
                        item.mjx_argv.index("--mjx-cem-update-min-top-score-gap")
                        + 1
                    ],
                    "0.015",
                )
                self.assertNotIn("--mjx-min-top-score-gap", item.replay_argv)
                self.assertNotIn(
                    "--mjx-cem-update-min-top-score-gap",
                    item.replay_argv,
                )

    def test_build_acceptance_plan_can_override_mjx_min_top_score_gap_by_row(
        self,
    ) -> None:
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
                    "--mjx-min-top-score-gap-override",
                    "jump:0:0.03",
                    "--mjx-min-top-score-gap-override",
                    "walk:2:0.01",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        by_row = {(item.motion, item.seed): item for item in plan}
        self.assertEqual(
            by_row[("jump", 0)].mjx_argv[
                by_row[("jump", 0)].mjx_argv.index("--mjx-min-top-score-gap")
                + 1
            ],
            "0.03",
        )
        self.assertEqual(
            by_row[("walk", 2)].mjx_argv[
                by_row[("walk", 2)].mjx_argv.index("--mjx-min-top-score-gap")
                + 1
            ],
            "0.01",
        )
        self.assertNotIn("--mjx-min-top-score-gap", by_row[("jump", 1)].mjx_argv)
        for item in plan:
            self.assertNotIn("--mjx-min-top-score-gap", item.replay_argv)

    def test_build_acceptance_plan_can_set_mjx_max_control_delta(self) -> None:
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
                    "--mjx-max-control-delta",
                    "0.25",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        self.assertEqual(len(plan), 6)
        for item in plan:
            with self.subTest(motion=item.motion, seed=item.seed):
                self.assertEqual(
                    item.mjx_argv[item.mjx_argv.index("--mjx-max-control-delta") + 1],
                    "0.25",
                )
                self.assertNotIn("--mjx-max-control-delta", item.replay_argv)

    def test_build_acceptance_plan_can_set_candidate_rank_diagnostics(
        self,
    ) -> None:
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
                    "--mjx-candidate-rank-diagnostics-top-k",
                    "8",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        self.assertEqual(len(plan), 6)
        for item in plan:
            with self.subTest(motion=item.motion, seed=item.seed):
                self.assertEqual(
                    item.mjx_argv[
                        item.mjx_argv.index(
                            "--mjx-candidate-rank-diagnostics-top-k"
                        )
                        + 1
                    ],
                    "8",
                )
                self.assertNotIn(
                    "--mjx-candidate-rank-diagnostics-top-k",
                    item.replay_argv,
                )

    def test_build_acceptance_plan_can_set_candidate_rescore_diagnostics(
        self,
    ) -> None:
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
                    "--mjx-candidate-rescore-diagnostics",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        self.assertEqual(len(plan), 6)
        for item in plan:
            with self.subTest(motion=item.motion, seed=item.seed):
                self.assertIn(
                    "--mjx-candidate-rescore-diagnostics",
                    item.mjx_argv,
                )
                self.assertNotIn(
                    "--mjx-candidate-rescore-diagnostics",
                    item.replay_argv,
                )

    def test_build_acceptance_plan_can_set_candidate_rescore_selection(
        self,
    ) -> None:
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
                    "--mjx-candidate-rescore-selection-top-k",
                    "4",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        self.assertEqual(len(plan), 6)
        for item in plan:
            with self.subTest(motion=item.motion, seed=item.seed):
                self.assertEqual(
                    item.mjx_argv[
                        item.mjx_argv.index(
                            "--mjx-candidate-rescore-selection-top-k"
                        )
                        + 1
                    ],
                    "4",
                )
                self.assertNotIn(
                    "--mjx-candidate-rescore-selection-top-k",
                    item.replay_argv,
                )

    def test_build_acceptance_plan_can_override_mjx_max_control_delta_by_row(
        self,
    ) -> None:
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
                    "--mjx-max-control-delta-override",
                    "walk:1:0.25",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        by_row = {(item.motion, item.seed): item for item in plan}
        self.assertEqual(
            by_row[("walk", 1)].mjx_argv[
                by_row[("walk", 1)].mjx_argv.index("--mjx-max-control-delta") + 1
            ],
            "0.25",
        )
        self.assertNotIn("--mjx-max-control-delta", by_row[("walk", 0)].mjx_argv)
        for item in plan:
            self.assertNotIn("--mjx-max-control-delta", item.replay_argv)

    def test_build_acceptance_plan_can_select_mjx_warp_impl(self) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--mjx-warp-naconmax",
                    "4096",
                    "--mjx-warp-njmax",
                    "512",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        self.assertEqual(len(plan), 1)
        item = plan[0]
        self.assertEqual(item.mjx_argv[item.mjx_argv.index("--mjx-impl") + 1], "warp")
        self.assertEqual(
            item.mjx_argv[item.mjx_argv.index("--mjx-warp-naconmax") + 1],
            "4096",
        )
        self.assertEqual(
            item.mjx_argv[item.mjx_argv.index("--mjx-warp-njmax") + 1],
            "512",
        )
        self.assertNotIn("--mjx-impl", item.replay_argv)
        self.assertNotIn("--mjx-warp-naconmax", item.replay_argv)
        self.assertNotIn("--mjx-warp-njmax", item.replay_argv)

    def test_build_acceptance_plan_can_override_naconmax_by_row(self) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--mjx-warp-naconmax",
                    "30000",
                    "--mjx-warp-naconmax-override",
                    "jump:1:24250",
                    "--mjx-warp-naconmax-override",
                    "walk:0:40000",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        by_row = {(item.motion, item.seed): item for item in plan}
        self.assertEqual(
            by_row[("jump", 1)].mjx_argv[
                by_row[("jump", 1)].mjx_argv.index("--mjx-warp-naconmax") + 1
            ],
            "24250",
        )
        self.assertEqual(
            by_row[("walk", 0)].mjx_argv[
                by_row[("walk", 0)].mjx_argv.index("--mjx-warp-naconmax") + 1
            ],
            "40000",
        )
        self.assertEqual(
            by_row[("jump", 0)].mjx_argv[
                by_row[("jump", 0)].mjx_argv.index("--mjx-warp-naconmax") + 1
            ],
            "30000",
        )
        for item in plan:
            self.assertNotIn("--mjx-warp-naconmax", item.replay_argv)

    def test_build_acceptance_plan_can_override_mjx_mpc_numeric_surface(self) -> None:
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
                    "--mjx-mpc-root-pos-sigma",
                    "0.03",
                    "--mjx-mpc-root-rot-sigma",
                    "0.08",
                    "--mjx-mpc-joint-sigma",
                    "0.12",
                    "--mjx-mpc-first-ctrl-noise-scale",
                    "0.8",
                    "--mjx-mpc-last-ctrl-noise-scale",
                    "0.9",
                    "--mjx-mpc-final-noise-scale",
                    "0.5",
                    "--mjx-mpc-sigma-decay",
                    "0.65",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        self.assertEqual(len(plan), 1)
        item = plan[0]
        expected_mjx = {
            "--mpc-root-pos-sigma": "0.03",
            "--mpc-root-rot-sigma": "0.08",
            "--mpc-joint-sigma": "0.12",
            "--mpc-first-ctrl-noise-scale": "0.8",
            "--mpc-last-ctrl-noise-scale": "0.9",
            "--mpc-final-noise-scale": "0.5",
            "--mpc-sigma-decay": "0.65",
        }
        for flag, value in expected_mjx.items():
            with self.subTest(flag=flag):
                self.assertEqual(
                    item.mjx_argv[item.mjx_argv.index(flag) + 1],
                    value,
                )
        expected_replay = {
            "--mpc-root-pos-sigma": "0.04",
            "--mpc-root-rot-sigma": "0.10",
            "--mpc-joint-sigma": "0.18",
            "--mpc-first-ctrl-noise-scale": "1.0",
            "--mpc-last-ctrl-noise-scale": "1.0",
            "--mpc-final-noise-scale": "1.0",
        }
        for flag, value in expected_replay.items():
            with self.subTest(flag=flag):
                self.assertEqual(
                    item.replay_argv[item.replay_argv.index(flag) + 1],
                    value,
                )
        self.assertNotIn("--mpc-sigma-decay", item.replay_argv)

    def test_build_acceptance_plan_can_override_mjx_sample_count(self) -> None:
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
                    "--mjx-mpc-samples",
                    "256",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        self.assertEqual(len(plan), 1)
        item = plan[0]
        self.assertEqual(
            item.mjx_argv[item.mjx_argv.index("--mpc-samples") + 1],
            "256",
        )
        self.assertEqual(
            item.replay_argv[item.replay_argv.index("--mpc-samples") + 1],
            "512",
        )

    def test_build_acceptance_plan_can_override_mjx_joint_sigma_by_row(self) -> None:
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
                    "--mjx-mpc-joint-sigma",
                    "0.14",
                    "--mjx-mpc-joint-sigma-override",
                    "walk:1:0.12",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        by_row = {(item.motion, item.seed): item for item in plan}
        self.assertEqual(
            by_row[("jump", 0)].mjx_argv[
                by_row[("jump", 0)].mjx_argv.index("--mpc-joint-sigma") + 1
            ],
            "0.14",
        )
        self.assertEqual(
            by_row[("walk", 1)].mjx_argv[
                by_row[("walk", 1)].mjx_argv.index("--mpc-joint-sigma") + 1
            ],
            "0.12",
        )
        for item in plan:
            self.assertEqual(
                item.replay_argv[item.replay_argv.index("--mpc-joint-sigma") + 1],
                "0.18",
            )

    def test_naconmax_row_override_rejects_invalid_values(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            base_args = [
                "--baseline-manifest",
                str(manifest_path),
                "--output-dir",
                str(root / "acceptance"),
                "--device",
                "cuda:0",
                "--mjx-warp-naconmax-override",
            ]
            for value in (
                "jump:1",
                "skip:1:20000",
                "jump:9:20000",
                "jump:1:0",
                "jump:1:abc",
            ):
                with self.subTest(value=value):
                    with self.assertRaises(SystemExit):
                        runner.parse_args([*base_args, value])
            with self.assertRaises(SystemExit):
                runner.parse_args(
                    [
                        *base_args,
                        "jump:1:20000",
                        "--mjx-warp-naconmax-override",
                        "jump:1:24000",
                    ]
                )

    def test_mjx_mpc_numeric_overrides_reject_invalid_values(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            base_args = [
                "--baseline-manifest",
                str(manifest_path),
                "--output-dir",
                str(root / "acceptance"),
                "--device",
                "cuda:0",
            ]
            for flag in (
                "--mjx-mpc-root-pos-sigma",
                "--mjx-mpc-root-rot-sigma",
                "--mjx-mpc-joint-sigma",
                "--mjx-mpc-first-ctrl-noise-scale",
                "--mjx-mpc-last-ctrl-noise-scale",
                "--mjx-mpc-sigma-decay",
            ):
                for value in ("0", "-0.1", "nan"):
                    with self.subTest(flag=flag, value=value):
                        with self.assertRaises(SystemExit):
                            runner.parse_args([*base_args, flag, value])
            for value in ("-0.1", "nan"):
                with self.subTest(flag="--mjx-mpc-final-noise-scale", value=value):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                *base_args,
                                "--mjx-mpc-final-noise-scale",
                                value,
                            ]
                        )
            runner.parse_args(
                [
                    *base_args,
                    "--mjx-mpc-final-noise-scale",
                    "0",
                ]
            )
            for value in ("0", "-1", "nan"):
                with self.subTest(flag="--mjx-mpc-samples", value=value):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                *base_args,
                                "--mjx-mpc-samples",
                                value,
                            ]
                        )

    def test_guided_candidate_period_rejects_invalid_values(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            base_args = [
                "--baseline-manifest",
                str(manifest_path),
                "--output-dir",
                str(root / "acceptance"),
                "--device",
                "cuda:0",
            ]
            with self.assertRaises(SystemExit):
                runner.parse_args(
                    [
                        *base_args,
                        "--mjx-guided-candidate-period",
                        "5",
                    ]
                )
            with self.assertRaises(SystemExit):
                runner.parse_args(
                    [
                        *base_args,
                        "--mjx-guided-candidate",
                        "--mjx-guided-candidate-period",
                        "0",
                    ]
                )
            for flag in (
                "--mjx-guided-candidate-period-override",
                "--mjx-model-iterations-override",
                "--mjx-model-ls-iterations-override",
                "--mjx-mpc-joint-sigma-override",
            ):
                with self.subTest(flag=flag):
                    with self.assertRaises(SystemExit):
                        runner.parse_args([*base_args, flag, "walk:2:0"])

    def test_mjx_min_score_improvement_rejects_invalid_values(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            base_args = [
                "--baseline-manifest",
                str(manifest_path),
                "--output-dir",
                str(root / "acceptance"),
                "--device",
                "cuda:0",
            ]
            for value in ("-0.01", "nan"):
                with self.subTest(value=value):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                *base_args,
                                "--mjx-min-score-improvement",
                                value,
                            ]
                        )
            for value in (
                "jump:2",
                "skip:2:0.01",
                "jump:9:0.01",
                "jump:2:-0.01",
                "jump:2:nan",
                "jump:2:abc",
            ):
                with self.subTest(value=value):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                *base_args,
                                "--mjx-min-score-improvement-override",
                                value,
                            ]
                        )

    def test_mjx_top_score_gap_guards_reject_invalid_values(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            base_args = [
                "--baseline-manifest",
                str(manifest_path),
                "--output-dir",
                str(root / "acceptance"),
                "--device",
                "cuda:0",
            ]
            for flag in (
                "--mjx-min-top-score-gap",
                "--mjx-cem-update-min-top-score-gap",
            ):
                for value in ("-0.01", "nan"):
                    with self.subTest(flag=flag, value=value):
                        with self.assertRaises(SystemExit):
                            runner.parse_args([*base_args, flag, value])
            for value in (
                "jump:2",
                "skip:2:0.03",
                "jump:9:0.03",
                "jump:2:-0.01",
                "jump:2:nan",
                "jump:2:abc",
            ):
                with self.subTest(value=value):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                *base_args,
                                "--mjx-min-top-score-gap-override",
                                value,
                            ]
                        )

    def test_mjx_max_control_delta_rejects_invalid_values(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            base_args = [
                "--baseline-manifest",
                str(manifest_path),
                "--output-dir",
                str(root / "acceptance"),
                "--device",
                "cuda:0",
            ]
            for value in ("0", "-0.01", "nan"):
                with self.subTest(value=value):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                *base_args,
                                "--mjx-max-control-delta",
                                value,
                            ]
                        )
            for value in (
                "jump:2",
                "skip:2:0.25",
                "jump:9:0.25",
                "jump:2:0",
                "jump:2:-0.01",
                "jump:2:nan",
                "jump:2:abc",
            ):
                with self.subTest(value=value):
                    with self.assertRaises(SystemExit):
                        runner.parse_args(
                            [
                                *base_args,
                                "--mjx-max-control-delta-override",
                                value,
                            ]
                        )

    def test_candidate_rank_diagnostics_rejects_invalid_values(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            base_args = [
                "--baseline-manifest",
                str(manifest_path),
                "--output-dir",
                str(root / "acceptance"),
                "--device",
                "cuda:0",
            ]
            with self.assertRaises(SystemExit):
                runner.parse_args(
                    [
                        *base_args,
                        "--mjx-candidate-rank-diagnostics-top-k",
                        "-1",
                    ]
                )

    def test_candidate_rescore_selection_rejects_invalid_values(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            base_args = [
                "--baseline-manifest",
                str(manifest_path),
                "--output-dir",
                str(root / "acceptance"),
                "--device",
                "cuda:0",
            ]
            with self.assertRaises(SystemExit):
                runner.parse_args(
                    [
                        *base_args,
                        "--mjx-candidate-rescore-selection-top-k",
                        "-1",
                    ]
                )

    def test_build_acceptance_plan_can_set_mjx_model_solver_options(self) -> None:
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
                    "--mjx-model-iterations",
                    "4",
                    "--mjx-model-ls-iterations",
                    "10",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )

            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )

        self.assertEqual(len(plan), 1)
        item = plan[0]
        self.assertEqual(
            item.mjx_argv[item.mjx_argv.index("--mjx-model-iterations") + 1],
            "4",
        )
        self.assertEqual(
            item.mjx_argv[item.mjx_argv.index("--mjx-model-ls-iterations") + 1],
            "10",
        )
        self.assertNotIn("--mjx-model-iterations", item.replay_argv)
        self.assertNotIn("--mjx-model-ls-iterations", item.replay_argv)

    def test_build_acceptance_plan_can_override_mjx_model_solver_options_by_row(
        self,
    ) -> None:
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
                    "--mjx-model-iterations",
                    "4",
                    "--mjx-model-ls-iterations",
                    "5",
                    "--mjx-model-iterations-override",
                    "walk:2:5",
                    "--mjx-model-ls-iterations-override",
                    "jump:0:10",
                ]
            )

            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )

        by_row = {(item.motion, item.seed): item for item in plan}
        self.assertEqual(
            by_row[("walk", 2)].mjx_argv[
                by_row[("walk", 2)].mjx_argv.index("--mjx-model-iterations") + 1
            ],
            "5",
        )
        self.assertEqual(
            by_row[("walk", 2)].mjx_argv[
                by_row[("walk", 2)].mjx_argv.index("--mjx-model-ls-iterations")
                + 1
            ],
            "5",
        )
        self.assertEqual(
            by_row[("jump", 0)].mjx_argv[
                by_row[("jump", 0)].mjx_argv.index("--mjx-model-iterations") + 1
            ],
            "4",
        )
        self.assertEqual(
            by_row[("jump", 0)].mjx_argv[
                by_row[("jump", 0)].mjx_argv.index("--mjx-model-ls-iterations")
                + 1
            ],
            "10",
        )
        for item in plan:
            self.assertNotIn("--mjx-model-iterations", item.replay_argv)
            self.assertNotIn("--mjx-model-ls-iterations", item.replay_argv)

    def test_acceptance_runner_rejects_non_positive_mjx_model_solver_options(
        self,
    ) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            base_args = [
                "--baseline-manifest",
                str(manifest_path),
                "--output-dir",
                str(root / "acceptance"),
            ]

            with self.assertRaises(SystemExit):
                runner.parse_args(base_args + ["--mjx-model-iterations", "0"])
            with self.assertRaises(SystemExit):
                runner.parse_args(base_args + ["--mjx-model-ls-iterations", "-1"])

    def test_mjx_metrics_with_wrong_impl_fail_closed(self) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["mjx_impl"] = "jax"
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_collision_profile_fail_closed(self) -> None:
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
                    "--collision-profile",
                    "wxy_explicit_floor_leg_pairs_7caps",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["collision_profile"] = "wxy_parity"
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_contact_false_positive_weight_fail_closed(
        self,
    ) -> None:
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
                    "--mjx-contact-false-positive-weight",
                    "1.2",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["reward_weights"] = {"contact_false_positive": 0.6}
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_sigma_decay_fail_closed(self) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["sigma_decay"] = None
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_numeric_surface_fail_closed(self) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--mjx-mpc-joint-sigma",
                    "0.12",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["joint_sigma"] = 0.18
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_min_score_improvement_fail_closed(self) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--mjx-min-score-improvement",
                    "0.01",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["mjx_min_score_improvement"] = 1.0e-9
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_top_score_gap_guard_fail_closed(self) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--mjx-min-top-score-gap",
                    "0.02",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["mjx_min_top_score_gap"] = 0.01
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_row_top_score_gap_guard_fail_closed(self) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--mjx-min-top-score-gap-override",
                    "jump:0:0.03",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["mjx_min_top_score_gap"] = 0.0
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_cem_update_top_score_gap_guard_fail_closed(
        self,
    ) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--mjx-cem-update-min-top-score-gap",
                    "0.015",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["mjx_cem_update_min_top_score_gap"] = 0.01
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_max_control_delta_fail_closed(self) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--mjx-max-control-delta",
                    "0.25",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["mjx_max_control_delta"] = 0.3
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_candidate_rank_diagnostics_fail_closed(
        self,
    ) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--mjx-candidate-rank-diagnostics-top-k",
                    "8",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["candidate_rank_diagnostics_top_k"] = 4
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_missing_candidate_rescore_diagnostics_fail_closed(
        self,
    ) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--mjx-candidate-rescore-diagnostics",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["mjx_candidate_rescore_diagnostics"] = False
            payload["mpc"]["candidate_rescore_diagnostics_windows"] = 0
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_candidate_rescore_selection_fail_closed(
        self,
    ) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--mjx-candidate-rescore-selection-top-k",
                    "4",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["candidate_rescore_selection_top_k"] = 2
            payload["mpc"]["candidate_rescore_selection_windows"] = 40
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_score_only_optimizer_fail_closed(self) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--mjx-score-only-optimizer",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["score_only_optimizer"] = False
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_score_only_rescore_diagnostics_fail_closed(
        self,
    ) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--mjx-score-only-rescore-diagnostics",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["score_only_rescore_diagnostics"] = False
            payload["mpc"]["score_only_rescore_diagnostics_windows"] = 0
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_score_only_output_rescore_diagnostics_fail_closed(
        self,
    ) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--mjx-score-only-output-rescore-diagnostics",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["score_only_output_rescore_diagnostics"] = False
            payload["mpc"]["score_only_output_rescore_diagnostics_windows"] = 0
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_contact_force_mode_fail_closed(self) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--mjx-contact-force-mode",
                    "first_row",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["contact_force_mode"] = "sum_rows"
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_default_contact_force_mode_rejects_legacy_missing(
        self,
    ) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"].pop("contact_force_mode", None)
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_sample_count_fail_closed(self) -> None:
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
                    "--mjx-impl",
                    "warp",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["sample_count"] = 256
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_wrong_model_options_fail_closed(self) -> None:
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
                    "--mjx-model-iterations",
                    "4",
                    "--mjx-model-ls-iterations",
                    "10",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["mjx_model_options"] = {
                "iterations": 3,
                "ls_iterations": 10,
            }
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_with_unplanned_model_options_fail_closed(self) -> None:
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
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"]["mjx_model_options"] = {"iterations": 4}
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

    def test_mjx_metrics_missing_planned_model_options_fail_closed(self) -> None:
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
                    "--mjx-model-iterations",
                    "4",
                    "--only-motion",
                    "jump",
                    "--only-seed",
                    "0",
                ]
            )
            plan = runner.build_acceptance_plan(
                args,
                json.loads(manifest_path.read_text()),
            )
            payload = _mjx_metrics_payload(plan[0])
            payload["mpc"].pop("mjx_model_options")
            metrics_path = Path(plan[0].output_dir) / "metrics.json"
            metrics_path.parent.mkdir(parents=True)
            metrics_path.write_text(json.dumps(payload))
            parsed = runner._row_from_metrics(metrics_path)

            self.assertFalse(
                runner._acceptance_metrics_provenance_matches(
                    plan[0],
                    kind="mjx",
                    payload=payload,
                    parsed=parsed,
                )
            )

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

    def test_build_acceptance_plan_rejects_missing_force_semantics(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            manifest.pop("contact_force_semantics")
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "contact_force_semantics"):
                runner.build_acceptance_plan(args, manifest)

    def test_build_acceptance_plan_rejects_legacy_force_semantics(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            manifest["contact_force_semantics"] = "first_solver_row_v0"
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "contact_force_semantics"):
                runner.build_acceptance_plan(args, manifest)

    def test_build_acceptance_plan_rejects_legacy_stage0_runner_provenance(
        self,
    ) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            row = manifest["rows"][0]
            provenance_path = Path(row["output_dir"]) / "stage0_runner_provenance.json"
            provenance = json.loads(provenance_path.read_text())
            provenance.pop("contact_force_semantics")
            provenance_path.write_text(json.dumps(provenance))
            args = runner.parse_args(
                [
                    "--baseline-manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(root / "acceptance"),
                ]
            )

            with self.assertRaisesRegex(ValueError, "stage0_runner_provenance"):
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

    def test_build_acceptance_plan_rejects_duplicate_formal_stage0_argv_flags(
        self,
    ) -> None:
        runner = load_runner()
        invalid_cases = (
            ("motion_type", ["--motion-type", "mujoco"]),
            ("samples", ["--mpc-samples", "128"]),
            ("guided_candidate", ["--mpc-guided-candidate"]),
            ("warm_start_conflict", ["--mpc-warm-start"]),
        )
        for name, duplicate in invalid_cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    root = Path(tmp_dir)
                    manifest_path = _baseline_manifest(root)
                    manifest = json.loads(manifest_path.read_text())
                    manifest["rows"][0]["argv"].extend(duplicate)
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
                        "formal Stage 0 sweetpoint",
                    ):
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
                        reward_weight_source=row["argv"][
                            row["argv"].index("--mpc-reward-weights") + 1
                        ],
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

    def test_build_acceptance_plan_rejects_baseline_metrics_provenance_mismatch(
        self,
    ) -> None:
        runner = load_runner()
        invalid_cases = (
            ("missing_motion_type", ("motion_type", None)),
            ("mismatched_motion_type", ("motion_type", "mujoco")),
            ("missing_reward_source", ("reward_weight_source", None)),
            ("mismatched_reward_source", ("reward_weight_source", "stale_reward.json")),
        )
        for name, (field, value) in invalid_cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    root = Path(tmp_dir)
                    manifest_path = _baseline_manifest(root)
                    manifest = json.loads(manifest_path.read_text())
                    row = manifest["rows"][0]
                    metrics_path = Path(row["artifacts"]["metrics_json"])
                    payload = json.loads(metrics_path.read_text())
                    if field == "motion_type":
                        if value is None:
                            payload.pop("motion_type")
                        else:
                            payload["motion_type"] = value
                    else:
                        if value is None:
                            payload["mpc"].pop("reward_weight_source")
                        else:
                            stale_reward = root / value
                            stale_reward.write_text("{}")
                            payload["mpc"]["reward_weight_source"] = str(stale_reward)
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

    def test_build_acceptance_plan_can_select_single_parallel_shard_row(self) -> None:
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
                    "--only-motion",
                    "walk",
                    "--only-seed",
                    "1",
                ]
            )
            manifest = json.loads(manifest_path.read_text())

            plan = runner.build_acceptance_plan(args, manifest)

        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].motion, "walk")
        self.assertEqual(plan[0].seed, 1)

    def test_main_skip_report_writes_row_sidecars_for_parallel_shards(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                out_dir = runner._output_dir_from_argv(argv)
                is_replay = "--saved-command" in argv
                _write_artifacts(out_dir, include_command=not is_replay)
                args = runner.parse_args(
                    [
                        "--baseline-manifest",
                        str(manifest_path),
                        "--output-dir",
                        str(output_dir),
                        "--device",
                        "cuda:0",
                        "--only-motion",
                        "jump",
                        "--only-seed",
                        "0",
                    ]
                )
                plan = runner.build_acceptance_plan(
                    args,
                    json.loads(manifest_path.read_text()),
                )
                planned = plan[0]
                payload = (
                    _replay_metrics_payload(planned)
                    if is_replay
                    else _mjx_metrics_payload(planned)
                )
                (out_dir / "metrics.json").write_text(json.dumps(payload))
                artifact_names = ["metrics.json", "rollout.npz"]
                if not is_replay:
                    artifact_names.append("mpc_command.npz")
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "command_wall_time_sec": 1.0,
                    "command_start_time_ns": min(
                        (out_dir / name).stat().st_mtime_ns for name in artifact_names
                    ) - 1_000_000,
                }
                row.update(runner._row_from_metrics(out_dir / "metrics.json"))
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
                        "--skip-report",
                        "--only-motion",
                        "jump",
                        "--only-seed",
                        "0",
                    ]
                )

            rows = runner.load_existing_acceptance_rows(output_dir)

        self.assertEqual(exit_code, 0)
        self.assertFalse((output_dir / "acceptance_report.json").exists())
        self.assertFalse((output_dir / "acceptance_report.partial.json").exists())
        self.assertEqual(len(rows["mjx"]), 1)
        self.assertEqual(len(rows["replay"]), 1)
        self.assertEqual(
            rows["mjx"][0]["same_seed_baseline_steady_state_wall_time_sec"],
            120.0,
        )
        self.assertEqual(
            rows["mjx"][0]["same_seed_target_steady_state_wall_time_sec"],
            10.0,
        )
        self.assertEqual(rows["mjx"][0]["same_seed_speedup"], 120.0)
        self.assertTrue(rows["mjx"][0]["same_seed_speedup_passed"])
        self.assertEqual(rows["mjx"][0]["realtime_factor"], 16.0)
        self.assertTrue(rows["mjx"][0]["realtime_passed"])

    def test_main_skip_report_fails_same_seed_speed_shortfall(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                out_dir = runner._output_dir_from_argv(argv)
                is_replay = "--saved-command" in argv
                _write_artifacts(out_dir, include_command=not is_replay)
                args = runner.parse_args(
                    [
                        "--baseline-manifest",
                        str(manifest_path),
                        "--output-dir",
                        str(output_dir),
                        "--device",
                        "cuda:0",
                        "--only-motion",
                        "jump",
                        "--only-seed",
                        "0",
                    ]
                )
                planned = runner.build_acceptance_plan(
                    args,
                    json.loads(manifest_path.read_text()),
                )[0]
                payload = (
                    _replay_metrics_payload(planned)
                    if is_replay
                    else _mjx_metrics_payload(planned)
                )
                if not is_replay:
                    payload["mpc"]["steady_state_wall_time_sec"] = 20.0
                (out_dir / "metrics.json").write_text(json.dumps(payload))
                artifact_names = ["metrics.json", "rollout.npz"]
                if not is_replay:
                    artifact_names.append("mpc_command.npz")
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "command_wall_time_sec": 1.0,
                    "command_start_time_ns": min(
                        (out_dir / name).stat().st_mtime_ns for name in artifact_names
                    ) - 1_000_000,
                }
                row.update(runner._row_from_metrics(out_dir / "metrics.json"))
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
                        "--skip-report",
                        "--only-motion",
                        "jump",
                        "--only-seed",
                        "0",
                    ]
                )

            rows = runner.load_existing_acceptance_rows(output_dir)

        self.assertEqual(exit_code, 1)
        self.assertEqual(len(rows["mjx"]), 1)
        self.assertEqual(rows["mjx"][0]["same_seed_speedup"], 6.0)
        self.assertFalse(rows["mjx"][0]["same_seed_speedup_passed"])
        self.assertEqual(rows["mjx"][0]["same_seed_speedup_shortfall_sec"], 10.0)
        self.assertEqual(rows["mjx"][0]["realtime_factor"], 0.8)
        self.assertFalse(rows["mjx"][0]["realtime_passed"])

    def test_main_skip_report_fails_failed_mjx_shard_even_when_process_ok(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            output_dir = root / "acceptance"

            def fake_run_command(argv, *, cwd):
                del cwd
                out_dir = runner._output_dir_from_argv(argv)
                is_replay = "--saved-command" in argv
                _write_artifacts(out_dir, include_command=not is_replay)
                args = runner.parse_args(
                    [
                        "--baseline-manifest",
                        str(manifest_path),
                        "--output-dir",
                        str(output_dir),
                        "--device",
                        "cuda:0",
                        "--only-motion",
                        "jump",
                        "--only-seed",
                        "0",
                    ]
                )
                planned = runner.build_acceptance_plan(
                    args,
                    json.loads(manifest_path.read_text()),
                )[0]
                payload = (
                    _replay_metrics_payload(planned)
                    if is_replay
                    else _mjx_metrics_payload(planned)
                )
                if not is_replay:
                    payload["mpc"]["accepted_windows"] = 39
                    payload["mpc"]["num_windows"] = 39
                (out_dir / "metrics.json").write_text(json.dumps(payload))
                artifact_names = ["metrics.json", "rollout.npz"]
                if not is_replay:
                    artifact_names.append("mpc_command.npz")
                row = {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "status": "ok",
                    "command_wall_time_sec": 1.0,
                    "command_start_time_ns": min(
                        (out_dir / name).stat().st_mtime_ns for name in artifact_names
                    ) - 1_000_000,
                }
                row.update(runner._row_from_metrics(out_dir / "metrics.json"))
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
                        "--skip-report",
                        "--only-motion",
                        "jump",
                        "--only-seed",
                        "0",
                    ]
                )

            rows = runner.load_existing_acceptance_rows(output_dir)

        self.assertEqual(exit_code, 1)
        self.assertFalse((output_dir / "acceptance_report.json").exists())
        self.assertEqual(rows["mjx"][0]["accepted_windows"], 39)

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

    def test_existing_acceptance_row_reuse_rejects_mismatched_mjx_guided_candidate(
        self,
    ) -> None:
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
                    "--mjx-guided-candidate",
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
            payload = _mjx_metrics_payload(planned)
            payload["mpc"]["use_guided_candidate"] = False
            (Path(planned.output_dir) / "metrics.json").write_text(json.dumps(payload))

            row = runner.load_existing_acceptance_row(
                planned,
                kind="mjx",
                existing_rows=mjx_rows,
            )

        self.assertIsNone(row)

    def test_existing_acceptance_row_reuse_rejects_mismatched_guided_period(
        self,
    ) -> None:
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
                    "--mjx-guided-candidate",
                    "--mjx-guided-candidate-period",
                    "5",
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
            payload = _mjx_metrics_payload(planned)
            payload["mpc"]["guided_candidate_period"] = 4
            (Path(planned.output_dir) / "metrics.json").write_text(json.dumps(payload))

            row = runner.load_existing_acceptance_row(
                planned,
                kind="mjx",
                existing_rows=mjx_rows,
            )

        self.assertIsNone(row)

    def test_existing_acceptance_row_reuse_rejects_mismatched_guided_windows(
        self,
    ) -> None:
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
                    "--mjx-guided-candidate",
                    "--mjx-guided-candidate-period",
                    "5",
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
            payload = _mjx_metrics_payload(planned)
            payload["mpc"]["guided_candidate_windows"] = 40
            (Path(planned.output_dir) / "metrics.json").write_text(json.dumps(payload))

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

        self.assertEqual(
            report["contact_force_semantics"],
            runner.CONTACT_FORCE_SEMANTICS,
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
                    "metrics": _metrics(
                        success=True,
                        contact_force_active_mean=65.0 if is_replay else 250.0,
                        contact_force_peak=450.0 if is_replay else 2400.0,
                    ),
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
                    "metrics": _metrics(
                        success=True,
                        contact_force_active_mean=65.0 if is_replay else 250.0,
                        contact_force_peak=450.0 if is_replay else 2400.0,
                    ),
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
                    "metrics": _metrics(
                        success=True,
                        contact_force_active_mean=65.0 if is_replay else 250.0,
                        contact_force_peak=450.0 if is_replay else 2400.0,
                    ),
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

    def test_replay_must_use_window_command_chunks(self) -> None:
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
            bad_row["mpc"]["saved_command_replay_source"] = "stitched_command_trajectory"

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
            "replay_window_command_source",
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
                    "metrics": _metrics(
                        success=True,
                        contact_force_active_mean=65.0 if is_replay else 250.0,
                        contact_force_peak=450.0 if is_replay else 2400.0,
                    ),
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
            report["contact_summary"]["jump"]["baseline"]["force"][
                "contact_force_active_mean"
            ]["values"],
            [60.0, 60.0, 60.0],
        )
        self.assertEqual(
            jump_contact["force"]["contact_force_active_mean"]["values"],
            [250.0, 250.0, 250.0],
        )
        self.assertEqual(
            report["contact_summary"]["jump"]["replay"]["force"][
                "contact_force_active_mean"
            ]["values"],
            [65.0, 65.0, 65.0],
        )
        jump_force_semantics = report["contact_summary"]["jump"][
            "force_semantics"
        ]
        self.assertEqual(
            jump_force_semantics["active"]["force_ratio_high_count"],
            3,
        )
        self.assertEqual(
            jump_force_semantics["active"]["isolated_raw_mjx_force_high_count"],
            3,
        )
        self.assertEqual(
            jump_force_semantics["active"]["shared_force_high_count"],
            0,
        )
        self.assertEqual(
            jump_force_semantics["active"]["replay_force_high_count"],
            0,
        )
        self.assertEqual(
            jump_force_semantics["active"]["semantic_classification_counts"],
            {"isolated_raw_mjx_force_high": 3},
        )
        self.assertEqual(
            jump_force_semantics["peak"]["force_ratio_high_count"],
            3,
        )
        self.assertEqual(
            jump_force_semantics["peak"]["isolated_raw_mjx_force_high_count"],
            3,
        )
        self.assertEqual(
            jump_force_semantics["active"]["mjx_to_baseline"]["values"],
            [250.0 / 60.0, 250.0 / 60.0, 250.0 / 60.0],
        )
        self.assertEqual(
            [
                row["classification"]
                for row in jump_force_semantics["active"]["rows"]
            ],
            ["force_ratio_high", "force_ratio_high", "force_ratio_high"],
        )
        self.assertEqual(
            [
                row["semantic_classification"]
                for row in jump_force_semantics["active"]["rows"]
            ],
            [
                "isolated_raw_mjx_force_high",
                "isolated_raw_mjx_force_high",
                "isolated_raw_mjx_force_high",
            ],
        )
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

    def test_mjx_rollout_ref_indices_must_use_legacy_frame_convention(self) -> None:
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
            frames = int(arrays["ref_indices"].shape[0])
            arrays["ref_indices"] = np.arange(frames, dtype=np.int64).reshape(frames, 1)
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

    def test_mjx_command_npz_requires_window_command_chunks(self) -> None:
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
            arrays.pop("window_starts")
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
            "mpc_command_npz_schema",
            report["motion_results"]["jump"]["mjx_failures"],
        )

    def test_mjx_command_refined_qpos_must_match_command_qpos(self) -> None:
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
            "mpc_command_qpos_mismatch",
            report["motion_results"]["jump"]["mjx_failures"],
        )

    def test_mjx_dynamic_rollout_qpos_may_differ_from_command_qpos(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            mjx_rows, replay_rows = _acceptance_rows_with_artifacts(root)
            mjx_row = next(
                row
                for row in mjx_rows
                if row["motion"] == "jump" and row["seed"] == 1
            )
            rollout_path = Path(mjx_row["artifacts"]["rollout_npz"])
            arrays = _valid_rollout_arrays()
            arrays["qpos"][:, 0, 0] = 0.5
            np.savez_compressed(rollout_path, **arrays)
            mjx_row["artifact_sha256"]["rollout_npz"] = _file_sha256(rollout_path)

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

        self.assertNotIn(
            "mpc_rollout_qpos_mismatch",
            report["motion_results"]["jump"]["mjx_failures"],
        )

    def test_mjx_rollout_must_report_dynamic_execute_trace_source(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            mjx_rows, replay_rows = _acceptance_rows_with_artifacts(root)
            mjx_row = next(
                row
                for row in mjx_rows
                if row["motion"] == "jump" and row["seed"] == 1
            )
            mjx_row["rollout_source"] = "static_qpos_fallback"
            mjx_row["rollout_dynamic_execute_trace"] = False
            mjx_row["execute_trace_chunks"] = 0

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
            "mjx_dynamic_execute_trace",
            report["motion_results"]["jump"]["mjx_failures"],
        )

    def test_mjx_warp_rejects_optimizer_selected_prefix_trace_source(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            mjx_rows, replay_rows = _acceptance_rows_with_artifacts(root)
            mjx_row = next(
                row
                for row in mjx_rows
                if row["motion"] == "jump" and row["seed"] == 1
            )
            mjx_row["mjx_impl"] = "warp"
            mjx_row["execute_trace_source_counts"] = {
                "rollout_tracer": 39,
                "optimizer_selected_prefix": 1,
            }

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
            "mjx_dynamic_execute_trace",
            report["motion_results"]["jump"]["mjx_failures"],
        )

    def test_baseline_command_refined_qpos_may_differ_from_rollout_qpos(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            manifest_path = _baseline_manifest(root)
            manifest = json.loads(manifest_path.read_text())
            baseline_row = next(
                row
                for row in manifest["rows"]
                if row["motion_name"] == "jump" and row["seed"] == 1
            )
            rollout_path = Path(baseline_row["artifacts"]["rollout_npz"])
            arrays = _valid_rollout_arrays()
            arrays["qpos"][:, 0, 0] = 0.5
            np.savez_compressed(rollout_path, **arrays)
            baseline_row["artifact_sha256"]["rollout_npz"] = _file_sha256(rollout_path)
            baseline_row["artifact_mtime_ns"]["rollout_npz"] = rollout_path.stat().st_mtime_ns
            mjx_rows, replay_rows = _acceptance_rows_with_artifacts(root)

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

        self.assertNotIn(
            "mpc_rollout_qpos_mismatch",
            report["motion_results"]["jump"]["baseline_failures"],
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

    def test_command_qvel_consistency_allows_float32_export_noise(self) -> None:
        runner = load_runner()
        with tempfile.TemporaryDirectory() as tmp_dir:
            command_path = Path(tmp_dir) / "mpc_command.npz"
            arrays = _valid_command_arrays()
            arrays["command_qvel_trajectory"][0, 0, 0] = 2.5e-5
            np.savez_compressed(command_path, **arrays)

            consistent = runner._command_qvel_is_consistent(command_path)

        self.assertTrue(consistent)

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

    def test_4090_target_fails_when_speedup_is_below_target(self) -> None:
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
                            "control_dt_sec": 0.1,
                            "evaluated_motion_duration_sec": 80.0,
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
        self.assertIn("speedup", report["speed_results"]["jump"]["failures"])

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
                            "optimizer_result_sync_wall_time_sec": 0.75,
                            "strip_live_mjx_data_between_windows": True,
                            "score_only_optimizer": True,
                            "contact_force_mode": "first_row",
                            "contact_force_first_row_diagnostics": True,
                            "candidate_rank_diagnostics_top_k": 8,
                            "candidate_rank_diagnostics_windows": 40,
                            "mjx_candidate_rescore_diagnostics": True,
                            "candidate_rescore_diagnostics_windows": 40,
                            "candidate_rescore_score_delta_max": 0.004,
                            "candidate_rescore_score_delta_mean": 0.00125,
                            "candidate_rescore_top1_changed_iteration_sum": 7,
                            "candidate_rescore_selection_top_k": 4,
                            "candidate_rescore_selection_windows": 40,
                            "candidate_rescore_selection_score_delta_max": 0.006,
                            "candidate_rescore_selection_score_delta_mean": 0.0025,
                            "candidate_rescore_selection_changed_iteration_sum": 5,
                            "zero_delta_noop_iteration_sum": 2,
                            "zero_delta_noop_windows": 1,
                            "zero_delta_noop_accepted_windows": 1,
                            "score_threshold_noop_iteration_sum": 3,
                            "score_threshold_noop_windows": 2,
                            "score_threshold_noop_accepted_windows": 2,
                            "control_delta_guard_noop_iteration_sum": 4,
                            "control_delta_guard_noop_windows": 2,
                            "control_delta_guard_noop_accepted_windows": 1,
                            "noop_candidate_iteration_sum": 5,
                            "noop_candidate_windows": 3,
                            "noop_candidate_accepted_windows": 3,
                            "iteration_zero_delta_noop_window_counts": [1, 1],
                            "iteration_score_threshold_noop_window_counts": [2, 1],
                            "iteration_control_delta_guard_noop_window_counts": [1, 3],
                            "iteration_noop_candidate_window_counts": [3, 2],
                            "top_score_gap_min": 0.01,
                            "top_score_gap_mean": 0.02,
                            "top_score_gap_max": 0.03,
                            "top_score_gap_windows": 40,
                            "iteration_top_score_gap_mins": [0.01, 0.02],
                            "iteration_top_score_gap_means": [0.015, 0.025],
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
        self.assertEqual(row["optimizer_result_sync_wall_time_sec"], 0.75)
        self.assertTrue(row["strip_live_mjx_data_between_windows"])
        self.assertTrue(row["score_only_optimizer"])
        self.assertEqual(row["contact_force_mode"], "first_row")
        self.assertTrue(row["contact_force_first_row_diagnostics"])
        self.assertEqual(row["candidate_rank_diagnostics_top_k"], 8)
        self.assertEqual(row["candidate_rank_diagnostics_windows"], 40)
        self.assertTrue(row["mjx_candidate_rescore_diagnostics"])
        self.assertEqual(row["candidate_rescore_diagnostics_windows"], 40)
        self.assertEqual(row["candidate_rescore_score_delta_max"], 0.004)
        self.assertEqual(row["candidate_rescore_score_delta_mean"], 0.00125)
        self.assertEqual(row["candidate_rescore_top1_changed_iteration_sum"], 7)
        self.assertEqual(row["candidate_rescore_selection_top_k"], 4)
        self.assertEqual(row["candidate_rescore_selection_windows"], 40)
        self.assertEqual(row["candidate_rescore_selection_score_delta_max"], 0.006)
        self.assertEqual(row["candidate_rescore_selection_score_delta_mean"], 0.0025)
        self.assertEqual(row["candidate_rescore_selection_changed_iteration_sum"], 5)
        self.assertEqual(row["top_score_gap_min"], 0.01)
        self.assertEqual(row["top_score_gap_mean"], 0.02)
        self.assertEqual(row["top_score_gap_max"], 0.03)
        self.assertEqual(row["top_score_gap_windows"], 40)
        self.assertEqual(row["iteration_top_score_gap_mins"], [0.01, 0.02])
        self.assertEqual(row["iteration_top_score_gap_means"], [0.015, 0.025])
        self.assertEqual(row["runtime_visible_devices"], ["0"])
        self.assertEqual(row["runtime_gpu_name"], "NVIDIA H100 80GB HBM3")
        self.assertEqual(row["control_dt_sec"], 0.02)
        self.assertEqual(row["evaluated_motion_duration_sec"], 16.0)
        self.assertEqual(row["zero_delta_noop_iteration_sum"], 2)
        self.assertEqual(row["zero_delta_noop_windows"], 1)
        self.assertEqual(row["zero_delta_noop_accepted_windows"], 1)
        self.assertEqual(row["score_threshold_noop_iteration_sum"], 3)
        self.assertEqual(row["score_threshold_noop_windows"], 2)
        self.assertEqual(row["score_threshold_noop_accepted_windows"], 2)
        self.assertEqual(row["control_delta_guard_noop_iteration_sum"], 4)
        self.assertEqual(row["control_delta_guard_noop_windows"], 2)
        self.assertEqual(row["control_delta_guard_noop_accepted_windows"], 1)
        self.assertEqual(row["noop_candidate_iteration_sum"], 5)
        self.assertEqual(row["noop_candidate_windows"], 3)
        self.assertEqual(row["noop_candidate_accepted_windows"], 3)
        self.assertEqual(row["iteration_zero_delta_noop_window_counts"], [1, 1])
        self.assertEqual(
            row["iteration_score_threshold_noop_window_counts"],
            [2, 1],
        )
        self.assertEqual(
            row["iteration_control_delta_guard_noop_window_counts"],
            [1, 3],
        )
        self.assertEqual(row["iteration_noop_candidate_window_counts"], [3, 2])
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
        self.assertIsNot(row["mpc_used_baseline_fallback"], False)

    def test_metrics_parser_invalid_accepted_windows_fails_closed(self) -> None:
        runner = load_runner()
        invalid_cases = (
            ("missing_with_num_windows", {"num_windows": 40}),
            ("string", {"accepted_windows": "40"}),
            ("bool", {"accepted_windows": True}),
        )
        for name, mpc in invalid_cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    path = Path(tmp_dir) / "metrics.json"
                    payload = {
                        "metrics": _metrics(success=True),
                        "mpc": {
                            "accepted": True,
                            "used_baseline_fallback": False,
                            **mpc,
                        },
                    }
                    path.write_text(json.dumps(payload))

                    row = runner._row_from_metrics(path)

                self.assertNotEqual(row["accepted_windows"], 40)


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
    collision_profile = _test_optional_argv_value(
        planned.mjx_argv,
        "--collision-profile",
        default="wxy_parity",
    )
    return {
        "method": "g1_wbc_joint_global",
        "motion": motion,
        "device": "cuda:0",
        "checkpoint": planned.mjx_argv[planned.mjx_argv.index("--checkpoint") + 1],
        "collision_profile": collision_profile,
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
            "collision_profile": collision_profile,
            "mjx_impl": planned.mjx_argv[planned.mjx_argv.index("--mjx-impl") + 1],
            "mjx_model_impl": planned.mjx_argv[
                planned.mjx_argv.index("--mjx-impl") + 1
            ],
            "mjx_model_options": _test_mjx_model_options(planned.mjx_argv),
            "mjx_warp_naconmax": int(
                planned.mjx_argv[planned.mjx_argv.index("--mjx-warp-naconmax") + 1]
            ),
            "mjx_warp_njmax": int(
                planned.mjx_argv[planned.mjx_argv.index("--mjx-warp-njmax") + 1]
            ),
            **_mjx_search_surface_evidence(planned.mjx_argv),
            "rollout_source": "dynamic_execute_trace",
            "rollout_dynamic_execute_trace": True,
            "execute_trace_chunks": 40,
            "execute_trace_source_counts": {"rollout_tracer": 40},
            "physics_step_count_min": 40,
            "physics_step_count_max": 40,
            "physics_step_count_windows": 40,
            "steady_state_wall_time_sec": 1.0,
            "use_guided_candidate": _test_argv_bool_optional(
                planned.mjx_argv,
                "--mjx-guided-candidate",
            ),
            "guided_candidate_period": _test_optional_argv_int(
                planned.mjx_argv,
                "--mjx-guided-candidate-period",
            ),
            "guided_candidate_windows": _test_expected_guided_candidate_windows(
                planned.mjx_argv
            ),
            "contact_force_first_row_diagnostics": (
                "--mjx-contact-force-first-row-diagnostics" in planned.mjx_argv
            ),
            "strip_live_mjx_data_between_windows": (
                "--mjx-strip-live-mjx-data" in planned.mjx_argv
            ),
            "score_only_optimizer": (
                "--mjx-score-only-optimizer" in planned.mjx_argv
            ),
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
                "G1WbcSamplingTask.replay_command_chunks"
            ),
            "saved_command": str(saved_command.resolve()),
            "saved_command_sha256": _file_sha256(saved_command),
            "replay_mode": "shared_execute_backend",
            "saved_command_replay_source": "window_command_chunks",
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
    ref_indices = np.concatenate(
        (np.array([0], dtype=np.int64), np.arange(steps, dtype=np.int64))
    ).reshape(frames, 1)
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
        "ref_indices": ref_indices,
        "dt": np.array(0.02, dtype=np.float32),
    }


def _valid_command_arrays() -> dict[str, np.ndarray]:
    frames = 801
    windows = 40
    horizon = 40
    starts = np.arange(windows, dtype=np.int32) * 20
    execute_steps = np.full(windows, 20, dtype=np.int32)
    horizons = np.full(windows, horizon, dtype=np.int32)
    return {
        "refined_qpos": np.zeros((frames, 36), dtype=np.float32),
        "candidate_scores": np.zeros((1,), dtype=np.float32),
        "window_command_schema_version": np.array(1, dtype=np.int32),
        "window_starts": starts,
        "window_execute_steps": execute_steps,
        "window_horizons": horizons,
        "window_command_joint_pos": np.zeros((windows, horizon, 1, 29), dtype=np.float32),
        "window_command_joint_vel": np.zeros((windows, horizon, 1, 29), dtype=np.float32),
        "window_command_body_pos_w": np.zeros((windows, horizon, 1, 30, 3), dtype=np.float32),
        "window_command_body_quat_w": np.zeros((windows, horizon, 1, 30, 4), dtype=np.float32),
        "window_command_body_lin_vel_w": np.zeros((windows, horizon, 1, 30, 3), dtype=np.float32),
        "window_command_body_ang_vel_w": np.zeros((windows, horizon, 1, 30, 3), dtype=np.float32),
        "window_command_qpos_chunks": np.zeros((windows, horizon, 1, 36), dtype=np.float32),
        "window_command_qvel_chunks": np.zeros((windows, horizon, 1, 35), dtype=np.float32),
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
                    "rollout_source": "dynamic_execute_trace",
                    "rollout_dynamic_execute_trace": True,
                    "execute_trace_chunks": 40,
                    "execute_trace_source_counts": {"rollout_tracer": 40},
                    "runtime_visible_devices": ("0",),
                    "sigma_decay": 0.75,
                    "mjx_min_score_improvement": 1.0e-9,
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
        "rollout_source": "dynamic_execute_trace",
        "rollout_dynamic_execute_trace": True,
        "execute_trace_chunks": 40,
        "execute_trace_source_counts": {"rollout_tracer": 40},
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
        "collision_profile": _test_optional_argv_value(
            argv,
            "--collision-profile",
            default="wxy_parity",
        ),
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
            "collision_profile": _test_optional_argv_value(
                argv,
                "--collision-profile",
                default="wxy_parity",
            ),
            "mjx_impl": _test_argv_value(argv, "--mjx-impl"),
            "mjx_model_impl": _test_argv_value(argv, "--mjx-impl"),
            "mjx_model_options": _test_mjx_model_options(argv),
            "mjx_warp_naconmax": int(_test_argv_value(argv, "--mjx-warp-naconmax")),
            "mjx_warp_njmax": int(_test_argv_value(argv, "--mjx-warp-njmax")),
            **_mjx_search_surface_evidence(argv),
            "physics_step_count_min": 40,
            "physics_step_count_max": 40,
            "physics_step_count_windows": 40,
            "steady_state_wall_time_sec": 1.0,
            "use_guided_candidate": _test_argv_bool_optional(
                argv,
                "--mjx-guided-candidate",
            ),
            "guided_candidate_period": _test_optional_argv_int(
                argv,
                "--mjx-guided-candidate-period",
            ),
            "guided_candidate_windows": _test_expected_guided_candidate_windows(argv),
        },
    }


def _mjx_search_surface_evidence(argv: list[str]) -> dict[str, object]:
    return {
        "sample_count": int(_test_argv_value(argv, "--mpc-samples")),
        "optimizer_iterations": int(_test_argv_value(argv, "--mpc-iterations")),
        "planning_horizon_steps": int(
            _test_argv_value(argv, "--mpc-planning-horizon-steps")
        ),
        "control_steps": int(_test_argv_value(argv, "--mpc-control-steps")),
        "knot_count": int(_test_argv_value(argv, "--mpc-knot-count")),
        "elite_frac": float(_test_argv_value(argv, "--mpc-elite-frac")),
        "temperature": float(_test_argv_value(argv, "--mpc-temperature")),
        "root_pos_sigma": float(_test_argv_value(argv, "--mpc-root-pos-sigma")),
        "root_rot_sigma": float(_test_argv_value(argv, "--mpc-root-rot-sigma")),
        "joint_sigma": float(_test_argv_value(argv, "--mpc-joint-sigma")),
        "first_ctrl_noise_scale": float(
            _test_argv_value(argv, "--mpc-first-ctrl-noise-scale")
        ),
        "last_ctrl_noise_scale": float(
            _test_argv_value(argv, "--mpc-last-ctrl-noise-scale")
        ),
        "final_noise_scale": float(_test_argv_value(argv, "--mpc-final-noise-scale")),
        "sigma_decay": float(_test_argv_value(argv, "--mpc-sigma-decay")),
        "mjx_min_score_improvement": _test_optional_argv_float(
            argv,
            "--mjx-min-score-improvement",
            default=1.0e-9,
        ),
        "mjx_min_top_score_gap": _test_optional_argv_float(
            argv,
            "--mjx-min-top-score-gap",
            default=0.0,
        ),
        "mjx_cem_update_min_top_score_gap": _test_optional_argv_float(
            argv,
            "--mjx-cem-update-min-top-score-gap",
            default=0.0,
        ),
        "mjx_max_control_delta": _test_optional_argv_float(
            argv,
            "--mjx-max-control-delta",
            default=None,
        ),
        "candidate_rank_diagnostics_top_k": int(
            _test_optional_argv_value(
                argv,
                "--mjx-candidate-rank-diagnostics-top-k",
                default="0",
            )
        ),
        "mjx_candidate_rescore_diagnostics": (
            "--mjx-candidate-rescore-diagnostics" in argv
        ),
        "candidate_rescore_diagnostics_windows": (
            40 if "--mjx-candidate-rescore-diagnostics" in argv else 0
        ),
        "candidate_rescore_selection_top_k": int(
            _test_optional_argv_value(
                argv,
                "--mjx-candidate-rescore-selection-top-k",
                default="0",
            )
        ),
        "candidate_rescore_selection_windows": (
            40
            if "--mjx-candidate-rescore-selection-top-k" in argv
            else 0
        ),
        "contact_force_mode": _test_optional_argv_value(
            argv,
            "--mjx-contact-force-mode",
            default="sum_rows",
        ),
        "use_warm_start": _test_argv_bool_optional(argv, "--mpc-warm-start"),
    }


def _test_argv_value(argv: list[str], flag: str) -> str:
    return str(argv[argv.index(flag) + 1])


def _test_optional_argv_value(
    argv: list[str],
    flag: str,
    *,
    default: str | None = None,
) -> str | None:
    if flag not in argv:
        return default
    return _test_argv_value(argv, flag)


def _test_optional_argv_int(argv: list[str], flag: str) -> int | None:
    if flag not in argv:
        return None
    return int(_test_argv_value(argv, flag))


def _test_optional_argv_float(
    argv: list[str],
    flag: str,
    *,
    default: float | None = None,
) -> float | None:
    if flag not in argv:
        return default
    return float(_test_argv_value(argv, flag))


def _test_expected_guided_candidate_windows(argv: list[str]) -> int:
    if not _test_argv_bool_optional(argv, "--mjx-guided-candidate"):
        return 0
    period = _test_optional_argv_int(argv, "--mjx-guided-candidate-period")
    if period is None:
        return 40
    return (40 + period - 1) // period


def _test_argv_bool_optional(argv: list[str], flag: str) -> bool:
    negative_flag = f"--no-{flag[2:]}"
    positive = flag in argv
    negative = negative_flag in argv
    if positive == negative:
        raise AssertionError(f"expected exactly one of {flag} or {negative_flag}")
    return positive


def _test_mjx_model_options(argv: list[str]) -> dict[str, int]:
    options: dict[str, int] = {}
    if "--mjx-model-iterations" in argv:
        options["iterations"] = int(_test_argv_value(argv, "--mjx-model-iterations"))
    if "--mjx-model-ls-iterations" in argv:
        options["ls_iterations"] = int(
            _test_argv_value(argv, "--mjx-model-ls-iterations")
        )
    return options


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
                "G1WbcSamplingTask.replay_command_chunks"
            ),
            "saved_command": str(Path(saved_command).expanduser().resolve()),
            "saved_command_sha256": _file_sha256(Path(saved_command)),
            "replay_mode": "shared_execute_backend",
            "saved_command_replay_source": "window_command_chunks",
            "control_steps": control_steps,
            "num_command_frames": num_command_frames,
            "num_replay_steps": num_replay_steps,
            "runtime_visible_devices": ["0"],
            "runtime_gpu_name": runtime_gpu_name,
        },
    }


if __name__ == "__main__":
    unittest.main()
