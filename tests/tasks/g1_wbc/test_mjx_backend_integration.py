from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

import numpy as np
import torch

from spider.config import Config
from spider.tasks.g1_wbc import mjx_backend as mjx_backend_module
from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    QPOS_DIM,
    QVEL_DIM,
)
from spider.tasks.g1_wbc.mjx_backend import run_g1_wbc_mjx_mpc
from spider.tasks.g1_wbc.mjx_components import build_mjx_rollout_components
from spider.tasks.g1_wbc.motion import G1Motion, qvel_from_qpos_trajectory
from spider.tasks.g1_wbc.policy import WbcActor
from spider.tasks.g1_wbc.result_types import G1WbcMpcRun


def _motion(frames: int = 801) -> G1Motion:
    bodies = len(MUJOCO_BODY_NAMES)
    qpos = torch.zeros(frames, QPOS_DIM)
    qpos[:, 3] = 1.0
    body_quat = torch.zeros(frames, bodies, 4)
    body_quat[..., 0] = 1.0
    return G1Motion(
        path=None,
        motion_type="mujoco",
        fps=50.0,
        joint_pos=qpos[:, 7:].clone(),
        joint_vel=torch.zeros(frames, ACTION_DIM),
        body_pos_w=torch.zeros(frames, bodies, 3),
        body_quat_w=body_quat,
        body_lin_vel_w=torch.zeros(frames, bodies, 3),
        body_ang_vel_w=torch.zeros(frames, bodies, 3),
        contact=torch.zeros(frames, 2),
    )


def _spider_config() -> Config:
    config = Config(
        device="cpu",
        num_samples=4,
        max_num_iterations=1,
        temperature=0.7,
        pos_noise_scale=0.04,
        rot_noise_scale=0.10,
        joint_noise_scale=0.18,
    )
    config.horizon_steps = 40
    config.ctrl_steps = 20
    config.num_samples = 4
    config.max_num_iterations = 1
    config.num_knot_points = 8
    config.temperature = 0.7
    config.pos_noise_scale = 0.04
    config.rot_noise_scale = 0.10
    config.joint_noise_scale = 0.18
    return config


class MjxBackendIntegrationTest(unittest.TestCase):
    def test_importing_backend_does_not_load_rollout_module(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        code = (
            "import sys; "
            "import spider.tasks.g1_wbc.mjx_backend; "
            "print('rollout_loaded', 'spider.tasks.g1_wbc.rollout' in sys.modules)"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root)

        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("rollout_loaded False", completed.stdout)
        self.assertNotIn("Warp", completed.stdout + completed.stderr)

    def test_injected_fake_success_path_does_not_load_rollout_module(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        code = textwrap.dedent(
            """
            from types import SimpleNamespace
            import sys
            import torch

            from spider.tasks.g1_wbc.constants import (
                ACTION_DIM,
                MUJOCO_BODY_NAMES,
                QPOS_DIM,
                QVEL_DIM,
            )
            from spider.tasks.g1_wbc.mjx_backend import run_g1_wbc_mjx_mpc
            from spider.tasks.g1_wbc.motion import G1Motion

            def motion(frames=801):
                bodies = len(MUJOCO_BODY_NAMES)
                qpos = torch.zeros(frames, QPOS_DIM)
                qpos[:, 3] = 1.0
                body_quat = torch.zeros(frames, bodies, 4)
                body_quat[..., 0] = 1.0
                return G1Motion(
                    path=None,
                    motion_type="mujoco",
                    fps=50.0,
                    joint_pos=qpos[:, 7:].clone(),
                    joint_vel=torch.zeros(frames, ACTION_DIM),
                    body_pos_w=torch.zeros(frames, bodies, 3),
                    body_quat_w=body_quat,
                    body_lin_vel_w=torch.zeros(frames, bodies, 3),
                    body_ang_vel_w=torch.zeros(frames, bodies, 3),
                    contact=torch.zeros(frames, 2),
                )

            def optimizer(**kwargs):
                del kwargs
                updated = torch.zeros(40, QPOS_DIM - 1)
                chunk = torch.zeros(21, QPOS_DIM - 1)
                return SimpleNamespace(
                    updated_controls=updated,
                    execute_chunk=chunk,
                    info={"best_score": torch.tensor(1.0), "accepted": True},
                )

            def rollout_factory(motion, total_steps, *, device, refined_qpos):
                del motion
                frames = int(total_steps) + 1
                bodies = len(MUJOCO_BODY_NAMES)
                qpos = refined_qpos.to(device).view(frames, 1, QPOS_DIM).clone()
                body_quat = torch.zeros(frames, 1, bodies, 4, device=device)
                body_quat[..., 0] = 1.0
                return SimpleNamespace(
                    qpos=qpos,
                    qvel=torch.zeros(frames, 1, QVEL_DIM, device=device),
                    body_pos_w=torch.zeros(frames, 1, bodies, 3, device=device),
                    body_quat_w=body_quat,
                    body_lin_vel_w=torch.zeros(frames, 1, bodies, 3, device=device),
                    body_ang_vel_w=torch.zeros(frames, 1, bodies, 3, device=device),
                    actions=torch.zeros(total_steps, 1, ACTION_DIM, device=device),
                    controls=torch.zeros(total_steps, 1, ACTION_DIM, device=device),
                    contact_indicator=torch.zeros(frames, 1, 2, device=device),
                    contact_force=torch.zeros(frames, 1, 2, device=device),
                    floor_contact_indicator=torch.zeros(frames, 1, 3, device=device),
                    floor_contact_force=torch.zeros(frames, 1, 3, device=device),
                    ref_indices=torch.arange(frames, device=device).view(frames, 1),
                )

            config = SimpleNamespace(
                horizon_steps=40,
                ctrl_steps=20,
                num_samples=4,
                max_num_iterations=1,
                num_knot_points=8,
                temperature=0.7,
                pos_noise_scale=0.04,
                rot_noise_scale=0.10,
                joint_noise_scale=0.18,
            )
            result = run_g1_wbc_mjx_mpc(
                spider_config=config,
                motion=motion(),
                actor=object(),
                rollout_config=SimpleNamespace(device="cpu"),
                execute_rollout_config=SimpleNamespace(device="cpu"),
                method="g1_wbc_joint_global",
                reward_weights=None,
                total_steps=800,
                seed=5,
                runtime=SimpleNamespace(jnp=SimpleNamespace(), jax=SimpleNamespace()),
                model_factory=lambda **kwargs: SimpleNamespace(
                    profile=SimpleNamespace(name=kwargs["profile_name"])
                ),
                policy_converter=lambda actor, *, jnp: SimpleNamespace(params=True),
                optimizer=optimizer,
                rollout_factory=rollout_factory,
            )
            print("accepted", result.metadata["accepted"])
            print("rollout_loaded", "spider.tasks.g1_wbc.rollout" in sys.modules)
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root)

        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("accepted True", completed.stdout)
        self.assertIn("rollout_loaded False", completed.stdout)
        self.assertNotIn("Warp", completed.stdout + completed.stderr)

    def test_mjx_backend_contract_with_injected_fakes(self) -> None:
        calls: list[str] = []

        def fake_model_factory(**kwargs):
            calls.append(f"model:{kwargs['profile_name']}")
            return SimpleNamespace(profile=SimpleNamespace(name="wxy_parity"))

        def fake_policy_converter(actor, *, jnp):
            del actor, jnp
            calls.append("policy")
            return SimpleNamespace(params=True)

        result = run_g1_wbc_mjx_mpc(
            spider_config=_spider_config(),
            motion=_motion(),
            actor=WbcActor(input_dim=4, hidden_dims=(), output_dim=2),
            rollout_config=SimpleNamespace(device="cpu", max_steps=800),
            execute_rollout_config=SimpleNamespace(device="cpu", max_steps=800),
            method="g1_wbc_joint_global",
            reward_weights=None,
            total_steps=800,
            seed=5,
            runtime=SimpleNamespace(jnp=SimpleNamespace(), jax=SimpleNamespace()),
            model_factory=fake_model_factory,
            policy_converter=fake_policy_converter,
            optimizer=_fake_optimizer,
            rollout_factory=_fake_rollout_result,
        )

        self.assertIsInstance(result, G1WbcMpcRun)
        self.assertEqual(result.metadata["backend"], "mjx")
        self.assertTrue(result.metadata["accepted"])
        self.assertFalse(result.metadata["used_baseline_fallback"])
        self.assertEqual(result.result.num_windows, 40)
        self.assertEqual(result.metadata["accepted_windows"], 40)
        self.assertEqual(result.result.rollout.qpos.shape, (801, 1, QPOS_DIM))
        self.assertEqual(result.result.refined_qpos.shape, (801, QPOS_DIM))
        self.assertTrue(
            torch.allclose(
                result.result.refined_qpos[:800, 0],
                torch.full((800,), 0.25),
            )
        )
        self.assertAlmostEqual(float(result.result.refined_qpos[800, 0]), 0.25)
        self.assertAlmostEqual(float(result.result.refined_qpos[800, 1]), 0.0)
        self.assertTrue(
            torch.allclose(
                result.result.command.qpos_trajectory[:800, 0, 0],
                torch.full((800,), 0.25),
            )
        )
        self.assertAlmostEqual(
            float(result.result.command.qpos_trajectory[800, 0, 0]),
            0.25,
        )
        self.assertAlmostEqual(float(result.result.rollout.qpos[800, 0, 0]), 0.25)
        self.assertEqual(result.result.scores.shape, (40,))
        self.assertTrue(torch.allclose(result.result.scores, torch.full((40,), 1.25)))
        self.assertEqual(result.receding.executed_steps, 800)
        self.assertIn("runtime_gpu_name", result.metadata)
        self.assertIsNone(result.metadata["runtime_gpu_name"])
        self.assertIn("model:wxy_parity", calls)
        self.assertIn("policy", calls)

    def test_command_from_refined_qpos_exports_finite_difference_qvel(self) -> None:
        refined_qpos = torch.zeros(3, QPOS_DIM)
        refined_qpos[:, 3] = 1.0
        refined_qpos[1:, 0] = torch.tensor([0.02, 0.04])
        refined_qpos[1:, 7] = torch.tensor([0.02, 0.04])
        rollout = _fake_rollout_result(
            _motion(frames=3),
            total_steps=2,
            device=torch.device("cpu"),
            refined_qpos=refined_qpos,
        )

        command = mjx_backend_module._command_from_refined_qpos(
            _motion(frames=3),
            refined_qpos,
            rollout,
        )

        expected_qvel = qvel_from_qpos_trajectory(refined_qpos[:, None, :])
        torch.testing.assert_close(command.qvel_trajectory, expected_qvel)
        torch.testing.assert_close(command.joint_vel, expected_qvel[..., 6:])
        torch.testing.assert_close(command.qpos_trajectory[:, 0], refined_qpos)

    def test_mjx_backend_emits_contact_capacity_metadata(self) -> None:
        def optimizer(**kwargs):
            del kwargs
            updated = torch.zeros(40, QPOS_DIM - 1)
            chunk = torch.zeros(21, QPOS_DIM - 1)
            chunk[:, 0] = 0.25
            return SimpleNamespace(
                updated_controls=updated,
                execute_chunk=chunk,
                info={
                    "best_score": torch.tensor(1.25),
                    "accepted": True,
                    "active_contact_count": torch.tensor(7),
                    "contact_pair_count": torch.tensor(13),
                },
            )

        result = _run_with_fakes(
            optimizer=optimizer,
            rollout_factory=_fake_rollout_result,
        )

        self.assertEqual(result.metadata["max_contact_points"], 512)
        self.assertEqual(result.metadata["max_geom_pairs"], 1024)
        self.assertEqual(result.metadata["active_contact_count"], 7)
        self.assertEqual(result.metadata["contact_pair_count"], 13)
        self.assertFalse(result.metadata["contact_saturated"])
        self.assertFalse(result.metadata["max_contact_points_saturated"])
        self.assertFalse(result.metadata["max_geom_pairs_saturated"])

    def test_mjx_backend_emits_replay_control_metadata(self) -> None:
        config = _spider_config()
        config.horizon_steps = 11
        config.ctrl_steps = 7

        def optimizer(**kwargs):
            window_config = kwargs["config"]
            updated = torch.zeros(int(window_config.horizon_steps), QPOS_DIM - 1)
            chunk = torch.zeros(int(window_config.control_steps) + 1, QPOS_DIM - 1)
            chunk[:, 0] = 0.25
            return SimpleNamespace(
                updated_controls=updated,
                execute_chunk=chunk,
                info={"best_score": torch.tensor(1.25), "accepted": True},
            )

        result = run_g1_wbc_mjx_mpc(
            spider_config=config,
            motion=_motion(frames=22),
            actor=WbcActor(input_dim=4, hidden_dims=(), output_dim=2),
            rollout_config=SimpleNamespace(device="cpu", max_steps=21),
            execute_rollout_config=SimpleNamespace(device="cpu", max_steps=21),
            method="g1_wbc_joint_global",
            reward_weights=None,
            total_steps=21,
            seed=5,
            runtime=SimpleNamespace(jnp=SimpleNamespace(), jax=SimpleNamespace()),
            model_factory=lambda **kwargs: _fake_model_bundle(
                profile_name=kwargs["profile_name"]
            ),
            policy_converter=lambda actor, *, jnp: SimpleNamespace(params=True),
            optimizer=optimizer,
            rollout_factory=_fake_rollout_result,
        )

        self.assertEqual(result.metadata["planning_horizon_steps"], 11)
        self.assertEqual(result.metadata["control_steps"], 7)

    def test_mjx_backend_honors_window_rejection_metadata(self) -> None:
        calls = 0

        def optimizer(**kwargs):
            nonlocal calls
            del kwargs
            calls += 1
            updated = torch.zeros(40, QPOS_DIM - 1)
            chunk = torch.zeros(21, QPOS_DIM - 1)
            chunk[:, 0] = 0.25
            return SimpleNamespace(
                updated_controls=updated,
                execute_chunk=chunk,
                info={
                    "best_score": torch.tensor(1.25),
                    "accepted": calls != 2,
                },
            )

        result = _run_with_fakes(
            optimizer=optimizer,
            rollout_factory=_fake_rollout_result,
        )

        self.assertFalse(result.metadata["accepted"])
        self.assertEqual(result.metadata["accepted_windows"], 39)
        self.assertFalse(result.result.infos[1]["accepted"])
        self.assertTrue(
            torch.allclose(
                result.result.refined_qpos[21:40, 0],
                torch.zeros(19),
            )
        )

    def test_default_optimizer_consumes_explicit_rollout_scorer(self) -> None:
        calls: list[tuple[int, ...]] = []

        def rollout_scorer(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            calls.append(tuple(int(dim) for dim in samples.shape))
            return np.arange(int(samples.shape[0]), dtype=np.float32)

        result = _run_with_fakes(
            optimizer=None,
            rollout_factory=_fake_rollout_result,
            runtime=_FakeOptimizerRuntime(),
            rollout_scorer=rollout_scorer,
        )

        self.assertTrue(result.metadata["accepted"])
        self.assertGreater(len(calls), 0)
        self.assertEqual(calls[0], (4, 40, QPOS_DIM - 1))

    def test_default_optimizer_rejects_when_scorer_favors_current_controls(self) -> None:
        def rollout_scorer(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            return -np.sum(np.asarray(samples, dtype=np.float32) ** 2, axis=(1, 2))

        result = _run_with_fakes(
            optimizer=None,
            rollout_factory=_fake_rollout_result,
            runtime=_FakeOptimizerRuntime(),
            rollout_scorer=rollout_scorer,
        )

        self.assertFalse(result.metadata["accepted"])
        self.assertEqual(result.metadata["accepted_windows"], 0)
        self.assertFalse(result.result.infos[0]["accepted"])

    def test_default_optimizer_warms_once_before_timed_windows(self) -> None:
        calls: list[dict[str, object]] = []

        def rollout_scorer(samples, reference, actor_params, model_bundle):
            del actor_params, model_bundle
            sample_array = np.asarray(samples, dtype=np.float32)
            calls.append(
                {
                    "shape": tuple(int(dim) for dim in sample_array.shape),
                    "start": int(reference["start"]),
                    "first_candidate_max_abs": float(np.max(np.abs(sample_array[0]))),
                }
            )
            return {
                "score": np.arange(int(sample_array.shape[0]), dtype=np.float32),
                "physics_step_count": np.full(
                    int(sample_array.shape[0]),
                    40,
                    dtype=np.float32,
                ),
            }

        result = _run_with_fakes(
            optimizer=None,
            rollout_factory=_fake_rollout_result,
            runtime=_FakeOptimizerRuntime(),
            rollout_scorer=rollout_scorer,
        )

        self.assertTrue(result.metadata["accepted"])
        self.assertEqual(result.metadata["num_windows"], 40)
        self.assertEqual(result.metadata["accepted_windows"], 40)
        self.assertEqual(result.result.num_windows, 40)
        self.assertEqual(len(result.result.infos), 40)
        self.assertEqual(len(result.receding.infos), 40)
        self.assertEqual(result.result.scores.shape, (40,))
        self.assertEqual(
            [info["sim_step"] for info in result.result.infos],
            list(range(0, 800, 20)),
        )
        self.assertEqual(len(calls), result.metadata["num_windows"] + 1)
        self.assertEqual(calls[0]["shape"], (4, 40, QPOS_DIM - 1))
        self.assertEqual(calls[0]["start"], 0)
        self.assertEqual(calls[1]["start"], 0)
        self.assertEqual(calls[0]["first_candidate_max_abs"], 0.0)
        self.assertEqual(calls[1]["first_candidate_max_abs"], 0.0)
        self.assertTrue(result.metadata["jit_warmup_enabled"])
        self.assertGreaterEqual(result.metadata["jit_warmup_wall_time_sec"], 0.0)
        self.assertEqual(result.metadata["physics_step_count_min"], 40)
        self.assertEqual(result.metadata["physics_step_count_max"], 40)
        self.assertEqual(result.metadata["physics_step_count_windows"], 40)

    def test_default_optimizer_uses_explicit_rollout_reference_factory(self) -> None:
        references: list[dict[str, object]] = []

        def rollout_scorer(samples, reference, actor_params, model_bundle):
            del actor_params, model_bundle
            references.append(dict(reference))
            scale = float(reference["score_scale"])
            return scale * np.arange(int(samples.shape[0]), dtype=np.float32)

        def rollout_reference_factory(**kwargs):
            return {
                "window_start": kwargs["start"],
                "score_scale": 0.5,
            }

        result = _run_with_fakes(
            optimizer=None,
            rollout_factory=_fake_rollout_result,
            runtime=_FakeOptimizerRuntime(),
            rollout_scorer=rollout_scorer,
            rollout_reference_factory=rollout_reference_factory,
        )

        self.assertTrue(result.metadata["accepted"])
        self.assertGreater(len(references), 0)
        self.assertEqual(references[0]["window_start"], 0)
        self.assertEqual(references[0]["score_scale"], 0.5)

    def test_mjx_backend_accepts_explicit_rollout_components(self) -> None:
        runtime = _FakeOptimizerRuntime()

        def physics_step_fn(
            model_bundle,
            robot_state,
            command_qpos,
            action,
            step_index,
            *,
            runtime,
        ):
            raise AssertionError("fake optimizer should not call rollout scorer")

        components = build_mjx_rollout_components(
            runtime=runtime,
            physics_step_fn=physics_step_fn,
            score_weights={"root_pos": 1.0},
        )

        result = run_g1_wbc_mjx_mpc(
            spider_config=_spider_config(),
            motion=_motion(),
            actor=WbcActor(input_dim=4, hidden_dims=(), output_dim=2),
            rollout_config=SimpleNamespace(device="cpu", max_steps=800),
            execute_rollout_config=SimpleNamespace(device="cpu", max_steps=800),
            method="g1_wbc_joint_global",
            reward_weights=None,
            total_steps=800,
            seed=5,
            runtime=runtime,
            model_factory=lambda **kwargs: _fake_model_bundle(
                profile_name=kwargs["profile_name"]
            ),
            policy_converter=lambda actor, *, jnp: SimpleNamespace(params=True),
            optimizer=_fake_optimizer,
            rollout_factory=_fake_rollout_result,
            rollout_scorer=components.rollout_scorer,
            rollout_reference_factory=components.rollout_reference_factory,
        )

        self.assertTrue(result.metadata["accepted"])
        self.assertEqual(result.metadata["backend"], "mjx")

    def test_scan_enabled_accepted_windows_require_physics_scan_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "physics scan"):
            _run_with_fakes(
                optimizer=_fake_optimizer,
                rollout_factory=_fake_rollout_result,
                runtime=_FakeOptimizerRuntime(),
                enable_physics_scan=True,
                reward_weights={
                    "body_global_pos_error": 4.0,
                    "ee_global_pos_error": 1.5,
                    "contact_false_positive": 1.5,
                    "contact_false_negative": 0.4,
                    "control_delta": 1.8,
                    "joint_acc": 0.006,
                },
            )

    def test_mjx_backend_default_path_stays_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            (RuntimeError, NotImplementedError),
            "--mpc-backend mjx|production MJX physics scan",
        ):
            run_g1_wbc_mjx_mpc(
                spider_config=_spider_config(),
                motion=_motion(),
                actor=WbcActor(input_dim=4, hidden_dims=(), output_dim=2),
                rollout_config=SimpleNamespace(device="cpu", max_steps=800),
                execute_rollout_config=SimpleNamespace(device="cpu", max_steps=800),
                method="g1_wbc_joint_global",
                reward_weights=None,
                total_steps=800,
                seed=5,
            )

    def test_mjx_backend_rejects_malformed_fake_rollout(self) -> None:
        with self.assertRaisesRegex(ValueError, "rollout.qvel"):
            _run_with_fakes(
                optimizer=_fake_optimizer,
                rollout_factory=_bad_qvel_rollout_result,
            )

    def test_mjx_backend_rejects_noop_rollout_that_ignores_refined_qpos(self) -> None:
        with self.assertRaisesRegex(ValueError, "rollout.qpos"):
            _run_with_fakes(
                optimizer=_fake_optimizer,
                rollout_factory=_baseline_rollout_result,
            )

    def test_mjx_backend_rejects_missing_best_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "best_score"):
            _run_with_fakes(
                optimizer=_missing_score_optimizer,
                rollout_factory=_fake_rollout_result,
            )

    def test_mjx_backend_rejects_non_finite_best_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "best_score"):
            _run_with_fakes(
                optimizer=_nan_score_optimizer,
                rollout_factory=_fake_rollout_result,
            )

    def test_mjx_backend_rejects_non_scalar_best_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "best_score"):
            _run_with_fakes(
                optimizer=_vector_score_optimizer,
                rollout_factory=_fake_rollout_result,
            )

    def test_mjx_backend_rejects_missing_or_invalid_accepted_metadata(self) -> None:
        invalid_values = {
            "missing": None,
            "none": None,
            "nan": torch.tensor(float("nan")),
            "string": "false",
        }
        for name, accepted in invalid_values.items():
            with self.subTest(name=name):
                include_field = name != "missing"

                def optimizer(**kwargs):
                    del kwargs
                    updated = torch.zeros(40, QPOS_DIM - 1)
                    chunk = torch.zeros(21, QPOS_DIM - 1)
                    chunk[:, 0] = 0.25
                    info = {"best_score": torch.tensor(1.25)}
                    if include_field:
                        info["accepted"] = accepted
                    return SimpleNamespace(
                        updated_controls=updated,
                        execute_chunk=chunk,
                        info=info,
                    )

                with self.assertRaisesRegex(ValueError, "accepted"):
                    _run_with_fakes(
                        optimizer=optimizer,
                        rollout_factory=_fake_rollout_result,
                    )

    def test_mjx_backend_shifts_controls_between_windows(self) -> None:
        captured: list[torch.Tensor] = []

        def optimizer(**kwargs):
            controls = kwargs["controls"].detach().clone()
            captured.append(controls)
            updated = torch.zeros_like(controls)
            updated[:, 0] = torch.arange(controls.shape[0], dtype=torch.float32)
            chunk = torch.zeros(21, QPOS_DIM - 1)
            chunk[:, 0] = 0.25
            return SimpleNamespace(
                updated_controls=updated,
                execute_chunk=chunk,
                info={"best_score": torch.tensor(1.25), "accepted": True},
            )

        _run_with_fakes(optimizer=optimizer, rollout_factory=_fake_rollout_result)

        self.assertGreaterEqual(len(captured), 2)
        self.assertTrue(
            torch.allclose(
                captured[1][:20, 0],
                torch.arange(20, 40, dtype=torch.float32),
            )
        )
        self.assertTrue(torch.allclose(captured[1][20:, 0], torch.zeros(20)))

    def test_mjx_backend_accepts_jax_optimizer_arrays(self) -> None:
        try:
            import jax.numpy as jnp
        except Exception as exc:
            self.skipTest(f"JAX is not available: {exc}")

        def optimizer(**kwargs):
            del kwargs
            return SimpleNamespace(
                updated_controls=jnp.zeros((40, QPOS_DIM - 1), dtype=jnp.float32),
                execute_chunk=jnp.zeros((21, QPOS_DIM - 1), dtype=jnp.float32),
                info={
                    "best_score": jnp.asarray(1.25, dtype=jnp.float32),
                    "accepted": True,
                },
            )

        result = _run_with_fakes(
            optimizer=optimizer,
            rollout_factory=_fake_rollout_result,
        )

        self.assertTrue(result.metadata["accepted"])

    def test_default_mjx_optimizer_keeps_full_horizon_controls_off_torch_per_window(
        self,
    ) -> None:
        converted_shapes: list[tuple[int, ...]] = []
        original_to_torch = mjx_backend_module._to_torch

        def recording_to_torch(value, *, device):
            tensor = original_to_torch(value, device=device)
            converted_shapes.append(tuple(tensor.shape))
            return tensor

        def rollout_scorer(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            return np.arange(int(samples.shape[0]), dtype=np.float32)

        def rollout_reference_factory(**kwargs):
            del kwargs
            return {}

        with mock.patch.object(
            mjx_backend_module,
            "_to_torch",
            side_effect=recording_to_torch,
        ):
            result = _run_with_fakes(
                optimizer=None,
                rollout_factory=_fake_rollout_result,
                runtime=_FakeOptimizerRuntime(),
                rollout_scorer=rollout_scorer,
                rollout_reference_factory=rollout_reference_factory,
            )

        self.assertTrue(result.metadata["accepted"])
        self.assertEqual(converted_shapes.count((40, QPOS_DIM - 1)), 1)
        self.assertEqual(converted_shapes.count((21, QPOS_DIM - 1)), 40)

    def test_default_mjx_optimizer_defers_best_score_scalarization(self) -> None:
        def rollout_scorer(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            return np.arange(int(samples.shape[0]), dtype=np.float32)

        def rollout_reference_factory(**kwargs):
            del kwargs
            return {}

        with mock.patch.object(
            mjx_backend_module,
            "_scalar_info",
            side_effect=AssertionError("default MJX path should defer score sync"),
        ):
            result = _run_with_fakes(
                optimizer=None,
                rollout_factory=_fake_rollout_result,
                runtime=_FakeOptimizerRuntime(),
                rollout_scorer=rollout_scorer,
                rollout_reference_factory=rollout_reference_factory,
            )

        self.assertTrue(result.metadata["accepted"])
        self.assertEqual(result.result.scores.shape, (40,))

    def test_mjx_backend_rejects_multiple_visible_devices(self) -> None:
        runtime = SimpleNamespace(
            jnp=SimpleNamespace(),
            jax=SimpleNamespace(),
            status=SimpleNamespace(visible_devices=("0", "1")),
        )
        with self.assertRaisesRegex(RuntimeError, "single GPU"):
            _run_with_fakes(
                optimizer=_fake_optimizer,
                rollout_factory=_fake_rollout_result,
                runtime=runtime,
            )

    def test_window_config_forwards_mpc_iterations(self) -> None:
        config = _spider_config()
        config.max_num_iterations = 3

        window_config = mjx_backend_module._window_config_from_spider(config)

        self.assertEqual(window_config.iterations, 3)


def _run_with_fakes(
    *,
    optimizer,
    rollout_factory,
    runtime=None,
    rollout_scorer=None,
    rollout_reference_factory=None,
    enable_physics_scan=False,
    reward_weights=None,
):
    kwargs = {}
    if optimizer is not None:
        kwargs["optimizer"] = optimizer
    if rollout_scorer is not None:
        kwargs["rollout_scorer"] = rollout_scorer
    if rollout_reference_factory is not None:
        kwargs["rollout_reference_factory"] = rollout_reference_factory
    return run_g1_wbc_mjx_mpc(
        spider_config=_spider_config(),
        motion=_motion(),
        actor=WbcActor(input_dim=4, hidden_dims=(), output_dim=2),
        rollout_config=SimpleNamespace(device="cpu", max_steps=800),
        execute_rollout_config=SimpleNamespace(device="cpu", max_steps=800),
        method="g1_wbc_joint_global",
        reward_weights=reward_weights,
        total_steps=800,
        seed=5,
        runtime=runtime or SimpleNamespace(jnp=SimpleNamespace(), jax=SimpleNamespace()),
        model_factory=lambda **kwargs: _fake_model_bundle(
            profile_name=kwargs["profile_name"]
        ),
        policy_converter=lambda actor, *, jnp: SimpleNamespace(params=True),
        rollout_factory=rollout_factory,
        enable_physics_scan=enable_physics_scan,
        **kwargs,
    )


class _FakeOptimizerJnp:
    @staticmethod
    def asarray(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def zeros(shape):
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def concatenate(values, axis=0):
        return np.concatenate(values, axis=axis)

    @staticmethod
    def repeat(value, repeats, axis=0):
        return np.repeat(value, repeats, axis=axis)

    @staticmethod
    def full(shape, value):
        return np.full(shape, value, dtype=np.float32)

    @staticmethod
    def sum(value, axis=None):
        return np.sum(value, axis=axis)

    @staticmethod
    def argmax(value):
        return np.argmax(value)

    @staticmethod
    def mean(value):
        return np.mean(value)

    @staticmethod
    def linspace(start, stop, num):
        return np.linspace(start, stop, int(num), dtype=np.float32)

    @staticmethod
    def floor(value):
        return np.floor(value)

    @staticmethod
    def minimum(left, right):
        return np.minimum(left, right)


class _FakeOptimizerRandom:
    @staticmethod
    def normal(key, shape):
        seed = int(np.asarray(key, dtype=np.uint32).sum())
        rng = np.random.default_rng(seed)
        return rng.normal(size=shape).astype(np.float32)


class _FakeOptimizerNN:
    @staticmethod
    def softmax(value):
        value = np.asarray(value, dtype=np.float32)
        shifted = value - np.max(value)
        exp = np.exp(shifted)
        return exp / np.sum(exp)


class _FakeOptimizerJax:
    random = _FakeOptimizerRandom()
    nn = _FakeOptimizerNN()


class _FakeOptimizerRuntime:
    jnp = _FakeOptimizerJnp()
    jax = _FakeOptimizerJax()


class _DlpackOnlyArray:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.dlpack_calls = 0
        self.array_calls = 0

    @property
    def shape(self):
        return self.tensor.shape

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()

    def __dlpack__(self, stream=None):
        del stream
        self.dlpack_calls += 1
        return torch.utils.dlpack.to_dlpack(self.tensor)

    def __array__(self, dtype=None):
        del dtype
        self.array_calls += 1
        raise AssertionError("DLPack arrays must not fall back to NumPy conversion")


class MjxBackendConversionTest(unittest.TestCase):
    def test_validated_execute_chunk_uses_dlpack_without_numpy_fallback(self) -> None:
        source = _DlpackOnlyArray(torch.ones(21, QPOS_DIM - 1, dtype=torch.float32))

        chunk = mjx_backend_module._validated_execute_chunk(
            source,
            execute_steps=20,
            device=torch.device("cpu"),
        )

        self.assertEqual(source.dlpack_calls, 1)
        self.assertEqual(source.array_calls, 0)
        self.assertEqual(tuple(chunk.shape), (21, QPOS_DIM - 1))
        self.assertTrue(torch.allclose(chunk, torch.ones_like(chunk)))

    def test_block_window_result_until_ready_blocks_outputs_and_info(self) -> None:
        class Blockable:
            def __init__(self) -> None:
                self.blocked = False

            def block_until_ready(self):
                self.blocked = True
                return self

        updated_controls = Blockable()
        execute_chunk = Blockable()
        best_score = Blockable()

        mjx_backend_module._block_window_result_until_ready(
            SimpleNamespace(
                updated_controls=updated_controls,
                execute_chunk=execute_chunk,
                info={"best_score": best_score},
            )
        )

        self.assertTrue(updated_controls.blocked)
        self.assertTrue(execute_chunk.blocked)
        self.assertTrue(best_score.blocked)


def _fake_model_bundle(*, profile_name: str = "wxy_parity"):
    return SimpleNamespace(
        profile=SimpleNamespace(name=profile_name),
        cpu_model=SimpleNamespace(
            jnt_limited=np.ones(ACTION_DIM, dtype=np.int32),
            jnt_range=np.stack(
                [
                    np.full(ACTION_DIM, -1.0, dtype=np.float32),
                    np.full(ACTION_DIM, 1.0, dtype=np.float32),
                ],
                axis=-1,
            ),
        ),
        joint_name_to_id={
            f"robot/{joint_name}": index
            for index, joint_name in enumerate(MUJOCO_JOINT_NAMES)
        },
    )


def _fake_optimizer(**kwargs):
    del kwargs
    updated = torch.zeros(40, QPOS_DIM - 1)
    chunk = torch.zeros(21, QPOS_DIM - 1)
    chunk[:, 0] = 0.25
    return SimpleNamespace(
        updated_controls=updated,
        execute_chunk=chunk,
        info={"best_score": torch.tensor(1.25), "accepted": True},
    )


def _missing_score_optimizer(**kwargs):
    del kwargs
    updated = torch.zeros(40, QPOS_DIM - 1)
    chunk = torch.zeros(21, QPOS_DIM - 1)
    chunk[:, 0] = 0.25
    return SimpleNamespace(
        updated_controls=updated,
        execute_chunk=chunk,
        info={"accepted": True},
    )


def _nan_score_optimizer(**kwargs):
    del kwargs
    updated = torch.zeros(40, QPOS_DIM - 1)
    chunk = torch.zeros(21, QPOS_DIM - 1)
    chunk[:, 0] = 0.25
    return SimpleNamespace(
        updated_controls=updated,
        execute_chunk=chunk,
        info={"best_score": torch.tensor(float("nan")), "accepted": True},
    )


def _vector_score_optimizer(**kwargs):
    del kwargs
    updated = torch.zeros(40, QPOS_DIM - 1)
    chunk = torch.zeros(21, QPOS_DIM - 1)
    chunk[:, 0] = 0.25
    return SimpleNamespace(
        updated_controls=updated,
        execute_chunk=chunk,
        info={"best_score": torch.tensor([1.0, 2.0]), "accepted": True},
    )


def _fake_rollout_result(
    motion: G1Motion,
    total_steps: int,
    *,
    device: torch.device,
    refined_qpos: torch.Tensor,
):
    frames = int(total_steps) + 1
    bodies = len(MUJOCO_BODY_NAMES)
    qpos = refined_qpos.to(device).view(frames, 1, QPOS_DIM).clone()
    qvel = torch.zeros(frames, 1, QVEL_DIM, device=device)
    body_quat = torch.zeros(frames, 1, bodies, 4, device=device)
    body_quat[..., 0] = 1.0
    return SimpleNamespace(
        qpos=qpos,
        qvel=qvel,
        body_pos_w=torch.zeros(frames, 1, bodies, 3, device=device),
        body_quat_w=body_quat,
        body_lin_vel_w=torch.zeros(frames, 1, bodies, 3, device=device),
        body_ang_vel_w=torch.zeros(frames, 1, bodies, 3, device=device),
        actions=torch.zeros(total_steps, 1, ACTION_DIM, device=device),
        controls=torch.zeros(total_steps, 1, ACTION_DIM, device=device),
        contact_indicator=torch.zeros(frames, 1, 2, device=device),
        contact_force=torch.zeros(frames, 1, 2, device=device),
        floor_contact_indicator=torch.zeros(frames, 1, 3, device=device),
        floor_contact_force=torch.zeros(frames, 1, 3, device=device),
        ref_indices=torch.arange(frames, device=device).view(frames, 1),
    )


def _bad_qvel_rollout_result(
    motion: G1Motion,
    total_steps: int,
    *,
    device: torch.device,
    refined_qpos: torch.Tensor,
):
    result = _fake_rollout_result(
        motion,
        total_steps,
        device=device,
        refined_qpos=refined_qpos,
    )
    result.qvel = torch.zeros(int(total_steps) + 1, 1, 1, device=device)
    return result


def _baseline_rollout_result(
    motion: G1Motion,
    total_steps: int,
    *,
    device: torch.device,
    refined_qpos: torch.Tensor,
):
    del refined_qpos
    baseline_qpos = motion.qpos()[: int(total_steps) + 1].to(device)
    return _fake_rollout_result(
        motion,
        total_steps,
        device=device,
        refined_qpos=baseline_qpos,
    )


if __name__ == "__main__":
    unittest.main()
