from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import torch

from spider.config import Config
from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    MUJOCO_BODY_NAMES,
    QPOS_DIM,
    QVEL_DIM,
)
from spider.tasks.g1_wbc.mjx_backend import run_g1_wbc_mjx_mpc
from spider.tasks.g1_wbc.motion import G1Motion
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
                    info={"best_score": torch.tensor(1.0)},
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
                result.result.refined_qpos[:800, 1],
                torch.full((800,), 0.25),
            )
        )
        self.assertAlmostEqual(float(result.result.refined_qpos[800, 1]), 0.25)
        self.assertTrue(
            torch.allclose(
                result.result.command.qpos_trajectory[:800, 0, 1],
                torch.full((800,), 0.25),
            )
        )
        self.assertAlmostEqual(
            float(result.result.command.qpos_trajectory[800, 0, 1]),
            0.25,
        )
        self.assertAlmostEqual(float(result.result.rollout.qpos[800, 0, 1]), 0.25)
        self.assertEqual(result.result.scores.shape, (40,))
        self.assertTrue(torch.allclose(result.result.scores, torch.full((40,), 1.25)))
        self.assertEqual(result.receding.executed_steps, 800)
        self.assertIn("model:wxy_parity", calls)
        self.assertIn("policy", calls)

    def test_mjx_backend_default_path_still_requires_runtime(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "--mpc-backend mjx"):
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
                info={"best_score": torch.tensor(1.25)},
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


def _run_with_fakes(*, optimizer, rollout_factory, runtime=None):
    return run_g1_wbc_mjx_mpc(
        spider_config=_spider_config(),
        motion=_motion(),
        actor=WbcActor(input_dim=4, hidden_dims=(), output_dim=2),
        rollout_config=SimpleNamespace(device="cpu", max_steps=800),
        execute_rollout_config=SimpleNamespace(device="cpu", max_steps=800),
        method="g1_wbc_joint_global",
        reward_weights=None,
        total_steps=800,
        seed=5,
        runtime=runtime or SimpleNamespace(jnp=SimpleNamespace(), jax=SimpleNamespace()),
        model_factory=lambda **kwargs: SimpleNamespace(
            profile=SimpleNamespace(name=kwargs["profile_name"])
        ),
        policy_converter=lambda actor, *, jnp: SimpleNamespace(params=True),
        optimizer=optimizer,
        rollout_factory=rollout_factory,
    )


def _fake_optimizer(**kwargs):
    del kwargs
    updated = torch.zeros(40, QPOS_DIM - 1)
    chunk = torch.zeros(21, QPOS_DIM - 1)
    chunk[:, 0] = 0.25
    return SimpleNamespace(
        updated_controls=updated,
        execute_chunk=chunk,
        info={"best_score": torch.tensor(1.25)},
    )


def _missing_score_optimizer(**kwargs):
    del kwargs
    updated = torch.zeros(40, QPOS_DIM - 1)
    chunk = torch.zeros(21, QPOS_DIM - 1)
    chunk[:, 0] = 0.25
    return SimpleNamespace(updated_controls=updated, execute_chunk=chunk, info={})


def _nan_score_optimizer(**kwargs):
    del kwargs
    updated = torch.zeros(40, QPOS_DIM - 1)
    chunk = torch.zeros(21, QPOS_DIM - 1)
    chunk[:, 0] = 0.25
    return SimpleNamespace(
        updated_controls=updated,
        execute_chunk=chunk,
        info={"best_score": torch.tensor(float("nan"))},
    )


def _vector_score_optimizer(**kwargs):
    del kwargs
    updated = torch.zeros(40, QPOS_DIM - 1)
    chunk = torch.zeros(21, QPOS_DIM - 1)
    chunk[:, 0] = 0.25
    return SimpleNamespace(
        updated_controls=updated,
        execute_chunk=chunk,
        info={"best_score": torch.tensor([1.0, 2.0])},
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
