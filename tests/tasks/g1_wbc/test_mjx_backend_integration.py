from types import SimpleNamespace
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
from spider.tasks.g1_wbc.rollout import WbcRolloutConfig
from spider.tasks.g1_wbc.spider_task import G1WbcMpcRun


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
            rollout_config=WbcRolloutConfig(device="cpu", max_steps=800),
            execute_rollout_config=WbcRolloutConfig(device="cpu", max_steps=800),
            method="g1_wbc_joint_global",
            reward_weights=None,
            total_steps=800,
            seed=5,
            runtime=SimpleNamespace(jnp=SimpleNamespace(), jax=SimpleNamespace()),
            model_factory=fake_model_factory,
            policy_converter=fake_policy_converter,
            optimizer=lambda **kwargs: SimpleNamespace(
                updated_controls=torch.zeros(40, QPOS_DIM - 1),
                execute_chunk=torch.zeros(21, QPOS_DIM - 1),
                info={"best_score": torch.tensor(1.25)},
            ),
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
                rollout_config=WbcRolloutConfig(device="cpu", max_steps=800),
                execute_rollout_config=WbcRolloutConfig(device="cpu", max_steps=800),
                method="g1_wbc_joint_global",
                reward_weights=None,
                total_steps=800,
                seed=5,
            )


def _fake_rollout_result(motion: G1Motion, total_steps: int, *, device: torch.device):
    from spider.tasks.g1_wbc.rollout import RolloutResult

    frames = int(total_steps) + 1
    bodies = len(MUJOCO_BODY_NAMES)
    qpos = torch.zeros(frames, 1, QPOS_DIM, device=device)
    qpos[..., 3] = 1.0
    qvel = torch.zeros(frames, 1, QVEL_DIM, device=device)
    body_quat = torch.zeros(frames, 1, bodies, 4, device=device)
    body_quat[..., 0] = 1.0
    return RolloutResult(
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


if __name__ == "__main__":
    unittest.main()
