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
    OBS_HISTORY_LENGTH,
    QPOS_DIM,
    QVEL_DIM,
)
from spider.tasks.g1_wbc.mjx_backend import run_g1_wbc_mjx_mpc
from spider.tasks.g1_wbc.mjx_components import build_mjx_rollout_components
from spider.tasks.g1_wbc.motion import G1CommandBatch, G1Motion, qvel_from_qpos_trajectory
from spider.tasks.g1_wbc.mpc import REWARD_WEIGHT_PRESETS
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

            def command_builder(motion, qpos_trajectory, config):
                del config
                frames = int(qpos_trajectory.shape[0])
                bodies = len(MUJOCO_BODY_NAMES)
                body_quat = torch.zeros(frames, 1, bodies, 4)
                body_quat[..., 0] = 1.0
                return SimpleNamespace(
                    path=motion.path,
                    motion_type=motion.motion_type,
                    fps=motion.fps,
                    joint_pos=qpos_trajectory[..., 7:].clone(),
                    joint_vel=torch.zeros(frames, 1, ACTION_DIM),
                    body_pos_w=torch.zeros(frames, 1, bodies, 3),
                    body_quat_w=body_quat,
                    body_lin_vel_w=torch.zeros(frames, 1, bodies, 3),
                    body_ang_vel_w=torch.zeros(frames, 1, bodies, 3),
                    qpos_trajectory=qpos_trajectory.clone(),
                    qvel_trajectory=torch.zeros(frames, 1, QVEL_DIM),
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
                command_builder=command_builder,
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

        model_kwargs = {}

        def fake_model_factory(**kwargs):
            model_kwargs.update(kwargs)
            calls.append(f"model:{kwargs['profile_name']}")
            return SimpleNamespace(
                profile=SimpleNamespace(name="wxy_parity"),
                mjx_impl=kwargs.get("mjx_impl"),
                mjx_warp_naconmax=kwargs.get("mjx_warp_naconmax"),
                mjx_warp_njmax=kwargs.get("mjx_warp_njmax"),
                mjx_model_options=kwargs.get("mjx_model_options"),
                mjx_model=SimpleNamespace(impl=kwargs.get("mjx_impl")),
            )

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
            command_builder=_fake_command_builder,
            mjx_impl="warp",
            mjx_warp_naconmax=4096,
            mjx_warp_njmax=512,
            mjx_model_options={"iterations": 6, "ls_iterations": 10},
        )

        self.assertIsInstance(result, G1WbcMpcRun)
        self.assertEqual(result.metadata["backend"], "mjx")
        self.assertTrue(result.metadata["accepted"])
        self.assertFalse(result.metadata["used_baseline_fallback"])
        self.assertFalse(result.metadata["use_guided_candidate"])
        self.assertEqual(result.metadata["rollout_source"], "static_qpos_fallback")
        self.assertFalse(result.metadata["rollout_dynamic_execute_trace"])
        self.assertEqual(result.metadata["execute_trace_chunks"], 0)
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
        chunks = result.result.executed_command_chunks
        self.assertEqual(len(chunks), 40)
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[0].execute_steps, 20)
        self.assertEqual(chunks[0].horizon_steps, 40)
        self.assertEqual(chunks[-1].start, 780)
        self.assertEqual(chunks[-1].execute_steps, 20)
        self.assertEqual(chunks[-1].horizon_steps, 40)
        self.assertEqual(chunks[0].command.qpos_trajectory.shape, (40, 1, QPOS_DIM))
        self.assertTrue(
            torch.allclose(
                chunks[0].command.qpos_trajectory[:21, 0, 0],
                torch.full((21,), 0.25),
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
        self.assertEqual(model_kwargs["mjx_impl"], "warp")
        self.assertEqual(model_kwargs["mjx_warp_naconmax"], 4096)
        self.assertEqual(model_kwargs["mjx_warp_njmax"], 512)
        self.assertEqual(
            model_kwargs["mjx_model_options"],
            {"iterations": 6, "ls_iterations": 10},
        )
        self.assertEqual(result.metadata["mjx_impl"], "warp")
        self.assertEqual(result.metadata["mjx_model_impl"], "warp")
        self.assertEqual(result.metadata["mjx_warp_naconmax"], 4096)
        self.assertEqual(result.metadata["mjx_warp_njmax"], 512)
        self.assertEqual(
            result.metadata["mjx_model_options"],
            {"iterations": 6, "ls_iterations": 10},
        )
        self.assertIn("model:wxy_parity", calls)
        self.assertIn("policy", calls)
        first_window = result.receding.infos[0]
        for field in (
            "reference_wall_time_sec",
            "optimizer_wall_time_sec",
            "optimizer_result_sync_wall_time_sec",
            "execute_trace_wall_time_sec",
            "shift_wall_time_sec",
            "window_wall_time_sec",
        ):
            self.assertIn(field, first_window)
            self.assertGreaterEqual(first_window[field], 0.0)
        self.assertIn("optimizer_result_sync_wall_time_sec", result.metadata)
        self.assertGreaterEqual(result.metadata["optimizer_result_sync_wall_time_sec"], 0.0)

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

        def command_builder(motion, qpos_trajectory, config):
            del motion, config
            qvel = qvel_from_qpos_trajectory(qpos_trajectory)
            return G1CommandBatch(
                path=None,
                motion_type="mujoco",
                fps=50.0,
                joint_pos=qpos_trajectory[..., 7:].contiguous(),
                joint_vel=qvel[..., 6:].contiguous(),
                body_pos_w=rollout.body_pos_w.clone(),
                body_quat_w=rollout.body_quat_w.clone(),
                body_lin_vel_w=rollout.body_lin_vel_w.clone(),
                body_ang_vel_w=rollout.body_ang_vel_w.clone(),
                qpos_trajectory=qpos_trajectory.contiguous(),
                qvel_trajectory=qvel.contiguous(),
            )

        command = mjx_backend_module._command_from_refined_qpos(
            _motion(frames=3),
            refined_qpos,
            rollout,
            command_builder=command_builder,
            rollout_config=SimpleNamespace(device="cpu"),
        )

        expected_qvel = qvel_from_qpos_trajectory(refined_qpos[:, None, :])
        torch.testing.assert_close(command.qvel_trajectory, expected_qvel)
        torch.testing.assert_close(command.joint_vel, expected_qvel[..., 6:])
        torch.testing.assert_close(command.qpos_trajectory[:, 0], refined_qpos)

    def test_command_from_refined_qpos_uses_command_builder_body_fields(self) -> None:
        refined_qpos = torch.zeros(3, QPOS_DIM)
        refined_qpos[:, 3] = 1.0
        rollout = _fake_rollout_result(
            _motion(frames=3),
            total_steps=2,
            device=torch.device("cpu"),
            refined_qpos=refined_qpos,
        )
        rollout.body_pos_w.fill_(-5.0)
        rollout.body_lin_vel_w.fill_(-6.0)
        builder_body_pos = torch.full_like(rollout.body_pos_w, 7.0)
        builder_body_lin_vel = torch.full_like(rollout.body_lin_vel_w, 8.0)

        def command_builder(motion, qpos_trajectory, config):
            del motion, config
            qvel = qvel_from_qpos_trajectory(qpos_trajectory)
            return G1CommandBatch(
                path=None,
                motion_type="mujoco",
                fps=50.0,
                joint_pos=qpos_trajectory[..., 7:].contiguous(),
                joint_vel=qvel[..., 6:].contiguous(),
                body_pos_w=builder_body_pos.clone(),
                body_quat_w=rollout.body_quat_w.clone(),
                body_lin_vel_w=builder_body_lin_vel.clone(),
                body_ang_vel_w=rollout.body_ang_vel_w.clone(),
                qpos_trajectory=qpos_trajectory.contiguous(),
                qvel_trajectory=qvel.contiguous(),
            )

        command = mjx_backend_module._command_from_refined_qpos(
            _motion(frames=3),
            refined_qpos,
            rollout,
            command_builder=command_builder,
            rollout_config=SimpleNamespace(device="cpu"),
        )

        torch.testing.assert_close(command.body_pos_w, builder_body_pos)
        torch.testing.assert_close(command.body_lin_vel_w, builder_body_lin_vel)

    def test_mjx_score_weights_maps_rotation_terms(self) -> None:
        weights = mjx_backend_module._mjx_score_weights(
            "g1_wbc_joint_global",
            {
                "root_rot_error": 0.5,
                "body_global_rot_error": 0.8,
                "ee_global_rot_error": 0.3,
            },
        )

        self.assertEqual(weights["root_rot"], 0.5)
        self.assertEqual(weights["body_global_rot"], 0.8)
        self.assertEqual(weights["ee_global_rot"], 0.3)

    def test_mjx_score_weights_maps_contact_and_action_terms_exactly(self) -> None:
        weights = mjx_backend_module._mjx_score_weights(
            "g1_wbc_joint_global",
            {
                "bad_floor_contact": 4.5,
                "bad_floor_force_excess": 0.7,
                "contact_force_active": 1.1,
                "contact_force_peak_excess": 0.9,
                "contact_force_delta": 2.5,
                "contact_mismatch": 2.0,
                "contact_false_positive": 1.5,
                "contact_false_negative": 0.4,
                "contact_switch": 1.2,
                "action_delta": 0.6,
                "joint_jerk": 0.0012,
            },
        )

        self.assertEqual(weights["bad_floor_contact"], 4.5)
        self.assertEqual(weights["bad_floor_force_excess"], 0.7)
        self.assertEqual(weights["contact_force_active"], 1.1)
        self.assertEqual(weights["contact_force_peak_excess"], 0.9)
        self.assertEqual(weights["contact_force_delta"], 2.5)
        self.assertEqual(weights["contact"], 2.0)
        self.assertEqual(weights["contact_false_positive"], 1.5)
        self.assertEqual(weights["contact_false_negative"], 0.4)
        self.assertEqual(weights["contact_switch"], 1.2)
        self.assertEqual(weights["action_delta"], 0.6)
        self.assertEqual(weights["joint_jerk"], 0.0012)

    def test_mjx_score_weights_cover_joint_global_default_preset(self) -> None:
        expected_mapping = {
            "bad_floor_contact": "bad_floor_contact",
            "bad_floor_force_excess": "bad_floor_force_excess",
            "contact_switch": "contact_switch",
            "contact_force_delta": "contact_force_delta",
            "contact_false_positive": "contact_false_positive",
            "contact_false_negative": "contact_false_negative",
            "control_delta": "control_delta",
            "action_delta": "action_delta",
            "joint_acc": "joint_acc",
            "joint_jerk": "joint_jerk",
            "body_global_pos_error": "body_global_pos",
            "body_global_rot_error": "body_global_rot",
            "ee_global_pos_error": "ee_global_pos",
            "ee_global_rot_error": "ee_global_rot",
        }
        preset = REWARD_WEIGHT_PRESETS["g1_wbc_joint_global"]
        weights = mjx_backend_module._mjx_score_weights(
            "g1_wbc_joint_global",
            preset,
        )

        self.assertEqual(sorted(preset), sorted(expected_mapping))
        for source_name, target_name in expected_mapping.items():
            self.assertEqual(weights[target_name], preset[source_name])

    def test_mjx_score_weights_cover_formal_v14_terms(self) -> None:
        expected_mapping = {
            "root_pos_error": "root_pos",
            "root_rot_error": "root_rot",
            "joint_pos_error": "joint_pos",
            "body_global_pos_error": "body_global_pos",
            "body_global_rot_error": "body_global_rot",
            "body_local_pos_error": "body_local_pos",
            "body_local_rot_error": "body_local_rot",
            "ee_global_pos_error": "ee_global_pos",
            "ee_global_rot_error": "ee_global_rot",
            "ee_local_pos_error": "ee_local_pos",
            "ee_local_rot_error": "ee_local_rot",
            "hand_global_pos_error": "hand_global_pos",
            "hand_global_rot_error": "hand_global_rot",
            "hand_local_pos_error": "hand_local_pos",
            "hand_local_rot_error": "hand_local_rot",
            "bad_floor_contact": "bad_floor_contact",
            "bad_floor_force_excess": "bad_floor_force_excess",
            "contact_force_active": "contact_force_active",
            "contact_force_peak_excess": "contact_force_peak_excess",
            "contact_force_delta": "contact_force_delta",
            "contact_mismatch": "contact",
            "contact_false_positive": "contact_false_positive",
            "contact_false_negative": "contact_false_negative",
            "contact_switch": "contact_switch",
            "control_delta": "control_delta",
            "action_delta": "action_delta",
            "joint_acc": "joint_acc",
            "joint_jerk": "joint_jerk",
        }
        reward_weights = {
            source_name: float(index + 1)
            for index, source_name in enumerate(expected_mapping)
        }

        weights = mjx_backend_module._mjx_score_weights(
            "g1_wbc_joint_global",
            reward_weights,
        )

        self.assertEqual(sorted(weights), sorted(expected_mapping.values()))
        for source_name, target_name in expected_mapping.items():
            self.assertEqual(weights[target_name], reward_weights[source_name])

    def test_mjx_score_weights_reject_unsupported_nonzero_terms(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported_term"):
            mjx_backend_module._mjx_score_weights(
                "g1_wbc_joint_global",
                {
                    "body_global_pos_error": 58.0,
                    "body_local_pos_error": 5.0,
                    "unsupported_term": 3.0,
                },
            )

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
        config.num_samples = 9
        config.max_num_iterations = 3
        config.horizon_steps = 11
        config.ctrl_steps = 7
        config.num_knot_points = 5
        config.temperature = 0.9
        config.pos_noise_scale = 0.03
        config.rot_noise_scale = 0.08
        config.joint_noise_scale = 0.12
        config.first_ctrl_noise_scale = 0.6
        config.last_ctrl_noise_scale = 0.7
        config.final_noise_scale = 0.8
        config.sigma_decay = 0.75
        config.mjx_min_score_improvement = 0.01
        config.use_warm_start = False

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
            command_builder=_fake_command_builder,
        )

        self.assertEqual(result.metadata["planning_horizon_steps"], 11)
        self.assertEqual(result.metadata["control_steps"], 7)
        self.assertEqual(result.metadata["sample_count"], 9)
        self.assertEqual(result.metadata["optimizer_iterations"], 3)
        self.assertEqual(result.metadata["knot_count"], 5)
        self.assertEqual(result.metadata["temperature"], 0.9)
        self.assertEqual(result.metadata["root_pos_sigma"], 0.03)
        self.assertEqual(result.metadata["root_rot_sigma"], 0.08)
        self.assertEqual(result.metadata["joint_sigma"], 0.12)
        self.assertEqual(result.metadata["first_ctrl_noise_scale"], 0.6)
        self.assertEqual(result.metadata["last_ctrl_noise_scale"], 0.7)
        self.assertEqual(result.metadata["final_noise_scale"], 0.8)
        self.assertEqual(result.metadata["sigma_decay"], 0.75)
        self.assertEqual(result.metadata["mjx_min_score_improvement"], 0.01)
        self.assertFalse(result.metadata["use_warm_start"])

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

    def test_default_optimizer_accepts_noop_when_scorer_favors_current_controls(
        self,
    ) -> None:
        def rollout_scorer(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            return -np.sum(np.asarray(samples, dtype=np.float32) ** 2, axis=(1, 2))

        result = _run_with_fakes(
            optimizer=None,
            rollout_factory=_fake_rollout_result,
            runtime=_FakeOptimizerRuntime(),
            rollout_scorer=rollout_scorer,
        )

        self.assertTrue(result.metadata["accepted"])
        self.assertEqual(result.metadata["accepted_windows"], 40)
        self.assertTrue(result.result.infos[0]["accepted"])
        self.assertEqual(int(result.result.infos[0]["accepted_iterations"]), 0)
        self.assertTrue(bool(result.result.infos[0]["current_controls_selected"]))

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

    def test_jax_warmup_primes_execute_rollout_tracer_before_steady_loop(self) -> None:
        events: list[str] = []
        include_guided_flags: list[bool] = []

        def rollout_scorer(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            events.append("score")
            sample_array = np.asarray(samples, dtype=np.float32)
            return {
                "score": np.arange(int(sample_array.shape[0]), dtype=np.float32),
                "physics_step_count": np.full(
                    int(sample_array.shape[0]),
                    40,
                    dtype=np.float32,
                ),
            }

        def rollout_reference_factory(**kwargs):
            events.append("reference")
            include_guided_flags.append(kwargs.get("include_guided_candidate", True))
            return {"start": kwargs["start"]}

        def rollout_tracer(samples, reference, actor_params, model_bundle):
            del samples, actor_params, model_bundle
            events.append("trace")
            bodies = len(MUJOCO_BODY_NAMES)
            qpos = np.zeros((1, QPOS_DIM), dtype=np.float32)
            qpos[:, 3] = 1.0
            qvel = np.zeros((1, QVEL_DIM), dtype=np.float32)
            body_quat = np.zeros((1, bodies, 4), dtype=np.float32)
            body_quat[..., 0] = 1.0
            return {
                "final_robot_state": {
                    "qpos": qpos,
                    "qvel": qvel,
                    "body_pos_w": np.zeros((1, bodies, 3), dtype=np.float32),
                    "body_quat_w": body_quat,
                    "body_lin_vel_w": np.zeros((1, bodies, 3), dtype=np.float32),
                    "body_ang_vel_w": np.zeros((1, bodies, 3), dtype=np.float32),
                },
                "final_obs_state": SimpleNamespace(history={}, last_action=None),
                "final_prev_control": np.zeros((1, QPOS_DIM - 1), dtype=np.float32),
                "final_prev_joint_acc": np.zeros((1, ACTION_DIM), dtype=np.float32),
                "final_prev_contact": np.zeros((1, 2), dtype=np.float32),
                "final_prev_contact_valid": np.ones((1,), dtype=np.float32),
                "final_prev_contact_force": np.zeros((1, 2), dtype=np.float32),
                "final_prev_contact_force_valid": np.ones((1,), dtype=np.float32),
            }

        result = _run_with_fakes(
            optimizer=None,
            rollout_factory=_fake_rollout_result,
            runtime=_FakeOptimizerRuntime(),
            rollout_scorer=rollout_scorer,
            rollout_reference_factory=rollout_reference_factory,
            rollout_tracer=rollout_tracer,
        )

        self.assertTrue(result.metadata["accepted"])
        score_indices = [index for index, event in enumerate(events) if event == "score"]
        trace_indices = [index for index, event in enumerate(events) if event == "trace"]
        self.assertGreaterEqual(len(score_indices), 2)
        self.assertGreaterEqual(len(trace_indices), 2)
        self.assertLess(trace_indices[0], score_indices[1])
        self.assertEqual(include_guided_flags[:4], [True, False, True, False])

    def test_guided_candidate_period_applies_to_steady_windows(self) -> None:
        reference_calls: list[dict[str, object]] = []

        def rollout_scorer(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            sample_array = np.asarray(samples, dtype=np.float32)
            return {
                "score": np.arange(int(sample_array.shape[0]), dtype=np.float32),
                "physics_step_count": np.full(
                    int(sample_array.shape[0]),
                    40,
                    dtype=np.float32,
                ),
            }

        def rollout_reference_factory(**kwargs):
            reference_calls.append(
                {
                    "include_guided_candidate": kwargs.get(
                        "include_guided_candidate",
                        True,
                    ),
                    "control_count": int(np.asarray(kwargs["controls"]).shape[0]),
                }
            )
            return {"start": kwargs["start"]}

        def rollout_tracer(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            bodies = len(MUJOCO_BODY_NAMES)
            qpos = np.zeros((1, QPOS_DIM), dtype=np.float32)
            qpos[:, 3] = 1.0
            qvel = np.zeros((1, QVEL_DIM), dtype=np.float32)
            body_quat = np.zeros((1, bodies, 4), dtype=np.float32)
            body_quat[..., 0] = 1.0
            return {
                "final_robot_state": {
                    "qpos": qpos,
                    "qvel": qvel,
                    "body_pos_w": np.zeros((1, bodies, 3), dtype=np.float32),
                    "body_quat_w": body_quat,
                    "body_lin_vel_w": np.zeros((1, bodies, 3), dtype=np.float32),
                    "body_ang_vel_w": np.zeros((1, bodies, 3), dtype=np.float32),
                },
                "final_obs_state": SimpleNamespace(history={}, last_action=None),
                "final_prev_control": np.zeros((1, QPOS_DIM - 1), dtype=np.float32),
                "final_prev_joint_acc": np.zeros((1, ACTION_DIM), dtype=np.float32),
                "final_prev_contact": np.zeros((1, 2), dtype=np.float32),
                "final_prev_contact_valid": np.ones((1,), dtype=np.float32),
                "final_prev_contact_force": np.zeros((1, 2), dtype=np.float32),
                "final_prev_contact_force_valid": np.ones((1,), dtype=np.float32),
            }

        spider_config = _spider_config()
        spider_config.use_guided_candidate = True
        spider_config.guided_candidate_period = 3
        result = _run_with_fakes(
            optimizer=None,
            rollout_factory=_fake_rollout_result,
            spider_config=spider_config,
            runtime=_FakeOptimizerRuntime(),
            rollout_scorer=rollout_scorer,
            rollout_reference_factory=rollout_reference_factory,
            rollout_tracer=rollout_tracer,
        )

        expected = [index % 3 == 0 for index in range(40)]
        steady_flags = [
            info["guided_candidate_included"] for info in result.result.infos
        ]
        horizon_reference_flags = [
            bool(call["include_guided_candidate"])
            for call in reference_calls
            if int(call["control_count"]) == 40
        ]
        self.assertEqual(steady_flags, expected)
        self.assertEqual(horizon_reference_flags[:3], [True, True, False])
        self.assertEqual(horizon_reference_flags[3:], expected)
        self.assertEqual(result.metadata["guided_candidate_period"], 3)
        self.assertEqual(result.metadata["guided_candidate_windows"], 14)

    def test_default_optimizer_outer_jit_keeps_rollout_tracer_execute_source(
        self,
    ) -> None:
        class RecordingJax(_FakeOptimizerJax):
            def __init__(self) -> None:
                self.jit_calls = 0

            def jit(self, fn):
                self.jit_calls += 1

                def wrapped(*args):
                    return fn(*args)

                return wrapped

        runtime = SimpleNamespace(jnp=_FakeOptimizerJnp(), jax=RecordingJax())

        def rollout_scorer(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            sample_array = np.asarray(samples, dtype=np.float32)
            return {
                "score": np.arange(int(sample_array.shape[0]), dtype=np.float32),
                "physics_step_count": np.full(
                    int(sample_array.shape[0]),
                    40,
                    dtype=np.float32,
                ),
            }

        def rollout_reference_factory(**kwargs):
            return {"start": kwargs["start"]}

        def rollout_tracer(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            bodies = len(MUJOCO_BODY_NAMES)
            qpos = np.zeros((1, QPOS_DIM), dtype=np.float32)
            qpos[:, 3] = 1.0
            qvel = np.zeros((1, QVEL_DIM), dtype=np.float32)
            body_quat = np.zeros((1, bodies, 4), dtype=np.float32)
            body_quat[..., 0] = 1.0
            return {
                "final_robot_state": {
                    "qpos": qpos,
                    "qvel": qvel,
                    "body_pos_w": np.zeros((1, bodies, 3), dtype=np.float32),
                    "body_quat_w": body_quat,
                    "body_lin_vel_w": np.zeros((1, bodies, 3), dtype=np.float32),
                    "body_ang_vel_w": np.zeros((1, bodies, 3), dtype=np.float32),
                },
                "final_obs_state": SimpleNamespace(history={}, last_action=None),
                "final_prev_control": np.zeros((1, QPOS_DIM - 1), dtype=np.float32),
                "final_prev_joint_acc": np.zeros((1, ACTION_DIM), dtype=np.float32),
                "final_prev_contact": np.zeros((1, 2), dtype=np.float32),
                "final_prev_contact_valid": np.ones((1,), dtype=np.float32),
                "final_prev_contact_force": np.zeros((1, 2), dtype=np.float32),
                "final_prev_contact_force_valid": np.ones((1,), dtype=np.float32),
            }

        result = _run_with_fakes(
            optimizer=None,
            rollout_factory=_fake_rollout_result,
            runtime=runtime,
            rollout_scorer=rollout_scorer,
            rollout_reference_factory=rollout_reference_factory,
            rollout_tracer=rollout_tracer,
        )

        self.assertGreaterEqual(runtime.jax.jit_calls, 1)
        self.assertEqual(
            result.metadata["execute_trace_source_counts"],
            {"rollout_tracer": 40},
        )
        self.assertTrue(
            all(
                info["execute_trace_source"] == "rollout_tracer"
                for info in result.result.infos
            )
        )

    def test_jax_warmup_primes_live_state_rollout_signature_before_steady_loop(
        self,
    ) -> None:
        events: list[str] = []

        def rollout_scorer(samples, reference, actor_params, model_bundle):
            del samples, actor_params, model_bundle
            events.append(
                "score_live" if "initial_robot_state" in reference else "score_cold"
            )
            return {
                "score": np.arange(4, dtype=np.float32),
                "physics_step_count": np.full(4, 40, dtype=np.float32),
            }

        def rollout_reference_factory(**kwargs):
            events.append(
                "reference_live"
                if "initial_robot_state" in kwargs
                else "reference_cold"
            )
            return dict(kwargs)

        def rollout_tracer(samples, reference, actor_params, model_bundle):
            del actor_params, model_bundle
            self.assertEqual(
                int(np.asarray(samples).shape[1]),
                int(np.asarray(reference["controls"]).shape[0]),
            )
            events.append("trace")
            bodies = len(MUJOCO_BODY_NAMES)
            qpos = np.zeros((1, QPOS_DIM), dtype=np.float32)
            qpos[:, 3] = 1.0
            qvel = np.zeros((1, QVEL_DIM), dtype=np.float32)
            body_quat = np.zeros((1, bodies, 4), dtype=np.float32)
            body_quat[..., 0] = 1.0
            return {
                "final_robot_state": {
                    "qpos": qpos,
                    "qvel": qvel,
                    "body_pos_w": np.zeros((1, bodies, 3), dtype=np.float32),
                    "body_quat_w": body_quat,
                    "body_lin_vel_w": np.zeros((1, bodies, 3), dtype=np.float32),
                    "body_ang_vel_w": np.zeros((1, bodies, 3), dtype=np.float32),
                },
                "final_obs_state": SimpleNamespace(history={}, last_action=None),
                "final_prev_control": np.zeros((1, QPOS_DIM - 1), dtype=np.float32),
                "final_prev_joint_acc": np.zeros((1, ACTION_DIM), dtype=np.float32),
                "final_prev_contact": np.zeros((1, 2), dtype=np.float32),
                "final_prev_contact_valid": np.ones((1,), dtype=np.float32),
                "final_prev_contact_force": np.zeros((1, 2), dtype=np.float32),
                "final_prev_contact_force_valid": np.ones((1,), dtype=np.float32),
            }

        _run_with_fakes(
            optimizer=None,
            rollout_factory=_fake_rollout_result,
            runtime=_FakeOptimizerRuntime(),
            rollout_scorer=rollout_scorer,
            rollout_reference_factory=rollout_reference_factory,
            rollout_tracer=rollout_tracer,
        )

        live_score_index = events.index("score_live")
        trace_indices = [index for index, event in enumerate(events) if event == "trace"]
        self.assertGreaterEqual(len(trace_indices), 2)
        self.assertLess(live_score_index, trace_indices[1])

    def test_default_optimizer_uses_explicit_rollout_reference_factory(self) -> None:
        references: list[dict[str, object]] = []
        captured_guided: list[np.ndarray] = []
        guided_controls = np.full((40, QPOS_DIM - 1), 0.75, dtype=np.float32)

        def rollout_scorer(samples, reference, actor_params, model_bundle):
            del actor_params, model_bundle
            references.append(dict(reference))
            captured_guided.append(np.asarray(samples[1], dtype=np.float32).copy())
            scale = float(reference["score_scale"])
            return scale * np.arange(int(samples.shape[0]), dtype=np.float32)

        def rollout_reference_factory(**kwargs):
            return {
                "window_start": kwargs["start"],
                "score_scale": 0.5,
                "guided_controls": guided_controls,
            }

        spider_config = _spider_config()
        spider_config.use_guided_candidate = True
        result = _run_with_fakes(
            optimizer=None,
            rollout_factory=_fake_rollout_result,
            spider_config=spider_config,
            runtime=_FakeOptimizerRuntime(),
            rollout_scorer=rollout_scorer,
            rollout_reference_factory=rollout_reference_factory,
        )

        self.assertTrue(result.metadata["accepted"])
        self.assertGreater(len(references), 0)
        self.assertEqual(references[0]["window_start"], 0)
        self.assertEqual(references[0]["score_scale"], 0.5)
        np.testing.assert_allclose(captured_guided[0], guided_controls)

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
            command_builder=_fake_command_builder,
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

    def test_mjx_backend_default_path_wires_production_scan_components(self) -> None:
        calls: list[str] = []

        def rollout_scorer(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            sample_count = int(samples.shape[0])
            return {
                "score": np.arange(sample_count, dtype=np.float32),
                "physics_step_count": np.full(sample_count, 40, dtype=np.float32),
            }

        def rollout_reference_factory(**kwargs):
            return {"start": kwargs["start"]}

        def default_components(**kwargs):
            calls.append("components")
            self.assertIs(kwargs["runtime"], runtime)
            self.assertTrue(kwargs["use_guided_candidate"])
            self.assertEqual(kwargs["guided_joint_gain"], 0.5)
            self.assertEqual(kwargs["contact_force_mode"], "sum_rows")
            self.assertFalse(kwargs["contact_force_first_row_diagnostics"])
            return SimpleNamespace(
                rollout_scorer=rollout_scorer,
                rollout_reference_factory=rollout_reference_factory,
            )

        def default_rollout_factory(config):
            calls.append("rollout_factory")
            self.assertEqual(config.device, "cpu")
            return _fake_rollout_result

        runtime = _FakeOptimizerRuntime()
        spider_config = _spider_config()
        spider_config.use_guided_candidate = True
        spider_config.guided_joint_gain = 0.5
        with (
            mock.patch.object(
                mjx_backend_module,
                "_default_rollout_components",
                side_effect=default_components,
            ),
            mock.patch.object(
                mjx_backend_module,
                "_default_static_rollout_factory",
                side_effect=default_rollout_factory,
            ),
        ):
            result = run_g1_wbc_mjx_mpc(
                spider_config=spider_config,
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
                command_builder=_fake_command_builder,
            )

        self.assertEqual(calls, ["components", "rollout_factory"])
        self.assertTrue(result.metadata["accepted"])
        self.assertTrue(result.metadata["physics_scan_enabled"])
        self.assertEqual(result.metadata["physics_step_count_min"], 40)
        self.assertEqual(result.metadata["physics_step_count_windows"], 40)
        self.assertEqual(result.metadata["contact_force_mode"], "sum_rows")
        self.assertFalse(result.metadata["contact_force_first_row_diagnostics"])

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

    def test_mjx_backend_can_disable_warm_start_shift(self) -> None:
        captured: list[torch.Tensor] = []
        spider_config = _spider_config()
        spider_config.use_warm_start = False

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

        _run_with_fakes(
            optimizer=optimizer,
            rollout_factory=_fake_rollout_result,
            spider_config=spider_config,
        )

        self.assertGreaterEqual(len(captured), 2)
        self.assertTrue(torch.allclose(captured[1][:, 0], torch.zeros(40)))

    def test_mjx_backend_carries_execute_trace_state_between_windows(self) -> None:
        optimizer_references: list[dict[str, object]] = []
        tracer_references: list[dict[str, object]] = []
        live_obs_state = SimpleNamespace(
            history={},
            last_action=np.zeros((1, ACTION_DIM), dtype=np.float32),
        )
        live_prev_control = np.full((1, QPOS_DIM - 1), 0.33, dtype=np.float32)
        live_prev_joint_acc = np.full((1, ACTION_DIM), 0.21, dtype=np.float32)
        live_prev_contact = np.array([[1.0, 0.0]], dtype=np.float32)
        live_prev_contact_force = np.array([[15.0, 25.0]], dtype=np.float32)

        def optimizer(**kwargs):
            optimizer_references.append(dict(kwargs["reference"]["kwargs"]))
            updated = torch.zeros(40, QPOS_DIM - 1)
            chunk = torch.zeros(21, QPOS_DIM - 1)
            return SimpleNamespace(
                updated_controls=updated,
                execute_chunk=chunk,
                info={"best_score": torch.tensor(1.25), "accepted": True},
            )

        def rollout_reference_factory(**kwargs):
            reference_kwargs = dict(kwargs)
            return {
                "start": kwargs["start"],
                "kwargs": reference_kwargs,
            }

        def rollout_tracer(samples, reference, actor_params, model_bundle):
            del samples, actor_params, model_bundle
            tracer_references.append(dict(reference["kwargs"]))
            bodies = len(MUJOCO_BODY_NAMES)
            qpos = np.zeros((1, QPOS_DIM), dtype=np.float32)
            qpos[:, 0] = 123.0 + float(reference["start"])
            qpos[:, 3] = 1.0
            qvel = np.full((1, QVEL_DIM), 0.5, dtype=np.float32)
            body_quat = np.zeros((1, bodies, 4), dtype=np.float32)
            body_quat[..., 0] = 1.0
            return {
                "final_robot_state": {
                    "qpos": qpos,
                    "qvel": qvel,
                    "body_pos_w": np.full((1, bodies, 3), 1.5, dtype=np.float32),
                    "body_quat_w": body_quat,
                    "body_lin_vel_w": np.full((1, bodies, 3), 2.5, dtype=np.float32),
                    "body_ang_vel_w": np.full((1, bodies, 3), 3.5, dtype=np.float32),
                },
                "final_obs_state": live_obs_state,
                "final_prev_control": live_prev_control,
                "final_prev_joint_acc": live_prev_joint_acc,
                "final_prev_contact": live_prev_contact,
                "final_prev_contact_valid": np.array([1.0], dtype=np.float32),
                "final_prev_contact_force": live_prev_contact_force,
                "final_prev_contact_force_valid": np.array([1.0], dtype=np.float32),
            }

        _run_with_fakes(
            optimizer=optimizer,
            rollout_factory=_fake_rollout_result,
            rollout_reference_factory=rollout_reference_factory,
            rollout_tracer=rollout_tracer,
        )

        self.assertGreaterEqual(len(optimizer_references), 2)
        self.assertGreaterEqual(len(tracer_references), 1)
        self.assertNotIn("initial_robot_state", optimizer_references[0])
        self.assertEqual(tracer_references[0]["start"], 0)
        second_reference = optimizer_references[1]
        np.testing.assert_allclose(
            second_reference["initial_robot_state"]["qpos"][:, 0],
            [123.0],
        )
        np.testing.assert_allclose(
            second_reference["initial_robot_state"]["qvel"],
            0.5,
        )
        self.assertIs(second_reference["obs_state"], live_obs_state)
        self.assertTrue(second_reference["obs_initialized"])
        np.testing.assert_allclose(second_reference["prev_control"], live_prev_control)
        np.testing.assert_allclose(
            second_reference["prev_joint_acc"],
            live_prev_joint_acc,
        )
        np.testing.assert_allclose(second_reference["prev_contact"], live_prev_contact)
        np.testing.assert_allclose(second_reference["prev_contact_valid"], [1.0])
        np.testing.assert_allclose(
            second_reference["prev_contact_force"],
            live_prev_contact_force,
        )
        np.testing.assert_allclose(
            second_reference["prev_contact_force_valid"],
            [1.0],
        )

    def test_mjx_backend_can_strip_live_mjx_data_between_windows(self) -> None:
        optimizer_references: list[dict[str, object]] = []
        mjx_data = object()

        def optimizer(**kwargs):
            optimizer_references.append(dict(kwargs["reference"]["kwargs"]))
            updated = torch.zeros(40, QPOS_DIM - 1)
            chunk = torch.zeros(21, QPOS_DIM - 1)
            return SimpleNamespace(
                updated_controls=updated,
                execute_chunk=chunk,
                info={"best_score": torch.tensor(1.25), "accepted": True},
            )

        def rollout_reference_factory(**kwargs):
            return {"start": kwargs["start"], "kwargs": dict(kwargs)}

        def rollout_tracer(samples, reference, actor_params, model_bundle):
            del samples, actor_params, model_bundle
            bodies = len(MUJOCO_BODY_NAMES)
            qpos = np.zeros((1, QPOS_DIM), dtype=np.float32)
            qpos[:, 3] = 1.0
            qvel = np.full((1, QVEL_DIM), 0.5, dtype=np.float32)
            body_quat = np.zeros((1, bodies, 4), dtype=np.float32)
            body_quat[..., 0] = 1.0
            return {
                "final_robot_state": {
                    "qpos": qpos,
                    "qvel": qvel,
                    "body_pos_w": np.full((1, bodies, 3), 1.5, dtype=np.float32),
                    "body_quat_w": body_quat,
                    "body_lin_vel_w": np.full((1, bodies, 3), 2.5, dtype=np.float32),
                    "body_ang_vel_w": np.full((1, bodies, 3), 3.5, dtype=np.float32),
                    "mjx_data": mjx_data,
                },
            }

        result = _run_with_fakes(
            optimizer=optimizer,
            rollout_factory=_fake_rollout_result,
            rollout_reference_factory=rollout_reference_factory,
            rollout_tracer=rollout_tracer,
            strip_live_mjx_data_between_windows=True,
        )

        second_reference = optimizer_references[1]
        second_robot_state = second_reference["initial_robot_state"]
        self.assertNotIn("mjx_data", second_robot_state)
        np.testing.assert_allclose(second_robot_state["qvel"], 0.5)
        self.assertTrue(result.metadata["strip_live_mjx_data_between_windows"])

    def test_mjx_backend_records_score_only_optimizer_diagnostic(self) -> None:
        def optimizer(**kwargs):
            window_config = kwargs["config"]
            updated = torch.zeros(
                int(window_config.horizon_steps),
                QPOS_DIM - 1,
            )
            chunk = torch.zeros(
                int(window_config.control_steps) + 1,
                QPOS_DIM - 1,
            )
            return SimpleNamespace(
                updated_controls=updated,
                execute_chunk=chunk,
                info={"best_score": torch.tensor(1.25), "accepted": True},
            )

        result = _run_with_fakes(
            optimizer=optimizer,
            rollout_factory=_fake_rollout_result,
            rollout_reference_factory=lambda **kwargs: {"kwargs": dict(kwargs)},
            score_only_optimizer=True,
        )

        self.assertTrue(result.metadata["score_only_optimizer"])

    def test_mjx_backend_records_contact_force_first_row_diagnostic(self) -> None:
        result = _run_with_fakes(
            optimizer=_fake_optimizer,
            rollout_factory=_fake_rollout_result,
            contact_force_first_row_diagnostics=True,
        )

        self.assertTrue(result.metadata["contact_force_first_row_diagnostics"])

    def test_mjx_backend_records_contact_force_mode_diagnostic(self) -> None:
        result = _run_with_fakes(
            optimizer=_fake_optimizer,
            rollout_factory=_fake_rollout_result,
            contact_force_mode="first_row",
        )

        self.assertEqual(result.metadata["contact_force_mode"], "first_row")

    def test_mjx_backend_builds_rollout_from_execute_trace_when_available(self) -> None:
        config = _spider_config()
        config.horizon_steps = 4
        config.ctrl_steps = 2
        total_steps = 4
        trace_starts: list[int] = []

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

        def rollout_reference_factory(**kwargs):
            return {"start": kwargs["start"], "kwargs": dict(kwargs)}

        def rollout_tracer(samples, reference, actor_params, model_bundle):
            del actor_params, model_bundle
            start = int(reference["start"])
            trace_starts.append(start)
            steps = int(samples.shape[1])
            frames = steps + 1
            bodies = len(MUJOCO_BODY_NAMES)
            qpos = np.zeros((frames, 1, QPOS_DIM), dtype=np.float32)
            qpos[..., 3] = 1.0
            qpos[:, 0, 0] = np.arange(frames, dtype=np.float32) + 10.0 + start
            qvel = np.full((frames, 1, QVEL_DIM), 0.5 + start, dtype=np.float32)
            body_quat = np.zeros((frames, 1, bodies, 4), dtype=np.float32)
            body_quat[..., 0] = 1.0
            actions = np.full((steps, 1, ACTION_DIM), 0.6 + start, dtype=np.float32)
            controls = np.full((steps, 1, ACTION_DIM), 0.7 + start, dtype=np.float32)
            contact = np.zeros((frames, 1, 2), dtype=np.float32)
            contact[1:, :, 0] = 1.0
            floor_contact = np.zeros((frames, 1, 3), dtype=np.float32)
            floor_contact[..., :2] = contact
            return {
                "qpos": qpos,
                "qvel": qvel,
                "body_pos_w": np.full((frames, 1, bodies, 3), 1.5 + start, dtype=np.float32),
                "body_quat_w": body_quat,
                "body_lin_vel_w": np.full((frames, 1, bodies, 3), 2.5 + start, dtype=np.float32),
                "body_ang_vel_w": np.full((frames, 1, bodies, 3), 3.5 + start, dtype=np.float32),
                "actions": actions,
                "controls": controls,
                "contact_indicator": contact,
                "contact_force": np.full((frames, 1, 2), 4.5 + start, dtype=np.float32),
                "contact_force_first_row": np.full(
                    (frames, 1, 2),
                    1.5 + start,
                    dtype=np.float32,
                ),
                "floor_contact_indicator": floor_contact,
                "floor_contact_force": np.full((frames, 1, 3), 5.5 + start, dtype=np.float32),
                "floor_contact_force_first_row": np.full(
                    (frames, 1, 3),
                    2.5 + start,
                    dtype=np.float32,
                ),
                "floor_contact_force_peak_source": np.full(
                    (frames, 1, 3, 8),
                    8.5 + start,
                    dtype=np.float32,
                ),
                "final_robot_state": {
                    "qpos": qpos[-1],
                    "qvel": qvel[-1],
                    "body_pos_w": np.full((1, bodies, 3), 1.5 + start, dtype=np.float32),
                    "body_quat_w": body_quat[-1],
                    "body_lin_vel_w": np.full((1, bodies, 3), 2.5 + start, dtype=np.float32),
                    "body_ang_vel_w": np.full((1, bodies, 3), 3.5 + start, dtype=np.float32),
                },
                "final_obs_state": SimpleNamespace(history={}, last_action=actions[-1]),
                "final_prev_control": samples[:, -1],
                "final_prev_joint_acc": np.zeros((1, ACTION_DIM), dtype=np.float32),
                "final_prev_contact": contact[-1, :, :],
                "final_prev_contact_valid": np.array([1.0], dtype=np.float32),
                "final_prev_contact_force": np.full((1, 2), 4.5 + start, dtype=np.float32),
                "final_prev_contact_force_valid": np.array([1.0], dtype=np.float32),
            }

        result = run_g1_wbc_mjx_mpc(
            spider_config=config,
            motion=_motion(frames=total_steps + 1),
            actor=WbcActor(input_dim=4, hidden_dims=(), output_dim=2),
            rollout_config=SimpleNamespace(device="cpu", max_steps=total_steps),
            execute_rollout_config=SimpleNamespace(device="cpu", max_steps=total_steps),
            method="g1_wbc_joint_global",
            reward_weights=None,
            total_steps=total_steps,
            seed=5,
            runtime=SimpleNamespace(jnp=SimpleNamespace(), jax=SimpleNamespace()),
            model_factory=lambda **kwargs: _fake_model_bundle(
                profile_name=kwargs["profile_name"]
            ),
            policy_converter=lambda actor, *, jnp: SimpleNamespace(params=True),
            optimizer=optimizer,
            rollout_factory=_fake_rollout_result,
            rollout_reference_factory=rollout_reference_factory,
            rollout_tracer=rollout_tracer,
            command_builder=_fake_command_builder,
        )

        self.assertEqual(trace_starts, [0, 2])
        self.assertEqual(result.metadata["rollout_source"], "dynamic_execute_trace")
        self.assertTrue(result.metadata["rollout_dynamic_execute_trace"])
        self.assertEqual(result.metadata["execute_trace_chunks"], 2)
        rollout = result.result.rollout
        self.assertEqual(tuple(rollout.qpos.shape), (total_steps + 1, 1, QPOS_DIM))
        torch.testing.assert_close(
            rollout.qpos[:, 0, 0],
            torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0]),
        )
        torch.testing.assert_close(rollout.actions[:2], torch.full((2, 1, ACTION_DIM), 0.6))
        torch.testing.assert_close(rollout.actions[2:], torch.full((2, 1, ACTION_DIM), 2.6))
        torch.testing.assert_close(rollout.controls[:2], torch.full((2, 1, ACTION_DIM), 0.7))
        self.assertIsNotNone(rollout.contact_force_first_row)
        self.assertIsNotNone(rollout.floor_contact_force_first_row)
        self.assertIsNotNone(rollout.floor_contact_force_peak_source)
        torch.testing.assert_close(
            rollout.contact_force_first_row[:3],
            torch.full((3, 1, 2), 1.5),
        )
        torch.testing.assert_close(
            rollout.contact_force_first_row[3:],
            torch.full((2, 1, 2), 3.5),
        )
        torch.testing.assert_close(
            rollout.floor_contact_force_peak_source[:3],
            torch.full((3, 1, 3, 8), 8.5),
        )
        torch.testing.assert_close(
            rollout.floor_contact_force_peak_source[3:],
            torch.full((2, 1, 3, 8), 10.5),
        )
        torch.testing.assert_close(rollout.controls[2:], torch.full((2, 1, ACTION_DIM), 2.7))
        torch.testing.assert_close(rollout.contact_indicator[1:, :, 0], torch.ones(4, 1))
        torch.testing.assert_close(
            rollout.ref_indices[:, 0],
            torch.tensor([0, *range(total_steps)]),
        )
        self.assertFalse(torch.allclose(rollout.qpos[:, 0], result.result.refined_qpos))

    def test_mjx_backend_exports_replay_state_for_executed_command_chunks(self) -> None:
        config = _spider_config()
        config.horizon_steps = 4
        config.ctrl_steps = 2
        total_steps = 4
        history = np.arange(
            OBS_HISTORY_LENGTH * ACTION_DIM,
            dtype=np.float32,
        ).reshape(1, OBS_HISTORY_LENGTH, ACTION_DIM)

        def optimizer(**kwargs):
            window_config = kwargs["config"]
            updated = torch.zeros(int(window_config.horizon_steps), QPOS_DIM - 1)
            chunk = torch.zeros(int(window_config.control_steps) + 1, QPOS_DIM - 1)
            return SimpleNamespace(
                updated_controls=updated,
                execute_chunk=chunk,
                info={"best_score": torch.tensor(1.25), "accepted": True},
            )

        def rollout_reference_factory(**kwargs):
            return {"start": kwargs["start"], "kwargs": dict(kwargs)}

        def rollout_tracer(samples, reference, actor_params, model_bundle):
            del actor_params, model_bundle
            start = int(reference["start"])
            steps = int(samples.shape[1])
            frames = steps + 1
            bodies = len(MUJOCO_BODY_NAMES)
            qpos = np.zeros((frames, 1, QPOS_DIM), dtype=np.float32)
            qpos[..., 3] = 1.0
            qpos[:, 0, 0] = np.arange(frames, dtype=np.float32) + 10.0 + start
            qvel = np.full((frames, 1, QVEL_DIM), 0.5 + start, dtype=np.float32)
            body_quat = np.zeros((frames, 1, bodies, 4), dtype=np.float32)
            body_quat[..., 0] = 1.0
            actions = np.full((steps, 1, ACTION_DIM), 0.6 + start, dtype=np.float32)
            controls = np.full((steps, 1, ACTION_DIM), 0.7 + start, dtype=np.float32)
            contact = np.zeros((frames, 1, 2), dtype=np.float32)
            floor_contact = np.zeros((frames, 1, 3), dtype=np.float32)
            floor_contact[..., :2] = contact
            return {
                "qpos": qpos,
                "qvel": qvel,
                "body_pos_w": np.full((frames, 1, bodies, 3), 1.5 + start, dtype=np.float32),
                "body_quat_w": body_quat,
                "body_lin_vel_w": np.full((frames, 1, bodies, 3), 2.5 + start, dtype=np.float32),
                "body_ang_vel_w": np.full((frames, 1, bodies, 3), 3.5 + start, dtype=np.float32),
                "actions": actions,
                "controls": controls,
                "contact_indicator": contact,
                "contact_force": np.full((frames, 1, 2), 4.5 + start, dtype=np.float32),
                "floor_contact_indicator": floor_contact,
                "floor_contact_force": np.full((frames, 1, 3), 5.5 + start, dtype=np.float32),
                "final_robot_state": {
                    "qpos": qpos[-1],
                    "qvel": qvel[-1],
                    "body_pos_w": np.full((1, bodies, 3), 1.5 + start, dtype=np.float32),
                    "body_quat_w": body_quat[-1],
                    "body_lin_vel_w": np.full((1, bodies, 3), 2.5 + start, dtype=np.float32),
                    "body_ang_vel_w": np.full((1, bodies, 3), 3.5 + start, dtype=np.float32),
                },
                "final_obs_state": SimpleNamespace(
                    history={"actions": history + start},
                    last_action=actions[-1],
                ),
                "final_prev_control": samples[:, -1],
                "final_prev_joint_acc": np.zeros((1, ACTION_DIM), dtype=np.float32),
                "final_prev_contact": contact[-1, :, :],
                "final_prev_contact_valid": np.array([1.0], dtype=np.float32),
                "final_prev_contact_force": np.full((1, 2), 4.5 + start, dtype=np.float32),
                "final_prev_contact_force_valid": np.array([1.0], dtype=np.float32),
            }

        result = run_g1_wbc_mjx_mpc(
            spider_config=config,
            motion=_motion(frames=total_steps + 1),
            actor=WbcActor(input_dim=4, hidden_dims=(), output_dim=2),
            rollout_config=SimpleNamespace(device="cpu", max_steps=total_steps),
            execute_rollout_config=SimpleNamespace(device="cpu", max_steps=total_steps),
            method="g1_wbc_joint_global",
            reward_weights=None,
            total_steps=total_steps,
            seed=5,
            runtime=SimpleNamespace(jnp=SimpleNamespace(), jax=SimpleNamespace()),
            model_factory=lambda **kwargs: _fake_model_bundle(
                profile_name=kwargs["profile_name"]
            ),
            policy_converter=lambda actor, *, jnp: SimpleNamespace(params=True),
            optimizer=optimizer,
            rollout_factory=_fake_rollout_result,
            rollout_reference_factory=rollout_reference_factory,
            rollout_tracer=rollout_tracer,
            command_builder=_fake_command_builder,
        )

        chunks = result.result.executed_command_chunks
        self.assertEqual(len(chunks), 2)
        first_state = chunks[0].replay_state
        second_state = chunks[1].replay_state
        self.assertIsNotNone(first_state)
        self.assertIsNotNone(second_state)
        assert first_state is not None
        assert second_state is not None
        expected_motion = _motion(5)
        torch.testing.assert_close(first_state.initial_qpos, expected_motion.qpos()[0])
        torch.testing.assert_close(first_state.initial_qvel, expected_motion.qvel()[0])
        self.assertIsNone(first_state.initial_last_action)
        torch.testing.assert_close(second_state.initial_qpos[0], torch.tensor(12.0))
        torch.testing.assert_close(second_state.initial_qvel, torch.full((QVEL_DIM,), 0.5))
        torch.testing.assert_close(
            second_state.initial_last_action,
            torch.full((ACTION_DIM,), 0.6),
        )
        actions_history = second_state.initial_history_state["actions"]
        self.assertEqual(actions_history["pointer"], OBS_HISTORY_LENGTH - 1)
        torch.testing.assert_close(
            actions_history["num_pushes"],
            torch.full((1,), OBS_HISTORY_LENGTH, dtype=torch.long),
        )
        torch.testing.assert_close(
            actions_history["buffer"],
            torch.tensor(history[0]).view(OBS_HISTORY_LENGTH, 1, ACTION_DIM),
        )

    def test_mjx_backend_defers_full_trace_when_state_advancer_available(self) -> None:
        config = _spider_config()
        config.horizon_steps = 4
        config.ctrl_steps = 2
        total_steps = 4
        optimizer_references: list[dict[str, object]] = []
        advancer_starts: list[int] = []
        trace_starts: list[int] = []

        def optimizer(**kwargs):
            optimizer_references.append(dict(kwargs["reference"]["kwargs"]))
            window_config = kwargs["config"]
            updated = torch.zeros(int(window_config.horizon_steps), QPOS_DIM - 1)
            chunk = torch.zeros(int(window_config.control_steps) + 1, QPOS_DIM - 1)
            chunk[:, 0] = 0.25
            return SimpleNamespace(
                updated_controls=updated,
                execute_chunk=chunk,
                info={"best_score": torch.tensor(1.25), "accepted": True},
            )

        def rollout_reference_factory(**kwargs):
            return {"start": kwargs["start"], "kwargs": dict(kwargs)}

        def final_fields(samples, reference):
            start = int(reference["start"])
            steps = int(samples.shape[1])
            bodies = len(MUJOCO_BODY_NAMES)
            qpos = np.zeros((1, QPOS_DIM), dtype=np.float32)
            qpos[:, 0] = 100.0 + start + steps
            qpos[:, 3] = 1.0
            qvel = np.full((1, QVEL_DIM), 0.5 + start, dtype=np.float32)
            body_quat = np.zeros((1, bodies, 4), dtype=np.float32)
            body_quat[..., 0] = 1.0
            return {
                "final_robot_state": {
                    "qpos": qpos,
                    "qvel": qvel,
                    "body_pos_w": np.full((1, bodies, 3), 1.5 + start, dtype=np.float32),
                    "body_quat_w": body_quat,
                    "body_lin_vel_w": np.full((1, bodies, 3), 2.5 + start, dtype=np.float32),
                    "body_ang_vel_w": np.full((1, bodies, 3), 3.5 + start, dtype=np.float32),
                },
                "final_obs_state": SimpleNamespace(history={}, last_action=np.zeros((1, ACTION_DIM), dtype=np.float32)),
                "final_prev_control": samples[:, -1],
                "final_prev_joint_acc": np.zeros((1, ACTION_DIM), dtype=np.float32),
                "final_prev_contact": np.array([[1.0, 0.0]], dtype=np.float32),
                "final_prev_contact_valid": np.array([1.0], dtype=np.float32),
                "final_prev_contact_force": np.full((1, 2), 4.5 + start, dtype=np.float32),
                "final_prev_contact_force_valid": np.array([1.0], dtype=np.float32),
            }

        def final_only_trace(samples, reference, actor_params, model_bundle):
            del actor_params, model_bundle
            advancer_starts.append(int(reference["start"]))
            return final_fields(samples, reference)

        def rollout_tracer(samples, reference, actor_params, model_bundle):
            del actor_params, model_bundle
            start = int(reference["start"])
            trace_starts.append(start)
            steps = int(samples.shape[1])
            frames = steps + 1
            bodies = len(MUJOCO_BODY_NAMES)
            qpos = np.zeros((frames, 1, QPOS_DIM), dtype=np.float32)
            qpos[..., 3] = 1.0
            qpos[:, 0, 0] = np.arange(frames, dtype=np.float32) + 10.0 + start
            qvel = np.full((frames, 1, QVEL_DIM), 0.5 + start, dtype=np.float32)
            body_quat = np.zeros((frames, 1, bodies, 4), dtype=np.float32)
            body_quat[..., 0] = 1.0
            actions = np.full((steps, 1, ACTION_DIM), 0.6 + start, dtype=np.float32)
            controls = np.full((steps, 1, ACTION_DIM), 0.7 + start, dtype=np.float32)
            contact = np.zeros((frames, 1, 2), dtype=np.float32)
            contact[1:, :, 0] = 1.0
            floor_contact = np.zeros((frames, 1, 3), dtype=np.float32)
            floor_contact[..., :2] = contact
            return {
                "qpos": qpos,
                "qvel": qvel,
                "body_pos_w": np.full((frames, 1, bodies, 3), 1.5 + start, dtype=np.float32),
                "body_quat_w": body_quat,
                "body_lin_vel_w": np.full((frames, 1, bodies, 3), 2.5 + start, dtype=np.float32),
                "body_ang_vel_w": np.full((frames, 1, bodies, 3), 3.5 + start, dtype=np.float32),
                "actions": actions,
                "controls": controls,
                "contact_indicator": contact,
                "contact_force": np.full((frames, 1, 2), 4.5 + start, dtype=np.float32),
                "floor_contact_indicator": floor_contact,
                "floor_contact_force": np.full((frames, 1, 3), 5.5 + start, dtype=np.float32),
                **final_fields(samples, reference),
            }

        result = run_g1_wbc_mjx_mpc(
            spider_config=config,
            motion=_motion(frames=total_steps + 1),
            actor=WbcActor(input_dim=4, hidden_dims=(), output_dim=2),
            rollout_config=SimpleNamespace(device="cpu", max_steps=total_steps),
            execute_rollout_config=SimpleNamespace(device="cpu", max_steps=total_steps),
            method="g1_wbc_joint_global",
            reward_weights=None,
            total_steps=total_steps,
            seed=5,
            runtime=SimpleNamespace(jnp=SimpleNamespace(), jax=SimpleNamespace()),
            model_factory=lambda **kwargs: _fake_model_bundle(
                profile_name=kwargs["profile_name"]
            ),
            policy_converter=lambda actor, *, jnp: SimpleNamespace(params=True),
            optimizer=optimizer,
            rollout_factory=_fake_rollout_result,
            rollout_reference_factory=rollout_reference_factory,
            rollout_tracer=rollout_tracer,
            rollout_state_advancer=final_only_trace,
            command_builder=_fake_command_builder,
        )

        self.assertEqual(advancer_starts, [0, 2])
        self.assertEqual(trace_starts, [0, 2])
        np.testing.assert_allclose(
            optimizer_references[1]["initial_robot_state"]["qpos"][:, 0],
            [102.0],
        )
        self.assertEqual(result.metadata["rollout_source"], "dynamic_execute_trace")
        self.assertTrue(result.metadata["rollout_dynamic_execute_trace"])
        self.assertEqual(result.metadata["execute_trace_chunks"], 2)
        self.assertEqual(result.metadata["deferred_execute_trace_chunks"], 2)
        self.assertEqual(
            result.metadata["execute_trace_source_counts"],
            {"rollout_tracer": 2},
        )
        rollout = result.result.rollout
        torch.testing.assert_close(
            rollout.qpos[:, 0, 0],
            torch.tensor([10.0, 11.0, 12.0, 13.0, 14.0]),
        )

    def test_mjx_backend_reuses_optimizer_execute_trace_when_available(self) -> None:
        config = _spider_config()
        config.horizon_steps = 4
        config.ctrl_steps = 2
        total_steps = 4
        optimizer_references: list[dict[str, object]] = []

        def execute_trace(start: int, steps: int) -> dict[str, object]:
            frames = int(steps) + 1
            bodies = len(MUJOCO_BODY_NAMES)
            qpos = np.zeros((frames, 1, QPOS_DIM), dtype=np.float32)
            qpos[..., 3] = 1.0
            qpos[:, 0, 0] = np.arange(frames, dtype=np.float32) + 50.0 + start
            qvel = np.full((frames, 1, QVEL_DIM), 0.5 + start, dtype=np.float32)
            body_quat = np.zeros((frames, 1, bodies, 4), dtype=np.float32)
            body_quat[..., 0] = 1.0
            actions = np.full((steps, 1, ACTION_DIM), 0.6 + start, dtype=np.float32)
            controls = np.full((steps, 1, ACTION_DIM), 0.7 + start, dtype=np.float32)
            contact = np.zeros((frames, 1, 2), dtype=np.float32)
            contact[1:, :, 0] = 1.0
            floor_contact = np.zeros((frames, 1, 3), dtype=np.float32)
            floor_contact[..., :2] = contact
            return {
                "qpos": qpos,
                "qvel": qvel,
                "body_pos_w": np.full(
                    (frames, 1, bodies, 3),
                    1.5 + start,
                    dtype=np.float32,
                ),
                "body_quat_w": body_quat,
                "body_lin_vel_w": np.full(
                    (frames, 1, bodies, 3),
                    2.5 + start,
                    dtype=np.float32,
                ),
                "body_ang_vel_w": np.full(
                    (frames, 1, bodies, 3),
                    3.5 + start,
                    dtype=np.float32,
                ),
                "actions": actions,
                "controls": controls,
                "contact_indicator": contact,
                "contact_force": np.full(
                    (frames, 1, 2),
                    4.5 + start,
                    dtype=np.float32,
                ),
                "floor_contact_indicator": floor_contact,
                "floor_contact_force": np.full(
                    (frames, 1, 3),
                    5.5 + start,
                    dtype=np.float32,
                ),
                "final_robot_state": {
                    "qpos": qpos[-1],
                    "qvel": qvel[-1],
                    "body_pos_w": np.full(
                        (1, bodies, 3),
                        1.5 + start,
                        dtype=np.float32,
                    ),
                    "body_quat_w": body_quat[-1],
                    "body_lin_vel_w": np.full(
                        (1, bodies, 3),
                        2.5 + start,
                        dtype=np.float32,
                    ),
                    "body_ang_vel_w": np.full(
                        (1, bodies, 3),
                        3.5 + start,
                        dtype=np.float32,
                    ),
                },
                "final_obs_state": SimpleNamespace(history={}, last_action=actions[-1]),
                "final_prev_control": controls[-1],
                "final_prev_joint_acc": np.zeros((1, ACTION_DIM), dtype=np.float32),
                "final_prev_contact": contact[-1],
                "final_prev_contact_valid": np.array([1.0], dtype=np.float32),
                "final_prev_contact_force": np.full(
                    (1, 2),
                    4.5 + start,
                    dtype=np.float32,
                ),
                "final_prev_contact_force_valid": np.array([1.0], dtype=np.float32),
            }

        def optimizer(**kwargs):
            start = int(kwargs["reference"]["start"])
            optimizer_references.append(dict(kwargs["reference"]["kwargs"]))
            window_config = kwargs["config"]
            updated = torch.zeros(int(window_config.horizon_steps), QPOS_DIM - 1)
            chunk = torch.zeros(int(window_config.control_steps) + 1, QPOS_DIM - 1)
            return SimpleNamespace(
                updated_controls=updated,
                execute_chunk=chunk,
                info={"best_score": torch.tensor(1.25), "accepted": True},
                execute_trace=execute_trace(
                    start,
                    int(window_config.control_steps),
                ),
            )

        def rollout_reference_factory(**kwargs):
            return {"start": kwargs["start"], "kwargs": dict(kwargs)}

        def rollout_tracer(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            raise AssertionError("optimizer execute_trace should avoid rollout_tracer")

        result = run_g1_wbc_mjx_mpc(
            spider_config=config,
            motion=_motion(frames=total_steps + 1),
            actor=WbcActor(input_dim=4, hidden_dims=(), output_dim=2),
            rollout_config=SimpleNamespace(device="cpu", max_steps=total_steps),
            execute_rollout_config=SimpleNamespace(device="cpu", max_steps=total_steps),
            method="g1_wbc_joint_global",
            reward_weights=None,
            total_steps=total_steps,
            seed=5,
            runtime=SimpleNamespace(jnp=SimpleNamespace(), jax=SimpleNamespace()),
            model_factory=lambda **kwargs: _fake_model_bundle(
                profile_name=kwargs["profile_name"]
            ),
            policy_converter=lambda actor, *, jnp: SimpleNamespace(params=True),
            optimizer=optimizer,
            rollout_factory=_fake_rollout_result,
            rollout_reference_factory=rollout_reference_factory,
            rollout_tracer=rollout_tracer,
            command_builder=_fake_command_builder,
        )

        self.assertGreaterEqual(len(optimizer_references), 2)
        np.testing.assert_allclose(
            optimizer_references[1]["initial_robot_state"]["qpos"][:, 0],
            [52.0],
        )
        self.assertEqual(result.metadata["rollout_source"], "dynamic_execute_trace")
        self.assertEqual(result.metadata["execute_trace_chunks"], 2)
        self.assertEqual(
            result.metadata["execute_trace_source_counts"],
            {"optimizer_selected_prefix": 2},
        )
        self.assertTrue(
            all(
                info["execute_trace_source"] == "optimizer_selected_prefix"
                for info in result.result.infos
            )
        )
        rollout = result.result.rollout
        torch.testing.assert_close(
            rollout.qpos[:, 0, 0],
            torch.tensor([50.0, 51.0, 52.0, 53.0, 54.0]),
        )

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
        self.assertEqual(converted_shapes.count((21, QPOS_DIM - 1)), 0)

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
        config.final_noise_scale = 0.25
        config.use_guided_candidate = False
        config.mjx_min_score_improvement = 0.01
        config.mjx_min_top_score_gap = 0.02
        config.mjx_cem_update_min_top_score_gap = 0.015
        config.mjx_max_control_delta = 0.25
        config.mjx_candidate_rank_diagnostics_top_k = 8
        config.mjx_candidate_rescore_diagnostics = True
        config.mjx_candidate_score_component_diagnostics_top_k = 6

        window_config = mjx_backend_module._window_config_from_spider(config)

        self.assertEqual(window_config.iterations, 3)
        self.assertEqual(window_config.final_noise_scale, 0.25)
        self.assertFalse(window_config.use_guided_candidate)
        self.assertEqual(window_config.min_score_improvement, 0.01)
        self.assertEqual(window_config.min_top_score_gap, 0.02)
        self.assertEqual(window_config.cem_update_min_top_score_gap, 0.015)
        self.assertEqual(window_config.max_control_delta, 0.25)
        self.assertEqual(window_config.candidate_rank_diagnostics_top_k, 8)
        self.assertTrue(window_config.candidate_rescore_diagnostics)
        self.assertEqual(window_config.candidate_score_component_diagnostics_top_k, 6)

    def test_mjx_backend_aggregates_iteration_diagnostics(self) -> None:
        def optimizer(**kwargs):
            window_config = kwargs["config"]
            updated = torch.zeros(int(window_config.horizon_steps), QPOS_DIM - 1)
            chunk = torch.zeros(int(window_config.control_steps) + 1, QPOS_DIM - 1)
            chunk[:, 0] = 0.25
            return SimpleNamespace(
                updated_controls=updated,
                execute_chunk=chunk,
                info={
                    "best_score": torch.tensor(1.25),
                    "accepted": True,
                    "accepted_iterations": 1,
                    "current_controls_selected": True,
                    "zero_delta_noop_selected": False,
                    "zero_delta_noop_iterations": 1,
                    "score_threshold_noop_selected": True,
                    "score_threshold_noop_iterations": 1,
                    "control_delta_guard_noop_selected": True,
                    "control_delta_guard_noop_iterations": 1,
                    "noop_candidate_selected": True,
                    "noop_candidate_iterations": 3,
                    "top_score_gap": torch.tensor(0.04),
                    "iteration_accepted_flags": (
                        torch.tensor(True),
                        torch.tensor(False),
                        torch.tensor(False),
                    ),
                    "iteration_current_controls_selected_flags": (
                        torch.tensor(False),
                        torch.tensor(True),
                        torch.tensor(False),
                    ),
                    "iteration_zero_delta_noop_flags": (
                        torch.tensor(False),
                        torch.tensor(False),
                        torch.tensor(True),
                    ),
                    "iteration_score_threshold_noop_flags": (
                        torch.tensor(False),
                        torch.tensor(True),
                        torch.tensor(False),
                    ),
                    "iteration_control_delta_guard_noop_flags": (
                        torch.tensor(True),
                        torch.tensor(False),
                        torch.tensor(False),
                    ),
                    "iteration_noop_candidate_flags": (
                        torch.tensor(True),
                        torch.tensor(True),
                        torch.tensor(True),
                    ),
                    "iteration_top_score_gaps": (
                        torch.tensor(0.03),
                        torch.tensor(0.04),
                        torch.tensor(0.05),
                    ),
                    "candidate_rank_diagnostics_top_k": 3,
                    "iteration_candidate_top_indices": (
                        (
                            torch.tensor(2),
                            torch.tensor(1),
                            torch.tensor(0),
                        ),
                    ),
                    "iteration_candidate_top_scores": (
                        (
                            torch.tensor(3.0),
                            torch.tensor(2.0),
                            torch.tensor(1.0),
                        ),
                    ),
                    "candidate_rescore_diagnostics": True,
                    "iteration_rescore_best_indices": (
                        torch.tensor(2),
                        torch.tensor(1),
                    ),
                    "iteration_rescore_top_score_gaps": (
                        torch.tensor(0.02),
                        torch.tensor(0.03),
                    ),
                    "iteration_rescore_score_delta_maxes": (
                        torch.tensor(0.001),
                        torch.tensor(0.004),
                    ),
                    "iteration_rescore_score_delta_means": (
                        torch.tensor(0.0005),
                        torch.tensor(0.002),
                    ),
                    "iteration_rescore_top1_changed_flags": (
                        torch.tensor(False),
                        torch.tensor(True),
                    ),
                },
            )

        result = _run_with_fakes(
            optimizer=optimizer,
            rollout_factory=_fake_rollout_result,
        )

        self.assertEqual(
            result.metadata["iteration_accepted_window_counts"],
            (40, 0, 0),
        )
        self.assertEqual(
            result.metadata["iteration_current_selected_window_counts"],
            (0, 40, 0),
        )
        self.assertEqual(result.metadata["iteration_noop_window_counts"], (0, 40, 0))
        self.assertEqual(result.metadata["zero_delta_noop_iteration_sum"], 40)
        self.assertEqual(result.metadata["zero_delta_noop_windows"], 0)
        self.assertEqual(result.metadata["zero_delta_noop_accepted_windows"], 0)
        self.assertEqual(result.metadata["score_threshold_noop_iteration_sum"], 40)
        self.assertEqual(result.metadata["score_threshold_noop_windows"], 40)
        self.assertEqual(result.metadata["score_threshold_noop_accepted_windows"], 0)
        self.assertEqual(result.metadata["control_delta_guard_noop_iteration_sum"], 40)
        self.assertEqual(result.metadata["control_delta_guard_noop_windows"], 40)
        self.assertEqual(
            result.metadata["control_delta_guard_noop_accepted_windows"],
            0,
        )
        self.assertEqual(result.metadata["noop_candidate_iteration_sum"], 120)
        self.assertEqual(result.metadata["noop_candidate_windows"], 40)
        self.assertEqual(result.metadata["noop_candidate_accepted_windows"], 0)
        self.assertEqual(result.metadata["candidate_rank_diagnostics_top_k"], 3)
        self.assertEqual(result.metadata["candidate_rank_diagnostics_windows"], 40)
        self.assertEqual(result.metadata["candidate_rescore_diagnostics_windows"], 40)
        self.assertAlmostEqual(
            result.metadata["candidate_rescore_score_delta_max"],
            0.004,
        )
        self.assertAlmostEqual(
            result.metadata["candidate_rescore_score_delta_mean"],
            0.00125,
        )
        self.assertEqual(
            result.metadata["candidate_rescore_top1_changed_iteration_sum"],
            40,
        )
        self.assertEqual(
            result.metadata["iteration_zero_delta_noop_window_counts"],
            (0, 0, 40),
        )
        self.assertEqual(
            result.metadata["iteration_score_threshold_noop_window_counts"],
            (0, 40, 0),
        )
        self.assertEqual(
            result.metadata["iteration_control_delta_guard_noop_window_counts"],
            (40, 0, 0),
        )
        self.assertEqual(
            result.metadata["iteration_noop_candidate_window_counts"],
            (40, 40, 40),
        )
        self.assertAlmostEqual(result.metadata["top_score_gap_min"], 0.04)
        self.assertAlmostEqual(result.metadata["top_score_gap_mean"], 0.04)
        self.assertAlmostEqual(result.metadata["top_score_gap_max"], 0.04)
        self.assertEqual(result.metadata["top_score_gap_windows"], 40)
        self.assertEqual(
            tuple(round(value, 2) for value in result.metadata["iteration_top_score_gap_mins"]),
            (0.03, 0.04, 0.05),
        )
        self.assertEqual(
            tuple(round(value, 2) for value in result.metadata["iteration_top_score_gap_means"]),
            (0.03, 0.04, 0.05),
        )

    def test_mjx_backend_rejects_nonfinite_optimizer_score_flag(self) -> None:
        def optimizer(**kwargs):
            window_config = kwargs["config"]
            updated = torch.zeros(int(window_config.horizon_steps), QPOS_DIM - 1)
            chunk = torch.zeros(int(window_config.control_steps) + 1, QPOS_DIM - 1)
            return SimpleNamespace(
                updated_controls=updated,
                execute_chunk=chunk,
                info={
                    "best_score": torch.tensor(1.0),
                    "accepted": True,
                    "scores_finite": False,
                },
            )

        with self.assertRaisesRegex(ValueError, "finite scores"):
            _run_with_fakes(
                optimizer=optimizer,
                rollout_factory=_fake_rollout_result,
            )


def _run_with_fakes(
    *,
    optimizer,
    rollout_factory,
    spider_config=None,
    runtime=None,
    rollout_scorer=None,
    rollout_reference_factory=None,
    rollout_tracer=None,
    enable_physics_scan=False,
    reward_weights=None,
    **backend_kwargs,
):
    kwargs = {}
    if optimizer is not None:
        kwargs["optimizer"] = optimizer
    if rollout_scorer is not None:
        kwargs["rollout_scorer"] = rollout_scorer
    if rollout_reference_factory is not None:
        kwargs["rollout_reference_factory"] = rollout_reference_factory
    if rollout_tracer is not None:
        kwargs["rollout_tracer"] = rollout_tracer
    return run_g1_wbc_mjx_mpc(
        spider_config=spider_config or _spider_config(),
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
        command_builder=_fake_command_builder,
        enable_physics_scan=enable_physics_scan,
        **backend_kwargs,
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
        nested_iteration_score = Blockable()

        mjx_backend_module._block_window_result_until_ready(
            SimpleNamespace(
                updated_controls=updated_controls,
                execute_chunk=execute_chunk,
                info={
                    "best_score": best_score,
                    "iteration_score_improvements": (nested_iteration_score,),
                },
            )
        )

        self.assertTrue(updated_controls.blocked)
        self.assertTrue(execute_chunk.blocked)
        self.assertTrue(best_score.blocked)
        self.assertTrue(nested_iteration_score.blocked)


def _fake_model_bundle(
    *,
    profile_name: str = "wxy_parity",
    mjx_impl: str | None = "jax",
    mjx_warp_naconmax: int | None = None,
    mjx_warp_njmax: int | None = None,
    mjx_model_options: dict[str, int] | None = None,
):
    return SimpleNamespace(
        profile=SimpleNamespace(name=profile_name),
        mjx_impl=mjx_impl,
        mjx_warp_naconmax=mjx_warp_naconmax,
        mjx_warp_njmax=mjx_warp_njmax,
        mjx_model_options=mjx_model_options or {},
        mjx_model=SimpleNamespace(impl=mjx_impl),
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


def _fake_command_builder(
    motion: G1Motion,
    qpos_trajectory: torch.Tensor,
    config,
):
    del config
    frames = int(qpos_trajectory.shape[0])
    bodies = len(MUJOCO_BODY_NAMES)
    device = qpos_trajectory.device
    body_quat = torch.zeros(frames, 1, bodies, 4, device=device)
    body_quat[..., 0] = 1.0
    return G1CommandBatch(
        path=motion.path,
        motion_type=motion.motion_type,
        fps=motion.fps,
        joint_pos=qpos_trajectory[..., 7:].contiguous(),
        joint_vel=torch.zeros(frames, 1, ACTION_DIM, device=device),
        body_pos_w=torch.zeros(frames, 1, bodies, 3, device=device),
        body_quat_w=body_quat,
        body_lin_vel_w=torch.zeros(frames, 1, bodies, 3, device=device),
        body_ang_vel_w=torch.zeros(frames, 1, bodies, 3, device=device),
        qpos_trajectory=qpos_trajectory.contiguous(),
        qvel_trajectory=torch.zeros(frames, 1, QVEL_DIM, device=device),
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
