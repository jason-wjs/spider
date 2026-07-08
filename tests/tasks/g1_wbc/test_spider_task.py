from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from spider.optimizers.receding import RecedingHorizonResult
from spider.tasks.g1_wbc.motion import G1CommandBatch
from spider.tasks.g1_wbc.result_types import (
    G1WbcExecutedCommandChunk,
    G1WbcWindowReplayState,
)
from spider.tasks.g1_wbc import spider_task


def _sampling_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_samples=512,
        rollout_batch_size=0,
        max_num_iterations=2,
        horizon_steps=40,
        ctrl_steps=20,
        control_update_mode="weighted_mean",
        noise_scale=torch.zeros(512, 8, 35),
        temperature=0.7,
        first_ctrl_noise_scale=1.0,
        last_ctrl_noise_scale=1.0,
        final_noise_scale=1.0,
        beta_traj=1.0,
        pos_noise_scale=0.04,
        rot_noise_scale=0.10,
        joint_noise_scale=0.18,
        exploit_ratio=0.0,
        exploit_noise_scale=0.0,
        use_torch_compile=False,
    )


class G1WbcSpiderTaskTest(unittest.TestCase):
    def test_sampling_mpc_metadata_includes_steady_state_wall_time(self) -> None:
        receding = RecedingHorizonResult(
            controls=torch.zeros(40, 35),
            infos=[],
            executed_steps=800,
        )
        task = mock.Mock()
        task.build_result.return_value = SimpleNamespace(
            rollout=None,
            command=None,
            refined_qpos=torch.zeros(1, 36),
            controls=torch.zeros(40, 35),
            infos=[],
            scores=torch.zeros(1),
            num_windows=40,
        )

        with mock.patch.object(
            spider_task,
            "run_sampling_receding_mpc",
            return_value=receding,
        ):
            result = spider_task.run_g1_wbc_sampling_mpc(
                _sampling_config(),
                task,
                total_steps=800,
            )

        self.assertIn("steady_state_wall_time_sec", result.metadata)
        self.assertIsInstance(result.metadata["steady_state_wall_time_sec"], float)
        self.assertGreaterEqual(result.metadata["steady_state_wall_time_sec"], 0.0)
        self.assertEqual(result.metadata["runtime_visible_devices"], ())
        self.assertIsNone(result.metadata["runtime_gpu_name"])

    def test_replay_command_chunks_executes_prefix_without_truncating_command(self) -> None:
        task = spider_task.G1WbcSamplingTask.__new__(spider_task.G1WbcSamplingTask)
        task.device = torch.device("cpu")
        task.reset_execution_state = mock.Mock()
        task._execute_command_batch = mock.Mock()
        task._stack_rollout = mock.Mock(return_value=SimpleNamespace(name="rollout"))
        chunk = G1WbcExecutedCommandChunk(
            start=0,
            execute_steps=3,
            horizon_steps=5,
            command=_command(steps=5),
            replay_state=G1WbcWindowReplayState(
                initial_qpos=torch.zeros(36),
                initial_qvel=torch.zeros(35),
                initial_last_action=torch.ones(29),
            ),
        )

        result = spider_task.G1WbcSamplingTask.replay_command_chunks(
            task,
            [chunk],
            total_steps=3,
        )

        self.assertEqual(result.name, "rollout")
        task._execute_command_batch.assert_called_once()
        command_arg, start_arg = task._execute_command_batch.call_args.args
        self.assertEqual(command_arg.num_frames, 5)
        self.assertEqual(start_arg, 0)
        self.assertEqual(
            task._execute_command_batch.call_args.kwargs["execute_steps"],
            3,
        )
        self.assertIs(
            task._execute_command_batch.call_args.kwargs["replay_state"],
            chunk.replay_state,
        )

def _command(steps: int) -> G1CommandBatch:
    qpos = torch.zeros(steps, 1, 36)
    qpos[..., 3] = 1.0
    qvel = torch.zeros(steps, 1, 35)
    return G1CommandBatch(
        path=Path("/tmp/motion.npz"),
        motion_type="mujoco",
        fps=50.0,
        joint_pos=torch.zeros(steps, 1, 29),
        joint_vel=torch.zeros(steps, 1, 29),
        body_pos_w=torch.zeros(steps, 1, 24, 3),
        body_quat_w=torch.zeros(steps, 1, 24, 4),
        body_lin_vel_w=torch.zeros(steps, 1, 24, 3),
        body_ang_vel_w=torch.zeros(steps, 1, 24, 3),
        qpos_trajectory=qpos,
        qvel_trajectory=qvel,
    )


if __name__ == "__main__":
    unittest.main()
