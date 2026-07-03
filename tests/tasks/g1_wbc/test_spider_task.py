from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from spider.optimizers.receding import RecedingHorizonResult
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


if __name__ == "__main__":
    unittest.main()
