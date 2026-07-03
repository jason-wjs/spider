from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from spider.tasks.g1_wbc import evaluate


class EvaluateReplayExportTest(unittest.TestCase):
    def test_replay_command_payload_includes_runtime_gpu_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            saved_command = root / "mpc_command.npz"
            saved_command.write_bytes(b"command")
            replayed_rollout = SimpleNamespace(name="rollout")

            class FakeTask:
                def __init__(self, *args, **kwargs):
                    pass

                def replay_qpos_command_sequence(self, *args, **kwargs):
                    return replayed_rollout

            argv = [
                "evaluate.py",
                "--motion",
                str(root / "motion.npz"),
                "--method",
                "replay_command",
                "--saved-command",
                str(saved_command),
                "--replay-control-steps",
                "2",
                "--device",
                "cpu",
            ]
            with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(
                    evaluate,
                    "load_motion",
                    return_value=SimpleNamespace(motion_type="isaaclab"),
                ), \
                mock.patch.object(evaluate, "validate_motion_dims"), \
                mock.patch.object(
                    evaluate,
                    "resolve_checkpoint_path",
                    return_value="/tmp/model.pt",
                ), \
                mock.patch.object(evaluate, "load_wbc_actor", return_value=object()), \
                mock.patch.object(
                    evaluate,
                    "_load_saved_command_trajectory",
                    return_value=(
                        torch.zeros(4, 36, dtype=torch.float32),
                        torch.zeros(4, 35, dtype=torch.float32),
                    ),
                ), \
                mock.patch.object(evaluate, "G1WbcSamplingTask", FakeTask), \
                mock.patch.object(
                    evaluate,
                    "compute_rollout_metrics",
                    return_value={"num_steps": 3},
                ), \
                mock.patch.object(
                    evaluate,
                    "_runtime_visible_devices",
                    return_value=("0",),
                ), \
                mock.patch.object(
                    evaluate,
                    "_runtime_gpu_name",
                    return_value="NVIDIA H100 80GB HBM3",
                ), \
                redirect_stdout(io.StringIO()) as stdout:
                evaluate.main()

            payload = json.loads(stdout.getvalue())

        self.assertEqual(payload["mpc"]["runtime_visible_devices"], ["0"])
        self.assertEqual(payload["mpc"]["runtime_gpu_name"], "NVIDIA H100 80GB HBM3")
        self.assertIn("steady_state_wall_time_sec", payload["mpc"])
        self.assertGreaterEqual(payload["mpc"]["steady_state_wall_time_sec"], 0.0)


if __name__ == "__main__":
    unittest.main()
