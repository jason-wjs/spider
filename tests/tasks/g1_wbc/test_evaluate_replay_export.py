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

import numpy as np
import torch

from spider.tasks.g1_wbc import evaluate
from spider.tasks.g1_wbc.motion import G1CommandBatch
from spider.tasks.g1_wbc.result_types import (
    G1WbcExecutedCommandChunk,
    G1WbcWindowReplayState,
)
from spider.tasks.g1_wbc.rollout import RolloutResult


class EvaluateReplayExportTest(unittest.TestCase):
    def test_save_rollout_preserves_floor_contact_force_peak_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rollout.npz"
            rollout = RolloutResult(
                qpos=torch.zeros(3, 1, 36),
                qvel=torch.zeros(3, 1, 35),
                body_pos_w=torch.zeros(3, 1, 30, 3),
                body_quat_w=torch.zeros(3, 1, 30, 4),
                body_lin_vel_w=torch.zeros(3, 1, 30, 3),
                body_ang_vel_w=torch.zeros(3, 1, 30, 3),
                actions=torch.zeros(2, 1, 29),
                controls=torch.zeros(2, 1, 29),
                contact_indicator=torch.zeros(3, 1, 2),
                contact_force=torch.zeros(3, 1, 2),
                floor_contact_indicator=torch.zeros(3, 1, 3),
                floor_contact_force=torch.zeros(3, 1, 3),
                floor_contact_force_peak_source=torch.arange(
                    3 * 1 * 3 * 8,
                    dtype=torch.float32,
                ).reshape(3, 1, 3, 8),
                ref_indices=torch.arange(3).reshape(3, 1),
            )

            evaluate._save_rollout(path, rollout)

            with np.load(path) as data:
                self.assertIn("floor_contact_force_peak_source", data.files)
                np.testing.assert_allclose(
                    data["floor_contact_force_peak_source"],
                    rollout.floor_contact_force_peak_source.numpy(),
                )

    def test_save_rollout_preserves_floor_contact_force_top_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "rollout.npz"
            rollout = RolloutResult(
                qpos=torch.zeros(3, 1, 36),
                qvel=torch.zeros(3, 1, 35),
                body_pos_w=torch.zeros(3, 1, 30, 3),
                body_quat_w=torch.zeros(3, 1, 30, 4),
                body_lin_vel_w=torch.zeros(3, 1, 30, 3),
                body_ang_vel_w=torch.zeros(3, 1, 30, 3),
                actions=torch.zeros(2, 1, 29),
                controls=torch.zeros(2, 1, 29),
                contact_indicator=torch.zeros(3, 1, 2),
                contact_force=torch.zeros(3, 1, 2),
                floor_contact_indicator=torch.zeros(3, 1, 3),
                floor_contact_force=torch.zeros(3, 1, 3),
                floor_contact_force_top_rows=torch.arange(
                    3 * 1 * 3 * 4 * 21,
                    dtype=torch.float32,
                ).reshape(3, 1, 3, 4, 21),
                ref_indices=torch.arange(3).reshape(3, 1),
            )

            evaluate._save_rollout(path, rollout)

            with np.load(path) as data:
                self.assertIn("floor_contact_force_top_rows", data.files)
                np.testing.assert_allclose(
                    data["floor_contact_force_top_rows"],
                    rollout.floor_contact_force_top_rows.numpy(),
                )

    def test_jsonable_infos_preserves_nested_iteration_scalars(self) -> None:
        class FakeScalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        rows = evaluate._jsonable_infos(
            [
                {
                    "iteration_score_improvements": (
                        FakeScalar(1.5),
                        np.array(0.25, dtype=np.float32),
                    ),
                    "nested": {"flag": FakeScalar(True)},
                }
            ]
        )

        self.assertEqual(rows[0]["iteration_score_improvements"], [1.5, 0.25])
        self.assertEqual(rows[0]["nested"], {"flag": True})

    def test_metrics_reference_motion_is_used_only_for_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_motion = SimpleNamespace(motion_type="mujoco", name="mpc_motion")
            reference_motion = SimpleNamespace(
                motion_type="isaaclab",
                name="original_motion",
            )
            rollout = SimpleNamespace(name="rollout")
            argv = [
                "evaluate.py",
                "--motion",
                str(root / "mpc_motion.npz"),
                "--motion-type",
                "mujoco",
                "--metrics-reference-motion",
                str(root / "motion.npz"),
                "--metrics-reference-motion-type",
                "isaaclab",
                "--method",
                "no_mpc",
                "--device",
                "cpu",
            ]
            with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(
                    evaluate,
                    "load_motion",
                    side_effect=[input_motion, reference_motion],
                ) as load_motion, \
                mock.patch.object(evaluate, "validate_motion_dims"), \
                mock.patch.object(
                    evaluate,
                    "resolve_checkpoint_path",
                    return_value="/tmp/model.pt",
                ), \
                mock.patch.object(evaluate, "load_wbc_actor", return_value=object()), \
                mock.patch.object(
                    evaluate,
                    "run_no_mpc_rollout",
                    return_value=rollout,
                ) as run_no_mpc_rollout, \
                mock.patch.object(
                    evaluate,
                    "compute_rollout_metrics",
                    return_value={"num_steps": 800},
                ) as compute_rollout_metrics, \
                redirect_stdout(io.StringIO()) as stdout:
                evaluate.main()

            payload = json.loads(stdout.getvalue())

        self.assertEqual(load_motion.call_count, 2)
        self.assertEqual(
            load_motion.call_args_list[0].args[0],
            str(root / "mpc_motion.npz"),
        )
        self.assertEqual(load_motion.call_args_list[0].kwargs["motion_type"], "mujoco")
        self.assertEqual(
            load_motion.call_args_list[1].args[0],
            str(root / "motion.npz"),
        )
        self.assertEqual(load_motion.call_args_list[1].kwargs["motion_type"], "isaaclab")
        run_no_mpc_rollout.assert_called_once()
        self.assertIs(run_no_mpc_rollout.call_args.args[0], input_motion)
        compute_rollout_metrics.assert_called_once_with(reference_motion, rollout)
        self.assertEqual(payload["motion_type"], "mujoco")
        self.assertEqual(payload["metrics_reference_motion_type"], "isaaclab")
        self.assertEqual(
            payload["metrics_reference_motion"],
            str((root / "motion.npz").resolve()),
        )

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
                    "_load_saved_command_chunks",
                    return_value=None,
                ), \
                mock.patch.object(
                    evaluate,
                    "_load_saved_command_trajectory",
                    return_value=(
                        torch.zeros(4, 36, dtype=torch.float32),
                        torch.zeros(4, 35, dtype=torch.float32),
                    ),
                ), \
                mock.patch.object(
                    evaluate,
                    "_saved_command_frame_count",
                    return_value=4,
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

    def test_replay_command_prefers_saved_window_command_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            saved_command = root / "mpc_command.npz"
            saved_command.write_bytes(b"command")
            replayed_rollout = SimpleNamespace(name="chunked_rollout")
            chunk = _command_chunk(start=0, steps=2)
            chunk.replay_state = G1WbcWindowReplayState(
                initial_qpos=torch.zeros(36),
                initial_qvel=torch.zeros(35),
            )

            class FakeTask:
                def __init__(self, *args, **kwargs):
                    self.chunk_calls = 0

                def replay_command_chunks(self, chunks, **kwargs):
                    self.chunk_calls += 1
                    self.chunks = chunks
                    self.kwargs = kwargs
                    return replayed_rollout

                def replay_qpos_command_sequence(self, *args, **kwargs):
                    raise AssertionError("stitched qpos replay should not be used")

            argv = [
                "evaluate.py",
                "--motion",
                str(root / "motion.npz"),
                "--method",
                "replay_command",
                "--saved-command",
                str(saved_command),
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
                    "_load_saved_command_chunks",
                    return_value=[chunk],
                ), \
                mock.patch.object(
                    evaluate,
                    "_saved_command_frame_count",
                    return_value=3,
                ), \
                mock.patch.object(evaluate, "G1WbcSamplingTask", FakeTask), \
                mock.patch.object(
                    evaluate,
                    "compute_rollout_metrics",
                    return_value={"num_steps": 2},
                ), \
                redirect_stdout(io.StringIO()) as stdout:
                evaluate.main()

            payload = json.loads(stdout.getvalue())

        self.assertEqual(
            payload["mpc"]["saved_command_replay_source"],
            "window_command_chunks",
        )
        self.assertEqual(
            payload["mpc"]["saved_command_replay_state_source"],
            "window_replay_state_chunks",
        )
        self.assertEqual(
            payload["mpc"]["backend"],
            "spider.tasks.g1_wbc.spider_task.G1WbcSamplingTask.replay_command_chunks",
        )
        self.assertEqual(payload["mpc"]["num_command_chunks"], 1)
        self.assertEqual(payload["mpc"]["num_command_replay_state_chunks"], 1)
        self.assertEqual(payload["mpc"]["num_command_frames"], 3)
        self.assertEqual(payload["mpc"]["num_replay_steps"], 2)

    def test_replay_command_marks_partial_replay_state_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            saved_command = root / "mpc_command.npz"
            saved_command.write_bytes(b"command")
            chunks = [_command_chunk(start=0, steps=2), _command_chunk(start=2, steps=2)]
            chunks[0].replay_state = G1WbcWindowReplayState(
                initial_qpos=torch.zeros(36),
                initial_qvel=torch.zeros(35),
            )

            class FakeTask:
                def __init__(self, *args, **kwargs):
                    pass

                def replay_command_chunks(self, chunks_arg, **kwargs):
                    return SimpleNamespace(name="chunked_rollout", chunks=chunks_arg)

            argv = [
                "evaluate.py",
                "--motion",
                str(root / "motion.npz"),
                "--method",
                "replay_command",
                "--saved-command",
                str(saved_command),
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
                    "_load_saved_command_chunks",
                    return_value=chunks,
                ), \
                mock.patch.object(
                    evaluate,
                    "_saved_command_frame_count",
                    return_value=4,
                ), \
                mock.patch.object(evaluate, "G1WbcSamplingTask", FakeTask), \
                mock.patch.object(
                    evaluate,
                    "compute_rollout_metrics",
                    return_value={"num_steps": 4},
                ), \
                redirect_stdout(io.StringIO()) as stdout:
                evaluate.main()

            payload = json.loads(stdout.getvalue())

        self.assertEqual(
            payload["mpc"]["saved_command_replay_state_source"],
            "partial_window_replay_state_chunks",
        )
        self.assertEqual(payload["mpc"]["num_command_chunks"], 2)
        self.assertEqual(payload["mpc"]["num_command_replay_state_chunks"], 1)

    def test_save_mpc_result_exports_window_command_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "mpc_command.npz"
            command = _command(steps=5)
            result = SimpleNamespace(
                refined_qpos=torch.zeros(4, 36),
                scores=torch.ones(2),
                command=_command(steps=4),
                executed_command_chunks=[
                    G1WbcExecutedCommandChunk(
                        start=0,
                        execute_steps=3,
                        horizon_steps=5,
                        command=command,
                    )
                ],
            )

            evaluate._save_mpc_result(path, result)
            chunks = evaluate._load_saved_command_chunks(str(path), device="cpu")

            with np.load(path) as data:
                starts = data["window_starts"]
                execute_steps = data["window_execute_steps"]
                horizons = data["window_horizons"]
                qvel = data["window_command_qvel_chunks"]

        self.assertEqual(starts.tolist(), [0])
        self.assertEqual(execute_steps.tolist(), [3])
        self.assertEqual(horizons.tolist(), [5])
        self.assertEqual(qvel.shape, (1, 5, 1, 35))
        self.assertIsNotNone(chunks)
        assert chunks is not None
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[0].execute_steps, 3)
        self.assertEqual(chunks[0].horizon_steps, 5)
        torch.testing.assert_close(chunks[0].command.qvel_trajectory, command.qvel_trajectory)

    def test_save_mpc_result_round_trips_window_replay_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "mpc_command.npz"
            command = _command(steps=5)
            replay_state = G1WbcWindowReplayState(
                initial_qpos=torch.arange(36, dtype=torch.float32),
                initial_qvel=torch.arange(35, dtype=torch.float32),
                initial_last_action=torch.arange(29, dtype=torch.float32),
                initial_history_state={
                    "actions": {
                        "buffer": torch.ones(3, 1, 29),
                        "pointer": 2,
                        "num_pushes": torch.tensor([3]),
                    }
                },
            )
            result = SimpleNamespace(
                refined_qpos=torch.zeros(4, 36),
                scores=torch.ones(2),
                command=_command(steps=4),
                executed_command_chunks=[
                    G1WbcExecutedCommandChunk(
                        start=0,
                        execute_steps=3,
                        horizon_steps=5,
                        command=command,
                        replay_state=replay_state,
                    )
                ],
            )

            evaluate._save_mpc_result(path, result)
            chunks = evaluate._load_saved_command_chunks(str(path), device="cpu")

            with np.load(path) as data:
                schema = int(data["window_replay_state_schema_version"])
                initial_qpos = data["window_replay_initial_qpos"]
                history_valid = data["window_replay_history__actions__valid"]

        self.assertEqual(schema, 1)
        self.assertEqual(initial_qpos.shape, (1, 36))
        self.assertEqual(history_valid.tolist(), [1])
        assert chunks is not None
        state = chunks[0].replay_state
        self.assertIsNotNone(state)
        assert state is not None
        torch.testing.assert_close(state.initial_qpos, replay_state.initial_qpos)
        torch.testing.assert_close(state.initial_qvel, replay_state.initial_qvel)
        torch.testing.assert_close(
            state.initial_last_action,
            replay_state.initial_last_action,
        )
        self.assertEqual(state.initial_history_state["actions"]["pointer"], 2)
        torch.testing.assert_close(
            state.initial_history_state["actions"]["buffer"],
            torch.ones(3, 1, 29),
        )

    def test_load_saved_command_chunks_rejects_unknown_window_command_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "mpc_command.npz"
            result = SimpleNamespace(
                refined_qpos=torch.zeros(4, 36),
                scores=torch.ones(2),
                command=_command(steps=4),
                executed_command_chunks=[
                    G1WbcExecutedCommandChunk(
                        start=0,
                        execute_steps=3,
                        horizon_steps=5,
                        command=_command(steps=5),
                    )
                ],
            )
            evaluate._save_mpc_result(path, result)
            with np.load(path) as data:
                arrays = {name: data[name] for name in data.files}
            arrays["window_command_schema_version"] = np.array(2, dtype=np.int32)
            np.savez_compressed(path, **arrays)

            with self.assertRaisesRegex(ValueError, "Unsupported window command schema"):
                evaluate._load_saved_command_chunks(str(path), device="cpu")

    def test_load_saved_command_chunks_rejects_replay_state_length_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "mpc_command.npz"
            replay_state = G1WbcWindowReplayState(
                initial_qpos=torch.zeros(36),
                initial_qvel=torch.zeros(35),
            )
            result = SimpleNamespace(
                refined_qpos=torch.zeros(4, 36),
                scores=torch.ones(2),
                command=_command(steps=4),
                executed_command_chunks=[
                    G1WbcExecutedCommandChunk(
                        start=0,
                        execute_steps=3,
                        horizon_steps=5,
                        command=_command(steps=5),
                        replay_state=replay_state,
                    ),
                    G1WbcExecutedCommandChunk(
                        start=3,
                        execute_steps=2,
                        horizon_steps=5,
                        command=_command(steps=5),
                        replay_state=replay_state,
                    ),
                ],
            )
            evaluate._save_mpc_result(path, result)
            with np.load(path) as data:
                arrays = {name: data[name] for name in data.files}
            arrays["window_replay_state_valid"] = arrays["window_replay_state_valid"][:1]
            np.savez_compressed(path, **arrays)

            with self.assertRaisesRegex(ValueError, "Malformed window replay state"):
                evaluate._load_saved_command_chunks(str(path), device="cpu")

    def test_load_saved_command_chunks_rejects_unknown_replay_state_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "mpc_command.npz"
            replay_state = G1WbcWindowReplayState(
                initial_qpos=torch.zeros(36),
                initial_qvel=torch.zeros(35),
            )
            result = SimpleNamespace(
                refined_qpos=torch.zeros(4, 36),
                scores=torch.ones(2),
                command=_command(steps=4),
                executed_command_chunks=[
                    G1WbcExecutedCommandChunk(
                        start=0,
                        execute_steps=3,
                        horizon_steps=5,
                        command=_command(steps=5),
                        replay_state=replay_state,
                    )
                ],
            )
            evaluate._save_mpc_result(path, result)
            with np.load(path) as data:
                arrays = {name: data[name] for name in data.files}
            arrays["window_replay_state_schema_version"] = np.array(2, dtype=np.int32)
            np.savez_compressed(path, **arrays)

            with self.assertRaisesRegex(ValueError, "Unsupported window replay state schema"):
                evaluate._load_saved_command_chunks(str(path), device="cpu")


def _command(steps: int) -> G1CommandBatch:
    qpos = torch.zeros(steps, 1, 36)
    qpos[..., 3] = 1.0
    qvel = torch.arange(steps * 35, dtype=torch.float32).view(steps, 1, 35)
    return G1CommandBatch(
        path=Path("/tmp/motion.npz"),
        motion_type="mujoco",
        fps=50.0,
        joint_pos=torch.zeros(steps, 1, 29),
        joint_vel=qvel[..., 6:].clone(),
        body_pos_w=torch.zeros(steps, 1, 24, 3),
        body_quat_w=torch.zeros(steps, 1, 24, 4),
        body_lin_vel_w=torch.zeros(steps, 1, 24, 3),
        body_ang_vel_w=torch.zeros(steps, 1, 24, 3),
        qpos_trajectory=qpos,
        qvel_trajectory=qvel,
    )


def _command_chunk(start: int, steps: int) -> G1WbcExecutedCommandChunk:
    return G1WbcExecutedCommandChunk(
        start=start,
        execute_steps=steps,
        horizon_steps=steps,
        command=_command(steps),
    )


if __name__ == "__main__":
    unittest.main()
