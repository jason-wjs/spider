from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

PROFILER_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "profile_g1_wbc_mpc_phases.py"
)


def load_profiler():
    spec = importlib.util.spec_from_file_location("g1_wbc_mpc_phase_profiler", PROFILER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load profiler from {PROFILER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MpcPhaseProfilerTest(unittest.TestCase):
    def test_main_fails_fast_when_profile_input_paths_are_missing(self) -> None:
        profiler = load_profiler()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir) / "profile"

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = profiler.main(
                    [
                        "--motion",
                        str(Path(tmp_dir) / "missing_motion.npz"),
                        "--checkpoint",
                        str(Path(tmp_dir) / "missing_model.pt"),
                        "--reward-weights",
                        str(Path(tmp_dir) / "missing_rewards.json"),
                        "--output-dir",
                        str(output_root),
                        "--dry-run",
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("missing input: motion", stderr.getvalue())
        self.assertFalse(output_root.exists())

    def test_build_evaluate_command_accepts_explicit_existing_inputs(self) -> None:
        profiler = load_profiler()
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            motion = root / "motion.npz"
            checkpoint = root / "model.pt"
            rewards = root / "rewards.json"
            for path in (motion, checkpoint, rewards):
                path.write_text("{}")
            args = profiler.parse_args(
                [
                    "--motion",
                    str(motion),
                    "--checkpoint",
                    str(checkpoint),
                    "--reward-weights",
                    str(rewards),
                    "--output-dir",
                    str(root / "profile"),
                    "--dry-run",
                ]
            )

            command = profiler.build_evaluate_command(args, root / "evaluate")

        self.assertIn(str(motion.resolve()), command)
        self.assertIn(str(checkpoint), command)
        self.assertIn(str(rewards.resolve()), command)


if __name__ == "__main__":
    unittest.main()
