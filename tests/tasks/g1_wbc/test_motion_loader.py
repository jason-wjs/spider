import tempfile
import unittest
from pathlib import Path

import numpy as np

from spider.tasks.g1_wbc.constants import ACTION_DIM, MUJOCO_BODY_NAMES
from spider.tasks.g1_wbc.motion import load_motion


class MotionLoaderTest(unittest.TestCase):
    def test_empty_fps_array_falls_back_to_target_dt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            motion_path = Path(tmp_dir) / "motion.npz"
            _write_minimal_motion(motion_path, fps=np.array([], dtype=np.int64))

            motion = load_motion(
                motion_path,
                motion_type="isaaclab",
                device="cpu",
                target_dt=0.02,
            )

        self.assertEqual(motion.num_frames, 4)
        self.assertEqual(motion.fps, 50.0)


def _write_minimal_motion(path: Path, *, fps: np.ndarray) -> None:
    frames = 4
    bodies = len(MUJOCO_BODY_NAMES)
    body_quat = np.zeros((frames, bodies, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    np.savez_compressed(
        path,
        fps=fps,
        joint_pos=np.zeros((frames, ACTION_DIM), dtype=np.float32),
        joint_vel=np.zeros((frames, ACTION_DIM), dtype=np.float32),
        body_pos_w=np.zeros((frames, bodies, 3), dtype=np.float32),
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros((frames, bodies, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, bodies, 3), dtype=np.float32),
    )


if __name__ == "__main__":
    unittest.main()
