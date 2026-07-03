import unittest

import numpy as np

from spider.tasks.g1_wbc.constants import ACTION_DIM, QPOS_DIM
from spider.tasks.g1_wbc.mjx_guided import guided_controls_from_trace


class _NumpyJnp:
    @staticmethod
    def asarray(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def concatenate(values, axis=0):
        return np.concatenate(values, axis=axis)

    @staticmethod
    def expand_dims(value, axis):
        return np.expand_dims(value, axis=axis)

    @staticmethod
    def sqrt(value):
        return np.sqrt(value)

    @staticmethod
    def sum(value, axis=None, keepdims=False):
        return np.sum(value, axis=axis, keepdims=keepdims)

    @staticmethod
    def atan2(y, x):
        return np.arctan2(y, x)

    @staticmethod
    def abs(value):
        return np.abs(value)

    @staticmethod
    def where(condition, x, y):
        return np.where(condition, x, y)

    @staticmethod
    def sin(value):
        return np.sin(value)

    @staticmethod
    def clip(value, low, high):
        return np.clip(value, low, high)


class MjxGuidedTest(unittest.TestCase):
    def test_guided_controls_from_trace_clips_and_freezes_first_frame(self) -> None:
        horizon = 4
        base_qpos = np.zeros((horizon, QPOS_DIM), dtype=np.float32)
        base_qpos[:, 0] = np.arange(horizon, dtype=np.float32)
        base_qpos[:, 3] = 1.0
        base_qpos[:, 7:] = 1.0
        executed = base_qpos.copy()
        executed[:, 0] -= 0.2
        executed[:, 7:] -= 1.0
        initial = np.zeros((1, 1, QPOS_DIM), dtype=np.float32)
        initial[:, :, 3] = 1.0
        rollout_qpos = np.concatenate([initial, executed[:, None, :]], axis=0)

        controls = guided_controls_from_trace(
            base_qpos,
            rollout_qpos,
            guided_root_pos_gain=0.5,
            guided_root_rot_gain=0.5,
            guided_joint_gain=0.5,
            guided_root_pos_clip=0.05,
            guided_root_rot_clip=0.12,
            guided_joint_clip=0.35,
            freeze_first_frame=True,
            jnp=_NumpyJnp,
        )

        self.assertEqual(controls.shape, (horizon, QPOS_DIM - 1))
        np.testing.assert_allclose(controls[0], 0.0)
        np.testing.assert_allclose(controls[1:, 0], 0.05)
        np.testing.assert_allclose(controls[1:, 3:6], 0.0, atol=1.0e-6)
        np.testing.assert_allclose(controls[1:, 6:], 0.35)
        self.assertEqual(controls[:, 6:].shape[-1], ACTION_DIM)


if __name__ == "__main__":
    unittest.main()
