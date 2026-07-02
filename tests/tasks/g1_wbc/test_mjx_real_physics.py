from types import SimpleNamespace
import unittest

import numpy as np

from spider.tasks.g1_wbc.constants import ACTION_DIM, DECIMATION, PHYSICS_DT, QPOS_DIM, QVEL_DIM
from spider.tasks.g1_wbc.mjx_model import build_mjx_model_bundle
from spider.tasks.g1_wbc.mjx_physics import (
    joint_order_to_model_ctrl,
    reset_forward_step_smoke,
)
from spider.tasks.g1_wbc.mjx_runtime import probe_mjx_runtime, require_mjx_runtime


class _NumpyJnp:
    @staticmethod
    def asarray(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def zeros_like(value):
        return np.zeros_like(value, dtype=np.float32)


class MjxRealPhysicsTest(unittest.TestCase):
    def test_joint_order_to_model_ctrl_maps_non_identity_actuator_ids(self) -> None:
        bundle = SimpleNamespace(
            actuator_name_to_id={
                "robot/left_hip_pitch_joint": 2,
                "robot/left_hip_roll_joint": 0,
                "robot/left_hip_yaw_joint": 1,
                **{
                    f"robot/joint_{index}": index
                    for index in range(3, ACTION_DIM)
                },
            }
        )
        # Fill the remaining expected G1 names after the deliberately permuted prefix.
        from spider.tasks.g1_wbc.constants import MUJOCO_JOINT_NAMES

        for index, joint_name in enumerate(MUJOCO_JOINT_NAMES[3:], start=3):
            bundle.actuator_name_to_id[f"robot/{joint_name}"] = index
        joint_ctrl = np.linspace(-1.0, 1.0, ACTION_DIM, dtype=np.float32)

        model_ctrl = joint_order_to_model_ctrl(bundle, joint_ctrl, jnp=_NumpyJnp)

        self.assertEqual(model_ctrl.shape, (ACTION_DIM,))
        self.assertEqual(float(model_ctrl[2]), float(joint_ctrl[0]))
        self.assertEqual(float(model_ctrl[0]), float(joint_ctrl[1]))
        self.assertEqual(float(model_ctrl[1]), float(joint_ctrl[2]))
        np.testing.assert_allclose(model_ctrl[3:], joint_ctrl[3:])

    def test_reset_forward_step_smoke_runs_real_mjx_when_runtime_available(self) -> None:
        if not probe_mjx_runtime().available:
            self.skipTest("jax/mujoco.mjx runtime is not available")
        runtime = require_mjx_runtime()
        devices = tuple(runtime.jax.devices())
        if not any(getattr(device, "platform", "") == "gpu" for device in devices):
            self.skipTest("JAX GPU device is not available")
        bundle = build_mjx_model_bundle(profile_name="wxy_parity", require_runtime=True)
        qpos = np.zeros(QPOS_DIM, dtype=np.float32)
        qpos[3] = 1.0
        qvel = np.zeros(QVEL_DIM, dtype=np.float32)
        joint_ctrl = np.linspace(-0.2, 0.2, ACTION_DIM, dtype=np.float32)

        result = reset_forward_step_smoke(
            bundle,
            qpos,
            qvel,
            joint_ctrl,
            runtime=runtime,
        )

        self.assertEqual(result["qpos"].shape, (QPOS_DIM,))
        self.assertEqual(result["qvel"].shape, (QVEL_DIM,))
        self.assertEqual(result["ctrl"].shape, (ACTION_DIM,))
        np.testing.assert_allclose(result["ctrl"], joint_ctrl, atol=1.0e-6)
        self.assertAlmostEqual(
            float(result["time"]),
            PHYSICS_DT * DECIMATION,
            places=5,
        )
        self.assertAlmostEqual(float(np.linalg.norm(result["qpos"][3:7])), 1.0, places=4)
        self.assertTrue(np.all(np.isfinite(result["qpos"])))
        self.assertTrue(np.all(np.isfinite(result["qvel"])))


if __name__ == "__main__":
    unittest.main()
