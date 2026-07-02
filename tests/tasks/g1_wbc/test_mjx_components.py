import os
from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    QPOS_DIM,
)
from spider.tasks.g1_wbc.mjx_components import build_mjx_rollout_components
from spider.tasks.g1_wbc.mjx_physics import default_action_scale
from spider.tasks.g1_wbc.motion import G1Motion


class _NumpyJnp:
    @staticmethod
    def asarray(value):
        return np.asarray(value, dtype=np.float32)


class _Runtime:
    jnp = _NumpyJnp()


def _motion(frames: int = 4) -> G1Motion:
    bodies = len(MUJOCO_BODY_NAMES)
    body_pos = torch.zeros(frames, bodies, 3)
    body_quat = torch.zeros(frames, bodies, 4)
    body_quat[..., 0] = 1.0
    return G1Motion(
        path=None,
        motion_type="mujoco",
        fps=50.0,
        joint_pos=torch.zeros(frames, ACTION_DIM),
        joint_vel=torch.zeros(frames, ACTION_DIM),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=torch.zeros(frames, bodies, 3),
        body_ang_vel_w=torch.zeros(frames, bodies, 3),
        contact=torch.zeros(frames, 2),
    )


def _model_bundle():
    return SimpleNamespace(
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


class MjxComponentsTest(unittest.TestCase):
    def test_importing_components_does_not_load_rollout_module(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        code = (
            "import sys; "
            "import spider.tasks.g1_wbc.mjx_components; "
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

    def test_default_action_scale_matches_warp_rollout_spec(self) -> None:
        expected = _warp_action_scale()

        actual = default_action_scale(jnp=_NumpyJnp())

        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-8)

    def test_build_components_exposes_scorer_and_reference_factory(self) -> None:
        def physics_step_fn(
            model_bundle,
            robot_state,
            command_qpos,
            action,
            step_index,
            *,
            runtime,
        ):
            raise AssertionError("not called in component wiring test")

        components = build_mjx_rollout_components(
            runtime=_Runtime(),
            physics_step_fn=physics_step_fn,
            score_weights={"root_pos": 1.0},
        )

        self.assertTrue(callable(components.rollout_scorer))
        self.assertTrue(callable(components.rollout_reference_factory))
        self.assertEqual(components.default_joint_pos.shape, (ACTION_DIM,))
        self.assertEqual(components.action_scale.shape, (ACTION_DIM,))

        reference = components.rollout_reference_factory(
            start=0,
            motion=_motion(),
            controls=torch.zeros(2, QPOS_DIM - 1),
            actor_params=object(),
            model_bundle=_model_bundle(),
            runtime=_Runtime(),
        )

        self.assertEqual(reference["score_weights"].terms, {"root_pos": 1.0})
        np.testing.assert_allclose(
            components.default_joint_pos,
            reference["default_joint_pos"],
        )


def _warp_action_scale() -> np.ndarray:
    repo_root = Path(__file__).resolve().parents[3]
    code = (
        "import json; "
        "from spider.tasks.g1_wbc.rollout import joint_actuator_specs; "
        "print('ACTION_SCALE_JSON', json.dumps("
        "joint_actuator_specs('cpu')['action_scale'].numpy().tolist()))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    for line in completed.stdout.splitlines():
        if line.startswith("ACTION_SCALE_JSON "):
            return np.array(
                json.loads(line.removeprefix("ACTION_SCALE_JSON ")),
                dtype=np.float32,
            )
    raise RuntimeError(completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
