import os
from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    OBS_DIM,
    QPOS_DIM,
    QVEL_DIM,
)
from spider.tasks.g1_wbc.mjx_components import build_mjx_rollout_components
from spider.tasks.g1_wbc.mjx_policy import JaxActorParams
from spider.tasks.g1_wbc.mjx_physics import default_action_scale
from spider.tasks.g1_wbc.motion import G1Motion


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
    def repeat(value, repeats, axis=0):
        return np.repeat(value, repeats, axis=axis)

    @staticmethod
    def linspace(start, stop, num):
        return np.linspace(start, stop, int(num), dtype=np.float32)

    @staticmethod
    def floor(value):
        return np.floor(value)

    @staticmethod
    def minimum(left, right):
        return np.minimum(left, right)

    @staticmethod
    def stack(values, axis=0):
        return np.stack(values, axis=axis)

    @staticmethod
    def take(value, indices, axis=0):
        return np.take(value, indices, axis=axis)

    @staticmethod
    def zeros(shape):
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def full(shape, value):
        return np.full(shape, value, dtype=np.float32)

    @staticmethod
    def sum(value, axis=None, keepdims=False):
        return np.sum(value, axis=axis, keepdims=keepdims)

    @staticmethod
    def sqrt(value):
        return np.sqrt(value)

    @staticmethod
    def sin(value):
        return np.sin(value)

    @staticmethod
    def cos(value):
        return np.cos(value)

    @staticmethod
    def atan2(y, x):
        return np.arctan2(y, x)

    @staticmethod
    def where(condition, x, y):
        return np.where(condition, x, y)

    @staticmethod
    def maximum(x, y):
        return np.maximum(x, y)

    @staticmethod
    def clip(value, low, high):
        return np.clip(value, low, high)

    @staticmethod
    def mean(value, axis=None):
        return np.mean(value, axis=axis)

    @staticmethod
    def abs(value):
        return np.abs(value)

    @staticmethod
    def logical_or(left, right):
        return np.logical_or(left, right)


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


def _constant_actor() -> JaxActorParams:
    return JaxActorParams(
        obs_mean=np.zeros((1, OBS_DIM), dtype=np.float32),
        obs_std=np.ones((1, OBS_DIM), dtype=np.float32),
        layers=(
            (
                np.zeros((OBS_DIM, ACTION_DIM), dtype=np.float32),
                np.zeros(ACTION_DIM, dtype=np.float32),
            ),
        ),
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
        self.assertTrue(callable(components.command_reference_fn))
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

    def test_build_components_can_request_score_only_optimizer_scorer(self) -> None:
        captured: dict[str, object] = {}

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

        def fake_make_rollout_scorer(**kwargs):
            captured["score_only"] = kwargs.get("score_only")

            def scorer(samples, reference, actor_params, model_bundle):
                del samples, reference, actor_params, model_bundle
                return {"score": np.zeros((1,), dtype=np.float32)}

            return scorer

        def fake_make_rollout_tracer(**kwargs):
            def tracer(samples, reference, actor_params, model_bundle):
                del samples, reference, actor_params, model_bundle
                return {}

            return tracer

        with (
            mock.patch(
                "spider.tasks.g1_wbc.mjx_rollout.make_rollout_scorer",
                side_effect=fake_make_rollout_scorer,
            ),
            mock.patch(
                "spider.tasks.g1_wbc.mjx_rollout.make_rollout_tracer",
                side_effect=fake_make_rollout_tracer,
            ),
        ):
            components = build_mjx_rollout_components(
                runtime=_Runtime(),
                physics_step_fn=physics_step_fn,
                score_only_optimizer=True,
            )

        self.assertTrue(callable(components.rollout_scorer))
        self.assertIs(captured["score_only"], True)

    def test_build_components_can_request_score_only_rescore_shadow_scorer(
        self,
    ) -> None:
        captured_score_only_flags: list[object] = []

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

        def fake_make_rollout_scorer(**kwargs):
            captured_score_only_flags.append(kwargs.get("score_only"))

            def scorer(samples, reference, actor_params, model_bundle):
                del samples, reference, actor_params, model_bundle
                return {"score": np.zeros((1,), dtype=np.float32)}

            return scorer

        def fake_make_rollout_tracer(**kwargs):
            def tracer(samples, reference, actor_params, model_bundle):
                del samples, reference, actor_params, model_bundle
                return {}

            return tracer

        with (
            mock.patch(
                "spider.tasks.g1_wbc.mjx_rollout.make_rollout_scorer",
                side_effect=fake_make_rollout_scorer,
            ),
            mock.patch(
                "spider.tasks.g1_wbc.mjx_rollout.make_rollout_tracer",
                side_effect=fake_make_rollout_tracer,
            ),
        ):
            components = build_mjx_rollout_components(
                runtime=_Runtime(),
                physics_step_fn=physics_step_fn,
                score_only_rescore_diagnostics=True,
            )

        self.assertTrue(callable(components.rollout_scorer))
        self.assertTrue(callable(components.score_only_rollout_scorer))
        self.assertEqual(captured_score_only_flags, [False, True])

    def test_build_components_can_request_score_only_output_rescore_scorer(
        self,
    ) -> None:
        captured_modes: list[tuple[object, object, object]] = []

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

        def fake_make_rollout_scorer(**kwargs):
            captured_modes.append(
                (
                    kwargs.get("score_only"),
                    kwargs.get("score_only_output"),
                    kwargs.get("score_only_output_full_metrics", False),
                )
            )

            def scorer(samples, reference, actor_params, model_bundle):
                del samples, reference, actor_params, model_bundle
                return {"score": np.zeros((1,), dtype=np.float32)}

            return scorer

        def fake_make_rollout_tracer(**kwargs):
            def tracer(samples, reference, actor_params, model_bundle):
                del samples, reference, actor_params, model_bundle
                return {}

            return tracer

        with (
            mock.patch(
                "spider.tasks.g1_wbc.mjx_rollout.make_rollout_scorer",
                side_effect=fake_make_rollout_scorer,
            ),
            mock.patch(
                "spider.tasks.g1_wbc.mjx_rollout.make_rollout_tracer",
                side_effect=fake_make_rollout_tracer,
            ),
        ):
            components = build_mjx_rollout_components(
                runtime=_Runtime(),
                physics_step_fn=physics_step_fn,
                score_only_output_rescore_diagnostics=True,
            )

        self.assertTrue(callable(components.rollout_scorer))
        self.assertTrue(callable(components.score_only_output_rollout_scorer))
        self.assertEqual(captured_modes, [(False, False, False), (False, True, True)])

    def test_score_only_output_rescore_can_use_first_row_diagnostic_physics(
        self,
    ) -> None:
        created_physics: list[object] = []
        scorer_calls: list[dict[str, object]] = []

        def fake_make_mjx_physics_step_fn(**kwargs):
            include_first_row = bool(
                kwargs.get("contact_force_first_row_diagnostics", False)
            )

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

            physics_step_fn.include_first_row = include_first_row
            created_physics.append(physics_step_fn)
            return physics_step_fn

        def fake_make_rollout_scorer(**kwargs):
            scorer_calls.append(
                {
                    "physics_step_fn": kwargs["physics_step_fn"],
                    "score_only_output": kwargs.get("score_only_output"),
                    "score_only_output_full_metrics": kwargs.get(
                        "score_only_output_full_metrics",
                        False,
                    ),
                }
            )

            def scorer(samples, reference, actor_params, model_bundle):
                del samples, reference, actor_params, model_bundle
                return {"score": np.zeros((1,), dtype=np.float32)}

            return scorer

        def fake_make_rollout_tracer(**kwargs):
            def tracer(samples, reference, actor_params, model_bundle):
                del samples, reference, actor_params, model_bundle
                return {}

            return tracer

        with (
            mock.patch(
                "spider.tasks.g1_wbc.mjx_components.make_mjx_physics_step_fn",
                side_effect=fake_make_mjx_physics_step_fn,
            ),
            mock.patch(
                "spider.tasks.g1_wbc.mjx_rollout.make_rollout_scorer",
                side_effect=fake_make_rollout_scorer,
            ),
            mock.patch(
                "spider.tasks.g1_wbc.mjx_rollout.make_rollout_tracer",
                side_effect=fake_make_rollout_tracer,
            ),
        ):
            components = build_mjx_rollout_components(
                runtime=_Runtime(),
                contact_force_first_row_diagnostics=True,
                score_only_output_rescore_diagnostics=True,
            )

        self.assertTrue(callable(components.rollout_scorer))
        self.assertTrue(callable(components.score_only_output_rollout_scorer))
        self.assertEqual(len(created_physics), 2)
        primary_scorer, score_only_output_scorer = scorer_calls
        self.assertFalse(primary_scorer["physics_step_fn"].include_first_row)
        self.assertTrue(score_only_output_scorer["score_only_output"])
        self.assertTrue(score_only_output_scorer["score_only_output_full_metrics"])
        self.assertTrue(
            score_only_output_scorer["physics_step_fn"].include_first_row
        )

    def test_build_components_keeps_first_row_diagnostics_out_of_scorer(self) -> None:
        created_physics: list[object] = []
        captured: dict[str, object] = {"tracers": []}

        def fake_make_mjx_physics_step_fn(**kwargs):
            include_first_row = bool(
                kwargs.get("contact_force_first_row_diagnostics", False)
            )
            contact_force_mode = kwargs.get("contact_force_mode", "sum_rows")

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

            physics_step_fn.include_first_row = include_first_row
            physics_step_fn.contact_force_mode = contact_force_mode
            created_physics.append(physics_step_fn)
            return physics_step_fn

        def fake_make_rollout_scorer(**kwargs):
            captured["scorer_physics_step_fn"] = kwargs["physics_step_fn"]

            def scorer(samples, reference, actor_params, model_bundle):
                del samples, reference, actor_params, model_bundle
                return {"score": np.zeros((1,), dtype=np.float32)}

            return scorer

        def fake_make_rollout_tracer(**kwargs):
            captured["tracers"].append(
                {
                    "physics_step_fn": kwargs["physics_step_fn"],
                    "record_trace": kwargs.get("record_trace", True),
                    "command_reference_fn": kwargs.get("command_reference_fn"),
                }
            )

            def tracer(samples, reference, actor_params, model_bundle):
                del samples, reference, actor_params, model_bundle
                return {}

            return tracer

        with (
            mock.patch(
                "spider.tasks.g1_wbc.mjx_components.make_mjx_physics_step_fn",
                side_effect=fake_make_mjx_physics_step_fn,
            ),
            mock.patch(
                "spider.tasks.g1_wbc.mjx_rollout.make_rollout_scorer",
                side_effect=fake_make_rollout_scorer,
            ),
            mock.patch(
                "spider.tasks.g1_wbc.mjx_rollout.make_rollout_tracer",
                side_effect=fake_make_rollout_tracer,
            ),
        ):
            components = build_mjx_rollout_components(
                runtime=_Runtime(),
                contact_force_mode="first_row",
                contact_force_first_row_diagnostics=True,
            )

        self.assertTrue(callable(components.rollout_scorer))
        self.assertEqual(len(created_physics), 2)
        scorer_physics = captured["scorer_physics_step_fn"]
        self.assertFalse(scorer_physics.include_first_row)
        self.assertEqual(scorer_physics.contact_force_mode, "first_row")
        rollout_tracer, state_advancer, guided_tracer = captured["tracers"]
        self.assertTrue(rollout_tracer["physics_step_fn"].include_first_row)
        self.assertEqual(
            rollout_tracer["physics_step_fn"].contact_force_mode,
            "first_row",
        )
        self.assertTrue(rollout_tracer["record_trace"])
        self.assertIsNotNone(rollout_tracer["command_reference_fn"])
        self.assertFalse(state_advancer["physics_step_fn"].include_first_row)
        self.assertEqual(
            state_advancer["physics_step_fn"].contact_force_mode,
            "first_row",
        )
        self.assertFalse(state_advancer["record_trace"])
        self.assertFalse(guided_tracer["physics_step_fn"].include_first_row)
        self.assertEqual(
            guided_tracer["physics_step_fn"].contact_force_mode,
            "first_row",
        )
        self.assertTrue(guided_tracer["record_trace"])
        self.assertIsNone(guided_tracer["command_reference_fn"])

    def test_build_components_keeps_top_rows_out_of_primary_scorer(self) -> None:
        created_physics: list[object] = []
        captured: dict[str, object] = {"tracers": []}

        def fake_make_mjx_physics_step_fn(**kwargs):
            include_top_rows = bool(
                kwargs.get("contact_force_top_row_diagnostics", False)
            )

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

            physics_step_fn.include_top_rows = include_top_rows
            created_physics.append(physics_step_fn)
            return physics_step_fn

        def fake_make_rollout_scorer(**kwargs):
            captured["scorer_physics_step_fn"] = kwargs["physics_step_fn"]

            def scorer(samples, reference, actor_params, model_bundle):
                del samples, reference, actor_params, model_bundle
                return {"score": np.zeros((1,), dtype=np.float32)}

            return scorer

        def fake_make_rollout_tracer(**kwargs):
            captured["tracers"].append(
                {
                    "physics_step_fn": kwargs["physics_step_fn"],
                    "record_trace": kwargs.get("record_trace", True),
                }
            )

            def tracer(samples, reference, actor_params, model_bundle):
                del samples, reference, actor_params, model_bundle
                return {}

            return tracer

        with (
            mock.patch(
                "spider.tasks.g1_wbc.mjx_components.make_mjx_physics_step_fn",
                side_effect=fake_make_mjx_physics_step_fn,
            ),
            mock.patch(
                "spider.tasks.g1_wbc.mjx_rollout.make_rollout_scorer",
                side_effect=fake_make_rollout_scorer,
            ),
            mock.patch(
                "spider.tasks.g1_wbc.mjx_rollout.make_rollout_tracer",
                side_effect=fake_make_rollout_tracer,
            ),
        ):
            components = build_mjx_rollout_components(
                runtime=_Runtime(),
                contact_force_top_row_diagnostics=True,
            )

        self.assertTrue(callable(components.rollout_scorer))
        self.assertEqual(len(created_physics), 2)
        self.assertFalse(captured["scorer_physics_step_fn"].include_top_rows)
        rollout_tracer, state_advancer, guided_tracer = captured["tracers"]
        self.assertTrue(rollout_tracer["physics_step_fn"].include_top_rows)
        self.assertTrue(rollout_tracer["record_trace"])
        self.assertFalse(state_advancer["physics_step_fn"].include_top_rows)
        self.assertFalse(state_advancer["record_trace"])
        self.assertFalse(guided_tracer["physics_step_fn"].include_top_rows)

    def test_reference_factory_attaches_generated_guided_controls(self) -> None:
        def physics_step_fn(
            model_bundle,
            robot_state,
            command_qpos,
            action,
            step_index,
            *,
            runtime,
        ):
            del model_bundle, robot_state, action, step_index, runtime
            samples = int(command_qpos.shape[0])
            bodies = len(MUJOCO_BODY_NAMES)
            qpos = np.asarray(command_qpos, dtype=np.float32).copy()
            qpos[:, 0] -= 0.2
            qpos[:, 7:] -= 1.0
            body_pos = np.zeros((samples, bodies, 3), dtype=np.float32)
            return {
                "qpos": qpos,
                "qvel": np.zeros((samples, QVEL_DIM), dtype=np.float32),
                "body_pos_w": body_pos,
                "body_quat_w": _identity_body_quat((samples, bodies)),
                "body_ang_vel_w": np.zeros((samples, bodies, 3), dtype=np.float32),
            }, {
                "root_pos": qpos[:, :3],
                "body_pos": body_pos[:, :2],
                "ee_pos": body_pos[:, :1],
                "contact": np.zeros((samples, 2), dtype=np.float32),
            }

        def command_reference_fn(*args, **kwargs):
            del args, kwargs
            raise AssertionError(
                "guided candidate trace should not compile command references"
            )

        components = build_mjx_rollout_components(
            runtime=_Runtime(),
            physics_step_fn=physics_step_fn,
            command_reference_fn=command_reference_fn,
            use_guided_candidate=True,
            guided_root_pos_gain=0.5,
            guided_joint_gain=0.5,
            guided_root_pos_clip=0.05,
            guided_joint_clip=0.35,
        )

        reference = components.rollout_reference_factory(
            start=0,
            motion=_motion(),
            controls=torch.zeros(3, QPOS_DIM - 1),
            actor_params=_constant_actor(),
            model_bundle=_model_bundle(),
            runtime=_Runtime(),
            obs_initialized=True,
        )

        guided = reference["guided_controls"]
        self.assertEqual(guided.shape, (3, QPOS_DIM - 1))
        np.testing.assert_allclose(guided[0], 0.0)
        np.testing.assert_allclose(guided[1:, 0], 0.05)
        np.testing.assert_allclose(guided[1:, 6:], 0.35)

    def test_reference_factory_can_skip_guided_controls_for_execute_trace(self) -> None:
        calls = 0

        def physics_step_fn(
            model_bundle,
            robot_state,
            command_qpos,
            action,
            step_index,
            *,
            runtime,
        ):
            del model_bundle, robot_state, action, step_index, runtime
            nonlocal calls
            calls += 1
            samples = int(command_qpos.shape[0])
            bodies = len(MUJOCO_BODY_NAMES)
            qpos = np.asarray(command_qpos, dtype=np.float32).copy()
            body_pos = np.zeros((samples, bodies, 3), dtype=np.float32)
            return {
                "qpos": qpos,
                "qvel": np.zeros((samples, QVEL_DIM), dtype=np.float32),
                "body_pos_w": body_pos,
                "body_quat_w": _identity_body_quat((samples, bodies)),
                "body_ang_vel_w": np.zeros((samples, bodies, 3), dtype=np.float32),
            }, {
                "root_pos": qpos[:, :3],
                "body_pos": body_pos[:, :2],
                "ee_pos": body_pos[:, :1],
                "contact": np.zeros((samples, 2), dtype=np.float32),
            }

        components = build_mjx_rollout_components(
            runtime=_Runtime(),
            physics_step_fn=physics_step_fn,
            use_guided_candidate=True,
        )

        reference = components.rollout_reference_factory(
            start=0,
            motion=_motion(),
            controls=torch.zeros(3, QPOS_DIM - 1),
            actor_params=_constant_actor(),
            model_bundle=_model_bundle(),
            runtime=_Runtime(),
            include_guided_candidate=False,
        )

        self.assertNotIn("guided_controls", reference)
        self.assertEqual(calls, 0)


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


def _identity_body_quat(shape: tuple[int, ...]) -> np.ndarray:
    quat = np.zeros((*shape, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    return quat


if __name__ == "__main__":
    unittest.main()
