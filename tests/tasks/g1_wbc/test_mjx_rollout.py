from dataclasses import is_dataclass
import unittest

import numpy as np

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    ANCHOR_BODY_NAME,
    COMMAND_BODY_NAMES,
    LIMB_EE_BODY_NAMES,
    MUJOCO_BODY_NAMES,
    OBS_DIM,
    QPOS_DIM,
    QVEL_DIM,
    TRACKING_ANCHOR_BODY_NAME,
)
from spider.tasks.g1_wbc.mjx_obs import (
    JaxObsIndices,
    JaxObsState,
    OBS_FIELD_ORDER,
    OBS_FIELD_SPECS,
)
from spider.tasks.g1_wbc.mjx_optimizer import JaxWindowOptimizerConfig, optimize_window
from spider.tasks.g1_wbc.mjx_policy import JaxActorParams
from spider.tasks.g1_wbc.mjx_rollout import (
    controls_to_qpos,
    make_rollout_scorer,
    score_candidate_controls,
)
from spider.tasks.g1_wbc.mjx_scoring import JaxScoreWeights


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
    def argmax(value):
        return np.argmax(value)

    @staticmethod
    def abs(value):
        return np.abs(value)

    @staticmethod
    def arange(stop):
        return np.arange(stop, dtype=np.int32)

    @staticmethod
    def logical_or(left, right):
        return np.logical_or(left, right)


class _FakeRandom:
    @staticmethod
    def normal(key, shape):
        seed = int(np.asarray(key, dtype=np.uint32).sum())
        rng = np.random.default_rng(seed)
        return rng.normal(size=shape).astype(np.float32)


class _FakeNN:
    @staticmethod
    def softmax(value):
        value = np.asarray(value, dtype=np.float32)
        shifted = value - np.max(value)
        exp = np.exp(shifted)
        return exp / np.sum(exp)


class _FakeJax:
    random = _FakeRandom()
    nn = _FakeNN()


class _RecordingJitJax:
    random = _FakeRandom()
    nn = _FakeNN()

    def __init__(self) -> None:
        self.jit_calls = 0

    def jit(self, fn):
        self.jit_calls += 1

        def wrapped(*args):
            return fn(*args)

        return wrapped


class _RecordingLax:
    def __init__(self) -> None:
        self.calls: list[dict[str, int]] = []

    def scan(self, fn, init, xs=None, length=None):
        if xs is None:
            steps = list(range(int(length)))
        else:
            steps = list(xs)
        self.calls.append({"steps": len(steps)})
        carry = init
        signature = self._carry_signature(carry)
        for step in steps:
            self.assert_valid_carry(carry, signature)
            carry, _ = fn(carry, step)
            self.assert_valid_carry(carry, signature)
        return carry, None

    def assert_valid_carry(self, value, signature) -> None:
        actual = self._carry_signature(value)
        if actual != signature:
            raise AssertionError(f"scan carry structure changed: {actual} != {signature}")

    def _carry_signature(self, value):
        if value is None:
            raise AssertionError("scan carry must not contain None")
        if is_dataclass(value):
            raise AssertionError("scan carry must not contain dataclass instances")
        if isinstance(value, tuple):
            return ("tuple", tuple(self._carry_signature(item) for item in value))
        if isinstance(value, dict):
            return (
                "dict",
                tuple(
                    (key, self._carry_signature(value[key]))
                    for key in sorted(value)
                ),
            )
        arr = np.asarray(value)
        return ("array", tuple(arr.shape), str(arr.dtype))


class _FakeRuntime:
    jnp = _NumpyJnp()
    jax = _FakeJax()


def _obs_indices() -> JaxObsIndices:
    return JaxObsIndices(
        command_body_indices=[
            MUJOCO_BODY_NAMES.index(name) for name in COMMAND_BODY_NAMES
        ],
        limb_indices=[COMMAND_BODY_NAMES.index(name) for name in LIMB_EE_BODY_NAMES],
        anchor_index=COMMAND_BODY_NAMES.index(ANCHOR_BODY_NAME),
        tracking_anchor_index=COMMAND_BODY_NAMES.index(TRACKING_ANCHOR_BODY_NAME),
    )


def _identity_body_quat(shape: tuple[int, ...]) -> np.ndarray:
    quat = np.zeros((*shape, 4), dtype=np.float32)
    quat[..., 0] = 1.0
    return quat


def _initial_robot_state(samples: int = 1) -> dict[str, np.ndarray]:
    bodies = len(MUJOCO_BODY_NAMES)
    qpos = np.zeros((samples, QPOS_DIM), dtype=np.float32)
    qpos[:, 3] = 1.0
    qvel = np.zeros((samples, QVEL_DIM), dtype=np.float32)
    return {
        "qpos": qpos,
        "qvel": qvel,
        "body_pos_w": np.zeros((samples, bodies, 3), dtype=np.float32),
        "body_quat_w": _identity_body_quat((samples, bodies)),
        "body_ang_vel_w": np.zeros((samples, bodies, 3), dtype=np.float32),
    }


def _rollout_reference(
    *,
    samples: int,
    horizon: int,
    actor_params: JaxActorParams | None = None,
) -> dict[str, object]:
    del actor_params
    command_bodies = len(COMMAND_BODY_NAMES)
    score_bodies = 2
    return {
        "initial_robot_state": _initial_robot_state(samples),
        "obs_reference": {
            "joint_pos": np.zeros((horizon, ACTION_DIM), dtype=np.float32),
            "joint_vel": np.zeros((horizon, ACTION_DIM), dtype=np.float32),
            "body_pos_w": np.zeros((horizon, command_bodies, 3), dtype=np.float32),
            "body_quat_w": _identity_body_quat((horizon, command_bodies)),
            "body_ang_vel_w": np.zeros(
                (horizon, command_bodies, 3),
                dtype=np.float32,
            ),
        },
        "score_reference": {
            "root_pos": np.zeros((horizon, 3), dtype=np.float32),
            "body_pos": np.zeros((horizon, score_bodies, 3), dtype=np.float32),
            "ee_pos": np.zeros((horizon, 1, 3), dtype=np.float32),
            "contact": np.zeros((horizon, 2), dtype=np.float32),
        },
        "obs_state": JaxObsState(
            history=None,
            last_action=np.zeros((samples, ACTION_DIM), dtype=np.float32),
        ),
        "obs_indices": _obs_indices(),
        "default_joint_pos": np.zeros(ACTION_DIM, dtype=np.float32),
        "joint_low": np.full(ACTION_DIM, -0.5, dtype=np.float32),
        "joint_high": np.full(ACTION_DIM, 0.5, dtype=np.float32),
        "prev_control": np.zeros((samples, QPOS_DIM - 1), dtype=np.float32),
        "score_weights": JaxScoreWeights({"root_pos": 1.0, "control_delta": 0.05}),
    }


def _constant_actor(action: np.ndarray) -> JaxActorParams:
    return JaxActorParams(
        obs_mean=np.zeros((1, OBS_DIM), dtype=np.float32),
        obs_std=np.ones((1, OBS_DIM), dtype=np.float32),
        layers=((np.zeros((OBS_DIM, ACTION_DIM), dtype=np.float32), action),),
    )


def _feedback_actor() -> JaxActorParams:
    bias = np.zeros(ACTION_DIM, dtype=np.float32)
    bias[0] = 0.1
    weight = np.zeros((OBS_DIM, ACTION_DIM), dtype=np.float32)
    offset = 0
    for name in OBS_FIELD_ORDER:
        if name == "actions":
            break
        offset += int(np.prod(OBS_FIELD_SPECS[name]))
    weight[offset + (5 * ACTION_DIM) - ACTION_DIM, 0] = 1.0
    return JaxActorParams(
        obs_mean=np.zeros((1, OBS_DIM), dtype=np.float32),
        obs_std=np.ones((1, OBS_DIM), dtype=np.float32),
        layers=((weight, bias),),
    )


def _physics_step(model_bundle, robot_state, command_qpos, action, step_index, *, runtime):
    del model_bundle, robot_state, step_index, runtime
    samples = int(command_qpos.shape[0])
    bodies = len(MUJOCO_BODY_NAMES)
    qvel = np.zeros((samples, QVEL_DIM), dtype=np.float32)
    qvel[:, 6:] = action
    body_pos = np.zeros((samples, bodies, 3), dtype=np.float32)
    body_pos[:, 0, :3] = command_qpos[:, :3]
    next_robot = {
        "qpos": command_qpos,
        "qvel": qvel,
        "body_pos_w": body_pos,
        "body_quat_w": _identity_body_quat((samples, bodies)),
        "body_ang_vel_w": np.zeros((samples, bodies, 3), dtype=np.float32),
    }
    score_state = {
        "root_pos": command_qpos[:, :3],
        "body_pos": body_pos[:, :2],
        "ee_pos": body_pos[:, :1],
        "contact": np.zeros((samples, 2), dtype=np.float32),
    }
    return next_robot, score_state


class MjxRolloutTest(unittest.TestCase):
    def test_controls_to_qpos_matches_spider_control_semantics(self) -> None:
        base_qpos = np.zeros((2, QPOS_DIM), dtype=np.float32)
        base_qpos[:, 3] = 1.0
        base_qpos[:, 7:] = 0.1
        controls = np.zeros((2, 3, QPOS_DIM - 1), dtype=np.float32)
        controls[:, :, 0] = np.array([[0.2, 0.4, 0.6], [-0.1, -0.2, -0.3]])
        controls[:, :, 6:] = 0.6

        qpos = controls_to_qpos(
            controls,
            base_qpos,
            joint_low=np.full(ACTION_DIM, -0.2, dtype=np.float32),
            joint_high=np.full(ACTION_DIM, 0.5, dtype=np.float32),
            jnp=_NumpyJnp,
        )

        self.assertEqual(qpos.shape, (2, 3, QPOS_DIM))
        np.testing.assert_allclose(qpos[..., 0], controls[..., 0])
        np.testing.assert_allclose(qpos[..., 3], 1.0)
        np.testing.assert_allclose(qpos[..., 4:7], 0.0)
        np.testing.assert_allclose(qpos[..., 7:], 0.5)

    def test_score_candidate_controls_returns_one_score_per_sample(self) -> None:
        samples = np.zeros((3, 4, QPOS_DIM - 1), dtype=np.float32)
        samples[1, :, 0] = 0.2
        samples[2, :, 0] = 0.4
        reference = _rollout_reference(samples=3, horizon=4)
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        first = score_candidate_controls(
            samples,
            reference,
            _constant_actor(action),
            model_bundle=object(),
            runtime=_FakeRuntime,
            physics_step_fn=_physics_step,
        )
        second = score_candidate_controls(
            samples,
            reference,
            _constant_actor(action),
            model_bundle=object(),
            runtime=_FakeRuntime,
            physics_step_fn=_physics_step,
        )

        self.assertEqual(first.shape, (3,))
        np.testing.assert_allclose(first, second)
        self.assertGreater(float(first[0]), float(first[1]))
        self.assertGreater(float(first[1]), float(first[2]))

    def test_score_candidate_controls_can_return_contact_diagnostics(self) -> None:
        samples = np.zeros((2, 3, QPOS_DIM - 1), dtype=np.float32)
        reference = _rollout_reference(samples=2, horizon=3)

        def diagnostic_step(
            model_bundle,
            robot_state,
            command_qpos,
            action,
            step_index,
            *,
            runtime,
        ):
            next_robot, score_state = _physics_step(
                model_bundle,
                robot_state,
                command_qpos,
                action,
                step_index,
                runtime=runtime,
            )
            step = int(step_index)
            score_state["active_contact_count"] = np.array(
                [step + 1, 5 - step],
                dtype=np.float32,
            )
            score_state["contact_pair_count"] = np.array(
                [2 * step, 3 + step],
                dtype=np.float32,
            )
            return next_robot, score_state

        metrics = score_candidate_controls(
            samples,
            reference,
            _constant_actor(np.zeros(ACTION_DIM, dtype=np.float32)),
            model_bundle=object(),
            runtime=_FakeRuntime,
            physics_step_fn=diagnostic_step,
            return_metrics=True,
        )

        self.assertEqual(metrics["score"].shape, (2,))
        np.testing.assert_allclose(metrics["active_contact_count"], [3.0, 5.0])
        np.testing.assert_allclose(metrics["contact_pair_count"], [4.0, 5.0])

    def test_score_candidate_controls_feeds_previous_action_into_next_observation(
        self,
    ) -> None:
        samples = np.zeros((1, 3, QPOS_DIM - 1), dtype=np.float32)
        reference = _rollout_reference(samples=1, horizon=3)
        actions: list[np.ndarray] = []

        def recording_step(
            model_bundle,
            robot_state,
            command_qpos,
            action,
            step_index,
            *,
            runtime,
        ):
            actions.append(np.asarray(action, dtype=np.float32).copy())
            return _physics_step(
                model_bundle,
                robot_state,
                command_qpos,
                action,
                step_index,
                runtime=runtime,
            )

        score_candidate_controls(
            samples,
            reference,
            _feedback_actor(),
            model_bundle=object(),
            runtime=_FakeRuntime,
            physics_step_fn=recording_step,
        )

        self.assertEqual(len(actions), 3)
        expected = [0.1]
        expected.append(0.1 + expected[-1] / 1.01)
        expected.append(0.1 + expected[-1] / 1.01)
        self.assertAlmostEqual(float(actions[0][0, 0]), expected[0], places=6)
        self.assertAlmostEqual(float(actions[1][0, 0]), expected[1], places=6)
        self.assertAlmostEqual(float(actions[2][0, 0]), expected[2], places=6)

    def test_score_candidate_controls_uses_lax_scan_when_available(self) -> None:
        samples = np.zeros((2, 3, QPOS_DIM - 1), dtype=np.float32)
        reference = _rollout_reference(samples=2, horizon=3)
        lax = _RecordingLax()
        runtime = type(
            "ScanRuntime",
            (),
            {
                "jnp": _NumpyJnp(),
                "jax": type("ScanJax", (), {"lax": lax})(),
            },
        )()

        scores = score_candidate_controls(
            samples,
            reference,
            _constant_actor(np.zeros(ACTION_DIM, dtype=np.float32)),
            model_bundle=object(),
            runtime=runtime,
            physics_step_fn=_physics_step,
        )

        self.assertEqual(scores.shape, (2,))
        self.assertEqual(lax.calls, [{"steps": 3}])

    def test_lax_scan_carry_keeps_base_ang_vel_structure_stable(self) -> None:
        samples = np.zeros((2, 3, QPOS_DIM - 1), dtype=np.float32)
        reference = _rollout_reference(samples=2, horizon=3)
        reference["initial_robot_state"]["base_ang_vel_b"] = np.ones(
            (2, 3),
            dtype=np.float32,
        )
        lax = _RecordingLax()
        runtime = type(
            "ScanRuntime",
            (),
            {
                "jnp": _NumpyJnp(),
                "jax": type("ScanJax", (), {"lax": lax})(),
            },
        )()

        scores = score_candidate_controls(
            samples,
            reference,
            _constant_actor(np.zeros(ACTION_DIM, dtype=np.float32)),
            model_bundle=object(),
            runtime=runtime,
            physics_step_fn=_physics_step,
        )

        self.assertEqual(scores.shape, (2,))
        self.assertEqual(lax.calls, [{"steps": 3}])

    def test_lax_scan_materializes_partial_obs_history(self) -> None:
        samples = np.zeros((2, 3, QPOS_DIM - 1), dtype=np.float32)
        reference = _rollout_reference(samples=2, horizon=3)
        reference["obs_state"] = JaxObsState(
            history={
                "actions": np.zeros(
                    (2, OBS_FIELD_SPECS["actions"][0], ACTION_DIM),
                    dtype=np.float32,
                )
            },
            last_action=np.zeros((2, ACTION_DIM), dtype=np.float32),
        )
        lax = _RecordingLax()
        runtime = type(
            "ScanRuntime",
            (),
            {
                "jnp": _NumpyJnp(),
                "jax": type("ScanJax", (), {"lax": lax})(),
            },
        )()

        scores = score_candidate_controls(
            samples,
            reference,
            _constant_actor(np.zeros(ACTION_DIM, dtype=np.float32)),
            model_bundle=object(),
            runtime=runtime,
            physics_step_fn=_physics_step,
        )

        self.assertEqual(scores.shape, (2,))
        self.assertEqual(lax.calls, [{"steps": 3}])

    def test_rollout_scorer_factory_matches_optimizer_callable_contract(self) -> None:
        config = JaxWindowOptimizerConfig(
            samples=2,
            horizon_steps=3,
            control_steps=1,
            knot_count=2,
            temperature=0.5,
            root_pos_sigma=0.01,
            root_rot_sigma=0.01,
            joint_sigma=0.01,
        )
        scorer = make_rollout_scorer(
            runtime=_FakeRuntime,
            physics_step_fn=_physics_step,
        )

        result = optimize_window(
            config,
            {"rollout_fn": scorer},
            np.zeros((3, QPOS_DIM - 1), dtype=np.float32),
            reference=_rollout_reference(samples=2, horizon=3),
            actor_params=_constant_actor(np.zeros(ACTION_DIM, dtype=np.float32)),
            model_bundle=object(),
            key=(0, 4),
            runtime=_FakeRuntime,
        )

        self.assertEqual(result.updated_controls.shape, (3, QPOS_DIM - 1))
        self.assertEqual(result.execute_chunk.shape, (2, QPOS_DIM - 1))

    def test_rollout_scorer_jits_once_per_model_bundle(self) -> None:
        jax = _RecordingJitJax()
        runtime = type("JitRuntime", (), {"jnp": _NumpyJnp(), "jax": jax})()
        scorer = make_rollout_scorer(
            runtime=runtime,
            physics_step_fn=_physics_step,
        )
        samples = np.zeros((2, 3, QPOS_DIM - 1), dtype=np.float32)
        reference = _rollout_reference(samples=2, horizon=3)
        actor_params = _constant_actor(np.zeros(ACTION_DIM, dtype=np.float32))
        model_bundle = object()

        first = scorer(samples, reference, actor_params, model_bundle)
        second = scorer(samples, reference, actor_params, model_bundle)

        self.assertEqual(jax.jit_calls, 1)
        np.testing.assert_allclose(first["score"], second["score"])
        self.assertIn("active_contact_count", first)


if __name__ == "__main__":
    unittest.main()
