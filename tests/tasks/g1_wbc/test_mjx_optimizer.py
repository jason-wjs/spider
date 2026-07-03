import unittest
from types import SimpleNamespace

import numpy as np

from spider.tasks.g1_wbc.mjx_optimizer import (
    JaxWindowOptimizerConfig,
    optimize_window,
    sample_residual_controls,
)


class _NumpyJnp:
    @staticmethod
    def asarray(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def concatenate(values, axis=0):
        return np.concatenate(values, axis=axis)

    @staticmethod
    def repeat(value, repeats, axis=0):
        return np.repeat(value, repeats, axis=axis)

    @staticmethod
    def full(shape, value):
        return np.full(shape, value, dtype=np.float32)

    @staticmethod
    def sum(value, axis=None):
        return np.sum(value, axis=axis)

    @staticmethod
    def argmax(value):
        return np.argmax(value)

    @staticmethod
    def mean(value):
        return np.mean(value)

    @staticmethod
    def max(value):
        return np.max(value)

    @staticmethod
    def linspace(start, stop, num):
        return np.linspace(start, stop, int(num), dtype=np.float32)

    @staticmethod
    def floor(value):
        return np.floor(value)

    @staticmethod
    def minimum(left, right):
        return np.minimum(left, right)


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


class _StrictRandom:
    @staticmethod
    def PRNGKey(seed):
        return ("key", int(seed))

    @staticmethod
    def fold_in(key, value):
        return ("fold", key, int(value))

    @staticmethod
    def normal(key, shape):
        if not (isinstance(key, tuple) and key and key[0] == "fold"):
            raise TypeError("normal requires folded PRNGKey")
        seed = int(np.asarray(key, dtype=object)[-1])
        rng = np.random.default_rng(seed)
        return rng.normal(size=shape).astype(np.float32)


class _StrictJax:
    random = _StrictRandom()
    nn = _FakeNN()


class _FakeRuntime:
    jnp = _NumpyJnp()
    jax = _FakeJax()


class _StrictRuntime:
    jnp = _NumpyJnp()
    jax = _StrictJax()


class _UnitStepRandom:
    def __init__(self) -> None:
        self.calls = 0

    def normal(self, key, shape):
        del key
        self.calls += 1
        noise = np.zeros(shape, dtype=np.float32)
        noise[1:] = 1.0
        return noise


class _RecordingKnotRandom:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    def normal(self, key, shape):
        del key
        self.shapes.append(tuple(int(dim) for dim in shape))
        noise = np.zeros(shape, dtype=np.float32)
        noise[1, :, 0] = np.linspace(0.0, 4.0, int(shape[1]), dtype=np.float32)
        return noise


def _config() -> JaxWindowOptimizerConfig:
    return JaxWindowOptimizerConfig(
        samples=4,
        horizon_steps=5,
        control_steps=2,
        knot_count=3,
        temperature=0.5,
        root_pos_sigma=0.01,
        root_rot_sigma=0.02,
        joint_sigma=0.03,
    )


class MjxOptimizerTest(unittest.TestCase):
    def test_sample_residual_controls_is_seed_deterministic(self) -> None:
        controls = np.zeros((5, 8), dtype=np.float32)
        config = _config()

        first = sample_residual_controls(config, controls, (0, 17), runtime=_FakeRuntime)
        second = sample_residual_controls(config, controls, (0, 17), runtime=_FakeRuntime)
        different = sample_residual_controls(config, controls, (0, 18), runtime=_FakeRuntime)

        np.testing.assert_allclose(first, second)
        self.assertGreater(np.max(np.abs(first - different)), 1.0e-6)
        np.testing.assert_allclose(first[0], controls)

    def test_sample_residual_controls_uses_segment_sigmas(self) -> None:
        controls = np.zeros((5, 8), dtype=np.float32)

        samples = sample_residual_controls(_config(), controls, (0, 3), runtime=_FakeRuntime)

        self.assertEqual(samples.shape, (4, 5, 8))
        nonzero = samples[1:]
        self.assertGreater(np.max(np.abs(nonzero[..., :3])), 0.0)
        self.assertGreater(np.max(np.abs(nonzero[..., 3:6])), 0.0)
        self.assertGreater(np.max(np.abs(nonzero[..., 6:])), 0.0)

    def test_sample_residual_controls_samples_knots_and_interpolates_horizon(self) -> None:
        random = _RecordingKnotRandom()
        runtime = SimpleNamespace(
            jnp=_NumpyJnp(),
            jax=SimpleNamespace(random=random, nn=_FakeNN()),
        )
        config = JaxWindowOptimizerConfig(
            samples=2,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=1.0,
            root_rot_sigma=1.0,
            joint_sigma=1.0,
        )

        samples = sample_residual_controls(
            config,
            np.zeros((5, 8), dtype=np.float32),
            (0, 3),
            runtime=runtime,
        )

        self.assertEqual(random.shapes, [(2, 3, 8)])
        self.assertEqual(samples.shape, (2, 5, 8))
        np.testing.assert_allclose(samples[0], 0.0)
        np.testing.assert_allclose(samples[1, :, 0], [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_sample_residual_controls_converts_tuple_seed_to_prng_key(self) -> None:
        controls = np.zeros((5, 8), dtype=np.float32)

        samples = sample_residual_controls(
            _config(),
            controls,
            (3, 17),
            runtime=_StrictRuntime,
        )

        self.assertEqual(samples.shape, (4, 5, 8))
        np.testing.assert_allclose(samples[0], controls)

    def test_optimize_window_weighted_update_and_execute_chunk(self) -> None:
        config = _config()
        controls = np.zeros((5, 8), dtype=np.float32)
        captured: dict[str, np.ndarray] = {}

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            captured["samples"] = np.asarray(samples, dtype=np.float32)
            return np.array([0.0, 1.0, 3.0, -2.0], dtype=np.float32)

        result = optimize_window(
            config,
            {"rollout_fn": rollout_fn},
            controls,
            reference={"unused": True},
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=_FakeRuntime,
        )

        scores = np.array([0.0, 1.0, 3.0, -2.0], dtype=np.float32)
        weights = _FakeNN.softmax(scores / config.temperature)
        expected = np.sum(captured["samples"] * weights[:, None, None], axis=0)
        self.assertEqual(result.updated_controls.shape, controls.shape)
        self.assertEqual(result.execute_chunk.shape, (config.control_steps + 1, 8))
        np.testing.assert_allclose(result.updated_controls, expected, rtol=1.0e-6)
        np.testing.assert_allclose(result.execute_chunk, expected[:3], rtol=1.0e-6)
        self.assertEqual(float(result.info["best_index"]), 2.0)
        self.assertEqual(float(result.info["best_score"]), 3.0)
        self.assertTrue(result.info["accepted"])
        self.assertEqual(float(result.info["control_score"]), 0.0)
        self.assertEqual(float(result.info["score_improvement"]), 3.0)

    def test_optimize_window_rejects_when_best_sample_is_current_controls(self) -> None:
        random = _UnitStepRandom()
        runtime = SimpleNamespace(
            jnp=_NumpyJnp(),
            jax=SimpleNamespace(random=random, nn=_FakeNN()),
        )
        config = JaxWindowOptimizerConfig(
            samples=3,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=1.0,
            root_rot_sigma=1.0,
            joint_sigma=1.0,
        )
        controls = np.full((5, 8), 0.25, dtype=np.float32)

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return np.array([10.0, 1.0, 5.0], dtype=np.float32)

        result = optimize_window(
            config,
            {"rollout_fn": rollout_fn},
            controls,
            reference=None,
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=runtime,
        )

        self.assertFalse(result.info["accepted"])
        self.assertEqual(float(result.info["best_index"]), 0.0)
        self.assertEqual(float(result.info["control_score"]), 10.0)
        self.assertEqual(float(result.info["score_improvement"]), 0.0)
        np.testing.assert_allclose(result.updated_controls, controls)
        np.testing.assert_allclose(result.execute_chunk, controls[:3])

    def test_optimize_window_rejects_positive_score_with_zero_control_delta(self) -> None:
        config = JaxWindowOptimizerConfig(
            samples=3,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.0,
            root_rot_sigma=0.0,
            joint_sigma=0.0,
        )
        controls = np.full((5, 8), 0.25, dtype=np.float32)

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return np.array([0.0, 5.0, 10.0], dtype=np.float32)

        result = optimize_window(
            config,
            {"rollout_fn": rollout_fn},
            controls,
            reference=None,
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=_FakeRuntime,
        )

        self.assertFalse(result.info["accepted"])
        self.assertEqual(float(result.info["best_index"]), 2.0)
        self.assertEqual(float(result.info["score_improvement"]), 10.0)
        self.assertEqual(float(result.info["control_delta_max"]), 0.0)
        np.testing.assert_allclose(result.updated_controls, controls)
        np.testing.assert_allclose(result.execute_chunk, controls[:3])

    def test_optimize_window_runs_configured_iterations(self) -> None:
        random = _UnitStepRandom()
        runtime = SimpleNamespace(
            jnp=_NumpyJnp(),
            jax=SimpleNamespace(random=random, nn=_FakeNN()),
        )
        config = JaxWindowOptimizerConfig(
            samples=3,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=1.0,
            root_rot_sigma=1.0,
            joint_sigma=1.0,
            iterations=3,
        )
        first_candidates: list[np.ndarray] = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            first_candidates.append(np.asarray(samples[0], dtype=np.float32).copy())
            return np.array([0.0, 100.0, 100.0], dtype=np.float32)

        result = optimize_window(
            config,
            {"rollout_fn": rollout_fn},
            np.zeros((5, 8), dtype=np.float32),
            reference={"unused": True},
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=runtime,
        )

        self.assertEqual(random.calls, 3)
        self.assertEqual(len(first_candidates), 3)
        np.testing.assert_allclose(first_candidates[0], 0.0)
        self.assertGreater(float(np.mean(first_candidates[1])), 0.9)
        self.assertGreater(
            float(np.mean(first_candidates[2])),
            float(np.mean(first_candidates[1])) + 0.9,
        )
        self.assertGreater(float(np.mean(result.updated_controls)), 2.9)
        self.assertTrue(result.info["accepted"])
        self.assertEqual(result.info["accepted_iterations"], 3)

    def test_optimize_window_decays_noise_by_iteration(self) -> None:
        random = _UnitStepRandom()
        runtime = SimpleNamespace(
            jnp=_NumpyJnp(),
            jax=SimpleNamespace(random=random, nn=_FakeNN()),
        )
        config = JaxWindowOptimizerConfig(
            samples=3,
            horizon_steps=5,
            control_steps=2,
            knot_count=5,
            temperature=0.5,
            root_pos_sigma=1.0,
            root_rot_sigma=1.0,
            joint_sigma=1.0,
            iterations=3,
            final_noise_scale=0.125,
        )
        sample_means: list[float] = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            sample_means.append(float(np.asarray(samples[1], dtype=np.float32).mean()))
            return np.array([100.0, 0.0, 0.0], dtype=np.float32)

        result = optimize_window(
            config,
            {"rollout_fn": rollout_fn},
            np.zeros((5, 8), dtype=np.float32),
            reference={"unused": True},
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=runtime,
        )

        self.assertEqual(random.calls, 3)
        np.testing.assert_allclose(sample_means, [1.0, 0.5, 0.25], rtol=1e-6)
        self.assertEqual(result.info["accepted_iterations"], 0)
        self.assertFalse(result.info["accepted"])

    def test_optimize_window_propagates_rollout_diagnostics(self) -> None:
        config = _config()

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return {
                "score": np.array([0.0, 1.0, 3.0, -2.0], dtype=np.float32),
                "active_contact_count": np.array([2, 5, 7, 1], dtype=np.float32),
                "contact_pair_count": np.array([4, 6, 9, 3], dtype=np.float32),
                "physics_step_count": np.array([3, 5, 2, 4], dtype=np.float32),
            }

        result = optimize_window(
            config,
            {"rollout_fn": rollout_fn},
            np.zeros((5, 8), dtype=np.float32),
            reference=None,
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=_FakeRuntime,
        )

        self.assertEqual(float(result.info["best_score"]), 3.0)
        self.assertEqual(float(result.info["active_contact_count"]), 7.0)
        self.assertEqual(float(result.info["contact_pair_count"]), 9.0)
        self.assertEqual(float(result.info["physics_step_count"]), 5.0)

    def test_optimize_window_rejects_column_vector_scores(self) -> None:
        config = _config()

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return np.zeros((config.samples, 1), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "scores shape"):
            optimize_window(
                config,
                {"rollout_fn": rollout_fn},
                np.zeros((5, 8), dtype=np.float32),
                reference=None,
                actor_params=None,
                model_bundle=None,
                key=(0, 7),
                runtime=_FakeRuntime,
            )

    def test_optimize_window_rejects_nonfinite_scores(self) -> None:
        config = _config()

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return np.array([0.0, np.nan, 1.0, 2.0], dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "finite"):
            optimize_window(
                config,
                {"rollout_fn": rollout_fn},
                np.zeros((5, 8), dtype=np.float32),
                reference=None,
                actor_params=None,
                model_bundle=None,
                key=(0, 7),
                runtime=_FakeRuntime,
            )

    def test_optimize_window_rejects_wrong_control_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected controls horizon"):
            sample_residual_controls(
                _config(),
                np.zeros((4, 8), dtype=np.float32),
                0,
                runtime=_FakeRuntime,
            )

    def test_optimizer_rejects_invalid_config_and_control_rank(self) -> None:
        config = _config()

        with self.assertRaisesRegex(ValueError, "at least one sample"):
            sample_residual_controls(
                JaxWindowOptimizerConfig(
                    samples=0,
                    horizon_steps=5,
                    control_steps=2,
                    knot_count=3,
                    temperature=0.5,
                    root_pos_sigma=0.01,
                    root_rot_sigma=0.02,
                    joint_sigma=0.03,
                ),
                np.zeros((5, 8), dtype=np.float32),
                0,
                runtime=_FakeRuntime,
            )

        with self.assertRaisesRegex(ValueError, "Expected 2D controls"):
            sample_residual_controls(
                config,
                np.zeros((5, 1, 8), dtype=np.float32),
                0,
                runtime=_FakeRuntime,
            )

        with self.assertRaisesRegex(ValueError, "smaller than horizon"):
            sample_residual_controls(
                JaxWindowOptimizerConfig(
                    samples=4,
                    horizon_steps=5,
                    control_steps=5,
                    knot_count=3,
                    temperature=0.5,
                    root_pos_sigma=0.01,
                    root_rot_sigma=0.02,
                    joint_sigma=0.03,
                ),
                np.zeros((5, 8), dtype=np.float32),
                0,
                runtime=_FakeRuntime,
            )

        with self.assertRaisesRegex(ValueError, "final_noise_scale"):
            sample_residual_controls(
                JaxWindowOptimizerConfig(
                    samples=4,
                    horizon_steps=5,
                    control_steps=2,
                    knot_count=3,
                    temperature=0.5,
                    root_pos_sigma=0.01,
                    root_rot_sigma=0.02,
                    joint_sigma=0.03,
                    final_noise_scale=-0.1,
                ),
                np.zeros((5, 8), dtype=np.float32),
                0,
                runtime=_FakeRuntime,
            )


if __name__ == "__main__":
    unittest.main()
