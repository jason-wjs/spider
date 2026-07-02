import unittest

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


class _FakeRandom:
    @staticmethod
    def normal(key, shape):
        rng = np.random.default_rng(int(key))
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


class _FakeRuntime:
    jnp = _NumpyJnp()
    jax = _FakeJax()


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

        first = sample_residual_controls(config, controls, 17, runtime=_FakeRuntime)
        second = sample_residual_controls(config, controls, 17, runtime=_FakeRuntime)
        different = sample_residual_controls(config, controls, 18, runtime=_FakeRuntime)

        np.testing.assert_allclose(first, second)
        self.assertGreater(np.max(np.abs(first - different)), 1.0e-6)
        np.testing.assert_allclose(first[0], controls)

    def test_sample_residual_controls_uses_segment_sigmas(self) -> None:
        controls = np.zeros((5, 8), dtype=np.float32)

        samples = sample_residual_controls(_config(), controls, 3, runtime=_FakeRuntime)

        self.assertEqual(samples.shape, (4, 5, 8))
        nonzero = samples[1:]
        self.assertGreater(np.max(np.abs(nonzero[..., :3])), 0.0)
        self.assertGreater(np.max(np.abs(nonzero[..., 3:6])), 0.0)
        self.assertGreater(np.max(np.abs(nonzero[..., 6:])), 0.0)

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
            key=11,
            runtime=_FakeRuntime,
        )

        scores = np.array([0.0, 1.0, 3.0, -2.0], dtype=np.float32)
        weights = _FakeNN.softmax(scores / config.temperature)
        expected = np.sum(captured["samples"] * weights[:, None, None], axis=0)
        self.assertEqual(result.updated_controls.shape, controls.shape)
        self.assertEqual(result.execute_chunk.shape, (config.control_steps + 1, 8))
        np.testing.assert_allclose(result.updated_controls, expected, rtol=1.0e-6)
        np.testing.assert_allclose(result.execute_chunk, expected[:3], rtol=1.0e-6)
        self.assertEqual(result.info["best_index"], 2.0)
        self.assertEqual(result.info["best_score"], 3.0)

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


if __name__ == "__main__":
    unittest.main()
