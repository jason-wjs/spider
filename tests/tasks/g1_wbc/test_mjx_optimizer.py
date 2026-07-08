import unittest
from types import SimpleNamespace

import numpy as np

from spider.tasks.g1_wbc.mjx_optimizer import (
    JaxWindowOptimizerConfig,
    _validate_scores,
    make_jitted_window_optimizer,
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
    def argsort(value):
        return np.argsort(value)

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

    @staticmethod
    def maximum(left, right):
        return np.maximum(left, right)

    @staticmethod
    def sqrt(value):
        return np.sqrt(value)

    @staticmethod
    def arange(stop, dtype=None):
        return np.arange(stop, dtype=dtype)


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


class _CountingNN:
    def __init__(self):
        self.softmax_calls = 0

    def softmax(self, value):
        self.softmax_calls += 1
        return _FakeNN.softmax(value)


class _BoolTrapScalar:
    shape = ()

    def __bool__(self):
        raise AssertionError("device scalar must not be converted to bool")


class _JaxLikeScores:
    __module__ = "jaxlib._jax"

    shape = (4,)


class _BoolTrapJnp:
    @staticmethod
    def isfinite(value):
        del value
        return _JaxLikeScores()

    @staticmethod
    def all(value):
        del value
        return _BoolTrapScalar()


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


class _RecordingJitJax:
    def __init__(self) -> None:
        self.random = _FakeRandom()
        self.nn = _FakeNN()
        self.jit_calls = 0

    def jit(self, fn):
        self.jit_calls += 1

        def wrapped(*args):
            return fn(*args)

        return wrapped


class _StringRejectingJitJax(_RecordingJitJax):
    def jit(self, fn):
        self.jit_calls += 1

        def wrapped(*args):
            output = fn(*args)
            _reject_string_leaves(output)
            return output

        return wrapped


def _reject_string_leaves(value):
    if isinstance(value, str):
        raise TypeError("JIT output cannot contain str")
    if isinstance(value, dict):
        for child in value.values():
            _reject_string_leaves(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_string_leaves(child)


class _UnitStepRandom:
    def __init__(self) -> None:
        self.calls = 0

    def normal(self, key, shape):
        del key
        self.calls += 1
        noise = np.zeros(shape, dtype=np.float32)
        noise[1:] = 1.0
        return noise


class _DistinctCandidateRandom:
    def normal(self, key, shape):
        del key
        noise = np.zeros(shape, dtype=np.float32)
        if int(shape[0]) > 1:
            noise[1] = 1.0
        if int(shape[0]) > 2:
            noise[2] = 2.0
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
        np.testing.assert_allclose(samples[:, 0], np.zeros_like(samples[:, 0]))

    def test_sample_residual_controls_reserves_current_anchor_and_center(
        self,
    ) -> None:
        controls = np.full((5, 8), 2.0, dtype=np.float32)
        anchor = np.zeros((5, 8), dtype=np.float32)
        guided = np.full((5, 8), 3.0, dtype=np.float32)
        runtime = SimpleNamespace(
            jnp=_NumpyJnp(),
            jax=SimpleNamespace(random=_UnitStepRandom(), nn=_FakeNN()),
        )

        samples = sample_residual_controls(
            _config(),
            controls,
            (0, 3),
            runtime=runtime,
            guided_controls=guided,
            anchor_controls=anchor,
            include_center_candidate=True,
        )

        np.testing.assert_allclose(samples[0], anchor)
        np.testing.assert_allclose(samples[1], guided)
        np.testing.assert_allclose(samples[2], controls)

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
        np.testing.assert_allclose(samples[1, 0], 0.0)

    def test_sample_residual_controls_uses_logspace_control_noise_profile(self) -> None:
        runtime = SimpleNamespace(
            jnp=_NumpyJnp(),
            jax=SimpleNamespace(random=_UnitStepRandom(), nn=_FakeNN()),
        )
        config = JaxWindowOptimizerConfig(
            samples=2,
            horizon_steps=3,
            control_steps=1,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=1.0,
            root_rot_sigma=1.0,
            joint_sigma=1.0,
            first_control_noise_scale=0.25,
            last_control_noise_scale=1.0,
            freeze_first_frame=False,
        )

        samples = sample_residual_controls(
            config,
            np.zeros((3, 8), dtype=np.float32),
            (0, 3),
            runtime=runtime,
        )

        np.testing.assert_allclose(samples[1, :, 0], [0.25, 0.5, 1.0])

    def test_sample_residual_controls_reserves_guided_candidate_slot(self) -> None:
        controls = np.zeros((5, 8), dtype=np.float32)
        guided_controls = np.full((5, 8), 2.0, dtype=np.float32)
        runtime = SimpleNamespace(
            jnp=_NumpyJnp(),
            jax=SimpleNamespace(random=_UnitStepRandom(), nn=_FakeNN()),
        )

        samples = sample_residual_controls(
            _config(),
            controls,
            (0, 3),
            runtime=runtime,
            guided_controls=guided_controls,
        )

        np.testing.assert_allclose(samples[0], controls)
        np.testing.assert_allclose(samples[1], guided_controls)
        self.assertGreater(float(np.mean(samples[2])), 0.0)

    def test_sample_residual_controls_rejects_malformed_guided_candidate(
        self,
    ) -> None:
        controls = np.zeros((5, 8), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "guided_controls shape"):
            sample_residual_controls(
                _config(),
                controls,
                (0, 3),
                runtime=_FakeRuntime,
                guided_controls=np.zeros((4, 8), dtype=np.float32),
            )

        nonfinite = np.zeros((5, 8), dtype=np.float32)
        nonfinite[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "guided_controls must be finite"):
            sample_residual_controls(
                _config(),
                controls,
                (0, 3),
                runtime=_FakeRuntime,
                guided_controls=nonfinite,
            )

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

    def test_optimize_window_executes_best_candidate(self) -> None:
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

        expected = captured["samples"][2]
        self.assertEqual(result.updated_controls.shape, controls.shape)
        self.assertEqual(result.execute_chunk.shape, (config.control_steps + 1, 8))
        np.testing.assert_allclose(result.updated_controls, expected, rtol=1.0e-6)
        np.testing.assert_allclose(result.execute_chunk, expected[:3], rtol=1.0e-6)
        self.assertEqual(float(result.info["best_index"]), 2.0)
        self.assertEqual(float(result.info["best_score"]), 3.0)
        self.assertEqual(float(result.info["second_best_score"]), 1.0)
        self.assertEqual(float(result.info["top_score_gap"]), 2.0)
        self.assertTrue(result.info["accepted"])
        self.assertEqual(float(result.info["control_score"]), 0.0)
        self.assertEqual(float(result.info["score_improvement"]), 3.0)
        self.assertEqual(
            tuple(float(v) for v in result.info["iteration_top_score_gaps"]),
            (2.0,),
        )

    def test_jitted_window_optimizer_reuses_outer_jit_for_same_signature(self) -> None:
        jax = _RecordingJitJax()
        runtime = SimpleNamespace(jnp=_NumpyJnp(), jax=jax)
        optimizer = make_jitted_window_optimizer(runtime=runtime)
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.04,
            root_rot_sigma=0.10,
            joint_sigma=0.18,
            iterations=2,
            elite_fraction=0.5,
        )
        controls = np.zeros((5, 8), dtype=np.float32)
        rollout_calls = 0

        def rollout_fn(samples, reference, actor_params, model_bundle):
            nonlocal rollout_calls
            del reference, actor_params, model_bundle
            rollout_calls += 1
            return np.asarray(samples, dtype=np.float32).mean(axis=(1, 2))

        model_bundle = object()
        for _ in range(2):
            result = optimizer(
                config=config,
                state={"rollout_fn": rollout_fn},
                controls=controls,
                reference={"unused": True},
                actor_params=None,
                model_bundle=model_bundle,
                key=(0, 11),
                runtime=runtime,
            )
            self.assertEqual(result.updated_controls.shape, controls.shape)

        self.assertEqual(jax.jit_calls, 1)
        self.assertEqual(rollout_calls, 4)
        self.assertEqual(int(result.info["iterations"]), 2)

    def test_jitted_window_optimizer_matches_python_optimizer_with_jax_arrays(
        self,
    ) -> None:
        try:
            import jax
            import jax.numpy as jnp
        except Exception as exc:
            self.skipTest(f"JAX unavailable: {exc}")

        runtime = SimpleNamespace(jnp=jnp, jax=jax)
        optimizer = make_jitted_window_optimizer(runtime=runtime)
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.04,
            root_rot_sigma=0.10,
            joint_sigma=0.18,
            iterations=2,
            elite_fraction=0.5,
            sigma_decay=0.75,
            min_root_pos_sigma=0.01,
            min_root_rot_sigma=0.01,
            min_joint_sigma=0.01,
        )
        controls = jnp.zeros((5, 8), dtype=jnp.float32)

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            score = jnp.sum(samples[..., 0], axis=1) + 0.1 * jnp.arange(
                samples.shape[0],
                dtype=jnp.float32,
            )
            return {
                "score": score,
                "physics_step_count": jnp.full(
                    (samples.shape[0],),
                    samples.shape[1],
                    dtype=jnp.float32,
                ),
            }

        kwargs = dict(
            config=config,
            state={"rollout_fn": rollout_fn},
            controls=controls,
            reference={"unused": jnp.asarray(1.0)},
            actor_params=None,
            model_bundle=object(),
            key=(0, 11),
            runtime=runtime,
        )

        expected = optimize_window(**kwargs)
        actual = optimizer(**kwargs)

        np.testing.assert_allclose(
            np.asarray(actual.updated_controls),
            np.asarray(expected.updated_controls),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            np.asarray(actual.execute_chunk),
            np.asarray(expected.execute_chunk),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        self.assertEqual(
            tuple(int(v) for v in actual.info["iteration_best_indices"]),
            tuple(int(v) for v in expected.info["iteration_best_indices"]),
        )
        self.assertEqual(
            tuple(bool(v) for v in actual.info["iteration_accepted_flags"]),
            tuple(bool(v) for v in expected.info["iteration_accepted_flags"]),
        )
        self.assertEqual(float(actual.info["physics_step_count"]), 5.0)

    def test_jitted_window_optimizer_returns_selected_execute_trace(self) -> None:
        try:
            import jax
            import jax.numpy as jnp
        except Exception as exc:
            self.skipTest(f"JAX unavailable: {exc}")

        runtime = SimpleNamespace(jnp=jnp, jax=jax)
        optimizer = make_jitted_window_optimizer(runtime=runtime)
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.01,
            root_rot_sigma=0.02,
            joint_sigma=0.03,
        )

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            qpos = jnp.zeros((3, config.samples, 4), dtype=jnp.float32)
            qpos = qpos.at[:, :, 0].set(
                jnp.arange(config.samples, dtype=jnp.float32)
            )
            return {
                "score": jnp.asarray([0.0, 1.0, 3.0, -2.0], dtype=jnp.float32),
                "execute_trace": {"qpos": qpos},
            }

        result = optimizer(
            config=config,
            state={"rollout_fn": rollout_fn},
            controls=jnp.zeros((5, 8), dtype=jnp.float32),
            reference={"unused": jnp.asarray(1.0)},
            actor_params=None,
            model_bundle=object(),
            key=(0, 11),
            runtime=runtime,
        )

        self.assertIsNotNone(result.execute_trace)
        self.assertEqual(tuple(np.asarray(result.execute_trace["qpos"]).shape), (3, 1, 4))
        np.testing.assert_allclose(
            np.asarray(result.execute_trace["qpos"])[:, 0, 0],
            2.0,
        )

    def test_optimize_window_returns_best_candidate_execute_trace(self) -> None:
        config = _config()
        controls = np.zeros((5, 8), dtype=np.float32)

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            qpos = np.zeros((3, config.samples, 4), dtype=np.float32)
            qpos[:, :, 0] = np.arange(config.samples, dtype=np.float32)
            actions = np.zeros((2, config.samples, 2), dtype=np.float32)
            actions[:, :, 0] = np.arange(config.samples, dtype=np.float32)
            final_qpos = np.zeros((config.samples, 4), dtype=np.float32)
            final_qpos[:, 0] = np.arange(config.samples, dtype=np.float32) + 10.0
            return {
                "score": np.array([0.0, 1.0, 3.0, -2.0], dtype=np.float32),
                "execute_trace": {
                    "qpos": qpos,
                    "actions": actions,
                    "final_robot_state": {"qpos": final_qpos},
                },
            }

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

        self.assertIsNotNone(result.execute_trace)
        self.assertEqual(result.execute_trace["qpos"].shape, (3, 1, 4))
        np.testing.assert_allclose(result.execute_trace["qpos"][:, 0, 0], 2.0)
        self.assertEqual(result.execute_trace["actions"].shape, (2, 1, 2))
        np.testing.assert_allclose(
            result.execute_trace["final_robot_state"]["qpos"][:, 0],
            [12.0],
        )

    def test_optimize_window_preserves_selected_mjx_data_execute_trace(self) -> None:
        config = _config()
        controls = np.zeros((5, 8), dtype=np.float32)
        contact_buffer = np.arange(8, dtype=np.int32).reshape(4, 2)

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return {
                "score": np.array([0.0, 1.0, 3.0, -2.0], dtype=np.float32),
                "execute_trace": {
                    "final_robot_state": {
                        "qpos": np.arange(16, dtype=np.float32).reshape(4, 4),
                        "mjx_data": {
                            "qpos": np.arange(8, dtype=np.float32).reshape(4, 2),
                            "_impl": {
                                "contact__geom": contact_buffer,
                                "efc__force": np.arange(
                                    12,
                                    dtype=np.float32,
                                ).reshape(4, 3),
                            },
                        },
                    },
                },
            }

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

        mjx_data = result.execute_trace["final_robot_state"]["mjx_data"]
        np.testing.assert_allclose(mjx_data["qpos"], [[4.0, 5.0]])
        np.testing.assert_allclose(mjx_data["_impl"]["efc__force"], [[6.0, 7.0, 8.0]])
        np.testing.assert_array_equal(mjx_data["_impl"]["contact__geom"], contact_buffer)

    def test_optimize_window_records_candidate_rank_diagnostics(self) -> None:
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.01,
            root_rot_sigma=0.02,
            joint_sigma=0.03,
            candidate_rank_diagnostics_top_k=3,
        )

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return np.array([0.0, 2.0, 1.0, 4.0], dtype=np.float32)

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

        self.assertEqual(result.info["candidate_rank_diagnostics_top_k"], 3)
        self.assertEqual(
            tuple(
                int(value)
                for value in result.info["iteration_candidate_top_indices"][0]
            ),
            (3, 1, 2),
        )
        self.assertEqual(
            tuple(
                float(value)
                for value in result.info["iteration_candidate_top_scores"][0]
            ),
            (4.0, 2.0, 1.0),
        )
        self.assertEqual(len(result.info["iteration_sample_sums"]), 1)
        self.assertEqual(len(result.info["iteration_sample_squared_sums"]), 1)
        self.assertEqual(len(result.info["iteration_sample_abs_maxes"]), 1)
        self.assertEqual(len(result.info["iteration_sample_checksums"]), 1)
        self.assertEqual(result.info["sample_diagnostics_version"], 3)
        self.assertEqual(tuple(result.info["sample_shape"]), (4, 5, 8))
        self.assertEqual(len(result.info["iteration_sample_checksums"][0]), 3)
        self.assertEqual(tuple(result.info["sample_checksum_slots"]), (0, 1, 2, 3))
        sample_slot_checksums = result.info["iteration_sample_slot_checksums"][0]
        self.assertEqual(len(sample_slot_checksums), 4)
        self.assertTrue(all(len(checksum) == 3 for checksum in sample_slot_checksums))
        self.assertIn("sample_probe_indices", result.info)
        self.assertIn("iteration_sample_probe_values", result.info)
        sample_probe_indices = result.info["sample_probe_indices"]
        sample_probe_values = result.info["iteration_sample_probe_values"][0]
        self.assertEqual(len(sample_probe_indices), len(sample_probe_values))
        self.assertTrue(sample_probe_indices)
        self.assertTrue(all(len(index) == 3 for index in sample_probe_indices))

    def test_optimize_window_omits_candidate_rank_diagnostics_by_default(
        self,
    ) -> None:
        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return np.array([0.0, 2.0, 1.0, 4.0], dtype=np.float32)

        result = optimize_window(
            _config(),
            {"rollout_fn": rollout_fn},
            np.zeros((5, 8), dtype=np.float32),
            reference=None,
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=_FakeRuntime,
        )

        self.assertNotIn("candidate_rank_diagnostics_top_k", result.info)
        self.assertNotIn("iteration_candidate_top_indices", result.info)
        self.assertNotIn("iteration_candidate_top_scores", result.info)
        self.assertNotIn("iteration_sample_sums", result.info)
        self.assertNotIn("iteration_sample_checksums", result.info)
        self.assertNotIn("sample_probe_indices", result.info)
        self.assertNotIn("iteration_sample_probe_values", result.info)

    def test_optimize_window_records_candidate_rescore_diagnostics(self) -> None:
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.01,
            root_rot_sigma=0.02,
            joint_sigma=0.03,
            candidate_rank_diagnostics_top_k=2,
            candidate_rescore_diagnostics=True,
        )
        calls = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            calls.append(len(calls))
            if len(calls) == 1:
                return np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
            return np.array([0.0, 4.0, 2.0, 3.0], dtype=np.float32)

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

        self.assertEqual(len(calls), 2)
        self.assertTrue(result.info["candidate_rescore_diagnostics"])
        self.assertEqual(
            tuple(int(value) for value in result.info["iteration_rescore_best_indices"]),
            (1,),
        )
        self.assertEqual(
            tuple(
                bool(value)
                for value in result.info["iteration_rescore_top1_changed_flags"]
            ),
            (True,),
        )
        self.assertEqual(
            tuple(
                float(value)
                for value in result.info["iteration_rescore_score_delta_maxes"]
            ),
            (3.0,),
        )
        self.assertEqual(
            tuple(
                float(value)
                for value in result.info["iteration_rescore_score_delta_means"]
            ),
            (0.75,),
        )
        self.assertEqual(
            tuple(
                int(value)
                for value in result.info["iteration_rescore_score_delta_max_indices"]
            ),
            (1,),
        )
        self.assertEqual(
            tuple(
                int(value)
                for value in result.info["iteration_rescore_candidate_top_indices"][0]
            ),
            (1, 3),
        )

    def test_optimize_window_omits_candidate_rescore_by_default(self) -> None:
        calls = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            calls.append(len(calls))
            return np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

        result = optimize_window(
            _config(),
            {"rollout_fn": rollout_fn},
            np.zeros((5, 8), dtype=np.float32),
            reference=None,
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=_FakeRuntime,
        )

        self.assertEqual(len(calls), 1)
        self.assertNotIn("candidate_rescore_diagnostics", result.info)
        self.assertNotIn("iteration_rescore_score_delta_maxes", result.info)

    def test_optimize_window_records_score_only_rescore_without_selecting_it(
        self,
    ) -> None:
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.01,
            root_rot_sigma=0.02,
            joint_sigma=0.03,
            score_only_rescore_diagnostics=True,
        )
        primary_calls = []
        shadow_calls = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            primary_calls.append(len(primary_calls))
            return np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

        def score_only_rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            shadow_calls.append(len(shadow_calls))
            return {"score": np.array([0.0, 4.0, 2.0, 1.0], dtype=np.float32)}

        result = optimize_window(
            config,
            {
                "rollout_fn": rollout_fn,
                "score_only_rollout_fn": score_only_rollout_fn,
            },
            np.zeros((5, 8), dtype=np.float32),
            reference=None,
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=_FakeRuntime,
        )

        self.assertEqual(len(primary_calls), 1)
        self.assertEqual(len(shadow_calls), 1)
        self.assertEqual(result.info["best_index"], 3)
        self.assertTrue(result.info["score_only_rescore_diagnostics"])
        self.assertEqual(
            tuple(
                int(value)
                for value in result.info[
                    "iteration_score_only_rescore_best_indices"
                ]
            ),
            (1,),
        )
        self.assertEqual(
            tuple(
                bool(value)
                for value in result.info[
                    "iteration_score_only_rescore_top1_changed_flags"
                ]
            ),
            (True,),
        )
        self.assertEqual(
            tuple(
                float(value)
                for value in result.info[
                    "iteration_score_only_rescore_score_delta_maxes"
                ]
            ),
            (3.0,),
        )

    def test_optimize_window_records_score_only_output_rescore_without_selecting_it(
        self,
    ) -> None:
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.01,
            root_rot_sigma=0.02,
            joint_sigma=0.03,
            score_only_output_rescore_diagnostics=True,
        )
        primary_calls = []
        shadow_calls = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            primary_calls.append(len(primary_calls))
            return np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

        def score_only_output_rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            shadow_calls.append(len(shadow_calls))
            return {"score": np.array([0.0, 4.0, 2.0, 1.0], dtype=np.float32)}

        result = optimize_window(
            config,
            {
                "rollout_fn": rollout_fn,
                "score_only_output_rollout_fn": score_only_output_rollout_fn,
            },
            np.zeros((5, 8), dtype=np.float32),
            reference=None,
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=_FakeRuntime,
        )

        self.assertEqual(len(primary_calls), 1)
        self.assertEqual(len(shadow_calls), 1)
        self.assertEqual(result.info["best_index"], 3)
        self.assertTrue(result.info["score_only_output_rescore_diagnostics"])
        self.assertEqual(
            tuple(
                int(value)
                for value in result.info[
                    "iteration_score_only_output_rescore_best_indices"
                ]
            ),
            (1,),
        )
        self.assertEqual(
            tuple(
                bool(value)
                for value in result.info[
                    "iteration_score_only_output_rescore_top1_changed_flags"
                ]
            ),
            (True,),
        )
        self.assertEqual(
            tuple(
                float(value)
                for value in result.info[
                    "iteration_score_only_output_rescore_score_delta_maxes"
                ]
            ),
            (3.0,),
        )

    def test_optimize_window_records_score_only_output_score_components(
        self,
    ) -> None:
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.01,
            root_rot_sigma=0.02,
            joint_sigma=0.03,
            score_only_output_rescore_diagnostics=True,
            candidate_score_component_diagnostics_top_k=2,
        )

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return {"score": np.array([0.0, 1.0, 3.0, 2.0], dtype=np.float32)}

        def score_only_output_rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return {
                "score": np.array([0.0, 4.0, 2.0, 1.0], dtype=np.float32),
                "root_pos_error_mean": np.array(
                    [0.40, 0.10, 0.20, 0.30],
                    dtype=np.float32,
                ),
                "contact_force_delta_mean": np.array(
                    [40.0, 10.0, 20.0, 30.0],
                    dtype=np.float32,
                ),
            }

        result = optimize_window(
            config,
            {
                "rollout_fn": rollout_fn,
                "score_only_output_rollout_fn": score_only_output_rollout_fn,
            },
            np.zeros((5, 8), dtype=np.float32),
            reference=None,
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=_FakeRuntime,
        )

        self.assertEqual(result.info["best_index"], 2)
        self.assertEqual(
            int(result.info["candidate_score_component_diagnostics_source_code"]),
            1,
        )
        self.assertEqual(result.info["candidate_score_component_diagnostics_field_count"], 2)
        self.assertEqual(
            tuple(int(value) for value in result.info["iteration_score_component_top_indices"][0]),
            (1, 2),
        )
        np.testing.assert_allclose(
            tuple(float(value) for value in result.info["iteration_score_component_selected_values"][0]),
            (0.20, 20.0),
        )
        top_values_by_field = result.info["iteration_score_component_top_values"][0]
        np.testing.assert_allclose(top_values_by_field[0], [0.10, 0.20])
        np.testing.assert_allclose(top_values_by_field[1], [10.0, 20.0])

    def test_optimize_window_records_score_component_peak_sources(self) -> None:
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.01,
            root_rot_sigma=0.02,
            joint_sigma=0.03,
            score_only_output_rescore_diagnostics=True,
            candidate_score_component_diagnostics_top_k=2,
        )
        peak_sources = np.arange(4 * 8, dtype=np.float32).reshape(4, 8)

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return {"score": np.array([0.0, 1.0, 3.0, 2.0], dtype=np.float32)}

        def score_only_output_rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return {
                "score": np.array([0.0, 4.0, 2.0, 1.0], dtype=np.float32),
                "contact_force_peak": np.array(
                    [100.0, 400.0, 200.0, 300.0],
                    dtype=np.float32,
                ),
                "contact_force_peak_source": peak_sources,
            }

        result = optimize_window(
            config,
            {
                "rollout_fn": rollout_fn,
                "score_only_output_rollout_fn": score_only_output_rollout_fn,
            },
            np.zeros((5, 8), dtype=np.float32),
            reference=None,
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=_FakeRuntime,
        )

        self.assertTrue(result.info["candidate_score_component_peak_source_available"])
        self.assertEqual(result.info["candidate_score_component_peak_source_width"], 8)
        np.testing.assert_allclose(
            result.info["iteration_score_component_peak_source_current_values"][0],
            peak_sources[0],
        )
        np.testing.assert_allclose(
            result.info["iteration_score_component_peak_source_selected_values"][0],
            peak_sources[2],
        )
        np.testing.assert_allclose(
            result.info["iteration_score_component_peak_source_top_values"][0],
            peak_sources[[1, 2]],
        )

    def test_score_only_output_score_components_skip_primary_rescore_components(
        self,
    ) -> None:
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.01,
            root_rot_sigma=0.02,
            joint_sigma=0.03,
            candidate_rescore_diagnostics=True,
            score_only_output_rescore_diagnostics=True,
            candidate_score_component_diagnostics_top_k=2,
        )
        primary_calls = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            primary_calls.append(len(primary_calls))
            if len(primary_calls) == 1:
                return {"score": np.array([0.0, 1.0, 3.0, 2.0], dtype=np.float32)}
            return {"score": np.array([0.0, 2.0, 1.0, 4.0], dtype=np.float32)}

        def score_only_output_rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return {
                "score": np.array([0.0, 4.0, 2.0, 1.0], dtype=np.float32),
                "root_pos_error_mean": np.array(
                    [0.40, 0.10, 0.20, 0.30],
                    dtype=np.float32,
                ),
            }

        result = optimize_window(
            config,
            {
                "rollout_fn": rollout_fn,
                "score_only_output_rollout_fn": score_only_output_rollout_fn,
            },
            np.zeros((5, 8), dtype=np.float32),
            reference=None,
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=_FakeRuntime,
        )

        self.assertEqual(len(primary_calls), 2)
        self.assertEqual(
            int(result.info["candidate_score_component_diagnostics_source_code"]),
            1,
        )
        self.assertEqual(
            tuple(result.info["iteration_rescore_score_component_current_values"]),
            (),
        )

    def test_jitted_score_component_source_is_jax_return_compatible(self) -> None:
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.01,
            root_rot_sigma=0.02,
            joint_sigma=0.03,
            score_only_output_rescore_diagnostics=True,
            candidate_score_component_diagnostics_top_k=2,
        )

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return {"score": np.array([0.0, 1.0, 3.0, 2.0], dtype=np.float32)}

        def score_only_output_rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return {
                "score": np.array([0.0, 4.0, 2.0, 1.0], dtype=np.float32),
                "root_pos_error_mean": np.array(
                    [0.40, 0.10, 0.20, 0.30],
                    dtype=np.float32,
                ),
            }

        runtime = type("Runtime", (), {"jnp": _NumpyJnp(), "jax": _StringRejectingJitJax()})()
        optimizer = make_jitted_window_optimizer(runtime=runtime)

        result = optimizer(
            config,
            {
                "rollout_fn": rollout_fn,
                "score_only_output_rollout_fn": score_only_output_rollout_fn,
            },
            np.zeros((5, 8), dtype=np.float32),
            reference=None,
            actor_params=None,
            model_bundle=object(),
            key=(0, 11),
            runtime=runtime,
        )

        self.assertEqual(
            int(result.info["candidate_score_component_diagnostics_source_code"]),
            1,
        )

    def test_optimize_window_records_candidate_score_component_diagnostics(
        self,
    ) -> None:
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.01,
            root_rot_sigma=0.02,
            joint_sigma=0.03,
            candidate_score_component_diagnostics_top_k=2,
        )

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return {
                "score": np.array([0.0, 2.0, 4.0, -1.0], dtype=np.float32),
                "root_pos_error_mean": np.array(
                    [0.40, 0.20, 0.10, 0.90],
                    dtype=np.float32,
                ),
                "contact_mismatch_rate": np.array(
                    [0.04, 0.02, 0.01, 0.09],
                    dtype=np.float32,
                ),
                "contact_force_delta_mean": np.array(
                    [40.0, 20.0, 10.0, 90.0],
                    dtype=np.float32,
                ),
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

        self.assertEqual(result.info["candidate_score_component_diagnostics_top_k"], 2)
        self.assertEqual(result.info["candidate_score_component_diagnostics_version"], 1)
        self.assertEqual(result.info["candidate_score_component_diagnostics_field_count"], 3)
        self.assertEqual(
            tuple(
                int(value)
                for value in result.info[
                    "candidate_score_component_diagnostics_field_indices"
                ]
            ),
            (0, 15, 22),
        )
        self.assertEqual(
            tuple(
                int(value)
                for value in result.info["iteration_score_component_field_indices"][0]
            ),
            (0, 15, 22),
        )
        self.assertEqual(
            tuple(int(value) for value in result.info["iteration_score_component_top_indices"][0]),
            (2, 1),
        )
        np.testing.assert_allclose(
            tuple(float(value) for value in result.info["iteration_score_component_current_values"][0]),
            (0.40, 0.04, 40.0),
        )
        np.testing.assert_allclose(
            tuple(float(value) for value in result.info["iteration_score_component_selected_values"][0]),
            (0.10, 0.01, 10.0),
        )
        top_values_by_field = result.info["iteration_score_component_top_values"][0]
        self.assertIsInstance(top_values_by_field[0], tuple)
        np.testing.assert_allclose(top_values_by_field[0], [0.10, 0.20])
        np.testing.assert_allclose(top_values_by_field[1], [0.01, 0.02])
        np.testing.assert_allclose(top_values_by_field[2], [10.0, 20.0])

    def test_optimize_window_records_rescore_score_component_diagnostics(
        self,
    ) -> None:
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.01,
            root_rot_sigma=0.02,
            joint_sigma=0.03,
            candidate_score_component_diagnostics_top_k=1,
            candidate_rescore_diagnostics=True,
        )
        calls = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            calls.append(len(calls))
            if len(calls) == 1:
                return {
                    "score": np.array([0.0, 1.0, 4.0, 2.0], dtype=np.float32),
                    "contact_force_delta_mean": np.array(
                        [4.0, 3.0, 10.0, 6.0],
                        dtype=np.float32,
                    ),
                }
            return {
                "score": np.array([5.0, 1.0, 2.0, 3.0], dtype=np.float32),
                "contact_force_delta_mean": np.array(
                    [1.0, 3.0, 12.0, 5.0],
                    dtype=np.float32,
                ),
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

        self.assertEqual(len(calls), 2)
        self.assertEqual(
            tuple(float(value) for value in result.info["iteration_score_component_selected_values"][0]),
            (10.0,),
        )
        self.assertEqual(
            tuple(float(value) for value in result.info["iteration_rescore_score_component_selected_values"][0]),
            (12.0,),
        )
        self.assertEqual(
            tuple(int(value) for value in result.info["iteration_rescore_score_component_best_indices"]),
            (0,),
        )
        self.assertEqual(
            tuple(float(value) for value in result.info["iteration_rescore_score_component_best_values"][0]),
            (1.0,),
        )
        self.assertEqual(
            tuple(
                int(value)
                for value in result.info[
                    "iteration_rescore_score_component_delta_max_indices"
                ]
            ),
            (0,),
        )
        self.assertEqual(
            tuple(
                float(value)
                for value in result.info[
                    "iteration_rescore_score_component_delta_max_original_values"
                ][0]
            ),
            (4.0,),
        )
        self.assertEqual(
            tuple(
                float(value)
                for value in result.info[
                    "iteration_rescore_score_component_delta_max_rescore_values"
                ][0]
            ),
            (1.0,),
        )

    def test_optimize_window_omits_score_component_diagnostics_by_default(
        self,
    ) -> None:
        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return {
                "score": np.array([0.0, 2.0, 4.0, -1.0], dtype=np.float32),
                "contact_force_delta_mean": np.array(
                    [40.0, 20.0, 10.0, 90.0],
                    dtype=np.float32,
                ),
            }

        result = optimize_window(
            _config(),
            {"rollout_fn": rollout_fn},
            np.zeros((5, 8), dtype=np.float32),
            reference=None,
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=_FakeRuntime,
        )

        self.assertNotIn("candidate_score_component_diagnostics_top_k", result.info)
        self.assertNotIn("iteration_score_component_current_values", result.info)

    def test_optimize_window_can_select_from_topk_rescore(self) -> None:
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.0,
            root_rot_sigma=0.0,
            joint_sigma=0.0,
            candidate_rescore_selection_top_k=2,
        )
        calls = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            calls.append(np.asarray(samples).shape[0])
            if len(calls) == 1:
                return np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
            return np.array([1.0, 5.0], dtype=np.float32)

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

        self.assertEqual(calls, [4, 2])
        self.assertEqual(float(result.info["best_index"]), 2.0)
        self.assertEqual(float(result.info["best_score"]), 3.5)
        self.assertEqual(result.info["candidate_rescore_selection_top_k"], 2)
        self.assertEqual(
            tuple(
                int(value)
                for value in result.info[
                    "iteration_rescore_selection_best_indices"
                ]
            ),
            (2,),
        )
        self.assertEqual(
            tuple(
                bool(value)
                for value in result.info[
                    "iteration_rescore_selection_changed_flags"
                ]
            ),
            (True,),
        )
        self.assertEqual(
            tuple(
                float(value)
                for value in result.info[
                    "iteration_rescore_selection_score_delta_maxes"
                ]
            ),
            (3.0,),
        )

    def test_optimize_window_accepts_noop_when_best_sample_is_current_controls(
        self,
    ) -> None:
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

        self.assertTrue(result.info["accepted"])
        self.assertEqual(result.info["accepted_iterations"], 0)
        self.assertTrue(result.info["current_controls_selected"])
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

    def test_optimize_window_accepts_zero_delta_below_explicit_threshold_as_noop(
        self,
    ) -> None:
        config = JaxWindowOptimizerConfig(
            samples=3,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=0.0,
            root_rot_sigma=0.0,
            joint_sigma=0.0,
            min_score_improvement=0.01,
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

        self.assertTrue(result.info["accepted"])
        self.assertFalse(result.info["iteration_accepted"])
        self.assertTrue(result.info["zero_delta_noop_selected"])
        self.assertFalse(result.info["score_threshold_noop_selected"])
        self.assertTrue(result.info["noop_candidate_selected"])
        self.assertEqual(result.info["accepted_iterations"], 0)
        self.assertEqual(result.info["zero_delta_noop_iterations"], 1)
        self.assertEqual(result.info["score_threshold_noop_iterations"], 0)
        self.assertEqual(result.info["noop_candidate_iterations"], 1)
        self.assertEqual(float(result.info["control_delta_max"]), 0.0)
        np.testing.assert_allclose(result.updated_controls, controls)
        np.testing.assert_allclose(result.execute_chunk, controls[:3])

    def test_optimize_window_accepts_score_improvement_below_threshold_as_noop(
        self,
    ) -> None:
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
            min_score_improvement=0.01,
        )
        controls = np.full((5, 8), 0.25, dtype=np.float32)

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return np.array([0.0, 0.005, 0.001], dtype=np.float32)

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

        self.assertTrue(result.info["accepted"])
        self.assertFalse(result.info["iteration_accepted"])
        self.assertEqual(result.info["accepted_iterations"], 0)
        self.assertEqual(result.info["zero_delta_noop_iterations"], 0)
        self.assertEqual(result.info["score_threshold_noop_iterations"], 1)
        self.assertEqual(result.info["noop_candidate_iterations"], 1)
        self.assertFalse(result.info["current_controls_selected"])
        self.assertFalse(result.info["zero_delta_noop_selected"])
        self.assertTrue(result.info["score_threshold_noop_selected"])
        self.assertTrue(result.info["noop_candidate_selected"])
        self.assertEqual(float(result.info["best_index"]), 1.0)
        self.assertAlmostEqual(float(result.info["score_improvement"]), 0.005)
        self.assertEqual(float(result.info["min_score_improvement"]), 0.01)
        self.assertEqual(
            tuple(bool(v) for v in result.info["iteration_zero_delta_noop_flags"]),
            (False,),
        )
        self.assertEqual(
            tuple(bool(v) for v in result.info["iteration_score_threshold_noop_flags"]),
            (True,),
        )
        self.assertEqual(
            tuple(bool(v) for v in result.info["iteration_noop_candidate_flags"]),
            (True,),
        )
        np.testing.assert_allclose(result.updated_controls, controls)
        np.testing.assert_allclose(result.execute_chunk, controls[:3])

    def test_optimize_window_accepts_score_improvement_above_threshold(self) -> None:
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
            min_score_improvement=0.01,
        )
        controls = np.full((5, 8), 0.25, dtype=np.float32)

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return np.array([0.0, 0.02, 0.001], dtype=np.float32)

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

        self.assertTrue(result.info["accepted"])
        self.assertTrue(result.info["iteration_accepted"])
        self.assertEqual(result.info["accepted_iterations"], 1)
        self.assertEqual(result.info["zero_delta_noop_iterations"], 0)
        self.assertEqual(result.info["score_threshold_noop_iterations"], 0)
        self.assertEqual(result.info["noop_candidate_iterations"], 0)
        self.assertEqual(float(result.info["best_index"]), 1.0)
        self.assertGreater(float(result.info["control_delta_max"]), 0.0)
        self.assertAlmostEqual(float(result.info["score_improvement"]), 0.02)
        self.assertEqual(float(result.info["min_score_improvement"]), 0.01)
        self.assertFalse(result.info["zero_delta_noop_selected"])
        self.assertFalse(result.info["score_threshold_noop_selected"])
        self.assertFalse(result.info["noop_candidate_selected"])
        self.assertFalse(np.allclose(result.updated_controls, controls))

    def test_optimize_window_keeps_low_top_score_gap_candidate_as_noop(self) -> None:
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
            min_top_score_gap=0.01,
        )
        controls = np.full((5, 8), 0.25, dtype=np.float32)

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return np.array([0.0, 1.0, 0.995], dtype=np.float32)

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

        self.assertTrue(result.info["accepted"])
        self.assertFalse(result.info["iteration_accepted"])
        self.assertEqual(result.info["accepted_iterations"], 0)
        self.assertTrue(result.info["top_score_gap_noop_selected"])
        self.assertEqual(result.info["top_score_gap_noop_iterations"], 1)
        self.assertTrue(result.info["noop_candidate_selected"])
        self.assertEqual(result.info["noop_candidate_iterations"], 1)
        self.assertEqual(float(result.info["best_index"]), 1.0)
        self.assertAlmostEqual(float(result.info["top_score_gap"]), 0.005, places=6)
        self.assertEqual(float(result.info["min_top_score_gap"]), 0.01)
        self.assertEqual(
            tuple(bool(v) for v in result.info["iteration_top_score_gap_noop_flags"]),
            (True,),
        )
        np.testing.assert_allclose(result.updated_controls, controls)
        np.testing.assert_allclose(result.execute_chunk, controls[:3])

    def test_optimize_window_default_accepts_low_top_score_gap_candidate(self) -> None:
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
            return np.array([0.0, 1.0, 0.995], dtype=np.float32)

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

        self.assertTrue(result.info["iteration_accepted"])
        self.assertFalse(result.info["top_score_gap_noop_selected"])
        self.assertEqual(float(result.info["min_top_score_gap"]), 0.0)
        self.assertFalse(np.allclose(result.updated_controls, controls))

    def test_optimize_window_keeps_current_when_control_delta_guard_trips(
        self,
    ) -> None:
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
            max_control_delta=0.01,
        )
        controls = np.full((5, 8), 0.25, dtype=np.float32)

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return np.array([0.0, 1.0, 0.5], dtype=np.float32)

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

        self.assertTrue(result.info["accepted"])
        self.assertFalse(result.info["iteration_accepted"])
        self.assertEqual(result.info["accepted_iterations"], 0)
        self.assertTrue(result.info["control_delta_guard_noop_selected"])
        self.assertEqual(result.info["control_delta_guard_noop_iterations"], 1)
        self.assertTrue(result.info["noop_candidate_selected"])
        self.assertEqual(result.info["noop_candidate_iterations"], 1)
        self.assertGreater(float(result.info["control_delta_max"]), 0.01)
        self.assertEqual(float(result.info["max_control_delta"]), 0.01)
        self.assertEqual(
            tuple(
                bool(v)
                for v in result.info["iteration_control_delta_guard_noop_flags"]
            ),
            (True,),
        )
        np.testing.assert_allclose(result.updated_controls, controls)
        np.testing.assert_allclose(result.execute_chunk, controls[:3])

    def test_optimize_window_ignores_late_below_threshold_candidate(self) -> None:
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
            iterations=2,
            min_score_improvement=0.01,
        )
        controls = np.full((5, 8), 0.25, dtype=np.float32)
        calls: list[np.ndarray] = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            calls.append(np.asarray(samples, dtype=np.float32))
            if len(calls) == 1:
                return np.array([0.0, 0.02, 0.0], dtype=np.float32)
            return np.array([0.0, 0.005, 0.0], dtype=np.float32)

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

        self.assertTrue(result.info["accepted"])
        self.assertEqual(result.info["accepted_iterations"], 1)
        self.assertEqual(result.info["zero_delta_noop_iterations"], 0)
        self.assertEqual(result.info["score_threshold_noop_iterations"], 1)
        self.assertEqual(result.info["noop_candidate_iterations"], 1)
        self.assertEqual(
            tuple(bool(v) for v in result.info["iteration_accepted_flags"]),
            (True, False),
        )
        self.assertEqual(
            tuple(bool(v) for v in result.info["iteration_zero_delta_noop_flags"]),
            (False, False),
        )
        self.assertEqual(
            tuple(bool(v) for v in result.info["iteration_score_threshold_noop_flags"]),
            (False, True),
        )
        self.assertEqual(
            tuple(bool(v) for v in result.info["iteration_noop_candidate_flags"]),
            (False, True),
        )
        np.testing.assert_allclose(result.updated_controls, calls[0][1])

    def test_rejected_candidate_does_not_block_later_accepted_candidate(
        self,
    ) -> None:
        runtime = SimpleNamespace(
            jnp=_NumpyJnp(),
            jax=SimpleNamespace(random=_DistinctCandidateRandom(), nn=_FakeNN()),
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
            iterations=2,
            min_top_score_gap=0.5,
        )
        controls = np.full((5, 8), 0.25, dtype=np.float32)
        calls: list[np.ndarray] = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            calls.append(np.asarray(samples, dtype=np.float32))
            qpos = np.zeros((3, config.samples, 4), dtype=np.float32)
            qpos[:, :, 0] = 10.0 * len(calls) + np.arange(
                config.samples,
                dtype=np.float32,
            )
            if len(calls) == 1:
                scores = np.array([0.0, 10.0, 10.1], dtype=np.float32)
            else:
                scores = np.array([0.0, 5.0, 1.0], dtype=np.float32)
            return {"score": scores, "execute_trace": {"qpos": qpos}}

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

        self.assertEqual(
            tuple(bool(value) for value in result.info["iteration_accepted_flags"]),
            (False, True),
        )
        self.assertEqual(
            tuple(int(value) for value in result.info["iteration_best_indices"]),
            (2, 1),
        )
        self.assertEqual(result.info["accepted_iterations"], 1)
        np.testing.assert_allclose(result.updated_controls, calls[1][1])
        self.assertIsNotNone(result.execute_trace)
        np.testing.assert_allclose(result.execute_trace["qpos"][:, 0, 0], 21.0)

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
        current_anchors: list[np.ndarray] = []
        center_candidates: list[np.ndarray] = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            current_anchors.append(np.asarray(samples[0], dtype=np.float32).copy())
            center_candidates.append(np.asarray(samples[1], dtype=np.float32).copy())
            return np.asarray(samples, dtype=np.float32).mean(axis=(1, 2))

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
        self.assertEqual(len(current_anchors), 3)
        for anchor in current_anchors:
            np.testing.assert_allclose(anchor, 0.0)
        self.assertGreater(float(np.mean(center_candidates[1])), 0.6)
        self.assertGreater(float(np.mean(center_candidates[2])), 0.6)
        self.assertGreater(float(np.mean(result.updated_controls)), 1.5)
        np.testing.assert_allclose(result.updated_controls[0], 0.0)
        self.assertTrue(result.info["accepted"])
        self.assertEqual(result.info["accepted_iterations"], 3)

    def test_optimize_window_updates_sampling_center_from_elites(self) -> None:
        runtime = SimpleNamespace(
            jnp=_NumpyJnp(),
            jax=SimpleNamespace(random=_UnitStepRandom(), nn=_FakeNN()),
        )
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=1.0,
            root_rot_sigma=1.0,
            joint_sigma=1.0,
            elite_fraction=0.5,
            iterations=2,
        )
        current_anchors: list[np.ndarray] = []
        sampling_centers: list[np.ndarray] = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            current_anchors.append(np.asarray(samples[0], dtype=np.float32).copy())
            sampling_centers.append(np.asarray(samples[1], dtype=np.float32).copy())
            return np.array([0.0, 10.0, -10.0, -20.0], dtype=np.float32)

        optimize_window(
            config,
            {"rollout_fn": rollout_fn},
            np.zeros((5, 8), dtype=np.float32),
            reference={"unused": True},
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=runtime,
        )

        self.assertEqual(len(sampling_centers), 2)
        for anchor in current_anchors:
            np.testing.assert_allclose(anchor, 0.0)
        self.assertGreater(float(sampling_centers[0].mean()), 0.6)
        self.assertGreater(float(sampling_centers[1].mean()), 0.6)
        np.testing.assert_allclose(sampling_centers[1][0], 0.0)

    def test_optimize_window_can_guard_cem_center_update_on_low_top_gap(self) -> None:
        runtime = SimpleNamespace(
            jnp=_NumpyJnp(),
            jax=SimpleNamespace(random=_UnitStepRandom(), nn=_FakeNN()),
        )
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=1.0,
            root_rot_sigma=1.0,
            joint_sigma=1.0,
            elite_fraction=0.5,
            iterations=2,
            cem_update_min_top_score_gap=0.01,
        )
        sampling_centers: list[np.ndarray] = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            sampling_centers.append(np.asarray(samples[1], dtype=np.float32).copy())
            return np.array([0.0, 10.0, 9.995, -20.0], dtype=np.float32)

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

        self.assertEqual(len(sampling_centers), 2)
        self.assertGreater(float(sampling_centers[0].mean()), 0.6)
        np.testing.assert_allclose(sampling_centers[1], 0.0)
        self.assertEqual(float(result.info["cem_update_min_top_score_gap"]), 0.01)
        self.assertFalse(result.info["cem_update_allowed"])

    def test_optimize_window_adaptive_sigma_keeps_parameter_center_candidate(
        self,
    ) -> None:
        runtime = SimpleNamespace(
            jnp=_NumpyJnp(),
            jax=SimpleNamespace(random=_UnitStepRandom(), nn=_FakeNN()),
        )
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=1.0,
            root_rot_sigma=1.0,
            joint_sigma=1.0,
            elite_fraction=0.5,
            iterations=2,
            sigma_decay=0.75,
            min_root_pos_sigma=0.2,
            min_root_rot_sigma=0.2,
            min_joint_sigma=0.2,
        )
        captured: list[np.ndarray] = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            captured.append(np.asarray(samples, dtype=np.float32).copy())
            return np.array([0.0, 10.0, -10.0, -20.0], dtype=np.float32)

        optimize_window(
            config,
            {"rollout_fn": rollout_fn},
            np.zeros((5, 8), dtype=np.float32),
            reference={"unused": True},
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=runtime,
        )

        self.assertEqual(len(captured), 2)
        np.testing.assert_allclose(captured[0][0], 0.0)
        np.testing.assert_allclose(captured[1][0], 0.0)
        self.assertGreater(float(captured[1][1].mean()), 0.1)
        np.testing.assert_allclose(captured[1][1, 0], 0.0)

    def test_optimize_window_skips_final_distribution_update(self) -> None:
        nn = _CountingNN()
        runtime = SimpleNamespace(
            jnp=_NumpyJnp(),
            jax=SimpleNamespace(random=_UnitStepRandom(), nn=nn),
        )
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=1.0,
            root_rot_sigma=1.0,
            joint_sigma=1.0,
            elite_fraction=0.5,
            iterations=3,
            sigma_decay=0.75,
            min_root_pos_sigma=0.2,
            min_root_rot_sigma=0.2,
            min_joint_sigma=0.2,
        )
        rollout_calls = 0

        def rollout_fn(samples, reference, actor_params, model_bundle):
            nonlocal rollout_calls
            del samples, reference, actor_params, model_bundle
            rollout_calls += 1
            return np.array([0.0, 10.0, -10.0, -20.0], dtype=np.float32)

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

        self.assertEqual(rollout_calls, 3)
        self.assertEqual(nn.softmax_calls, 2)
        self.assertEqual(result.info["iteration"], 2)

    def test_optimize_window_records_iteration_diagnostics(self) -> None:
        runtime = SimpleNamespace(
            jnp=_NumpyJnp(),
            jax=SimpleNamespace(random=_UnitStepRandom(), nn=_FakeNN()),
        )
        config = JaxWindowOptimizerConfig(
            samples=4,
            horizon_steps=5,
            control_steps=2,
            knot_count=3,
            temperature=0.5,
            root_pos_sigma=1.0,
            root_rot_sigma=1.0,
            joint_sigma=1.0,
            elite_fraction=0.5,
            iterations=2,
            sigma_decay=0.75,
            min_root_pos_sigma=0.2,
            min_root_rot_sigma=0.2,
            min_joint_sigma=0.2,
        )

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del samples, reference, actor_params, model_bundle
            return np.array([0.0, 10.0, -10.0, -20.0], dtype=np.float32)

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

        self.assertEqual(tuple(int(v) for v in result.info["iteration_best_indices"]), (1, 1))
        self.assertEqual(
            tuple(bool(v) for v in result.info["iteration_accepted_flags"]),
            (True, True),
        )
        self.assertEqual(
            tuple(bool(v) for v in result.info["iteration_current_controls_selected_flags"]),
            (False, False),
        )
        self.assertEqual(len(result.info["iteration_score_improvements"]), 2)
        self.assertEqual(len(result.info["iteration_control_delta_maxes"]), 2)

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
            sample_means.append(float(np.asarray(samples[2], dtype=np.float32).mean()))
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
        np.testing.assert_allclose(sample_means, [0.8, 0.4, 0.2], rtol=1e-6)
        self.assertEqual(result.info["accepted_iterations"], 0)
        self.assertTrue(result.info["accepted"])
        self.assertTrue(result.info["current_controls_selected"])

    def test_optimize_window_uses_guided_controls_from_reference(self) -> None:
        random = _UnitStepRandom()
        runtime = SimpleNamespace(
            jnp=_NumpyJnp(),
            jax=SimpleNamespace(random=random, nn=_FakeNN()),
        )
        guided_controls = np.full((5, 8), 3.0, dtype=np.float32)
        captured_guided: list[np.ndarray] = []

        def rollout_fn(samples, reference, actor_params, model_bundle):
            del reference, actor_params, model_bundle
            captured_guided.append(np.asarray(samples[1], dtype=np.float32).copy())
            return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        optimize_window(
            _config(),
            {"rollout_fn": rollout_fn},
            np.zeros((5, 8), dtype=np.float32),
            reference={"guided_controls": guided_controls},
            actor_params=None,
            model_bundle=None,
            key=(0, 11),
            runtime=runtime,
        )

        self.assertEqual(random.calls, 1)
        np.testing.assert_allclose(captured_guided[0], guided_controls)

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

    def test_validate_scores_defers_jax_finite_check_without_bool_sync(self) -> None:
        status = _validate_scores(
            _JaxLikeScores(),
            _config(),
            jnp=_BoolTrapJnp(),
        )

        self.assertIsInstance(status, _BoolTrapScalar)

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

        with self.assertRaisesRegex(ValueError, "elite_fraction"):
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
                    elite_fraction=0.0,
                ),
                np.zeros((5, 8), dtype=np.float32),
                0,
                runtime=_FakeRuntime,
            )

        with self.assertRaisesRegex(ValueError, "candidate_rank_diagnostics_top_k"):
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
                    candidate_rank_diagnostics_top_k=5,
                ),
                np.zeros((5, 8), dtype=np.float32),
                0,
                runtime=_FakeRuntime,
            )

        with self.assertRaisesRegex(ValueError, "candidate_rescore_selection_top_k"):
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
                    candidate_rescore_selection_top_k=5,
                ),
                np.zeros((5, 8), dtype=np.float32),
                0,
                runtime=_FakeRuntime,
            )

        with self.assertRaisesRegex(
            ValueError,
            "candidate_score_component_diagnostics_top_k",
        ):
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
                    candidate_score_component_diagnostics_top_k=5,
                ),
                np.zeros((5, 8), dtype=np.float32),
                0,
                runtime=_FakeRuntime,
            )


if __name__ == "__main__":
    unittest.main()
