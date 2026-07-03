import unittest

import numpy as np

from spider.tasks.g1_wbc.mjx_scoring import (
    ACCUMULATOR_KEYS,
    JaxScoreWeights,
    finalize_score,
    init_score_accumulator,
    score_step,
)


class _NumpyJnp:
    @staticmethod
    def asarray(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def zeros(shape):
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def mean(value, axis=None):
        return np.mean(value, axis=axis)

    @staticmethod
    def abs(value):
        return np.abs(value)

    @staticmethod
    def maximum(x, y):
        return np.maximum(x, y)

    @staticmethod
    def max(value, axis=None):
        return np.max(value, axis=axis)

    @staticmethod
    def sum(value, axis=None, keepdims=False):
        return np.sum(value, axis=axis, keepdims=keepdims)

    @staticmethod
    def sqrt(value):
        return np.sqrt(value)

    @staticmethod
    def clip(value, low, high):
        return np.clip(value, low, high)

    @staticmethod
    def arccos(value):
        return np.arccos(value)


def _step_state(offset: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "root_pos": np.array([1.0 + offset, 2.0, 3.0], dtype=np.float32),
        "body_pos": np.array(
            [[0.0, 1.0 + offset, 2.0], [3.0, 4.0, 5.0]],
            dtype=np.float32,
        ),
        "ee_pos": np.array([[1.0, -1.0 + offset, 0.5]], dtype=np.float32),
        "contact": np.array([1.0, 0.0], dtype=np.float32),
        "control": np.array([0.2, 0.4, 0.6], dtype=np.float32),
        "prev_control": np.array([0.1, 0.5, 0.3], dtype=np.float32),
        "joint_vel": np.array([-0.2, 0.1, 0.3], dtype=np.float32),
        "prev_joint_vel": np.array([-0.1, 0.0, 0.5], dtype=np.float32),
    }


def _reference_state() -> dict[str, np.ndarray]:
    return {
        "root_pos": np.array([1.5, 1.0, 3.0], dtype=np.float32),
        "body_pos": np.array(
            [[0.0, 2.0, 2.0], [2.0, 4.0, 7.0]],
            dtype=np.float32,
        ),
        "ee_pos": np.array([[0.5, -1.0, 1.5]], dtype=np.float32),
        "contact": np.array([0.0, 0.0], dtype=np.float32),
    }


def _expected_terms(step_state, reference_state) -> dict[str, float]:
    root_error = np.linalg.norm(step_state["root_pos"] - reference_state["root_pos"])
    body_error = np.mean(
        np.linalg.norm(step_state["body_pos"] - reference_state["body_pos"], axis=-1)
    )
    ee_error = np.mean(
        np.linalg.norm(step_state["ee_pos"] - reference_state["ee_pos"], axis=-1)
    )
    contact_error = np.mean(np.abs(step_state["contact"] - reference_state["contact"]))
    control_delta = np.mean((step_state["control"] - step_state["prev_control"]) ** 2)
    joint_acc = np.mean((step_state["joint_vel"] - step_state["prev_joint_vel"]) ** 2)
    return {
        "root_pos_error_mean": float(root_error),
        "body_global_pos_error_mean": float(body_error),
        "ee_global_pos_error_mean": float(ee_error),
        "contact_mismatch_rate": float(contact_error),
        "control_delta_mean": float(control_delta),
        "joint_acc_mean": float(joint_acc),
    }


class MjxScoringTest(unittest.TestCase):
    def test_score_step_matches_numpy_expected_terms(self) -> None:
        step_state = _step_state()
        reference_state = _reference_state()
        weights = JaxScoreWeights(
            {
                "root_pos": 1.5,
                "body_global_pos": 4.0,
                "ee_global_pos": 3.0,
                "contact": 2.0,
                "control_delta": 0.5,
                "joint_acc": 0.25,
            }
        )

        accumulator = init_score_accumulator((), jnp=_NumpyJnp)
        accumulator = score_step(
            accumulator,
            step_state,
            reference_state,
            weights,
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        expected = _expected_terms(step_state, reference_state)
        for name, value in expected.items():
            self.assertAlmostEqual(float(metrics[name]), value, places=6)

        expected_penalty = (
            1.5 * expected["root_pos_error_mean"]
            + 4.0 * expected["body_global_pos_error_mean"]
            + 3.0 * expected["ee_global_pos_error_mean"]
            + 2.0 * expected["contact_mismatch_rate"]
            + 0.5 * expected["control_delta_mean"]
            + 0.25 * expected["joint_acc_mean"]
        )
        self.assertAlmostEqual(float(metrics["score"]), -expected_penalty, places=6)

    def test_score_step_accumulates_rotation_error_terms(self) -> None:
        identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        half_turn_x = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        step_state = {
            **_step_state(),
            "root_quat": half_turn_x,
            "body_quat": np.stack([half_turn_x, identity], axis=0),
            "ee_quat": np.stack([half_turn_x], axis=0),
        }
        reference_state = {
            **_reference_state(),
            "root_quat": identity,
            "body_quat": np.stack([identity, identity], axis=0),
            "ee_quat": np.stack([identity], axis=0),
        }
        weights = JaxScoreWeights(
            {
                "root_rot": 0.5,
                "body_global_rot": 0.8,
                "ee_global_rot": 0.3,
            }
        )

        accumulator = score_step(
            init_score_accumulator((), jnp=_NumpyJnp),
            step_state,
            reference_state,
            weights,
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        self.assertAlmostEqual(
            float(metrics["root_rot_error_mean"]),
            np.pi,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["body_global_rot_error_mean"]),
            np.pi / 2.0,
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["ee_global_rot_error_mean"]),
            np.pi,
            places=6,
        )
        expected_penalty = 0.5 * np.pi + 0.8 * (np.pi / 2.0) + 0.3 * np.pi
        self.assertAlmostEqual(float(metrics["score"]), -expected_penalty, places=6)

    def test_score_step_accumulates_contact_classification_and_action_delta(
        self,
    ) -> None:
        step_state = {
            **_step_state(),
            "contact": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "action": np.array([0.2, -0.4], dtype=np.float32),
            "prev_action": np.array([-0.1, 0.1], dtype=np.float32),
        }
        reference_state = {
            **_reference_state(),
            "contact": np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
        }
        weights = JaxScoreWeights(
            {
                "contact": 2.0,
                "contact_false_positive": 3.0,
                "contact_false_negative": 5.0,
                "action_delta": 7.0,
            }
        )

        accumulator = score_step(
            init_score_accumulator((), jnp=_NumpyJnp),
            step_state,
            reference_state,
            weights,
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        self.assertAlmostEqual(float(metrics["contact_mismatch_rate"]), 0.5)
        self.assertAlmostEqual(float(metrics["contact_false_positive"]), 0.25)
        self.assertAlmostEqual(float(metrics["contact_false_negative"]), 0.25)
        expected_action_delta = float(
            np.linalg.norm(step_state["action"] - step_state["prev_action"])
        )
        self.assertAlmostEqual(float(metrics["action_delta_mean"]), expected_action_delta)
        expected_penalty = (
            2.0 * 0.5
            + 3.0 * 0.25
            + 5.0 * 0.25
            + 7.0 * expected_action_delta
        )
        self.assertAlmostEqual(float(metrics["score"]), -expected_penalty, places=6)

    def test_finalize_score_returns_compute_rollout_scores_term_aliases(self) -> None:
        step_state = _step_state()
        reference_state = _reference_state()
        accumulator = init_score_accumulator((), jnp=_NumpyJnp)
        accumulator = score_step(
            accumulator,
            step_state,
            reference_state,
            JaxScoreWeights({"root_pos_error": 1.0}),
            jnp=_NumpyJnp,
        )

        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        self.assertEqual(metrics["root_pos_error"], metrics["root_pos_error_mean"])
        self.assertEqual(
            metrics["body_global_pos_error"],
            metrics["body_global_pos_error_mean"],
        )
        self.assertEqual(
            metrics["ee_global_pos_error"],
            metrics["ee_global_pos_error_mean"],
        )
        self.assertEqual(metrics["contact_mismatch"], metrics["contact_mismatch_rate"])
        self.assertEqual(metrics["control_delta"], metrics["control_delta_mean"])
        self.assertEqual(metrics["joint_acc"], metrics["joint_acc_mean"])

    def test_accumulator_averages_multiple_steps(self) -> None:
        reference_state = _reference_state()
        first = _step_state(offset=0.0)
        second = _step_state(offset=0.3)
        accumulator = init_score_accumulator((), jnp=_NumpyJnp)

        accumulator = score_step(
            accumulator,
            first,
            reference_state,
            JaxScoreWeights({"root_pos": 1.0}),
            jnp=_NumpyJnp,
        )
        accumulator = score_step(
            accumulator,
            second,
            reference_state,
            JaxScoreWeights({"root_pos": 1.0}),
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        expected_root = (
            _expected_terms(first, reference_state)["root_pos_error_mean"]
            + _expected_terms(second, reference_state)["root_pos_error_mean"]
        ) / 2.0
        self.assertAlmostEqual(
            float(metrics["root_pos_error_mean"]),
            expected_root,
            places=6,
        )
        self.assertAlmostEqual(float(metrics["score"]), -expected_root, places=6)

    def test_accumulator_tracks_contact_diagnostic_maxima(self) -> None:
        reference_state = _reference_state()
        accumulator = init_score_accumulator((2,), jnp=_NumpyJnp)
        base_step = {
            name: np.stack([value, value], axis=0)
            for name, value in _step_state().items()
        }
        batched_reference = {
            name: np.stack([value, value], axis=0)
            for name, value in reference_state.items()
        }
        first = {
            **base_step,
            "active_contact_count": np.array([1.0, 4.0], dtype=np.float32),
            "contact_pair_count": np.array([3.0, 2.0], dtype=np.float32),
        }
        second = {
            **base_step,
            "active_contact_count": np.array([5.0, 2.0], dtype=np.float32),
            "contact_pair_count": np.array([4.0, 8.0], dtype=np.float32),
        }

        accumulator = score_step(
            accumulator,
            first,
            batched_reference,
            JaxScoreWeights({"root_pos": 1.0}),
            jnp=_NumpyJnp,
        )
        accumulator = score_step(
            accumulator,
            second,
            batched_reference,
            JaxScoreWeights({"root_pos": 1.0}),
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        np.testing.assert_allclose(metrics["active_contact_count"], [5.0, 4.0])
        np.testing.assert_allclose(metrics["contact_pair_count"], [4.0, 8.0])

    def test_batched_score_keeps_sample_axis(self) -> None:
        first = _step_state(offset=0.0)
        second = _step_state(offset=2.0)
        batched_step = {
            name: np.stack([first[name], second[name]], axis=0)
            for name in first
        }
        ref = _reference_state()
        batched_ref = {
            name: np.stack([ref[name], ref[name]], axis=0)
            for name in ref
        }
        accumulator = init_score_accumulator((2,), jnp=_NumpyJnp)

        accumulator = score_step(
            accumulator,
            batched_step,
            batched_ref,
            JaxScoreWeights({"root_pos": 1.0}),
            jnp=_NumpyJnp,
        )
        metrics = finalize_score(accumulator, jnp=_NumpyJnp)

        self.assertEqual(metrics["score"].shape, (2,))
        expected = np.array(
            [
                _expected_terms(first, ref)["root_pos_error_mean"],
                _expected_terms(second, ref)["root_pos_error_mean"],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(metrics["root_pos_error_mean"], expected)
        np.testing.assert_allclose(metrics["score"], -expected)

    def test_score_accumulator_has_stable_keys(self) -> None:
        accumulator = init_score_accumulator((2,), jnp=_NumpyJnp)

        self.assertEqual(tuple(accumulator), ACCUMULATOR_KEYS)
        for value in accumulator.values():
            self.assertEqual(value.shape, (2,))

    def test_score_step_rejects_missing_accumulator_key(self) -> None:
        accumulator = init_score_accumulator((), jnp=_NumpyJnp)
        del accumulator["joint_acc_sum"]

        with self.assertRaisesRegex(KeyError, "joint_acc_sum"):
            score_step(
                accumulator,
                _step_state(),
                _reference_state(),
                JaxScoreWeights({"root_pos": 1.0}),
                jnp=_NumpyJnp,
            )

    def test_larger_errors_lower_total_score(self) -> None:
        reference_state = _reference_state()
        weights = JaxScoreWeights(
            {
                "root_pos": 1.0,
                "body_global_pos": 1.0,
                "ee_global_pos": 1.0,
                "contact": 1.0,
                "control_delta": 1.0,
                "joint_acc": 1.0,
            }
        )

        small = finalize_score(
            score_step(
                init_score_accumulator((), jnp=_NumpyJnp),
                _step_state(offset=0.0),
                reference_state,
                weights,
                jnp=_NumpyJnp,
            ),
            jnp=_NumpyJnp,
        )
        large = finalize_score(
            score_step(
                init_score_accumulator((), jnp=_NumpyJnp),
                _step_state(offset=2.0),
                reference_state,
                weights,
                jnp=_NumpyJnp,
            ),
            jnp=_NumpyJnp,
        )

        self.assertLess(float(large["score"]), float(small["score"]))

    def test_score_step_rejects_empty_accumulator(self) -> None:
        with self.assertRaisesRegex(KeyError, "init_score_accumulator"):
            score_step(
                {},
                _step_state(),
                _reference_state(),
                JaxScoreWeights({"root_pos": 1.0}),
                jnp=_NumpyJnp,
            )

    def test_finalize_score_handles_empty_accumulator(self) -> None:
        metrics = finalize_score({}, jnp=_NumpyJnp)

        self.assertEqual(float(metrics["score"]), 0.0)
        self.assertEqual(float(metrics["root_pos_error_mean"]), 0.0)

    def test_score_weights_are_static_jax_pytree_node(self) -> None:
        try:
            import jax
        except Exception as exc:
            self.skipTest(f"JAX is not available: {exc}")

        weights = JaxScoreWeights({"root_pos": 1.0, "control_delta": 0.5})

        leaves, treedef = jax.tree_util.tree_flatten(weights)

        self.assertEqual(leaves, [])
        self.assertIn("JaxScoreWeights", str(treedef))


if __name__ == "__main__":
    unittest.main()
