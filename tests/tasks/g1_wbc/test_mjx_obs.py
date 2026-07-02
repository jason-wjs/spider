import unittest

import numpy as np

from spider.tasks.g1_wbc.constants import ACTION_DIM, OBS_DIM, OBS_HISTORY_LENGTH
from spider.tasks.g1_wbc.mjx_obs import (
    OBS_FIELD_ORDER,
    build_wbc_observation,
    update_obs_history,
)


class _NumpyJnp:
    @staticmethod
    def asarray(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def ravel(value):
        return np.ravel(value)

    @staticmethod
    def concatenate(values, axis=0):
        return np.concatenate(values, axis=axis)

    @staticmethod
    def pad(value, pad_width):
        return np.pad(value, pad_width)

    @staticmethod
    def repeat(value, repeats, axis=0):
        return np.repeat(value, repeats, axis=axis)


def _synthetic_fields() -> dict[str, np.ndarray]:
    return {
        "projected_gravity": np.array([0.0, 0.0, -1.0], dtype=np.float32),
        "base_ang_vel": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "joint_pos_error": np.linspace(-0.2, 0.2, ACTION_DIM, dtype=np.float32),
        "joint_vel": np.linspace(-0.5, 0.5, ACTION_DIM, dtype=np.float32),
        "command": np.linspace(-1.0, 1.0, ACTION_DIM * 2, dtype=np.float32),
        "last_action": np.linspace(0.4, -0.4, ACTION_DIM, dtype=np.float32),
    }


class MjxObsTest(unittest.TestCase):
    def test_build_wbc_observation_pads_to_actor_dim_in_field_order(self) -> None:
        fields = _synthetic_fields()

        obs = build_wbc_observation(fields, jnp=_NumpyJnp)

        expected_prefix = np.concatenate(
            [
                np.ravel(np.asarray(fields[name], dtype=np.float32))
                for name in OBS_FIELD_ORDER
            ],
            axis=0,
        )
        self.assertEqual(obs.shape, (OBS_DIM,))
        np.testing.assert_allclose(obs[: expected_prefix.shape[0]], expected_prefix)
        np.testing.assert_allclose(obs[expected_prefix.shape[0] :], 0.0)

    def test_build_wbc_observation_rejects_missing_field(self) -> None:
        fields = _synthetic_fields()
        del fields["joint_vel"]

        with self.assertRaisesRegex(KeyError, "joint_vel"):
            build_wbc_observation(fields, jnp=_NumpyJnp)

    def test_build_wbc_observation_rejects_oversized_observation(self) -> None:
        fields = _synthetic_fields()
        fields["command"] = np.ones(OBS_DIM, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "expected <="):
            build_wbc_observation(fields, jnp=_NumpyJnp)

    def test_update_obs_history_backfills_first_frame(self) -> None:
        obs = np.linspace(-1.0, 1.0, OBS_DIM, dtype=np.float32)

        history = update_obs_history(None, obs, initialized=False, jnp=_NumpyJnp)

        self.assertEqual(history.shape[-2:], (OBS_HISTORY_LENGTH, OBS_DIM))
        np.testing.assert_allclose(
            history,
            np.repeat(obs[None, :], OBS_HISTORY_LENGTH, axis=0),
        )

    def test_update_obs_history_shifts_left_and_inserts_latest(self) -> None:
        first = np.linspace(-1.0, 1.0, OBS_DIM, dtype=np.float32)
        second = np.linspace(1.0, -1.0, OBS_DIM, dtype=np.float32)
        history = update_obs_history(None, first, initialized=False, jnp=_NumpyJnp)

        updated = update_obs_history(history, second, initialized=True, jnp=_NumpyJnp)

        self.assertEqual(updated.shape[-2:], (OBS_HISTORY_LENGTH, OBS_DIM))
        np.testing.assert_allclose(updated[:-1], history[1:])
        np.testing.assert_allclose(updated[-1], second)

    def test_update_obs_history_rejects_wrong_history_shape(self) -> None:
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        bad_history = np.zeros((OBS_HISTORY_LENGTH - 1, OBS_DIM), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "Expected history shape"):
            update_obs_history(bad_history, obs, initialized=True, jnp=_NumpyJnp)

    def test_update_obs_history_rejects_wrong_observation_dim(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected observation dim"):
            update_obs_history(
                None,
                np.zeros(OBS_DIM - 1, dtype=np.float32),
                initialized=False,
                jnp=_NumpyJnp,
            )


if __name__ == "__main__":
    unittest.main()
