import unittest

import numpy as np

from spider.tasks.g1_wbc.constants import ACTION_DIM, OBS_DIM, OBS_HISTORY_LENGTH
from spider.tasks.g1_wbc.mjx_obs import (
    LIMB_POSE_DIM,
    OBS_FIELD_ORDER,
    OBS_FIELD_SPECS,
    build_wbc_observation,
    update_obs_history,
)


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
    def where(condition, x, y):
        return np.where(condition, x, y)


def _values(shape: tuple[int, ...], start: int) -> np.ndarray:
    size = int(np.prod(shape))
    return np.arange(start, start + size, dtype=np.float32).reshape(shape)


def _synthetic_fields(batch_shape: tuple[int, ...] = ()) -> dict[str, np.ndarray]:
    fields: dict[str, np.ndarray] = {}
    offset = 0
    for name in OBS_FIELD_ORDER:
        shape = (*batch_shape, *OBS_FIELD_SPECS[name])
        fields[name] = _values(shape, offset)
        offset += int(np.prod(OBS_FIELD_SPECS[name]))
    return fields


def _flatten_reference(fields: dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for name in OBS_FIELD_ORDER:
        value = fields[name]
        width = int(np.prod(OBS_FIELD_SPECS[name]))
        parts.append(value.reshape((*value.shape[: -len(OBS_FIELD_SPECS[name])], width)))
    return np.concatenate(parts, axis=-1)


class MjxObsTest(unittest.TestCase):
    def test_field_specs_sum_to_actor_observation_dim(self) -> None:
        width = sum(int(np.prod(OBS_FIELD_SPECS[name])) for name in OBS_FIELD_ORDER)

        self.assertEqual(width, OBS_DIM)
        self.assertEqual(OBS_FIELD_ORDER[0], "command")
        self.assertEqual(OBS_FIELD_SPECS["command"], (ACTION_DIM * 2,))
        self.assertEqual(
            OBS_FIELD_SPECS["ref_limb_ee_pose_b"],
            (OBS_HISTORY_LENGTH, LIMB_POSE_DIM),
        )

    def test_build_wbc_observation_matches_existing_actor_slice_order(self) -> None:
        fields = _synthetic_fields()

        obs = build_wbc_observation(fields, jnp=_NumpyJnp)

        expected = _flatten_reference(fields)
        self.assertEqual(obs.shape, (OBS_DIM,))
        np.testing.assert_allclose(obs, expected)

        offset = 0
        for name in OBS_FIELD_ORDER:
            width = int(np.prod(OBS_FIELD_SPECS[name]))
            np.testing.assert_allclose(
                obs[offset : offset + width],
                fields[name].reshape(width),
                err_msg=f"bad slice for {name}",
            )
            offset += width

    def test_build_wbc_observation_preserves_batch_axis(self) -> None:
        fields = _synthetic_fields(batch_shape=(2,))

        obs = build_wbc_observation(fields, jnp=_NumpyJnp)

        self.assertEqual(obs.shape, (2, OBS_DIM))
        np.testing.assert_allclose(obs, _flatten_reference(fields))

    def test_build_wbc_observation_accepts_preflattened_history_fields(self) -> None:
        fields = _synthetic_fields()
        fields["ref_limb_ee_pose_b"] = fields["ref_limb_ee_pose_b"].reshape(
            OBS_HISTORY_LENGTH * LIMB_POSE_DIM
        )

        obs = build_wbc_observation(fields, jnp=_NumpyJnp)

        self.assertEqual(obs.shape, (OBS_DIM,))

    def test_build_wbc_observation_rejects_missing_field(self) -> None:
        fields = _synthetic_fields()
        del fields["joint_vel"]

        with self.assertRaisesRegex(KeyError, "joint_vel"):
            build_wbc_observation(fields, jnp=_NumpyJnp)

    def test_build_wbc_observation_rejects_wrong_tail_shape(self) -> None:
        fields = _synthetic_fields()
        fields["actions"] = np.ones((OBS_HISTORY_LENGTH, ACTION_DIM + 1), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "actions"):
            build_wbc_observation(fields, jnp=_NumpyJnp)

    def test_build_wbc_observation_rejects_mismatched_batch_shape(self) -> None:
        fields = _synthetic_fields(batch_shape=(2,))
        fields["motion_ref_ang_vel"] = np.zeros((3,), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "batch shape"):
            build_wbc_observation(fields, jnp=_NumpyJnp)

    def test_update_obs_history_backfills_first_frame(self) -> None:
        obs = np.linspace(-1.0, 1.0, 4, dtype=np.float32)

        history = update_obs_history(None, obs, initialized=False, jnp=_NumpyJnp)

        self.assertEqual(history.shape[-2:], (OBS_HISTORY_LENGTH, 4))
        np.testing.assert_allclose(
            history,
            np.repeat(obs[None, :], OBS_HISTORY_LENGTH, axis=0),
        )

    def test_update_obs_history_shifts_left_and_inserts_latest(self) -> None:
        obs = np.array([9.0, 10.0, 11.0], dtype=np.float32)
        history = np.arange(OBS_HISTORY_LENGTH * 3, dtype=np.float32).reshape(
            OBS_HISTORY_LENGTH,
            3,
        )

        updated = update_obs_history(history, obs, initialized=True, jnp=_NumpyJnp)

        self.assertEqual(updated.shape[-2:], (OBS_HISTORY_LENGTH, 3))
        np.testing.assert_allclose(updated[:-1], history[1:])
        np.testing.assert_allclose(updated[-1], obs)

    def test_update_obs_history_supports_batched_mixed_initialized_mask(self) -> None:
        obs = np.array([[1.0, 2.0], [10.0, 20.0]], dtype=np.float32)
        history = np.arange(2 * OBS_HISTORY_LENGTH * 2, dtype=np.float32).reshape(
            2,
            OBS_HISTORY_LENGTH,
            2,
        )
        initialized = np.array([True, False])

        updated = update_obs_history(history, obs, initialized=initialized, jnp=_NumpyJnp)

        expected_first = np.concatenate([history[0, 1:], obs[0][None, :]], axis=0)
        expected_second = np.repeat(obs[1][None, :], OBS_HISTORY_LENGTH, axis=0)
        np.testing.assert_allclose(updated[0], expected_first)
        np.testing.assert_allclose(updated[1], expected_second)

    def test_update_obs_history_rejects_wrong_history_shape(self) -> None:
        obs = np.zeros(3, dtype=np.float32)
        bad_history = np.zeros((OBS_HISTORY_LENGTH - 1, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "Expected history shape"):
            update_obs_history(bad_history, obs, initialized=True, jnp=_NumpyJnp)


if __name__ == "__main__":
    unittest.main()
