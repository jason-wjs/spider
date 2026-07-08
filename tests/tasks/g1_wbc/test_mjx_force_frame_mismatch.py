import numpy as np

from scripts.diagnose_g1_wbc_mjx_force_frame_mismatch import (
    FORCE_HIGH_STATE_DIVERGED,
    FORCE_HIGH_STATE_NEAR,
    analyze_force_frame_mismatch,
)


def test_force_frame_mismatch_classifies_high_force_with_near_state(tmp_path):
    mjx = tmp_path / "mjx.npz"
    replay = tmp_path / "replay.npz"
    qpos = np.zeros((5, 1, 3), dtype=np.float32)
    qvel = np.zeros((5, 1, 3), dtype=np.float32)
    contact = np.ones((5, 1, 2), dtype=np.float32)
    mjx_force = np.array(
        [[[0.0, 0.0]], [[120.0, 10.0]], [[420.0, 20.0]], [[80.0, 0.0]], [[0.0, 0.0]]],
        dtype=np.float32,
    )
    replay_force = np.array(
        [[[0.0, 0.0]], [[100.0, 10.0]], [[100.0, 20.0]], [[90.0, 0.0]], [[0.0, 0.0]]],
        dtype=np.float32,
    )
    np.savez(
        mjx,
        qpos=qpos,
        qvel=qvel,
        contact_indicator=contact,
        contact_force=mjx_force,
    )
    np.savez(
        replay,
        qpos=qpos.copy(),
        qvel=qvel.copy(),
        contact_indicator=contact.copy(),
        contact_force=replay_force,
    )

    report = analyze_force_frame_mismatch(mjx, replay)

    left = report["feet"]["left"]
    assert left["frame"] == 2
    assert left["classification"] == FORCE_HIGH_STATE_NEAR
    assert left["mjx_force"] == 420.0
    assert left["replay_force"] == 100.0
    assert left["force_ratio"] == 4.2
    assert left["qpos_norm"] == 0.0
    assert left["same_contact"]


def test_force_frame_mismatch_distinguishes_state_divergence(tmp_path):
    mjx = tmp_path / "mjx.npz"
    replay = tmp_path / "replay.npz"
    mjx_qpos = np.zeros((4, 1, 3), dtype=np.float32)
    replay_qpos = mjx_qpos.copy()
    replay_qpos[2, 0, 0] = 0.25
    qvel = np.zeros((4, 1, 3), dtype=np.float32)
    contact = np.ones((4, 1, 2), dtype=np.float32)
    mjx_force = np.array(
        [[[0.0, 0.0]], [[100.0, 0.0]], [[500.0, 0.0]], [[0.0, 0.0]]],
        dtype=np.float32,
    )
    replay_force = np.array(
        [[[0.0, 0.0]], [[100.0, 0.0]], [[100.0, 0.0]], [[0.0, 0.0]]],
        dtype=np.float32,
    )
    np.savez(
        mjx,
        qpos=mjx_qpos,
        qvel=qvel,
        contact_indicator=contact,
        contact_force=mjx_force,
    )
    np.savez(
        replay,
        qpos=replay_qpos,
        qvel=qvel.copy(),
        contact_indicator=contact.copy(),
        contact_force=replay_force,
    )

    report = analyze_force_frame_mismatch(mjx, replay, qpos_near_threshold=0.05)

    assert report["feet"]["left"]["classification"] == FORCE_HIGH_STATE_DIVERGED
    assert report["feet"]["left"]["qpos_norm"] == 0.25
