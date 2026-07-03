"""Guided-candidate controls for the G1 WBC MJX optimizer."""

from __future__ import annotations

from spider.tasks.g1_wbc.constants import QPOS_DIM


def guided_controls_from_trace(
    base_qpos,
    rollout_qpos,
    *,
    guided_root_pos_gain: float,
    guided_root_rot_gain: float,
    guided_joint_gain: float,
    guided_root_pos_clip: float,
    guided_root_rot_clip: float,
    guided_joint_clip: float,
    freeze_first_frame: bool = True,
    jnp,
):
    """Convert a no-MPC rollout trace into clipped residual controls."""

    base_qpos = jnp.asarray(base_qpos)
    rollout_qpos = jnp.asarray(rollout_qpos)
    _validate_shapes(base_qpos, rollout_qpos)
    horizon = int(base_qpos.shape[0])
    executed = rollout_qpos[1 : horizon + 1, 0]

    root_pos = (base_qpos[:, :3] - executed[:, :3]) * float(guided_root_pos_gain)
    quat_err = _quat_mul(base_qpos[:, 3:7], _quat_inv(executed[:, 3:7], jnp=jnp), jnp=jnp)
    root_rot = _axis_angle_from_quat(quat_err, jnp=jnp) * float(guided_root_rot_gain)
    joints = (base_qpos[:, 7:] - executed[:, 7:]) * float(guided_joint_gain)

    controls = jnp.concatenate(
        [
            jnp.clip(
                root_pos,
                -float(guided_root_pos_clip),
                float(guided_root_pos_clip),
            ),
            jnp.clip(
                root_rot,
                -float(guided_root_rot_clip),
                float(guided_root_rot_clip),
            ),
            jnp.clip(
                joints,
                -float(guided_joint_clip),
                float(guided_joint_clip),
            ),
        ],
        axis=-1,
    )
    if freeze_first_frame:
        controls = _set_first_zero(controls)
    return controls


def _validate_shapes(base_qpos, rollout_qpos) -> None:
    if len(base_qpos.shape) != 2 or int(base_qpos.shape[-1]) != QPOS_DIM:
        raise ValueError(
            f"Expected base_qpos shape (horizon, {QPOS_DIM}), got {base_qpos.shape}"
        )
    if len(rollout_qpos.shape) != 3 or int(rollout_qpos.shape[-1]) != QPOS_DIM:
        raise ValueError(
            "Expected rollout_qpos shape "
            f"(horizon + 1, samples, {QPOS_DIM}), got {rollout_qpos.shape}"
        )
    if int(rollout_qpos.shape[0]) < int(base_qpos.shape[0]) + 1:
        raise ValueError(
            "rollout_qpos must include the initial frame plus the full horizon"
        )
    if int(rollout_qpos.shape[1]) < 1:
        raise ValueError("rollout_qpos must include at least one sample")


def _set_first_zero(value):
    if hasattr(value, "at"):
        return value.at[0].set(0.0)
    out = value.copy()
    out[0] = 0.0
    return out


def _quat_mul(q1, q2, *, jnp):
    w1, x1, y1, z1 = (q1[..., i] for i in range(4))
    w2, x2, y2, z2 = (q2[..., i] for i in range(4))
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return jnp.concatenate(
        [
            jnp.expand_dims(w, axis=-1),
            jnp.expand_dims(x, axis=-1),
            jnp.expand_dims(y, axis=-1),
            jnp.expand_dims(z, axis=-1),
        ],
        axis=-1,
    )


def _quat_inv(quat, *, jnp):
    conjugate = jnp.concatenate([quat[..., 0:1], -quat[..., 1:]], axis=-1)
    return conjugate / jnp.sum(quat * quat, axis=-1, keepdims=True)


def _axis_angle_from_quat(quat, *, jnp):
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = jnp.sqrt(jnp.sum(quat[..., 1:] * quat[..., 1:], axis=-1))
    half_angle = jnp.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    safe_angle = jnp.where(jnp.abs(angle) > 1.0e-6, angle, 1.0)
    denom = jnp.where(
        jnp.abs(angle) > 1.0e-6,
        jnp.sin(half_angle) / safe_angle,
        0.5 - angle * angle / 48.0,
    )
    return quat[..., 1:4] / jnp.expand_dims(denom, axis=-1)


__all__ = ["guided_controls_from_trace"]
