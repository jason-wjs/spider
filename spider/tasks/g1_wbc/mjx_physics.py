"""Small MJX physics helpers used by real-runtime smoke tests."""

from __future__ import annotations

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    DECIMATION,
    MUJOCO_JOINT_NAMES,
    QPOS_DIM,
    QVEL_DIM,
)


def joint_order_to_model_ctrl(bundle, joint_ctrl, *, jnp):
    """Map G1 joint-order controls into MuJoCo actuator-order ctrl slots."""

    joint_ctrl = jnp.asarray(joint_ctrl)
    if int(joint_ctrl.shape[-1]) != ACTION_DIM:
        raise ValueError(
            f"Expected joint control width {ACTION_DIM}, got {joint_ctrl.shape}"
        )
    actuator_ids = _actuator_ids_for_joint_order(bundle)
    model_ctrl = jnp.zeros_like(joint_ctrl)
    if hasattr(model_ctrl, "at"):
        return model_ctrl.at[..., actuator_ids].set(joint_ctrl)
    model_ctrl = model_ctrl.copy()
    model_ctrl[..., actuator_ids] = joint_ctrl
    return model_ctrl


def reset_forward_step_smoke(
    bundle,
    qpos,
    qvel,
    joint_ctrl,
    *,
    runtime,
    steps: int = DECIMATION,
) -> dict[str, object]:
    """Reset an MJX data object, run forward and a short fixed step loop."""

    if getattr(bundle, "mjx_model", None) is None:
        raise ValueError("MJX model bundle must include mjx_model")
    jnp = runtime.jnp
    mjx = runtime.mjx
    qpos = jnp.asarray(qpos)
    qvel = jnp.asarray(qvel)
    if tuple(int(dim) for dim in qpos.shape) != (QPOS_DIM,):
        raise ValueError(f"Expected qpos shape {(QPOS_DIM,)}, got {qpos.shape}")
    if tuple(int(dim) for dim in qvel.shape) != (QVEL_DIM,):
        raise ValueError(f"Expected qvel shape {(QVEL_DIM,)}, got {qvel.shape}")

    ctrl = joint_order_to_model_ctrl(bundle, joint_ctrl, jnp=jnp)
    data = mjx.make_data(bundle.mjx_model)
    data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl, time=0.0)
    data = mjx.forward(bundle.mjx_model, data)
    data = _step_fixed_count(bundle.mjx_model, data, runtime=runtime, steps=steps)
    return {
        "qpos": runtime.jax.device_get(data.qpos),
        "qvel": runtime.jax.device_get(data.qvel),
        "ctrl": runtime.jax.device_get(data.ctrl),
        "time": runtime.jax.device_get(data.time),
    }


def _step_fixed_count(model, data, *, runtime, steps: int):
    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if steps == 0:
        return data

    def step_once(carry, _unused):
        return runtime.mjx.step(model, carry), None

    data, _ = runtime.jax.lax.scan(step_once, data, xs=None, length=steps)
    return data


def _actuator_ids_for_joint_order(bundle) -> tuple[int, ...]:
    name_to_id = getattr(bundle, "actuator_name_to_id", None)
    if name_to_id is None:
        raise ValueError("MJX model bundle must expose actuator_name_to_id")
    actuator_ids = tuple(_lookup_actuator_id(name_to_id, name) for name in MUJOCO_JOINT_NAMES)
    if len(set(actuator_ids)) != ACTION_DIM:
        raise ValueError("Actuator ids for G1 joint order must be unique")
    return actuator_ids


def _lookup_actuator_id(name_to_id: dict[str, int], joint_name: str) -> int:
    for candidate in (f"robot/{joint_name}", joint_name):
        if candidate in name_to_id:
            return int(name_to_id[candidate])
    raise ValueError(f"MJX model bundle is missing actuator for joint {joint_name!r}")


__all__ = [
    "joint_order_to_model_ctrl",
    "reset_forward_step_smoke",
]
