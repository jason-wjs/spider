"""Small MJX physics helpers used by real-runtime smoke tests."""

from __future__ import annotations

import re

from dataclasses import dataclass

from spider.tasks.g1_wbc.constants import (
    ACTION_DIM,
    ACTUATOR_GROUPS,
    COMMAND_BODY_NAMES,
    DECIMATION,
    MUJOCO_BODY_NAMES,
    MUJOCO_JOINT_NAMES,
    QPOS_DIM,
    QVEL_DIM,
    TASK_EE_BODY_NAMES,
)


@dataclass(frozen=True)
class FootContactGeomGroups:
    floor_geom_ids: tuple[int, ...]
    left_foot_geom_ids: tuple[int, ...]
    right_foot_geom_ids: tuple[int, ...]


def default_action_scale(*, jnp):
    """Return WXY raw-action scale factors in MuJoCo joint order."""

    return jnp.asarray(_action_scale_values())


def foot_contact_geom_groups(bundle) -> FootContactGeomGroups:
    """Resolve floor and left/right foot collision geom ids from a model bundle."""

    profile = getattr(bundle, "profile", None)
    geom_name_to_id = getattr(bundle, "geom_name_to_id", None)
    if profile is None or geom_name_to_id is None:
        raise ValueError("MJX model bundle must expose profile and geom_name_to_id")
    floor_geom_ids = tuple(
        _lookup_geom_id(geom_name_to_id, name)
        for name in getattr(profile, "floor_geom_names", ())
    )
    left_names = tuple(
        name
        for name in getattr(profile, "foot_collision_geom_names", ())
        if "left_" in name
    )
    right_names = tuple(
        name
        for name in getattr(profile, "foot_collision_geom_names", ())
        if "right_" in name
    )
    if not floor_geom_ids or not left_names or not right_names:
        raise ValueError(
            "Contact profile must define floor, left foot, and right foot geoms"
        )
    return FootContactGeomGroups(
        floor_geom_ids=floor_geom_ids,
        left_foot_geom_ids=tuple(
            _lookup_geom_id(geom_name_to_id, name) for name in left_names
        ),
        right_foot_geom_ids=tuple(
            _lookup_geom_id(geom_name_to_id, name) for name in right_names
        ),
    )


def foot_contact_indicator_from_contact(
    contact,
    *,
    floor_geom_ids: tuple[int, ...],
    left_foot_geom_ids: tuple[int, ...],
    right_foot_geom_ids: tuple[int, ...],
    jnp,
):
    """Return left/right foot-floor contact indicators from fixed-shape MJX contact."""

    geom = jnp.asarray(contact.geom)
    dist = jnp.asarray(contact.dist)
    includemargin = jnp.asarray(contact.includemargin)
    if len(geom.shape) != 2 or int(geom.shape[-1]) != 2:
        raise ValueError(f"Expected contact.geom shape (contacts, 2), got {geom.shape}")
    valid = (geom[:, 0] >= 0) & (geom[:, 1] >= 0)
    active = valid & (dist <= includemargin + 1.0e-5)
    has_floor = _contact_has_any_geom(geom, floor_geom_ids, jnp=jnp)
    left = _contact_has_any_geom(geom, left_foot_geom_ids, jnp=jnp)
    right = _contact_has_any_geom(geom, right_foot_geom_ids, jnp=jnp)
    return jnp.asarray(
        [
            _as_indicator(jnp.any(active & has_floor & left), jnp=jnp),
            _as_indicator(jnp.any(active & has_floor & right), jnp=jnp),
        ]
    )


def contact_count_diagnostics(contact, *, jnp) -> dict[str, object]:
    """Return fixed-buffer contact counts without synchronizing to host."""

    geom = jnp.asarray(contact.geom)
    dist = jnp.asarray(contact.dist)
    includemargin = jnp.asarray(contact.includemargin)
    if len(geom.shape) != 2 or int(geom.shape[-1]) != 2:
        raise ValueError(f"Expected contact.geom shape (contacts, 2), got {geom.shape}")
    valid = (geom[:, 0] >= 0) & (geom[:, 1] >= 0)
    active = valid & (dist <= includemargin + 1.0e-5)
    return {
        "active_contact_count": jnp.sum(active),
        "contact_pair_count": jnp.sum(valid),
    }


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


def action_to_model_ctrl(
    bundle,
    action,
    default_joint_pos,
    action_scale,
    *,
    jnp,
):
    """Map raw WBC actor actions to MuJoCo actuator-order ctrl slots."""

    action = jnp.asarray(action)
    if int(action.shape[-1]) != ACTION_DIM:
        raise ValueError(f"Expected action width {ACTION_DIM}, got {action.shape}")
    default_joint_pos = _jnp_vector(
        "default_joint_pos",
        default_joint_pos,
        ACTION_DIM,
        jnp=jnp,
    )
    action_scale = _jnp_vector("action_scale", action_scale, ACTION_DIM, jnp=jnp)
    joint_ctrl = action * action_scale + default_joint_pos
    return joint_order_to_model_ctrl(bundle, joint_ctrl, jnp=jnp)


def make_mjx_physics_step_fn(
    *,
    default_joint_pos,
    action_scale,
    score_body_names: tuple[str, ...] = MUJOCO_BODY_NAMES,
    ee_body_names: tuple[str, ...] = TASK_EE_BODY_NAMES,
    decimation: int = DECIMATION,
):
    """Create a batched MJX physics step function for rollout scoring."""

    default_joint_pos = _float_tuple(
        "default_joint_pos",
        default_joint_pos,
        ACTION_DIM,
    )
    action_scale = _float_tuple("action_scale", action_scale, ACTION_DIM)
    score_body_names = tuple(score_body_names)
    ee_body_names = tuple(ee_body_names)
    decimation = int(decimation)
    if decimation < 0:
        raise ValueError("decimation must be non-negative")

    def physics_step_fn(
        bundle,
        robot_state,
        command_qpos,
        action,
        step_index,
        *,
        runtime,
    ):
        del step_index
        if getattr(bundle, "mjx_model", None) is None:
            raise ValueError("MJX model bundle must include mjx_model")
        jnp = runtime.jnp
        qpos = _batched_vector(
            "robot_state['qpos']",
            robot_state["qpos"],
            QPOS_DIM,
            jnp=jnp,
        )
        command_qpos = _batched_vector("command_qpos", command_qpos, QPOS_DIM, jnp=jnp)
        qvel = _batched_vector(
            "robot_state['qvel']",
            robot_state["qvel"],
            QVEL_DIM,
            jnp=jnp,
        )
        action_array = _batched_vector("action", action, ACTION_DIM, jnp=jnp)
        sample_count = int(qpos.shape[0])
        if int(qvel.shape[0]) != sample_count:
            raise ValueError(
                f"Expected qvel batch {sample_count}, got {int(qvel.shape[0])}"
            )
        if int(action_array.shape[0]) != sample_count:
            raise ValueError(
                f"Expected action batch {sample_count}, got {int(action_array.shape[0])}"
            )
        if int(command_qpos.shape[0]) != sample_count:
            raise ValueError(
                f"Expected command_qpos batch {sample_count}, "
                f"got {int(command_qpos.shape[0])}"
            )

        body_ids = _body_ids_for_names(bundle, MUJOCO_BODY_NAMES)
        score_body_ids = _body_ids_for_names(bundle, score_body_names)
        ee_body_ids = _body_ids_for_names(bundle, ee_body_names)
        body_id_array = jnp.asarray(body_ids)
        score_body_id_array = jnp.asarray(score_body_ids)
        ee_body_id_array = jnp.asarray(ee_body_ids)
        root_body_id = _lookup_body_id(bundle, MUJOCO_BODY_NAMES[0])
        contact_groups = foot_contact_geom_groups(bundle)

        def step_one(qpos_one, qvel_one, action_one):
            model_ctrl = action_to_model_ctrl(
                bundle,
                action_one,
                default_joint_pos,
                action_scale,
                jnp=jnp,
            )
            data = runtime.mjx.make_data(bundle.mjx_model)
            data = data.replace(qpos=qpos_one, qvel=qvel_one, ctrl=model_ctrl, time=0.0)
            data = runtime.mjx.forward(bundle.mjx_model, data)
            data = _step_fixed_count(
                bundle.mjx_model,
                data,
                runtime=runtime,
                steps=decimation,
            )
            body_pos_w = jnp.take(data.xpos, body_id_array, axis=0)
            body_quat_w = jnp.take(data.xquat, body_id_array, axis=0)
            body_cvel = jnp.take(data.cvel, body_id_array, axis=0)
            body_ang_vel_w = body_cvel[..., 0:3]
            root_subtree_com = data.subtree_com[root_body_id]
            body_lin_vel_w = body_cvel[..., 3:6] - jnp.cross(
                body_ang_vel_w,
                root_subtree_com - body_pos_w,
            )
            next_robot = {
                "qpos": data.qpos,
                "qvel": data.qvel,
                "body_pos_w": body_pos_w,
                "body_quat_w": body_quat_w,
                "body_lin_vel_w": body_lin_vel_w,
                "body_ang_vel_w": body_ang_vel_w,
            }
            score_state = {
                "root_pos": data.qpos[:3],
                "body_pos": jnp.take(data.xpos, score_body_id_array, axis=0),
                "ee_pos": jnp.take(data.xpos, ee_body_id_array, axis=0),
                "contact": foot_contact_indicator_from_contact(
                    data._impl.contact,
                    floor_geom_ids=contact_groups.floor_geom_ids,
                    left_foot_geom_ids=contact_groups.left_foot_geom_ids,
                    right_foot_geom_ids=contact_groups.right_foot_geom_ids,
                    jnp=jnp,
                ),
                **contact_count_diagnostics(data._impl.contact, jnp=jnp),
                "model_ctrl": data.ctrl,
                "time": data.time,
            }
            return next_robot, score_state

        return runtime.jax.vmap(step_one)(qpos, qvel, action_array)

    return physics_step_fn


def make_mjx_command_reference_fn(
    *,
    command_body_names: tuple[str, ...] = COMMAND_BODY_NAMES,
):
    """Create a batched MJX forward-kinematics function for actor command refs."""

    command_body_names = tuple(command_body_names)

    def command_reference_fn(bundle, command_qpos, command_qvel, *, runtime):
        if getattr(bundle, "mjx_model", None) is None:
            raise ValueError("MJX model bundle must include mjx_model")
        jnp = runtime.jnp
        command_qpos = _batched_trajectory(
            "command_qpos",
            command_qpos,
            QPOS_DIM,
            jnp=jnp,
        )
        command_qvel = _batched_trajectory(
            "command_qvel",
            command_qvel,
            QVEL_DIM,
            jnp=jnp,
        )
        sample_count = int(command_qpos.shape[0])
        horizon = int(command_qpos.shape[1])
        if tuple(int(dim) for dim in command_qvel.shape[:2]) != (
            sample_count,
            horizon,
        ):
            raise ValueError(
                "Expected command_qvel batch/horizon "
                f"{(sample_count, horizon)}, got {command_qvel.shape[:2]}"
            )

        body_ids = _body_ids_for_names(bundle, command_body_names)
        body_id_array = jnp.asarray(body_ids)
        root_body_id = _lookup_body_id(bundle, MUJOCO_BODY_NAMES[0])
        flat_qpos = command_qpos.reshape((sample_count * horizon, QPOS_DIM))
        flat_qvel = command_qvel.reshape((sample_count * horizon, QVEL_DIM))

        def forward_one(qpos_one, qvel_one):
            data = runtime.mjx.make_data(bundle.mjx_model)
            data = data.replace(qpos=qpos_one, qvel=qvel_one, time=0.0)
            data = runtime.mjx.forward(bundle.mjx_model, data)
            body_pos_w = jnp.take(data.xpos, body_id_array, axis=0)
            body_quat_w = jnp.take(data.xquat, body_id_array, axis=0)
            body_cvel = jnp.take(data.cvel, body_id_array, axis=0)
            body_ang_vel_w = body_cvel[..., 0:3]
            root_subtree_com = data.subtree_com[root_body_id]
            body_lin_vel_w = body_cvel[..., 3:6] - jnp.cross(
                body_ang_vel_w,
                root_subtree_com - body_pos_w,
            )
            return {
                "body_pos_w": body_pos_w,
                "body_quat_w": body_quat_w,
                "body_lin_vel_w": body_lin_vel_w,
                "body_ang_vel_w": body_ang_vel_w,
            }

        values = runtime.jax.vmap(forward_one)(flat_qpos, flat_qvel)
        return {
            name: value.reshape((sample_count, horizon, *value.shape[1:]))
            for name, value in values.items()
        }

    return command_reference_fn


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


def _batched_vector(name: str, value, width: int, *, jnp):
    value = jnp.asarray(value)
    if len(value.shape) == 1:
        value = jnp.expand_dims(value, axis=0)
    if len(value.shape) != 2 or int(value.shape[-1]) != int(width):
        raise ValueError(f"Expected {name} shape (batch, {width}), got {value.shape}")
    return value


def _batched_trajectory(name: str, value, width: int, *, jnp):
    value = jnp.asarray(value)
    if len(value.shape) == 2:
        value = jnp.expand_dims(value, axis=1)
    if len(value.shape) != 3 or int(value.shape[-1]) != int(width):
        raise ValueError(
            f"Expected {name} shape (batch, horizon, {width}), got {value.shape}"
        )
    return value


def _jnp_vector(name: str, value, width: int, *, jnp):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = jnp.asarray(value)
    if len(value.shape) != 1 or int(value.shape[0]) != int(width):
        raise ValueError(f"Expected {name} shape ({width},), got {value.shape}")
    return value


def _float_tuple(name: str, value, width: int) -> tuple[float, ...]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    values = tuple(float(item) for item in value)
    if len(values) != int(width):
        raise ValueError(f"Expected {name} width {width}, got {len(values)}")
    return values


def _body_ids_for_names(bundle, body_names: tuple[str, ...]) -> tuple[int, ...]:
    body_ids = tuple(_lookup_body_id(bundle, name) for name in body_names)
    if len(set(body_ids)) != len(body_ids):
        raise ValueError("Body ids must be unique")
    return body_ids


def _lookup_body_id(bundle, body_name: str) -> int:
    name_to_id = getattr(bundle, "body_name_to_id", None)
    if name_to_id is None:
        raise ValueError("MJX model bundle must expose body_name_to_id")
    for candidate in (f"robot/{body_name}", body_name):
        if candidate in name_to_id:
            return int(name_to_id[candidate])
    raise ValueError(f"MJX model bundle is missing body {body_name!r}")


def _lookup_geom_id(name_to_id: dict[str, int], geom_name: str) -> int:
    if geom_name in name_to_id:
        return int(name_to_id[geom_name])
    raise ValueError(f"MJX model bundle is missing geom {geom_name!r}")


def _contact_has_any_geom(geom, geom_ids: tuple[int, ...], *, jnp):
    geom_ids = jnp.asarray(tuple(int(value) for value in geom_ids))
    return jnp.any(geom[..., None] == geom_ids, axis=(1, 2))


def _as_indicator(value, *, jnp):
    return jnp.asarray(value) * 1.0


def _action_scale_values() -> tuple[float, ...]:
    values: list[float] = []
    for joint_name in MUJOCO_JOINT_NAMES:
        joint_kp, _joint_kd, joint_effort, _joint_armature = _match_actuator_group(
            joint_name
        )
        values.append(float(joint_effort) / (4.0 * float(joint_kp)))
    return tuple(values)


def _match_actuator_group(joint_name: str) -> tuple[float, float, float, float]:
    matches: list[tuple[float, float, float, float]] = []
    for patterns, kp, kd, effort, armature in ACTUATOR_GROUPS:
        if any(re.fullmatch(pattern, joint_name) for pattern in patterns):
            matches.append((float(kp), float(kd), float(effort), float(armature)))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one actuator group for {joint_name}, got {len(matches)}"
        )
    return matches[0]


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
    "action_to_model_ctrl",
    "default_action_scale",
    "FootContactGeomGroups",
    "foot_contact_geom_groups",
    "foot_contact_indicator_from_contact",
    "joint_order_to_model_ctrl",
    "make_mjx_command_reference_fn",
    "make_mjx_physics_step_fn",
    "reset_forward_step_smoke",
]
