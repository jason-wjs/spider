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
    other_robot_geom_ids: tuple[int, ...]


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
    left_foot_geom_ids = tuple(
        _lookup_geom_id(geom_name_to_id, name) for name in left_names
    )
    right_foot_geom_ids = tuple(
        _lookup_geom_id(geom_name_to_id, name) for name in right_names
    )
    foot_geom_ids = set(left_foot_geom_ids) | set(right_foot_geom_ids)
    other_robot_geom_ids = tuple(
        sorted(
            geom_id
            for name, geom_id in geom_name_to_id.items()
            if int(geom_id) not in foot_geom_ids
            and int(geom_id) not in floor_geom_ids
            and _is_robot_collision_geom_name(name)
        )
    )
    return FootContactGeomGroups(
        floor_geom_ids=floor_geom_ids,
        left_foot_geom_ids=left_foot_geom_ids,
        right_foot_geom_ids=right_foot_geom_ids,
        other_robot_geom_ids=other_robot_geom_ids,
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

    return floor_contact_indicator_from_contact(
        contact,
        floor_geom_ids=floor_geom_ids,
        left_foot_geom_ids=left_foot_geom_ids,
        right_foot_geom_ids=right_foot_geom_ids,
        other_robot_geom_ids=(),
        jnp=jnp,
    )[:2]


def floor_contact_indicator_from_contact(
    contact,
    *,
    floor_geom_ids: tuple[int, ...],
    left_foot_geom_ids: tuple[int, ...],
    right_foot_geom_ids: tuple[int, ...],
    other_robot_geom_ids: tuple[int, ...],
    jnp,
):
    """Return left foot, right foot, and other-robot floor contact indicators."""

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
    other = _contact_has_any_geom(geom, other_robot_geom_ids, jnp=jnp)
    return jnp.asarray(
        [
            _as_indicator(jnp.any(active & has_floor & left), jnp=jnp),
            _as_indicator(jnp.any(active & has_floor & right), jnp=jnp),
            _as_indicator(jnp.any(active & has_floor & other), jnp=jnp),
        ]
    )


def floor_contact_force_from_contact(
    contact,
    efc_force,
    *,
    floor_geom_ids: tuple[int, ...],
    left_foot_geom_ids: tuple[int, ...],
    right_foot_geom_ids: tuple[int, ...],
    other_robot_geom_ids: tuple[int, ...],
    jnp,
):
    """Return grouped floor normal forces for left, right, and other geoms."""

    return _floor_contact_force_from_contact_force(
        contact,
        _contact_normal_force(contact, efc_force, jnp=jnp),
        floor_geom_ids=floor_geom_ids,
        left_foot_geom_ids=left_foot_geom_ids,
        right_foot_geom_ids=right_foot_geom_ids,
        other_robot_geom_ids=other_robot_geom_ids,
        jnp=jnp,
    )


def floor_contact_force_first_row_from_contact(
    contact,
    efc_force,
    *,
    floor_geom_ids: tuple[int, ...],
    left_foot_geom_ids: tuple[int, ...],
    right_foot_geom_ids: tuple[int, ...],
    other_robot_geom_ids: tuple[int, ...],
    jnp,
):
    """Return grouped floor force using only the first solver row per contact."""

    return _floor_contact_force_from_contact_force(
        contact,
        _contact_first_solver_row_force(contact, efc_force, jnp=jnp),
        floor_geom_ids=floor_geom_ids,
        left_foot_geom_ids=left_foot_geom_ids,
        right_foot_geom_ids=right_foot_geom_ids,
        other_robot_geom_ids=other_robot_geom_ids,
        jnp=jnp,
    )


def _floor_contact_force_from_contact_force(
    contact,
    normal_force,
    *,
    floor_geom_ids: tuple[int, ...],
    left_foot_geom_ids: tuple[int, ...],
    right_foot_geom_ids: tuple[int, ...],
    other_robot_geom_ids: tuple[int, ...],
    jnp,
):
    geom = jnp.asarray(contact.geom)
    dist = jnp.asarray(contact.dist)
    includemargin = jnp.asarray(contact.includemargin)
    if len(geom.shape) != 2 or int(geom.shape[-1]) != 2:
        raise ValueError(f"Expected contact.geom shape (contacts, 2), got {geom.shape}")
    valid = (geom[:, 0] >= 0) & (geom[:, 1] >= 0)
    active = valid & (dist <= includemargin + 1.0e-5)
    has_floor = _contact_has_any_geom(geom, floor_geom_ids, jnp=jnp)
    return jnp.asarray(
        [
            _sum_group_force(
                active,
                has_floor,
                _contact_has_any_geom(geom, left_foot_geom_ids, jnp=jnp),
                normal_force,
                jnp=jnp,
            ),
            _sum_group_force(
                active,
                has_floor,
                _contact_has_any_geom(geom, right_foot_geom_ids, jnp=jnp),
                normal_force,
                jnp=jnp,
            ),
            _sum_group_force(
                active,
                has_floor,
                _contact_has_any_geom(geom, other_robot_geom_ids, jnp=jnp),
                normal_force,
                jnp=jnp,
            ),
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


def floor_contact_indicator_from_warp_impl(
    impl,
    world_id,
    *,
    floor_geom_ids: tuple[int, ...],
    left_foot_geom_ids: tuple[int, ...],
    right_foot_geom_ids: tuple[int, ...],
    other_robot_geom_ids: tuple[int, ...],
    jnp,
):
    """Return grouped floor contact indicators from MJX-Warp flat contact buffers."""

    geom = jnp.asarray(impl.contact__geom)
    active = _warp_active_contacts(impl, world_id, jnp=jnp)
    has_floor = _contact_has_any_geom(geom, floor_geom_ids, jnp=jnp)
    left = _contact_has_any_geom(geom, left_foot_geom_ids, jnp=jnp)
    right = _contact_has_any_geom(geom, right_foot_geom_ids, jnp=jnp)
    other = _contact_has_any_geom(geom, other_robot_geom_ids, jnp=jnp)
    return jnp.asarray(
        [
            _as_indicator(jnp.any(active & has_floor & left), jnp=jnp),
            _as_indicator(jnp.any(active & has_floor & right), jnp=jnp),
            _as_indicator(jnp.any(active & has_floor & other), jnp=jnp),
        ]
    )


def floor_contact_force_from_warp_impl(
    impl,
    world_id,
    *,
    floor_geom_ids: tuple[int, ...],
    left_foot_geom_ids: tuple[int, ...],
    right_foot_geom_ids: tuple[int, ...],
    other_robot_geom_ids: tuple[int, ...],
    jnp,
):
    """Return grouped floor normal forces from MJX-Warp flat contact buffers."""

    return _floor_contact_force_from_warp_contact_force(
        impl,
        world_id,
        _warp_contact_normal_force(impl, world_id, jnp=jnp),
        floor_geom_ids=floor_geom_ids,
        left_foot_geom_ids=left_foot_geom_ids,
        right_foot_geom_ids=right_foot_geom_ids,
        other_robot_geom_ids=other_robot_geom_ids,
        jnp=jnp,
    )


def floor_contact_force_first_row_from_warp_impl(
    impl,
    world_id,
    *,
    floor_geom_ids: tuple[int, ...],
    left_foot_geom_ids: tuple[int, ...],
    right_foot_geom_ids: tuple[int, ...],
    other_robot_geom_ids: tuple[int, ...],
    jnp,
):
    """Return grouped floor force using only the first MJX-Warp solver row."""

    return _floor_contact_force_from_warp_contact_force(
        impl,
        world_id,
        _warp_contact_first_solver_row_force(impl, world_id, jnp=jnp),
        floor_geom_ids=floor_geom_ids,
        left_foot_geom_ids=left_foot_geom_ids,
        right_foot_geom_ids=right_foot_geom_ids,
        other_robot_geom_ids=other_robot_geom_ids,
        jnp=jnp,
    )


def _floor_contact_force_from_warp_contact_force(
    impl,
    world_id,
    normal_force,
    *,
    floor_geom_ids: tuple[int, ...],
    left_foot_geom_ids: tuple[int, ...],
    right_foot_geom_ids: tuple[int, ...],
    other_robot_geom_ids: tuple[int, ...],
    jnp,
):
    geom = jnp.asarray(impl.contact__geom)
    active = _warp_active_contacts(impl, world_id, jnp=jnp)
    has_floor = _contact_has_any_geom(geom, floor_geom_ids, jnp=jnp)
    return jnp.asarray(
        [
            _sum_group_force(
                active,
                has_floor,
                _contact_has_any_geom(geom, left_foot_geom_ids, jnp=jnp),
                normal_force,
                jnp=jnp,
            ),
            _sum_group_force(
                active,
                has_floor,
                _contact_has_any_geom(geom, right_foot_geom_ids, jnp=jnp),
                normal_force,
                jnp=jnp,
            ),
            _sum_group_force(
                active,
                has_floor,
                _contact_has_any_geom(geom, other_robot_geom_ids, jnp=jnp),
                normal_force,
                jnp=jnp,
            ),
        ]
    )


def contact_count_diagnostics_from_warp_impl(impl, world_id, *, jnp) -> dict[str, object]:
    """Return per-world fixed-buffer contact counts for MJX-Warp data."""

    geom = jnp.asarray(impl.contact__geom)
    valid = _warp_valid_contacts(impl, world_id, jnp=jnp) & (
        (geom[:, 0] >= 0) & (geom[:, 1] >= 0)
    )
    active = _warp_active_contacts(impl, world_id, jnp=jnp)
    return {
        "active_contact_count": jnp.sum(active),
        "contact_pair_count": jnp.sum(valid),
    }


def _warp_floor_contact_summary(
    impl,
    world_id,
    *,
    contact_groups: FootContactGeomGroups,
    include_peak_source: bool = False,
    jnp,
) -> dict[str, object]:
    geom = jnp.asarray(impl.contact__geom)
    valid = _warp_valid_contacts(impl, world_id, jnp=jnp) & (
        (geom[:, 0] >= 0) & (geom[:, 1] >= 0)
    )
    active = valid & (
        jnp.asarray(impl.contact__dist)
        <= jnp.asarray(impl.contact__includemargin) + 1.0e-5
    )
    has_floor = _contact_has_any_geom(
        geom,
        contact_groups.floor_geom_ids,
        jnp=jnp,
    )
    left = _contact_has_any_geom(geom, contact_groups.left_foot_geom_ids, jnp=jnp)
    right = _contact_has_any_geom(geom, contact_groups.right_foot_geom_ids, jnp=jnp)
    other = _contact_has_any_geom(geom, contact_groups.other_robot_geom_ids, jnp=jnp)
    normal_force = _warp_contact_normal_force(impl, world_id, jnp=jnp)
    first_row_force = _warp_contact_first_solver_row_force(impl, world_id, jnp=jnp)
    summary = {
        "floor_contact": jnp.asarray(
            [
                _as_indicator(jnp.any(active & has_floor & left), jnp=jnp),
                _as_indicator(jnp.any(active & has_floor & right), jnp=jnp),
                _as_indicator(jnp.any(active & has_floor & other), jnp=jnp),
            ]
        ),
        "floor_contact_force": jnp.asarray(
            [
                _sum_group_force(active, has_floor, left, normal_force, jnp=jnp),
                _sum_group_force(active, has_floor, right, normal_force, jnp=jnp),
                _sum_group_force(active, has_floor, other, normal_force, jnp=jnp),
            ]
        ),
        "floor_contact_force_first_row": jnp.asarray(
            [
                _sum_group_force(active, has_floor, left, first_row_force, jnp=jnp),
                _sum_group_force(active, has_floor, right, first_row_force, jnp=jnp),
                _sum_group_force(active, has_floor, other, first_row_force, jnp=jnp),
            ]
        ),
        "contact_counts": {
            "active_contact_count": jnp.sum(active),
            "contact_pair_count": jnp.sum(valid),
        },
    }
    if include_peak_source:
        summary["floor_contact_force_peak_source"] = jnp.asarray(
            [
                _warp_contact_force_peak_source_row(
                    geom,
                    active,
                    has_floor,
                    left,
                    normal_force,
                    first_row_force,
                    impl=impl,
                    jnp=jnp,
                ),
                _warp_contact_force_peak_source_row(
                    geom,
                    active,
                    has_floor,
                    right,
                    normal_force,
                    first_row_force,
                    impl=impl,
                    jnp=jnp,
                ),
                _warp_contact_force_peak_source_row(
                    geom,
                    active,
                    has_floor,
                    other,
                    normal_force,
                    first_row_force,
                    impl=impl,
                    jnp=jnp,
                ),
            ]
        )
    return summary


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
    contact_force_mode: str = "sum_rows",
    contact_force_first_row_diagnostics: bool = False,
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
    contact_force_mode = str(contact_force_mode)
    if contact_force_mode not in {"sum_rows", "first_row"}:
        raise ValueError("contact_force_mode must be 'sum_rows' or 'first_row'")
    contact_force_first_row_diagnostics = bool(contact_force_first_row_diagnostics)
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

        def outputs_from_data(
            data,
            joint_control,
            *,
            include_mjx_data: bool,
            world_id=None,
        ):
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
            if include_mjx_data:
                next_robot["mjx_data"] = data
            data_impl = data._impl
            need_first_row_force = (
                contact_force_mode == "first_row"
                or contact_force_first_row_diagnostics
            )
            floor_contact_force_peak_source = None
            if _is_warp_data_impl(data_impl):
                contact_summary = _warp_floor_contact_summary(
                    data_impl,
                    world_id,
                    contact_groups=contact_groups,
                    include_peak_source=contact_force_first_row_diagnostics,
                    jnp=jnp,
                )
                floor_contact = contact_summary["floor_contact"]
                floor_contact_force = contact_summary["floor_contact_force"]
                contact_counts = contact_summary["contact_counts"]
                if need_first_row_force:
                    floor_contact_force_first_row = contact_summary[
                        "floor_contact_force_first_row"
                    ]
                floor_contact_force_peak_source = contact_summary.get(
                    "floor_contact_force_peak_source"
                )
            else:
                floor_contact = floor_contact_indicator_from_contact(
                    data_impl.contact,
                    floor_geom_ids=contact_groups.floor_geom_ids,
                    left_foot_geom_ids=contact_groups.left_foot_geom_ids,
                    right_foot_geom_ids=contact_groups.right_foot_geom_ids,
                    other_robot_geom_ids=contact_groups.other_robot_geom_ids,
                    jnp=jnp,
                )
                floor_contact_force = floor_contact_force_from_contact(
                    data_impl.contact,
                    data_impl.efc_force,
                    floor_geom_ids=contact_groups.floor_geom_ids,
                    left_foot_geom_ids=contact_groups.left_foot_geom_ids,
                    right_foot_geom_ids=contact_groups.right_foot_geom_ids,
                    other_robot_geom_ids=contact_groups.other_robot_geom_ids,
                    jnp=jnp,
                )
                if need_first_row_force:
                    floor_contact_force_first_row = (
                        floor_contact_force_first_row_from_contact(
                            data_impl.contact,
                            data_impl.efc_force,
                            floor_geom_ids=contact_groups.floor_geom_ids,
                            left_foot_geom_ids=contact_groups.left_foot_geom_ids,
                            right_foot_geom_ids=contact_groups.right_foot_geom_ids,
                            other_robot_geom_ids=contact_groups.other_robot_geom_ids,
                            jnp=jnp,
                        )
                    )
                contact_counts = contact_count_diagnostics(data_impl.contact, jnp=jnp)
            scoring_floor_contact_force = (
                floor_contact_force_first_row
                if contact_force_mode == "first_row"
                else floor_contact_force
            )
            score_state = {
                "root_pos": data.qpos[:3],
                "root_quat": data.qpos[3:7],
                "body_pos": jnp.take(data.xpos, score_body_id_array, axis=0),
                "body_quat": jnp.take(data.xquat, score_body_id_array, axis=0),
                "ee_pos": jnp.take(data.xpos, ee_body_id_array, axis=0),
                "ee_quat": jnp.take(data.xquat, ee_body_id_array, axis=0),
                "floor_contact": floor_contact,
                "floor_contact_force": scoring_floor_contact_force,
                **contact_counts,
                "model_ctrl": data.ctrl,
                "joint_control": joint_control,
                "time": data.time,
            }
            if contact_force_first_row_diagnostics:
                score_state["floor_contact_force_first_row"] = (
                    floor_contact_force_first_row
                )
                if floor_contact_force_peak_source is not None:
                    score_state["floor_contact_force_peak_source"] = (
                        floor_contact_force_peak_source
                    )
            score_state["contact"] = score_state["floor_contact"][:2]
            score_state["contact_force"] = score_state["floor_contact_force"][:2]
            if contact_force_first_row_diagnostics:
                score_state["contact_force_first_row"] = score_state[
                    "floor_contact_force_first_row"
                ][:2]
            return next_robot, score_state

        def step_one_from_state(qpos_one, qvel_one, action_one, world_id):
            joint_control = action_one * jnp.asarray(action_scale) + jnp.asarray(
                default_joint_pos
            )
            model_ctrl = action_to_model_ctrl(
                bundle,
                action_one,
                default_joint_pos,
                action_scale,
                jnp=jnp,
            )
            data = _make_mjx_data(bundle, runtime=runtime)
            data = data.replace(
                qpos=qpos_one,
                qvel=qvel_one,
                ctrl=model_ctrl,
                time=jnp.asarray(0.0),
            )
            data = runtime.mjx.forward(bundle.mjx_model, data)
            data = _step_fixed_count(
                bundle.mjx_model,
                data,
                runtime=runtime,
                steps=decimation,
            )
            return outputs_from_data(
                data,
                joint_control,
                include_mjx_data=False,
                world_id=world_id,
            )

        def step_one_from_data(data_one, action_one, world_id):
            joint_control = action_one * jnp.asarray(action_scale) + jnp.asarray(
                default_joint_pos
            )
            model_ctrl = action_to_model_ctrl(
                bundle,
                action_one,
                default_joint_pos,
                action_scale,
                jnp=jnp,
            )
            data = data_one.replace(ctrl=model_ctrl)
            data = _step_fixed_count(
                bundle.mjx_model,
                data,
                runtime=runtime,
                steps=decimation,
            )
            return outputs_from_data(
                data,
                joint_control,
                include_mjx_data=True,
                world_id=world_id,
            )

        world_ids = jnp.arange(sample_count)
        mjx_data = robot_state.get("mjx_data")
        if mjx_data is not None and _mjx_data_batch_size(mjx_data) == sample_count:
            return runtime.jax.vmap(step_one_from_data)(
                mjx_data,
                action_array,
                world_ids,
            )
        return runtime.jax.vmap(step_one_from_state)(
            qpos,
            qvel,
            action_array,
            world_ids,
        )

    def initialize_robot_state(bundle, robot_state, sample_count: int, *, runtime):
        if getattr(bundle, "mjx_model", None) is None:
            raise ValueError("MJX model bundle must include mjx_model")
        sample_count = int(sample_count)
        jnp = runtime.jnp
        existing = robot_state.get("mjx_data")
        if existing is not None:
            existing = _broadcast_mjx_data(existing, sample_count, runtime=runtime)
            if existing is not None:
                initialized = dict(robot_state)
                initialized["mjx_data"] = existing
                return initialized
        qpos = _batched_vector(
            "robot_state['qpos']",
            robot_state["qpos"],
            QPOS_DIM,
            jnp=jnp,
        )
        qvel = _batched_vector(
            "robot_state['qvel']",
            robot_state["qvel"],
            QVEL_DIM,
            jnp=jnp,
        )
        zero_ctrl = jnp.zeros((sample_count, ACTION_DIM))
        template = _make_mjx_data(bundle, runtime=runtime)

        def forward_one(qpos_one, qvel_one, ctrl_one):
            data = template.replace(
                qpos=qpos_one,
                qvel=qvel_one,
                ctrl=ctrl_one,
                time=jnp.asarray(0.0),
            )
            return runtime.mjx.forward(bundle.mjx_model, data)

        initialized = dict(robot_state)
        initialized["mjx_data"] = runtime.jax.vmap(forward_one)(qpos, qvel, zero_ctrl)
        return initialized

    physics_step_fn.initialize_robot_state = initialize_robot_state
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
            data = _make_mjx_data(bundle, runtime=runtime)
            data = data.replace(qpos=qpos_one, qvel=qvel_one, time=jnp.asarray(0.0))
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
    data = _make_mjx_data(bundle, runtime=runtime)
    data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl, time=jnp.asarray(0.0))
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


def _mjx_data_batch_size(data) -> int | None:
    qpos = getattr(data, "qpos", None)
    shape = getattr(qpos, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    if int(shape[-1]) != QPOS_DIM:
        return None
    return int(shape[0])


def _broadcast_mjx_data(data, sample_count: int, *, runtime):
    current = _mjx_data_batch_size(data)
    if current == int(sample_count):
        return data
    if current != 1:
        return None
    tree_util = getattr(getattr(runtime, "jax", None), "tree_util", None)
    tree_map = getattr(tree_util, "tree_map", None)
    if not callable(tree_map):
        return None

    def broadcast_leaf(value):
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) > 0 and int(shape[0]) == 1:
            return runtime.jnp.repeat(value, int(sample_count), axis=0)
        return value

    return tree_map(broadcast_leaf, data)


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


def _is_robot_collision_geom_name(name: str) -> bool:
    bare_name = str(name).removeprefix("robot/")
    return bare_name.endswith("_collision")


def _contact_normal_force(contact, efc_force, *, jnp):
    dim = jnp.asarray(contact.dim).astype("int32")
    efc_address = jnp.asarray(contact.efc_address).astype("int32")
    efc_force = jnp.asarray(efc_force)
    if len(efc_address.shape) == 2:
        row_offsets = jnp.arange(int(efc_address.shape[1]))
        rows = efc_address
        row_valid = (efc_address >= 0) & (
            row_offsets[None, :] < _contact_solver_row_count(dim, jnp=jnp)[:, None]
        )
    else:
        row_offsets = jnp.arange(10)
        rows = efc_address[:, None] + row_offsets[None, :]
        row_valid = (efc_address[:, None] >= 0) & (
            row_offsets[None, :] < _contact_solver_row_count(dim, jnp=jnp)[:, None]
        )
    clipped_rows = jnp.clip(rows, 0, int(efc_force.shape[0]) - 1)
    force_rows = efc_force[clipped_rows]
    normal_force = jnp.sum(jnp.where(row_valid, force_rows, 0.0), axis=1)
    return jnp.maximum(normal_force, 0.0)


def _contact_first_solver_row_force(contact, efc_force, *, jnp):
    efc_address = jnp.asarray(contact.efc_address).astype("int32")
    efc_force = jnp.asarray(efc_force)
    address = efc_address[:, 0] if len(efc_address.shape) == 2 else efc_address
    clipped = jnp.clip(address, 0, int(efc_force.shape[0]) - 1)
    force = jnp.where(address >= 0, efc_force[clipped], 0.0)
    return jnp.maximum(force, 0.0)


def _make_mjx_data(bundle, *, runtime):
    from spider.tasks.g1_wbc.mjx_model import make_mjx_data

    return make_mjx_data(bundle, runtime=runtime)


def _is_warp_data_impl(data_impl) -> bool:
    return hasattr(data_impl, "contact__geom") and hasattr(data_impl, "contact__worldid")


def _warp_valid_contacts(impl, world_id, *, jnp):
    geom = jnp.asarray(impl.contact__geom)
    ids = jnp.arange(int(geom.shape[0]))
    world_ids = jnp.asarray(impl.contact__worldid)
    same_world = world_ids == world_id
    nacon = jnp.asarray(impl.nacon)
    nacon_limit = nacon[0] if len(nacon.shape) > 0 else nacon
    nacon_limit = jnp.minimum(nacon_limit, int(geom.shape[0]))
    return (ids < nacon_limit) & same_world


def _warp_active_contacts(impl, world_id, *, jnp):
    geom = jnp.asarray(impl.contact__geom)
    valid = _warp_valid_contacts(impl, world_id, jnp=jnp)
    return (
        valid
        & (geom[:, 0] >= 0)
        & (geom[:, 1] >= 0)
        & (jnp.asarray(impl.contact__dist) <= jnp.asarray(impl.contact__includemargin) + 1.0e-5)
    )


def _warp_contact_normal_force(impl, world_id, *, jnp):
    dim = jnp.asarray(impl.contact__dim).astype("int32")
    efc_address = jnp.asarray(impl.contact__efc_address).astype("int32")
    efc_force = jnp.asarray(impl.efc__force)
    if len(efc_force.shape) == 2:
        efc_force = efc_force[world_id]
    if len(efc_address.shape) == 2:
        row_offsets = jnp.arange(int(efc_address.shape[1]))
        rows = efc_address
        row_valid = (efc_address >= 0) & (
            row_offsets[None, :] < _contact_solver_row_count(dim, jnp=jnp)[:, None]
        )
    else:
        row_offsets = jnp.arange(10)
        rows = efc_address[:, None] + row_offsets[None, :]
        row_valid = (efc_address[:, None] >= 0) & (
            row_offsets[None, :] < _contact_solver_row_count(dim, jnp=jnp)[:, None]
        )
    clipped_rows = jnp.clip(rows, 0, int(efc_force.shape[0]) - 1)
    force_rows = efc_force[clipped_rows]
    normal_force = jnp.sum(jnp.where(row_valid, force_rows, 0.0), axis=1)
    return jnp.maximum(normal_force, 0.0)


def _warp_contact_first_solver_row_force(impl, world_id, *, jnp):
    efc_address = jnp.asarray(impl.contact__efc_address).astype("int32")
    efc_force = jnp.asarray(impl.efc__force)
    if len(efc_force.shape) == 2:
        efc_force = efc_force[world_id]
    address = efc_address[:, 0] if len(efc_address.shape) == 2 else efc_address
    clipped = jnp.clip(address, 0, int(efc_force.shape[0]) - 1)
    force = jnp.where(address >= 0, efc_force[clipped], 0.0)
    return jnp.maximum(force, 0.0)


def _warp_contact_force_peak_source_row(
    geom,
    active,
    has_floor,
    group,
    normal_force,
    first_row_force,
    *,
    impl,
    jnp,
):
    mask = active & has_floor & group
    masked_force = jnp.where(mask, normal_force, 0.0)
    row_id = jnp.argmax(masked_force)
    peak_force = masked_force[row_id]
    present = peak_force > 0.0
    efc_address = jnp.asarray(impl.contact__efc_address).astype("int32")
    first_efc = (
        efc_address[:, 0]
        if len(efc_address.shape) == 2
        else efc_address
    )
    dim = jnp.asarray(impl.contact__dim).astype("int32")
    row_value = jnp.where(present, row_id, -1)
    geom0 = jnp.where(present, geom[row_id, 0], -1)
    geom1 = jnp.where(present, geom[row_id, 1], -1)
    first_efc_value = jnp.where(present, first_efc[row_id], -1)
    return jnp.asarray(
        [
            row_value,
            geom0,
            geom1,
            peak_force,
            jnp.where(present, first_row_force[row_id], 0.0),
            jnp.sum(masked_force),
            jnp.where(present, dim[row_id], 0),
            first_efc_value,
        ]
    )


def _contact_solver_row_count(dim, *, jnp):
    return jnp.where(dim == 1, 1, 2 * (dim - 1))


def _sum_group_force(active, has_floor, group, normal_force, *, jnp):
    return jnp.sum(jnp.where(active & has_floor & group, normal_force, 0.0))


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
    "floor_contact_indicator_from_contact",
    "floor_contact_force_from_contact",
    "floor_contact_force_first_row_from_contact",
    "floor_contact_force_from_warp_impl",
    "floor_contact_force_first_row_from_warp_impl",
    "joint_order_to_model_ctrl",
    "make_mjx_command_reference_fn",
    "make_mjx_physics_step_fn",
    "reset_forward_step_smoke",
]
