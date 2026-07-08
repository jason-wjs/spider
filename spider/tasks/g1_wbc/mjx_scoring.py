"""JAX-shaped scoring helpers for G1 WBC MJX rollouts."""

from __future__ import annotations

from dataclasses import dataclass

from spider.tasks.g1_wbc.constants import (
    ANCHOR_BODY_NAME,
    HAND_EE_BODY_NAMES,
    MUJOCO_BODY_NAMES,
    POLICY_DT,
    TASK_EE_BODY_NAMES,
)


_CONTACT_FORCE_SCALE = 300.0
_CONTACT_FORCE_PEAK_SOURCE_WIDTH = 8
_CONTACT_FORCE_PEAK_SOURCE_COLUMNS = tuple(
    f"contact_force_peak_source_{index}"
    for index in range(_CONTACT_FORCE_PEAK_SOURCE_WIDTH)
)
_CONTACT_FORCE_PEAK_SOURCE_DEFAULTS = (-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
_ANCHOR_INDEX = MUJOCO_BODY_NAMES.index(ANCHOR_BODY_NAME)
_LOCAL_BODY_INDICES = tuple(
    index for index, name in enumerate(MUJOCO_BODY_NAMES) if name != ANCHOR_BODY_NAME
)
_EE_BODY_INDICES = tuple(MUJOCO_BODY_NAMES.index(name) for name in TASK_EE_BODY_NAMES)
_HAND_BODY_INDICES = tuple(MUJOCO_BODY_NAMES.index(name) for name in HAND_EE_BODY_NAMES)

try:
    from jax import tree_util as _jax_tree_util
except Exception:
    _jax_tree_util = None


def _register_pytree_node_class(cls):
    if _jax_tree_util is None:
        return cls
    return _jax_tree_util.register_pytree_node_class(cls)


@_register_pytree_node_class
@dataclass(frozen=True)
class JaxScoreWeights:
    terms: dict[str, float]

    def tree_flatten(self):
        return (), tuple(
            sorted((name, float(value)) for name, value in self.terms.items())
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(dict(aux_data))


ACCUMULATOR_KEYS = (
    "score_sum",
    "root_pos_error_sum",
    "root_rot_error_sum",
    "joint_pos_error_sum",
    "body_global_pos_error_sum",
    "body_global_rot_error_sum",
    "body_local_pos_error_sum",
    "body_local_rot_error_sum",
    "ee_global_pos_error_sum",
    "ee_global_rot_error_sum",
    "ee_local_pos_error_sum",
    "ee_local_rot_error_sum",
    "hand_global_pos_error_sum",
    "hand_global_rot_error_sum",
    "hand_local_pos_error_sum",
    "hand_local_rot_error_sum",
    "contact_mismatch_sum",
    "contact_false_positive_sum",
    "contact_false_negative_sum",
    "contact_switch_sum",
    "bad_floor_contact_sum",
    "bad_floor_force_excess_sum",
    "contact_force_active_sum",
    "contact_force_active_count",
    "contact_force_peak_max",
    *_CONTACT_FORCE_PEAK_SOURCE_COLUMNS,
    "contact_force_peak_excess_sum",
    "contact_force_delta_sum",
    "control_delta_sum",
    "action_delta_sum",
    "joint_acc_sum",
    "joint_jerk_sum",
    "count",
    "active_contact_count",
    "contact_pair_count",
)


def init_score_accumulator(batch_shape=(), *, jnp):
    """Create a stable score accumulator PyTree for JAX scan carries."""

    shape = _shape_tuple(batch_shape)
    accumulator = {name: jnp.zeros(shape) for name in ACCUMULATOR_KEYS}
    for name, default in zip(
        _CONTACT_FORCE_PEAK_SOURCE_COLUMNS,
        _CONTACT_FORCE_PEAK_SOURCE_DEFAULTS,
    ):
        accumulator[name] = accumulator[name] + float(default)
    return accumulator


def score_step(
    accumulator,
    step_state,
    reference_state,
    weights: JaxScoreWeights,
    *,
    jnp,
    collect_metrics: bool = True,
):
    """Accumulate one step of core WBC rollout score terms."""

    terms = _require_accumulator(accumulator)
    batch_shape = _shape_tuple(terms["count"].shape)
    root_error = _mean_l2_delta(
        step_state["root_pos"],
        reference_state["root_pos"],
        batch_shape=batch_shape,
        jnp=jnp,
    )
    root_rot_error = _mean_quat_error(
        step_state,
        reference_state,
        "root_quat",
        required=_weight(weights, "root_rot", "root_rot_error") != 0.0,
        batch_shape=batch_shape,
        jnp=jnp,
    )
    joint_pos_error = _mean_optional_l2_delta(
        step_state,
        "joint_pos",
        "joint_pos",
        required=_weight(weights, "joint_pos", "joint_pos_error") != 0.0,
        batch_shape=batch_shape,
        jnp=jnp,
        expected_container=reference_state,
    )
    body_error = _mean_l2_delta(
        step_state["body_pos"],
        reference_state["body_pos"],
        batch_shape=batch_shape,
        jnp=jnp,
    )
    body_rot_error = _mean_quat_error(
        step_state,
        reference_state,
        "body_quat",
        required=_weight(weights, "body_global_rot", "body_global_rot_error") != 0.0,
        batch_shape=batch_shape,
        jnp=jnp,
    )
    body_local_pos_error, body_local_rot_error = _local_body_errors(
        step_state,
        reference_state,
        "body_pos",
        "body_quat",
        _LOCAL_BODY_INDICES,
        required=(
            _weight(weights, "body_local_pos", "body_local_pos_error") != 0.0
            or _weight(weights, "body_local_rot", "body_local_rot_error") != 0.0
        ),
        batch_shape=batch_shape,
        jnp=jnp,
    )
    ee_error = _mean_l2_delta(
        step_state["ee_pos"],
        reference_state["ee_pos"],
        batch_shape=batch_shape,
        jnp=jnp,
    )
    ee_rot_error = _mean_quat_error(
        step_state,
        reference_state,
        "ee_quat",
        required=_weight(weights, "ee_global_rot", "ee_global_rot_error") != 0.0,
        batch_shape=batch_shape,
        jnp=jnp,
    )
    ee_local_pos_error, ee_local_rot_error = _local_body_errors(
        step_state,
        reference_state,
        "body_pos",
        "body_quat",
        _EE_BODY_INDICES,
        required=(
            _weight(weights, "ee_local_pos", "ee_local_pos_error") != 0.0
            or _weight(weights, "ee_local_rot", "ee_local_rot_error") != 0.0
        ),
        batch_shape=batch_shape,
        jnp=jnp,
    )
    hand_global_pos_error = _mean_indexed_l2_delta(
        step_state,
        reference_state,
        "body_pos",
        _HAND_BODY_INDICES,
        required=_weight(weights, "hand_global_pos", "hand_global_pos_error") != 0.0,
        batch_shape=batch_shape,
        jnp=jnp,
    )
    hand_global_rot_error = _mean_indexed_quat_error(
        step_state,
        reference_state,
        "body_quat",
        _HAND_BODY_INDICES,
        required=_weight(weights, "hand_global_rot", "hand_global_rot_error") != 0.0,
        batch_shape=batch_shape,
        jnp=jnp,
    )
    hand_local_pos_error, hand_local_rot_error = _local_body_errors(
        step_state,
        reference_state,
        "body_pos",
        "body_quat",
        _HAND_BODY_INDICES,
        required=(
            _weight(weights, "hand_local_pos", "hand_local_pos_error") != 0.0
            or _weight(weights, "hand_local_rot", "hand_local_rot_error") != 0.0
        ),
        batch_shape=batch_shape,
        jnp=jnp,
    )
    contact_error = _mean_abs(
        step_state["contact"],
        reference_state["contact"],
        batch_shape=batch_shape,
        jnp=jnp,
    )
    contact_false_positive = _mean_contact_indicator(
        step_state["contact"],
        reference_state["contact"],
        positive=True,
        batch_shape=batch_shape,
        jnp=jnp,
    )
    contact_false_negative = _mean_contact_indicator(
        step_state["contact"],
        reference_state["contact"],
        positive=False,
        batch_shape=batch_shape,
        jnp=jnp,
    )
    contact_switch = _mean_optional_l2_delta(
        step_state,
        "contact",
        "prev_contact",
        required=_weight(weights, "contact_switch") != 0.0,
        batch_shape=batch_shape,
        jnp=jnp,
    ) * _validity_value(step_state, "prev_contact_valid", batch_shape, jnp=jnp)
    bad_floor_contact = _mean_optional_feature_tail(
        step_state,
        "floor_contact",
        start=2,
        required=_weight(weights, "bad_floor_contact") != 0.0,
        batch_shape=batch_shape,
        jnp=jnp,
    )
    bad_floor_force_excess = _mean_optional_force_excess(
        step_state,
        "floor_contact_force",
        start=2,
        required=_weight(weights, "bad_floor_force_excess") != 0.0,
        batch_shape=batch_shape,
        jnp=jnp,
    )
    contact_force_delta = (
        _mean_optional_l2_delta(
            step_state,
            "contact_force",
            "prev_contact_force",
            required=_weight(weights, "contact_force_delta") != 0.0,
            batch_shape=batch_shape,
            jnp=jnp,
        )
        / _CONTACT_FORCE_SCALE
        * _validity_value(
            step_state,
            "prev_contact_force_valid",
            batch_shape,
            jnp=jnp,
        )
    )
    contact_force_active_sum, contact_force_active_count = (
        _active_contact_force_stats(
            step_state,
            required=_weight(weights, "contact_force_active") != 0.0,
            batch_shape=batch_shape,
            jnp=jnp,
        )
    )
    contact_force_active = (
        contact_force_active_sum / jnp.maximum(contact_force_active_count, 1.0)
    ) / _CONTACT_FORCE_SCALE
    contact_force_peak, contact_force_peak_excess = _contact_force_peak_excess(
        step_state,
        required=_weight(weights, "contact_force_peak_excess") != 0.0,
        batch_shape=batch_shape,
        jnp=jnp,
    )
    contact_force_peak_source = _contact_force_peak_source(
        step_state,
        batch_shape=batch_shape,
        jnp=jnp,
    )
    control_delta = _mean_l2_delta(
        step_state["control"],
        step_state["prev_control"],
        batch_shape=batch_shape,
        jnp=jnp,
    )
    action_delta = _mean_optional_l2_delta(
        step_state,
        "action",
        "prev_action",
        required=_weight(weights, "action_delta") != 0.0,
        batch_shape=batch_shape,
        jnp=jnp,
    )
    joint_acc_delta = jnp.asarray(step_state["joint_vel"]) - jnp.asarray(
        step_state["prev_joint_vel"]
    )
    joint_acc = _mean_l2(
        joint_acc_delta,
        batch_shape=batch_shape,
        jnp=jnp,
    ) / POLICY_DT
    if "prev_joint_acc" in step_state:
        joint_jerk = _mean_l2_delta(
            joint_acc_delta,
            step_state["prev_joint_acc"],
            batch_shape=batch_shape,
            jnp=jnp,
        ) / POLICY_DT
    elif _weight(weights, "joint_jerk") != 0.0:
        raise KeyError("Missing score field 'prev_joint_acc'")
    else:
        joint_jerk = jnp.zeros(batch_shape)

    penalty = (
        _weight(weights, "root_pos", "root_pos_error") * root_error
        + _weight(weights, "root_rot", "root_rot_error") * root_rot_error
        + _weight(weights, "joint_pos", "joint_pos_error") * joint_pos_error
        + _weight(weights, "body_global_pos", "body_global_pos_error") * body_error
        + _weight(weights, "body_global_rot", "body_global_rot_error")
        * body_rot_error
        + _weight(weights, "body_local_pos", "body_local_pos_error")
        * body_local_pos_error
        + _weight(weights, "body_local_rot", "body_local_rot_error")
        * body_local_rot_error
        + _weight(weights, "ee_global_pos", "ee_global_pos_error") * ee_error
        + _weight(weights, "ee_global_rot", "ee_global_rot_error") * ee_rot_error
        + _weight(weights, "ee_local_pos", "ee_local_pos_error")
        * ee_local_pos_error
        + _weight(weights, "ee_local_rot", "ee_local_rot_error")
        * ee_local_rot_error
        + _weight(weights, "hand_global_pos", "hand_global_pos_error")
        * hand_global_pos_error
        + _weight(weights, "hand_global_rot", "hand_global_rot_error")
        * hand_global_rot_error
        + _weight(weights, "hand_local_pos", "hand_local_pos_error")
        * hand_local_pos_error
        + _weight(weights, "hand_local_rot", "hand_local_rot_error")
        * hand_local_rot_error
        + _weight(weights, "contact", "contact_mismatch") * contact_error
        + _weight(weights, "contact_false_positive") * contact_false_positive
        + _weight(weights, "contact_false_negative") * contact_false_negative
        + _weight(weights, "contact_switch") * contact_switch
        + _weight(weights, "bad_floor_contact") * bad_floor_contact
        + _weight(weights, "bad_floor_force_excess") * bad_floor_force_excess
        + _weight(weights, "contact_force_active") * contact_force_active
        + _weight(weights, "contact_force_peak_excess") * contact_force_peak_excess
        + _weight(weights, "contact_force_delta") * contact_force_delta
        + _weight(weights, "control_delta") * control_delta
        + _weight(weights, "action_delta") * action_delta
        + _weight(weights, "joint_acc") * joint_acc
        + _weight(weights, "joint_jerk") * joint_jerk
    )
    if collect_metrics:
        terms["root_pos_error_sum"] = terms["root_pos_error_sum"] + root_error
        terms["root_rot_error_sum"] = terms["root_rot_error_sum"] + root_rot_error
        terms["joint_pos_error_sum"] = terms["joint_pos_error_sum"] + joint_pos_error
        terms["body_global_pos_error_sum"] = (
            terms["body_global_pos_error_sum"] + body_error
        )
        terms["body_global_rot_error_sum"] = (
            terms["body_global_rot_error_sum"] + body_rot_error
        )
        terms["body_local_pos_error_sum"] = (
            terms["body_local_pos_error_sum"] + body_local_pos_error
        )
        terms["body_local_rot_error_sum"] = (
            terms["body_local_rot_error_sum"] + body_local_rot_error
        )
        terms["ee_global_pos_error_sum"] = terms["ee_global_pos_error_sum"] + ee_error
        terms["ee_global_rot_error_sum"] = (
            terms["ee_global_rot_error_sum"] + ee_rot_error
        )
        terms["ee_local_pos_error_sum"] = (
            terms["ee_local_pos_error_sum"] + ee_local_pos_error
        )
        terms["ee_local_rot_error_sum"] = (
            terms["ee_local_rot_error_sum"] + ee_local_rot_error
        )
        terms["hand_global_pos_error_sum"] = (
            terms["hand_global_pos_error_sum"] + hand_global_pos_error
        )
        terms["hand_global_rot_error_sum"] = (
            terms["hand_global_rot_error_sum"] + hand_global_rot_error
        )
        terms["hand_local_pos_error_sum"] = (
            terms["hand_local_pos_error_sum"] + hand_local_pos_error
        )
        terms["hand_local_rot_error_sum"] = (
            terms["hand_local_rot_error_sum"] + hand_local_rot_error
        )
        terms["contact_mismatch_sum"] = terms["contact_mismatch_sum"] + contact_error
        terms["contact_false_positive_sum"] = (
            terms["contact_false_positive_sum"] + contact_false_positive
        )
        terms["contact_false_negative_sum"] = (
            terms["contact_false_negative_sum"] + contact_false_negative
        )
        terms["contact_switch_sum"] = terms["contact_switch_sum"] + contact_switch
        terms["bad_floor_contact_sum"] = (
            terms["bad_floor_contact_sum"] + bad_floor_contact
        )
        terms["bad_floor_force_excess_sum"] = (
            terms["bad_floor_force_excess_sum"] + bad_floor_force_excess
        )
        terms["contact_force_active_sum"] = (
            terms["contact_force_active_sum"] + contact_force_active_sum
        )
        terms["contact_force_active_count"] = (
            terms["contact_force_active_count"] + contact_force_active_count
        )
        contact_force_peak_source_update = contact_force_peak > terms[
            "contact_force_peak_max"
        ]
        for index, name in enumerate(_CONTACT_FORCE_PEAK_SOURCE_COLUMNS):
            terms[name] = _where(
                contact_force_peak_source_update,
                contact_force_peak_source[..., index],
                terms[name],
                jnp=jnp,
            )
        terms["contact_force_peak_max"] = jnp.maximum(
            terms["contact_force_peak_max"],
            contact_force_peak,
        )
        terms["contact_force_peak_excess_sum"] = (
            terms["contact_force_peak_excess_sum"] + contact_force_peak_excess
        )
        terms["contact_force_delta_sum"] = (
            terms["contact_force_delta_sum"] + contact_force_delta
        )
        terms["control_delta_sum"] = terms["control_delta_sum"] + control_delta
        terms["action_delta_sum"] = terms["action_delta_sum"] + action_delta
        terms["joint_acc_sum"] = terms["joint_acc_sum"] + joint_acc
        terms["joint_jerk_sum"] = terms["joint_jerk_sum"] + joint_jerk
        terms["active_contact_count"] = jnp.maximum(
            terms["active_contact_count"],
            _diagnostic_value(step_state, "active_contact_count", batch_shape, jnp=jnp),
        )
        terms["contact_pair_count"] = jnp.maximum(
            terms["contact_pair_count"],
            _diagnostic_value(step_state, "contact_pair_count", batch_shape, jnp=jnp),
        )
    terms["count"] = terms["count"] + 1.0
    terms["score_sum"] = terms["score_sum"] - penalty
    return terms


def finalize_score_only(accumulator, *, jnp):
    """Return only the mean rollout score from a streaming score accumulator."""

    terms = _empty_or_valid_accumulator(accumulator, jnp=jnp)
    count = jnp.maximum(terms["count"], 1.0)
    return {"score": terms["score_sum"] / count}


def finalize_score(accumulator, *, jnp):
    """Return mean score metrics from a streaming score accumulator."""

    terms = _empty_or_valid_accumulator(accumulator, jnp=jnp)
    count = jnp.maximum(terms["count"], 1.0)
    score = terms["score_sum"] / count
    root_pos_error = terms["root_pos_error_sum"] / count
    root_rot_error = terms["root_rot_error_sum"] / count
    joint_pos_error = terms["joint_pos_error_sum"] / count
    body_global_pos_error = terms["body_global_pos_error_sum"] / count
    body_global_rot_error = terms["body_global_rot_error_sum"] / count
    body_local_pos_error = terms["body_local_pos_error_sum"] / count
    body_local_rot_error = terms["body_local_rot_error_sum"] / count
    ee_global_pos_error = terms["ee_global_pos_error_sum"] / count
    ee_global_rot_error = terms["ee_global_rot_error_sum"] / count
    ee_local_pos_error = terms["ee_local_pos_error_sum"] / count
    ee_local_rot_error = terms["ee_local_rot_error_sum"] / count
    hand_global_pos_error = terms["hand_global_pos_error_sum"] / count
    hand_global_rot_error = terms["hand_global_rot_error_sum"] / count
    hand_local_pos_error = terms["hand_local_pos_error_sum"] / count
    hand_local_rot_error = terms["hand_local_rot_error_sum"] / count
    contact_mismatch = terms["contact_mismatch_sum"] / count
    contact_false_positive = terms["contact_false_positive_sum"] / count
    contact_false_negative = terms["contact_false_negative_sum"] / count
    contact_switch = terms["contact_switch_sum"] / count
    bad_floor_contact = terms["bad_floor_contact_sum"] / count
    bad_floor_force_excess = terms["bad_floor_force_excess_sum"] / count
    contact_force_active_mean = terms["contact_force_active_sum"] / jnp.maximum(
        terms["contact_force_active_count"],
        1.0,
    )
    contact_force_active = contact_force_active_mean / _CONTACT_FORCE_SCALE
    contact_force_peak = terms["contact_force_peak_max"]
    contact_force_peak_source = jnp.stack(
        [terms[name] for name in _CONTACT_FORCE_PEAK_SOURCE_COLUMNS],
        axis=-1,
    )
    contact_force_peak_excess_mean = terms["contact_force_peak_excess_sum"] / count
    contact_force_delta = terms["contact_force_delta_sum"] / count
    control_delta = terms["control_delta_sum"] / count
    action_delta = terms["action_delta_sum"] / count
    joint_acc = terms["joint_acc_sum"] / count
    joint_jerk = terms["joint_jerk_sum"] / count
    return {
        "score": score,
        "root_pos_error_mean": root_pos_error,
        "root_rot_error_mean": root_rot_error,
        "joint_pos_error_mean": joint_pos_error,
        "body_global_pos_error_mean": body_global_pos_error,
        "body_global_rot_error_mean": body_global_rot_error,
        "body_local_pos_error_mean": body_local_pos_error,
        "body_local_rot_error_mean": body_local_rot_error,
        "ee_global_pos_error_mean": ee_global_pos_error,
        "ee_global_rot_error_mean": ee_global_rot_error,
        "ee_local_pos_error_mean": ee_local_pos_error,
        "ee_local_rot_error_mean": ee_local_rot_error,
        "hand_global_pos_error_mean": hand_global_pos_error,
        "hand_global_rot_error_mean": hand_global_rot_error,
        "hand_local_pos_error_mean": hand_local_pos_error,
        "hand_local_rot_error_mean": hand_local_rot_error,
        "contact_mismatch_rate": contact_mismatch,
        "contact_false_positive_rate": contact_false_positive,
        "contact_false_negative_rate": contact_false_negative,
        "contact_switch_rate": contact_switch,
        "bad_floor_contact_rate": bad_floor_contact,
        "bad_floor_force_excess_mean": bad_floor_force_excess,
        "contact_force_active_mean": contact_force_active_mean,
        "contact_force_peak": contact_force_peak,
        "contact_force_peak_source": contact_force_peak_source,
        "contact_force_peak_excess_mean": contact_force_peak_excess_mean,
        "contact_force_delta_mean": contact_force_delta,
        "control_delta_mean": control_delta,
        "action_delta_mean": action_delta,
        "joint_acc_mean": joint_acc,
        "joint_jerk_mean": joint_jerk,
        "root_pos_error": root_pos_error,
        "root_rot_error": root_rot_error,
        "joint_pos_error": joint_pos_error,
        "body_global_pos_error": body_global_pos_error,
        "body_global_rot_error": body_global_rot_error,
        "body_local_pos_error": body_local_pos_error,
        "body_local_rot_error": body_local_rot_error,
        "ee_global_pos_error": ee_global_pos_error,
        "ee_global_rot_error": ee_global_rot_error,
        "ee_local_pos_error": ee_local_pos_error,
        "ee_local_rot_error": ee_local_rot_error,
        "hand_global_pos_error": hand_global_pos_error,
        "hand_global_rot_error": hand_global_rot_error,
        "hand_local_pos_error": hand_local_pos_error,
        "hand_local_rot_error": hand_local_rot_error,
        "contact_mismatch": contact_mismatch,
        "contact_false_positive": contact_false_positive,
        "contact_false_negative": contact_false_negative,
        "contact_switch": contact_switch,
        "bad_floor_contact": bad_floor_contact,
        "bad_floor_force_excess": bad_floor_force_excess,
        "contact_force_active": contact_force_active,
        "contact_force_peak_excess": contact_force_peak_excess_mean,
        "contact_force_delta": contact_force_delta,
        "control_delta": control_delta,
        "action_delta": action_delta,
        "joint_acc": joint_acc,
        "joint_jerk": joint_jerk,
        "active_contact_count": terms["active_contact_count"],
        "contact_pair_count": terms["contact_pair_count"],
    }


def _mean_squared(actual, expected, *, batch_shape: tuple[int, ...], jnp):
    delta = jnp.asarray(actual) - jnp.asarray(expected)
    return _mean_feature_axes(delta * delta, batch_shape=batch_shape, jnp=jnp)


def _mean_l2_delta(actual, expected, *, batch_shape: tuple[int, ...], jnp):
    delta = jnp.asarray(actual) - jnp.asarray(expected)
    return _mean_l2(delta, batch_shape=batch_shape, jnp=jnp)


def _mean_l2(value, *, batch_shape: tuple[int, ...], jnp):
    value = jnp.asarray(value)
    norm = jnp.sqrt(jnp.sum(value * value, axis=-1))
    return _mean_feature_axes(norm, batch_shape=batch_shape, jnp=jnp)


def _mean_abs(actual, expected, *, batch_shape: tuple[int, ...], jnp):
    delta = jnp.asarray(actual) - jnp.asarray(expected)
    return _mean_feature_axes(jnp.abs(delta), batch_shape=batch_shape, jnp=jnp)


def _mean_contact_indicator(
    actual,
    expected,
    *,
    positive: bool,
    batch_shape: tuple[int, ...],
    jnp,
):
    actual = jnp.asarray(actual)
    expected = jnp.asarray(expected)
    if positive:
        indicator = (actual > 0.5) & (expected <= 0.5)
    else:
        indicator = (actual <= 0.5) & (expected > 0.5)
    return _mean_feature_axes(jnp.asarray(indicator), batch_shape=batch_shape, jnp=jnp)


def _mean_optional_l2_delta(
    step_state,
    actual_name: str,
    expected_name: str,
    *,
    required: bool,
    batch_shape: tuple[int, ...],
    jnp,
    expected_container=None,
):
    expected_source = step_state if expected_container is None else expected_container
    if actual_name not in step_state or expected_name not in expected_source:
        if required:
            raise KeyError(
                f"Missing score fields {actual_name!r} and/or {expected_name!r}"
            )
        return jnp.zeros(batch_shape)
    delta = jnp.asarray(step_state[actual_name]) - jnp.asarray(
        expected_source[expected_name]
    )
    norm = jnp.sqrt(jnp.sum(delta * delta, axis=-1))
    return _mean_feature_axes(norm, batch_shape=batch_shape, jnp=jnp)


def _mean_indexed_l2_delta(
    step_state,
    reference_state,
    name: str,
    indices: tuple[int, ...],
    *,
    required: bool,
    batch_shape: tuple[int, ...],
    jnp,
):
    if not required:
        return jnp.zeros(batch_shape)
    if name not in step_state or name not in reference_state:
        raise KeyError(f"Missing indexed score field {name!r}")
    take_indices = _take_indices(indices, jnp=jnp)
    actual = jnp.take(jnp.asarray(step_state[name]), take_indices, axis=-2)
    expected = jnp.take(jnp.asarray(reference_state[name]), take_indices, axis=-2)
    return _mean_l2_delta(actual, expected, batch_shape=batch_shape, jnp=jnp)


def _mean_indexed_quat_error(
    step_state,
    reference_state,
    name: str,
    indices: tuple[int, ...],
    *,
    required: bool,
    batch_shape: tuple[int, ...],
    jnp,
):
    if not required:
        return jnp.zeros(batch_shape)
    if name not in step_state or name not in reference_state:
        raise KeyError(f"Missing indexed quaternion score field {name!r}")
    take_indices = _take_indices(indices, jnp=jnp)
    actual = jnp.take(jnp.asarray(step_state[name]), take_indices, axis=-2)
    expected = jnp.take(jnp.asarray(reference_state[name]), take_indices, axis=-2)
    return _mean_quat_values(actual, expected, batch_shape=batch_shape, jnp=jnp)


def _local_body_errors(
    step_state,
    reference_state,
    pos_name: str,
    quat_name: str,
    indices: tuple[int, ...],
    *,
    required: bool,
    batch_shape: tuple[int, ...],
    jnp,
):
    if not required:
        return jnp.zeros(batch_shape), jnp.zeros(batch_shape)
    missing = [
        name
        for name in (pos_name, quat_name)
        if name not in step_state or name not in reference_state
    ]
    if missing:
        raise KeyError(f"Missing local body score fields: {sorted(set(missing))}")
    actual_pos, actual_quat = _body_pose_in_anchor(
        step_state[pos_name],
        step_state[quat_name],
        indices,
        jnp=jnp,
    )
    expected_pos, expected_quat = _body_pose_in_anchor(
        reference_state[pos_name],
        reference_state[quat_name],
        indices,
        jnp=jnp,
    )
    return (
        _mean_l2_delta(actual_pos, expected_pos, batch_shape=batch_shape, jnp=jnp),
        _mean_quat_values(
            actual_quat,
            expected_quat,
            batch_shape=batch_shape,
            jnp=jnp,
        ),
    )


def _body_pose_in_anchor(body_pos, body_quat, indices: tuple[int, ...], *, jnp):
    body_pos = jnp.asarray(body_pos)
    body_quat = jnp.asarray(body_quat)
    take_indices = _take_indices(indices, jnp=jnp)
    target_pos = jnp.take(body_pos, take_indices, axis=-2)
    target_quat = jnp.take(body_quat, take_indices, axis=-2)
    anchor_pos = jnp.expand_dims(jnp.take(body_pos, _ANCHOR_INDEX, axis=-2), axis=-2)
    anchor_quat = jnp.expand_dims(
        jnp.take(body_quat, _ANCHOR_INDEX, axis=-2),
        axis=-2,
    )
    anchor_pos = anchor_pos + jnp.zeros_like(target_pos)
    anchor_quat = anchor_quat + jnp.zeros_like(target_quat)
    inv_anchor = _quat_inv(anchor_quat, jnp=jnp)
    return (
        _quat_apply(inv_anchor, target_pos - anchor_pos, jnp=jnp),
        _quat_mul(inv_anchor, target_quat, jnp=jnp),
    )


def _take_indices(indices: tuple[int, ...], *, jnp):
    try:
        return jnp.asarray(indices, dtype="int32")
    except TypeError:
        return indices


def _mean_optional_feature_tail(
    step_state,
    name: str,
    *,
    start: int,
    required: bool,
    batch_shape: tuple[int, ...],
    jnp,
):
    if name not in step_state:
        if required:
            raise KeyError(f"Missing score field {name!r}")
        return jnp.zeros(batch_shape)
    value = jnp.asarray(step_state[name])
    if int(value.shape[-1]) <= int(start):
        if required:
            raise ValueError(
                f"Expected score field {name!r} to have channel > {int(start)}"
            )
        return jnp.zeros(batch_shape)
    return _mean_feature_axes(value[..., int(start) :], batch_shape=batch_shape, jnp=jnp)


def _mean_optional_force_excess(
    step_state,
    name: str,
    *,
    start: int,
    required: bool,
    batch_shape: tuple[int, ...],
    jnp,
):
    if name not in step_state:
        if required:
            raise KeyError(f"Missing score field {name!r}")
        return jnp.zeros(batch_shape)
    value = jnp.asarray(step_state[name])
    if int(value.shape[-1]) <= int(start):
        if required:
            raise ValueError(
                f"Expected score field {name!r} to have channel > {int(start)}"
            )
        return jnp.zeros(batch_shape)
    excess = jnp.maximum(value[..., int(start) :] - _CONTACT_FORCE_SCALE, 0.0)
    return _mean_feature_axes(
        excess / _CONTACT_FORCE_SCALE,
        batch_shape=batch_shape,
        jnp=jnp,
    )


def _active_contact_force_stats(
    step_state,
    *,
    required: bool,
    batch_shape: tuple[int, ...],
    jnp,
):
    if "contact_force" not in step_state:
        if required:
            raise KeyError("Missing score field 'contact_force'")
        return jnp.zeros(batch_shape), jnp.zeros(batch_shape)
    if "contact" not in step_state:
        if required:
            raise KeyError("Missing score field 'contact'")
        return jnp.zeros(batch_shape), jnp.zeros(batch_shape)
    contact = jnp.asarray(step_state["contact"])
    contact_force = jnp.asarray(step_state["contact_force"])
    _validate_batch_prefix(contact, batch_shape)
    _validate_batch_prefix(contact_force, batch_shape)
    if tuple(int(dim) for dim in contact.shape) != tuple(
        int(dim) for dim in contact_force.shape
    ):
        raise ValueError(
            "Expected score fields 'contact' and 'contact_force' to have matching shapes"
        )
    active = jnp.asarray(contact > 0.5)
    reduce_axes = tuple(range(len(batch_shape), len(contact_force.shape)))
    if reduce_axes:
        return (
            jnp.sum(contact_force * active, axis=reduce_axes),
            jnp.sum(active, axis=reduce_axes),
        )
    return contact_force * active, active


def _contact_force_peak_excess(
    step_state,
    *,
    required: bool,
    batch_shape: tuple[int, ...],
    jnp,
):
    if "contact_force" not in step_state:
        if required:
            raise KeyError("Missing score field 'contact_force'")
        return jnp.zeros(batch_shape), jnp.zeros(batch_shape)
    contact_force = jnp.asarray(step_state["contact_force"])
    _validate_batch_prefix(contact_force, batch_shape)
    reduce_axes = tuple(range(len(batch_shape), len(contact_force.shape)))
    if reduce_axes:
        peak = jnp.max(contact_force, axis=reduce_axes)
    else:
        peak = contact_force
    excess = jnp.maximum(peak - _CONTACT_FORCE_SCALE, 0.0) / _CONTACT_FORCE_SCALE
    return peak, excess


def _contact_force_peak_source(step_state, *, batch_shape: tuple[int, ...], jnp):
    default = _default_contact_force_peak_source(batch_shape, jnp=jnp)
    if (
        "contact_force" not in step_state
        or "floor_contact_force_peak_source" not in step_state
    ):
        return default
    contact_force = jnp.asarray(step_state["contact_force"])
    source = jnp.asarray(step_state["floor_contact_force_peak_source"])
    _validate_batch_prefix(contact_force, batch_shape)
    _validate_batch_prefix(source, batch_shape)
    if len(contact_force.shape) != len(batch_shape) + 1:
        return default
    if len(source.shape) != len(batch_shape) + 2:
        return default
    if int(source.shape[-1]) != _CONTACT_FORCE_PEAK_SOURCE_WIDTH:
        return default
    if int(source.shape[-2]) < int(contact_force.shape[-1]):
        return default
    peak_index = jnp.argmax(contact_force, axis=-1)
    group_ids = jnp.arange(int(source.shape[-2]))
    selector = group_ids == jnp.expand_dims(peak_index, axis=-1)
    return jnp.sum(source * jnp.expand_dims(selector, axis=-1), axis=-2)


def _default_contact_force_peak_source(batch_shape: tuple[int, ...], *, jnp):
    return jnp.zeros((*batch_shape, _CONTACT_FORCE_PEAK_SOURCE_WIDTH)) + jnp.asarray(
        _CONTACT_FORCE_PEAK_SOURCE_DEFAULTS
    )


def _mean_quat_error(
    step_state,
    reference_state,
    name: str,
    *,
    required: bool,
    batch_shape: tuple[int, ...],
    jnp,
):
    if name not in step_state or name not in reference_state:
        if required:
            raise KeyError(f"Missing quaternion score field {name!r}")
        return jnp.zeros(batch_shape)
    return _mean_quat_values(
        step_state[name],
        reference_state[name],
        batch_shape=batch_shape,
        jnp=jnp,
    )


def _mean_quat_values(actual, expected, *, batch_shape: tuple[int, ...], jnp):
    actual = _normalize_quat(actual, jnp=jnp)
    expected = _normalize_quat(expected, jnp=jnp)
    dot = jnp.sum(actual * expected, axis=-1)
    angle = 2.0 * jnp.arccos(jnp.clip(jnp.abs(dot), -1.0, 1.0))
    return _mean_feature_axes(angle, batch_shape=batch_shape, jnp=jnp)


def _normalize_quat(value, *, jnp):
    value = jnp.asarray(value)
    norm = jnp.sqrt(jnp.maximum(jnp.sum(value * value, axis=-1, keepdims=True), 1.0e-12))
    return value / norm


def _quat_inv(quat, *, jnp):
    quat = jnp.asarray(quat)
    conj = jnp.concatenate([quat[..., :1], -quat[..., 1:]], axis=-1)
    denom = jnp.maximum(jnp.sum(quat * quat, axis=-1, keepdims=True), 1.0e-12)
    return conj / denom


def _quat_mul(q1, q2, *, jnp):
    q1 = jnp.asarray(q1)
    q2 = jnp.asarray(q2)
    w1, x1, y1, z1 = [q1[..., index] for index in range(4)]
    w2, x2, y2, z2 = [q2[..., index] for index in range(4)]
    return jnp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _quat_apply(quat, vec, *, jnp):
    quat = jnp.asarray(quat)
    vec = jnp.asarray(vec)
    xyz = quat[..., 1:]
    t = jnp.cross(xyz, vec, axis=-1) * 2.0
    return vec + quat[..., :1] * t + jnp.cross(xyz, t, axis=-1)


def _weight(weights: JaxScoreWeights, *names: str) -> float:
    for name in names:
        if name in weights.terms:
            return weights.terms[name]
    return 0.0


def _mean_feature_axes(value, *, batch_shape: tuple[int, ...], jnp):
    value = jnp.asarray(value)
    _validate_batch_prefix(value, batch_shape)
    reduce_axes = tuple(range(len(batch_shape), len(value.shape)))
    if not reduce_axes:
        return value
    if len(reduce_axes) == len(value.shape):
        return jnp.mean(value)
    return jnp.mean(value, axis=reduce_axes)


def _diagnostic_value(step_state, name: str, batch_shape: tuple[int, ...], *, jnp):
    value = step_state.get(name)
    if value is None:
        return jnp.zeros(batch_shape)
    value = jnp.asarray(value)
    if not value.shape:
        return value + jnp.zeros(batch_shape)
    _validate_batch_prefix(value, batch_shape)
    reduce_axes = tuple(range(len(batch_shape), len(value.shape)))
    if reduce_axes:
        return jnp.max(value, axis=reduce_axes)
    return value


def _validity_value(step_state, name: str, batch_shape: tuple[int, ...], *, jnp):
    if name not in step_state:
        return jnp.zeros(batch_shape) + 1.0
    return _diagnostic_value(step_state, name, batch_shape, jnp=jnp)


def _where(condition, true_value, false_value, *, jnp):
    where = getattr(jnp, "where", None)
    if callable(where):
        return where(condition, true_value, false_value)
    return true_value if bool(condition) else false_value


def _require_accumulator(accumulator):
    if not accumulator:
        raise KeyError("score_step requires init_score_accumulator output")
    missing = [name for name in ACCUMULATOR_KEYS if name not in accumulator]
    if missing:
        raise KeyError(f"Missing score accumulator keys: {missing}")
    return dict(accumulator)


def _empty_or_valid_accumulator(accumulator, *, jnp):
    if not accumulator:
        return init_score_accumulator((), jnp=jnp)
    return _require_accumulator(accumulator)


def _leading_shape(value, *, jnp) -> tuple[int, ...]:
    arr = jnp.asarray(value)
    if len(arr.shape) <= 1:
        return ()
    return tuple(int(dim) for dim in arr.shape[:-1])


def _validate_batch_prefix(value, batch_shape: tuple[int, ...]) -> None:
    actual = tuple(int(dim) for dim in value.shape[: len(batch_shape)])
    if actual != batch_shape:
        raise ValueError(f"Expected batch shape {batch_shape}, got prefix {actual}")


def _shape_tuple(value) -> tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(dim) for dim in value)


__all__ = [
    "ACCUMULATOR_KEYS",
    "JaxScoreWeights",
    "finalize_score",
    "finalize_score_only",
    "init_score_accumulator",
    "score_step",
]
