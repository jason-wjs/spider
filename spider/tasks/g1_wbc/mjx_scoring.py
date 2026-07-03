"""JAX-shaped scoring helpers for G1 WBC MJX rollouts."""

from __future__ import annotations

from dataclasses import dataclass

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
    "body_global_pos_error_sum",
    "body_global_rot_error_sum",
    "ee_global_pos_error_sum",
    "ee_global_rot_error_sum",
    "contact_mismatch_sum",
    "contact_false_positive_sum",
    "contact_false_negative_sum",
    "control_delta_sum",
    "action_delta_sum",
    "joint_acc_sum",
    "count",
    "active_contact_count",
    "contact_pair_count",
)


def init_score_accumulator(batch_shape=(), *, jnp):
    """Create a stable score accumulator PyTree for JAX scan carries."""

    shape = _shape_tuple(batch_shape)
    return {name: jnp.zeros(shape) for name in ACCUMULATOR_KEYS}


def score_step(accumulator, step_state, reference_state, weights: JaxScoreWeights, *, jnp):
    """Accumulate one step of core WBC rollout score terms."""

    terms = _require_accumulator(accumulator)
    batch_shape = _shape_tuple(terms["count"].shape)
    root_error = _mean_squared(
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
    body_error = _mean_squared(
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
    ee_error = _mean_squared(
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
    control_delta = _mean_squared(
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
    joint_acc = _mean_squared(
        step_state["joint_vel"],
        step_state["prev_joint_vel"],
        batch_shape=batch_shape,
        jnp=jnp,
    )

    terms["root_pos_error_sum"] = terms["root_pos_error_sum"] + root_error
    terms["root_rot_error_sum"] = terms["root_rot_error_sum"] + root_rot_error
    terms["body_global_pos_error_sum"] = terms["body_global_pos_error_sum"] + body_error
    terms["body_global_rot_error_sum"] = (
        terms["body_global_rot_error_sum"] + body_rot_error
    )
    terms["ee_global_pos_error_sum"] = terms["ee_global_pos_error_sum"] + ee_error
    terms["ee_global_rot_error_sum"] = terms["ee_global_rot_error_sum"] + ee_rot_error
    terms["contact_mismatch_sum"] = terms["contact_mismatch_sum"] + contact_error
    terms["contact_false_positive_sum"] = (
        terms["contact_false_positive_sum"] + contact_false_positive
    )
    terms["contact_false_negative_sum"] = (
        terms["contact_false_negative_sum"] + contact_false_negative
    )
    terms["control_delta_sum"] = terms["control_delta_sum"] + control_delta
    terms["action_delta_sum"] = terms["action_delta_sum"] + action_delta
    terms["joint_acc_sum"] = terms["joint_acc_sum"] + joint_acc
    terms["count"] = terms["count"] + 1.0
    terms["active_contact_count"] = jnp.maximum(
        terms["active_contact_count"],
        _diagnostic_value(step_state, "active_contact_count", batch_shape, jnp=jnp),
    )
    terms["contact_pair_count"] = jnp.maximum(
        terms["contact_pair_count"],
        _diagnostic_value(step_state, "contact_pair_count", batch_shape, jnp=jnp),
    )

    penalty = (
        _weight(weights, "root_pos", "root_pos_error") * root_error
        + _weight(weights, "root_rot", "root_rot_error") * root_rot_error
        + _weight(weights, "body_global_pos", "body_global_pos_error") * body_error
        + _weight(weights, "body_global_rot", "body_global_rot_error")
        * body_rot_error
        + _weight(weights, "ee_global_pos", "ee_global_pos_error") * ee_error
        + _weight(weights, "ee_global_rot", "ee_global_rot_error") * ee_rot_error
        + _weight(weights, "contact", "contact_mismatch") * contact_error
        + _weight(weights, "contact_false_positive") * contact_false_positive
        + _weight(weights, "contact_false_negative") * contact_false_negative
        + _weight(weights, "control_delta") * control_delta
        + _weight(weights, "action_delta") * action_delta
        + _weight(weights, "joint_acc") * joint_acc
    )
    terms["score_sum"] = terms["score_sum"] - penalty
    return terms


def finalize_score(accumulator, *, jnp):
    """Return mean score metrics from a streaming score accumulator."""

    terms = _empty_or_valid_accumulator(accumulator, jnp=jnp)
    count = jnp.maximum(terms["count"], 1.0)
    score = terms["score_sum"] / count
    root_pos_error = terms["root_pos_error_sum"] / count
    root_rot_error = terms["root_rot_error_sum"] / count
    body_global_pos_error = terms["body_global_pos_error_sum"] / count
    body_global_rot_error = terms["body_global_rot_error_sum"] / count
    ee_global_pos_error = terms["ee_global_pos_error_sum"] / count
    ee_global_rot_error = terms["ee_global_rot_error_sum"] / count
    contact_mismatch = terms["contact_mismatch_sum"] / count
    contact_false_positive = terms["contact_false_positive_sum"] / count
    contact_false_negative = terms["contact_false_negative_sum"] / count
    control_delta = terms["control_delta_sum"] / count
    action_delta = terms["action_delta_sum"] / count
    joint_acc = terms["joint_acc_sum"] / count
    return {
        "score": score,
        "root_pos_error_mean": root_pos_error,
        "root_rot_error_mean": root_rot_error,
        "body_global_pos_error_mean": body_global_pos_error,
        "body_global_rot_error_mean": body_global_rot_error,
        "ee_global_pos_error_mean": ee_global_pos_error,
        "ee_global_rot_error_mean": ee_global_rot_error,
        "contact_mismatch_rate": contact_mismatch,
        "contact_false_positive_rate": contact_false_positive,
        "contact_false_negative_rate": contact_false_negative,
        "control_delta_mean": control_delta,
        "action_delta_mean": action_delta,
        "joint_acc_mean": joint_acc,
        "root_pos_error": root_pos_error,
        "root_rot_error": root_rot_error,
        "body_global_pos_error": body_global_pos_error,
        "body_global_rot_error": body_global_rot_error,
        "ee_global_pos_error": ee_global_pos_error,
        "ee_global_rot_error": ee_global_rot_error,
        "contact_mismatch": contact_mismatch,
        "contact_false_positive": contact_false_positive,
        "contact_false_negative": contact_false_negative,
        "control_delta": control_delta,
        "action_delta": action_delta,
        "joint_acc": joint_acc,
        "active_contact_count": terms["active_contact_count"],
        "contact_pair_count": terms["contact_pair_count"],
    }


def _mean_squared(actual, expected, *, batch_shape: tuple[int, ...], jnp):
    delta = jnp.asarray(actual) - jnp.asarray(expected)
    return _mean_feature_axes(delta * delta, batch_shape=batch_shape, jnp=jnp)


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
):
    if actual_name not in step_state or expected_name not in step_state:
        if required:
            raise KeyError(
                f"Missing score fields {actual_name!r} and/or {expected_name!r}"
            )
        return jnp.zeros(batch_shape)
    delta = jnp.asarray(step_state[actual_name]) - jnp.asarray(step_state[expected_name])
    norm = jnp.sqrt(jnp.sum(delta * delta, axis=-1))
    return _mean_feature_axes(norm, batch_shape=batch_shape, jnp=jnp)


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
    actual = _normalize_quat(step_state[name], jnp=jnp)
    expected = _normalize_quat(reference_state[name], jnp=jnp)
    dot = jnp.sum(actual * expected, axis=-1)
    angle = 2.0 * jnp.arccos(jnp.clip(jnp.abs(dot), -1.0, 1.0))
    return _mean_feature_axes(angle, batch_shape=batch_shape, jnp=jnp)


def _normalize_quat(value, *, jnp):
    value = jnp.asarray(value)
    norm = jnp.sqrt(jnp.maximum(jnp.sum(value * value, axis=-1, keepdims=True), 1.0e-12))
    return value / norm


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
    "init_score_accumulator",
    "score_step",
]
