"""JAX-shaped scoring helpers for G1 WBC MJX rollouts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JaxScoreWeights:
    terms: dict[str, float]


def score_step(accumulator, step_state, reference_state, weights: JaxScoreWeights, *, jnp):
    """Accumulate one step of core WBC rollout score terms."""

    terms = dict(accumulator)
    root_error = _mean_squared(step_state["root_pos"], reference_state["root_pos"], jnp=jnp)
    body_error = _mean_squared(step_state["body_pos"], reference_state["body_pos"], jnp=jnp)
    ee_error = _mean_squared(step_state["ee_pos"], reference_state["ee_pos"], jnp=jnp)
    contact_error = _mean_abs(step_state["contact"], reference_state["contact"], jnp=jnp)
    control_delta = _mean_squared(
        step_state["control"], step_state["prev_control"], jnp=jnp
    )
    joint_acc = _mean_squared(
        step_state["joint_vel"], step_state["prev_joint_vel"], jnp=jnp
    )

    terms["root_pos_error_sum"] = terms.get("root_pos_error_sum", 0.0) + root_error
    terms["body_global_pos_error_sum"] = (
        terms.get("body_global_pos_error_sum", 0.0) + body_error
    )
    terms["ee_global_pos_error_sum"] = (
        terms.get("ee_global_pos_error_sum", 0.0) + ee_error
    )
    terms["contact_mismatch_sum"] = (
        terms.get("contact_mismatch_sum", 0.0) + contact_error
    )
    terms["control_delta_sum"] = terms.get("control_delta_sum", 0.0) + control_delta
    terms["joint_acc_sum"] = terms.get("joint_acc_sum", 0.0) + joint_acc
    terms["count"] = terms.get("count", 0.0) + 1.0

    penalty = (
        _weight(weights, "root_pos", "root_pos_error") * root_error
        + _weight(weights, "body_global_pos", "body_global_pos_error") * body_error
        + _weight(weights, "ee_global_pos", "ee_global_pos_error") * ee_error
        + _weight(weights, "contact", "contact_mismatch") * contact_error
        + _weight(weights, "control_delta") * control_delta
        + _weight(weights, "joint_acc") * joint_acc
    )
    terms["score_sum"] = terms.get("score_sum", 0.0) - penalty
    return terms


def finalize_score(accumulator, *, jnp):
    """Return mean score metrics from a streaming score accumulator."""

    count = jnp.maximum(accumulator.get("count", 0.0), 1.0)
    score = accumulator.get("score_sum", 0.0) / count
    root_pos_error = accumulator.get("root_pos_error_sum", 0.0) / count
    body_global_pos_error = accumulator.get("body_global_pos_error_sum", 0.0) / count
    ee_global_pos_error = accumulator.get("ee_global_pos_error_sum", 0.0) / count
    contact_mismatch = accumulator.get("contact_mismatch_sum", 0.0) / count
    control_delta = accumulator.get("control_delta_sum", 0.0) / count
    joint_acc = accumulator.get("joint_acc_sum", 0.0) / count
    return {
        "score": score,
        "root_pos_error_mean": root_pos_error,
        "body_global_pos_error_mean": body_global_pos_error,
        "ee_global_pos_error_mean": ee_global_pos_error,
        "contact_mismatch_rate": contact_mismatch,
        "control_delta_mean": control_delta,
        "joint_acc_mean": joint_acc,
        "root_pos_error": root_pos_error,
        "body_global_pos_error": body_global_pos_error,
        "ee_global_pos_error": ee_global_pos_error,
        "contact_mismatch": contact_mismatch,
        "control_delta": control_delta,
        "joint_acc": joint_acc,
    }


def _mean_squared(actual, expected, *, jnp):
    delta = jnp.asarray(actual) - jnp.asarray(expected)
    return jnp.mean(delta * delta)


def _mean_abs(actual, expected, *, jnp):
    delta = jnp.asarray(actual) - jnp.asarray(expected)
    return jnp.mean(jnp.abs(delta))


def _weight(weights: JaxScoreWeights, *names: str) -> float:
    for name in names:
        if name in weights.terms:
            return float(weights.terms[name])
    return 0.0


__all__ = [
    "JaxScoreWeights",
    "finalize_score",
    "score_step",
]
