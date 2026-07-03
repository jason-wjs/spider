"""Scan-shaped rollout scoring helpers for the G1 WBC MJX backend."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from spider.tasks.g1_wbc.constants import ACTION_DIM, POLICY_DT, QPOS_DIM, QVEL_DIM
from spider.tasks.g1_wbc.mjx_obs import (
    JaxObsIndices,
    JaxObsState,
    OBS_FIELD_SPECS,
    build_wbc_observation_from_state,
)
from spider.tasks.g1_wbc.mjx_policy import jax_actor_forward
from spider.tasks.g1_wbc.mjx_scoring import (
    JaxScoreWeights,
    finalize_score,
    init_score_accumulator,
    score_step,
)


PhysicsStepFn = Callable[..., tuple[Mapping[str, object], Mapping[str, object]]]
CommandReferenceFn = Callable[..., Mapping[str, object]]


def make_rollout_scorer(
    *,
    runtime,
    physics_step_fn: PhysicsStepFn,
    command_reference_fn: CommandReferenceFn | None = None,
):
    """Bind runtime dependencies into the optimizer's four-argument scorer."""

    jitted_by_model_id: dict[int, Callable[..., object]] = {}

    def rollout_fn(samples, reference, actor_params, model_bundle):
        jit = getattr(getattr(runtime, "jax", None), "jit", None)
        if jit is None:
            return score_candidate_controls(
                samples,
                reference,
                actor_params,
                model_bundle,
                runtime=runtime,
                physics_step_fn=physics_step_fn,
                command_reference_fn=command_reference_fn,
                return_metrics=True,
            )
        model_key = id(model_bundle)
        compiled = jitted_by_model_id.get(model_key)
        if compiled is None:

            def score_for_model(samples, reference, actor_params):
                return score_candidate_controls(
                    samples,
                    reference,
                    actor_params,
                    model_bundle,
                    runtime=runtime,
                    physics_step_fn=physics_step_fn,
                    command_reference_fn=command_reference_fn,
                    return_metrics=True,
                )

            compiled = jit(score_for_model)
            jitted_by_model_id[model_key] = compiled
        return compiled(samples, reference, actor_params)

    return rollout_fn


def controls_to_qpos(controls, base_qpos, joint_low, joint_high, *, jnp):
    """Apply SPIDER residual controls to a base qpos trajectory."""

    controls = jnp.asarray(controls)
    if len(controls.shape) != 3:
        raise ValueError(
            f"Expected controls shape (samples, horizon, width), got {controls.shape}"
        )
    if int(controls.shape[-1]) != QPOS_DIM - 1:
        raise ValueError(
            f"Expected control width {QPOS_DIM - 1}, got {int(controls.shape[-1])}"
        )

    samples = int(controls.shape[0])
    horizon = int(controls.shape[1])
    base = _base_qpos_trajectory(
        jnp.asarray(base_qpos),
        sample_count=samples,
        horizon=horizon,
        jnp=jnp,
    )
    root_pos = base[..., :3] + controls[..., :3]
    delta_quat = _quat_from_axis_angle(controls[..., 3:6], jnp=jnp)
    root_quat = _normalize(
        _quat_mul(delta_quat, base[..., 3:7], jnp=jnp),
        jnp=jnp,
    )
    joints = base[..., 7:] + controls[..., 6:]
    joints = jnp.clip(
        joints,
        _joint_limit_array(joint_low, jnp=jnp),
        _joint_limit_array(joint_high, jnp=jnp),
    )
    return jnp.concatenate([root_pos, root_quat, joints], axis=-1)


def _base_qpos_trajectory(base_qpos, *, sample_count: int, horizon: int, jnp):
    if int(base_qpos.shape[-1]) != QPOS_DIM:
        raise ValueError(f"Expected base qpos width {QPOS_DIM}, got {base_qpos.shape}")
    rank = len(base_qpos.shape)
    if rank == 1:
        base = jnp.repeat(jnp.expand_dims(base_qpos, axis=0), sample_count, axis=0)
        return jnp.repeat(jnp.expand_dims(base, axis=1), horizon, axis=1)
    if rank == 2:
        first = int(base_qpos.shape[0])
        if first == int(sample_count):
            return jnp.repeat(jnp.expand_dims(base_qpos, axis=1), horizon, axis=1)
        if first == int(horizon):
            return jnp.repeat(jnp.expand_dims(base_qpos, axis=0), sample_count, axis=0)
    if rank == 3 and tuple(int(dim) for dim in base_qpos.shape[:2]) == (
        int(sample_count),
        int(horizon),
    ):
        return base_qpos
    raise ValueError(
        "Expected base qpos shape "
        f"({QPOS_DIM},), ({sample_count}, {QPOS_DIM}), "
        f"({horizon}, {QPOS_DIM}), or "
        f"({sample_count}, {horizon}, {QPOS_DIM}); got {base_qpos.shape}"
    )


def _reference_base_qpos(base_qpos, *, sample_count: int, horizon: int, jnp):
    base_qpos = jnp.asarray(base_qpos)
    if len(base_qpos.shape) == 2 and int(base_qpos.shape[0]) == int(horizon):
        return jnp.repeat(jnp.expand_dims(base_qpos, axis=0), sample_count, axis=0)
    return base_qpos


def score_candidate_controls(
    samples,
    reference: Mapping[str, object],
    actor_params,
    model_bundle,
    *,
    runtime,
    physics_step_fn: PhysicsStepFn,
    command_reference_fn: CommandReferenceFn | None = None,
    return_metrics: bool = False,
):
    """Score sampled high-level controls with a scan-compatible rollout loop."""

    jnp = runtime.jnp
    samples = jnp.asarray(samples)
    _validate_samples(samples)
    sample_count = int(samples.shape[0])
    horizon = int(samples.shape[1])

    robot_state = _batched_robot_state(
        _required(reference, "initial_robot_state"),
        sample_count,
        jnp=jnp,
    )
    obs_state = _initial_obs_state(reference, sample_count, jnp=jnp)
    obs_indices = _required(reference, "obs_indices")
    if not isinstance(obs_indices, JaxObsIndices):
        raise TypeError("reference['obs_indices'] must be a JaxObsIndices")
    default_joint_pos = _required(reference, "default_joint_pos")
    base_qpos = (
        _reference_base_qpos(
            reference["base_qpos"],
            sample_count=sample_count,
            horizon=horizon,
            jnp=jnp,
        )
        if "base_qpos" in reference
        else robot_state["qpos"]
    )
    commanded_qpos = controls_to_qpos(
        samples,
        base_qpos,
        _required(reference, "joint_low"),
        _required(reference, "joint_high"),
        jnp=jnp,
    )
    commanded_qvel = _commanded_qvel(commanded_qpos, jnp=jnp)
    commanded_joint_vel = commanded_qvel[..., 6:]
    command_reference = _command_reference(
        command_reference_fn,
        model_bundle,
        commanded_qpos,
        commanded_qvel,
        runtime=runtime,
        jnp=jnp,
    )
    prev_control = _ensure_batch(
        jnp.asarray(
            reference.get("prev_control", jnp.zeros((sample_count, QPOS_DIM - 1)))
        ),
        sample_count,
        jnp=jnp,
    )
    prev_joint_vel = _joint_vel(robot_state)
    weights = reference.get("score_weights", JaxScoreWeights({}))
    if not isinstance(weights, JaxScoreWeights):
        weights = JaxScoreWeights(dict(weights))
    accumulator = init_score_accumulator((sample_count,), jnp=jnp)
    obs_initialized = reference.get("obs_initialized", False)
    step_reference = _with_commanded_qpos(
        reference,
        commanded_qpos,
        commanded_joint_vel,
        command_reference,
    )

    scan = _lax_scan(runtime)
    if scan is not None:
        obs_state = _materialized_obs_state(obs_state, sample_count, jnp=jnp)
        carry = (
            robot_state,
            obs_state.history,
            obs_state.last_action,
            prev_control,
            prev_joint_vel,
            accumulator,
        )

        def scan_step(carry, step_index):
            (
                robot_state,
                obs_history,
                last_action,
                prev_control,
                prev_joint_vel,
                accumulator,
            ) = carry
            next_values = _score_rollout_step(
                step_index,
                robot_state=robot_state,
                obs_state=JaxObsState(history=obs_history, last_action=last_action),
                prev_control=prev_control,
                prev_joint_vel=prev_joint_vel,
                accumulator=accumulator,
                samples=samples,
                reference=step_reference,
                obs_indices=obs_indices,
                default_joint_pos=default_joint_pos,
                actor_params=actor_params,
                model_bundle=model_bundle,
                weights=weights,
                obs_initialized=_step_obs_initialized(
                    obs_initialized,
                    step_index,
                    jnp=jnp,
                ),
                physics_step_fn=physics_step_fn,
                runtime=runtime,
                sample_count=sample_count,
            )
            return (
                next_values["robot_state"],
                next_values["obs_state"].history,
                next_values["obs_state"].last_action,
                next_values["prev_control"],
                next_values["prev_joint_vel"],
                next_values["accumulator"],
            ), None

        carry, _ = scan(scan_step, carry, jnp.arange(horizon))
        accumulator = carry[-1]
        metrics = finalize_score(accumulator, jnp=jnp)
        if return_metrics:
            return _with_physics_step_count(
                metrics,
                sample_count=sample_count,
                horizon=horizon,
                jnp=jnp,
            )
        return metrics["score"]

    for step_index in range(horizon):
        next_values = _score_rollout_step(
            step_index,
            robot_state=robot_state,
            obs_state=obs_state,
            prev_control=prev_control,
            prev_joint_vel=prev_joint_vel,
            accumulator=accumulator,
            samples=samples,
            reference=step_reference,
            obs_indices=obs_indices,
            default_joint_pos=default_joint_pos,
            actor_params=actor_params,
            model_bundle=model_bundle,
            weights=weights,
            obs_initialized=obs_initialized if step_index == 0 else True,
            physics_step_fn=physics_step_fn,
            runtime=runtime,
            sample_count=sample_count,
        )
        robot_state = next_values["robot_state"]
        obs_state = next_values["obs_state"]
        prev_control = next_values["prev_control"]
        prev_joint_vel = next_values["prev_joint_vel"]
        accumulator = next_values["accumulator"]

    metrics = finalize_score(accumulator, jnp=jnp)
    if return_metrics:
        return _with_physics_step_count(
            metrics,
            sample_count=sample_count,
            horizon=horizon,
            jnp=jnp,
        )
    return metrics["score"]


def rollout_candidate_controls(
    samples,
    reference: Mapping[str, object],
    actor_params,
    model_bundle,
    *,
    runtime,
    physics_step_fn: PhysicsStepFn,
    command_reference_fn: CommandReferenceFn | None = None,
) -> dict[str, object]:
    """Roll out sampled high-level controls and return robot-state traces."""

    jnp = runtime.jnp
    samples = jnp.asarray(samples)
    _validate_samples(samples)
    sample_count = int(samples.shape[0])
    horizon = int(samples.shape[1])

    robot_state = _batched_robot_state(
        _required(reference, "initial_robot_state"),
        sample_count,
        jnp=jnp,
    )
    obs_state = _initial_obs_state(reference, sample_count, jnp=jnp)
    obs_indices = _required(reference, "obs_indices")
    if not isinstance(obs_indices, JaxObsIndices):
        raise TypeError("reference['obs_indices'] must be a JaxObsIndices")
    default_joint_pos = _required(reference, "default_joint_pos")
    base_qpos = (
        _reference_base_qpos(
            reference["base_qpos"],
            sample_count=sample_count,
            horizon=horizon,
            jnp=jnp,
        )
        if "base_qpos" in reference
        else robot_state["qpos"]
    )
    commanded_qpos = controls_to_qpos(
        samples,
        base_qpos,
        _required(reference, "joint_low"),
        _required(reference, "joint_high"),
        jnp=jnp,
    )
    commanded_qvel = _commanded_qvel(commanded_qpos, jnp=jnp)
    commanded_joint_vel = commanded_qvel[..., 6:]
    command_reference = _command_reference(
        command_reference_fn,
        model_bundle,
        commanded_qpos,
        commanded_qvel,
        runtime=runtime,
        jnp=jnp,
    )
    prev_control = _ensure_batch(
        jnp.asarray(
            reference.get("prev_control", jnp.zeros((sample_count, QPOS_DIM - 1)))
        ),
        sample_count,
        jnp=jnp,
    )
    prev_joint_vel = _joint_vel(robot_state)
    obs_initialized = reference.get("obs_initialized", False)
    step_reference = _with_commanded_qpos(
        reference,
        commanded_qpos,
        commanded_joint_vel,
        command_reference,
    )

    qpos_trace = [robot_state["qpos"]]
    qvel_trace = [robot_state["qvel"]]
    for step_index in range(horizon):
        next_values = _rollout_trace_step(
            step_index,
            robot_state=robot_state,
            obs_state=obs_state,
            prev_control=prev_control,
            prev_joint_vel=prev_joint_vel,
            samples=samples,
            reference=step_reference,
            obs_indices=obs_indices,
            default_joint_pos=default_joint_pos,
            actor_params=actor_params,
            model_bundle=model_bundle,
            obs_initialized=obs_initialized if step_index == 0 else True,
            physics_step_fn=physics_step_fn,
            runtime=runtime,
            sample_count=sample_count,
        )
        robot_state = next_values["robot_state"]
        obs_state = next_values["obs_state"]
        prev_control = next_values["prev_control"]
        prev_joint_vel = next_values["prev_joint_vel"]
        qpos_trace.append(robot_state["qpos"])
        qvel_trace.append(robot_state["qvel"])

    return {
        "qpos": jnp.stack(qpos_trace, axis=0),
        "qvel": jnp.stack(qvel_trace, axis=0),
        "final_robot_state": robot_state,
        "final_obs_state": obs_state,
        "final_prev_control": prev_control,
        "final_prev_joint_vel": prev_joint_vel,
    }


def _with_physics_step_count(
    metrics: Mapping[str, object],
    *,
    sample_count: int,
    horizon: int,
    jnp,
) -> dict[str, object]:
    enriched = dict(metrics)
    enriched["physics_step_count"] = jnp.full(
        (int(sample_count),),
        int(horizon),
    )
    return enriched


def _score_rollout_step(
    step_index,
    *,
    robot_state,
    obs_state: JaxObsState,
    prev_control,
    prev_joint_vel,
    accumulator,
    samples,
    reference: Mapping[str, object],
    obs_indices: JaxObsIndices,
    default_joint_pos,
    actor_params,
    model_bundle,
    weights: JaxScoreWeights,
    obs_initialized,
    physics_step_fn: PhysicsStepFn,
    runtime,
    sample_count: int,
) -> dict[str, object]:
    jnp = runtime.jnp
    commanded_qpos = _required(reference, "commanded_qpos")
    command_qpos = commanded_qpos[:, step_index]
    commanded_joint_vel = _required(reference, "commanded_joint_vel")
    command_joint_vel = commanded_joint_vel[:, step_index]
    command_reference = _time_slice_command_reference(
        reference.get("command_reference", {}),
        step_index,
        jnp=jnp,
    )
    obs_reference = _time_slice_reference(
        _required(reference, "obs_reference"),
        step_index,
        sample_count,
        jnp=jnp,
    )
    obs_reference = _with_commanded_joint_reference(
        obs_reference,
        command_qpos,
        command_joint_vel,
        command_reference,
    )
    obs, next_obs_state = build_wbc_observation_from_state(
        robot_state=robot_state,
        reference_state=obs_reference,
        obs_state=obs_state,
        indices=obs_indices,
        default_joint_pos=default_joint_pos,
        initialized=obs_initialized,
        jnp=jnp,
    )
    action = jax_actor_forward(actor_params, obs, jnp=jnp)
    robot_state, physics_score_state = physics_step_fn(
        model_bundle,
        robot_state,
        command_qpos,
        action,
        step_index,
        runtime=runtime,
    )
    robot_state = _batched_robot_state(robot_state, sample_count, jnp=jnp)
    step_control = samples[:, step_index]
    step_state = dict(physics_score_state)
    step_state.setdefault("control", step_control)
    step_state.setdefault("prev_control", prev_control)
    step_state.setdefault("joint_vel", _joint_vel(robot_state))
    step_state.setdefault("prev_joint_vel", prev_joint_vel)
    score_reference = _time_slice_reference(
        _required(reference, "score_reference"),
        step_index,
        sample_count,
        jnp=jnp,
    )
    accumulator = score_step(
        accumulator,
        step_state,
        score_reference,
        weights,
        jnp=jnp,
    )
    return {
        "robot_state": robot_state,
        "obs_state": JaxObsState(history=next_obs_state.history, last_action=action),
        "prev_control": step_control,
        "prev_joint_vel": _joint_vel(robot_state),
        "accumulator": accumulator,
    }


def _rollout_trace_step(
    step_index,
    *,
    robot_state,
    obs_state: JaxObsState,
    prev_control,
    prev_joint_vel,
    samples,
    reference: Mapping[str, object],
    obs_indices: JaxObsIndices,
    default_joint_pos,
    actor_params,
    model_bundle,
    obs_initialized,
    physics_step_fn: PhysicsStepFn,
    runtime,
    sample_count: int,
) -> dict[str, object]:
    jnp = runtime.jnp
    commanded_qpos = _required(reference, "commanded_qpos")
    command_qpos = commanded_qpos[:, step_index]
    commanded_joint_vel = _required(reference, "commanded_joint_vel")
    command_joint_vel = commanded_joint_vel[:, step_index]
    command_reference = _time_slice_command_reference(
        reference.get("command_reference", {}),
        step_index,
        jnp=jnp,
    )
    obs_reference = _time_slice_reference(
        _required(reference, "obs_reference"),
        step_index,
        sample_count,
        jnp=jnp,
    )
    obs_reference = _with_commanded_joint_reference(
        obs_reference,
        command_qpos,
        command_joint_vel,
        command_reference,
    )
    obs, next_obs_state = build_wbc_observation_from_state(
        robot_state=robot_state,
        reference_state=obs_reference,
        obs_state=obs_state,
        indices=obs_indices,
        default_joint_pos=default_joint_pos,
        initialized=obs_initialized,
        jnp=jnp,
    )
    action = jax_actor_forward(actor_params, obs, jnp=jnp)
    robot_state, _physics_score_state = physics_step_fn(
        model_bundle,
        robot_state,
        command_qpos,
        action,
        step_index,
        runtime=runtime,
    )
    robot_state = _batched_robot_state(robot_state, sample_count, jnp=jnp)
    step_control = samples[:, step_index]
    return {
        "robot_state": robot_state,
        "obs_state": JaxObsState(history=next_obs_state.history, last_action=action),
        "prev_control": step_control,
        "prev_joint_vel": _joint_vel(robot_state),
    }


def _lax_scan(runtime):
    lax = getattr(getattr(runtime, "jax", None), "lax", None)
    return getattr(lax, "scan", None)


def _step_obs_initialized(initialized, step_index, *, jnp):
    if isinstance(initialized, bool):
        if initialized:
            return True
        return step_index > 0
    return jnp.logical_or(initialized, step_index > 0)


def _materialized_obs_state(
    obs_state: JaxObsState,
    sample_count: int,
    *,
    jnp,
) -> JaxObsState:
    history = _zero_obs_history(sample_count, jnp=jnp)
    if obs_state.history is not None:
        for name, value in obs_state.history.items():
            if name in history and value is not None:
                history[name] = value
    return JaxObsState(history=history, last_action=obs_state.last_action)


def _zero_obs_history(sample_count: int, *, jnp):
    history = {}
    for name, spec in OBS_FIELD_SPECS.items():
        if name in {"command", "motion_ref_ang_vel"}:
            continue
        history[name] = jnp.zeros((int(sample_count), *spec))
    return history


def _with_commanded_qpos(
    reference: Mapping[str, object],
    commanded_qpos,
    commanded_joint_vel,
    command_reference,
) -> dict[str, object]:
    values = dict(reference)
    values["commanded_qpos"] = commanded_qpos
    values["commanded_joint_vel"] = commanded_joint_vel
    values["command_reference"] = command_reference
    return values


def _command_reference(
    command_reference_fn: CommandReferenceFn | None,
    model_bundle,
    commanded_qpos,
    commanded_qvel,
    *,
    runtime,
    jnp,
) -> dict[str, object]:
    if command_reference_fn is None:
        return {}
    values = command_reference_fn(
        model_bundle,
        commanded_qpos,
        commanded_qvel,
        runtime=runtime,
    )
    return {name: jnp.asarray(value) for name, value in values.items()}


def _commanded_qvel(commanded_qpos, *, jnp):
    if int(commanded_qpos.shape[1]) <= 1:
        return jnp.zeros((*commanded_qpos.shape[:2], QVEL_DIM))

    lin_vel = _differentiate_trajectory(commanded_qpos[..., :3], jnp=jnp)
    delta_quat = _quat_mul(
        commanded_qpos[:, 1:, 3:7],
        _quat_inv(commanded_qpos[:, :-1, 3:7], jnp=jnp),
        jnp=jnp,
    )
    ang_vel = _axis_angle_from_quat(delta_quat, jnp=jnp) / POLICY_DT
    ang_vel = jnp.concatenate([ang_vel, ang_vel[:, -1:]], axis=1)
    root_qvel = _world_velocity_to_qvel(
        commanded_qpos[..., :7],
        jnp.concatenate([lin_vel, ang_vel], axis=-1),
        jnp=jnp,
    )
    joint_vel = _differentiate_trajectory(commanded_qpos[..., 7:], jnp=jnp)
    return jnp.concatenate([root_qvel, joint_vel], axis=-1)


def _differentiate_trajectory(values, *, jnp):
    if int(values.shape[1]) <= 1:
        return jnp.zeros(values.shape)
    velocity = (values[:, 1:] - values[:, :-1]) / POLICY_DT
    return jnp.concatenate([velocity, velocity[:, -1:]], axis=1)


def _time_slice_command_reference(
    values: Mapping[str, object],
    step_index: int,
    *,
    jnp,
) -> dict[str, object]:
    return {name: jnp.asarray(value)[:, step_index] for name, value in values.items()}


def _with_commanded_joint_reference(
    obs_reference: Mapping[str, object],
    command_qpos,
    command_joint_vel,
    command_reference: Mapping[str, object],
) -> dict[str, object]:
    values = dict(obs_reference)
    values["joint_pos"] = command_qpos[:, 7:]
    values["joint_vel"] = command_joint_vel
    for name in ("body_pos_w", "body_quat_w", "body_ang_vel_w"):
        if name in command_reference:
            values[name] = command_reference[name]
    return values


def _validate_samples(samples) -> None:
    if len(samples.shape) != 3:
        raise ValueError(
            f"Expected samples shape (samples, horizon, width), got {samples.shape}"
        )
    if int(samples.shape[0]) < 1:
        raise ValueError("At least one rollout sample is required")
    if int(samples.shape[1]) < 1:
        raise ValueError("At least one rollout horizon step is required")
    if int(samples.shape[-1]) != QPOS_DIM - 1:
        raise ValueError(
            f"Expected control width {QPOS_DIM - 1}, got {samples.shape[-1]}"
        )


def _batched_robot_state(
    state: Mapping[str, object],
    sample_count: int,
    *,
    jnp,
) -> dict[str, object]:
    qpos = _ensure_robot_state_batch(
        "qpos",
        jnp.asarray(state["qpos"]),
        sample_count,
        jnp=jnp,
    )
    qvel = _ensure_robot_state_batch(
        "qvel",
        jnp.asarray(state["qvel"]),
        sample_count,
        jnp=jnp,
    )
    body_pos_w = _ensure_robot_state_batch(
        "body_pos_w",
        jnp.asarray(state["body_pos_w"]),
        sample_count,
        jnp=jnp,
    )
    body_lin_vel_source = state.get("body_lin_vel_w")
    body_lin_vel_w = (
        jnp.zeros(body_pos_w.shape)
        if body_lin_vel_source is None
        else _ensure_robot_state_batch(
            "body_lin_vel_w",
            jnp.asarray(body_lin_vel_source),
            sample_count,
            jnp=jnp,
        )
    )
    body_quat_w = _ensure_robot_state_batch(
        "body_quat_w",
        jnp.asarray(state["body_quat_w"]),
        sample_count,
        jnp=jnp,
    )
    body_ang_vel_w = _ensure_robot_state_batch(
        "body_ang_vel_w",
        jnp.asarray(state["body_ang_vel_w"]),
        sample_count,
        jnp=jnp,
    )
    base_ang_vel_b = _batched_base_ang_vel(
        state.get("base_ang_vel_b"),
        qpos=qpos,
        body_ang_vel_w=body_ang_vel_w,
        sample_count=sample_count,
        jnp=jnp,
    )
    return {
        "qpos": qpos,
        "qvel": qvel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "base_ang_vel_b": base_ang_vel_b,
    }


def _batched_base_ang_vel(
    base_ang_vel_b,
    *,
    qpos,
    body_ang_vel_w,
    sample_count: int,
    jnp,
):
    if base_ang_vel_b is not None:
        return _ensure_batch(jnp.asarray(base_ang_vel_b), sample_count, jnp=jnp)
    return _quat_apply_inverse(qpos[:, 3:7], body_ang_vel_w[:, 0], jnp=jnp)


def _initial_obs_state(reference: Mapping[str, object], sample_count: int, *, jnp):
    obs_state = reference.get("obs_state")
    if obs_state is None:
        return JaxObsState(
            history=None,
            last_action=jnp.zeros((sample_count, ACTION_DIM)),
        )
    if not isinstance(obs_state, JaxObsState):
        raise TypeError("reference['obs_state'] must be a JaxObsState")
    return JaxObsState(
        history=obs_state.history,
        last_action=_ensure_batch(
            jnp.asarray(obs_state.last_action),
            sample_count,
            jnp=jnp,
        ),
    )


def _time_slice_reference(
    values: Mapping[str, object],
    step_index: int,
    sample_count: int,
    *,
    jnp,
) -> dict[str, object]:
    return {
        name: _ensure_reference_batch(
            name,
            jnp.asarray(value)[step_index],
            sample_count,
            jnp=jnp,
        )
        for name, value in values.items()
    }


def _ensure_robot_state_batch(name: str, value, sample_count: int, *, jnp):
    if _is_unbatched_robot_state(name, value):
        return _broadcast_batch(value, sample_count, jnp=jnp)
    return _ensure_batch(value, sample_count, jnp=jnp)


def _ensure_reference_batch(name: str, value, sample_count: int, *, jnp):
    if _is_unbatched_reference(name, value):
        return _broadcast_batch(value, sample_count, jnp=jnp)
    return _ensure_batch(value, sample_count, jnp=jnp)


def _ensure_batch(value, sample_count: int, *, jnp):
    if len(value.shape) > 0 and int(value.shape[0]) == int(sample_count):
        return value
    if len(value.shape) > 0 and int(value.shape[0]) == 1:
        return jnp.repeat(value, int(sample_count), axis=0)
    return _broadcast_batch(value, sample_count, jnp=jnp)


def _broadcast_batch(value, sample_count: int, *, jnp):
    return jnp.repeat(jnp.expand_dims(value, axis=0), int(sample_count), axis=0)


def _is_unbatched_robot_state(name: str, value) -> bool:
    shape = tuple(int(dim) for dim in value.shape)
    if name == "qpos":
        return shape == (QPOS_DIM,)
    if name == "qvel":
        return len(shape) == 1
    if name in {"body_pos_w", "body_lin_vel_w", "body_ang_vel_w"}:
        return len(shape) == 2 and shape[-1] == 3
    if name == "body_quat_w":
        return len(shape) == 2 and shape[-1] == 4
    return False


def _is_unbatched_reference(name: str, value) -> bool:
    shape = tuple(int(dim) for dim in value.shape)
    if name in {"joint_pos", "joint_vel"}:
        return shape == (ACTION_DIM,)
    if name == "root_pos":
        return shape == (3,)
    if name in {"body_pos_w", "body_ang_vel_w", "body_pos", "ee_pos"}:
        return len(shape) == 2 and shape[-1] == 3
    if name == "body_quat_w":
        return len(shape) == 2 and shape[-1] == 4
    if name == "contact":
        return len(shape) == 1
    return False


def _joint_vel(robot_state: Mapping[str, object]):
    return robot_state["qvel"][:, 6:]


def _joint_limit_array(value, *, jnp):
    arr = jnp.asarray(value)
    while len(arr.shape) < 3:
        arr = jnp.expand_dims(arr, axis=0)
    return arr


def _quat_from_axis_angle(axis_angle, *, jnp):
    angle = jnp.sqrt(jnp.sum(axis_angle * axis_angle, axis=-1, keepdims=True))
    half_angle = angle * 0.5
    safe_angle = jnp.where(angle > 1.0e-8, angle, 1.0)
    scale = jnp.where(
        angle > 1.0e-8,
        jnp.sin(half_angle) / safe_angle,
        0.5 - (angle * angle) / 48.0,
    )
    return jnp.concatenate([jnp.cos(half_angle), axis_angle * scale], axis=-1)


def _normalize(value, *, jnp):
    norm = jnp.sqrt(jnp.sum(value * value, axis=-1, keepdims=True))
    return value / jnp.maximum(norm, 1.0e-9)


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
    return jnp.stack([w, x, y, z], axis=-1)


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


def _world_velocity_to_qvel(qpos, world_vel, *, jnp):
    return jnp.concatenate(
        [
            world_vel[..., :3],
            _quat_apply_inverse(qpos[..., 3:7], world_vel[..., 3:6], jnp=jnp),
        ],
        axis=-1,
    )


def _quat_apply_inverse(quat, vec, *, jnp):
    xyz = quat[..., 1:]
    t = _cross(xyz, vec, jnp=jnp) * 2.0
    return vec - quat[..., 0:1] * t + _cross(xyz, t, jnp=jnp)


def _cross(a, b, *, jnp):
    return jnp.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


def _required(values: Mapping[str, object], name: str):
    if name not in values:
        raise KeyError(f"Missing rollout reference field {name!r}")
    return values[name]


__all__ = [
    "controls_to_qpos",
    "make_rollout_scorer",
    "rollout_candidate_controls",
    "score_candidate_controls",
]
