"""One-window JAX-shaped optimizer skeleton for the G1 WBC MJX backend."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace


@dataclass(frozen=True)
class JaxWindowOptimizerConfig:
    samples: int
    horizon_steps: int
    control_steps: int
    knot_count: int
    temperature: float
    root_pos_sigma: float
    root_rot_sigma: float
    joint_sigma: float
    first_control_noise_scale: float = 1.0
    last_control_noise_scale: float = 1.0
    elite_fraction: float = 1.0
    iterations: int = 1
    final_noise_scale: float = 1.0
    sigma_decay: float | None = None
    min_root_pos_sigma: float = 0.0
    min_root_rot_sigma: float = 0.0
    min_joint_sigma: float = 0.0
    use_guided_candidate: bool = True
    freeze_first_frame: bool = True
    min_score_improvement: float = 1.0e-9
    min_top_score_gap: float = 0.0
    cem_update_min_top_score_gap: float = 0.0
    max_control_delta: float | None = None
    candidate_rank_diagnostics_top_k: int = 0
    candidate_rescore_diagnostics: bool = False
    candidate_rescore_selection_top_k: int = 0
    score_only_rescore_diagnostics: bool = False
    score_only_output_rescore_diagnostics: bool = False
    candidate_score_component_diagnostics_top_k: int = 0


@dataclass(frozen=True)
class JaxWindowResult:
    updated_controls: object
    execute_chunk: object
    info: dict[str, object]
    execute_trace: object | None = None


_SCORE_COMPONENT_DIAGNOSTIC_FIELDS = (
    "root_pos_error_mean",
    "root_rot_error_mean",
    "joint_pos_error_mean",
    "body_global_pos_error_mean",
    "body_global_rot_error_mean",
    "body_local_pos_error_mean",
    "body_local_rot_error_mean",
    "ee_global_pos_error_mean",
    "ee_global_rot_error_mean",
    "ee_local_pos_error_mean",
    "ee_local_rot_error_mean",
    "hand_global_pos_error_mean",
    "hand_global_rot_error_mean",
    "hand_local_pos_error_mean",
    "hand_local_rot_error_mean",
    "contact_mismatch_rate",
    "contact_false_positive_rate",
    "contact_false_negative_rate",
    "contact_switch_rate",
    "bad_floor_contact_rate",
    "bad_floor_force_excess_mean",
    "contact_force_active",
    "contact_force_delta_mean",
    "contact_force_peak",
    "contact_force_peak_excess_mean",
    "control_delta_mean",
    "action_delta_mean",
    "joint_acc_mean",
    "joint_jerk_mean",
)
_SCORE_COMPONENT_SOURCE_ROLLOUT = 0
_SCORE_COMPONENT_SOURCE_SCORE_ONLY_OUTPUT_RESCORE = 1


def sample_residual_controls(
    config: JaxWindowOptimizerConfig,
    controls,
    key,
    *,
    runtime,
    guided_controls=None,
    anchor_controls=None,
    include_center_candidate: bool = False,
):
    """Sample residual control candidates around a control mean."""

    _validate_config(config)
    jnp = runtime.jnp
    controls = _validate_control_array(
        controls,
        config,
        name="controls",
        jnp=jnp,
    )
    anchor_controls = (
        controls
        if anchor_controls is None
        else _validate_control_array(
            anchor_controls,
            config,
            name="anchor_controls",
            jnp=jnp,
        )
    )
    parameter_steps = _sample_parameter_steps(config)
    sigma = _control_sigma(
        config,
        controls.shape[-1],
        steps=parameter_steps,
        jnp=jnp,
    )
    noise = runtime.jax.random.normal(
        _prng_key(key, runtime=runtime),
        (int(config.samples), int(parameter_steps), int(controls.shape[-1])),
    )
    delta = noise * sigma[None, :, :]
    if bool(config.freeze_first_frame):
        delta = _set_first_frame(delta, 0.0)
    if int(parameter_steps) != int(config.horizon_steps):
        delta = _interpolate_knot_samples(
            delta,
            target_steps=int(config.horizon_steps),
            jnp=jnp,
        )
    if bool(config.freeze_first_frame):
        delta = _set_first_frame(delta, 0.0)
    samples = controls[None, :, :] + delta
    samples = _set_sample(samples, 0, anchor_controls)
    next_reserved = 1
    if (
        bool(config.use_guided_candidate)
        and guided_controls is not None
        and int(config.samples) > next_reserved
    ):
        guided_controls = _validate_guided_controls(
            guided_controls,
            controls,
            jnp=jnp,
        )
        samples = _set_sample(samples, next_reserved, guided_controls)
        next_reserved += 1
    if bool(include_center_candidate) and int(config.samples) > next_reserved:
        samples = _set_sample(samples, next_reserved, controls)
    return samples


def _sample_parameter_residual_controls(
    config: JaxWindowOptimizerConfig,
    anchor_controls,
    center_delta,
    sigma,
    key,
    *,
    runtime,
    guided_controls=None,
    include_center_candidate: bool = False,
):
    """Sample candidates from a CEM center in knot/parameter space."""

    _validate_config(config)
    jnp = runtime.jnp
    anchor_controls = _validate_control_array(
        anchor_controls,
        config,
        name="anchor_controls",
        jnp=jnp,
    )
    center_delta = jnp.asarray(center_delta)
    sigma = jnp.asarray(sigma)
    parameter_steps = _sample_parameter_steps(config)
    expected_parameter_shape = (
        int(parameter_steps),
        int(anchor_controls.shape[-1]),
    )
    if tuple(int(dim) for dim in center_delta.shape) != expected_parameter_shape:
        raise ValueError(
            "Expected center_delta shape "
            f"{expected_parameter_shape}, got {center_delta.shape}"
        )
    if tuple(int(dim) for dim in sigma.shape) != expected_parameter_shape:
        raise ValueError(
            f"Expected sigma shape {expected_parameter_shape}, got {sigma.shape}"
        )
    noise = runtime.jax.random.normal(
        _prng_key(key, runtime=runtime),
        (
            int(config.samples),
            int(parameter_steps),
            int(anchor_controls.shape[-1]),
        ),
    )
    parameter_deltas = center_delta[None, :, :] + noise * sigma[None, :, :]
    if bool(config.freeze_first_frame):
        parameter_deltas = _set_first_frame(parameter_deltas, 0.0)
    horizon_deltas = _parameter_delta_to_horizon_delta(
        parameter_deltas,
        config,
        jnp=jnp,
    )
    if bool(config.freeze_first_frame):
        horizon_deltas = _set_first_frame(horizon_deltas, 0.0)
    samples = anchor_controls[None, :, :] + horizon_deltas
    zero_delta = center_delta * 0.0
    samples = _set_sample(samples, 0, anchor_controls)
    parameter_deltas = _set_sample(parameter_deltas, 0, zero_delta)
    next_reserved = 1
    if (
        bool(config.use_guided_candidate)
        and guided_controls is not None
        and int(config.samples) > next_reserved
    ):
        guided_controls = _validate_guided_controls(
            guided_controls,
            anchor_controls,
            jnp=jnp,
        )
        guided_parameter_delta = _horizon_delta_to_parameter_delta(
            guided_controls - anchor_controls,
            config,
            jnp=jnp,
        )
        samples = _set_sample(samples, next_reserved, guided_controls)
        parameter_deltas = _set_sample(
            parameter_deltas,
            next_reserved,
            guided_parameter_delta,
        )
        next_reserved += 1
    if bool(include_center_candidate) and int(config.samples) > next_reserved:
        center_controls = anchor_controls + _parameter_delta_to_horizon_delta(
            center_delta,
            config,
            jnp=jnp,
        )
        samples = _set_sample(samples, next_reserved, center_controls)
        parameter_deltas = _set_sample(parameter_deltas, next_reserved, center_delta)
    return samples, parameter_deltas


def optimize_window(
    config: JaxWindowOptimizerConfig,
    state,
    controls,
    reference,
    actor_params,
    model_bundle,
    key,
    *,
    runtime,
):
    """Optimize one receding MPC window using an injected rollout scorer."""

    return _optimize_window_impl(
        config,
        state,
        controls,
        reference,
        actor_params,
        model_bundle,
        key,
        runtime=runtime,
        collect_execute_trace=True,
    )


def make_jitted_window_optimizer(*, runtime):
    """Build a cached outer-JIT optimizer for the default no-prefix scoring path."""

    compiled_by_signature: dict[tuple[object, ...], object] = {}

    def optimizer(
        config: JaxWindowOptimizerConfig,
        state,
        controls,
        reference,
        actor_params,
        model_bundle,
        key,
        *,
        runtime=runtime,
    ):
        jit = getattr(getattr(runtime, "jax", None), "jit", None)
        if jit is None:
            return optimize_window(
                config,
                state,
                controls,
                reference,
                actor_params,
                model_bundle,
                key,
                runtime=runtime,
            )
        rollout_fn = state["rollout_fn"]
        score_only_rollout_fn = state.get("score_only_rollout_fn")
        score_only_output_rollout_fn = state.get("score_only_output_rollout_fn")
        cache_key = (
            id(rollout_fn),
            id(score_only_rollout_fn) if score_only_rollout_fn is not None else None,
            (
                id(score_only_output_rollout_fn)
                if score_only_output_rollout_fn is not None
                else None
            ),
            id(model_bundle),
            _optimizer_static_signature(config),
        )
        compiled = compiled_by_signature.get(cache_key)
        if compiled is None:

            def optimize_for_signature(controls, reference, actor_params, iteration_keys):
                result = _optimize_window_impl(
                    config,
                    {
                        "rollout_fn": rollout_fn,
                        "score_only_rollout_fn": score_only_rollout_fn,
                        "score_only_output_rollout_fn": (
                            score_only_output_rollout_fn
                        ),
                    },
                    controls,
                    reference,
                    actor_params,
                    model_bundle,
                    None,
                    runtime=runtime,
                    iteration_keys=iteration_keys,
                    collect_execute_trace=True,
                )
                return (
                    result.updated_controls,
                    result.execute_chunk,
                    result.info,
                    result.execute_trace,
                )

            compiled = jit(optimize_for_signature)
            compiled_by_signature[cache_key] = compiled
        iteration_keys = tuple(
            _prng_key(_iteration_key(key, iteration, config=config), runtime=runtime)
            for iteration in range(int(config.iterations))
        )
        updated_controls, execute_chunk, info, execute_trace = compiled(
            controls,
            reference,
            actor_params,
            iteration_keys,
        )
        return JaxWindowResult(
            updated_controls=updated_controls,
            execute_chunk=execute_chunk,
            info=info,
            execute_trace=execute_trace,
        )

    return optimizer


def _optimize_window_impl(
    config: JaxWindowOptimizerConfig,
    state,
    controls,
    reference,
    actor_params,
    model_bundle,
    key,
    *,
    runtime,
    iteration_keys=None,
    collect_execute_trace: bool,
):
    _validate_config(config)
    jnp = runtime.jnp
    rollout_fn = state["rollout_fn"]
    sampling_center_controls = controls
    adaptive_sigma = _adaptive_sigma_enabled(config)
    parameter_steps = _sample_parameter_steps(config)
    width = int(jnp.asarray(controls).shape[-1])
    sampling_center_delta = jnp.full((parameter_steps, width), 0.0)
    sigma = _control_sigma(config, width, steps=parameter_steps, jnp=jnp)
    min_sigma = _minimum_control_sigma(config, width, steps=parameter_steps, jnp=jnp)
    best_controls = controls
    best_observed_score = -math.inf
    best_execute_trace = None
    info: dict[str, object] = {}
    accepted_iterations = 0
    zero_delta_noop_iterations = 0
    score_threshold_noop_iterations = 0
    top_score_gap_noop_iterations = 0
    control_delta_guard_noop_iterations = 0
    noop_candidate_iterations = 0
    scores_finite = True
    iteration_best_indices = []
    iteration_score_improvements = []
    iteration_second_best_scores = []
    iteration_top_score_gaps = []
    iteration_control_delta_maxes = []
    iteration_accepted_flags = []
    iteration_current_selected_flags = []
    iteration_zero_delta_noop_flags = []
    iteration_score_threshold_noop_flags = []
    iteration_top_score_gap_noop_flags = []
    iteration_control_delta_guard_noop_flags = []
    iteration_noop_candidate_flags = []
    iteration_candidate_top_indices = []
    iteration_candidate_top_scores = []
    iteration_sample_sums = []
    iteration_sample_squared_sums = []
    iteration_sample_abs_maxes = []
    iteration_sample_checksums = []
    sample_checksum_slots = ()
    iteration_sample_slot_checksums = []
    sample_probe_indices = ()
    iteration_sample_probe_values = []
    iteration_rescore_best_indices = []
    iteration_rescore_top_score_gaps = []
    iteration_rescore_score_delta_maxes = []
    iteration_rescore_score_delta_means = []
    iteration_rescore_score_delta_max_indices = []
    iteration_rescore_top1_changed_flags = []
    iteration_rescore_candidate_top_indices = []
    iteration_rescore_candidate_top_scores = []
    iteration_rescore_selection_candidate_indices = []
    iteration_rescore_selection_original_scores = []
    iteration_rescore_selection_rescore_scores = []
    iteration_rescore_selection_average_scores = []
    iteration_rescore_selection_best_indices = []
    iteration_rescore_selection_changed_flags = []
    iteration_rescore_selection_score_delta_maxes = []
    iteration_rescore_selection_score_delta_means = []
    iteration_rescore_selection_top_score_gaps = []
    iteration_score_only_rescore_best_indices = []
    iteration_score_only_rescore_top_score_gaps = []
    iteration_score_only_rescore_score_delta_maxes = []
    iteration_score_only_rescore_score_delta_means = []
    iteration_score_only_rescore_score_delta_max_indices = []
    iteration_score_only_rescore_top1_changed_flags = []
    iteration_score_only_output_rescore_best_indices = []
    iteration_score_only_output_rescore_top_score_gaps = []
    iteration_score_only_output_rescore_score_delta_maxes = []
    iteration_score_only_output_rescore_score_delta_means = []
    iteration_score_only_output_rescore_score_delta_max_indices = []
    iteration_score_only_output_rescore_top1_changed_flags = []
    iteration_score_component_top_indices = []
    iteration_score_component_current_values = []
    iteration_score_component_selected_values = []
    iteration_score_component_top_values = []
    iteration_score_component_field_indices = []
    iteration_score_component_peak_source_current_values = []
    iteration_score_component_peak_source_selected_values = []
    iteration_score_component_peak_source_top_values = []
    iteration_rescore_score_component_current_values = []
    iteration_rescore_score_component_selected_values = []
    iteration_rescore_score_component_top_values = []
    iteration_rescore_score_component_best_indices = []
    iteration_rescore_score_component_best_values = []
    iteration_rescore_score_component_delta_max_indices = []
    iteration_rescore_score_component_delta_max_original_values = []
    iteration_rescore_score_component_delta_max_rescore_values = []
    candidate_rank_diagnostics_top_k = int(config.candidate_rank_diagnostics_top_k)
    candidate_rescore_diagnostics_enabled = bool(
        config.candidate_rescore_diagnostics
    )
    candidate_rescore_selection_top_k = int(config.candidate_rescore_selection_top_k)
    score_only_rescore_diagnostics_enabled = bool(
        config.score_only_rescore_diagnostics
    )
    score_only_output_rescore_diagnostics_enabled = bool(
        config.score_only_output_rescore_diagnostics
    )
    candidate_score_component_diagnostics_top_k = int(
        config.candidate_score_component_diagnostics_top_k
    )
    score_component_diagnostics_field_count = 0
    score_component_diagnostics_field_indices = ()
    score_component_diagnostics_source_code = _SCORE_COMPONENT_SOURCE_ROLLOUT
    score_component_peak_source_available = False
    score_component_peak_source_width = 0
    for iteration in range(int(config.iterations)):
        sample_key = (
            iteration_keys[int(iteration)]
            if iteration_keys is not None
            else _iteration_key(key, iteration, config=config)
        )
        previous_sampling_center_controls = sampling_center_controls
        has_next_iteration = int(iteration) + 1 < int(config.iterations)
        iteration_config = (
            config if adaptive_sigma else _iteration_noise_config(config, iteration)
        )
        if adaptive_sigma:
            samples, parameter_deltas = _sample_parameter_residual_controls(
                iteration_config,
                controls,
                sampling_center_delta,
                sigma,
                sample_key,
                runtime=runtime,
                guided_controls=_guided_controls_from_reference(reference),
                include_center_candidate=iteration > 0,
            )
        else:
            samples = sample_residual_controls(
                iteration_config,
                sampling_center_controls,
                sample_key,
                runtime=runtime,
                guided_controls=_guided_controls_from_reference(reference),
                anchor_controls=controls,
                include_center_candidate=iteration > 0,
            )
            parameter_deltas = None
        rollout_result = rollout_fn(samples, reference, actor_params, model_bundle)
        scores = _rollout_scores(rollout_result, jnp=jnp)
        scores_finite = _logical_and(
            scores_finite,
            _validate_scores(scores, config, jnp=jnp),
            jnp=jnp,
        )
        selection_scores = scores
        rescore_selection_diagnostics = {}
        if candidate_rescore_selection_top_k > 0:
            (
                selection_scores,
                rescore_selection_scores,
                rescore_selection_diagnostics,
            ) = _candidate_rescore_selection_scores(
                scores,
                samples,
                rollout_fn,
                reference,
                actor_params,
                model_bundle,
                top_k=candidate_rescore_selection_top_k,
                jnp=jnp,
            )
            scores_finite = _logical_and(
                scores_finite,
                _validate_scores(
                    rescore_selection_scores,
                    replace(config, samples=candidate_rescore_selection_top_k),
                    jnp=jnp,
                ),
                jnp=jnp,
            )
        best_index = jnp.argmax(selection_scores)
        control_score = selection_scores[0]
        best_score = selection_scores[best_index]
        second_best_score, top_score_gap = _second_best_score_and_gap(
            selection_scores,
            best_score,
            jnp=jnp,
        )
        candidate_rank_diagnostics = _candidate_rank_diagnostics(
            scores,
            top_k=candidate_rank_diagnostics_top_k,
            jnp=jnp,
        )
        if candidate_rank_diagnostics:
            sample_diagnostics = _candidate_sample_diagnostics(samples, jnp=jnp)
        else:
            sample_diagnostics = {}
        rescore_diagnostics = {}
        if candidate_rescore_diagnostics_enabled:
            rescore_result = rollout_fn(samples, reference, actor_params, model_bundle)
            rescore_scores = _rollout_scores(rescore_result, jnp=jnp)
            scores_finite = _logical_and(
                scores_finite,
                _validate_scores(rescore_scores, config, jnp=jnp),
                jnp=jnp,
            )
            rescore_diagnostics = _candidate_rescore_diagnostics(
                scores,
                rescore_scores,
                top_k=candidate_rank_diagnostics_top_k,
                jnp=jnp,
            )
        score_only_rescore_diagnostics = {}
        if score_only_rescore_diagnostics_enabled:
            score_only_rollout_fn = state.get("score_only_rollout_fn")
            if score_only_rollout_fn is None:
                raise ValueError(
                    "score_only_rescore_diagnostics requires "
                    "state['score_only_rollout_fn']"
                )
            score_only_rescore_result = score_only_rollout_fn(
                samples,
                reference,
                actor_params,
                model_bundle,
            )
            score_only_rescore_scores = _rollout_scores(
                score_only_rescore_result,
                jnp=jnp,
            )
            scores_finite = _logical_and(
                scores_finite,
                _validate_scores(score_only_rescore_scores, config, jnp=jnp),
                jnp=jnp,
            )
            score_only_rescore_diagnostics = _candidate_rescore_diagnostics(
                scores,
                score_only_rescore_scores,
                top_k=candidate_rank_diagnostics_top_k,
                jnp=jnp,
            )
        score_only_output_rescore_result = None
        score_only_output_rescore_scores = None
        score_only_output_rescore_diagnostics = {}
        if score_only_output_rescore_diagnostics_enabled:
            score_only_output_rollout_fn = state.get(
                "score_only_output_rollout_fn"
            )
            if score_only_output_rollout_fn is None:
                raise ValueError(
                    "score_only_output_rescore_diagnostics requires "
                    "state['score_only_output_rollout_fn']"
                )
            score_only_output_rescore_result = score_only_output_rollout_fn(
                samples,
                reference,
                actor_params,
                model_bundle,
            )
            score_only_output_rescore_scores = _rollout_scores(
                score_only_output_rescore_result,
                jnp=jnp,
            )
            scores_finite = _logical_and(
                scores_finite,
                _validate_scores(score_only_output_rescore_scores, config, jnp=jnp),
                jnp=jnp,
            )
            score_only_output_rescore_diagnostics = (
                _candidate_rescore_diagnostics(
                    scores,
                    score_only_output_rescore_scores,
                    top_k=candidate_rank_diagnostics_top_k,
                    jnp=jnp,
                )
            )
        score_improvement = best_score - control_score
        score_component_diagnostics = {}
        rescore_score_component_diagnostics = {}
        if candidate_score_component_diagnostics_top_k > 0:
            score_component_diagnostics = _candidate_score_component_diagnostics(
                rollout_result,
                scores,
                selected_index=best_index,
                top_k=candidate_score_component_diagnostics_top_k,
                jnp=jnp,
            )
            score_component_source_code = _SCORE_COMPONENT_SOURCE_ROLLOUT
            if (
                int(score_component_diagnostics["field_count"]) == 0
                and score_only_output_rescore_result is not None
                and score_only_output_rescore_scores is not None
            ):
                score_only_output_score_component_diagnostics = (
                    _candidate_score_component_diagnostics(
                        score_only_output_rescore_result,
                        score_only_output_rescore_scores,
                        selected_index=best_index,
                        top_k=candidate_score_component_diagnostics_top_k,
                        jnp=jnp,
                    )
                )
                if int(score_only_output_score_component_diagnostics["field_count"]) > 0:
                    score_component_diagnostics = (
                        score_only_output_score_component_diagnostics
                    )
                    score_component_source_code = (
                        _SCORE_COMPONENT_SOURCE_SCORE_ONLY_OUTPUT_RESCORE
                    )
            score_component_field_count = int(
                score_component_diagnostics["field_count"]
            )
            if score_component_field_count > score_component_diagnostics_field_count:
                score_component_diagnostics_field_count = score_component_field_count
                score_component_diagnostics_field_indices = (
                    score_component_diagnostics["field_indices"]
                )
                score_component_diagnostics_source_code = score_component_source_code
            if (
                rescore_diagnostics
                and score_component_source_code == _SCORE_COMPONENT_SOURCE_ROLLOUT
            ):
                rescore_score_component_diagnostics = (
                    _candidate_rescore_score_component_diagnostics(
                        rollout_result,
                        rescore_result,
                        fields=score_component_diagnostics["fields"],
                        selected_index=best_index,
                        rescore_best_index=rescore_diagnostics["rescore_best_index"],
                        delta_max_index=rescore_diagnostics[
                            "rescore_score_delta_max_index"
                        ],
                        top_indices=score_component_diagnostics["top_indices_array"],
                        jnp=jnp,
                    )
                )
        temperature = max(float(config.temperature), 1.0e-6)
        if adaptive_sigma and has_next_iteration:
            elite_param_scores, elite_parameter_deltas = _elite_scores_and_samples(
                scores,
                parameter_deltas,
                config,
                jnp=jnp,
            )
            param_weights = runtime.jax.nn.softmax(
                (elite_param_scores - jnp.mean(elite_param_scores)) / temperature
            )
            sampling_center_delta_candidate = jnp.sum(
                elite_parameter_deltas * param_weights[:, None, None],
                axis=0,
            )
            sampling_center_candidate = controls + _parameter_delta_to_horizon_delta(
                sampling_center_delta_candidate,
                config,
                jnp=jnp,
            )
            centered_parameter_deltas = (
                elite_parameter_deltas - sampling_center_delta_candidate[None, :, :]
            )
            elite_std = _sqrt(
                jnp.sum(
                    centered_parameter_deltas
                    * centered_parameter_deltas
                    * param_weights[:, None, None],
                    axis=0,
                )
                + 1.0e-8,
                jnp=jnp,
            )
            sigma_candidate = _maximum(
                min_sigma,
                (0.5 * sigma + 0.5 * elite_std) * float(config.sigma_decay),
                jnp=jnp,
            )
            sampling_center_delta_candidate = _freeze_parameter_first_row(
                sampling_center_delta_candidate,
                enabled=bool(config.freeze_first_frame),
            )
            sigma_candidate = _freeze_parameter_first_row(
                sigma_candidate,
                enabled=bool(config.freeze_first_frame),
            )
        elif not adaptive_sigma and has_next_iteration:
            elite_scores, elite_samples = _elite_scores_and_samples(
                scores,
                samples,
                config,
                jnp=jnp,
            )
            weights = runtime.jax.nn.softmax(
                (elite_scores - jnp.mean(elite_scores)) / temperature
            )
            sampling_center_candidate = jnp.sum(
                elite_samples * weights[:, None, None],
                axis=0,
            )
        else:
            sampling_center_candidate = sampling_center_controls
        candidate_controls = samples[best_index]
        control_delta_max = _max_abs_delta(
            candidate_controls,
            controls,
            jnp=jnp,
        )
        sampling_center_delta_max = _max_abs_delta(
            sampling_center_candidate,
            sampling_center_controls,
            jnp=jnp,
        )
        iteration_accepted = _accepted_candidate(
            score_improvement,
            control_delta_max,
            top_score_gap,
            min_score_improvement=float(config.min_score_improvement),
            min_top_score_gap=float(config.min_top_score_gap),
            max_control_delta=config.max_control_delta,
            jnp=jnp,
        )
        accepted_iterations = accepted_iterations + _accepted_iteration_increment(
            iteration_accepted
        )
        current_controls_selected = _current_controls_selected(
            best_index,
            score_improvement,
            jnp=jnp,
        )
        zero_delta_noop_selected = _zero_delta_noop_selected(
            best_index,
            score_improvement,
            control_delta_max,
            min_score_improvement=float(config.min_score_improvement),
            jnp=jnp,
        )
        score_threshold_noop_selected = _score_threshold_noop_selected(
            best_index,
            score_improvement,
            control_delta_max,
            min_score_improvement=float(config.min_score_improvement),
            jnp=jnp,
        )
        top_score_gap_noop_selected = _top_score_gap_noop_selected(
            best_index,
            score_improvement,
            control_delta_max,
            top_score_gap,
            min_top_score_gap=float(config.min_top_score_gap),
            jnp=jnp,
        )
        control_delta_guard_noop_selected = _control_delta_guard_noop_selected(
            best_index,
            score_improvement,
            control_delta_max,
            max_control_delta=config.max_control_delta,
            jnp=jnp,
        )
        noop_candidate_selected = _logical_or(
            _logical_or(
                zero_delta_noop_selected,
                score_threshold_noop_selected,
                jnp=jnp,
            ),
            _logical_or(
                top_score_gap_noop_selected,
                control_delta_guard_noop_selected,
                jnp=jnp,
            ),
            jnp=jnp,
        )
        zero_delta_noop_iterations = (
            zero_delta_noop_iterations
            + _accepted_iteration_increment(zero_delta_noop_selected)
        )
        score_threshold_noop_iterations = (
            score_threshold_noop_iterations
            + _accepted_iteration_increment(score_threshold_noop_selected)
        )
        top_score_gap_noop_iterations = (
            top_score_gap_noop_iterations
            + _accepted_iteration_increment(top_score_gap_noop_selected)
        )
        control_delta_guard_noop_iterations = (
            control_delta_guard_noop_iterations
            + _accepted_iteration_increment(control_delta_guard_noop_selected)
        )
        noop_candidate_iterations = (
            noop_candidate_iterations
            + _accepted_iteration_increment(noop_candidate_selected)
        )
        iteration_best_indices.append(best_index)
        iteration_score_improvements.append(score_improvement)
        iteration_second_best_scores.append(second_best_score)
        iteration_top_score_gaps.append(top_score_gap)
        iteration_control_delta_maxes.append(control_delta_max)
        iteration_accepted_flags.append(iteration_accepted)
        iteration_current_selected_flags.append(current_controls_selected)
        iteration_zero_delta_noop_flags.append(zero_delta_noop_selected)
        iteration_score_threshold_noop_flags.append(score_threshold_noop_selected)
        iteration_top_score_gap_noop_flags.append(top_score_gap_noop_selected)
        iteration_control_delta_guard_noop_flags.append(
            control_delta_guard_noop_selected
        )
        iteration_noop_candidate_flags.append(noop_candidate_selected)
        if candidate_rank_diagnostics:
            iteration_candidate_top_indices.append(
                candidate_rank_diagnostics["candidate_top_indices"]
            )
            iteration_candidate_top_scores.append(
                candidate_rank_diagnostics["candidate_top_scores"]
            )
            sample_probe_indices = sample_diagnostics["sample_probe_indices"]
            iteration_sample_sums.append(sample_diagnostics["sample_sum"])
            iteration_sample_squared_sums.append(
                sample_diagnostics["sample_squared_sum"]
            )
            iteration_sample_abs_maxes.append(sample_diagnostics["sample_abs_max"])
            iteration_sample_checksums.append(sample_diagnostics["sample_checksum"])
            sample_checksum_slots = sample_diagnostics["sample_checksum_slots"]
            iteration_sample_slot_checksums.append(
                sample_diagnostics["sample_slot_checksums"]
            )
            iteration_sample_probe_values.append(
                sample_diagnostics["sample_probe_values"]
            )
        if rescore_diagnostics:
            iteration_rescore_best_indices.append(
                rescore_diagnostics["rescore_best_index"]
            )
            iteration_rescore_top_score_gaps.append(
                rescore_diagnostics["rescore_top_score_gap"]
            )
            iteration_rescore_score_delta_maxes.append(
                rescore_diagnostics["rescore_score_delta_max"]
            )
            iteration_rescore_score_delta_means.append(
                rescore_diagnostics["rescore_score_delta_mean"]
            )
            iteration_rescore_score_delta_max_indices.append(
                rescore_diagnostics["rescore_score_delta_max_index"]
            )
            iteration_rescore_top1_changed_flags.append(
                rescore_diagnostics["rescore_top1_changed"]
            )
            if "rescore_candidate_top_indices" in rescore_diagnostics:
                iteration_rescore_candidate_top_indices.append(
                    rescore_diagnostics["rescore_candidate_top_indices"]
                )
                iteration_rescore_candidate_top_scores.append(
                    rescore_diagnostics["rescore_candidate_top_scores"]
                )
        if rescore_selection_diagnostics:
            iteration_rescore_selection_candidate_indices.append(
                rescore_selection_diagnostics[
                    "rescore_selection_candidate_indices"
                ]
            )
            iteration_rescore_selection_original_scores.append(
                rescore_selection_diagnostics[
                    "rescore_selection_original_scores"
                ]
            )
            iteration_rescore_selection_rescore_scores.append(
                rescore_selection_diagnostics[
                    "rescore_selection_rescore_scores"
                ]
            )
            iteration_rescore_selection_average_scores.append(
                rescore_selection_diagnostics[
                    "rescore_selection_average_scores"
                ]
            )
            iteration_rescore_selection_best_indices.append(
                rescore_selection_diagnostics[
                    "rescore_selection_best_index"
                ]
            )
            iteration_rescore_selection_changed_flags.append(
                rescore_selection_diagnostics[
                    "rescore_selection_changed"
                ]
            )
            iteration_rescore_selection_score_delta_maxes.append(
                rescore_selection_diagnostics[
                    "rescore_selection_score_delta_max"
                ]
            )
            iteration_rescore_selection_score_delta_means.append(
                rescore_selection_diagnostics[
                    "rescore_selection_score_delta_mean"
                ]
            )
            iteration_rescore_selection_top_score_gaps.append(
                rescore_selection_diagnostics[
                    "rescore_selection_top_score_gap"
                ]
            )
        if score_only_rescore_diagnostics:
            iteration_score_only_rescore_best_indices.append(
                score_only_rescore_diagnostics["rescore_best_index"]
            )
            iteration_score_only_rescore_top_score_gaps.append(
                score_only_rescore_diagnostics["rescore_top_score_gap"]
            )
            iteration_score_only_rescore_score_delta_maxes.append(
                score_only_rescore_diagnostics["rescore_score_delta_max"]
            )
            iteration_score_only_rescore_score_delta_means.append(
                score_only_rescore_diagnostics["rescore_score_delta_mean"]
            )
            iteration_score_only_rescore_score_delta_max_indices.append(
                score_only_rescore_diagnostics["rescore_score_delta_max_index"]
            )
            iteration_score_only_rescore_top1_changed_flags.append(
                score_only_rescore_diagnostics["rescore_top1_changed"]
            )
        if score_only_output_rescore_diagnostics:
            iteration_score_only_output_rescore_best_indices.append(
                score_only_output_rescore_diagnostics["rescore_best_index"]
            )
            iteration_score_only_output_rescore_top_score_gaps.append(
                score_only_output_rescore_diagnostics["rescore_top_score_gap"]
            )
            iteration_score_only_output_rescore_score_delta_maxes.append(
                score_only_output_rescore_diagnostics["rescore_score_delta_max"]
            )
            iteration_score_only_output_rescore_score_delta_means.append(
                score_only_output_rescore_diagnostics["rescore_score_delta_mean"]
            )
            iteration_score_only_output_rescore_score_delta_max_indices.append(
                score_only_output_rescore_diagnostics[
                    "rescore_score_delta_max_index"
                ]
            )
            iteration_score_only_output_rescore_top1_changed_flags.append(
                score_only_output_rescore_diagnostics["rescore_top1_changed"]
            )
        if score_component_diagnostics:
            iteration_score_component_field_indices.append(
                score_component_diagnostics["field_indices"]
            )
            iteration_score_component_top_indices.append(
                score_component_diagnostics["top_indices"]
            )
            iteration_score_component_current_values.append(
                score_component_diagnostics["current_values"]
            )
            iteration_score_component_selected_values.append(
                score_component_diagnostics["selected_values"]
            )
            iteration_score_component_top_values.append(
                score_component_diagnostics["top_values"]
            )
            if score_component_diagnostics.get("peak_source_available", False):
                score_component_peak_source_available = True
                score_component_peak_source_width = max(
                    score_component_peak_source_width,
                    int(score_component_diagnostics["peak_source_width"]),
                )
                iteration_score_component_peak_source_current_values.append(
                    score_component_diagnostics["peak_source_current_values"]
                )
                iteration_score_component_peak_source_selected_values.append(
                    score_component_diagnostics["peak_source_selected_values"]
                )
                iteration_score_component_peak_source_top_values.append(
                    score_component_diagnostics["peak_source_top_values"]
                )
        if rescore_score_component_diagnostics:
            iteration_rescore_score_component_current_values.append(
                rescore_score_component_diagnostics["current_values"]
            )
            iteration_rescore_score_component_selected_values.append(
                rescore_score_component_diagnostics["selected_values"]
            )
            iteration_rescore_score_component_top_values.append(
                rescore_score_component_diagnostics["top_values"]
            )
            iteration_rescore_score_component_best_indices.append(
                rescore_score_component_diagnostics["best_index"]
            )
            iteration_rescore_score_component_best_values.append(
                rescore_score_component_diagnostics["best_values"]
            )
            iteration_rescore_score_component_delta_max_indices.append(
                rescore_score_component_diagnostics["delta_max_index"]
            )
            iteration_rescore_score_component_delta_max_original_values.append(
                rescore_score_component_diagnostics["delta_max_original_values"]
            )
            iteration_rescore_score_component_delta_max_rescore_values.append(
                rescore_score_component_diagnostics["delta_max_rescore_values"]
            )
        window_accepted = _accepted_window(
            accepted_iterations,
            current_controls_selected,
            noop_candidate_selected,
            jnp=jnp,
        )
        cem_update_allowed = _top_score_gap_ok(
            top_score_gap,
            min_top_score_gap=float(config.cem_update_min_top_score_gap),
            jnp=jnp,
        )
        sampling_center_candidate = _freeze_first_frame_to_anchor(
            sampling_center_candidate,
            controls,
            enabled=bool(config.freeze_first_frame),
        )
        sampling_center_controls = _where_controls(
            cem_update_allowed,
            sampling_center_candidate,
            previous_sampling_center_controls,
            jnp=jnp,
        )
        if adaptive_sigma and has_next_iteration:
            sampling_center_delta = _where_controls(
                cem_update_allowed,
                sampling_center_delta_candidate,
                sampling_center_delta,
                jnp=jnp,
            )
            sigma = _where_controls(
                cem_update_allowed,
                sigma_candidate,
                sigma,
                jnp=jnp,
            )
        observed_score_improved = _observed_score_improved(
            best_observed_score,
            best_score,
            jnp=jnp,
        )
        accepted_observed_score_improved = _logical_and(
            iteration_accepted,
            observed_score_improved,
            jnp=jnp,
        )
        best_controls = _where_controls(
            accepted_observed_score_improved,
            candidate_controls,
            best_controls,
            jnp=jnp,
        )
        best_observed_score = _where_score(
            accepted_observed_score_improved,
            best_score,
            best_observed_score,
            jnp=jnp,
        )
        if bool(collect_execute_trace):
            best_candidate_execute_trace = _rollout_execute_trace(
                rollout_result,
                best_index,
                runtime=runtime,
                jnp=jnp,
            )
            previous_execute_trace = best_execute_trace
            if previous_execute_trace is None:
                previous_execute_trace = _rollout_execute_trace(
                    rollout_result,
                    0,
                    runtime=runtime,
                    jnp=jnp,
                )
            best_execute_trace = _where_execute_trace(
                accepted_observed_score_improved,
                best_candidate_execute_trace,
                previous_execute_trace,
                runtime=runtime,
                jnp=jnp,
            )
        info = {
            "best_index": best_index,
            "best_score": best_score,
            "second_best_score": second_best_score,
            "top_score_gap": top_score_gap,
            "control_score": control_score,
            "score_improvement": score_improvement,
            "control_delta_max": control_delta_max,
            "sampling_center_delta_max": sampling_center_delta_max,
            "accepted": window_accepted,
            "iteration_accepted": iteration_accepted,
            "accepted_iterations": accepted_iterations,
            "current_controls_selected": current_controls_selected,
            "zero_delta_noop_selected": zero_delta_noop_selected,
            "zero_delta_noop_iterations": zero_delta_noop_iterations,
            "score_threshold_noop_selected": score_threshold_noop_selected,
            "score_threshold_noop_iterations": score_threshold_noop_iterations,
            "top_score_gap_noop_selected": top_score_gap_noop_selected,
            "top_score_gap_noop_iterations": top_score_gap_noop_iterations,
            "control_delta_guard_noop_selected": control_delta_guard_noop_selected,
            "control_delta_guard_noop_iterations": (
                control_delta_guard_noop_iterations
            ),
            "noop_candidate_selected": noop_candidate_selected,
            "noop_candidate_iterations": noop_candidate_iterations,
            "scores_finite": scores_finite,
            "mean_score": jnp.mean(scores),
            "iteration": int(iteration),
            "iterations": int(config.iterations),
            "min_score_improvement": float(config.min_score_improvement),
            "min_top_score_gap": float(config.min_top_score_gap),
            "cem_update_min_top_score_gap": float(
                config.cem_update_min_top_score_gap
            ),
            "cem_update_allowed": cem_update_allowed,
            "max_control_delta": (
                None
                if config.max_control_delta is None
                else float(config.max_control_delta)
            ),
            "iteration_best_indices": tuple(iteration_best_indices),
            "iteration_score_improvements": tuple(iteration_score_improvements),
            "iteration_second_best_scores": tuple(iteration_second_best_scores),
            "iteration_top_score_gaps": tuple(iteration_top_score_gaps),
            "iteration_control_delta_maxes": tuple(iteration_control_delta_maxes),
            "iteration_accepted_flags": tuple(iteration_accepted_flags),
            "iteration_current_controls_selected_flags": tuple(
                iteration_current_selected_flags
            ),
            "iteration_zero_delta_noop_flags": tuple(
                iteration_zero_delta_noop_flags
            ),
            "iteration_score_threshold_noop_flags": tuple(
                iteration_score_threshold_noop_flags
            ),
            "iteration_top_score_gap_noop_flags": tuple(
                iteration_top_score_gap_noop_flags
            ),
            "iteration_control_delta_guard_noop_flags": tuple(
                iteration_control_delta_guard_noop_flags
            ),
            "iteration_noop_candidate_flags": tuple(iteration_noop_candidate_flags),
        }
        if candidate_rank_diagnostics_top_k > 0:
            info.update(
                {
                    "candidate_rank_diagnostics_top_k": (
                        candidate_rank_diagnostics_top_k
                    ),
                    "iteration_candidate_top_indices": tuple(
                        iteration_candidate_top_indices
                    ),
                    "iteration_candidate_top_scores": tuple(
                        iteration_candidate_top_scores
                    ),
                    "iteration_sample_sums": tuple(iteration_sample_sums),
                    "iteration_sample_squared_sums": tuple(
                        iteration_sample_squared_sums
                    ),
                    "iteration_sample_abs_maxes": tuple(iteration_sample_abs_maxes),
                    "iteration_sample_checksums": tuple(iteration_sample_checksums),
                    "sample_diagnostics_version": 3,
                    "sample_shape": tuple(int(dim) for dim in samples.shape),
                    "sample_checksum_slots": tuple(sample_checksum_slots),
                    "iteration_sample_slot_checksums": tuple(
                        iteration_sample_slot_checksums
                    ),
                    "sample_probe_indices": tuple(sample_probe_indices),
                    "iteration_sample_probe_values": tuple(
                        iteration_sample_probe_values
                    ),
                }
            )
        if candidate_rescore_diagnostics_enabled:
            info.update(
                {
                    "candidate_rescore_diagnostics": True,
                    "iteration_rescore_best_indices": tuple(
                        iteration_rescore_best_indices
                    ),
                    "iteration_rescore_top_score_gaps": tuple(
                        iteration_rescore_top_score_gaps
                    ),
                    "iteration_rescore_score_delta_maxes": tuple(
                        iteration_rescore_score_delta_maxes
                    ),
                    "iteration_rescore_score_delta_means": tuple(
                        iteration_rescore_score_delta_means
                    ),
                    "iteration_rescore_score_delta_max_indices": tuple(
                        iteration_rescore_score_delta_max_indices
                    ),
                    "iteration_rescore_top1_changed_flags": tuple(
                        iteration_rescore_top1_changed_flags
                    ),
                }
            )
            if candidate_rank_diagnostics_top_k > 0:
                info.update(
                    {
                        "iteration_rescore_candidate_top_indices": tuple(
                            iteration_rescore_candidate_top_indices
                        ),
                        "iteration_rescore_candidate_top_scores": tuple(
                            iteration_rescore_candidate_top_scores
                        ),
                    }
                )
        if candidate_rescore_selection_top_k > 0:
            info.update(
                {
                    "candidate_rescore_selection_top_k": (
                        candidate_rescore_selection_top_k
                    ),
                    "iteration_rescore_selection_candidate_indices": tuple(
                        iteration_rescore_selection_candidate_indices
                    ),
                    "iteration_rescore_selection_original_scores": tuple(
                        iteration_rescore_selection_original_scores
                    ),
                    "iteration_rescore_selection_rescore_scores": tuple(
                        iteration_rescore_selection_rescore_scores
                    ),
                    "iteration_rescore_selection_average_scores": tuple(
                        iteration_rescore_selection_average_scores
                    ),
                    "iteration_rescore_selection_best_indices": tuple(
                        iteration_rescore_selection_best_indices
                    ),
                    "iteration_rescore_selection_changed_flags": tuple(
                        iteration_rescore_selection_changed_flags
                    ),
                    "iteration_rescore_selection_score_delta_maxes": tuple(
                        iteration_rescore_selection_score_delta_maxes
                    ),
                    "iteration_rescore_selection_score_delta_means": tuple(
                        iteration_rescore_selection_score_delta_means
                    ),
                    "iteration_rescore_selection_top_score_gaps": tuple(
                        iteration_rescore_selection_top_score_gaps
                    ),
                }
            )
        if score_only_rescore_diagnostics_enabled:
            info.update(
                {
                    "score_only_rescore_diagnostics": True,
                    "iteration_score_only_rescore_best_indices": tuple(
                        iteration_score_only_rescore_best_indices
                    ),
                    "iteration_score_only_rescore_top_score_gaps": tuple(
                        iteration_score_only_rescore_top_score_gaps
                    ),
                    "iteration_score_only_rescore_score_delta_maxes": tuple(
                        iteration_score_only_rescore_score_delta_maxes
                    ),
                    "iteration_score_only_rescore_score_delta_means": tuple(
                        iteration_score_only_rescore_score_delta_means
                    ),
                    "iteration_score_only_rescore_score_delta_max_indices": tuple(
                        iteration_score_only_rescore_score_delta_max_indices
                    ),
                    "iteration_score_only_rescore_top1_changed_flags": tuple(
                        iteration_score_only_rescore_top1_changed_flags
                    ),
                }
            )
        if score_only_output_rescore_diagnostics_enabled:
            info.update(
                {
                    "score_only_output_rescore_diagnostics": True,
                    "iteration_score_only_output_rescore_best_indices": tuple(
                        iteration_score_only_output_rescore_best_indices
                    ),
                    "iteration_score_only_output_rescore_top_score_gaps": tuple(
                        iteration_score_only_output_rescore_top_score_gaps
                    ),
                    "iteration_score_only_output_rescore_score_delta_maxes": tuple(
                        iteration_score_only_output_rescore_score_delta_maxes
                    ),
                    "iteration_score_only_output_rescore_score_delta_means": tuple(
                        iteration_score_only_output_rescore_score_delta_means
                    ),
                    "iteration_score_only_output_rescore_score_delta_max_indices": tuple(
                        iteration_score_only_output_rescore_score_delta_max_indices
                    ),
                    "iteration_score_only_output_rescore_top1_changed_flags": tuple(
                        iteration_score_only_output_rescore_top1_changed_flags
                    ),
                }
            )
        if candidate_score_component_diagnostics_top_k > 0:
            info.update(
                {
                    "candidate_score_component_diagnostics_top_k": (
                        candidate_score_component_diagnostics_top_k
                    ),
                    "candidate_score_component_diagnostics_version": 1,
                    "candidate_score_component_diagnostics_field_count": (
                        score_component_diagnostics_field_count
                    ),
                    "candidate_score_component_diagnostics_field_indices": (
                        tuple(score_component_diagnostics_field_indices)
                    ),
                    "candidate_score_component_diagnostics_source_code": (
                        score_component_diagnostics_source_code
                    ),
                    "candidate_score_component_peak_source_available": (
                        score_component_peak_source_available
                    ),
                    "candidate_score_component_peak_source_width": (
                        score_component_peak_source_width
                    ),
                    "iteration_score_component_field_indices": tuple(
                        iteration_score_component_field_indices
                    ),
                    "iteration_score_component_top_indices": tuple(
                        iteration_score_component_top_indices
                    ),
                    "iteration_score_component_current_values": tuple(
                        iteration_score_component_current_values
                    ),
                    "iteration_score_component_selected_values": tuple(
                        iteration_score_component_selected_values
                    ),
                    "iteration_score_component_top_values": tuple(
                        iteration_score_component_top_values
                    ),
                }
            )
            if score_component_peak_source_available:
                info.update(
                    {
                        "iteration_score_component_peak_source_current_values": tuple(
                            iteration_score_component_peak_source_current_values
                        ),
                        "iteration_score_component_peak_source_selected_values": tuple(
                            iteration_score_component_peak_source_selected_values
                        ),
                        "iteration_score_component_peak_source_top_values": tuple(
                            iteration_score_component_peak_source_top_values
                        ),
                    }
                )
            if candidate_rescore_diagnostics_enabled:
                info.update(
                    {
                        "iteration_rescore_score_component_current_values": tuple(
                            iteration_rescore_score_component_current_values
                        ),
                        "iteration_rescore_score_component_selected_values": tuple(
                            iteration_rescore_score_component_selected_values
                        ),
                        "iteration_rescore_score_component_top_values": tuple(
                            iteration_rescore_score_component_top_values
                        ),
                        "iteration_rescore_score_component_best_indices": tuple(
                            iteration_rescore_score_component_best_indices
                        ),
                        "iteration_rescore_score_component_best_values": tuple(
                            iteration_rescore_score_component_best_values
                        ),
                        "iteration_rescore_score_component_delta_max_indices": tuple(
                            iteration_rescore_score_component_delta_max_indices
                        ),
                        "iteration_rescore_score_component_delta_max_original_values": (
                            tuple(
                                iteration_rescore_score_component_delta_max_original_values
                            )
                        ),
                        "iteration_rescore_score_component_delta_max_rescore_values": (
                            tuple(
                                iteration_rescore_score_component_delta_max_rescore_values
                            )
                        ),
                    }
                )
        info.update(_rollout_diagnostics(rollout_result, jnp=jnp))
    updated_controls = _where_controls(
        accepted_iterations > 0,
        best_controls,
        controls,
        jnp=jnp,
    )
    execute_chunk = updated_controls[: int(config.control_steps) + 1]
    return JaxWindowResult(
        updated_controls=updated_controls,
        execute_chunk=execute_chunk,
        info=info,
        execute_trace=best_execute_trace,
    )


def _optimizer_static_signature(config: JaxWindowOptimizerConfig) -> tuple[object, ...]:
    return tuple(getattr(config, field.name) for field in fields(config))


def _rollout_scores(rollout_result, *, jnp):
    if isinstance(rollout_result, Mapping):
        return jnp.asarray(rollout_result["score"])
    return jnp.asarray(rollout_result)


def _candidate_score_component_diagnostics(
    rollout_result,
    scores,
    *,
    selected_index,
    top_k: int,
    jnp,
) -> dict[str, object]:
    top_k = min(max(0, int(top_k)), int(scores.shape[0]))
    top_indices_array = _top_score_indices(scores, top_k=top_k, jnp=jnp)
    fields = _score_component_diagnostic_fields(
        rollout_result,
        scores,
        jnp=jnp,
    )
    peak_source = _score_component_peak_source_diagnostics(
        rollout_result,
        scores,
        selected_index=selected_index,
        top_indices=top_indices_array,
        jnp=jnp,
    )
    return {
        "field_count": len(fields),
        "fields": fields,
        "field_indices": _score_component_diagnostic_field_indices(fields),
        "top_indices_array": top_indices_array,
        "top_indices": tuple(top_indices_array[index] for index in range(top_k)),
        "current_values": _score_component_values(
            rollout_result,
            fields=fields,
            index=0,
            jnp=jnp,
        ),
        "selected_values": _score_component_values(
            rollout_result,
            fields=fields,
            index=selected_index,
            jnp=jnp,
        ),
        "top_values": _score_component_top_values(
            rollout_result,
            fields=fields,
            top_indices=top_indices_array,
            jnp=jnp,
        ),
        **peak_source,
    }


def _candidate_rescore_score_component_diagnostics(
    original_rollout_result,
    rollout_result,
    *,
    fields: tuple[str, ...],
    selected_index,
    rescore_best_index,
    delta_max_index,
    top_indices,
    jnp,
) -> dict[str, object]:
    return {
        "current_values": _score_component_values(
            rollout_result,
            fields=fields,
            index=0,
            jnp=jnp,
        ),
        "selected_values": _score_component_values(
            rollout_result,
            fields=fields,
            index=selected_index,
            jnp=jnp,
        ),
        "top_values": _score_component_top_values(
            rollout_result,
            fields=fields,
            top_indices=top_indices,
            jnp=jnp,
        ),
        "best_index": rescore_best_index,
        "best_values": _score_component_values(
            rollout_result,
            fields=fields,
            index=rescore_best_index,
            jnp=jnp,
        ),
        "delta_max_index": delta_max_index,
        "delta_max_original_values": _score_component_values(
            original_rollout_result,
            fields=fields,
            index=delta_max_index,
            jnp=jnp,
        ),
        "delta_max_rescore_values": _score_component_values(
            rollout_result,
            fields=fields,
            index=delta_max_index,
            jnp=jnp,
        ),
    }


def _score_component_diagnostic_fields(
    rollout_result,
    scores,
    *,
    jnp,
) -> tuple[str, ...]:
    if not isinstance(rollout_result, Mapping):
        return ()
    expected_shape = tuple(int(dim) for dim in scores.shape)
    fields = []
    for name in _SCORE_COMPONENT_DIAGNOSTIC_FIELDS:
        if name not in rollout_result:
            continue
        value = jnp.asarray(rollout_result[name])
        if tuple(int(dim) for dim in value.shape) == expected_shape:
            fields.append(name)
    return tuple(fields)


def _score_component_diagnostic_field_indices(fields: tuple[str, ...]):
    return tuple(
        _SCORE_COMPONENT_DIAGNOSTIC_FIELDS.index(name)
        for name in fields
    )


def _score_component_values(rollout_result, *, fields: tuple[str, ...], index, jnp):
    return tuple(jnp.asarray(rollout_result[name])[index] for name in fields)


def _score_component_top_values(
    rollout_result,
    *,
    fields: tuple[str, ...],
    top_indices,
    jnp,
):
    top_k = int(top_indices.shape[0])
    return tuple(
        tuple(
            jnp.asarray(rollout_result[name])[top_indices[index]]
            for index in range(top_k)
        )
        for name in fields
    )


def _score_component_peak_source_diagnostics(
    rollout_result,
    scores,
    *,
    selected_index,
    top_indices,
    jnp,
) -> dict[str, object]:
    if not isinstance(rollout_result, Mapping):
        return _empty_score_component_peak_source_diagnostics()
    if "contact_force_peak_source" not in rollout_result:
        return _empty_score_component_peak_source_diagnostics()
    source = jnp.asarray(rollout_result["contact_force_peak_source"])
    expected_prefix = tuple(int(dim) for dim in scores.shape)
    if len(source.shape) != len(expected_prefix) + 1:
        return _empty_score_component_peak_source_diagnostics()
    if tuple(int(dim) for dim in source.shape[:-1]) != expected_prefix:
        return _empty_score_component_peak_source_diagnostics()
    width = int(source.shape[-1])
    return {
        "peak_source_available": True,
        "peak_source_width": width,
        "peak_source_current_values": _score_component_peak_source_values(
            source,
            0,
            width=width,
        ),
        "peak_source_selected_values": _score_component_peak_source_values(
            source,
            selected_index,
            width=width,
        ),
        "peak_source_top_values": tuple(
            _score_component_peak_source_values(
                source,
                top_indices[index],
                width=width,
            )
            for index in range(int(top_indices.shape[0]))
        ),
    }


def _score_component_peak_source_values(source, index, *, width: int):
    return tuple(source[index, column] for column in range(int(width)))


def _empty_score_component_peak_source_diagnostics() -> dict[str, object]:
    return {
        "peak_source_available": False,
        "peak_source_width": 0,
    }


def _second_best_score_and_gap(scores, best_score, *, jnp):
    if int(scores.shape[0]) <= 1:
        return best_score, best_score * 0.0
    argsort = getattr(jnp, "argsort", None)
    if callable(argsort):
        sorted_indices = argsort(scores)
        second_best_score = scores[sorted_indices[-2]]
    else:
        masked_scores = scores.copy()
        masked_scores[int(jnp.argmax(scores))] = -math.inf
        second_best_score = scores[int(jnp.argmax(masked_scores))]
    return second_best_score, best_score - second_best_score


def _candidate_rank_diagnostics(scores, *, top_k: int, jnp) -> dict[str, object]:
    top_k = min(max(0, int(top_k)), int(scores.shape[0]))
    if top_k <= 0:
        return {}
    top_indices = _top_score_indices(scores, top_k=top_k, jnp=jnp)
    top_scores = scores[top_indices]
    return {
        "candidate_top_indices": tuple(top_indices[index] for index in range(top_k)),
        "candidate_top_scores": tuple(top_scores[index] for index in range(top_k)),
    }


def _candidate_rescore_diagnostics(
    scores,
    rescore_scores,
    *,
    top_k: int,
    jnp,
) -> dict[str, object]:
    rescore_best_index = jnp.argmax(rescore_scores)
    rescore_best_score = rescore_scores[rescore_best_index]
    _, rescore_top_score_gap = _second_best_score_and_gap(
        rescore_scores,
        rescore_best_score,
        jnp=jnp,
    )
    score_delta = rescore_scores - scores
    abs_fn = getattr(jnp, "abs", None)
    score_delta = abs_fn(score_delta) if callable(abs_fn) else abs(score_delta)
    score_delta_max_index = jnp.argmax(score_delta)
    diagnostics: dict[str, object] = {
        "rescore_best_index": rescore_best_index,
        "rescore_top_score_gap": rescore_top_score_gap,
        "rescore_score_delta_max": _jnp_max(score_delta, jnp=jnp),
        "rescore_score_delta_mean": jnp.mean(score_delta),
        "rescore_score_delta_max_index": score_delta_max_index,
        "rescore_top1_changed": rescore_best_index != jnp.argmax(scores),
    }
    rank = _candidate_rank_diagnostics(
        rescore_scores,
        top_k=top_k,
        jnp=jnp,
    )
    if rank:
        diagnostics["rescore_candidate_top_indices"] = rank["candidate_top_indices"]
        diagnostics["rescore_candidate_top_scores"] = rank["candidate_top_scores"]
    return diagnostics


def _candidate_rescore_selection_scores(
    scores,
    samples,
    rollout_fn,
    reference,
    actor_params,
    model_bundle,
    *,
    top_k: int,
    jnp,
):
    top_k = min(max(0, int(top_k)), int(scores.shape[0]))
    top_indices = _top_score_indices(scores, top_k=top_k, jnp=jnp)
    original_scores = scores[top_indices]
    rescore_result = rollout_fn(
        samples[top_indices],
        reference,
        actor_params,
        model_bundle,
    )
    rescore_scores = _rollout_scores(rescore_result, jnp=jnp)
    _validate_candidate_score_shape(
        rescore_scores,
        expected=(top_k,),
        name="candidate rescore selection scores",
    )
    average_scores = 0.5 * (original_scores + rescore_scores)
    selection_scores = _set_score_subset(scores, top_indices, average_scores)
    selection_best_local_index = jnp.argmax(average_scores)
    selection_best_index = top_indices[selection_best_local_index]
    selection_best_score = average_scores[selection_best_local_index]
    _, selection_top_score_gap = _second_best_score_and_gap(
        average_scores,
        selection_best_score,
        jnp=jnp,
    )
    score_delta = rescore_scores - original_scores
    abs_fn = getattr(jnp, "abs", None)
    score_delta = abs_fn(score_delta) if callable(abs_fn) else abs(score_delta)
    diagnostics = {
        "rescore_selection_candidate_indices": tuple(
            top_indices[index] for index in range(top_k)
        ),
        "rescore_selection_original_scores": tuple(
            original_scores[index] for index in range(top_k)
        ),
        "rescore_selection_rescore_scores": tuple(
            rescore_scores[index] for index in range(top_k)
        ),
        "rescore_selection_average_scores": tuple(
            average_scores[index] for index in range(top_k)
        ),
        "rescore_selection_best_index": selection_best_index,
        "rescore_selection_changed": selection_best_index != jnp.argmax(scores),
        "rescore_selection_score_delta_max": _jnp_max(score_delta, jnp=jnp),
        "rescore_selection_score_delta_mean": jnp.mean(score_delta),
        "rescore_selection_top_score_gap": selection_top_score_gap,
    }
    return selection_scores, rescore_scores, diagnostics


def _top_score_indices(scores, *, top_k: int, jnp):
    argsort = getattr(jnp, "argsort", None)
    if callable(argsort):
        sorted_indices = argsort(scores)
    else:
        sorted_indices = scores.argsort()
    return sorted_indices[-int(top_k) :][::-1]


def _validate_candidate_score_shape(scores, *, expected: tuple[int, ...], name: str):
    actual = tuple(int(dim) for dim in scores.shape)
    if actual != tuple(int(dim) for dim in expected):
        raise ValueError(f"Expected {name} shape {expected}, got {actual}")


def _candidate_sample_diagnostics(samples, *, jnp) -> dict[str, object]:
    abs_fn = getattr(jnp, "abs", None)
    abs_samples = abs_fn(samples) if callable(abs_fn) else abs(samples)
    checksum_slots = _candidate_sample_checksum_slots(samples)
    probe_indices = _candidate_sample_probe_indices(samples)
    return {
        "sample_sum": jnp.sum(samples),
        "sample_squared_sum": jnp.sum(samples * samples),
        "sample_abs_max": _jnp_max(abs_samples, jnp=jnp),
        "sample_checksum": _candidate_sample_checksums(samples, jnp=jnp),
        "sample_checksum_slots": checksum_slots,
        "sample_slot_checksums": _candidate_sample_slot_checksums(
            samples,
            checksum_slots,
            jnp=jnp,
        ),
        "sample_probe_indices": probe_indices,
        "sample_probe_values": tuple(
            samples[sample_index, step_index, control_index]
            for sample_index, step_index, control_index in probe_indices
        ),
    }


def _candidate_sample_checksums(samples, *, jnp) -> tuple[object, object, object]:
    flat = samples.reshape((-1,))
    sample_count = int(flat.shape[0])
    indices = jnp.arange(sample_count, dtype=flat.dtype) + 1.0
    linear_weights = indices / float(sample_count)
    modulo_weights = ((indices % 997.0) + 1.0) / 997.0
    quadratic_weights = (((indices * indices) % 991.0) + 1.0) / 991.0
    return (
        jnp.sum(flat * linear_weights),
        jnp.sum(flat * flat * modulo_weights),
        jnp.sum(flat * quadratic_weights),
    )


def _candidate_sample_slot_checksums(
    samples,
    slots: tuple[int, ...],
    *,
    jnp,
) -> tuple[tuple[object, object, object], ...]:
    return tuple(
        _candidate_sample_checksums(samples[int(slot)], jnp=jnp)
        for slot in slots
    )


def _candidate_sample_checksum_slots(samples) -> tuple[int, ...]:
    sample_count = int(samples.shape[0])
    if sample_count <= 0:
        return ()
    candidates = (
        0,
        min(1, sample_count - 1),
        min(2, sample_count - 1),
        max(0, sample_count - 1),
    )
    unique: list[int] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _candidate_sample_probe_indices(samples) -> tuple[tuple[int, int, int], ...]:
    sample_count = int(samples.shape[0])
    horizon_steps = int(samples.shape[1])
    control_dim = int(samples.shape[2])
    if sample_count <= 0 or horizon_steps <= 0 or control_dim <= 0:
        return ()
    candidates = (
        (0, 0, 0),
        (min(1, sample_count - 1), min(1, horizon_steps - 1), min(1, control_dim - 1)),
        (
            min(2, sample_count - 1),
            max(0, horizon_steps // 2),
            max(0, control_dim // 2),
        ),
        (
            max(0, sample_count - 1),
            max(0, horizon_steps - 1),
            max(0, control_dim - 1),
        ),
    )
    unique: list[tuple[int, int, int]] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return tuple(unique)


def _rollout_diagnostics(rollout_result, *, jnp) -> dict[str, object]:
    if not isinstance(rollout_result, Mapping):
        return {}
    diagnostics: dict[str, object] = {}
    for name in ("active_contact_count", "contact_pair_count", "physics_step_count"):
        if name in rollout_result:
            diagnostics[name] = _jnp_max(jnp.asarray(rollout_result[name]), jnp=jnp)
    return diagnostics


def _rollout_execute_trace(rollout_result, best_index, *, runtime, jnp):
    if not isinstance(rollout_result, Mapping) or "execute_trace" not in rollout_result:
        return None
    return _take_trace_sample(
        rollout_result["execute_trace"],
        best_index,
        sample_axis=1,
        runtime=runtime,
        jnp=jnp,
    )


def _take_trace_sample(value, index, *, sample_axis: int, runtime, jnp):
    if isinstance(value, Mapping):
        selected = {}
        for key, item in value.items():
            if key == "mjx_data":
                selected[key] = _take_mjx_data_sample(
                    item,
                    index,
                    runtime=runtime,
                    jnp=jnp,
                )
            else:
                selected[key] = _take_trace_sample(
                    item,
                    index,
                    sample_axis=_trace_sample_axis(key, sample_axis),
                    runtime=runtime,
                    jnp=jnp,
                )
        return selected
    if is_dataclass(value):
        return type(value)(
            **{
                field.name: _take_trace_sample(
                    getattr(value, field.name),
                    index,
                    sample_axis=sample_axis,
                    runtime=runtime,
                    jnp=jnp,
                )
                for field in fields(value)
            }
        )
    tree_map = _jax_tree_map(runtime)
    if callable(tree_map) and not _is_plain_array_like(value):
        return tree_map(
            lambda leaf: _take_array_sample(
                leaf,
                index,
                axis=sample_axis,
                jnp=jnp,
            ),
            value,
        )
    return _take_array_sample(value, index, axis=sample_axis, jnp=jnp)


def _take_mjx_data_sample(value, index, *, runtime, jnp):
    sample_count = _mjx_data_sample_count(value)
    if sample_count is None:
        return value
    return _take_mjx_data_value(
        value,
        index,
        sample_count=sample_count,
        path=(),
        runtime=runtime,
        jnp=jnp,
    )


def _take_mjx_data_value(value, index, *, sample_count: int, path, runtime, jnp):
    if _is_mjx_warp_flat_contact_path(path):
        return value
    if isinstance(value, Mapping):
        return {
            key: _take_mjx_data_value(
                item,
                index,
                sample_count=sample_count,
                path=(*path, str(key)),
                runtime=runtime,
                jnp=jnp,
            )
            for key, item in value.items()
        }
    if is_dataclass(value):
        return type(value)(
            **{
                field.name: _take_mjx_data_value(
                    getattr(value, field.name),
                    index,
                    sample_count=sample_count,
                    path=(*path, field.name),
                    runtime=runtime,
                    jnp=jnp,
                )
                for field in fields(value)
            }
        )
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0 and int(shape[0]) == int(sample_count):
        return _take_array_sample(value, index, axis=0, jnp=jnp)
    tree_map = _jax_tree_map(runtime)
    if callable(tree_map) and not _is_plain_array_like(value):
        return tree_map(
            lambda leaf: _take_mjx_data_leaf(
                leaf,
                index,
                sample_count=sample_count,
                jnp=jnp,
            ),
            value,
        )
    return value


def _take_mjx_data_leaf(leaf, index, *, sample_count: int, jnp):
    shape = getattr(leaf, "shape", None)
    if shape is not None and len(shape) > 0 and int(shape[0]) == int(sample_count):
        return _take_array_sample(leaf, index, axis=0, jnp=jnp)
    return leaf


def _mjx_data_sample_count(value) -> int | None:
    qpos = _mapping_or_attr(value, "qpos")
    shape = getattr(qpos, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[0])


def _mapping_or_attr(value, name: str):
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _is_mjx_warp_flat_contact_path(path) -> bool:
    return bool(path) and str(path[-1]).startswith("contact__")


def _trace_sample_axis(name: str, default_axis: int) -> int:
    if name.startswith("final_"):
        return 0
    return int(default_axis)


def _take_array_sample(value, index, *, axis: int, jnp):
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) <= int(axis):
        return value
    take = getattr(jnp, "take", None)
    indices = _sample_index_vector(index, jnp=jnp)
    if callable(take):
        return take(value, indices, axis=int(axis))
    return value.take(indices, axis=int(axis))


def _sample_index_vector(index, *, jnp):
    asarray = getattr(jnp, "asarray", None)
    if callable(asarray):
        values = asarray([index])
        astype = getattr(values, "astype", None)
        return astype("int32") if callable(astype) else values
    return [int(index)]


def _is_plain_array_like(value) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        or isinstance(value, (int, float, bool))
    )


def _jax_tree_map(runtime):
    jax = getattr(runtime, "jax", None)
    tree_util = getattr(jax, "tree_util", None)
    tree_map = getattr(tree_util, "tree_map", None)
    if callable(tree_map):
        return tree_map
    tree = getattr(jax, "tree", None)
    return getattr(tree, "map", None)


def _jnp_max(value, *, jnp):
    max_fn = getattr(jnp, "max", None)
    if callable(max_fn):
        return max_fn(value)
    return value.max()


def _maximum(left, right, *, jnp):
    maximum = getattr(jnp, "maximum", None)
    if callable(maximum):
        return maximum(left, right)
    return left if left >= right else right


def _sqrt(value, *, jnp):
    sqrt = getattr(jnp, "sqrt", None)
    if callable(sqrt):
        return sqrt(value)
    return value**0.5


def _sample_parameter_steps(config: JaxWindowOptimizerConfig) -> int:
    horizon = int(config.horizon_steps)
    if horizon <= 1:
        return 1
    return max(2, min(int(config.knot_count), horizon))


def _control_sigma(
    config: JaxWindowOptimizerConfig,
    width: int,
    *,
    steps: int | None = None,
    jnp,
):
    steps = int(config.horizon_steps) if steps is None else int(steps)
    root_pos_width = min(3, int(width))
    root_rot_width = min(3, max(0, int(width) - root_pos_width))
    joint_width = max(0, int(width) - root_pos_width - root_rot_width)
    sigma = jnp.concatenate(
        [
            jnp.full((steps, root_pos_width), config.root_pos_sigma),
            jnp.full((steps, root_rot_width), config.root_rot_sigma),
            jnp.full((steps, joint_width), config.joint_sigma),
        ],
        axis=-1,
    )
    profile = 10.0 ** jnp.linspace(
        math.log10(float(config.first_control_noise_scale)),
        math.log10(float(config.last_control_noise_scale)),
        int(steps),
    )
    return sigma * profile[:, None]


def _minimum_control_sigma(
    config: JaxWindowOptimizerConfig,
    width: int,
    *,
    steps: int,
    jnp,
):
    min_config = replace(
        config,
        root_pos_sigma=float(config.min_root_pos_sigma),
        root_rot_sigma=float(config.min_root_rot_sigma),
        joint_sigma=float(config.min_joint_sigma),
        first_control_noise_scale=1.0,
        last_control_noise_scale=1.0,
    )
    return _control_sigma(min_config, width, steps=int(steps), jnp=jnp)


def _adaptive_sigma_enabled(config: JaxWindowOptimizerConfig) -> bool:
    return config.sigma_decay is not None


def _parameter_delta_to_horizon_delta(delta, config: JaxWindowOptimizerConfig, *, jnp):
    if int(delta.shape[-2]) == int(config.horizon_steps):
        return delta
    if len(delta.shape) == 2:
        expanded = _interpolate_knot_samples(
            delta[None, :, :],
            target_steps=int(config.horizon_steps),
            jnp=jnp,
        )
        return expanded[0]
    return _interpolate_knot_samples(
        delta,
        target_steps=int(config.horizon_steps),
        jnp=jnp,
    )


def _horizon_delta_to_parameter_delta(delta, config: JaxWindowOptimizerConfig, *, jnp):
    parameter_steps = _sample_parameter_steps(config)
    if int(parameter_steps) == int(config.horizon_steps):
        return delta
    positions = jnp.linspace(
        0.0,
        float(int(config.horizon_steps) - 1),
        int(parameter_steps),
    )
    indices = jnp.floor(positions + 0.5).astype("int32")
    if len(delta.shape) == 2:
        return delta[indices]
    return delta[:, indices, :]


def _validate_control_array(value, config: JaxWindowOptimizerConfig, *, name: str, jnp):
    value = jnp.asarray(value)
    if len(value.shape) != 2:
        raise ValueError(f"Expected 2D {name}, got shape {value.shape}")
    if int(value.shape[0]) != int(config.horizon_steps):
        raise ValueError(
            f"Expected {name} horizon {config.horizon_steps}, got {value.shape[0]}"
        )
    if not _all_finite(value, jnp=jnp):
        raise ValueError(f"JAX window optimizer {name} must be finite")
    return value


def _freeze_parameter_first_row(value, *, enabled: bool):
    if not enabled or int(value.shape[0]) < 1:
        return value
    if hasattr(value, "at"):
        return value.at[0].set(0.0)
    out = value.copy()
    out[0] = 0.0
    return out


def _set_first_frame(value, replacement):
    if int(value.shape[1]) < 1:
        return value
    if hasattr(value, "at"):
        return value.at[:, 0, :].set(replacement)
    out = value.copy()
    out[:, 0, :] = replacement
    return out


def _freeze_first_frame_to_anchor(candidate, anchor, *, enabled: bool):
    if not enabled or int(candidate.shape[0]) < 1:
        return candidate
    if hasattr(candidate, "at"):
        return candidate.at[0].set(anchor[0])
    out = candidate.copy()
    out[0] = anchor[0]
    return out


def _interpolate_knot_samples(knot_samples, *, target_steps: int, jnp):
    target_steps = int(target_steps)
    if target_steps < 1:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    knot_count = int(knot_samples.shape[1])
    if target_steps == 1:
        return knot_samples[:, :1, :]
    if knot_count <= 1:
        return jnp.repeat(knot_samples[:, :1, :], target_steps, axis=1)
    positions = jnp.linspace(0.0, float(knot_count - 1), target_steps)
    left = jnp.floor(positions).astype("int32")
    right = jnp.minimum(left + 1, knot_count - 1)
    blend = (positions - left.astype(positions.dtype))[None, :, None]
    left_values = knot_samples[:, left, :]
    right_values = knot_samples[:, right, :]
    return left_values * (1.0 - blend) + right_values * blend


def _validate_config(config: JaxWindowOptimizerConfig) -> None:
    if int(config.samples) < 1:
        raise ValueError("JAX window optimizer requires at least one sample")
    if int(config.horizon_steps) < 1:
        raise ValueError("JAX window optimizer horizon_steps must be positive")
    if int(config.control_steps) < 0:
        raise ValueError("JAX window optimizer control_steps must be non-negative")
    if int(config.control_steps) >= int(config.horizon_steps):
        raise ValueError("control_steps must be smaller than horizon_steps")
    if int(config.knot_count) < 2 and int(config.horizon_steps) > 1:
        raise ValueError("JAX window optimizer knot_count must be at least 2")
    if (
        not math.isfinite(float(config.elite_fraction))
        or float(config.elite_fraction) <= 0.0
        or float(config.elite_fraction) > 1.0
    ):
        raise ValueError("JAX window optimizer elite_fraction must be in (0, 1]")
    if int(config.iterations) < 1:
        raise ValueError("JAX window optimizer iterations must be positive")
    if (
        not math.isfinite(float(config.final_noise_scale))
        or float(config.final_noise_scale) < 0.0
    ):
        raise ValueError("JAX window optimizer final_noise_scale must be non-negative")
    for name in (
        "first_control_noise_scale",
        "last_control_noise_scale",
    ):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"JAX window optimizer {name} must be positive")
    for name in (
        "min_root_pos_sigma",
        "min_root_rot_sigma",
        "min_joint_sigma",
    ):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"JAX window optimizer {name} must be non-negative")
    if config.sigma_decay is not None:
        sigma_decay = float(config.sigma_decay)
        if not math.isfinite(sigma_decay) or sigma_decay <= 0.0:
            raise ValueError("JAX window optimizer sigma_decay must be positive")
    min_score_improvement = float(config.min_score_improvement)
    if not math.isfinite(min_score_improvement) or min_score_improvement < 0.0:
        raise ValueError(
            "JAX window optimizer min_score_improvement must be non-negative"
        )
    for name in ("min_top_score_gap", "cem_update_min_top_score_gap"):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"JAX window optimizer {name} must be non-negative")
    if config.max_control_delta is not None:
        max_control_delta = float(config.max_control_delta)
        if not math.isfinite(max_control_delta) or max_control_delta <= 0.0:
            raise ValueError("JAX window optimizer max_control_delta must be positive")
    candidate_rank_diagnostics_top_k = int(config.candidate_rank_diagnostics_top_k)
    if candidate_rank_diagnostics_top_k < 0:
        raise ValueError(
            "JAX window optimizer candidate_rank_diagnostics_top_k "
            "must be non-negative"
        )
    if candidate_rank_diagnostics_top_k > int(config.samples):
        raise ValueError(
            "JAX window optimizer candidate_rank_diagnostics_top_k "
            "must not exceed samples"
        )
    candidate_rescore_selection_top_k = int(config.candidate_rescore_selection_top_k)
    if candidate_rescore_selection_top_k < 0:
        raise ValueError(
            "JAX window optimizer candidate_rescore_selection_top_k "
            "must be non-negative"
        )
    if candidate_rescore_selection_top_k > int(config.samples):
        raise ValueError(
            "JAX window optimizer candidate_rescore_selection_top_k "
            "must not exceed samples"
        )
    candidate_score_component_diagnostics_top_k = int(
        config.candidate_score_component_diagnostics_top_k
    )
    if candidate_score_component_diagnostics_top_k < 0:
        raise ValueError(
            "JAX window optimizer candidate_score_component_diagnostics_top_k "
            "must be non-negative"
        )
    if candidate_score_component_diagnostics_top_k > int(config.samples):
        raise ValueError(
            "JAX window optimizer candidate_score_component_diagnostics_top_k "
            "must not exceed samples"
        )


def _validate_scores(scores, config: JaxWindowOptimizerConfig, *, jnp):
    expected = (int(config.samples),)
    actual = tuple(int(dim) for dim in scores.shape)
    if actual != expected:
        raise ValueError(f"Expected rollout scores shape {expected}, got {actual}")
    finite = _finite_status(scores, jnp=jnp)
    if isinstance(finite, bool) and not finite:
        raise ValueError("JAX window optimizer rollout scores must be finite")
    return finite


def _guided_controls_from_reference(reference):
    if isinstance(reference, Mapping):
        return reference.get("guided_controls")
    return None


def _validate_guided_controls(guided_controls, controls, *, jnp):
    guided_controls = jnp.asarray(guided_controls)
    expected = tuple(int(dim) for dim in controls.shape)
    actual = tuple(int(dim) for dim in guided_controls.shape)
    if actual != expected:
        raise ValueError(f"Expected guided_controls shape {expected}, got {actual}")
    if not _all_finite(guided_controls, jnp=jnp):
        raise ValueError("JAX window optimizer guided_controls must be finite")
    return guided_controls


def _iteration_noise_config(
    config: JaxWindowOptimizerConfig,
    iteration: int,
) -> JaxWindowOptimizerConfig:
    beta = float(config.final_noise_scale) ** (1.0 / float(config.iterations))
    scale = beta ** int(iteration)
    return replace(
        config,
        root_pos_sigma=float(config.root_pos_sigma) * scale,
        root_rot_sigma=float(config.root_rot_sigma) * scale,
        joint_sigma=float(config.joint_sigma) * scale,
    )


def _all_finite(value, *, jnp) -> bool:
    finite = _finite_status(value, jnp=jnp)
    if isinstance(finite, bool):
        return finite
    return True


def _finite_status(value, *, jnp):
    isfinite = getattr(jnp, "isfinite", None)
    all_fn = getattr(jnp, "all", None)
    if callable(isfinite) and callable(all_fn):
        finite = all_fn(isfinite(value))
        if _is_device_array_like(value) or _is_device_array_like(finite):
            return finite
        try:
            return bool(finite)
        except (TypeError, ValueError):
            pass
    try:
        flat = value.ravel() if hasattr(value, "ravel") else value
        return all(math.isfinite(float(item)) for item in flat)
    except (TypeError, ValueError, OverflowError):
        return False


def _max_abs_delta(candidate, previous, *, jnp):
    delta = candidate - previous
    abs_fn = getattr(jnp, "abs", None)
    delta = abs_fn(delta) if callable(abs_fn) else abs(delta)
    max_fn = getattr(jnp, "max", None)
    if callable(max_fn):
        return max_fn(delta)
    value_max = getattr(delta, "max", None)
    return value_max() if callable(value_max) else max(delta)


def _accepted_candidate(
    score_improvement,
    control_delta_max,
    top_score_gap,
    *,
    min_score_improvement: float,
    min_top_score_gap: float,
    max_control_delta: float | None,
    jnp,
):
    score_ok = score_improvement > float(min_score_improvement)
    delta_ok = control_delta_max > 1.0e-9
    top_gap_ok = _top_score_gap_ok(
        top_score_gap,
        min_top_score_gap=float(min_top_score_gap),
        jnp=jnp,
    )
    guard_ok = _control_delta_within_guard(
        control_delta_max,
        max_control_delta=max_control_delta,
        jnp=jnp,
    )
    logical_and = getattr(jnp, "logical_and", None)
    if callable(logical_and):
        return logical_and(logical_and(logical_and(score_ok, delta_ok), top_gap_ok), guard_ok)
    return score_ok and delta_ok and top_gap_ok and guard_ok


def _top_score_gap_ok(top_score_gap, *, min_top_score_gap: float, jnp):
    del jnp
    if float(min_top_score_gap) <= 0.0:
        return True
    return top_score_gap >= float(min_top_score_gap)


def _control_delta_within_guard(control_delta_max, *, max_control_delta, jnp):
    del jnp
    if max_control_delta is None:
        return True
    return control_delta_max <= float(max_control_delta)


def _score_threshold_noop_selected(
    best_index,
    score_improvement,
    control_delta_max,
    *,
    min_score_improvement: float,
    jnp,
):
    if float(min_score_improvement) <= 1.0e-9:
        return False
    index_ok = best_index != 0
    score_nonnegative = score_improvement >= -1.0e-9
    score_below_threshold = score_improvement <= float(min_score_improvement)
    threshold_noop = _logical_and(
        score_below_threshold,
        control_delta_max > 1.0e-9,
        jnp=jnp,
    )
    return _logical_and(
        _logical_and(index_ok, score_nonnegative, jnp=jnp),
        threshold_noop,
        jnp=jnp,
    )


def _top_score_gap_noop_selected(
    best_index,
    score_improvement,
    control_delta_max,
    top_score_gap,
    *,
    min_top_score_gap: float,
    jnp,
):
    if float(min_top_score_gap) <= 0.0:
        return False
    index_ok = best_index != 0
    score_nonnegative = score_improvement >= -1.0e-9
    delta_ok = control_delta_max > 1.0e-9
    top_gap_below_threshold = top_score_gap < float(min_top_score_gap)
    return _logical_and(
        _logical_and(index_ok, score_nonnegative, jnp=jnp),
        _logical_and(delta_ok, top_gap_below_threshold, jnp=jnp),
        jnp=jnp,
    )


def _control_delta_guard_noop_selected(
    best_index,
    score_improvement,
    control_delta_max,
    *,
    max_control_delta: float | None,
    jnp,
):
    if max_control_delta is None:
        return False
    index_ok = best_index != 0
    score_positive = score_improvement > 1.0e-9
    delta_exceeds_guard = control_delta_max > float(max_control_delta)
    return _logical_and(
        _logical_and(index_ok, score_positive, jnp=jnp),
        delta_exceeds_guard,
        jnp=jnp,
    )


def _zero_delta_noop_selected(
    best_index,
    score_improvement,
    control_delta_max,
    *,
    min_score_improvement: float,
    jnp,
):
    if float(min_score_improvement) <= 1.0e-9:
        return False
    index_ok = best_index != 0
    score_nonnegative = score_improvement >= -1.0e-9
    delta_noop = control_delta_max <= 1.0e-9
    return _logical_and(
        _logical_and(index_ok, score_nonnegative, jnp=jnp),
        delta_noop,
        jnp=jnp,
    )


def _current_controls_selected(best_index, score_improvement, *, jnp):
    index_ok = best_index == 0
    score_ok = score_improvement >= -1.0e-9
    logical_and = getattr(jnp, "logical_and", None)
    if callable(logical_and):
        return logical_and(index_ok, score_ok)
    return index_ok and score_ok


def _accepted_window(
    accepted_iterations,
    current_controls_selected,
    score_threshold_noop_selected=False,
    *,
    jnp,
):
    improved = accepted_iterations > 0
    logical_or = getattr(jnp, "logical_or", None)
    if callable(logical_or):
        return logical_or(
            logical_or(improved, current_controls_selected),
            score_threshold_noop_selected,
        )
    return improved or current_controls_selected or score_threshold_noop_selected


def _logical_or(left, right, *, jnp):
    if isinstance(left, bool) and not left:
        return right
    if isinstance(right, bool) and not right:
        return left
    logical_or = getattr(jnp, "logical_or", None)
    if callable(logical_or):
        return logical_or(left, right)
    return bool(left) or bool(right)


def _accepted_iteration_increment(iteration_accepted):
    astype = getattr(iteration_accepted, "astype", None)
    if callable(astype):
        return astype("int32")
    return int(bool(iteration_accepted))


def _elite_scores_and_samples(scores, samples, config: JaxWindowOptimizerConfig, *, jnp):
    elite_count = max(
        1,
        min(
            int(config.samples),
            int(round(int(config.samples) * float(config.elite_fraction))),
        ),
    )
    if elite_count >= int(config.samples):
        return scores, samples
    sorted_indices = jnp.argsort(scores)
    elite_indices = sorted_indices[-elite_count:]
    return scores[elite_indices], samples[elite_indices]


def _observed_score_improved(previous, current, *, jnp) -> bool:
    del jnp
    if previous is None:
        return True
    return current > previous


def _where_score(condition, candidate, previous, *, jnp):
    if previous is None:
        return candidate
    where = getattr(jnp, "where", None)
    if callable(where):
        return where(condition, candidate, previous)
    return candidate if bool(condition) else previous


def _where_execute_trace(condition, candidate, previous, *, runtime, jnp):
    if candidate is None:
        return previous
    if previous is None:
        return candidate
    tree_map = _jax_tree_map(runtime)
    where = getattr(jnp, "where", None)
    if callable(tree_map) and callable(where):
        return tree_map(lambda new, old: where(condition, new, old), candidate, previous)
    return candidate if bool(condition) else previous


def _where_controls(condition, candidate, previous, *, jnp):
    where = getattr(jnp, "where", None)
    if callable(where):
        return where(condition, candidate, previous)
    return candidate if bool(condition) else previous


def _logical_and(left, right, *, jnp):
    if isinstance(left, bool) and left:
        return right
    if isinstance(right, bool) and right:
        return left
    logical_and = getattr(jnp, "logical_and", None)
    if callable(logical_and):
        return logical_and(left, right)
    return bool(left) and bool(right)


def _is_device_array_like(value) -> bool:
    module = type(value).__module__
    return module.startswith("jax") or module.startswith("jaxlib")


def _prng_key(key, *, runtime):
    random = runtime.jax.random
    if not hasattr(random, "PRNGKey"):
        return key
    if isinstance(key, tuple):
        if not key:
            raise ValueError("JAX optimizer key tuple must not be empty")
        prng = random.PRNGKey(int(key[0]))
        fold_in = getattr(random, "fold_in", None)
        if fold_in is None:
            return prng
        for value in key[1:]:
            prng = fold_in(prng, int(value))
        return prng
    if isinstance(key, int):
        return random.PRNGKey(int(key))
    return key


def _iteration_key(key, iteration: int, *, config: JaxWindowOptimizerConfig):
    if int(config.iterations) == 1:
        return key
    if isinstance(key, tuple):
        return (*key, int(iteration))
    if isinstance(key, int):
        return (int(key), int(iteration))
    return key


def _set_sample(samples, index: int, controls):
    if hasattr(samples, "at"):
        return samples.at[int(index)].set(controls)
    out = samples.copy()
    out[int(index)] = controls
    return out


def _set_score_subset(scores, indices, values):
    if hasattr(scores, "at"):
        return scores.at[indices].set(values)
    out = scores.copy()
    out[indices] = values
    return out


__all__ = [
    "JaxWindowOptimizerConfig",
    "JaxWindowResult",
    "make_jitted_window_optimizer",
    "optimize_window",
    "sample_residual_controls",
]
