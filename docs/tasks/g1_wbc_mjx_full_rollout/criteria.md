# G1 WBC MJX Full-Rollout Final Acceptance Criteria

## Purpose

This file is the durable acceptance contract for the G1 WBC MJX/JAX
full-rollout migration. The first practical milestone is sweetpoint quality
with at least `>= 12x` steady-state speedup over the current sweetpoint
baseline. The final deployment target is real-time inference on a desktop
`RTX 4090`, but H100 is allowed for optimization and first milestone
acceptance.

Fast results that fail quality, replay, provenance, single GPU, or report gates
do not count as a successful migration.

## Scope

Hard-gate motions:

- `jump`
- `walk`

Formal baseline name:

- `g1_wbc_stage0_mujoco_warp_sweetpoint`

Backend under test:

- `mjx_canonical`, selected explicitly with `--mpc-backend mjx`

Reference backend:

- MuJoCo-Warp, selected with `--mpc-backend mujoco_warp`

The default backend for existing commands must remain MuJoCo-Warp. MJX must be
an explicit backend path and must fail loudly when its runtime or parity
requirements are unavailable.

## Hardware Rules

- One motion inference command may use only one visible GPU.
- Use `CUDA_VISIBLE_DEVICES=<one id>` and `--device cuda:0` for formal CUDA
  runs.
- The current machine may have four H100 GPUs, but one motion inference command
  must never use multiple GPUs.
- H100 reports must include runtime GPU names for Stage0 baseline, MJX, and
  replay rows when applicable.
- The first milestone may require `H100` in the runtime GPU name.
- The final real-time target must be measured on `RTX 4090` and must not be
  inferred from H100 timing.

## Timing Scope

The formal speed metric is steady-state end-to-end MPC wall-clock time.

Included in timed MJX steady-state measurement:

- policy observation assembly needed by MPC
- actor calls used in the planning path
- candidate sampling
- rollout
- scoring
- weighted update or selection
- required device synchronization

Excluded from steady-state timing but reported separately:

- first-call JIT compile
- model loading and MJX model construction
- warmup run
- rendering and video writing
- heavy debug artifact serialization

The report must include compile/init timing, warmup timing, steady-state repeat
timing, per-window timing, number of windows, visible GPU ids, and GPU model.

## Stage0 Sweetpoint Baseline

Stage0 freezes the MuJoCo-Warp denominator for quality and speed. The formal
runner is `scripts/run_g1_wbc_stage0_baseline.py`.

Required fixed command properties:

| Field | Required value |
| --- | --- |
| method | `g1_wbc_joint_global` |
| backend | `mujoco_warp` |
| max steps | `800` |
| samples | `512` |
| iterations | `2` |
| planning horizon steps | `40` |
| control steps | `20` |
| sampling mode | `knot` |
| knot count | `8` |
| temperature | `0.7` |
| root position sigma | `0.04` |
| root rotation sigma | `0.10` |
| joint sigma | `0.18` |
| smooth passes | `0` |
| command regularization | `0.0` |
| command smoothness | `0.0` |
| guided candidate | enabled |
| acceptance gate | enabled |
| save rollout | enabled |

The manifest must use explicit file paths for motion, checkpoint, and reward
weights. Aliases such as `--checkpoint bc` are not formal acceptance inputs
unless they have already been resolved to a concrete checkpoint file in the
manifest rows.

## Stage0 Manifest Requirements

Before MJX acceptance can launch, Stage0 must produce
`baseline_manifest.json` with schema version `1`.

Required top-level fields:

- `schema_version`
- `baseline_name`
- `motions`
- `seeds`
- `rows`
- `provenance.worktree_path`
- `provenance.git_commit`
- `input_sha256.jump_motion`
- `input_sha256.walk_motion`
- `input_sha256.checkpoint`
- `input_sha256.reward_weights`
- `baseline_envelopes`
- `promoted_seeds`

Required row evidence for each motion and seed:

- motion name and resolved motion path
- seed in `0`, `1`, `2`
- exact argv used for Stage0
- `status=ok`
- `returncode=0`
- `num_steps=800`
- `mpc_accepted=true`
- `accepted_windows=40`
- `mpc_used_baseline_fallback=false`
- `steady_state_wall_time_sec`
- `runtime_visible_devices`
- `runtime_gpu_name`
- artifact paths for `metrics_json`, `rollout_npz`, and `mpc_command_npz`
- `artifact_mtime_ns`
- `artifact_sha256`

Formal acceptance recomputes input hashes from the manifest row argv paths.
Hashes must be lowercase SHA256 hex strings. Acceptance also recomputes baseline
artifact hashes and rejects stale or replaced baseline files.

`baseline_envelopes` are frozen Stage0 metric envelopes per motion. They are the
only allowed quality reference for MJX and MuJoCo-Warp replay gates. Acceptance
must reject a manifest when frozen envelopes or `promoted_seeds` are missing or
do not match the Stage0 rows.

## Stage0 Baseline Eligibility

Every Stage0 repeat must satisfy:

- `status=ok`
- `returncode=0`
- exactly `800` evaluated steps
- `mpc_accepted=true`
- `accepted_windows=40`
- `mpc_used_baseline_fallback=false`
- no NaN or Inf in primary metrics
- all required artifacts exist at acceptance time
- artifact hashes match the manifest
- runtime evidence shows one visible GPU

The Stage0 group must contain exactly seeds `0`, `1`, and `2` for each hard-gate
motion.

### Jump Baseline Minimum

The `jump` Stage0 baseline must satisfy all group-level thresholds:

| Metric | Required standard |
| --- | ---: |
| success count | at least `2/3` |
| score mean | `>= -2.12` |
| root position error mean | `<= 0.065` |
| body global position error mean | `<= 0.075` |
| end-effector global position error mean | `<= 0.085` |
| contact mismatch rate | `<= 0.36` |
| control delta mean | `<= 0.46` |
| joint acceleration mean | `<= 225.0` |

### Walk Baseline Minimum

The `walk` Stage0 baseline must satisfy all group-level thresholds:

| Metric | Required standard |
| --- | ---: |
| success count | `3/3` |
| score mean | `>= -1.27` |
| root position error mean | `<= 0.075` |
| body global position error mean | `<= 0.080` |
| end-effector global position error mean | `<= 0.082` |
| end-effector local position error mean | `<= 0.038` |
| contact mismatch rate | `<= 0.155` |
| control delta mean | `<= 0.25` |
| joint acceleration mean | `<= 120.0` |

If either baseline group fails, the project has not established a valid
sweetpoint denominator and must stop formal MJX claims until Stage0 is repaired.

## Frozen Baseline Envelope

For each motion and primary metric, Stage0 freezes:

- `mean`
- `std`
- `min`
- `max`
- `median`
- `count` for success

The promoted seed is the best-score seed among the passing Stage0 group and is
stored in `promoted_seeds`. The promoted seed is used for review and debugging;
the formal quality denominator is still the full frozen envelope.

MJX acceptance must compare against the manifest envelope, not against a
baseline rerun selected after MJX results are known.

## MJX Functional Gate

The MJX backend must:

- run through the explicit `--mpc-backend mjx` path
- never silently fall back to MuJoCo-Warp during timed MJX inference
- complete `800` policy steps for each repeat
- export metrics compatible with the existing G1 WBC report schema
- export `rollout.npz`
- export `mpc_command.npz`
- report JIT compile/init timing separately from steady-state timing
- emit runtime GPU and device visibility diagnostics
- emit contact capacity diagnostics

Unsupported MJX model, contact, actor, observation, or scoring features must
produce a clear failure rather than an optimistic benchmark.

## MJX Quality Gate

Quality is evaluated over paired seeds `0`, `1`, and `2` for both motions.

Each MJX repeat must satisfy:

- `status=ok`
- `returncode=0`
- exactly `800` evaluated steps
- no baseline fallback
- no contact saturation
- no `max_contact_points` saturation
- no `max_geom_pairs` saturation
- required metrics present and finite
- required artifacts present and fresh
- one visible GPU

For each motion, MJX must satisfy:

| Metric group | Required standard |
| --- | ---: |
| success count | `>=` baseline success count |
| score mean, `jump` | `>= baseline mean - 0.08` |
| score worst repeat, `jump` | `>= baseline worst - 0.08` |
| score mean, `walk` | `>= baseline mean - 0.05` |
| score worst repeat, `walk` | `>= baseline worst - 0.05` |
| root position error mean | `<= 1.10x` baseline mean and below absolute cap |
| body global position error mean | `<= 1.10x` baseline mean and below absolute cap |
| end-effector global position error mean | `<= 1.10x` baseline mean and below absolute cap |
| end-effector local position error mean | `<= 1.10x` baseline mean and below absolute cap |
| contact mismatch rate | `<= baseline mean + 0.02` |
| contact false positive rate | `<= baseline mean + 0.02` |
| contact false negative rate | `<= baseline mean + 0.02` |
| bad floor contact rate | `<= baseline mean + 0.01` |
| control delta mean | `<= 1.15x` baseline mean |
| joint acceleration mean | `<= 1.15x` baseline mean |
| joint jerk mean | `<= 1.20x` baseline mean |

Absolute caps:

| Motion | root | body global | EE global | EE local | contact mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `jump` | `0.075` | `0.085` | `0.095` | `0.050` | `0.375` |
| `walk` | `0.082` | `0.088` | `0.090` | `0.042` | `0.175` |

Video or trajectory review remains a manual stop condition: an obvious physical
failure hidden by aggregate metrics invalidates promotion.

## MuJoCo-Warp Replay Gate

Every selected MJX command must be replayed through MuJoCo-Warp using
`--method replay_command`.

The MuJoCo-Warp replay gate must:

- use the MJX run's current `mpc_command.npz`
- not contribute to MJX speed timing
- produce `metrics.json` and `rollout.npz`
- report replay provenance in `metrics.json`
- use `replay_mode=shared_execute_backend`
- use the planned replay control steps
- report `num_command_frames >= num_replay_steps + 1`
- compare replay quality against the same frozen baseline envelope

If MJX metrics pass but MuJoCo-Warp replay fails, the result is a parity
failure. It is not a successful backend replacement.

## H100 Speed Gate

The first milestone passes speed only when both hard-gate motions reach:

```text
baseline_mean_steady_state_wall_time / mjx_mean_steady_state_wall_time >= 12.0
```

Required class on success:

- `pass_h100_milestone`

This claim is limited to one H100 and does not imply RTX 4090 real-time
readiness.

## RTX 4090 Real-Time Gate

The final deployment target passes only when all quality, replay, and metadata
gates pass on RTX 4090 and:

```text
evaluated_motion_duration_sec / mjx_steady_state_wall_time_sec >= 1.0
```

Required command intent:

```bash
--target 4090_realtime --required-gpu-name-fragment 4090
```

Each MJX row must include:

- `control_dt_sec`
- `evaluated_motion_duration_sec`
- runtime GPU name containing `4090`
- one visible GPU

Required class on success:

- `pass_4090_realtime`

## Formal Report Requirements

The formal report is `acceptance_report.json` from
`scripts/run_g1_wbc_mjx_acceptance.py`.

Required top-level sections:

- baseline manifest path
- frozen `baseline_envelopes`
- planned MJX and replay runs
- baseline rows
- MJX rows
- replay rows
- motion quality results
- replay results
- speed results
- realtime results
- timing summary
- contact summary
- final classification

Required final classifications:

| Class | Meaning |
| --- | --- |
| `pass_h100_milestone` | H100 quality, replay, and `>= 12x` speed gates all pass. |
| `pass_4090_realtime` | RTX 4090 quality, replay, and real-time gates all pass. |
| `quality_regression` | Speed passes but quality fails. |
| `speed_regression` | Quality passes but speed fails. |
| `parity_failure` | Replay, contact, model, actor, scoring, or static parity fails. |
| `invalid_benchmark` | Manifest, artifact, timing, GPU, fallback, saturation, or provenance evidence is invalid. |

## Non-Negotiable Failure Conditions

The result fails immediately if any of these occur:

- a motion inference command uses more than one GPU
- MJX timed inference silently falls back to MuJoCo-Warp
- Stage0 manifest lacks `baseline_envelopes` or `promoted_seeds`
- Stage0 baseline fails the motion-specific minimum thresholds
- Stage0 artifact hashes do not match
- required MJX or replay artifacts are missing or stale
- contact capacity saturates
- any hard-gate repeat has fewer than `800` evaluated steps
- selected MJX commands are not replayed in MuJoCo-Warp
- compile/init time is mixed into steady-state timing without separate report
- H100 timing is reported as RTX 4090 real-time readiness
- reward weights, horizon, iterations, samples, control steps, or motion suite
  are changed to make the first milestone easier

## Stop Conditions

Stop claiming progress and repair the lower layer first when:

- Stage0 cannot produce a formal sweetpoint manifest
- model/contact/actor/observation/scoring parity cannot be explained
- MJX quality passes only by weakening the frozen baseline envelope
- speedup passes only by excluding required MPC work from timing
- replay fails for selected MJX commands
- the report cannot reproduce the benchmark from paths, hashes, and argv

The full migration goal is complete only after the H100 milestone passes with
the above evidence and the RTX 4090 path has an explicit reproducible real-time
validation plan or result.
