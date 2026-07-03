# G1 WBC MJX Full-Rollout Acceptance Criteria

## Purpose

This document defines the final evaluation standard for the MJX/JAX
full-rollout G1 WBC backend. It also defines what qualifies as the current
sweetpoint baseline. A result that is fast but fails these quality criteria is
not a successful migration.

The criteria apply in the worktree:

```text
/data_team/junsong/model-based/spider_worktrees/g1-wbc-mjx-full-rollout
```

## Scope

Hard-gate motions:

- `jump`
- `walk`

Backend under test:

- `mjx_canonical`, the MJX/JAX full-rollout backend that preserves current WXY
  model semantics before using MJX-specific simplifications.

Hardware rules:

- One motion inference run may use only one GPU.
- H100 is allowed for the first implementation milestone.
- RTX 4090 is the final deployment target.
- Four available H100s may be used for separate experiments, but never for one
  motion inference run.
- The H100 milestone report must include the runtime GPU name for every MJX
  repeat, and the default formal acceptance gate requires the name to contain
  `H100`. Later RTX 4090 validation should use a different explicit required
  GPU-name fragment rather than reusing the H100 milestone label.

Timing rules:

- The speed metric is steady-state end-to-end MPC wall-clock.
- Timing includes sampling, rollout, scoring, action selection, and required
  synchronization.
- Timing excludes first-call JIT compilation, model initialization, rendering,
  video writing, and heavy debug artifact writing.
- JIT compilation and initialization time must be reported separately.

## Sweetpoint Baseline Definition

The sweetpoint baseline is the frozen MuJoCo-Warp reference used as the quality
and speed denominator for this migration.

Default baseline configuration:

```text
method: g1_wbc_joint_global
samples: 512
iterations: 2
planning_horizon_steps: 40
control_steps: 20
knot_count: 8
sampling_mode: knot
temperature: 0.7
root_pos_sigma: 0.04
root_rot_sigma: 0.10
joint_sigma: 0.18
smooth_passes: 0
command_reg_weight: 0.0
command_smooth_weight: 0.0
guided_candidate: true
acceptance_gate: true
reward_weights: g1_wbc_reward_weights_method_specific_v14_20260612.json
max_steps: 800
checkpoint: bc
```

If Stage 0 proves that `walk` has an already-promoted lower-cost sweetpoint that
dominates the default `s512` configuration on quality-per-second, the baseline
manifest may use that `walk`-specific baseline. The manifest must state this
explicitly and record the exact configuration.

The historical `8192/h80` baseline is a quality reference, not the speed
denominator for this migration.

Concrete local inputs verified for this worktree on 2026-07-03:

```text
jump_motion: /data_team/junsong/model-based/wbc_results/assets/motion_data/jump/motion.npz
walk_motion: /data_team/junsong/model-based/wbc_results/assets/motion_data/walk/motion.npz
checkpoint: /data_team/junsong/model-based/wbc_results/assets/checkpoints/model_8000.pt
reward_weights: /data_team/junsong/model-based/wbc_results/g1_body_tracking_wbc/spider/2026-06-23-mechanism-quality-speed-wjs/configs/g1_wbc_reward_weights_method_specific_v14_20260612.json
```

The checkpoint is an RSL-RL training package whose `actor_state_dict` contains
`obs_normalizer.*` and `mlp.*` keys, so it is compatible with the current WBC
MLP policy loader. Formal Stage 0 runs in this worktree should pass these paths
explicitly instead of relying on the runner defaults, because the old packaged
testbed path is not present in this worktree. The same reward-weight file also
exists under the `2026-06-17-testbed-motion-baselines-xwj` artifact directory
with the same contents, but the `2026-06-23` path is the preferred explicit
formal input because it is the one referenced by current sweetpoint commands.

Verified dry-run command:

```bash
PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_stage0_baseline.py \
  --jump-motion /data_team/junsong/model-based/wbc_results/assets/motion_data/jump/motion.npz \
  --walk-motion /data_team/junsong/model-based/wbc_results/assets/motion_data/walk/motion.npz \
  --checkpoint /data_team/junsong/model-based/wbc_results/assets/checkpoints/model_8000.pt \
  --reward-weights /data_team/junsong/model-based/wbc_results/g1_body_tracking_wbc/spider/2026-06-23-mechanism-quality-speed-wjs/configs/g1_wbc_reward_weights_method_specific_v14_20260612.json \
  --output-dir /tmp/g1_wbc_stage0_formal_inputs_dryrun \
  --dry-run
```

## Baseline Manifest Requirements

Before any MJX result can be evaluated, Stage 0 must create a baseline manifest
for `jump` and `walk`.

Required fields:

- motion name and motion path
- baseline configuration
- seed and repeat id
- commit hash
- worktree path
- model path
- checkpoint path or alias
- reward weight path and hash
- MuJoCo, MuJoCo-Warp, Torch, CUDA, driver, JAX, and MJX versions when available
- GPU model and visible GPU ids
- wall-clock timing
- compile/init timing if any
- `metrics.json`
- `rollout.npz`
- `mpc_command.npz`
- accepted windows
- fallback status
- contact capacity/saturation diagnostics if available

All manifest artifact paths must resolve to existing files at acceptance time.
The MJX acceptance runner treats a missing baseline `metrics.json`,
`rollout.npz`, or `mpc_command.npz` as an invalid benchmark even if the manifest
contains otherwise valid metric fields.

The acceptance runner must also reject non-formal manifests before launching MJX
runs. A formal manifest has schema version `1`, baseline name
`g1_wbc_stage0_mujoco_warp_sweetpoint`, motions `jump/walk`, seeds `0/1/2`, and
per-row argv values matching the frozen sweetpoint configuration. The row argv
must contain explicit existing files for `--motion`, `--checkpoint`, and
`--mpc-reward-weights`; an alias such as `--checkpoint bc` is not sufficient for
formal acceptance in this worktree.

The manifest must contain at least three repeated runs per hard-gate motion,
using seeds `0`, `1`, and `2`, unless a historical artifact is explicitly
declared as the only available baseline. Historical single-run artifacts may be
used for context but not as the only final quality gate.

## Baseline Eligibility Criteria

A sweetpoint baseline is eligible only if it satisfies all global criteria and
the motion-specific minimum standards below.

### Global Baseline Criteria

Each baseline repeat must satisfy:

- `status=ok`
- `num_steps=800`
- `mpc_accepted=true`
- `accepted_windows=40`
- `mpc_used_baseline_fallback=false`
- no NaN or Inf in primary metrics
- output contains `metrics.json`, `rollout.npz`, and `mpc_command.npz`, and
  the manifest paths for those artifacts still exist as files

The baseline group must satisfy:

- at least three repeats per motion
- no missing primary metrics
- no visible severe failure in video review for the promoted baseline repeat
- no unexplained timing outlier greater than `1.25x` the median repeat time

### `jump` Baseline Minimum

The `jump` sweetpoint baseline must satisfy these group-level standards:

| Metric | Required standard |
| --- | ---: |
| success count | at least `2/3` repeats |
| score mean | `>= -2.12` |
| root position error mean | `<= 0.065` |
| body global position error mean | `<= 0.075` |
| EE global position error mean | `<= 0.085` |
| contact mismatch rate | `<= 0.36` |
| control delta mean | `<= 0.46` |
| joint acceleration mean | `<= 225.0` |
| mean 800-step wall time | recorded; used as speed denominator |

These thresholds reflect the current `s512/i2/h40/c20/k8` jump sweetpoint
evidence: score around `-2.05`, global sum around `0.19`, contact around
`0.33-0.35`, and 4070 Laptop wall time around `188.7 s`. They are not a target
for MJX to tune against; they define whether the reference is good enough to be
called the sweetpoint baseline.

### `walk` Baseline Minimum

The `walk` sweetpoint baseline must satisfy these group-level standards:

| Metric | Required standard |
| --- | ---: |
| success count | `3/3` repeats |
| score mean | `>= -1.27` |
| root position error mean | `<= 0.075` |
| body global position error mean | `<= 0.080` |
| EE global position error mean | `<= 0.082` |
| EE local position error mean | `<= 0.038` |
| contact mismatch rate | `<= 0.155` |
| control delta mean | `<= 0.25` |
| joint acceleration mean | `<= 120.0` |
| mean 800-step wall time | recorded; used as speed denominator |

These thresholds match the known `walk` low-cost sweetpoint envelope where
`best_s128` reached score `-1.2671`, root `0.0732`, body global `0.0776`,
EE global `0.0791`, contact `0.1500`, control delta `0.2441`, and joint
acceleration `117.789`. A higher-sample `walk` baseline is allowed, but it must
meet or beat these minima.

## Baseline Envelope

After baseline eligibility is established, Stage 0 computes a baseline envelope
for each metric and motion:

- `mean`
- `std`
- `min`
- `max`
- `median`
- promoted repeat id
- lower quality bound for score
- upper quality bound for error metrics

The baseline envelope is frozen in the manifest. MJX quality gates compare
against this frozen envelope, not against later reruns chosen after seeing MJX
results.

## Final MJX Implementation Pass Criteria

The MJX backend passes the first full-rollout milestone only if all sections in
this chapter pass.

### Functional Criteria

The implementation must:

- expose a selectable backend flag such as `--mpc-backend mjx`
- keep MuJoCo-Warp as the default backend
- run `jump` and `walk` for `800` policy steps
- produce existing-compatible metrics and rollout summaries
- export the selected command in a form replayable by MuJoCo-Warp
- report compile/init time separately from steady-state inference time
- fail loudly on unsupported model/contact/scoring features
- avoid silent fallback to MuJoCo-Warp during MJX timing

### Model And Contact Criteria

The MJX run must satisfy:

- all required bodies, joints, geoms, and actuators resolve by name
- no visual mesh participates in contact
- `mjx_canonical` uses primitive contact geoms
- explicit contact-pair manifest is emitted
- no `max_contact_points` saturation
- no `max_geom_pairs` saturation
- contact-pair count and active-contact count are logged per run
- `mjx_reference_variant` is not used for final replacement claims unless the
  claim is explicitly labeled as a changed-model variant

### Parity Criteria

Static parity must satisfy:

- body and EE position mismatch `<= 5e-4 m`
- body and EE rotation mismatch `<= 5e-4 rad`
- action-to-control max absolute mismatch `<= 1e-5`
- actor output max absolute mismatch versus Torch `<= 1e-4`
- scoring-term max absolute mismatch on replayed traces `<= 1e-4`

Contact probe parity must satisfy:

- `walk` foot contact onset differs by at most one policy step
- `jump` landing contact onset differs by at most one policy step
- no new persistent non-foot floor contact appears in static or no-MPC probes

### Quality Criteria

Quality is evaluated over paired seeds `0`, `1`, and `2` for each hard-gate
motion, unless the baseline manifest declares a different fixed repeat set
before MJX evaluation.

For each motion, MJX must satisfy:

- success count is at least the baseline success count
- no repeat has fewer than `800` steps
- no repeat uses baseline fallback
- no repeat has contact capacity saturation
- mean score is no worse than baseline mean by more than the motion allowance
- primary tracking metrics do not exceed the baseline-relative caps
- smoothness metrics do not exceed the baseline-relative caps
- video review shows no new catastrophic artifact

Motion-specific score allowances:

| Motion | Mean score allowance | Worst-repeat score allowance |
| --- | ---: | ---: |
| `jump` | baseline mean minus `0.08` | baseline worst minus `0.08` |
| `walk` | baseline mean minus `0.05` | baseline worst minus `0.05` |

Metric caps:

| Metric group | Required MJX standard |
| --- | ---: |
| root position error mean | `<= 1.10x` baseline mean and below absolute cap |
| body global position error mean | `<= 1.10x` baseline mean and below absolute cap |
| EE global position error mean | `<= 1.10x` baseline mean and below absolute cap |
| EE local position error mean | `<= 1.10x` baseline mean and below absolute cap |
| contact mismatch rate | `<= baseline mean + 0.02` |
| contact false positive rate | `<= baseline mean + 0.02` |
| contact false negative rate | `<= baseline mean + 0.02` |
| bad floor contact rate | `<= baseline mean + 0.01` |
| control delta mean | `<= 1.15x` baseline mean |
| joint acceleration mean | `<= 1.15x` baseline mean |
| joint jerk mean | `<= 1.20x` baseline mean |

Absolute caps:

| Motion | root | body global | EE global | EE local | contact |
| --- | ---: | ---: | ---: | ---: | ---: |
| `jump` | `0.075` | `0.085` | `0.095` | `0.050` | `0.375` |
| `walk` | `0.082` | `0.088` | `0.090` | `0.042` | `0.175` |

The absolute caps prevent a weak baseline run from making the relative gate too
easy.

### MuJoCo-Warp Replay Criteria

For each MJX run that passes MJX-side quality:

- replay the selected `mpc_command.npz` in MuJoCo-Warp
- do not include replay time in MJX speed
- compute the same primary metrics
- compare replay metrics against the same baseline envelope

The replay must satisfy the same quality criteria, except contact force scale
may be reported separately if contact timing and tracking pass. If MJX metrics
pass but MuJoCo-Warp replay fails, the implementation is classified as
`mjx_contact_or_model_parity_failure`, not as a successful backend.

### Speed Criteria

For the H100 first milestone:

| Motion | Required speedup |
| --- | ---: |
| `jump` | `>= 12x` steady-state end-to-end MPC wall-clock |
| `walk` | `>= 12x` steady-state end-to-end MPC wall-clock |

Speedup is:

```text
baseline_mean_steady_state_wall_time / mjx_mean_steady_state_wall_time
```

The denominator is measured after JIT warmup. The numerator is the frozen
MuJoCo-Warp sweetpoint baseline mean from Stage 0, measured with the same timing
scope.

Minimum timing report:

- first-call compile/init time
- warmup run time
- steady-state repeat times
- mean, median, min, max, std
- per-window timing distribution
- number of windows
- visible GPU id
- GPU model
- peak GPU memory if available

For the RTX 4090 final deployment target:

- same quality gates
- same one-GPU rule
- no H100-only code path
- report real-time factor against the motion duration
- target is at least `1.0x` real-time end-to-end steady-state inference

The H100 milestone can pass before the RTX 4090 target passes, but it must not
be described as desktop real-time readiness.

## Classification

Every formal evaluation is assigned exactly one class.

| Class | Meaning |
| --- | --- |
| `pass_h100_milestone` | Quality gates pass and H100 speedup is `>= 12x` on both motions. |
| `pass_4090_realtime` | Quality gates pass and RTX 4090 reaches at least `1.0x` real time. |
| `quality_regression` | Speed passes but one or more quality gates fail. |
| `speed_regression` | Quality passes but speed is below target. |
| `parity_failure` | Static, actor, scoring, contact, or replay parity fails. |
| `invalid_benchmark` | Missing manifest, fallback used, saturation occurred, wrong GPU scope, or timing scope is incomplete. |

## Required Report Tables

The final report must include:

1. Baseline manifest summary.
2. Baseline eligibility table.
3. MJX quality table by motion and seed.
4. Baseline-relative quality ratio table.
5. MuJoCo-Warp replay table.
6. H100 speed table.
7. RTX 4090 speed table when available.
8. Contact capacity/saturation table.
9. Compile/init timing table.
10. Final classification.

## Non-Negotiable Failure Conditions

The result fails regardless of average score if any of these occur:

- uses more than one GPU for one motion inference run
- silently falls back to MuJoCo-Warp in the MJX timed path
- changes samples, horizon, iterations, reward weights, or baseline motion suite
  to satisfy the first milestone
- uses `mjx_reference_variant` while claiming unchanged WXY sweetpoint semantics
- omits MuJoCo-Warp replay of selected MJX commands
- omits compile/init timing
- reports cold-start and steady-state timing as one number
- has contact capacity saturation
- has fewer than `800` rollout steps on a hard-gate motion
- lacks enough metadata to reproduce the benchmark
