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

The Stage0 sweetpoint denominator is narrower than the generic MuJoCo-Warp
backend: it must use `--mpc-optimizer legacy` so the historical
`G1WbcMpcConfig` guided candidate, acceptance gate, regularization, and
per-window metadata semantics are actually active.

MJX acceptance candidates must convert that frozen Stage0 argv to
`--mpc-backend mjx --mpc-optimizer generic`; MJX backend runs with
`--mpc-optimizer legacy` are invalid setup, not benchmark failures.

## Hardware Rules

- One motion inference command may use only one visible GPU.
- Use `CUDA_VISIBLE_DEVICES=<one id>` and `--device cuda:0` for formal CUDA
  runs.
- The single visible GPU for formal Stage0 and acceptance timing must have no
  pre-existing compute processes. The Stage0 runner fails fast on GPU
  contention because a busy H100 would invalidate denominator timing.
- The current machine may have four H100 GPUs, but one motion inference command
  must never use multiple GPUs.
- Do not run two MJX/JAX motion inference commands concurrently on the same
  visible GPU during smoke or formal acceptance. Each command can still satisfy
  the one-GPU rule, but concurrent JAX warmup on one H100 can exhaust memory or
  fail cuSolver handle creation. Run such commands serially on one GPU, or give
  independent commands different `CUDA_VISIBLE_DEVICES` values.
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
| optimizer | `legacy` |
| preset | `aggressive` |
| max steps | `800` |
| samples | `512` |
| iterations | `2` |
| planning horizon steps | `40` |
| control steps | `20` |
| sampling mode | `knot` |
| knot count | `8` |
| elite fraction | `0.125` |
| temperature | `0.7` |
| root position sigma | `0.04` |
| root rotation sigma | `0.10` |
| joint sigma | `0.18` |
| sigma decay | `0.75` |
| smooth passes | `0` |
| command regularization | `0.0` |
| command smoothness | `0.0` |
| guided root position gain | `0.50` |
| guided root rotation gain | `0.50` |
| guided joint gain | `0.50` |
| guided root position clip | `0.05` |
| guided root rotation clip | `0.12` |
| guided joint clip | `0.35` |
| guided candidate | enabled |
| acceptance gate | enabled |
| warm start | disabled |
| warm start source | `best` |
| warm start decay | `1.0` |
| nconmax per env | `512` |
| njmax per env | `2048` |
| save rollout | enabled |

Each baseline row `metrics.json` must include `mpc.config` matching the
effective legacy `G1WbcMpcConfig` derived from the row argv. That config must
also record the fixed defaults not exposed as Stage0 argv: `min_root_pos_sigma`
`0.002`, `min_root_rot_sigma` `0.004`, `min_joint_sigma` `0.008`, and
`freeze_first_frame` enabled. The config `seed` must equal the row seed.
`reward_weights` are recorded separately as `reward_weight_source` and
`reward_weights`, not inside `mpc.config`.

The manifest must use explicit file paths for motion, checkpoint, and reward
weights. Aliases such as `--checkpoint bc` are not formal acceptance inputs
unless they have already been resolved to a concrete checkpoint file in the
manifest rows.

The Stage0 motion inputs must be motions for which the current MuJoCo-Warp
sweetpoint can actually pass the Stage0 baseline eligibility gates below. A
file path and hash are not sufficient evidence: the input set is formal only
after the real `baseline_manifest.json` contains passing `baseline_envelopes`
and `promoted_seeds`.

Interrupted Stage0 runs may be resumed with `--reuse-existing-ok`, but only for
rows whose existing `metrics.json`, `rollout.npz`, and `mpc_command.npz` are
present, report `800` steps, `40` accepted windows, no baseline fallback, and
match the current command provenance for motion, checkpoint, method, backend,
optimizer, and effective `mpc.config`. Incomplete or provenance-mismatched rows
must be rerun.

Historical sweetpoint evidence available in `wbc_results` comes from the
versioned testbed motion set now stored under
`wbc_results/assets/motion_data/{jump,walk}/motion.npz` and from the
single-motion `homejrhangmr` run recorded under
`2026-06-26-homejrhangmr-three-way-wjs`. These motion assets are the official
migrated copies of the historical testbed package inputs, but a file path and
hash still are not enough to accept Stage0; the current code must reproduce the
six-row MuJoCo-Warp sweetpoint manifest and pass the gates below.

Current input status in this worktree on 2026-07-03:

| Input | Path | SHA256 | Status |
| --- | --- | --- | --- |
| WBC MLP checkpoint | `/data_team/junsong/model-based/wbc_results/assets/checkpoints/model_8000.pt` | `98738b9214d12146dc7f4669cb65dfde9d835f4a133e5f2cbaef4e60b1e5b88f` | accepted checkpoint input |
| reward weights | `/data_team/junsong/model-based/wbc_results/g1_body_tracking_wbc/spider/2026-06-23-mechanism-quality-speed-wjs/configs/g1_wbc_reward_weights_method_specific_v14_20260612.json` | `bb0490a71a27a29480f13ce77bc00c13a900845f0f52641b5ec20d51ebaa535d` | accepted reward input |
| jump motion | `/data_team/junsong/model-based/wbc_results/assets/motion_data/jump/motion.npz` | `07b3b8e1bf9ba3f94dfbe552819cd792f81a06a3c4ff6e5029b55b2897b7c544` | official migrated testbed input; current-code Stage0 reproduction still must pass |
| walk motion | `/data_team/junsong/model-based/wbc_results/assets/motion_data/walk/motion.npz` | `a9baaa714d61da19c6114077cf0c919c965ad6802f770cc83ed695396c4c8c9f` | official migrated testbed input; current-code Stage0 reproduction still must pass |

The 2026-07-03 diagnostic run
`/data_team/junsong/model-based/g1_wbc_mjx_runs/stage0_baseline_wbc_mlp_20260703_h100_gpu1`
completed `jump/seed_0` and `jump/seed_1`; those two rows alone made the jump
group mathematically unable to pass (`root_pos_error_mean` of `0.567` and
`0.134` versus the group mean cap `0.065`). Because the motion files are now
confirmed to be official migrated testbed inputs, that run is treated as a
current-code Stage0 reproduction failure, not as evidence that the assets are
invalid. See
`docs/tasks/g1_wbc_mjx_full_rollout/stage0_input_diagnosis_20260703.md`.

Historical single-motion sweetpoint reference:

| Input | Path | SHA256 |
| --- | --- | --- |
| homejrhangmr motion | `/data_zcy/wxy/test_motion/homejrhangmr_dataset_pbhc_contact_maskACCADFemale1Walking_c3dB19-walktopickupbox_posespkl/motion.npz` | `1fa518f5b80b675e3a89ff8aad501e031b64cd7ab9c0b5a1a9e2cd36331ff829` |
| sweetpoint rollout artifact | `/data_team/junsong/model-based/wbc_results/assets/results/2026-06-26-homejrhangmr-three-way-wjs/sweetpoint_rollout.npz` | indexed in `configs/rollout_index.json` |
| sweetpoint command artifact | `/data_team/junsong/model-based/wbc_results/assets/results/2026-06-26-homejrhangmr-three-way-wjs/sweetpoint_mpc_command.npz` | indexed in `configs/rollout_index.json` |

The checkpoint must be a WBC MLP actor checkpoint with `actor_state_dict`,
`obs_normalizer.*`, and `mlp.*` weights. The user-provided SparseTrack
Transformer checkpoint
`/data_team/junsong/general_controller/ScaleTrack-mj/ScaleTrack/logs/rsl_rl/mjlab_g1_bfm_transformer_tracking_exp/mjlab_myrsl_g1_global_transformer_sparse_tracking/model_11800.pt`
is a valid reference for `SparseTrack-Tracking-Flat-G1-Global-Transformer-v0`,
but it is not a valid Stage0 checkpoint for this WBC MLP runner until a separate
SparseTrack Transformer policy/observation adapter exists.

The raw `/data_team/zcy/motion_data/...` motions are allowed for loader and
runtime smoke tests, but they are not byte-identical to the packaged testbed
`jump` and `walk` motions from the historical baseline. They must not replace
Stage0 inputs unless the Stage0 manifest explicitly records their paths and
hashes and the resulting baseline passes the gates below.

Verified Stage0 dry-run command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_stage0_baseline.py \
  --jump-motion /data_team/junsong/model-based/wbc_results/assets/motion_data/jump/motion.npz \
  --walk-motion /data_team/junsong/model-based/wbc_results/assets/motion_data/walk/motion.npz \
  --motion-type isaaclab \
  --checkpoint /data_team/junsong/model-based/wbc_results/assets/checkpoints/model_8000.pt \
  --reward-weights /data_team/junsong/model-based/wbc_results/g1_body_tracking_wbc/spider/2026-06-23-mechanism-quality-speed-wjs/configs/g1_wbc_reward_weights_method_specific_v14_20260612.json \
  --output-dir /data_team/junsong/model-based/g1_wbc_mjx_runs/stage0_dryrun_testbed_wbc_mlp \
  --device cuda:0 \
  --dry-run
```

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

`rollout.npz` must be a valid NPZ containing the rollout core schema:
`qpos`, `qvel`, body position/quaternion/linear-velocity/angular-velocity
arrays, `actions`, `controls`, foot contact arrays, floor contact arrays,
`ref_indices`, and scalar `dt`, with shapes consistent with
`num_steps + 1` frames. Invalid rollout content is reported as
`rollout_npz_schema`.

`mpc_command.npz` must be a valid NPZ containing `refined_qpos`,
`candidate_scores`, command joint/body trajectories, and command
`qpos`/`qvel` trajectories with shapes consistent with the same frame count.
Invalid command content is reported as `mpc_command_npz_schema`.
Within a command NPZ, `refined_qpos` must match `command_qpos_trajectory`
within the runtime acceptance tolerance. Mismatch is reported as
`mpc_command_qpos_mismatch`.
`command_qvel_trajectory` must match the finite-difference MuJoCo qvel
trajectory derived from `refined_qpos` at the policy timestep. Mismatch is
reported as `mpc_command_qvel_mismatch`.
For MJX and Stage0 MPC rows, `rollout.npz["qpos"][:, 0]` must match
`mpc_command.npz["refined_qpos"]` within the runtime acceptance tolerance.
Mismatch is reported as `mpc_rollout_qpos_mismatch`.

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
- reject a replay saved command that does not match the same motion and seed's
  MJX artifact, reported as `replay_saved_command_source`
- record `saved_command_sha256` in replay `metrics.json` and match it to the
  MJX command artifact hash, reported as `replay_saved_command_hash`
- not contribute to MJX speed timing
- produce `metrics.json` and `rollout.npz`
- report replay provenance in `metrics.json`
- report top-level `method=replay_command` in `metrics.json`, with mismatch
  reported as `replay_metrics_provenance`
- use `replay_mode=shared_execute_backend`
- use the planned replay control steps
- report `num_command_frames >= num_replay_steps + 1`
- match `num_command_frames` to the saved command NPZ frame count, reported as
  `replay_command_npz_frames`
- require replay `rollout.npz["ref_indices"]` to stay within the saved command
  frame range, reported as `replay_rollout_ref_indices`
- compare replay quality against the same frozen baseline envelope

If MJX metrics pass but MuJoCo-Warp replay fails, the result is a parity
failure. It is not a successful backend replacement.

## H100 Speed Gate

The first milestone passes speed only when both hard-gate motions reach:

```text
baseline_mean_steady_state_wall_time / mjx_mean_steady_state_wall_time >= 12.0
```

It must also pass the seed-paired worst-repeat gate:

```text
worst_speedup = min(baseline_seed_wall_time / mjx_same_seed_wall_time) >= 12.0
```

The report uses `speedup` for the mean ratio and `worst_speedup` for the
seed-paired lower bound. A failure of only the worst-repeat gate is reported as
`speedup_worst` so mean speed and outlier speed failures are distinguishable.

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

Interrupted acceptance runs may resume with `--reuse-existing-ok`. During a
real run the runner must write `acceptance_report.partial.json` after each
completed MJX/replay pair. Reuse is valid only for rows from an existing final
or partial report whose status is `ok`, whose command argv/text match the
current plan, whose artifact hashes still match disk, and whose `metrics.json`
provenance matches the planned method, motion, checkpoint, device, max steps,
backend, optimizer or replay command source. Any incomplete, failed,
hash-mismatched, schema-invalid, or provenance-mismatched row must be rerun.

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
