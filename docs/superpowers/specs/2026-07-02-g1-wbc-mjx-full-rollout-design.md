# G1 WBC MJX Full-Rollout Migration Design

## Goal

Build a selectable MJX/JAX full-rollout backend for G1 WBC MPC that preserves
current sweetpoint quality while achieving at least `12x` steady-state
end-to-end MPC wall-clock speedup on a single GPU.

The first milestone is deliberately narrow:

- Motions: `jump` and `walk`.
- Backend target: MJX/JAX full rollout, not low-sample retuning.
- Hardware rule: one motion inference run uses one GPU only.
- Development hardware: H100 is allowed for optimization.
- Deployment direction: desktop RTX 4090 real-time inference.
- Speed metric: steady-state end-to-end MPC wall-clock, excluding first-call
  JIT compilation and initialization.
- Quality metric: strict baseline-relative quality gate against the current
  sweetpoint.

The repository's primary worktree must remain untouched. Development for this
track happens in:

```text
/data_team/junsong/model-based/spider_worktrees/g1-wbc-mjx-full-rollout
branch: codex/g1-wbc-mjx-full-rollout
```

## Baseline Contract

Stage 0 freezes the current MuJoCo-Warp sweetpoint as a manifest before any MJX
result is allowed to claim speedup or quality parity.

Default sweetpoint configuration:

```text
method: g1_wbc_joint_global
samples: 512
iterations: 2
planning_horizon_steps: 40
control_steps: 20
knot_count: 8
sampling_mode: knot
elite_frac: 0.125
temperature: 0.7
root_pos_sigma: 0.04
root_rot_sigma: 0.10
joint_sigma: 0.18
sigma_decay: 0.75
smooth_passes: 0
command_reg_weight: 0.0
command_smooth_weight: 0.0
guided_candidate: true
acceptance_gate: true
warm_start: false
warm_start_source: best
warm_start_decay: 1.0
reward_weights: g1_wbc_reward_weights_method_specific_v14_20260612.json
max_steps: 800
```

The manifest records the exact per-motion command, seed, metrics, wall time,
commit, model path, checkpoint, reward weights, effective legacy `mpc.config`,
GPU, driver, MuJoCo, MuJoCo-Warp, Torch, JAX, and MJX versions. If existing
project artifacts prove a different promoted sweetpoint for `walk`, Stage 0
records that per-motion baseline explicitly before MJX work proceeds. Later
gates compare only against the frozen manifest.

The historical `8192/h80` packaged baseline remains a reference for quality and
smoothness context, but it is not the speed denominator for the first MJX
milestone.

## Architecture

The migration keeps Spider's receding-horizon control structure and replaces
the per-window optimization hot path.

The retained Python-level loop is:

```text
for each MPC window:
    prepare current state, history, last action, and reference slice
    call one compiled JAX window optimizer
    receive updated controls and compact diagnostics
    execute the first control chunk
    shift the horizon
```

The compiled JAX window optimizer owns:

- residual command sampling
- knot interpolation
- `controls_to_qpos`
- command forward kinematics
- WBC observation construction
- WBC actor MLP inference
- MJX policy-in-the-loop rollout
- streaming score accumulation
- top-k or weighted-mean control update

This boundary is important. Replacing only `mj_step` with MJX while leaving
observation building, scoring, or sample selection in Python/Torch would not be
allowed to claim the `12x` milestone unless profiling proves those boundaries
are outside the steady-state hot path.

## Model Strategy

Use three model variants.

### `warp_canonical`

This is the current Spider MuJoCo-Warp model after dynamic WXY patching:

- terrain insertion
- visual geoms disabled for contact
- collision geoms enabled
- foot collision `condim=3`
- non-foot collision `condim=1`
- WBC actuator gains, limits, damping, frictionloss, and armature
- current solver option semantics

It is the semantic baseline, not a new asset.

### `mjx_canonical`

This is the first MJX target. It should preserve the current WXY model semantics
while using an MJX-friendly representation:

- visual meshes stay visual only
- contact geoms are primitive collision geoms
- broadphase contact set is constrained by explicit `<pair>` entries
- current WXY seven-capsule foot model is preserved first
- foot-floor, body-floor, and necessary self-contact pairs are generated
  intentionally
- `max_contact_points` and `max_geom_pairs` are fixed and monitored

This variant is the only one eligible for first-pass quality parity.

### `mjx_reference_variant`

This variant borrows patterns from:

```text
/data_team/junsong/tracking/Humanoid-GPT/storage/assets/unitree_g1_4010
```

Useful references:

- `g1_mjx.xml`
- `g1_mjx_track.xml`
- `scene_mjx_track.xml`

The reference model uses visual meshes, primitive collision geoms, disabled
default contact, and explicit contact pairs. It also uses a three-capsule plus
foot-box foot model. That makes it useful as an MJX performance and contact
structure template, but it changes foot semantics relative to the current WXY
model and is therefore a fallback/performance variant, not the first parity
target.

## Contact Policy

MJX-JAX is sensitive to mesh collision, broadphase branching, and large contact
sets on GPU SIMD hardware. The MJX backend therefore avoids open-ended contact
generation.

Rules:

- Mesh geoms never participate in collision.
- Primitive collision geoms are the only contact candidates.
- Foot-floor pairs are explicit.
- Non-foot floor pairs are limited to bodies used by scoring, termination, or
  failure diagnostics.
- Self-contact pairs are explicit and task-relevant.
- The backend records active contact counts, pair counts, saturation, and
  dropped-contact indicators.
- A run with saturated `max_contact_points` or `max_geom_pairs` cannot pass a
  quality gate.

Solver/contact tuning has two modes:

- `parity`: closest practical match to the current Spider MuJoCo-Warp semantics.
- `perf`: MJX-friendly solver settings such as lower iterations or disabled
  Euler damping, used only after parity tests pass.

## JAX Data Model

Static objects prepared once:

- `JaxModelBundle`: MJX model, geom/body/joint/actuator indices, contact pair
  metadata, default pose, action scale, and scoring body indices.
- `JaxMotionCache`: resampled motion tensors, reference body fields, qpos/qvel,
  contact labels, and padded indexing helpers.
- `JaxActorParams`: actor MLP weights and observation normalizer converted from
  the Torch checkpoint.
- `JaxMpcConfig`: fixed shape and scalar parameters for samples, horizon,
  control steps, knots, score weights, and update mode.

Dynamic per-window input:

- current qpos/qvel
- current observation history buffers
- last actor action
- current receding controls
- window start index
- PRNG key

Per-window output:

- updated controls
- execute chunk
- next observation history
- next last action
- final simulated state for the planner path when needed
- compact score and timing diagnostics
- optional selected-candidate trace in debug mode only

Full candidate traces are not returned in the normal benchmark path.

## Rollout And Scoring

The MJX rollout uses `jax.lax.scan` over policy steps. Each policy step performs:

1. observation construction
2. actor MLP inference
3. action-to-control mapping
4. `DECIMATION` MJX physics steps
5. body/contact extraction
6. streaming score-term updates

The streaming scorer computes the terms currently used by G1 WBC rewards and
metrics:

- root position and rotation error
- joint position error
- body global position and rotation error
- EE global and local position/rotation error
- hand/body local diagnostics when required by the selected reward weights
- foot contact mismatch, false positive, false negative, and switch rate
- bad floor contact and bad floor force excess
- contact force excess and force delta
- action delta
- control delta
- joint acceleration
- joint jerk

For final reports, the selected executed trajectory is converted back into the
existing `RolloutResult`-compatible schema so current metric and rendering tools
remain usable.

## Migration Phases

### Phase 0: Worktree And Baseline Lock

Deliverables:

- clean worktree on `codex/g1-wbc-mjx-full-rollout`
- `baseline_manifest.json` for `jump` and `walk`
- baseline metrics, commands, rollout outputs, and timing logs

Gate:

- no MJX speedup or quality claim can be made before the manifest exists

### Phase 1: MJX Model Build

Deliverables:

- `mjx_canonical` model generation path
- model metadata dump with joint/body/geom/actuator mapping
- explicit contact-pair manifest
- contact capacity settings

Gate:

- qpos/qvel/control dimensions match
- required bodies, joints, geoms, and actuators resolve by name
- mesh collision is absent from the contact set

### Phase 2: Static Parity Harness

Deliverables:

- FK parity checks on representative qpos frames
- actuator mapping checks
- foot contact probe suite
- contact capacity/saturation report

Gate:

- static body and EE position mismatch is at most `5e-4` meters per body
- static body and EE rotation mismatch is at most `5e-4` radians per body
- foot contact onset differs by at most one policy step for `walk` stance and
  `jump` landing probes
- no contact capacity saturation

### Phase 3: Actor And Observation Port

Deliverables:

- Torch checkpoint to JAX parameter converter
- JAX observation builder
- observation parity tests against Torch for fixed states and reference indices
- actor-output parity tests against Torch

Gate:

- observation vector shape is `886`
- fixed-state actor outputs match Torch with max absolute error at most `1e-4`
- history buffer behavior matches first-frame backfill semantics

### Phase 4: No-MPC Rollout Parity

Deliverables:

- MJX policy-only rollout for `jump` and `walk`
- MuJoCo-Warp policy-only comparison report
- per-term metric deltas

Gate:

- no divergence before contact-sensitive events
- action/control traces are close enough to attribute remaining differences to
  model/contact behavior rather than actor or observation bugs

### Phase 5: Streaming Scoring Port

Deliverables:

- JAX streaming scorer
- scorer parity report against Torch `compute_rollout_scores`
- score-weight compatibility for `g1_wbc_joint_global`

Gate:

- score and primary score terms match Torch on replayed traces with max absolute
  error at most `1e-4`
- missing or unused score terms are explicitly reported as zero-weight only

### Phase 6: JAX Window Optimizer

Deliverables:

- fixed-shape compiled optimizer for one MPC window
- sampling, interpolation, score, and weighted-mean update parity tests
- window diagnostics compatible with existing reports

Gate:

- one warm JIT call completes a full window without Python step-loop callbacks
- repeated same-shape calls do not recompile
- update mode matches current `weighted_mean` semantics

### Phase 7: Spider Integration

Deliverables:

- selectable backend flag such as `--mpc-backend mujoco_warp|mjx`
- MJX result metadata in the existing output schema
- final selected trajectory export compatible with rendering and metrics tools

Gate:

- the default backend remains MuJoCo-Warp
- existing MuJoCo-Warp tests and CLI behavior remain unchanged
- MJX backend failures surface clear diagnostics instead of silent fallback

### Phase 8: Quality Gate

Hard gate for both `jump` and `walk`:

- full `800` policy steps complete
- no baseline fallback
- no contact capacity saturation
- score remains inside the frozen baseline envelope
- root/body/EE global tracking has only small baseline-relative regression
- contact mismatch has only small absolute regression
- smoothness metrics, including control delta and joint acceleration, have only
  small baseline-relative regression
- video review does not reveal a failure hidden by aggregate metrics

The exact numeric tolerances are frozen in the baseline manifest before formal
MJX evaluation. Suggested initial tolerances:

- global tracking terms: at most `1.10x` baseline
- contact mismatch: at most baseline `+0.02` absolute
- control delta and joint acceleration: at most `1.15x` baseline
- score: no worse than the baseline repeated-run lower envelope

The same MJX-selected command is replayed offline in MuJoCo-Warp. That replay is
not counted in speed, but it must pass quality review for MJX to be considered a
replacement backend. If MJX passes its own metrics but MuJoCo-Warp replay fails,
the result is a model/contact parity failure.

### Phase 9: Speed Gate

Formal speed measurement:

- one visible GPU via `CUDA_VISIBLE_DEVICES`
- H100 allowed for first milestone validation
- JIT warmup excluded from milestone timing
- compile/init time reported separately
- at least three steady-state repeats per motion
- sampling, rollout, scoring, action selection, and necessary synchronization
  included
- rendering and disk-heavy artifact writing excluded from measured inference
  time

Pass condition:

- `jump` steady-state end-to-end speedup is at least `12x`
- `walk` steady-state end-to-end speedup is at least `12x`
- neither motion trades away the strict quality gate

## Fallbacks

### Quality Failure

Debug order:

1. observation and actor parity
2. actuator and default pose mapping
3. static FK parity
4. foot-floor contact timing
5. non-foot floor contact filtering
6. contact force scale
7. solver/contact parameters
8. `mjx_reference_variant` comparison

Only after these steps should the project consider a hybrid design.

### Speed Failure

Debug order:

1. confirm no recompilation
2. profile host-device transfers
3. remove full trace returns from the hot path
4. profile actor, observation, physics, scoring, and update separately
5. reduce explicit pair count without breaking quality
6. tune `max_contact_points` and `max_geom_pairs`
7. evaluate solver settings in `perf` mode

Do not reduce samples, horizon, iterations, or reward weights to make the first
MJX milestone pass.

### Hybrid Fallback

If MJX is fast but final-score contact parity fails:

1. MJX proposes all candidates.
2. MJX keeps top-K plus the zero-residual baseline.
3. MuJoCo-Warp re-scores top-K.
4. MuJoCo-Warp winner is executed.

This is a fallback path, not the main first milestone.

## Testing Strategy

Unit-level:

- model mapping checks
- checkpoint conversion checks
- observation parity checks
- actor parity checks
- scoring parity checks
- sampling and weighted-mean update checks

Integration-level:

- static FK/contact probes
- no-MPC rollout parity
- saved-command replay parity
- one-window optimizer smoke tests
- full `jump` and `walk` evaluation with output schema validation

Benchmark-level:

- baseline lock run
- compile-time report
- steady-state repeat report
- quality table
- speed table
- MuJoCo-Warp replay confirmation

## Non-Goals

- Do not tune sweetpoint parameters for this milestone.
- Do not replace the default MuJoCo-Warp backend.
- Do not broaden the hard gate beyond `jump` and `walk`.
- Do not use multiple GPUs for one motion inference run.
- Do not claim RTX 4090 real-time readiness from H100-only timing.
- Do not move to MPPI/iLQR/SQP or another optimizer family in this migration.
- Do not make the Humanoid-GPT model the baseline model.

## Success Criteria

The first milestone succeeds only when all are true:

- `mjx_canonical` passes model, observation, actor, rollout, and scoring parity
  gates.
- `jump` and `walk` pass strict baseline-relative quality gates.
- steady-state end-to-end MPC wall-clock speedup is at least `12x` on one H100
  for both motions.
- compile/init costs are reported separately.
- the selected MJX commands replay in MuJoCo-Warp without invalidating quality.
- the backend is selectable and the existing MuJoCo-Warp path remains intact.
