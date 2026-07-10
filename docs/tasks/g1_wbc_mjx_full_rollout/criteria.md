# G1 WBC MJX RTX 4090 Acceptance Criteria

## Purpose and Success Definition

This file is the normative acceptance contract for the G1 WBC MJX
full-rollout migration on the current single-RTX-4090 workstation.

The migration succeeds only when MPC plus policy-in-the-loop:

1. substantially improves over the current No-MPC policy rollout;
2. meets or beats MuJoCo-Warp sweetpoint quality;
3. preserves control, smoothness, force, and physical-behavior guardrails;
4. reproduces across seeds and independent processes;
5. replays through MuJoCo-Warp with matching provenance and quality; and
6. achieves `realtime_factor >= 0.8` on every formal MJX row.

The evaluation order is fail-closed:

```text
evidence validity
-> functional and resource validity
-> No-MPC dominance
-> sweetpoint quality
-> control/smoothness/force guardrails
-> repeat stability
-> replay parity
-> RTX 4090 speed
-> optional n8192 parity
```

Speed never compensates for failed quality, provenance, repeatability, or
replay. Historical context is in [HISTORY.md](../../HISTORY.md); the previous
non-normative contract is [archived here](criteria_legacy_20260710.md).

## Frozen Evaluation Surface

Hard-gate motions:

- `jump`
- `walk`

Every formal row evaluates:

- `800` policy steps;
- `16.0` seconds of motion at `control_dt_sec=0.02`;
- `40` MPC windows when `control_steps=20`;
- the original motion as the metrics reference;
- the same WBC policy checkpoint and reward weights;
- exactly one visible GPU.

Formal input identities are path-independent; manifests resolve local files
and recompute SHA256 values.

| Logical input | Required SHA256 |
| --- | --- |
| WBC MLP checkpoint | `98738b9214d12146dc7f4669cb65dfde9d835f4a133e5f2cbaef4e60b1e5b88f` |
| reward weights | `bb0490a71a27a29480f13ce77bc00c13a900845f0f52641b5ec20d51ebaa535d` |
| jump motion | `07b3b8e1bf9ba3f94dfbe552819cd792f81a06a3c4ff6e5029b55b2897b7c544` |
| walk motion | `a9baaa714d61da19c6114077cf0c919c965ad6802f770cc83ed695396c4c8c9f` |

The formal sweetpoint baseline name is:

```text
g1_wbc_stage0_mujoco_warp_sweetpoint
```

Its frozen search surface is:

| Field | Required value |
| --- | --- |
| method | `g1_wbc_joint_global` |
| backend | `mujoco_warp` |
| optimizer | `legacy` |
| preset | `aggressive` |
| samples | `512` |
| optimizer iterations | `2` |
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
| warm-start source | `best` |
| warm-start decay | `1.0` |
| save rollout | enabled |

The formal candidate uses explicit `--mpc-backend mjx` and
`--mpc-optimizer generic`. Its manifest freezes every search and execution
parameter, including:

- `mjx_impl` and model implementation;
- sample, iteration, horizon, control, and knot counts;
- elite fraction, temperature, sigmas, and sigma decay;
- candidate and warm-start policy;
- score-only or full-metrics optimizer mode;
- solver and line-search options;
- collision profile and force semantics;
- contact and constraint capacities.

One formal matrix may select `jax` or `warp`, but must not mix them. A candidate
change creates a new matrix; rows from different configurations cannot be stitched.

## Reference Levels

### Current No-MPC Reference

The formal No-MPC reference is regenerated from the current code and frozen
inputs:

```text
jump/walk x seeds 0/1/2 = 6 No-MPC rows
```

Each row uses `--method no_mpc`, completes 800 steps, and records the same
metrics and provenance. Historical values are explanatory only. Promotion is
blocked until the current manifest is valid.

### MuJoCo-Warp Sweetpoint Reference

The minimum acceptable optimization quality is the validated RTX 4090 Stage0
manifest:

```text
s512 / i2 / h40 / c20 / k8
jump/walk x seeds 0/1/2 = 6 sweetpoint rows
```

The manifest freezes rows, metric envelopes, promoted seeds, input hashes, and
artifact hashes before candidate evaluation. A baseline rerun selected after
candidate results are known is invalid.

### Historical n8192 Stretch Reference

The historical quality-oriented configuration is:

```text
s8192 / i2 / h80 / c20 / k8
```

Historical metrics may guide quality work but do not block sweetpoint-level
promotion. A formal `n8192_parity_candidate` claim requires a new
current-code, current-input, RTX 4090 manifest. OOM or impractical runtime for
this stretch workload does not invalidate a candidate that passes the main
contract.

## Formal Candidate and Replay Matrix

The candidate matrix is:

```text
jump/walk x seeds 0/1/2 x 2 independent processes = 12 MJX rows
```

Repeats must use separate OS processes; an in-memory repeat does not count.

Every MJX row has one matching MuJoCo-Warp `replay_command` row:

```text
12 MJX rows + 12 replay rows
```

Within one matrix, only these fields may vary:

- motion;
- seed;
- repeat ID;
- output path.

All optimizer, scoring, physics, collision, capacity, and timing settings are
otherwise identical.

## Hardware and Timing Contract

Formal evidence must report runtime GPU name:

```text
NVIDIA GeForce RTX 4090
```

Hardware rules:

- expose exactly one GPU to each command;
- use that same GPU class for No-MPC, Stage0, MJX, and replay evidence;
- do not run concurrent formal commands on the visible GPU;
- start with no unrelated compute process on the GPU;
- record visible device IDs and runtime GPU name in every row;
- treat results from H100 or any other GPU as historical or diagnostic.

Steady-state MJX timing includes:

- observation assembly required by MPC;
- actor calls in the planning path;
- candidate sampling;
- physics rollout;
- scoring;
- optimizer update or selection;
- dynamic execute tracing;
- required device synchronization.

Report separately and exclude from steady state:

- model and runtime initialization;
- first-call JIT compilation;
- warmup;
- rendering and video writing;
- heavy diagnostic serialization.

Replay time is separate and never contributes to MJX RTF.

## Evidence Validity

Formal rows share one clean Git commit. Dirty runs are diagnostic only.

Every manifest and report must record:

- commit SHA and clean-worktree status;
- resolved input paths and recomputed SHA256 values;
- effective argv and normalized configuration;
- motion, seed, repeat ID, backend, optimizer, and implementation;
- runtime GPU and visible devices;
- compile/init, warmup, steady-state, and per-window timing;
- accepted windows and dynamic execute-trace source;
- contact/constraint capacity and saturation status;
- metrics, rollout, and command artifact paths and SHA256 values;
- final gate results and classification.

Every MJX and replay row must:

- return successfully;
- complete exactly 800 steps;
- contain all required finite metrics;
- avoid baseline fallback;
- avoid contact, geometry-pair, and constraint-capacity saturation;
- emit fresh `metrics.json` and `rollout.npz`;
- emit a fresh `mpc_command.npz` for MJX rows;
- preserve the accepted 40-window dynamic execute trace.

These conditions make evidence invalid rather than low quality:

- missing or mismatched hashes;
- incomplete matrices;
- stale or replaced artifacts;
- mixed configurations or implementations;
- static/fabricated rollout traces;
- reuse under a different motion, seed, repeat, or command source;
- OOM, NaN/Inf, fallback, or capacity overflow.

OOM is a functional/resource failure and produces no valid speed measurement.

## Quality Gates

### Primary Outcome Metrics

Higher is better only for `score`. Lower is better for:

- `root_pos_error_mean`;
- `body_global_pos_error_mean`;
- `ee_global_pos_error_mean`;
- `ee_local_pos_error_mean`;
- `contact_mismatch_rate`;
- `contact_false_positive_rate`;
- `contact_false_negative_rate`.

All 12 MJX rows and all 12 replay rows must have:

- `success=true`;
- finite primary and guardrail metrics;
- `bad_floor_contact_rate=0`;
- `bad_floor_force_excess_mean=0`.

Score, the three global position metrics, and contact mismatch participate in
No-MPC improvement capture. Local EE does not: a globally displaced rollout
can retain a deceptively low local-coordinate error. Local EE and contact
false-positive/false-negative rates remain hard sweetpoint-relative gates.

### No-MPC Improvement Capture

For lower-is-better dominance metric `x`:

```text
capture(x) = (NoMPC(x) - candidate(x))
             / (NoMPC(x) - sweetpoint(x))
```

For score:

```text
capture(score) = (candidate(score) - NoMPC(score))
                 / (sweetpoint(score) - NoMPC(score))
```

For every motion and dominance metric:

- each MJX row uses its same-seed No-MPC and sweetpoint rows and must capture
  at least `80%` of the sweetpoint improvement;
- the six-row MJX motion mean uses the No-MPC and sweetpoint motion means and
  must capture at least `100%`;
- the six replay rows independently satisfy the same row and mean rules.

For lower-is-better metrics, a valid denominator requires
`NoMPC > sweetpoint`. For score it requires `sweetpoint > NoMPC`. If the
reference does not improve in the required direction, that reference metric
is invalid; do not reverse or omit the comparison.

### Sweetpoint Mean and Per-Row Gates

For every primary metric and motion:

- the six-row MJX mean must meet or beat the three-seed sweetpoint mean;
- the six-row replay mean must independently meet or beat that mean.

For lower-is-better metric `x`, every MJX and replay row must satisfy:

```text
candidate(motion, seed, repeat, x)
    <= 1.05 * sweetpoint(motion, seed, x)
```

For score, every row must satisfy:

```text
candidate_score
    >= sweetpoint_score - 0.05 * abs(sweetpoint_score)
```

An excellent row cannot hide a failing row.

### Absolute Safety Caps

Every row must also remain below these fail-closed caps:

| Motion | Root | Body global | EE global | EE local | Contact mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `jump` | `0.075` | `0.085` | `0.095` | `0.050` | `0.375` |
| `walk` | `0.082` | `0.088` | `0.090` | `0.042` | `0.175` |

Passing an absolute cap does not excuse missing a tighter relative gate.

### Control, Smoothness, and Force Guardrails

These metrics need not beat No-MPC. Every MJX and replay row must remain
within its same-seed sweetpoint multiplier:

| Metric | Maximum multiplier |
| --- | ---: |
| `control_delta_mean` | `1.15x` |
| `joint_acc_mean` | `1.15x` |
| `joint_jerk_mean` | `1.20x` |
| `contact_force_active_mean` | `1.50x` |
| `contact_force_excess_mean` | `1.50x` |
| `contact_force_peak` | `2.00x` |

Force comparisons are valid only when candidate, replay, and Stage0 declare
the same canonical force semantics. All force fields must be finite, and no
contact or constraint buffer may saturate.

The wider peak multiplier guards explosions, not exact single-frame parity.

## Independent-Process Repeat Gate

For each motion, seed, and nonzero primary metric, compare the two MJX process
repeats:

```text
relative_pair_gap = abs(r0 - r1)
                    / max(abs((r0 + r1) / 2), 1e-12)
```

Every `relative_pair_gap` must be at most `5%`. Zero-valued safety metrics use
their explicit zero gates. Passing row limits or group means cannot override a
repeat failure.

## MuJoCo-Warp Replay Gate

Every candidate command is replayed with `--method replay_command` through the
shared MuJoCo-Warp execution backend.

Each replay row must:

- use the corresponding MJX row's current command artifact;
- match motion, seed, repeat ID, command source, and command SHA256;
- consume the saved per-window command and replay-state chunks;
- use the original motion as metrics reference;
- complete 800 steps without fallback, non-finite output, or saturation;
- produce fresh metrics and rollout artifacts;
- pass the same No-MPC, sweetpoint, absolute, and guardrail gates as MJX.

For every primary metric in each paired MJX/replay row:

```text
paired_gap = 2 * abs(MJX - replay)
             / max(abs(MJX) + abs(replay), 1e-12)
```

Every `paired_gap` must be at most `5%`. A replay failure is a backend parity
failure even when MJX quality and speed pass.

## Visual Review

Inspect each motion's worst-score and worst-contact row before promotion. One
row may satisfy both selections.

Promotion stops on any obvious:

- fall;
- body or floor penetration;
- explosive jitter or force event;
- materially incorrect motion hidden by aggregate metrics.

Visual review cannot rescue a numeric failure.

## RTX 4090 Speed Gate

Use the repository's existing speed definition:

```text
realtime_factor = evaluated_motion_duration_sec
                  / steady_state_wall_time_sec
```

Every formal MJX row must satisfy:

```text
realtime_factor >= 0.8
```

The aggregate hard gate is therefore:

```text
min(realtime_factor) >= 0.8
```

The minimum is taken across all motions, seeds, and process repeats. No mean,
median, percentile, or best row may hide a slow row. For a 16-second motion,
every row must have `steady_state_wall_time_sec <= 20.0`.

Compile/init and warmup remain separate report fields. Same-seed speedup over
Stage0 is report-only and is not a hard gate.

The speed decision is evaluated only after evidence, functionality, quality,
guardrail, repeat, replay, and visual gates pass.

## Classifications and Stop Rules

Report the first applicable failure layer:

| Classification | Meaning |
| --- | --- |
| `invalid_evidence` | provenance, hardware, matrix, or artifact contract failed |
| `functional_or_resource_failure` | crash, OOM, fallback, non-finite output, or saturation |
| `no_mpc_dominance_failure` | required No-MPC improvement capture failed |
| `sweetpoint_quality_failure` | sweetpoint mean, per-row, or absolute quality gate failed |
| `guardrail_failure` | control, smoothness, force, or visual guardrail failed |
| `repeat_stability_failure` | independent-process repeat gap failed |
| `replay_parity_failure` | replay provenance, quality, or paired gap failed |
| `speed_failure` | a valid MJX row had RTF below 0.8 |
| `promotion_candidate` | every mandatory layer passed |
| `n8192_parity_candidate` | promotion passed and a valid current n8192 parity gate passed |

Repair the lowest failing layer first:

- invalid or resource-failing rows cannot support quality or speed claims;
- do not tune speed while No-MPC or sweetpoint quality fails;
- do not claim backend replacement while repeat or replay fails;
- optimize runtime only after quality, guardrails, repeat, and replay pass;
- do not let n8192 resource failure block sweetpoint-level promotion.

## Current Runner Enforcement Gap

This file remains normative when the runner implements an older contract.

Until runner tests enforce all of the following, its output alone cannot claim
`promotion_candidate`:

- the current six-row No-MPC manifest;
- two independent processes for every motion and seed;
- 12 paired replay rows;
- 80% per-row and 100% mean improvement capture;
- sweetpoint mean and same-seed 5% row limits;
- repeat and replay paired-gap formulas;
- `min(realtime_factor)` across all 12 MJX rows;
- the classifications and stop order defined above.

Runner alignment must strengthen code to match this contract, not weaken it to match an older runner.
