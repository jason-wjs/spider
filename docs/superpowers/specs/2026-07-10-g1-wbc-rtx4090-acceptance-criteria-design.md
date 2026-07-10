# G1 WBC RTX 4090 Acceptance Criteria Redesign

**Date:** 2026-07-10
**Status:** Approved design
**Scope:** Acceptance-contract documentation and its document-level tests

## Objective

Replace the current H100-era G1 WBC MJX acceptance criteria with a concise,
quality-first contract for the single-RTX-4090 workstation.

The contract answers one promotion question:

> Does MPC plus policy-in-the-loop substantially improve over the current
> No-MPC rollout, recover at least MuJoCo-Warp sweetpoint quality, reproduce
> across processes and MuJoCo-Warp replay, and complete every formal row with
> `realtime_factor >= 0.8`?

Speed never compensates for failed quality, replay, provenance, or stability.
Historical `n8192` quality remains a stretch reference until it is reproduced
on the current code, inputs, and RTX 4090.

## Selected Documentation Structure

The approved approach is a complete archive-and-rewrite:

1. Preserve the current criteria body unchanged as
   `criteria_legacy_20260710.md` and prepend a non-normative archive banner.
2. Replace `criteria.md` with a durable contract targeting 350--500 lines and
   never exceeding 1,000 lines.
3. Add only durable decisions to `docs/HISTORY.md`.
4. Leave raw run paths, probe tables, and chronological diagnosis in existing
   handoff and diagnostic documents.

The new criteria stays in one file. It must not be split into several small
normative documents because that would make the acceptance order and final
classification harder to audit.

## Normative Decision Order

The criteria evaluates layers in this order and reports the first failing
layer:

1. evidence validity
2. functional and resource validity
3. No-MPC dominance
4. sweetpoint-relative quality
5. control, smoothness, and force guardrails
6. independent-process repeat stability
7. MuJoCo-Warp replay parity
8. RTX 4090 real-time factor
9. optional current-manifest `n8192` parity

Later layers cannot override earlier failures.

## Frozen Evaluation Surface

Hard-gate motions are `jump` and `walk`. Each row uses:

- the same original motion asset and motion-reference convention;
- the same WBC policy checkpoint;
- the same reward-weight file;
- 800 evaluated policy steps;
- 40 MPC windows for the current 20-step execution interval;
- one visible RTX 4090;
- explicit backend, optimizer, MJX implementation, solver, collision, and
  contact-capacity settings.

The normative file records stable logical names and expected SHA256 values,
not machine-specific absolute paths. Formal manifests resolve paths and record
the hashes again. All rows in one candidate matrix share the same resolved
inputs and code snapshot.

## Reference Levels

### Current No-MPC reference

Build a current-code/current-input No-MPC manifest containing:

```text
jump/walk x seeds 0/1/2 = 6 rows
```

Historical No-MPC values explain the project goal but are not formal evidence.
The current manifest is required before a new promotion claim can pass.

### MuJoCo-Warp sweetpoint reference

The formal minimum quality level is the validated RTX 4090 Stage0
MuJoCo-Warp sweetpoint manifest:

```text
s512 / i2 / h40 / c20 / k8
jump/walk x seeds 0/1/2 = 6 rows
```

The frozen manifest rows, envelopes, input hashes, and artifact hashes are the
reference. A baseline rerun selected after seeing candidate results is invalid.

### Historical `n8192` reference

The historical shape is:

```text
s8192 / i2 / h80 / c20 / k8
```

Its packaged metrics may be shown as stretch guidance. They cannot block
sweetpoint-level acceptance and cannot support a formal parity claim until a
current-code/current-input RTX 4090 manifest exists. OOM or impractical runtime
on this shape is recorded as a stretch-target resource result, not as failure
of an otherwise valid sweetpoint-level candidate.

## Formal Candidate and Replay Matrix

The MJX candidate matrix is:

```text
jump/walk x seeds 0/1/2 x 2 independent processes = 12 MJX rows
```

Every MJX row has one corresponding MuJoCo-Warp `replay_command` row, producing
12 replay rows. Within one formal candidate, only motion, seed, repeat ID, and
output path may differ. All search, physics, scoring, solver, collision, and
capacity settings must remain identical.

The two repeats must be separate OS processes. Reusing one in-memory compiled
runtime does not count as independent-process evidence.

## Quality Metrics

### Primary outcome metrics

The primary set is:

- `score` (higher is better);
- `root_pos_error_mean`;
- `body_global_pos_error_mean`;
- `ee_global_pos_error_mean`;
- `contact_mismatch_rate`;
- `contact_false_positive_rate`;
- `contact_false_negative_rate`;
- `ee_local_pos_error_mean`.

No-MPC dominance is applied to score, the three global position metrics, and
contact mismatch. It is not applied to local EE error: a globally displaced
rollout can retain a deceptively small local-coordinate error. Local EE remains
a hard sweetpoint-relative consistency metric.

All 12 MJX rows and all 12 replay rows must report `success=true`, finite
required metrics, and `bad_floor_contact_rate=0`.

## No-MPC Improvement Capture

For lower-is-better metric `x`, define:

```text
capture(x) = (NoMPC(x) - candidate(x))
             / (NoMPC(x) - sweetpoint(x))
```

For score, define:

```text
capture(score) = (candidate(score) - NoMPC(score))
                 / (sweetpoint(score) - NoMPC(score))
```

The denominator must be positive in the improvement direction. If a frozen
reference does not satisfy that condition, the manifest is invalid for this
metric and must be repaired rather than silently reversing the comparison.

For each motion and each No-MPC-dominance metric:

- every individual candidate row must capture at least 80% of the same seed's
  sweetpoint improvement over the same-seed No-MPC row;
- the six-row candidate mean must capture at least 100%, computed from the
  candidate, No-MPC, and sweetpoint motion-level means;
- the same rules apply independently to the six replay rows.

This quantifies "substantially better than No-MPC" while preventing a strong
row from hiding a weak one.

## Sweetpoint-Relative Gates

For each motion, the mean over six candidate rows must meet or beat the mean
over the three frozen sweetpoint seeds on every primary outcome metric. The
same group rule applies independently to replay rows.

For lower-is-better metric `x`, every row must satisfy:

```text
candidate(motion, seed, repeat, x)
    <= 1.05 * sweetpoint(motion, seed, x)
```

For score, every row must satisfy:

```text
candidate_score
    >= sweetpoint_score - 0.05 * abs(sweetpoint_score)
```

Replay rows use the same formulas. Existing absolute caps remain as
fail-closed safety ceilings:

| Motion | Root | Body global | EE global | EE local | Contact mismatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| `jump` | 0.075 | 0.085 | 0.095 | 0.050 | 0.375 |
| `walk` | 0.082 | 0.088 | 0.090 | 0.042 | 0.175 |

Meeting an absolute cap does not excuse missing the tighter relative gate.

## Control, Smoothness, and Force Guardrails

These metrics need not beat No-MPC, but each candidate and replay row must stay
within its same-seed sweetpoint guardrail:

| Metric | Maximum relative to same-seed sweetpoint |
| --- | ---: |
| `control_delta_mean` | 1.15x |
| `joint_acc_mean` | 1.15x |
| `joint_jerk_mean` | 1.20x |
| `contact_force_active_mean` | 1.50x |
| `contact_force_excess_mean` | 1.50x |
| `contact_force_peak` | 2.00x |

Additionally:

- `bad_floor_force_excess_mean=0`;
- all force metrics are finite;
- force semantics match the frozen manifest;
- contact and geometry capacities are not saturated.

The wider peak-force limit catches explosive behavior without treating a
single-frame peak as precise backend parity.

## Repeat Stability

For each motion, seed, and nonzero primary metric, compare its two independent
process repeats using:

```text
relative_pair_gap = abs(r0 - r1)
                    / max(abs((r0 + r1) / 2), 1e-12)
```

`relative_pair_gap` must not exceed 5%. Zero-valued safety metrics use their
explicit zero gates instead. Passing group means does not override a repeat
gap failure.

## Replay Parity

Each replay row must:

- consume the corresponding MJX row's saved command and replay-state chunks;
- match motion, seed, repeat ID, command source, and command SHA256;
- complete 800 steps without fallback, NaN/Inf, or saturation;
- produce fresh metrics and rollout artifacts;
- pass the same No-MPC, sweetpoint, absolute, and guardrail quality gates as
  the MJX matrix.

For every paired MJX/replay row and every primary outcome metric, use the
symmetric difference:

```text
paired_gap = 2 * abs(MJX - replay)
             / max(abs(MJX) + abs(replay), 1e-12)
```

`paired_gap` must not exceed 5%. Replay runtime is reported separately and is
never included in MJX real-time factor.

## RTX 4090 Timing Contract

Use the repository's existing definition:

```text
realtime_factor = evaluated_motion_duration_sec
                  / steady_state_wall_time_sec
```

Every one of the 12 MJX rows must satisfy:

```text
realtime_factor >= 0.8
```

Therefore the reported aggregate is the minimum over all motions, seeds, and
repeats. For the current 16-second motions, every row must finish steady-state
optimization in at most 20 seconds.

Steady-state timing includes observation and actor work required by MPC,
sampling, rollout, scoring, optimizer update or selection, execute tracing,
and required device synchronization. Initialization, JIT compilation, warmup,
rendering, and heavy debug serialization are reported separately.

Same-seed speedup over Stage0 is report-only. H100 timing is historical only.
OOM, fallback, non-finite output, or capacity saturation produces no valid RTF.

## Hardware and Evidence Validity

Formal evidence requires:

- runtime GPU name `NVIDIA GeForce RTX 4090`;
- exactly one visible GPU;
- no concurrent formal run or pre-existing compute process on that GPU;
- one clean Git commit shared by all formal rows;
- exact input and artifact SHA256 values;
- consistent config provenance across the matrix;
- fresh `metrics.json`, `rollout.npz`, and `mpc_command.npz` artifacts;
- dynamic execute-trace and accepted-window evidence;
- no mixed backend, implementation, solver, collision, or capacity settings.

Dirty-worktree runs, partial matrices, stitched rows, stale artifacts, or rows
reused from another config are diagnostic evidence only.

## Visual Review

Before final promotion, review at least the worst-score and worst-contact row
for each motion. An obvious fall, penetration, explosive jitter, or materially
incorrect motion invalidates promotion even if aggregate metrics pass.

## Classifications and Stop Rules

Report the first applicable failure classification:

1. `invalid_evidence`
2. `functional_or_resource_failure`
3. `no_mpc_dominance_failure`
4. `sweetpoint_quality_failure`
5. `guardrail_failure`
6. `repeat_stability_failure`
7. `replay_parity_failure`
8. `speed_failure`
9. `promotion_candidate`
10. `n8192_parity_candidate`, only with a valid current n8192 manifest

The team repairs the lowest failing layer first. It does not tune speed while
quality fails, and it does not claim backend replacement while repeat or replay
parity fails. Only a quality-, guardrail-, repeat-, and replay-valid matrix is
eligible for the RTF promotion decision.

## Current Evidence Interpretation

The current canonical full-metrics MJX-Warp report is speed-qualified under the
new `RTF >= 0.8` threshold, but it is not promotion-qualified. It lacks the
second independent-process repeat matrix and misses the proposed quality bar
on local EE and parts of walk score/contact behavior.

The historical n8192 workload demonstrates why backend speedup does not erase
workload scaling: its much larger sample and horizon product can still OOM or
run impractically slowly. This rationale belongs in history, not as a current
hard gate.

## Implementation Scope

This documentation change will:

1. archive and rewrite the criteria files;
2. add a concise durable history entry where it is not already present;
3. update the criteria document contract test, including the `<1000` line
   invariant;
4. run focused documentation tests and whitespace/link checks.

It will not modify the acceptance runner, MJX runtime, optimizer, or physics
implementation in the same change. The new criteria is normative; any runner
behavior that does not yet enforce the 12-row repeat/replay matrix, No-MPC
capture, per-row RTF minimum, or new classifications is an explicit follow-up
implementation gap. Until that gap is closed, runner output cannot by itself
claim `promotion_candidate` under this contract.

## Verification

The documentation implementation is complete when:

- the active criteria is below 1,000 lines and contains no H100 acceptance
  gate or `>=12x` hard gate;
- it defines the exact matrices, formulas, thresholds, classifications, and
  stop order approved above;
- the legacy criteria is clearly marked non-normative;
- HISTORY contains only concise durable context, without raw diagnostic logs;
- the focused criteria-document tests pass;
- `git diff --check` passes;
- existing unrelated worktree changes remain untouched.
