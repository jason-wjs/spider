# SPIDER Development History

> This file is a concise record of past engineering decisions. It is not an
> acceptance contract, benchmark specification, or current task handoff.

Current G1 WBC MJX requirements are in the
[acceptance criteria](tasks/g1_wbc_mjx_full_rollout/criteria.md). That task
directory intentionally contains only the current acceptance contract.

Raw commands, local paths, candidate matrices, and implementation checklists are
omitted. Former tracked handoffs and diagnostics remain available in Git
history; unique findings from untracked July notes were summarized here before
their removal.

## 2026-06-18: G1 WBC Benchmark Foundation

**Question:** How should policy-only and MPC retargeting runs be compared repeatably?

**Findings:** The project needed one benchmark runner that fixed the input
motion, checkpoint, MPC configuration, output schemas, and primary metrics.
Ad-hoc runs could not reliably distinguish configuration changes from input or
artifact changes.

**Decision:** Introduce the first `bench_data` benchmark workflow and structured
artifacts. Subsequent experiments would compare tracking, contact, control, and
smoothness together rather than promote a candidate from aggregate score alone.

## 2026-06-22: Low-Sample MPC

**Question:** Could the historical `n8192` quality anchor be approached with a
much smaller rollout budget?

**Findings:** The `s128/i2/h40/c20/k8` candidate preserved much of the global
tracking improvement over the No-MPC rollout and was substantially faster than
the packaged `n8192` result. The comparison was not a strict same-hardware speed
benchmark, and the candidate was still far from real time. It also regressed
control smoothness, showed visible jitter, and remained motion-dependent on
local tracking and contact.

**Decision:** Keep `s128` as a useful search anchor, not as a replacement for
the high-sample baseline. Future promotion gates had to include contact,
smoothness, and visual review as well as global tracking.

## 2026-06-22: Local-Frontier Experiments

**Question:** Could stronger local and posture rewards improve motion fidelity
without increasing the `s128` rollout budget?

**Findings:** Several reward variants improved local metrics in the initial
screen. Transfer and saved-rollout review were less convincing: the strongest
three-motion candidate remained viable on `walk` and `qixing`, but `jump` was
unstable and visibly worse on repeated runs. Improvements in selected local
metrics did not overcome regressions in smoothness, global tracking, or
repeatability.

**Decision:** Do not promote the local-frontier variants. Return to the original
low-sample reward anchor and strengthen baseline-relative smoothness and repeat
gates. Treat the variants as diagnostic ablations only.

## 2026-06-23: Fast-Quality Pareto Search

**Question:** Was the remaining quality gap caused mainly by too few samples,
by reward design, or by run-to-run instability?

**Findings:** A `jump` ladder from `s64` through `s256` improved the continuous
quality signal at the upper end, but every budget failed the repeat-stability
classification. `s256` came close to the historical baseline on average quality
while succeeding on only one of three seeds. The failed rows were narrow contact
gate misses rather than broad tracking failures.

**Decision:** Classify stability and contact handling as the immediate
bottleneck. Do not transfer or visually promote a budget based on its best or
mean row when the repeat set is incomplete or inconsistent.

## 2026-06-24: Quality-Speed Mechanism Study

**Question:** Which configuration dimensions actually control quality per
second, and which merely add rollout work?

**Findings:** Increasing samples was useful but not monotonic. On the tested
legacy backend, `s512/i2/h40/c20/k8` became the practical `jump` sweetpoint:
better than `s128` on average while avoiding the cost and variance of larger
budgets. A third optimizer iteration acted as a quality-ceiling lever. Small
command regularization was the strongest low-budget mechanism found. Changing
horizon/control or knot count away from the anchor frequently damaged tracking
or contact.

Repeated saved rollouts also showed that the same explicit configuration could
produce materially different outcomes. A single attractive trajectory was not
sufficient evidence for promotion.

**Decision:** Use the `s512/i2/h40/c20/k8` family as the historical sweetpoint,
keep `n8192` as a quality reference rather than a usable runtime target, and
require multi-seed or independent-process evidence for future comparisons.

## 2026-06-26: MPC Motion Export Contract

**Question:** How should optimized motion and executed commands be exported so
that evaluation and replay can verify what actually ran?

**Findings:** A rollout alone did not capture enough provenance to distinguish
reference motion, planned commands, executed commands, and replay inputs.

**Decision:** Define separate rollout and command export schemas with explicit
frame, timing, state, and reference-index information. Later acceptance work
extended this principle with command hashes and replay provenance.

## 2026-07-01 to 2026-07-03: JAX/MJX Full-Rollout Migration

**Question:** Could JAX orchestration and MJX-compatible rollout remove the
legacy backend's optimizer bottleneck without changing policy or scoring
semantics?

**Findings:** Backend replacement touched more than physics stepping. Model
construction, actor inference, observation assembly, contact interpretation,
streaming score accumulation, optimizer updates, and saved replay all required
independent parity checks. A fast kernel was not sufficient if its inputs or
metrics differed from the baseline contract.

**Decision:** Migrate in parity-gated phases: runtime and model probes, actor and
observation parity, no-MPC rollout parity, scoring parity, one-window optimizer,
full runner integration, and finally replay and speed gates. Keep canonical
model roles explicit instead of silently substituting a reduced physics model.

The acceptance tooling was hardened around a fixed policy checkpoint, artifact
hashes, input and output schemas, single-GPU evidence, steady-state timing, saved
command provenance, and MuJoCo-Warp replay. These checks made stale or
incompatible benchmark artifacts invalid rather than merely low quality.

## 2026-07-03: Stage0 Reproduction and Optimizer Routing

**Question:** Why did current-code Stage0 fail to reproduce the historical
sweetpoint even though it used the packaged testbed motions?

**Findings:** The versioned `jump` and `walk` assets and their hashes were the
official migrated testbed inputs. The failure came from routing the
MuJoCo-Warp run through the generic optimizer while historical sweetpoint
artifacts used legacy guided-candidate, acceptance, regularization, and
per-window semantics. Explicit legacy routing recovered the expected short-run
behavior; a complete manifest was still required for formal use.

**Decision:** Do not reject the official motion assets by hash. Stage0 must
select the legacy optimizer explicitly, resolve the checkpoint to a concrete
file, freeze effective configuration and artifact provenance, and run only on
an idle visible GPU.

## 2026-07-04: Move to RTX 4090 Evaluation

**Question:** Which earlier conclusions survived the move from the H100 test
environment to the current single-RTX-4090 machine?

**Findings:** Historical H100 timings and hardware-specific thresholds were not
portable evidence for the new machine. The RTX 4090 required fresh compile,
warmup, steady-state, memory, quality, and replay measurements. The repository's
existing `realtime_factor` remained the appropriate runtime measure.

**Decision:** Treat H100 results as historical provenance only. Run formal work
on one visible RTX 4090, report compile/warmup separately, and judge steady-state
optimization with the existing realtime-factor field.

## 2026-07-06 to 2026-07-08: Quality and Repeatability Recovery

**Question:** Why did the Warp-backed MJX path become fast at the sweetpoint
without consistently preserving the desired optimization quality?

**Findings:** Score-only and contact/collision experiments could improve runtime
or individual rows while degrading `jump` quality, replay, or cross-process
ranking. Contact-force diagnostics also exposed a semantic mismatch risk: MJX
and the legacy MuJoCo-Warp path were not proven to aggregate solver rows in the
same way. Experimental physics and scoring changes were therefore not safe to
promote from isolated wins.

The sweetpoint-sized workload demonstrated that the new execution path could be
fast on the RTX 4090. Larger `n8192`-style workloads remained constrained by
memory and total rollout work, showing that a backend speedup does not make
sample count, horizon, or optimizer depth free. Quality and repeat stability,
not only raw throughput, became the active bottleneck.

Cross-process diagnostics narrowed the instability further. Score-only
optimization removed the observed within-run full-metric rescore rank change,
but checksum-consistent first-iteration samples could still receive different
scores and ranks across processes. The earliest trustworthy component split
was `contact_force_peak` on terrain-foot solver rows, implicating contact-row
grouping and force aggregation semantics. Full-metric shadow scoring and
top-row traces were useful diagnostic oracles but too expensive to count as
formal speed candidates.

**Decision:** Restore canonical behavior after rejected experiments. Preserve
force-row work as a diagnostic, require independent-process repeats, and keep
the optimization objective quality-first: outperform No-MPC, recover at least
the historical sweetpoint quality, and use `n8192` only as a stretch reference
until comparable current-machine evidence exists.

## 2026-07-10: RTX 4090 Quality-First Contract and Documentation Cleanup

**Question:** How should promotion be judged after the hardware move and the
quality/repeatability findings, and which task documents should remain active?

**Findings:** H100-specific speedup gates and chronological probe logs no
longer described the current workstation or decision. The repository already
defined `realtime_factor`, while current evidence showed that optimizer
quality, independent-process stability, and replay parity were the limiting
layers.

**Decision:** Make the single RTX 4090 the formal platform. Require a current
No-MPC reference, MuJoCo-Warp sweetpoint-level quality, twelve MJX rows, twelve
paired replay rows, and `realtime_factor >= 0.8` on every MJX row. Keep
historical `n8192` as a stretch reference until a comparable current manifest
exists. Consolidate durable findings into this history, keep thresholds in the
acceptance criteria, and remove timestamped task handoffs, diagnostics, and the
superseded criteria archive from the active documentation tree.

## Durable Lessons

- Backend speed and end-to-end optimizer speed are different measurements.
- Work scales with the experiment shape; a fast sweetpoint does not imply that
  an `n8192` configuration fits memory or finishes quickly.
- Aggregate score alone is not a promotion criterion. Tracking, contact,
  smoothness, replay, and visual behavior must agree.
- Sample count is an important quality lever, but it is neither monotonic nor
  independent of horizon, control interval, knot count, and iteration count.
- Single-run wins are exploratory evidence. Promotion requires repeated,
  provenance-valid comparisons on the frozen evaluation surface.
- Historical hardware measurements explain past decisions; they do not define
  current RTX 4090 acceptance thresholds.
- Active thresholds and classification rules belong in the current acceptance
  criteria, not in this history file.
