# G1 WBC MJX RTX 4090 Repeat Rank Diagnostic Update

Date: 2026-07-07

Workspace:
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/PITL_worktrees/spider`

Run root:
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs`

## Goal Slice

This update continues the `>=11x` RTX 4090 quality-recovery sub-goal from
`handoff_4090_repeat_stability_20260707.md`. It does not claim acceptance.
The focus was same-argv score-only `jump/seed1/js0.18` repeat instability,
because current evidence showed score-only optimizer improves force behavior
and speed but still produces different commands across runs.

Binding constraints preserved:

- no actuator, `sim_dt`, policy cadence, policy weight, or training-bound change
- no asset restructuring
- `wxy_parity` remains the quality anchor
- `wxy_explicit_floor_leg_foot147_pairs_7caps` remains a reversible diagnostic
  collision profile, not a promoted replacement
- no result below repeat, quality, replay, provenance, single-GPU, contact/force,
  or speed gates is promoted

## Runs

### Rank diagnostic A/B

Runs:

- `foot147_scoreonly_jump_seed1_js018_rankdiag32_repeatA_20260707_4090_gpu0`
- `foot147_scoreonly_jump_seed1_js018_rankdiag32_repeatB_20260707_4090_gpu0`

Reports:

- `foot147_scoreonly_jump_seed1_js018_rankdiag32_repeat_probe_20260707_4090_gpu0.json`
- `foot147_scoreonly_jump_seed1_js018_rankdiag32_rank_divergence_20260707_4090_gpu0.json`

Result:

- repeat stability failed: `classification=repeat_instability`
- rank diagnostics failed: `classification=rank_divergence`
- command content hashes differed
- first command qpos delta exceeded `1e-3` at frame `101`; max qpos delta
  was about `1.204`
- metric ranges exceeded score/root/body/global-EE/local-EE/peak-force gates

Key root-cause evidence:

- sample signature/probe first diverged at `window=0, iteration=1`
- candidate top order/set first diverged earlier at `window=0, iteration=0`
- same-index score delta was already large at `window=0, iteration=0`:
  candidate `352` had score delta `+0.99027`
- within-run candidate rescore stayed stable:
  `candidate_rescore_score_delta_max=0.0`,
  `candidate_rescore_top1_changed_iteration_sum=0`

Interpretation:

The first visible divergence is cross-run score/rank drift on apparently
matching first-iteration samples, not within-run scorer non-idempotence. The
later sample divergence is likely an amplification through CEM elite/update
state after the rank drift.

### Score-component diagnostic C/D

Runs:

- `foot147_scoreonly_jump_seed1_js018_rankscorediag32_repeatC_20260707_4090_gpu0`
- `foot147_scoreonly_jump_seed1_js018_rankscorediag32_repeatD_20260707_4090_gpu0`

Reports:

- `foot147_scoreonly_jump_seed1_js018_rankscorediag32_repeat_probe_20260707_4090_gpu0.json`
- `foot147_scoreonly_jump_seed1_js018_rankscorediag32_rank_divergence_20260707_4090_gpu0.json`

Result:

- repeat stability still failed: `classification=repeat_instability`
- rank diagnostics still failed: `classification=rank_divergence`
- first same-index score delta at `window=0, iteration=0` was candidate `384`,
  delta `-0.66299`
- same-index score delta max later reached about `10.09`
- score-component diagnostics were not useful under `--mjx-score-only-optimizer`:
  `candidate_score_component_diagnostics_field_count=0`

Interpretation:

The score-only optimizer path emits only compact scores, so component-level
diagnostics cannot currently explain which tracking/contact/control component
drives the cross-run score drift. This is a diagnostic limitation, not a
quality pass or failure by itself.

### JAX backend comparison E

Run:

- `jax_scoreonly_jump_seed1_js018_rankdiag32_repeatE_20260707_4090_gpu0`

Result:

- MJX row status was `ok`, but the runner returned nonzero because the row is
  not acceptance-worthy
- `mjx_impl=jax`, `mjx_model_impl=jax`
- steady-state time was about `28.366s`, below the `>=11x` speed envelope
- quality regressed for `jump/seed1`:
  `score=-2.05919`, `root=0.07461`, `body=0.08657`,
  `ee_global=0.09493`, `ee_local=0.05058`

Interpretation:

The JAX backend comparison is rejected as a practical `>=11x` quality-recovery
path on this configuration. A second repeat was not run because this direction
already missed speed and did not improve quality.

### CEM update gap G/H

Runs:

- `foot147_scoreonly_jump_seed1_js018_cemgap05_rankdiag32_repeatG_20260707_4090_gpu0`
- `foot147_scoreonly_jump_seed1_js018_cemgap05_rankdiag32_repeatH_20260707_4090_gpu0`

Reports:

- `foot147_scoreonly_jump_seed1_js018_cemgap05_rankdiag32_repeat_probe_20260707_4090_gpu0.json`
- `foot147_scoreonly_jump_seed1_js018_cemgap05_rankdiag32_rank_divergence_20260707_4090_gpu0.json`

Single-variable change:

- added `--mjx-cem-update-min-top-score-gap 0.5`

Result:

- speed stayed above the local `>=11x` line for the two single-seed rows:
  about `18.25s` and `18.20s`
- repeat stability got worse, not better:
  score range `0.2845`, root/body/global-EE ranges about `0.083`
- one repeat quality collapsed:
  `root=0.15040`, `body=0.15992`, `ee_global=0.16658`
- rank divergence still began at `window=0, iteration=0`
- first same-index score delta was `+1.09513`; later max was about `37.05`

Interpretation:

`--mjx-cem-update-min-top-score-gap 0.5` is rejected. It preserves speed but
does not stabilize the cross-run score/rank drift and can severely regress
quality.

### Sample-checksum diagnostic I/J

Runs:

- `foot147_scoreonly_jump_seed1_js018_checksumdiag32_repeatI_20260707_4090_gpu0`
- `foot147_scoreonly_jump_seed1_js018_checksumdiag32_repeatJ_20260707_4090_gpu0`

Reports:

- `foot147_scoreonly_jump_seed1_js018_checksumdiag32_repeat_probe_20260707_4090_gpu0.json`
- `foot147_scoreonly_jump_seed1_js018_checksumdiag32_rank_divergence_20260707_4090_gpu0.json`

Code/tooling delta before this run:

- `spider/tasks/g1_wbc/mjx_optimizer.py` now emits
  `iteration_sample_checksums` under the existing
  `candidate_rank_diagnostics_top_k > 0` diagnostic gate.
- `scripts/diagnose_g1_wbc_mjx_candidate_rank_divergence.py` now compares those
  checksums and emits `candidate_sample_checksum_divergence`.

Result:

- repeat stability still failed: `classification=repeat_instability`
- rank diagnostics still failed: `classification=rank_divergence`
- both rows used one visible RTX 4090:
  `runtime_gpu_name=NVIDIA GeForce RTX 4090`, `runtime_visible_devices=["0"]`
- speed remained above the local `>=11x` line:
  repeatI `11.404x`, repeatJ `11.365x`
- repeatI quality:
  `score=-1.99822`, `root=0.08174`, `body=0.09067`,
  `ee_global=0.09813`, `ee_local=0.04772`,
  `contact_mismatch_rate=0.265625`,
  `contact_force_active_mean=244.355`,
  `contact_force_delta_mean=0.32192`, `contact_force_peak=2053.90`
- repeatJ quality:
  `score=-1.88917`, `root=0.05691`, `body=0.06523`,
  `ee_global=0.07258`, `ee_local=0.04963`,
  `contact_mismatch_rate=0.265625`,
  `contact_force_active_mean=242.879`,
  `contact_force_delta_mean=0.30864`, `contact_force_peak=1965.66`
- within-run rescore was still stable:
  `candidate_rescore_score_delta_max=0.0`

New root-cause evidence:

- sample checksum diverged immediately at `window=0, iteration=0`
  with max checksum delta `0.0024719`
- sample signature also diverged at `window=0, iteration=0`
  with sample-sum delta about `0.00351`
- sample probe first diverged at `window=0, iteration=1`
- candidate top order/set diverged at `window=0, iteration=0`
- same-index score delta also diverged at `window=0, iteration=0`:
  candidate `352` delta `-0.99332`
- later same-index score delta max reached `25.8713`
- final repeat instability remained material:
  score delta `+0.10905`, root delta `-0.02482`,
  body delta `-0.02544`, global-EE delta `-0.02554`,
  first command qpos delta exceeded `1e-5` at frame `1`,
  `1e-3` at frame `121`, and max qpos delta was about `1.3974`

Interpretation:

The earlier A/B interpretation was too weak: the first visible divergence is
not only score/rank drift on apparently matching samples. The stronger
checksum evidence shows the sampled candidate tensor already differs across
independent same-argv processes at `window=0, iteration=0`. The next root cause
slice should therefore audit the MJX optimizer sampling/PRNG path and any
process-level nondeterminism that can perturb the first candidate batch. Warp
rollout/scoring drift may still exist, but it is no longer the earliest proven
divergence on this diagnostic surface.

### No-guided checksum diagnostic K/L

Runs:

- `foot147_scoreonly_jump_seed1_js018_noguided_checksumdiag32_repeatK_20260707_4090_gpu0`
- `foot147_scoreonly_jump_seed1_js018_noguided_checksumdiag32_repeatL_20260707_4090_gpu0`

Reports:

- `foot147_scoreonly_jump_seed1_js018_noguided_checksumdiag32_repeat_probe_20260707_4090_gpu0.json`
- `foot147_scoreonly_jump_seed1_js018_noguided_checksumdiag32_rank_divergence_20260707_4090_gpu0.json`

Single-variable diagnostic change:

- omitted `--mjx-guided-candidate` and
  `--mjx-guided-candidate-period 4`

Result:

- repeat stability still failed: `classification=repeat_instability`
- rank diagnostics still failed: `classification=rank_divergence`
- `guided_candidate_windows=0` in both rows
- speed stayed above the local `>=11x` line and improved slightly:
  repeatK `11.918x`, repeatL `11.891x`
- repeatK quality:
  `score=-1.98293`, `root=0.07063`, `body=0.07897`,
  `ee_global=0.08405`, `ee_local=0.04717`,
  `contact_mismatch_rate=0.27000`,
  `contact_force_active_mean=251.531`,
  `contact_force_delta_mean=0.32806`, `contact_force_peak=1886.22`
- repeatL quality:
  `score=-2.03486`, `root=0.05832`, `body=0.06904`,
  `ee_global=0.07779`, `ee_local=0.04822`,
  `contact_mismatch_rate=0.29375`,
  `contact_force_active_mean=249.689`,
  `contact_force_delta_mean=0.32209`, `contact_force_peak=1385.74`
- within-run rescore remained stable:
  `candidate_rescore_score_delta_max=0.0`

Key evidence:

- with guided disabled, sample checksum/signature/probe no longer first
  diverged at `window=0, iteration=0`
- first sample checksum/signature/probe divergence moved to
  `window=0, iteration=1`, after the first rank/score decision
- candidate top order/set still diverged at `window=0, iteration=0`
- same-index score delta still diverged at `window=0, iteration=0`:
  candidate `380` delta `-0.98316`
- later same-index score delta max reached `17.0232`

Interpretation:

The guided candidate or an upstream control/reference source contributes to the
I/J first-iteration sample checksum drift, but it is not the only blocker. K/L
restores first-iteration sample consistency, yet MJX-Warp scoring/rank still
drifts on the same first candidate batch. Therefore the main root-cause path
returns to cross-process MJX-Warp rollout/scoring nondeterminism, with guided
candidate stability as a secondary cleanup item before promotion.

### Slot diagnostic M/N

Runs:

- `foot147_scoreonly_jump_seed1_js018_slotdiag32_repeatM_20260707_4090_gpu0`
- `foot147_scoreonly_jump_seed1_js018_slotdiag32_repeatN_20260707_4090_gpu0`

Reports:

- `foot147_scoreonly_jump_seed1_js018_slotdiag32_repeat_probe_20260707_4090_gpu0.json`
- `foot147_scoreonly_jump_seed1_js018_slotdiag32_rank_divergence_20260707_4090_gpu0.json`

Code/tooling delta before this run:

- `sample_diagnostics_version=3`
- added `sample_checksum_slots`
- added `iteration_sample_slot_checksums`

Result:

- repeat stability still failed: `classification=repeat_instability`
- rank diagnostics still failed: `classification=rank_divergence`
- both rows used `sample_diagnostics_version=3`
- both rows used one visible RTX 4090:
  `runtime_gpu_name=NVIDIA GeForce RTX 4090`
- speed stayed above the local `>=11x` line:
  repeatM `11.348x`, repeatN `11.347x`
- repeatM quality:
  `score=-1.95684`, `root=0.06486`, `body=0.07515`,
  `ee_global=0.08419`, `ee_local=0.04785`,
  `contact_mismatch_rate=0.270625`,
  `contact_force_active_mean=242.971`,
  `contact_force_delta_mean=0.32445`, `contact_force_peak=1794.69`
- repeatN quality:
  `score=-2.01556`, `root=0.06306`, `body=0.07296`,
  `ee_global=0.08302`, `ee_local=0.04876`,
  `contact_mismatch_rate=0.289375`,
  `contact_force_active_mean=251.344`,
  `contact_force_delta_mean=0.32862`, `contact_force_peak=1912.52`

Key evidence:

- first whole-sample checksum/signature divergence was again at
  `window=0, iteration=0`
- the new per-slot checksum localized the first sample drift to slot `1`:
  `window=0, iteration=0`, max delta about `1.0069e-4`
- slot `1` is the guided candidate slot for this configuration
- candidate top order/set still diverged at `window=0, iteration=0`
- same-index score delta still diverged at `window=0, iteration=0`:
  candidate `352` delta `-0.99418`
- later same-index score delta max reached `14.3907`
- within-run rescore remained stable:
  `candidate_rescore_score_delta_max=0.0`

Interpretation:

The guided-on first-iteration sample checksum drift is now localized to the
guided candidate slot, consistent with Planck's static audit of the zero-control
guided trace path. This explains the tiny whole-sample drift, but not the main
rank failure: M/N and K/L both show same-index score drift around `0.99` at the
first candidate ranking step. The main blocker remains MJX-Warp rollout/scoring
nondeterminism on effectively identical first-iteration candidate batches.

### First-row force diagnostic O/P

Runs:

- `foot147_scoreonly_jump_seed1_js018_noguided_firstrow_rankdiag32_repeatO_20260707_4090_gpu0`
- `foot147_scoreonly_jump_seed1_js018_noguided_firstrow_rankdiag32_repeatP_20260707_4090_gpu0`

Reports:

- `foot147_scoreonly_jump_seed1_js018_noguided_firstrow_rankdiag32_repeat_probe_20260707_4090_gpu0.json`
- `foot147_scoreonly_jump_seed1_js018_noguided_firstrow_rankdiag32_rank_divergence_20260707_4090_gpu0.json`

Single-variable diagnostic change relative to K/L:

- added `--mjx-contact-force-mode first_row`
- added `--mjx-contact-force-first-row-diagnostics`

Result:

- repeat stability still failed: `classification=repeat_instability`
- rank diagnostics still failed: `classification=rank_divergence`
- speed improved above `12x` for this single-seed diagnostic:
  repeatO `12.128x`, repeatP `12.143x`
- repeatO quality:
  `score=-1.99112`, `root=0.05572`, `body=0.06814`,
  `ee_global=0.07848`, `ee_local=0.04843`,
  `contact_mismatch_rate=0.286875`,
  `contact_force_active_mean=58.595`,
  `contact_force_delta_mean=0.10548`, `contact_force_peak=413.46`
- repeatP quality:
  `score=-1.90818`, `root=0.06458`, `body=0.07277`,
  `ee_global=0.07843`, `ee_local=0.04869`,
  `contact_mismatch_rate=0.255625`,
  `contact_force_active_mean=60.719`,
  `contact_force_delta_mean=0.11724`, `contact_force_peak=380.04`

Key evidence:

- first candidate top order/set still diverged at `window=0, iteration=0`
- first same-index score delta still diverged at `window=0, iteration=0`:
  candidate `115` delta `-0.69477`
- later same-index score delta max reached `12.3052`
- first-row mode greatly reduced force scale and peak versus sum-row mode, but
  it did not remove rank/repeat drift
- repeat instability still included contact mismatch and command qpos drift

Interpretation:

`contact_force_mode=first_row` is a useful diagnostic and may reduce force-scale
variance, but it is not a standalone repeat-stability fix. The result weakens
the hypothesis that summed solver-row force alone causes the first-rank drift.
The next scoring/contact diagnostic should inspect first-step physics/contact
state fields directly, especially `_warp_valid_contacts()` and flat Warp contact
row filtering.

### Score-component shadow diagnostic Q3/R3

Runs:

- `foot147_scoreonly_jump_seed1_js018_noguided_scorecomp32_repeatQ3_20260707_4090_gpu0`
- `foot147_scoreonly_jump_seed1_js018_noguided_scorecomp32_repeatR3_20260707_4090_gpu0`

Report:

- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_scorecomp32_Q3_R3_rank_divergence.json`

Diagnostic change relative to K/L:

- kept `--mjx-score-only-optimizer`
- kept no-guided candidate
- added `--mjx-score-only-output-rescore-diagnostics`
- added `--mjx-candidate-score-component-diagnostics-top-k 32`

Result:

- MJX and replay rows completed with artifacts for both repeats
- replay status was `ok` for both repeats
- rank/component divergence failed:
  `classification=rank_divergence`
- Q3 speedup was `6.530x`, R3 speedup was `6.527x`; this is expected
  diagnostic overhead from the full-metric shadow scorer and is not a formal
  speed candidate
- Q3 quality:
  `score=-1.91278`, `root=0.06678`, `body=0.07777`,
  `ee_global=0.08516`, `ee_local=0.04863`,
  `contact_mismatch_rate=0.25500`,
  `contact_force_active_mean=249.88`,
  `contact_force_delta_mean=0.32135`, `contact_force_peak=1910.01`
- R3 quality:
  `score=-2.00399`, `root=0.06494`, `body=0.07389`,
  `ee_global=0.08124`, `ee_local=0.04941`,
  `contact_mismatch_rate=0.28312`,
  `contact_force_active_mean=251.11`,
  `contact_force_delta_mean=0.31415`, `contact_force_peak=1626.48`

Key evidence:

- first rank divergence still occurred at `window=0, iteration=0`
- first same-index score drift was candidate `115`, delta `+0.71512`
- whole-sample checksum drift first appeared at `window=0, iteration=1`
- per-slot checksum drift first appeared at `window=0, iteration=1`, slot `1`
- component source was `score_only_output_rescore` for both repeats
- first score-component delta was `contact_force_peak` for candidate `52`,
  delta `-754.51` at `window=0, iteration=0`
- max score-component delta was also `contact_force_peak`: candidate `403`,
  delta `1708.63` at `window=37, iteration=0`

Interpretation:

The new component diagnostic localizes the earliest observable component drift
to contact-force peak, while root/body/EE tracking metrics remain in the same
quality band across repeats. This strengthens the contact-force/contact-row
diagnosis without making `first_row` a fix: O/P showed first-row force mode
reduces force scale but does not remove rank drift. Q3/R3 is diagnostic-only
because full-metric shadow rescoring drops speed below the formal `11x` gate.

## Code And Tooling Changes

Added an execution plan:

- `docs/superpowers/plans/2026-07-07-g1-wbc-mjx-4090-quality-recovery-execution-plan.md`

Enhanced the offline rank diagnostic:

- `scripts/diagnose_g1_wbc_mjx_candidate_rank_divergence.py`

New report fields in each pairwise entry:

- `compared_same_index_score_values`
- `same_index_score_delta_max`
- `same_index_score_delta_max_location`
- `first_same_index_score_delta`
- `compared_sample_slot_checksum_values`
- `first_sample_slot_checksum_divergence`
- `sample_slot_checksum_max_abs_delta`
- `score_component_sources`
- `compared_score_component_values`
- `first_score_component_delta`
- `score_component_delta_max`
- `score_component_delta_max_location`

New failure label:

- `pair_X_Y:candidate_same_index_score_delta`
- `pair_X_Y:candidate_sample_checksum_divergence`
- `pair_X_Y:candidate_sample_slot_checksum_divergence`
- `pair_X_Y:candidate_score_component_delta`

Test coverage:

- `tests/tasks/g1_wbc/test_mjx_candidate_rank_divergence.py`
  now covers same-index score drift, sample-checksum drift, and per-slot
  sample-checksum drift, plus score-component drift from optimizer history.
- `tests/tasks/g1_wbc/test_mjx_optimizer.py`
  now covers optimizer emission of `iteration_sample_checksums`,
  `sample_checksum_slots`, `iteration_sample_slot_checksums`, score-only-output
  score-component fallback, and JIT-compatible numeric source codes under rank
  diagnostics.
- `tests/tasks/g1_wbc/test_mjx_rollout.py` now covers the diagnostic-only
  `score_only_output_full_metrics` path while preserving the existing compact
  score-only-output behavior.
- `tests/tasks/g1_wbc/test_mjx_components.py` now verifies that only the
  score-only-output diagnostic shadow scorer requests full metrics.

After the K/L diagnostics, the optimizer diagnostics were advanced to
`sample_diagnostics_version=3`. Version 3 adds per-slot sample checksums for
reserved candidate localization. The I/J and K/L GPU reports above were
generated before this v3 field existed, so they only contain whole-tensor
checksum evidence; M/N contains the v3 per-slot evidence.

Verification after the v3 tooling change:

- `./.venv/bin/python -m pytest tests/tasks/g1_wbc/test_mjx_candidate_rank_divergence.py tests/tasks/g1_wbc/test_mjx_repeat_stability.py tests/tasks/g1_wbc/test_mjx_optimizer.py -q`
  -> `60 passed`
- `./.venv/bin/python -m py_compile scripts/diagnose_g1_wbc_mjx_candidate_rank_divergence.py spider/tasks/g1_wbc/mjx_optimizer.py`
  -> passed
- old v2 rank reports still parse through
  `scripts/diagnose_g1_wbc_mjx_candidate_rank_divergence.py`

Verification after the score-component diagnostic tooling change:

- `./.venv/bin/python -m pytest tests/tasks/g1_wbc/test_mjx_optimizer.py::MjxOptimizerTest::test_jitted_score_component_source_is_jax_return_compatible tests/tasks/g1_wbc/test_mjx_optimizer.py tests/tasks/g1_wbc/test_mjx_candidate_rank_divergence.py -q`
  -> `60 passed`
- `./.venv/bin/python -m pytest tests/tasks/g1_wbc/test_mjx_rollout.py tests/tasks/g1_wbc/test_mjx_components.py -q`
  -> `43 passed`
- `./.venv/bin/python -m py_compile scripts/diagnose_g1_wbc_mjx_candidate_rank_divergence.py spider/tasks/g1_wbc/mjx_optimizer.py spider/tasks/g1_wbc/mjx_rollout.py spider/tasks/g1_wbc/mjx_components.py`
  -> passed

### Offline force-frame/oracle diagnostics for Q3/R3

Reports:

- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_scorecomp32_Q3_force_frame_mismatch.json`
- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_scorecomp32_R3_force_frame_mismatch.json`
- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_scorecomp32_Q3_contact_force_oracle.json`
- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_scorecomp32_R3_contact_force_oracle.json`

Key evidence:

- Q3 MJX-vs-replay force spike checks classified both feet as
  `force_high_state_diverged`; the worst left/right deltas were about `515 N`
  and `535 N` with `qpos_norm` around `0.14-0.15`.
- R3 left had a near-state force spike:
  frame `445`, MJX `1626.48 N`, replay `603.78 N`, delta `1022.70 N`,
  `qpos_norm=0.0322`, same contact. R3 right was state-diverged.
- Q3 oracle comparison: raw peak `1910.01 N`, oracle peak `1515.47 N`,
  mean abs delta `23.19 N`, max abs delta `654.00 N`.
- R3 oracle comparison: raw peak `1626.48 N`, oracle peak `1816.16 N`,
  mean abs delta `25.68 N`, max abs delta `1107.19 N`.

Interpretation:

The replay mismatch is not only a late command-quality issue: there are
individual frames where saved MJX force and MuJoCo oracle force disagree at
hundreds of Newtons. R3 frame `445` is especially important because the state is
near enough that contact-force semantics or row aggregation remain plausible
root causes.

### First-row trace-only diagnostic S/T

Runs:

- `foot147_scoreonly_jump_seed1_js018_noguided_firstrowtrace_rankdiag32_repeatS_20260708_4090_gpu0`
- `foot147_scoreonly_jump_seed1_js018_noguided_firstrowtrace_rankdiag32_repeatT_20260708_4090_gpu0`

Reports:

- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_firstrowtrace_S_T_rank_divergence.json`
- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_firstrowtrace_S_force_frame_mismatch.json`
- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_firstrowtrace_T_force_frame_mismatch.json`
- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_firstrowtrace_S_contact_force_oracle.json`
- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_firstrowtrace_T_contact_force_oracle.json`

Single-variable diagnostic change relative to K/L:

- kept `contact_force_mode=sum_rows`
- kept no-guided candidate and score-only optimizer
- added `--mjx-contact-force-first-row-diagnostics` only, so first-row force was
  traced but did not replace the scoring force

Result:

- S speedup `11.910x`, T speedup `11.913x`; both MJX/replay rows completed.
- S MJX quality:
  `score=-1.91097`, `root=0.05312`, `body=0.06355`,
  `ee_global=0.07287`, `ee_local=0.05049`, `contact=0.26750`,
  `force_active=250.21`, `force_peak=1296.22`.
- T MJX quality:
  `score=-1.89935`, `root=0.06172`, `body=0.07067`,
  `ee_global=0.07746`, `ee_local=0.04843`, `contact=0.260625`,
  `force_active=252.58`, `force_peak=1700.10`.
- Rank comparison still failed with `classification=rank_divergence`;
  first top divergence remained at `window=0, iteration=0`.
- S first-row peak was `427.16 N`, while sum-row peak was `1296.22 N`;
  max absolute sum-minus-first delta was about `1008 N`.
- T first-row peak was `510.18 N`, while sum-row peak was `1700.10 N`;
  max absolute sum-minus-first delta was about `1349 N`.
- S/T force-frame mismatch classified both feet as
  `force_high_state_diverged`; oracle comparisons still showed hundreds of
  Newtons of raw-vs-oracle frame deltas.

Interpretation:

The sum-row aggregation can amplify the force peak by roughly `1.0-1.35 kN`
relative to first-row diagnostics on the same rollout, while preserving the
`>=11x` single-seed speed envelope. However, trace-only first-row diagnostics
do not solve rank/repeat instability. This made the next useful hook a
row-level source trace for the summed floor-contact peak.

### Peak-source trace hook and U diagnostic

Code/tooling delta:

- `spider/tasks/g1_wbc/mjx_physics.py` can now emit
  `floor_contact_force_peak_source` for Warp floor-contact summaries when
  first-row diagnostics are enabled.
- `spider/tasks/g1_wbc/mjx_rollout.py` carries the trace as
  `(frames, envs, 3, 8)` with columns
  `row_id, geom0, geom1, normal_force, first_row_force, group_sum_force, dim, efc0`.
- `spider/tasks/g1_wbc/mjx_backend.py`, `spider/tasks/g1_wbc/rollout.py`, and
  `spider/tasks/g1_wbc/evaluate.py` preserve and export the optional field in
  `rollout.npz`.

Verification:

- `./.venv/bin/python -m pytest tests/tasks/g1_wbc/test_mjx_real_physics.py::MjxRealPhysicsTest::test_warp_floor_contact_summary_reports_peak_source_rows tests/tasks/g1_wbc/test_mjx_backend_integration.py::MjxBackendIntegrationTest::test_mjx_backend_builds_rollout_from_execute_trace_when_available tests/tasks/g1_wbc/test_evaluate_replay_export.py::EvaluateReplayExportTest::test_save_rollout_preserves_floor_contact_force_peak_source -q`
  -> `3 passed`
- `./.venv/bin/python -m pytest tests/tasks/g1_wbc/test_mjx_real_physics.py tests/tasks/g1_wbc/test_mjx_backend_integration.py::MjxBackendIntegrationTest::test_mjx_backend_builds_rollout_from_execute_trace_when_available tests/tasks/g1_wbc/test_evaluate_replay_export.py tests/tasks/g1_wbc/test_mjx_rollout.py -q`
  -> `63 passed, 6 skipped`
- `./.venv/bin/python -m py_compile spider/tasks/g1_wbc/rollout.py spider/tasks/g1_wbc/mjx_backend.py spider/tasks/g1_wbc/evaluate.py spider/tasks/g1_wbc/mjx_physics.py spider/tasks/g1_wbc/mjx_rollout.py`
  -> passed

Run:

- `foot147_scoreonly_jump_seed1_js018_noguided_peaksource_rankdiag32_repeatU_20260708_4090_gpu0`

Result:

- MJX status `ok`; replay status `ok`.
- Same-seed speedup `11.885x` on `NVIDIA GeForce RTX 4090`, visible device
  `["0"]`.
- MJX quality:
  `score=-1.99172`, `root=0.06062`, `body=0.06857`,
  `ee_global=0.07410`, `ee_local=0.04637`, `contact=0.28750`,
  `force_active=248.67`, `force_delta=0.31348`, `force_peak=1523.09`.
- Replay quality:
  `score=-1.98680`, `root=0.06075`, `body=0.06840`,
  `ee_global=0.07386`, `ee_local=0.04595`, `contact=0.28375`,
  `force_active=243.63`, `force_delta=0.29047`, `force_peak=1793.32`.
- U is diagnostic-only: it is a single run and does not prove repeat stability
  or full `jump/walk` acceptance.

Peak-source extraction from U:

- Global foot peak: frame `430`, right foot, sum force `1523.09 N`.
  Source row: `row_id=1`, `geom=(0,34)`, `normal_force=783.08 N`,
  `first_row_force=208.55 N`, `group_sum_force=1523.09 N`, `dim=3`,
  `efc0=4`.
- At the same frame, traced right first-row group force was `317.17 N`, so
  sum-minus-first at the peak was about `1205.93 N`.
- Left-foot max: frame `503`, sum force `1459.12 N`, source
  `row_id=1`, `geom=(0,17)`, `normal_force=494.49 N`,
  `first_row_force=141.07 N`, `dim=3`, `efc0=4`.
- Geom ids on this profile:
  `0=terrain`, `17=robot/left_foot4_collision`,
  `20=robot/left_foot7_collision`,
  `31=robot/right_foot1_collision`,
  `34=robot/right_foot4_collision`,
  `37=robot/right_foot7_collision`.

Offline U diagnostics:

- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_U_force_frame_mismatch.json`
  classified both feet as `force_high_state_diverged`.
  Left worst: frame `640`, MJX `974.83 N`, replay `326.46 N`,
  delta `648.37 N`, `qpos_norm=0.1998`.
  Right worst: frame `638`, MJX `876.16 N`, replay `0.00 N`,
  delta `876.16 N`, `qpos_norm=0.1998`, contact mismatch.
- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_U_contact_force_oracle.json`
  reported raw peak `1523.09 N`, oracle peak `1344.35 N`,
  mean abs delta `23.43 N`, max abs delta `783.31 N`.
- Worst oracle frame `503` had raw `[1459.12, 1383.67, 0]` versus oracle
  `[850.86, 600.36, 0]`, so both feet over-reported by roughly
  `608 N` and `783 N`.

Interpretation:

The new hook confirms that the largest summed peaks are not anonymous metric
spikes: they come from specific terrain-foot capsule rows, often with `dim=3`
and group sums much larger than the first row force. It does not by itself
prove whether the bug is row ordering, solver-row aggregation semantics, or
state divergence, but it gives the next diagnostic a concrete target:
compare row-level active masks, `nacon[0]`, row order, `dim/efc0`, and
per-row solver force for the same peak frames across same-argv repeats.

### U/V same-argv repeat with executed peak-source traces

Runs:

- `foot147_scoreonly_jump_seed1_js018_noguided_peaksource_rankdiag32_repeatU_20260708_4090_gpu0`
- `foot147_scoreonly_jump_seed1_js018_noguided_peaksource_rankdiag32_repeatV_20260708_4090_gpu0`

Result:

- Both runs were MJX/replay `ok` on RTX 4090 and preserved the formal-speed
  envelope for this one-seed diagnostic:
  U `same_seed_speedup=11.885x`, V `same_seed_speedup=11.874x`.
- U MJX quality:
  `score=-1.99172`, `root=0.06062`, `body=0.06857`,
  `ee_global=0.07410`, `ee_local=0.04637`, `contact=0.28750`,
  `force_active=248.67`, `force_delta=0.31348`, `force_peak=1523.09`.
- V MJX quality:
  `score=-1.89276`, `root=0.06144`, `body=0.07238`,
  `ee_global=0.08246`, `ee_local=0.04834`, `contact=0.26125`,
  `force_active=244.72`, `force_delta=0.28234`, `force_peak=1448.34`.
- V replay quality:
  `score=-1.88362`, `root=0.06168`, `body=0.07230`,
  `ee_global=0.08227`, `ee_local=0.04794`, `contact=0.25437`,
  `force_active=238.98`, `force_delta=0.28309`, `force_peak=1471.02`.

Peak-source comparison:

- U global peak: frame `430`, right foot, source
  `row_id=1`, `geom=(0,34)`, `normal_force=783.08 N`,
  `first_row_force=208.55 N`, `group_sum_force=1523.09 N`,
  `dim=3`, `efc0=4`.
- V global peak: frame `727`, left foot, source
  `row_id=2`, `geom=(0,20)`, `normal_force=337.64 N`,
  `first_row_force=93.69 N`, `group_sum_force=1448.34 N`,
  `dim=3`, `efc0=8`.
- V right-foot max: frame `507`, source
  `row_id=1`, `geom=(0,34)`, `normal_force=425.36 N`,
  `first_row_force=135.76 N`, `group_sum_force=1039.22 N`,
  `dim=3`, `efc0=4`.

Offline reports:

- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_U_V_rank_divergence.json`
  classified `rank_divergence`. First top-order/top-set divergence was already
  at `window=0, iteration=0` with overlap `24/32`; the first same-index score
  delta was candidate `64`, `-1.06699`. Sample checksum drift at that point
  was tiny (`1.53e-05`), so contact scoring remains the leading amplifier.
- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_U_V_repeat_stability.json`
  classified `repeat_instability`. Score range was `0.09896`, contact range
  `0.02625`, and command `qpos` divergence crossed `1e-3` by frame `102`,
  `0.05` by frame `121`, and `0.1` by frame `122`.
- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_V_force_frame_mismatch.json`
  classified both feet as `force_high_state_diverged`. Left worst was frame
  `727`, MJX `1448.34 N` vs replay `322.55 N`, delta `1125.79 N`,
  `qpos_norm=0.0638`, with same contact. Right worst was frame `507`, delta
  `549.37 N`, `qpos_norm=0.0952`.
- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_V_contact_force_oracle.json`
  reported raw peak `1448.34 N`, oracle peak `1766.28 N`, mean abs delta
  `22.54 N`, and max abs delta `1093.74 N`.

Interpretation:

U/V proves the executed trace hook can stay within the `>=11x` same-seed speed
envelope, but it also proves the no-guided score-only surface is not repeat
stable. The executed peak-source rows are useful after divergence, yet the
rank report still says the first material split is at the very first optimizer
iteration, before the later rollout peak frames.

### Candidate-level peak-source companion diagnostics and AA/AB

Code/tooling delta:

- `spider/tasks/g1_wbc/mjx_scoring.py` now carries
  `contact_force_peak_source` through the score accumulator and returns the
  source row for the candidate's maximum `contact_force_peak`.
- `spider/tasks/g1_wbc/mjx_optimizer.py` exports optional
  `iteration_score_component_peak_source_*` fields alongside
  score-component diagnostics.
- `spider/tasks/g1_wbc/mjx_components.py` routes
  `--mjx-score-only-output-rescore-diagnostics` through diagnostic physics when
  first-row diagnostics are requested, so the full-metric shadow scorer can see
  `floor_contact_force_peak_source` while the primary score-only optimizer
  remains lightweight.
- `scripts/diagnose_g1_wbc_mjx_candidate_rank_divergence.py` reports
  `left_peak_source` and `right_peak_source` for the first and maximum
  score-component delta.

Verification:

- `./.venv/bin/python -m pytest tests/tasks/g1_wbc/test_mjx_candidate_rank_divergence.py -q`
  -> `10 passed`
- Earlier targeted red/green coverage added:
  `test_contact_force_peak_source_tracks_max_peak`,
  `test_optimize_window_records_score_component_peak_sources`,
  `test_candidate_rank_divergence_reports_score_component_peak_sources`, and
  `test_score_only_output_rescore_can_use_first_row_diagnostic_physics`.

Runs:

- `foot147_scoreonly_jump_seed1_js018_noguided_peaksource_scorecomp32_repeatAA_20260708_4090_gpu0`
- `foot147_scoreonly_jump_seed1_js018_noguided_peaksource_scorecomp32_repeatAB_20260708_4090_gpu0`

Diagnostic caveat:

- AA/AB are not formal candidates. The full-metric shadow scorer and
  score-component source trace intentionally reduce same-seed speedup to about
  `6.52x` and `6.51x`; these runs are for row-level diagnosis only.
- Both runs produced valid candidate source fields:
  `candidate_score_component_peak_source_available=True`, width `8`, and
  `2560/2560` top-k source entries were non-sentinel in each run.

Report:

- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_scorecomp32_AA_AB_rank_divergence.json`
  classified `rank_divergence`.
- First top-order/top-set divergence: `window=0, iteration=0`,
  overlap `25/32`.
- First top1 divergence: `window=6, iteration=1`, overlap `24/32`.
- First same-index score delta: `window=0, iteration=0`, candidate `384`,
  delta `-0.62525`.
- First sample checksum divergence at the same first iteration was still tiny:
  max abs delta `1.53e-05`.
- First score-component delta:
  `window=0, iteration=0`, candidate `52`,
  field `contact_force_peak`, AA `1379.45 N`, AB `709.39 N`,
  delta `-670.05 N`.
  AA source was `row_id=840`, `geom=(0,34)`, `normal_force=369.07 N`,
  `first_row_force=79.20 N`, `group_sum_force=1379.45 N`, `dim=3`,
  `efc0=4`.
  AB source was `row_id=796`, `geom=(0,34)`, `normal_force=281.21 N`,
  `first_row_force=71.03 N`, `group_sum_force=709.39 N`, `dim=3`,
  `efc0=28`.
- Maximum score-component delta:
  `window=21, iteration=0`, candidate `208`,
  field `contact_force_peak`, AA `3357.15 N`, AB `1272.50 N`,
  delta `-2084.66 N`.
  AA source was `row_id=1064`, `geom=(0,31)`, `normal_force=729.08 N`,
  `first_row_force=117.07 N`, `group_sum_force=3357.15 N`, `dim=3`,
  `efc0=0`.
  AB source was `row_id=1661`, `geom=(0,37)`, `normal_force=712.54 N`,
  `first_row_force=112.60 N`, `group_sum_force=1272.50 N`, `dim=3`,
  `efc0=8`.
- Geom ids on this profile:
  `31=robot/right_foot1_collision`,
  `34=robot/right_foot4_collision`,
  `37=robot/right_foot7_collision`.

Interpretation:

AA/AB localizes the first trustworthy candidate-level component drift to
`contact_force_peak`, not to root/body tracking or candidate sampling. The
first component split compares the same terrain/right-foot capsule pair
`geom=(0,34)` but lands on different solver rows/`efc0` values and a roughly
`670 N` summed-force difference. The maximum split later moves across
right-foot capsules and exceeds `2 kN`. The next fix should therefore target
deterministic contact row grouping / force aggregation semantics under
MJX-Warp, while keeping `wxy_parity` fixed and leaving actuator/timing/policy
parameters untouched.

## Current Diagnosis

The strongest current diagnosis is cross-process MJX-Warp rollout/scoring
nondeterminism, with contact-force peak now the best localized scoring
component. The no-guided K/L diagnostic shows first-iteration samples can be
made checksum-consistent, while score/rank still diverge at
`window=0, iteration=0`. Guided-candidate/upstream control drift is a secondary
contributor because I/J showed sample checksum drift at the same first
iteration when guided candidates were enabled.

Contact/force is now the leading localized scoring component:

- Q3/R3 shows earliest component drift at `contact_force_peak`
- O/P shows `first_row` force mode reduces force scale but does not remove
  first-rank drift
- S/T shows trace-only first-row diagnostics preserve the `>=11x` envelope but
  sum-row force can exceed first-row force by more than `1 kN`
- U localizes a representative peak to terrain-foot capsule rows and exposes
  `row_id/geom/dim/efc0` for follow-up comparisons
- U/V confirms the no-shadow surface remains fast but repeat-unstable
- AA/AB localizes first trustworthy candidate-level component drift to
  `contact_force_peak` on terrain/right-foot solver rows
- active force is near Stage0 scale in the promising score-only matrices, but
  contact mismatch and peak-force variance remain promotion blockers
- repeat instability blocks promotion before another contact/profile tweak is
  justified

## Rejected Directions

- JAX backend on this single-seed score-only surface: too slow and not quality
  improving.
- `--mjx-cem-update-min-top-score-gap 0.5`: speed ok, repeat and quality worse.
- score-component top-k from the primary score-only optimizer result: no
  component fields; use the score-only-output full-metric shadow diagnostic
  instead.
- `--mjx-contact-force-mode first_row` as a standalone fix: useful force-scale
  diagnostic and fast, but rank/repeat still fail.
- Q3/R3 score-component diagnostics as a formal candidate: useful diagnosis,
  but full-metric shadow rescoring reduces single-seed speedup to about `6.53x`.
- AA/AB score-component peak-source diagnostics as a formal candidate: useful
  diagnosis, but intentionally slow (`~6.5x`) because of the full-metric shadow
  scorer and source trace.

## Next Judgment

Do not run a full `jump/walk` promotion matrix from the current score-only
surface yet. The matrix would be expensive and still blocked by repeat
instability.

Recommended next work:

1. Audit MJX-Warp rollout/contact scoring for cross-process score drift on
   checksum-consistent first-iteration samples, with priority on
   `contact_force_peak`, `_warp_valid_contacts()`, `nacon[0]` flat-row
   filtering, first-step contact row ordering, and the new
   `floor_contact_force_peak_source` row/geoms/`dim`/`efc0` trace. AA/AB now
   gives concrete candidate-level rows on `right_foot*_collision`, including a
   same-geom first split with different `efc0` values.
2. Add a narrower diagnostic for reserved candidate slots (`sample0` current
   controls and `sample1` guided controls) before re-enabling guided-candidate
   promotion experiments, because I/J showed guided-enabled sample drift at the
   first iteration. M/N confirms slot `1`, the guided candidate, is the first
   drifting slot under guided-on repeats.
3. Keep full-metric score-component diagnostics out of formal speed candidates.
   Use it only for focused one-seed diagnosis, then rerun the no-shadow surface
   for speed verification.
4. Only after repeat stability improves, rerun a full `jump/walk` matrix and
   gate it against `criteria.md`.
