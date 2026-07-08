# G1 WBC MJX RTX 4090 Handoff: Repeat Stability And Quality Recovery

Date: 2026-07-07

Branch: `codex/g1-wbc-mjx-full-rollout`

Workspace:
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/PITL_worktrees/spider`

Primary run directory:
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs`

## Purpose

Continue the RTX 4090 MJX/JAX migration debugging in a fresh thread. The active
sub-goal is to keep worst same-seed speedup at or above `11x` while recovering
MJX MPC optimization quality, contact, and force behavior to the Stage0
MuJoCo-Warp sweetpoint level for `jump` and `walk`.

Do not treat the current state as complete. The strongest candidate reaches the
speed target and has a promising collision surface, but repeat stability is not
solved and `jump` quality is not reliably passing.

## Suggested Skills

- `superpowers:systematic-debugging`: use for the next repeat-instability
  diagnosis loop.
- `superpowers:dispatching-parallel-agents`: use when splitting independent
  artifact inspection, candidate-rank comparison, and code-path audit work.
- `superpowers:verification-before-completion`: use before claiming a candidate
  passes acceptance.
- `handoff`: use again when compacting this work for another continuation.

## Acceptance Contract

Use the criteria in:

- `docs/tasks/g1_wbc_mjx_full_rollout/criteria.md`

Important local sub-goal constraint from this thread:

- Current practical target is `>=11x` same-seed Stage0 speedup on RTX 4090.
- Do not change actuator, `sim_dt`, policy cadence, policy weights, or other
  training-bound parameters.
- Keep the WXY parity physical surface as the quality anchor.
- Collision/profile changes are allowed only as single-variable, reversible
  experiments.
- Formal candidates must pass Stage0 denominator, MJX quality, replay,
  contact/force, provenance, single-GPU, and speed gates.

## GPU And Sandbox Notes

See `AGENTS.md` for the durable RTX 4090 visibility note. In short:

- Sandboxed `nvidia-smi` may fail because Codex cannot see NVIDIA device nodes.
- With user-approved unsandboxed execution, this machine sees
  `NVIDIA GeForce RTX 4090`, driver `580.126.09`.
- Formal/diagnostic GPU runs should use one visible GPU:
  `CUDA_VISIBLE_DEVICES=0`, `--device cuda:0`.
- Use escalated execution for real GPU runs when the sandbox hides the device.

Recommended command prefix for real runs:

```bash
env CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. ./.venv/bin/python
```

## Current State

The thread is paused; no experiment process was running at handoff time.

The active goal is still open: recover MJX MPC quality/contact/force to
sweetpoint level on RTX 4090 while keeping worst same-seed speedup `>=11x`.

The latest result is a repeat-stability failure:

- `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/foot147_scoreonly_jump_seed1_js018_repeat_probe_20260707_4090_gpu0.json`

Classification:

- `repeat_instability`

Failures include:

- `metric_range:score`
- `metric_range:root_pos_error_mean`
- `metric_range:body_global_pos_error_mean`
- `metric_range:ee_global_pos_error_mean`
- `metric_range:ee_local_pos_error_mean`
- `metric_range:contact_mismatch_rate`
- `metric_range:contact_force_peak`
- `pair_0_1:best_index_divergence`
- `pair_0_1:current_selected_divergence`
- `pair_0_1:command_qpos_delta`

Interpretation: score-only fixed within-run candidate rescore instability, but
same-argv cross-run command selection is still unstable.

## Important Artifacts

Stage0 denominator:

- `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/stage0_baseline_current_schema_20260706_4090_gpu0/baseline_manifest.json`
- Schema version: `1`
- Contact force semantics: `pyramidal_contact_normal_v1`
- Promoted seeds: `jump: 1`, `walk: 0`

Best prior pass-like candidate, not final due repeat risk:

- `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/foot147_quality_jump_p4_js012_gap001_walk_canonical_20260707_4090_gpu0/acceptance_report.json`
- `passed=True`, classification `pass_h100_milestone`
- Collision profile: `wxy_explicit_floor_leg_foot147_pairs_7caps`
- Speed passed:
  - jump worst `11.295905x`
  - walk worst `11.462182x`
- MJX quality passed for both `jump` and `walk`
- Do not promote this as final without resolving repeat stability.

Current strongest score-only full matrix:

- `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/foot147_scoreonly_jump_p4_walk_p15_js012_gap001_20260707_4090_gpu0/acceptance_report.json`
- `passed=False`, classification `parity_failure`
- Speed passed:
  - jump worst `11.196153x`
  - walk worst `11.257481x`
- `walk` MJX quality passed.
- `jump` MJX quality failed on score/root/body/EE metrics, mainly seed 1.

Score-only `jump` joint-sigma `0.18` probe:

- `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/foot147_scoreonly_jump_p4_js018_gap001_probe_20260707_4090_gpu0/acceptance_report.json`
- Report classification is `invalid_benchmark` because it only covers `jump`.
- Useful diagnostic data: speed remained above `11x`, but quality varied and
  repeat probe later showed same-argv instability.

Earlier non-score-only repeat and divergence diagnostics:

- `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/foot147_jump_seed0_repeat_probe_20260707_4090_gpu0.json`
- `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/foot147_jump_seed0_command_divergence_20260707_4090_gpu0.json`

## What Has Been Learned

Collision simplification is useful but not sufficient.

- Original WBC runtime model dynamically built from
  `spider/assets/robots/unitree_g1/wxy_g1.xml` had broad collision behavior:
  `npair=0`, default broadphase, all `_collision` geoms collidable, and each
  foot had 7 capsule geoms.
- MJX training assets in `g1_assets/` use a different collision strategy:
  geoms default to non-collidable and scene `<pair>` entries explicitly enable
  contacts; feet are often simplified to 1 box + 3 capsules.
- The active experimental profile keeps the WXY model surface but adds explicit
  floor/leg/foot pairs while retaining the 7 capsule foot structure:
  `wxy_explicit_floor_leg_foot147_pairs_7caps`.
- This profile reduces contact surface pressure enough to make `>=11x` viable,
  but it does not by itself guarantee optimization quality.

Force behavior improved materially.

- Earlier observations showed very high force peaks and active force means.
- In the score-only full matrix, active force is close to Stage0 baseline:
  about `1.01x` for `jump` and `1.02x` for `walk`.
- Peak force is still less clean:
  `jump` peak max ratio is about `1.56x`; `walk` replay still has one high
  peak-force case.

Full-metrics MJX scorer had within-run rescore instability.

- A diagnostic with candidate rescore showed the same candidates could change
  top-1 selection or score when rescored.
- Strip-live-MJX-data did not fix it.
- Score-only optimizer eliminated this within-run rescore issue in the checked
  diagnostic: top-1 changed iteration sum became `0`.

Score-only optimizer is promising but not done.

- It preserves the `>=11x` speed envelope.
- It improves/normalizes active force.
- It still suffers cross-run instability: the same effective argv can produce
  different selected commands and materially different `jump` quality.

Rejected or weak branches so far:

- `--mjx-max-control-delta 0.35` on score-only `jump/seed1` caused severe
  quality collapse.
- Lowering joint sigma to `0.08` for score-only `jump/seed1` caused severe
  quality collapse.
- Raising joint sigma to `0.18` looked good in one single-seed run but failed
  repeat stability when compared against the matrix run.

## Code Changes Known From This Thread

The worktree is dirty with broad pre-existing changes. Do not revert unrelated
files.

One recent runner edit made during this thread:

- `scripts/run_g1_wbc_mjx_acceptance.py`

Purpose:

- Added `--mjx-candidate-score-component-diagnostics-top-k`.
- Validates non-negative values.
- Passes the flag into MJX planned argv.
- Drops the flag from replay argv.
- Adds provenance consistency checks and row flattening for
  `candidate_score_component_diagnostics_top_k`.

Verification already run:

- `python3 -m py_compile scripts/run_g1_wbc_mjx_acceptance.py`
- `env PYTHONPATH=. python3 scripts/run_g1_wbc_mjx_acceptance.py --help`
- Dry-run confirmed the new score-component diagnostic flag appears in MJX argv
  and not replay argv.

Suggested follow-up tests:

- `env PYTHONPATH=. pytest tests/tasks/g1_wbc/test_mjx_acceptance_runner.py -q`
- Add or inspect targeted coverage for the new diagnostics flag if continuing
  runner edits.

## Next Recommended Work

First priority: diagnose cross-run repeat instability.

Run two repeated score-only `jump/seed1/js0.18` jobs with rank diagnostics. The
purpose is to determine whether divergence comes from different sampled
candidate fingerprints or from scoring/selection of the same samples.

Use the same config for repeat A and B, changing only `--output-dir`.

Template for repeat A:

```bash
env CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=. ./.venv/bin/python scripts/run_g1_wbc_mjx_acceptance.py \
  --baseline-manifest /home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/stage0_baseline_current_schema_20260706_4090_gpu0/baseline_manifest.json \
  --output-dir /home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/foot147_scoreonly_jump_seed1_js018_rankdiag32_repeatA_20260707_4090_gpu0 \
  --device cuda:0 --target h100_speedup --min-speedup 11 --required-gpu-name-fragment 4090 \
  --collision-profile wxy_explicit_floor_leg_foot147_pairs_7caps \
  --only-motion jump --only-seed 1 --skip-report \
  --mjx-impl warp --mjx-warp-naconmax 20000 --mjx-warp-njmax 192 \
  --mjx-model-iterations 4 --mjx-model-ls-iterations 5 \
  --mjx-mpc-joint-sigma 0.18 \
  --mjx-guided-candidate --mjx-guided-candidate-period 4 \
  --mjx-min-top-score-gap 0.01 \
  --mjx-score-only-optimizer \
  --mjx-candidate-rank-diagnostics-top-k 32 \
  --mjx-candidate-rescore-diagnostics
```

Then repeat with `repeatB`.

Useful comparison tools already present:

- `scripts/check_g1_wbc_mjx_repeat_stability.py`
- `scripts/diagnose_g1_wbc_mjx_command_divergence.py`
- `scripts/diagnose_g1_wbc_mjx_candidate_rank_divergence.py`

Inspect CLI help before use:

```bash
env PYTHONPATH=. python3 scripts/diagnose_g1_wbc_mjx_candidate_rank_divergence.py --help
```

Expected interpretations:

- If sample fingerprints differ early despite same seed and effective argv,
  inspect PRNG/sample generation and seed construction.
- If sample fingerprints match but scores/ranks differ, inspect score-only
  scoring, MJX-Warp physics determinism, synchronization, and device-side state.
- If early ranks match but selected commands diverge later, focus on tie/gap
  sensitivity and acceptance/selection stabilization.

## Stop Conditions For Next Thread

Do not declare success until a full `jump/walk` acceptance matrix passes:

- speed worst same-seed `>=11x`
- MJX quality
- replay parity
- contact/force semantics
- provenance checks
- single RTX 4090 GPU evidence

If the next repeated rank diagnostic still fails without exposing fingerprints
or enough candidate detail, add the smallest possible diagnostic field rather
than changing physics parameters.

If a candidate only passes once but fails repeat stability, keep it as a probe,
not a final migration result.
