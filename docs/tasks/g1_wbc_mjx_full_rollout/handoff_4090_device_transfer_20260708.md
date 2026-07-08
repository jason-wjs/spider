# G1 WBC MJX RTX 4090 Device Transfer Handoff

Date: 2026-07-08

Branch: `codex/g1-wbc-mjx-full-rollout`

Workspace on the source machine:
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/PITL_worktrees/spider`

Primary external run directory on the source machine:
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs`

## Purpose

This handoff prepares the work for continuation on another RTX 4090 machine.
The current checkpoint is exploratory and intentionally not cleaned up. The
next machine should first organize the repository state, then continue the
MJX/MuJoCo-Warp contact-force diagnosis.

Do not treat this branch as a completed migration. The active goal remains:
recover G1 WBC MJX MPC optimization quality, contact, and force behavior to the
Stage0 MuJoCo-Warp sweetpoint level while keeping worst same-seed speedup
`>=11x`.

## Suggested Skills

- `superpowers:systematic-debugging`: continue the repeat-instability diagnosis
  loop.
- `superpowers:verification-before-completion`: use before claiming any gate,
  candidate, test, or branch state is passing.
- `superpowers:dispatching-parallel-agents`: use for independent cleanup,
  artifact inventory, and code-path audits after the branch is pulled on the
  new machine.
- `handoff`: use again after the new-machine cleanup and first verification
  pass.

## Must-Preserve Constraints

- Keep Stage0 MuJoCo-Warp sweetpoint and `wxy_parity` as the quality anchor.
- Do not change actuator, `sim_dt`, policy cadence, policy weights, or other
  training-bound parameters.
- Any collision/profile change must be single-variable and reversible.
- Do not call a run successful unless Stage0 denominator, MJX quality, replay,
  contact/force, provenance, single-GPU evidence, and speed gates all pass.
- The current focus is MJX/MuJoCo-Warp contact-force diagnosis, quality-first
  experiments, conservative speed work, and necessary tooling. No training, no
  policy-weight changes, and no large asset restructuring.

## Repository State At Transfer

This checkpoint is being committed and pushed before repository cleanup because
the user plans to do that cleanup on the next RTX 4090 machine.

Before this handoff file was added, the worktree contained a broad exploratory
diff across runtime code, runners, diagnostics, tests, docs, and lockfile
changes. The latest local status showed:

- `37` tracked files changed.
- About `25k` insertions and `2.7k` deletions in tracked diffs.
- Multiple untracked diagnostic scripts, tests, and docs.

This is expected for the transfer checkpoint. The next step should be cleanup
and classification, not more feature growth.

Recommended cleanup buckets on the new machine:

1. `formal path`: code/flags needed for no-shadow speed-gated acceptance runs.
2. `diagnostic-only`: slow shadow scorer, peak-source, row/source traces, and
   offline report scripts.
3. `evidence/docs`: criteria, handoffs, diagnostic updates, and report paths.
4. `candidate for rollback/shelve`: exploratory changes not needed for either
   formal path or current contact-force diagnosis.

## Existing Reference Documents

Use these instead of re-deriving the full history:

- Acceptance contract:
  `docs/tasks/g1_wbc_mjx_full_rollout/criteria.md`
- Previous repeat-stability handoff:
  `docs/tasks/g1_wbc_mjx_full_rollout/handoff_4090_repeat_stability_20260707.md`
- Current diagnostic log:
  `docs/tasks/g1_wbc_mjx_full_rollout/diagnostic_update_4090_repeat_rank_20260707.md`
- Older H100-to-4090 handoff:
  `docs/tasks/g1_wbc_mjx_full_rollout/handoff_h100_to_4090_20260704.md`
- GPU sandbox note:
  `AGENTS.md`

## Current Diagnosis

The strongest current diagnosis is cross-process MJX-Warp scoring/contact-force
nondeterminism. The useful signal has narrowed from general quality drift to
`contact_force_peak` differences in terrain-foot contact rows and solver-row
aggregation.

Key evidence is recorded in:

- `docs/tasks/g1_wbc_mjx_full_rollout/diagnostic_update_4090_repeat_rank_20260707.md`

Short version:

- U/V no-shadow same-argv repeats stayed inside the speed envelope:
  U `11.885x`, V `11.874x`.
- U/V still classified as repeat/rank unstable. Divergence appears at
  `window=0, iteration=0`, while sample checksum drift is only about
  `1.53e-05`.
- AA/AB candidate-level source diagnostics show the first trustworthy
  score-component split at `contact_force_peak`, not root/body tracking.
- First AA/AB component split:
  `window=0`, `iteration=0`, candidate `52`,
  AA `1379.45 N` vs AB `709.39 N`, both on `geom=(0,34)`
  (`robot/right_foot4_collision`) but with different `efc0` values.
- Maximum AA/AB component split later exceeds `2 kN`.
- AA/AB are diagnostic-only because the full-metric shadow scorer reduces
  same-seed speedup to about `6.5x`.

## Important External Artifacts

These paths were on the source machine and may not exist on the new machine
unless copied separately. The key numbers are summarized in the docs above.

Stage0 denominator:

- `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/stage0_baseline_current_schema_20260706_4090_gpu0/baseline_manifest.json`

No-shadow speed-preserving peak-source repeats:

- `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_rankdiag32_repeatU_20260708_4090_gpu0`
- `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_rankdiag32_repeatV_20260708_4090_gpu0`

Diagnostic-only candidate-level source repeats:

- `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_scorecomp32_repeatAA_20260708_4090_gpu0`
- `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_scorecomp32_repeatAB_20260708_4090_gpu0`

Offline JSON reports on the source machine:

- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_U_V_rank_divergence.json`
- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_U_V_repeat_stability.json`
- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_V_force_frame_mismatch.json`
- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_V_contact_force_oracle.json`
- `/tmp/foot147_scoreonly_jump_seed1_js018_noguided_peaksource_scorecomp32_AA_AB_rank_divergence.json`

## Verification Already Run Before Transfer

Fresh verification before this handoff:

```bash
./.venv/bin/python -m pytest \
  tests/tasks/g1_wbc/test_mjx_candidate_rank_divergence.py \
  tests/tasks/g1_wbc/test_mjx_scoring.py \
  tests/tasks/g1_wbc/test_mjx_optimizer.py \
  tests/tasks/g1_wbc/test_mjx_rollout.py \
  tests/tasks/g1_wbc/test_mjx_components.py \
  tests/tasks/g1_wbc/test_evaluate_replay_export.py -q
```

Result:

- `140 passed in 8.49s`

Fresh syntax verification:

```bash
./.venv/bin/python -m py_compile \
  scripts/diagnose_g1_wbc_mjx_candidate_rank_divergence.py \
  spider/tasks/g1_wbc/mjx_scoring.py \
  spider/tasks/g1_wbc/mjx_optimizer.py \
  spider/tasks/g1_wbc/mjx_components.py \
  spider/tasks/g1_wbc/mjx_physics.py \
  spider/tasks/g1_wbc/mjx_rollout.py \
  spider/tasks/g1_wbc/mjx_backend.py \
  spider/tasks/g1_wbc/evaluate.py \
  spider/tasks/g1_wbc/rollout.py
```

Result:

- passed

These checks are not proof of final task completion. They only verify the
current diagnostic/tooling edits.

## Recommended First Steps On The New Machine

1. Pull the remote branch and confirm it matches the pushed transfer commit.
2. Confirm RTX 4090 visibility outside the Codex sandbox if needed, following
   `AGENTS.md`.
3. Run a lightweight repository inventory before adding new diagnostics:
   classify files into formal, diagnostic-only, docs/evidence, and rollback
   buckets.
4. Re-run the verification commands above in the new environment.
5. If external artifacts are not copied, rerun the minimal U/V and AA/AB
   diagnostics only after the repository inventory is complete.
6. Continue with the narrow row-level contact diagnostic proposed by the
   subagent audit:
   `floor_contact_force_top_rows` with row id, `worldid`, `nacon0`, active flag,
   geoms, distance/margin, `dim`, `efc*`, `force*`, first-row force, normal sum,
   group sum, and active-row count.

## Do Not Do Next

- Do not promote the current branch as a final `jump/walk` success.
- Do not run a full promotion matrix before repeat instability is improved.
- Do not add another collision/profile experiment before the contact row
  aggregation evidence is exhausted.
- Do not let diagnostic-only shadow scorer flags enter formal speed candidates.
- Do not clean by reverting broad files blindly; the branch contains useful
  tooling and evidence, but it needs classification first.

