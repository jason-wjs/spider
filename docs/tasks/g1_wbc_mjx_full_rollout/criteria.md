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

Current 2026-07-06 RTX 4090 execution note: the active local milestone is a
quality-recovery sub-goal at `>=11x` same-seed Stage0 speedup, while preserving
all frozen quality, replay, provenance, single-GPU, and trace-source gates. This
sub-goal is useful migration evidence, but it does not change the formal
`>=12x` milestone or the final RTX 4090 real-time acceptance target below.

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
Shared sampling knobs that define the frozen search surface, including
`--mpc-samples`, `--mpc-iterations`, `--mpc-knot-count`,
`--mpc-elite-frac`, and `--mpc-temperature`, must remain part of the MJX
planned command unless a replacement is explicitly documented and tested.
The MJX generic acceptance command must also pin the control-noise profile to
`--mpc-first-ctrl-noise-scale 1.0 --mpc-last-ctrl-noise-scale 1.0
--mpc-final-noise-scale 1.0`. This is not a quality relaxation: the 2026-07-06
RTX 4090 audit showed the generic default `final_noise_scale=0.1` changes the
search surface enough to make `jump/seed_0` regress from near-cap quality to a
large global-tracking failure.
The Stage0 legacy `--mpc-sigma-decay 0.75` setting is part of the frozen search
surface and must remain in the formal MJX planned command. Earlier 2026-07-05
RTX 4090 adaptive-sigma probes failed when combined with the wrong generic
noise profile, but a 2026-07-06 RTX 4090 single-variable retest on the current
flat-noise, no-warm-start, no-guided MJX-Warp surface passed the `jump/seed_0`
root/body/EE quality caps. Formal MJX metrics must record `sigma_decay=0.75`
and provenance checks must reject reused metrics whose recorded value differs
from the planned argv.

The MJX implementation must be explicit in the planned command and recorded
metrics. `--mjx-impl jax` and `--mjx-impl warp` are different benchmark
surfaces: they must not be mixed in one acceptance row or reused across each
other. If `--mjx-impl warp` is selected, the MJX-Warp contact/constraint
allocation values (`--mjx-warp-naconmax` and `--mjx-warp-njmax`) are part of
the command provenance and must be recorded in row metrics.
MJX model solver overrides (`--mjx-model-iterations` and
`--mjx-model-ls-iterations`) are diagnostic by default. They are not formal
acceptance evidence until this criteria file promotes exact values into the
frozen surface. Once promoted, the exact values must appear in the planned MJX
command, `metrics.mpc.mjx_model_options`, row sidecars, and acceptance report
provenance; missing, extra, or mismatched values must fail closed. Replay rows
must not receive these MJX-only flags.

## Hardware Rules

- One motion inference command may use only one visible GPU.
- Use `CUDA_VISIBLE_DEVICES=<one id>` and `--device cuda:0` for formal CUDA
  runs.
- The single visible GPU for formal Stage0 and acceptance timing must have no
  pre-existing compute processes. The Stage0 runner fails fast on GPU
  contention because a busy H100 would invalidate denominator timing.
- The historical H100 optimization server may have four H100 GPUs, but one
  motion inference command must never use multiple GPUs.
- The current RTX 4090 workstation is expected to run with exactly one visible
  GPU (`CUDA_VISIBLE_DEVICES=0`) for Stage0, MJX, and replay evidence.
- Do not run two MJX/JAX motion inference commands concurrently on the same
  visible GPU during smoke or formal acceptance. Each command can still satisfy
  the one-GPU rule, but concurrent JAX warmup on one H100 can exhaust memory or
  fail cuSolver handle creation. Run such commands serially on one GPU, or give
  independent commands different `CUDA_VISIBLE_DEVICES` values.
- H100 reports must include runtime GPU names for Stage0 baseline, MJX, and
  replay rows when applicable.
- The first milestone may require `H100` in the runtime GPU name.
- The final real-time target must be measured on `RTX 4090` for Stage0
  baseline, MJX, and replay rows, and must not be inferred from H100 timing.

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
present, report `800` steps, `40` accepted windows, no baseline fallback, carry
the required `mpc_command.npz` window-command schema below, and match the
current command provenance for motion, checkpoint, method, backend, optimizer,
and effective `mpc.config`. Incomplete, provenance-mismatched, or
stitched-command-only rows must be rerun.

Historical sweetpoint evidence available in `wbc_results` comes from the
versioned testbed motion set now stored under
`wbc_results/assets/motion_data/{jump,walk}/motion.npz` and from the
single-motion `homejrhangmr` run recorded under
`2026-06-26-homejrhangmr-three-way-wjs`. These motion assets are the official
migrated copies of the historical testbed package inputs, but a file path and
hash still are not enough to accept Stage0; the current code must reproduce the
six-row MuJoCo-Warp sweetpoint manifest and pass the gates below.

Historical H100/server input status in the 2026-07-03 worktree:

| Input | Path | SHA256 | Status |
| --- | --- | --- | --- |
| WBC MLP checkpoint | `/data_team/junsong/model-based/wbc_results/assets/checkpoints/WXY_8000/model_8000.pt` | `98738b9214d12146dc7f4669cb65dfde9d835f4a133e5f2cbaef4e60b1e5b88f` | accepted checkpoint input |
| reward weights | `/data_team/junsong/model-based/wbc_results/g1_body_tracking_wbc/spider/2026-06-23-mechanism-quality-speed-wjs/configs/g1_wbc_reward_weights_method_specific_v14_20260612.json` | `bb0490a71a27a29480f13ce77bc00c13a900845f0f52641b5ec20d51ebaa535d` | accepted reward input |
| jump motion | `/data_team/junsong/model-based/wbc_results/assets/motion_data/jump/motion.npz` | `07b3b8e1bf9ba3f94dfbe552819cd792f81a06a3c4ff6e5029b55b2897b7c544` | official migrated testbed input; current-code Stage0 reproduction still must pass |
| walk motion | `/data_team/junsong/model-based/wbc_results/assets/motion_data/walk/motion.npz` | `a9baaa714d61da19c6114077cf0c919c965ad6802f770cc83ed695396c4c8c9f` | official migrated testbed input; current-code Stage0 reproduction still must pass |

Current RTX 4090 workstation input status on 2026-07-05:

| Input | Path | SHA256 | Status |
| --- | --- | --- | --- |
| WBC MLP checkpoint | `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/wbc_results/assets/checkpoints/WXY_8000/model_8000.pt` | `98738b9214d12146dc7f4669cb65dfde9d835f4a133e5f2cbaef4e60b1e5b88f` | accepted checkpoint input |
| reward weights | `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/wbc_results/g1_body_tracking_wbc/spider/2026-06-23-mechanism-quality-speed-wjs/configs/g1_wbc_reward_weights_method_specific_v14_20260612.json` | `bb0490a71a27a29480f13ce77bc00c13a900845f0f52641b5ec20d51ebaa535d` | accepted reward input |
| jump motion | `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/wbc_results/assets/motion_data/jump/motion.npz` | `07b3b8e1bf9ba3f94dfbe552819cd792f81a06a3c4ff6e5029b55b2897b7c544` | official migrated testbed input; current-code Stage0 reproduction still must pass on this machine |
| walk motion | `/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/wbc_results/assets/motion_data/walk/motion.npz` | `a9baaa714d61da19c6114077cf0c919c965ad6802f770cc83ed695396c4c8c9f` | official migrated testbed input; current-code Stage0 reproduction still must pass on this machine |

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
  --jump-motion /home/humanoid/Projects/Junsong_WU/policy-in-the-loop/wbc_results/assets/motion_data/jump/motion.npz \
  --walk-motion /home/humanoid/Projects/Junsong_WU/policy-in-the-loop/wbc_results/assets/motion_data/walk/motion.npz \
  --motion-type isaaclab \
  --checkpoint /home/humanoid/Projects/Junsong_WU/policy-in-the-loop/wbc_results/assets/checkpoints/WXY_8000/model_8000.pt \
  --reward-weights /home/humanoid/Projects/Junsong_WU/policy-in-the-loop/wbc_results/g1_body_tracking_wbc/spider/2026-06-23-mechanism-quality-speed-wjs/configs/g1_wbc_reward_weights_method_specific_v14_20260612.json \
  --output-dir /home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/stage0_dryrun_testbed_wbc_mlp_4090 \
  --device cuda:0 \
  --dry-run
```

Current RTX 4090 runtime setup status, audited 2026-07-05:

- Hardware is present outside the sandbox: `NVIDIA GeForce RTX 4090`, driver
  `580.126.09`, CUDA runtime reported by `nvidia-smi` as `13.0`.
- Dependency migration is explicit: `pyproject.toml` includes `jax[cuda13]` and
  `mujoco-mjx==3.7.0`; `uv.lock` resolves from public PyPI plus
  `https://pypi.nvidia.com/` instead of the private CodeArtifact mirror.
- `uv sync --frozen` completed in `.venv` on this workstation. The environment
  imports `torch==2.11.0+cu130`, `jax==0.10.2`, `jaxlib==0.10.2`,
  `mujoco==3.7.0`, `mujoco.mjx`, `mujoco_warp==3.7.0.1`, and `warp==1.12.1`.
- `scripts/check_g1_wbc_4090_runtime.py --required-gpu-name-fragment 4090`
  passes with `CUDA_VISIBLE_DEVICES=0` and one visible JAX GPU:
  `NVIDIA GeForce RTX 4090`.
- Use `scripts/check_g1_wbc_4090_runtime.py` after dependency setup and before
  formal Stage0/MJX commands. This preflight is not acceptance evidence; it only
  proves the single-GPU Python/JAX/MJX runtime is ready to begin experiments.

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
`num_steps + 1` frames. `ref_indices` must use the Stage0/MuJoCo-Warp replay
artifact convention for executed rollouts: the initial frame and the first
post-step frame both reference motion frame `0`, followed by `1...num_steps-1`
(`800` steps therefore writes `[0, 0, 1, ..., 799]`). A raw `0...num_steps`
sequence is a metric-reference convention mismatch and is reported as
`rollout_npz_schema`. This artifact convention is separate from the MJX
optimizer scorer's post-physics-step `score_reference` alignment, which remains
`start + 1...` inside the scorer.
Invalid rollout content is reported as `rollout_npz_schema`.

`mpc_command.npz` must be a valid NPZ containing `refined_qpos`,
`candidate_scores`, command joint/body trajectories, and command
`qpos`/`qvel` trajectories with shapes consistent with the same frame count.
Invalid command content is reported as `mpc_command_npz_schema`.
It must also contain the exact receding-horizon window commands consumed during
MPC execution:

- `window_command_schema_version`
- `window_starts`, `window_execute_steps`, and `window_horizons`
- full-window `window_command_*` joint/body fields
- `window_command_qpos_chunks` and `window_command_qvel_chunks`

The window starts and execute steps must be contiguous and cover `num_steps`.
Each command chunk stores the full original planning horizon, while
`window_execute_steps` records how many frames were physically executed from
that horizon. A stitched full-run command without these per-window chunks is
not replay-proof evidence and is reported as `mpc_command_npz_schema`.
Current code may also write stronger diagnostic replay-state provenance:
`window_replay_state_schema_version`, per-window initial `qpos`/`qvel`, optional
previous action, and observation-history buffers. These state chunks are not a
replacement for `window_command_*` command chunks; they are an explicit
closed-loop carry-state oracle. Current code can export these chunks from the
legacy MuJoCo-Warp path and from MJX windows that have live execute state, but
until a full formal MJX/replay row matrix proves parity and this criteria file
promotes the exact schema, replay-state chunks remain diagnostic/strengthening
evidence rather than a standalone formal pass condition.
Within a command NPZ, `refined_qpos` must match `command_qpos_trajectory`
within the runtime acceptance tolerance. Mismatch is reported as
`mpc_command_qpos_mismatch`.
`command_qvel_trajectory` must match the finite-difference MuJoCo qvel
trajectory derived from `refined_qpos` at the policy timestep. Mismatch is
reported as `mpc_command_qvel_mismatch`.
Stage0 and MJX MPC `rollout.npz` artifacts are dynamic execute traces. Their
`qpos` records the simulator state produced by applying the MPC controls, and
is not required to equal the command/refined qpos trajectory frame-for-frame.
Static rollout artifacts, if explicitly emitted for a diagnostic path, may opt
into a rollout-command qpos equality check; mismatch in that static diagnostic
mode is reported as `mpc_rollout_qpos_mismatch`.

For 2026-07-06 RTX 4090 local evidence, the historical full Stage0 manifest at
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/stage0_baseline_legacy_20260705_4090_gpu0/baseline_manifest.json`
has a complete six-row matrix and passes formal manifest metadata validation,
but all six command NPZ artifacts fail the current artifact preflight as
`mpc_command_npz_schema` because they predate the per-window command chunks. It
is usable only as historical denominator evidence until Stage0 is rerun or the
current command schema, hashes, freshness, and provenance are regenerated. The
single-row
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/stage0_baseline_legacy_20260705_4090_windowcmd_probe`
can validate artifact shape for `jump/seed_0`, but it is not an acceptance
baseline manifest: the runner validates the complete `jump`/`walk` x seeds
`0,1,2` matrix before applying `--only-motion` / `--only-seed` filters.
The regenerated current-schema manifest at
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/stage0_baseline_current_schema_20260706_4090_gpu0/baseline_manifest.json`
passes formal manifest validation and current artifact preflight
(`artifact_preflight=()`), includes `baseline_envelopes` and `promoted_seeds`,
and is the current RTX 4090 denominator for downstream formal acceptance. Its
first full run failed the tight jump Stage0 mean gate until `jump/seed_0` was
rerun in place, so keep the accepted manifest and artifact hashes fixed when
computing speedups; do not mix rows from a later rerun without regenerating and
revalidating the full manifest.

Formal MJX rows must positively report `rollout_source=dynamic_execute_trace`,
`rollout_dynamic_execute_trace=true`, and an `execute_trace_chunks` count equal
to `accepted_windows`. Missing or static-fallback rollout source evidence is
reported as `mjx_dynamic_execute_trace`.

Formal acceptance recomputes input hashes from the manifest row argv paths.
Hashes must be lowercase SHA256 hex strings. Acceptance also recomputes baseline
artifact hashes and rejects stale or replaced baseline files.

Formal baseline manifests must declare
`contact_force_semantics=pyramidal_contact_normal_v1`. This is the canonical
force denominator after decoding pyramidal contact rows; older first-solver-row
force manifests must not be mixed with current MJX/contact-force acceptance.

`baseline_envelopes` are frozen Stage0 metric envelopes per motion. They are the
only allowed quality reference for MJX and MuJoCo-Warp replay gates. Acceptance
must reject a manifest when frozen envelopes or `promoted_seeds` are missing or
do not match the Stage0 rows.

## Per-Motion Metrics Comparison Contract

For every future single-motion quality discussion or tuning report, the primary
metrics table must use exactly these four business-comparison columns:

| Column | Required source | Metrics reference |
| --- | --- | --- |
| MJX MPC | The selected `--mpc-backend mjx` row and its `metrics.json` | original `motion.npz` |
| No-MPC original | `--method no_mpc --motion <original motion.npz>` | original `motion.npz` |
| No-MPC on MJX `mpc_motion` | `--method no_mpc --motion <MJX row>/mpc_motion.npz --motion-type mujoco --metrics-reference-motion <original motion.npz> --metrics-reference-motion-type isaaclab` | original `motion.npz` |
| MuJoCo-Warp MPC | The same-motion/same-seed Stage0 sweetpoint row, `--mpc-backend mujoco_warp --mpc-optimizer legacy` | original `motion.npz` |

The `No-MPC on MJX mpc_motion` column measures how well the bare policy tracks
the original motion when the MJX-optimized `mpc_motion.npz` is used only as the
policy input/reference. Its metrics must be computed against the original
motion, not against `mpc_motion.npz`. Valid evidence must record both top-level
`motion=<.../mpc_motion.npz>` and `metrics_reference_motion=<original
motion.npz>` in `metrics.json`.

`replay_command` metrics are not part of this primary comparison table. Replay
remains a separate backend/parity gate for formal acceptance, but it must not be
used as a replacement for MJX MPC quality or for the `No-MPC on MJX
mpc_motion` diagnostic.

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
Formal v14 reward weights must not be partially applied: any nonzero reward
term that MJX cannot score, such as local body/EE/hand terms or joint position
terms, invalidates the row until implemented. Dropping unsupported terms is a
scoring-surface change, not an optimization.
As of the 2026-07-05 RTX 4090 audit, the known formal v14 terms are expected to
be implemented in the MJX scorer; any newly introduced nonzero term must still
fail closed until implemented and tested.
As of the 2026-07-06 RTX 4090 audit, early v14 MJX-Warp 800-step
`jump/seed_0` formal-shaped probes with the wrong generic search profile did
not pass this gate. The unguided probe at
`/tmp/g1_wbc_mjx_warp_jump_seed0_formal_probe_20260706` reported
`root/body/EE=0.49479/0.50056/0.50341` and `steady_state_wall_time_sec=23.28066`;
the guided-candidate probe at
`/tmp/g1_wbc_mjx_warp_jump_seed0_guided_probe_20260706` improved to
`0.20171/0.21160/0.21667` but still failed quality and slowed to `27.62754s`.
The later flat-noise, no-warm-start, no-guided adaptive-sigma retest at
`/tmp/g1_wbc_mjx_warp_jump_seed0_flat_nowarm_adaptive075_from_runner_20260706`
passed the single `jump/seed_0` root/body/EE quality caps, but it is still not
acceptance evidence until the full matrix, replay, speed, and real-time gates
pass.
Later 2026-07-06 solver-option diagnostics showed that
`mjx_model.opt.iterations=4` preserves single-row `jump/seed_0` quality and
improves steady time to roughly `19.57s`; `iterations=3` and `2` were faster
but failed root/body/EE quality. Lower `ls_iterations` at `iterations=4`
improved timing only slightly (`18.95s` at `ls_iterations=5` after adaptive
optimizer dead-work removal) and remains below the `>=12x` gate. These rows are
diagnostic until exact solver options are promoted and the full acceptance
matrix passes.

A 2026-07-06 selected-prefix trace reuse diagnostic at
`/tmp/g1_wbc_mjx_warp_jump_seed0_adaptive075_model_iter4_ls5_prefixreuse_20260706`
is explicitly rejected. It removed separate execute-tracer wall time by using
`execute_trace_source=optimizer_selected_prefix`, but the scorer's batched
MJX-Warp `mjx_data` was not preserved; reconstructing state from selected
qpos/qvel regressed `jump/seed_0` to
`root/body/EE=0.09310/0.10061/0.10612` and `steady_state_wall_time_sec=20.32592`.
This is neither speed nor quality evidence. The default MJX-Warp path must keep
the exact separate `rollout_tracer` unless an exact selected-world MJX-Warp data
extraction method is proven in a full 800-step row. A later helper-level fix
can preserve selected `mjx_data` by slicing world-batched leaves while keeping
MJX-Warp flat global `contact__*` buffers intact, but that helper test and a
one-step 4090 introspection are not acceptance evidence. For MJX-Warp rows, the
acceptance runner must reject `execute_trace_source_counts` containing any
nonzero source other than `rollout_tracer` until this paragraph is explicitly
revised with full quality/replay/speed evidence. Formal MJX-Warp acceptance
therefore requires `execute_trace_source_counts` to be exactly
`{"rollout_tracer": accepted_windows}`; any nonzero `optimizer_selected_prefix`
source remains diagnostic-only.
Two later outer-JIT selected-prefix diagnostics did preserve single-row
`jump/seed_0` quality and crossed the same-seed `>=12x` speed line:
`/tmp/g1_wbc_mjx_warp_jump_seed0_outerjit_prefix20_800_20260706`
reported `root/body/EE=0.07048/0.08025/0.08824` and `16.38116s`, while
`/tmp/g1_wbc_mjx_warp_jump_seed0_outerjit_prefix20_800_r2_20260706`
reported `0.07075/0.08136/0.08970` and `16.40470s`. Against the regenerated
current-schema `jump/seed_0` denominator (`206.82522s`), these are about
`12.63x` and `12.61x`. These results are useful optimization evidence only.
They are not formal acceptance evidence under the current runner because both
rows report
`execute_trace_source_counts={"optimizer_selected_prefix": 40}`, and they still
miss the RTX 4090 realtime factor of `1.0`.
MJX metrics provenance must also match the effective search surface planned by
the runner: sample count, optimizer iterations, horizon/control/knot counts,
elite fraction, temperature, root/rotation/joint sigmas, first/last/final noise
scales, sigma decay, warm-start, guided-candidate, solver options, and MJX-Warp
buffer sizes. Missing or mismatched fields are a metrics-provenance failure; a
short real 4090 smoke confirmed the current backend writes those fields at
`/tmp/g1_wbc_mjx_warp_jump_seed0_metadata_smoke_searchprov_20260706`.

A 2026-07-06 score-only optimizer diagnostic at
`/tmp/g1_wbc_mjx_warp_jump_seed0_adaptive075_model_iter4_ls5_scoreonly_20260706`
is explicitly rejected. It skipped per-term metric accumulation inside the
optimizer scorer, but still reported only about `10.97x` speedup over the local
`jump/seed_0` Stage0 denominator and regressed quality to
`root/body/EE=0.11989/0.12957/0.13739`. Formal MJX rows must keep the full
scorer metrics enabled unless a later exact-equivalence proof passes the same
quality, replay, speed, and provenance gates.
A later explicit diagnostic flag, `--mjx-score-only-optimizer`, produced the
same conclusion on the post-step-reference surface:
`/tmp/g1_wbc_mjx_poststep_ref_jump_seed0_model_iter4_ls5_scoreonlyopt_20260706`
reported only `17.77378s` (`11.637x`) and regressed `jump/seed_0` to
`root/body/EE=0.08406/0.09470/0.10220`, with `ee_local=0.04821` and
`score=-1.98727`. Formal rows must leave `score_only_optimizer=false` unless
this criteria file is revised after an exact-equivalence proof.
Two follow-up full-metrics default reruns on the same surface were mixed:
`/tmp/g1_wbc_mjx_warp_jump_seed0_adaptive075_model_iter4_ls5_post_scoreonly_default_20260706`
missed the root cap by `0.00078`, while
`/tmp/g1_wbc_mjx_warp_jump_seed0_adaptive075_model_iter4_ls5_post_scoreonly_default_r2_20260706`
passed `jump/seed_0` root/body/EE quality at `0.07180/0.08063/0.08683`. This
fragile single-row margin is not formal acceptance evidence.
The 2026-07-06 final CEM distribution dead-work cleanup at
`/tmp/g1_wbc_mjx_warp_jump_seed0_adaptive075_model_iter4_ls5_finaldist_noguided_20260706`
is quality-preserving for the no-guided/no-warm diagnostic surface
(`root/body/EE=0.06274/0.07220/0.07921`, `rollout_tracer` source count `40`),
but the steady-state time is still about `18.88s`, below neither the
`>=12x` milestone nor the realtime target.
A 2026-07-06 per-iteration diagnostic at
`/tmp/g1_wbc_mjx_warp_jump_seed0_iterdiag_800_20260706` passed single-row
quality (`root/body/EE=0.05417/0.06564/0.07428`) but remained at `18.91s`.
Its window diagnostics showed `iteration_accepted_window_counts=[20,36]` and
`iteration_noop_window_counts=[20,4]`; `16` first-iteration no-op/current
windows were improved by the second iteration. Therefore naive early-stop after
the first current/no-op selection is an explicit quality risk, not a valid
speed shortcut.
The optimizer host-sync cleanup reduced Python boolean conversion on JAX
finite-score checks and cross-iteration best selection, while preserving a
fail-closed `scores_finite` flag. It produced one quality-failing repeat at
`/tmp/g1_wbc_mjx_warp_jump_seed0_hostsync_800_20260706`
(`root/body/EE=0.07981/0.08717/0.09109`, `18.69s`) and one quality-passing
repeat at `/tmp/g1_wbc_mjx_warp_jump_seed0_hostsync_800_r2_20260706`
(`root/body/EE=0.05562/0.06506/0.07167`, `18.77s`). This cleanup is useful
engineering, but it still leaves roughly `1.5s` of same-seed timing gap to the
regenerated `>=12x` target time of about `17.24s`.
The follow-up outer-JIT default optimizer path preserves the formal
`rollout_tracer` source and same search surface. Repeat 1 at
`/tmp/g1_wbc_mjx_warp_jump_seed0_outerjit_direct_800_20260706` ran in
`17.84790s` but missed quality (`root/body/EE=0.08012/0.08971/0.09677`).
Repeat 2 at `/tmp/g1_wbc_mjx_warp_jump_seed0_outerjit_direct_800_r2_20260706`
passed single-row quality (`0.05729/0.06858/0.07800`) with
`execute_trace_source_counts={"rollout_tracer": 40}` and ran in `17.84408s`.
That is about `11.59x` over the regenerated local `jump/seed_0` denominator
`206.82522s` and remains short of both `>=12x` and realtime.
The later timing-attribution cleanup records
`optimizer_result_sync_wall_time_sec` separately. A short 80-step 4090 smoke at
`/tmp/g1_wbc_mjx_sync_timing_smoke_20260706` measured `1.79715s` steady time,
of which `0.50840s` was optimizer-result synchronization after enqueue. The
800-step post-step-reference shard above measured `4.82887s` of that sync time
inside the `17.82799s` steady window. This timing is required compute/sync
tail, not compile or warmup; it should guide optimizer fusion work rather than
be removed from the denominator.
Two acceptance-runner repeats against the regenerated current-schema baseline
confirm the same formal-path block. The first at
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/mjx_acceptance_current_baseline_jump_seed0_20260706`
reported `root/body/EE=0.11823/0.12645/0.13282`, `17.84580s`,
same-seed speedup `11.5896x`, realtime factor `0.8966`, and replay
`root/body/EE=0.44779/0.44991/0.45431`. The second at
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/mjx_acceptance_current_baseline_jump_seed0_r2_20260706`
reported `0.07099/0.08068/0.08950`, `17.81840s`, same-seed speedup
`11.6074x`, realtime factor `0.8979`, and replay
`0.62777/0.63097/0.63036`. Both rows used
`execute_trace_source_counts={"rollout_tracer": 40}`. The second row is close
to the absolute caps but still fails the regenerated baseline `1.10x` ratio
quality gate, and both rows fail speed, realtime, and replay gates.
The later deferred refined-qpos artifact diagnostic at
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/mjx_acceptance_current_baseline_jump_seed0_deferred_qpos_20260706`
kept the formal `{"rollout_tracer": 40}` source, improved steady time only to
`17.75836s`, and still reached only about `11.65x` same-seed speedup. A
final-only state-advancer diagnostic at
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/mjx_acceptance_current_baseline_jump_seed0_finalonly_trace_20260706`
also kept formal source/count and reported `deferred_execute_trace_chunks=40`,
but steady time remained `17.76596s` while full trace materialization cost
`4.85s` outside steady. This is rejected as a default formal path: the required
closed-loop state advancement, not trace payload materialization alone, appears
to dominate the remaining gap.
A follow-up low-risk cleanup deferred JAX `execute_chunk` to Torch
materialization from the steady loop into post-steady artifact generation. The
single-row 4090 repeat at
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/mjx_jump_seed0_delayed_chunk_20260706`
kept `execute_trace_source_counts={"rollout_tracer": 40}` but reported
`root/body/EE=0.10138/0.11030/0.11844` and `17.83s`, so it did not provide
speed or quality evidence.
After the replay-state schema was added to MJX artifacts, a strict default-model
acceptance-runner shard at
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/mjx_acceptance_replaystate_jump_seed0_20260706`
exported `40/40` replay-state chunks and replay consumed all of them, but the
MJX row still failed quality and speed:
`root/body/EE=0.08148/0.09141/0.09876`, `ee_local=0.04676`,
`score=-2.01364`, `contact_mismatch=0.26812`, `22.77009s`,
same-seed speedup `9.083x`, realtime factor `0.703`. The replay row improved
to `0.07540/0.08379/0.09130` with
`saved_command_replay_state_source=window_replay_state_chunks`, but it still
missed the tight jump root cap and is not a pass.
A matched diagnostic solver-option shard at
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/mjx_acceptance_replaystate_jump_seed0_model_iter4_ls5_20260706`
also exported and replay-loaded `40/40` state chunks. It restored current-best
timing (`17.80671s`, same-seed speedup `11.615x`, realtime factor `0.899`) and
passed root/body/global-EE caps (`0.06172/0.07086/0.07909`), with replay at
`0.05730/0.06534/0.07371`. It still fails the full criteria surface:
diagnostic solver options are not promoted formal settings, `score=-1.98096`
is below the Stage0 jump envelope min `-1.91816`, `ee_local=0.04798` exceeds the
Stage0 max `0.04407`, `contact_mismatch=0.28000` exceeds the Stage0 max
`0.26625`, and speed remains below `>=12x`.
The follow-up scorer-reference audit found a real timing bug in the MJX
rollout reference: `_score_rollout_step` advances physics before scoring, but
`build_mjx_rollout_reference` fed `score_reference` from the pre-step frame.
`score_reference` now uses the post-step motion window (`start + 1 ...`) while
the command/observation/base windows keep their original `start ...` alignment.
This is a correctness fix, not a criteria relaxation. The formal-shaped
`jump/seed_0` retest at
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/mjx_acceptance_poststep_ref_jump_seed0_model_iter4_ls5_20260706`
reported MJX `root/body/EE=0.05229/0.06248/0.07094`,
`ee_local=0.04699`, `score=-1.99532`, `contact_mismatch=0.29063`,
`steady_state_wall_time_sec=17.82799`, same-seed speedup `11.601x`, and
realtime factor `0.897`. MuJoCo-Warp replay consumed `40/40` replay-state
chunks and reported `0.04882/0.05859/0.06713`, `ee_local=0.04659`,
`score=-1.97099`, and `contact_mismatch=0.27875`. This row remains a failing
single-shard diagnostic: the full matrix is missing, solver options are still
diagnostic, score/contact are not closed, and speed/realtime remain below
target. The MJX `ee_local` value narrowly passes the current runner's `1.10x`
gate for this row; local-EE remains a fragile margin, not the coded failure for
this particular report. This row predates the saved-rollout `ref_indices` schema
fix described above; its timing remains useful, but its saved-rollout quality
metrics must be regenerated before any fresh acceptance decision.
The faster `mjx_model.opt.iterations=3` post-step-reference diagnostic at
`/tmp/g1_wbc_mjx_poststep_ref_jump_seed0_model_iter3_ls5_20260706` crossed the
same-seed `12x` line (`16.77558s`, `12.329x`), but quality regressed to
`root/body/EE=0.07370/0.08386/0.09186`, `ee_local=0.04996`, and
`score=-2.01596`; it also remained below realtime. Increasing line-search to
`10` at
`/tmp/g1_wbc_mjx_poststep_ref_jump_seed0_model_iter3_ls10_20260706` did not
recover quality (`root/body/EE=0.07504/0.08571/0.09265`,
`score=-2.07638`). These rows prove speed-only feasibility, not a valid
quality-preserving milestone.
MJX-Warp buffer-size diagnostics on the same post-step solver surface are useful
timing evidence, but they also predate the saved-rollout `ref_indices` schema
fix and must be rerun before making fresh quality claims. `naconmax=20000` at
`/tmp/g1_wbc_mjx_poststep_ref_jump_seed0_model_iter4_ls5_nacon20000_20260706`
ran at `17.46575s` (`11.842x`) with no reported saturation but still failed
score and narrowly failed contact. `naconmax=16000` at
`/tmp/g1_wbc_mjx_poststep_ref_jump_seed0_model_iter4_ls5_nacon16000_20260706`
is the best safe buffer-size probe so far (`17.37317s`, `11.905x`, contact
passes, no reported saturation), but it still fails speed, realtime, score, and
local-EE. `naconmax=14336` was similar (`17.36892s`, `11.908x`) and still
failed score/local-EE. `naconmax=12000` is rejected even though it reached
`17.27175s`: stdout emitted MJX-Warp narrowphase overflow warnings, and the row
also failed contact/score. Do not use this pre-ref-index-fix family as quality
evidence.

The first post-ref-index-fix rerun at
`/tmp/g1_wbc_mjx_post_refindexfix_nacon16000_jump_seed0_20260706` confirmed the
artifact fix (`rollout.npz/ref_indices` head `[0,0,1,2,3,4,5,6]`, tail
`[792,...,799]`) and materially changed quality: `score=-1.93704` and
`contact_mismatch=0.25125` now pass the current `jump` envelope. It still fails
the formal gates with `root/body/EE=0.07720/0.08822/0.09656`,
`ee_local=0.04864`, `steady_state_wall_time_sec=17.36303`, same-seed speedup
`11.912x`, and realtime factor `0.9215`. Replay consumed `40/40` state chunks
and reported `root/body/EE=0.07710/0.08726/0.09582`, `ee_local=0.04850`,
`score=-2.00024`, and `contact_mismatch=0.25813`.

The follow-up post-ref-index-fix `jump/seed_0` `naconmax=18000` row at
`/tmp/g1_wbc_mjx_post_refindexfix_nacon18000_jump_seed0_20260706` ran at
`17.48163s` (`11.831x`) with no reported saturation and formal
`rollout_tracer=40`, and it passed MJX root/body/EE, score, and contact. It is
still rejected as a quality pass because local EE was `0.04927`, and replay
also missed local EE and score. The `naconmax=20000` rerun at
`/tmp/g1_wbc_mjx_post_refindexfix_nacon20000_jump_seed0_20260706` is the current
`jump/seed_0` `>=11x` milestone row: `17.48371s`, `11.830x`, realtime factor
`0.9151`, no reported saturation, `rollout_tracer=40`, MJX
`root/body/EE/local=0.05558/0.06705/0.07502/0.04610`, `score=-1.88914`,
`contact=0.26125`, and MuJoCo-Warp replay also passes the frozen quality
envelope. This is valid `>=11x` quality-recovery evidence, not a formal
`>=12x` or real-time pass.

The current 2026-07-06 RTX 4090 `>=11x` quality-recovery matrix is:

| row | path | steady / speedup | MJX quality | replay quality |
| --- | --- | ---: | --- | --- |
| `jump/seed_0`, `naconmax=16000` | `/tmp/g1_wbc_mjx_post_refindexfix_nacon16000_jump_seed0_20260706` | `17.36303s / 11.912x` | fails root/body/EE/local-EE | fails root/body/EE/local-EE/score |
| `jump/seed_0`, `naconmax=18000` | `/tmp/g1_wbc_mjx_post_refindexfix_nacon18000_jump_seed0_20260706` | `17.48163s / 11.831x` | fails local EE only | fails local EE and score |
| `jump/seed_0`, `naconmax=20000` | `/tmp/g1_wbc_mjx_post_refindexfix_nacon20000_jump_seed0_20260706` | `17.48371s / 11.830x` | pass | pass |
| `walk/seed_1`, `naconmax=20000` | `/tmp/g1_wbc_mjx_milestone_nacon20000_walk_seed1_20260706` | `17.53605s / 11.770x` | fails local EE only | fails local EE only |
| `walk/seed_1`, `naconmax=30000` | `/tmp/g1_wbc_mjx_milestone_nacon30000_walk_seed1_20260706` | `17.86752s / 11.551x` | pass | pass |
| `jump/seed_1`, `naconmax=20000` | `/tmp/g1_wbc_mjx_milestone_nacon20000_jump_seed1_20260706` | `17.47431s / 11.758x` | fails local EE only | fails local EE only |
| `jump/seed_1`, `naconmax=30000` | `/tmp/g1_wbc_mjx_milestone_nacon30000_jump_seed1_20260706` | `17.75360s / 11.573x` | fails root/body/EE and score | fails root/body/EE and score |
| `jump/seed_2`, `naconmax=20000` | `/tmp/g1_wbc_mjx_milestone_nacon20000_jump_seed2_20260706` | `17.46997s / 11.794x` | fails local EE and score | fails local EE |

All rows in this matrix used the local current-schema Stage0 manifest at
`/home/humanoid/Projects/Junsong_WU/policy-in-the-loop/g1_wbc_mjx_runs/stage0_baseline_current_schema_20260706_4090_gpu0/baseline_manifest.json`,
one visible RTX 4090, `--mjx-impl warp`, `--mjx-model-iterations 4`,
`--mjx-model-ls-iterations 5`, no guided candidate, and no selected-prefix,
score-only, reduced-sample, reduced-iteration, reduced-horizon, or overflow
shortcut as evidence. Solver options remain an evidence row parameter, not a
formal acceptance default, unless this criteria document later promotes exact
values.

The follow-up unbatched `score_reference` scorer-slice diagnostic at
`/tmp/g1_wbc_mjx_warp_jump_seed0_adaptive075_model_iter4_ls5_unbatched_score_ref_20260706`
is explicitly rejected and reverted: it kept steady-state timing near
`18.88s` and regressed `jump/seed_0` quality to
`root/body/EE=0.07709/0.08619/0.09291`, crossing the root cap.
Another 2026-07-06 diagnostic at
`/tmp/g1_wbc_mjx_warp_jump_seed0_adaptive075_model_iter4_ls5_fast_no_bodylin_20260706`
removed the scorer's `body_lin_vel_w` output, which is not directly consumed by
the actor observation or scorer. It is still explicitly rejected: timing stayed
near `18.94s` and quality regressed to `root/body/EE=0.13205/0.13992/0.14878`.
The default physics output structure must remain intact unless a later
equivalence proof passes the frozen gates.
Lowering line search further is also not a shortcut: the diagnostic
`/tmp/g1_wbc_mjx_warp_jump_seed0_adaptive075_model_iter4_ls1_20260706` improved
steady time to `18.39846s`, but quality regressed to
`root/body/EE=0.11431/0.12496/0.13090` and speedup was still only about `11.20x`.

An MJX window may count as accepted when the current controls are already the
best candidate with non-regressing score. This is a no-op MPC acceptance, not a
baseline fallback and not proof of speed or quality by itself. The dynamic
execute trace, replay, quality, speed, and real-time gates remain decisive.

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

`contact_force_active_mean` and `contact_force_peak` are report-only force
semantics diagnostics, not frozen MJX quality gates. If raw MJX force is high
but MuJoCo-Warp replay force remains near the same-seed baseline and all contact
quality, replay quality, saturation, provenance, and speed gates pass, classify
the row as `isolated_raw_mjx_force_high`. This is a diagnostic warning, not a
physical quality failure.

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
- report `saved_command_replay_source=window_command_chunks`; stitched-command
  replay is reported as `replay_window_command_source`
- use the planned replay control steps
- report `num_command_frames >= num_replay_steps + 1`
- match `num_command_frames` to the saved command NPZ frame count, reported as
  `replay_command_npz_frames`
- require replay `rollout.npz["ref_indices"]` to stay within the saved command
  frame range, reported as `replay_rollout_ref_indices`
- compare replay quality against the same frozen baseline envelope

The `isolated_raw_mjx_force_high` classification does not weaken replay quality
or replay provenance gates. `shared_force_high` and `replay_force_high` mean the
MuJoCo-Warp replay oracle is also off sweetpoint-scale force, and must remain a
replay/contact parity warning or blocker until explained or explicitly promoted
by this criteria document.

Current 2026-07-06 RTX 4090 diagnostics show that even the new-schema Stage0
`jump/seed_0` saved command does not self-replay inside the frozen quality
envelope. The best window-command replay probe was
`/tmp/g1_wbc_replay_matrix_no_cuda_graph_20260706` with
`root/body/EE=0.12878/0.13604/0.14346`, and a Stage0-style per-window new-env
diagnostic at `/tmp/g1_wbc_direct_chunk_replay_diag_20260706.json` still
reported `0.11848/0.12588/0.13287`. This does not weaken the replay gate; it
means the saved-command replay oracle remains a known blocker and may require
richer execution-state provenance before formal acceptance can pass.
For the current deferred-qpos MJX artifact, removing window chunks and forcing
stitched replay still failed (`/tmp/g1_wbc_replay_stitched_noqvel_20260706`,
`root/body/EE=0.69791/0.70134/0.69932`), and replaying the same stitched command
with saved qvel was worse (`/tmp/g1_wbc_replay_stitched_savedqvel_20260706`,
`0.73429/0.73951/0.74292`). Therefore the current replay blocker is not just the
window-chunk loader or missing qvel; it is a deeper MJX-command to MuJoCo-Warp
closed-loop parity problem.
Replaying the MJX dynamic rollout qpos/qvel itself as a stitched command also
failed and is not a valid formal path:
`/tmp/g1_wbc_replay_mjx_dynamic_as_command_savedqvel_20260706` reported
`root/body/EE=1.33445/1.33447/1.33269`. Dynamic execute state may be used as a
diagnostic oracle, but it must not be disguised as
`saved_command_replay_source=window_command_chunks` or as the exact selected MPC
command consumed during optimization.

The replay parity diagnostic
`scripts/diagnose_g1_wbc_mjx_replay_parity.py` can compare MJX, replay, and
command artifacts without changing any gate. On the current deferred-qpos row it
wrote `/tmp/g1_wbc_mjx_replay_parity_deferred_qpos_20260706.json`: replay and
MJX dynamic root trajectories are close early (`0.00119m` at frame `20`,
`0.00830m` at frame `160`), cross `0.1m` at frame `373`, and reach `2.559m` at
frame `800`; actions cross `0.1` norm at frame `89`, while contact
force/contact indicator differences appear much earlier. Treat this as evidence
of long-horizon closed-loop amplification of MJX-Warp vs MuJoCo-Warp
contact/velocity/policy-input differences, not a first-window qpos command shape
bug.
The Stage0 window carry-state diagnostic
`scripts/diagnose_g1_wbc_window_state_replay.py` can rerun a short legacy
MuJoCo-Warp prefix, capture per-window closed-loop state, and replay ablations
without changing the formal replay gate. On the current RTX 4090 `jump/seed_0`
Stage0 source it wrote `/tmp/g1_wbc_window_state_replay_jump0_20260706.json`.
For windows starting at `20` and `40`, `state_all` replay matched the captured
window rollout to about `1e-7m` root max and `1e-5` action norm max, while
`command_only` had immediate action drift and root max `0.03179m` / `0.14549m`.
Dropping only history stayed below `0.00039m` root max, but dropping previous
action reached about `0.0156m` root max with large action drift. This strongly
indicates that saved command chunks alone are missing closed-loop policy carry
state, especially previous action, and that richer replay provenance should be
designed explicitly rather than disguised as command trajectory data.
A follow-up 2026-07-06 RTX 4090 smoke promoted that finding into artifact
plumbing for the legacy MuJoCo-Warp path. The short two-window source run at
`/tmp/g1_wbc_replay_state_smoke_source_20260706` wrote an `mpc_command.npz`
containing `window_replay_state_schema_version=1`,
`window_replay_state_valid=[1,1]`, and
`window_replay_initial_last_action_valid=[0,1]`; replay at
`/tmp/g1_wbc_replay_state_smoke_replay_20260706` reported
`saved_command_replay_source=window_command_chunks`,
`saved_command_replay_state_source=window_replay_state_chunks`, and
`num_command_replay_state_chunks=2`. Its replay rollout matched the source qpos
with max norm `7.1e-6` over 40 steps. This is still a smoke, not full Stage0 or
MJX formal acceptance, but it proves the richer state schema is consumable by
the shared MuJoCo-Warp replay backend.
A second 2026-07-06 RTX 4090 smoke extended the same replay-state schema to the
MJX-Warp path. The short two-window source run at
`/tmp/g1_wbc_mjx_replay_state_smoke_20260706` wrote schema `1`, valid state
chunks `[1,1]`, and previous-action validity `[0,1]`; MuJoCo-Warp replay at
`/tmp/g1_wbc_mjx_replay_state_smoke_replay_20260706` reported
`saved_command_replay_source=window_command_chunks`,
`saved_command_replay_state_source=window_replay_state_chunks`, and
`num_command_replay_state_chunks=2`. This only proves MJX artifacts can now
carry and replay-load closed-loop state. It is not parity evidence: over the
20-step smoke, replay-vs-MJX qpos max norm was about `0.0186`, and contact force
diverged immediately, so the formal MJX-vs-MuJoCo-Warp replay gate remains open.
The follow-up 800-step replay-state-enabled shards above show the same pattern
at formal length. For the diagnostic solver-option row
`/tmp/g1_wbc_mjx_replay_parity_replaystate_jump0_model_iter4_ls5_20260706.json`
reported replay-vs-MJX root position mean/max `0.00267m/0.03302m`, with root
never crossing `0.1m`; this is much better than the earlier deferred-qpos row
that reached `2.559m` root divergence. However qpos still crossed `0.1` norm at
frame `372`, actions crossed `0.1` at frame `99`, controls crossed `0.1` at
frame `117`, and contact force diverged immediately. Replay-state chunks
therefore narrow the replay blocker but do not close the formal parity gate.
The diagnostic flag `--mjx-strip-live-mjx-data` strips the live MJX `Data` carry
between windows while preserving explicit qpos/qvel/action/history carry. It is
for replay-carry investigation only and must not be used for formal acceptance.
On `/tmp/g1_wbc_mjx_strip_mjx_data_smoke_20260706`, an 80-step paired smoke kept
quality nearly unchanged versus the normal path and contact indicators matched,
while qpos/qvel drift appeared later and one window incurred a large timing
tail. This lowers the priority of hidden live `mjx_data` as the primary replay
divergence cause; contact/velocity/policy-input differences and formal
full-length replay remain the blockers.

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

## RTX 4090 Speed and Real-Time Gate

The final deployment target passes only when all quality, replay, metadata, and
baseline-relative speed gates pass on RTX 4090, and:

```text
evaluated_motion_duration_sec / mjx_steady_state_wall_time_sec >= 1.0
```

Required command intent:

```bash
--target 4090_realtime --required-gpu-name-fragment 4090
```

The same `4090` runtime evidence requirement applies to the local Stage0
baseline rows, MJX rows, and replay rows. H100 baseline manifests are historical
evidence only and cannot satisfy the final 4090 target.

The `4090_realtime` target must still satisfy the same `>= 12x` mean and worst
seed steady-state speedup requirements before it can report
`pass_4090_realtime`.

Each MJX row must include:

- `control_dt_sec`
- `evaluated_motion_duration_sec`
- `mjx_impl` and `mjx_model_impl`
- if `mjx_impl=warp`, `mjx_warp_naconmax` and `mjx_warp_njmax`
- same-seed baseline steady-state time, same-seed speedup, target row time, and
  row real-time factor diagnostics in row sidecars
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

When `--skip-report` is used for sharded rows, exit code `0` means the row
evidence is structurally reusable and any deterministic row-level same-seed
speed diagnostic did not already prove final worst-speedup failure. It is not a
motion-level quality, replay-parity, speed, or real-time pass. The row sidecar
speed diagnostics and metrics must be inspected, and only the final report can
classify `pass_h100_milestone`, `pass_4090_realtime`,
`quality_regression`, `parity_failure`, or `speed_regression`.

Required top-level sections:

- baseline manifest path
- frozen `baseline_envelopes`
- planned MJX and replay runs
- baseline rows
- MJX rows
- per-motion four-way comparison rows
- replay rows
- motion quality results
- replay results
- speed results
- realtime results
- timing summary
- contact summary
- final classification

The contact summary must include active and peak force ratios plus semantic
classification counts for `isolated_raw_mjx_force_high`, `shared_force_high`,
`replay_force_high`, and `force_ratio_nominal`. These counts are required
report evidence even when force semantics do not independently fail the MJX
quality gate.

Required final classifications:

| Class | Meaning |
| --- | --- |
| `pass_h100_milestone` | H100 quality, replay, and `>= 12x` speed gates all pass. |
| `pass_4090_realtime` | RTX 4090 quality, replay, `>= 12x` speed, and real-time gates all pass. |
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
