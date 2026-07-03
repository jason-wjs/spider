# Stage0 Reproduction Diagnosis 2026-07-03

## Summary

The 2026-07-03 H100 Stage0 attempt failed to reproduce the historical
MuJoCo-Warp sweetpoint quality on `jump`, but the attempted motion files are
not rejected Stage0 inputs. They are the official versioned copies of the
historical testbed motion package inputs according to
`/data_team/junsong/model-based/wbc_results/assets/motion_data/README.md`.

This diagnosis is therefore a current-code reproduction failure record. It is
not proof that `wbc_results/assets/motion_data/{jump,walk}/motion.npz` are the
wrong motions.

The run was stopped after `jump/seed_0` and `jump/seed_1` because the jump group
was already mathematically unable to pass Stage0 eligibility. Continuing to
`jump/seed_2` and `walk` would have consumed several more H100 hours without
producing a valid `baseline_manifest.json`.

## Attempted Command

Output root:

`/data_team/junsong/model-based/g1_wbc_mjx_runs/stage0_baseline_wbc_mlp_20260703_h100_gpu1`

Inputs:

| Input | Path | SHA256 |
| --- | --- | --- |
| jump motion | `/data_team/junsong/model-based/wbc_results/assets/motion_data/jump/motion.npz` | `07b3b8e1bf9ba3f94dfbe552819cd792f81a06a3c4ff6e5029b55b2897b7c544` |
| walk motion | `/data_team/junsong/model-based/wbc_results/assets/motion_data/walk/motion.npz` | `a9baaa714d61da19c6114077cf0c919c965ad6802f770cc83ed695396c4c8c9f` |
| checkpoint | `/data_team/junsong/model-based/wbc_results/assets/checkpoints/model_8000.pt` | `98738b9214d12146dc7f4669cb65dfde9d835f4a133e5f2cbaef4e60b1e5b88f` |
| reward weights | `/data_team/junsong/model-based/wbc_results/g1_body_tracking_wbc/spider/2026-06-23-mechanism-quality-speed-wjs/configs/g1_wbc_reward_weights_method_specific_v14_20260612.json` | `bb0490a71a27a29480f13ce77bc00c13a900845f0f52641b5ec20d51ebaa535d` |

The command used `CUDA_VISIBLE_DEVICES=1` and `--device cuda:0`; completed rows
recorded `runtime_visible_devices=["1"]` and `NVIDIA H100 80GB HBM3`.

## Completed Evidence

| Motion | Seed | Steps | Accepted windows | Fallback | Score | Root mean | Body mean | EE mean | Contact mismatch | Control delta | Joint acc | Steady-state sec |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| jump | 0 | 800 | 40 | false | -4.1838 | 0.5674 | 0.5718 | 0.5779 | 0.2606 | 0.4157 | 204.94 | 654.34 |
| jump | 1 | 800 | 40 | false | -2.1596 | 0.1336 | 0.1386 | 0.1437 | 0.2531 | 0.4272 | 214.11 | 661.01 |

Jump Stage0 caps are score mean `>= -2.12`, root mean `<= 0.065`, body mean
`<= 0.075`, and EE mean `<= 0.085`. After the two completed rows, even a
perfect `jump/seed_2` would leave root/body/EE means above the caps. The run was
therefore stopped before generating a full formal manifest.

## Corrected Input Provenance

`wbc_results/assets/motion_data/README.md` records these motion assets as the
migrated shared testbed inputs:

| Motion | Asset SHA256 | Recorded source |
| --- | --- | --- |
| jump | `07b3b8e1bf9ba3f94dfbe552819cd792f81a06a3c4ff6e5029b55b2897b7c544` | `../g1_wbc_testbed_motion_package_20260617/input_motions/jump/motion.npz` |
| walk | `a9baaa714d61da19c6114077cf0c919c965ad6802f770cc83ed695396c4c8c9f` | `../g1_wbc_testbed_motion_package_20260617/input_motions/walk/motion.npz` |

The earlier conclusion that these hashes were known-bad non-sweetpoint inputs
was incorrect. The Stage0 runner must not reject them by hash.

## Historical Sweetpoint Evidence

The historical jump/walk sweetpoint evidence came from the same testbed motion
package lineage:

- `2026-06-17-testbed-motion-baselines-xwj/primary_metrics.csv` reports
  `jump/g1_wbc_joint_global` at score `-2.0780`, root mean `0.0425`, body mean
  `0.0530`, EE mean `0.0608`, contact mismatch `0.3550`, control delta
  `0.3372`, and joint acc `191.87`.
- `2026-06-17-testbed-motion-baselines-xwj/primary_metrics.csv` reports
  `walk/g1_wbc_joint_global` at score `-1.0360`, root mean `0.0432`, body mean
  `0.0453`, EE mean `0.0471`, contact mismatch `0.1344`, control delta
  `0.1489`, and joint acc `84.32`.
- `2026-06-23-mechanism-quality-speed-wjs` repeats the sweetpoint family with
  explicit `aggressive`, knot, guided-candidate, and contact-buffer parameters
  in command artifacts.

The historical single-motion `homejrhangmr` sweetpoint reference is a different
trajectory and remains useful only for a separate single-motion milestone.

## Current Root Cause

The strongest current-code difference is optimizer routing. The failed
2026-07-03 attempt ran `--mpc-backend mujoco_warp`, but the current
`evaluate.py` MuJoCo-Warp path was using SPIDER's generic sampled MPC optimizer
metadata (`spider.optimizers.sampling`, `control_update_mode=weighted_mean`).
The historical sweetpoint artifacts use the legacy `G1WbcMpcConfig` optimizer
semantics, including guided-candidate insertion, acceptance-gate fallback,
command regularization, per-window zero-delta comparisons, and legacy
`mpc_*` metadata.

The historical flags were accepted by the CLI, but before this repair many of
them did not affect the generic optimizer path. Making those flags explicit in
the runner was necessary for auditability but not sufficient for quality
reproduction.

The Stage0 runner now emits `--mpc-optimizer legacy`, and `evaluate.py` routes
that mode to `spider.tasks.g1_wbc.mpc.optimize_mpc_command()`. The runner also
defaults to the versioned asset checkpoint instead of the worktree-dependent
`bc` alias.

## Post-Repair Smoke Evidence

A 40-step single-GPU H100 smoke run on `jump/seed_1` after the legacy optimizer
repair succeeded:

`/data_team/junsong/model-based/g1_wbc_mjx_runs/stage0_legacy_smoke_20260703_jump_seed1_40`

| Steps | Optimizer | Root mean | Body mean | EE mean | Contact mismatch | Control delta | GPU visibility | GPU |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 40 | `legacy` | `0.0126` | `0.0161` | `0.0188` | `0.0000` | `0.1027` | `["1"]` | `NVIDIA H100 80GB HBM3` |

The smoke also wrote `metrics.json`, `rollout.npz`, and `mpc_command.npz` with
consistent 41-frame command/rollout qpos arrays. This is not a formal Stage0
baseline because it used only 40 steps and one seed, but it supports the
optimizer-routing root cause and justifies rerunning the full six-row Stage0
baseline through the corrected runner.

## Remaining Reproduction Hypotheses

If the corrected legacy Stage0 path still fails, investigate:

- MuJoCo/MuJoCo-Warp model, XML, contact, or loader drift.
- Evaluation code drift after the historical `wbc_results` runs were produced.
- Environment or dependency drift on the current H100 machine.

A new full Stage0 manifest is still required before MJX formal acceptance can
launch.

## Decision

Do not launch formal MJX acceptance until a six-row Stage0 MuJoCo-Warp run on
the official `wbc_results/assets` jump/walk motions produces passing
`baseline_envelopes` and `promoted_seeds`.

If the corrected Stage0 runner still fails, continue diagnosis by comparing
current `evaluate.py` inputs and runtime artifacts against the historical
command artifacts, checkpoint payload, XML/model assets, and archived metrics.
