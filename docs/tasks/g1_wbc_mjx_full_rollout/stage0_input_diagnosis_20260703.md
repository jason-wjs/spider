# Stage0 Input Diagnosis 2026-07-03

## Summary

The 2026-07-03 H100 Stage0 attempt showed that the current
`wbc_results/assets/motion_data/{jump,walk}` motion files cannot be treated as
the frozen sweetpoint baseline inputs. They are valid motion files, but they do
not match the historical motion package that produced the current jump/walk
sweetpoint quality gates.

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

## Root Cause

The historical jump/walk sweetpoint evidence came from
`g1_wbc_testbed_motion_package_20260617/input_motions/{jump,walk}/motion.npz`.
Examples:

- `2026-06-17-testbed-motion-baselines-xwj/primary_metrics.csv` reports
  `jump/g1_wbc_joint_global` at score `-2.0780`, root mean `0.0425`, body mean
  `0.0530`, EE mean `0.0608`, contact mismatch `0.3550`, control delta
  `0.3372`, and joint acc `191.87`.
- `2026-06-17-testbed-motion-baselines-xwj/primary_metrics.csv` reports
  `walk/g1_wbc_joint_global` at score `-1.0360`, root mean `0.0432`, body mean
  `0.0453`, EE mean `0.0471`, contact mismatch `0.1344`, control delta
  `0.1489`, and joint acc `84.32`.
- `2026-06-23-mechanism-quality-speed-wjs` and
  `2026-06-23-fast-quality-pareto-wjs` continue to reference the same testbed
  motion package in their command text and reference metrics.

The rejected `wbc_results/assets/motion_data/jump/motion.npz` file is a much
longer motion with 12,224 frames. The historical single-motion sweetpoint
reference `homejrhangmr` has 345 frames and is a different trajectory:

| Motion file | Frames | Root range after loader | SHA256 |
| --- | ---: | --- | --- |
| rejected jump candidate | 12,224 | `[8.6073, 5.5045, 0.3546]` | `07b3b8e1bf9ba3f94dfbe552819cd792f81a06a3c4ff6e5029b55b2897b7c544` |
| rejected walk candidate | 12,332 | `[9.0360, 5.3269, 0.7610]` | `a9baaa714d61da19c6114077cf0c919c965ad6802f770cc83ed695396c4c8c9f` |
| `homejrhangmr` sweetpoint reference | 345 | `[2.3553, 2.0781, 0.4948]` | `1fa518f5b80b675e3a89ff8aad501e031b64cd7ab9c0b5a1a9e2cd36331ff829` |

The first 345 loaded frames of the rejected jump candidate differ from
`homejrhangmr` by root RMS `1.786m` and joint RMS `0.588`, so this is not a
byte-level packaging change of the same benchmark.

## Decision

Do not launch formal MJX acceptance from the rejected `wbc_results/assets`
motion candidates. A valid next Stage0 attempt must use one of these routes:

1. Restore the historical `g1_wbc_testbed_motion_package_20260617` input motions
   and run the six-row Stage0 baseline against those files.
2. Promote a new motion package only after a full six-row Stage0 run produces
   passing `baseline_envelopes` and `promoted_seeds`.
3. Use the `homejrhangmr` single-motion sweetpoint only for a separate
   single-motion milestone, not as proof for the jump/walk Stage0 gate.

The quality gate remains unchanged; the rejected inputs failed the gate rather
than proving the gate too strict.
