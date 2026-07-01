# G1 WBC JAX/MJX Speed-Parity Experiment Design

## Goal

Determine whether a Dial-MPC-style JAX/JIT/vmap execution path can materially
reduce G1 WBC sampling-MPC inference time without prematurely replacing the
current MuJoCo-Warp backend.

This is a feasibility and parity experiment, not a tuning sweep. The experiment
keeps the current SPIDER sweetpoint configuration fixed and separates two
questions:

- Speed: can a JAX/JIT/vmap implementation reduce steady-state MPC window time?
- Trust: can an MJX rollout reproduce the MuJoCo-Warp G1 WBC model, policy
  rollout, and contact metrics closely enough to use it for quality decisions?

## Baseline Anchor

Use the current SPIDER sweetpoint as the frozen baseline:

```text
motions: jump, walk
method: g1_wbc_joint_global
samples: 512
iterations: 2
planning_horizon_steps: 40
control_steps: 20
knot_count: 8
temperature: 0.7
root_pos_sigma: 0.04
root_rot_sigma: 0.10
joint_sigma: 0.18
reward_weights: g1_wbc_reward_weights_method_specific_v14_20260612.json
seed: 0
```

No sample, horizon, noise, reward, or acceptance tuning belongs in this
experiment. Any quality differences should be attributable to the execution
path, model semantics, or numerical backend.

## Stage 0: Baseline Lock

Run or register the existing MuJoCo-Warp sweetpoint outputs for `jump` and
`walk` before implementing the JAX/MJX path.

Required artifacts per motion:

- `metrics.json`
- `rollout.npz`
- `mpc_command.npz`
- total wall time
- per-window timing, if available
- backend metadata, including MuJoCo, MuJoCo-Warp, Torch, checkpoint, and reward
  weight source

Primary comparison metrics:

- score
- root position and rotation error
- body global position and rotation error
- EE global and local position error
- contact mismatch rate
- non-foot floor contact rate
- contact force diagnostics
- control delta
- joint acceleration
- total wall time and steady-state window time

The Stage 0 result is the only quality anchor for later comparisons.

## Stage 1: Model And Kinematics Parity

Validate that the MJX model can represent the same G1 WBC semantics before
running policy rollouts.

Checks:

- qpos and qvel dimensions match the current G1 WBC model.
- MuJoCo joint order matches `MUJOCO_JOINT_NAMES`.
- Body order covers all bodies used by observations and metrics.
- Actuator order, default joint pose, action scale, armature, damping,
  frictionloss, force limits, and PD gain semantics are mapped intentionally.
- Foot collision geoms and non-foot robot collision geoms are classified in the
  same groups as the MuJoCo-Warp backend.
- Static forward kinematics for reference qpos frames matches MuJoCo-Warp for
  pelvis, task EEs, hands, feet, and torso.

Gate:

- If joint/body/control mapping is wrong, stop and repair the model adapter.
- If static body/EE kinematics mismatch is large enough to affect tracking
  metrics, do not proceed to rollout parity.
- If only contact force details differ, continue to Stage 2 but mark contact
  parity as high risk.

## Stage 2: Rollout And Contact Parity

Validate rollout behavior before running sampled MPC.

Run the following on both `jump` and `walk`:

1. default-pose standing rollout with fixed controls
2. short saved-command replay from the MuJoCo-Warp baseline
3. full `no_mpc` policy rollout
4. full replay of Stage 0 `mpc_command.npz`

Compare MJX against MuJoCo-Warp on:

- root/body/EE trajectory drift
- actor action trajectory
- joint position target trajectory
- left and right foot contact timing
- non-foot floor contact timing
- contact force scale and spikes
- metric deltas from `compute_rollout_metrics`
- runtime after JIT warmup

Gate:

- If body tracking and actor/control trajectories are close but contact force
  scale differs, MJX may be used as a proposal backend only.
- If contact timing differs materially on `walk`, MJX cannot be used for final
  scoring.
- If `jump` contact differs but `walk` is close, continue with a motion-specific
  risk label and do not generalize quality claims.
- If `no_mpc` policy rollout diverges before contact events, stop and repair the
  observation, actor, or actuator mapping.

## Stage 3: JAX Sweetpoint Speed Spike

Only enter this stage after Stage 1 succeeds and Stage 2 is at least usable as a
proposal backend.

Implement the smallest JAX path that can evaluate the frozen sweetpoint:

- WBC actor MLP forward with checkpoint weights converted from Torch.
- Required observation-builder terms for the WXY G1 WBC actor.
- Command residual to qpos conversion for `g1_wbc_joint_global`.
- Fixed-shape batched rollout over samples and horizon.
- Score terms required by the v14 `g1_wbc_joint_global` weights.
- Knot interpolation, sample generation, softmax update, horizon shift, and
  execute chunk behavior matching the SPIDER optimizer.

Timing rules:

- Record JIT compile time separately.
- Report steady-state per-window time.
- Report full rollout wall time excluding and including first-call compilation.
- Use the same GPU when comparing to Stage 0.
- Keep Python outside the inner rollout/update path.

Quality rules:

- Do not change sweetpoint parameters to improve MJX quality.
- Compare against Stage 0 metrics and replay artifacts.
- Report quality only after parity gates are labeled.

## Fallback Branches

### Branch A: MJX Parity Passes

Promote the JAX/MJX path from feasibility spike to backend candidate.

Next design should cover:

- registering a selectable `g1_wbc_jax_mjx` backend
- packaging model conversion
- checkpoint conversion
- repeated-seed validation
- broader bench_data transfer

### Branch B: MJX Is Fast But Contact Parity Fails

Use MJX only for candidate proposal or pruning.

Candidate hybrid flow:

1. Run JAX/MJX sampling over all samples.
2. Keep the top-K candidate command sequences by MJX score.
3. Re-score top-K plus the zero-residual baseline in MuJoCo-Warp.
4. Execute the MuJoCo-Warp winner.

This branch preserves the MuJoCo-Warp quality anchor while using JAX/MJX to
reduce the number of expensive final rollouts.

### Branch C: JAX/MJX Is Not Fast Enough Or Parity Is Poor

Stop the migration track and pursue Dial-style execution improvements inside the
current Torch/MuJoCo-Warp backend:

- static rollout buffers
- fewer Python state transitions per window
- stronger CUDA graph coverage
- persistent candidate command tensors
- asynchronous planner and action-buffer execution
- top-K or early-pruning within MuJoCo-Warp

## Success Criteria

Speed feasibility succeeds if the JAX/JIT/vmap path shows at least a 3x
steady-state per-window speedup on the frozen sweetpoint workload, excluding
first-call compilation.

Backend replacement feasibility succeeds only if Stage 1 and Stage 2 pass and
Stage 3 preserves the Stage 0 quality envelope on both `jump` and `walk`.

Hybrid feasibility succeeds if MJX is fast enough for proposal generation and
MuJoCo-Warp top-K re-scoring reduces total wall time without changing final
selection quality materially.

The experiment fails constructively if it identifies one of these blockers:

- MJX cannot represent the WXY G1 WBC model semantics closely enough.
- Policy rollout parity fails before contact-sensitive events.
- Contact timing mismatch invalidates MJX final scoring.
- JAX/MJX steady-state speedup is too small to justify migration.

## Non-Goals

- Do not tune sweetpoint parameters.
- Do not replace the existing MuJoCo-Warp evaluator during this experiment.
- Do not add new reward weights.
- Do not broaden beyond `jump` and `walk`.
- Do not claim real-time deployment readiness.
- Do not use MPX/iLQR/SQP as an implementation dependency in this stage.

## Deliverables

- Stage 0 baseline package for `jump` and `walk`.
- Stage 1 model/kinematics parity report.
- Stage 2 rollout/contact parity report.
- Stage 3 speed report with compile time, steady-state window time, and full
  wall time.
- Decision note selecting Branch A, B, or C.
