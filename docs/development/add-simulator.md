# Add a Simulator Backend

SPIDER backend integration is entrypoint-owned. Each runnable workflow imports a
simulator adapter directly; there is no runtime registry or fully uniform
`Simulator` protocol.

Examples:

```text
examples/run_mjwp.py        -> spider/simulators/mjwp.py
examples/run_dexmachina.py  -> spider/simulators/dexmachina.py
examples/run_hdmi.py        -> spider/simulators/hdmi.py
examples/run_maniptrans.py  -> spider/simulators/maniptrans.py
```

Start from the pair whose state model and optimizer behavior most closely match
the new backend.

## Files to Add

```text
spider/simulators/my_backend.py
examples/run_my_backend.py
examples/config/my_backend.yaml
tests or an environment-specific validation script
```

Add the backend to [Optional Backends](/workflows/optional-backends) until its
environment and verification surface are reproducible.

## Adapter Responsibilities

Sampling entrypoints typically need functions in these groups:

| Group | Typical functions |
| --- | --- |
| Lifecycle | `setup_env`, `step_env` |
| Optimizer state | `save_state`, `load_state`, `copy_sample_state` |
| Scoring | `get_reward`, `get_terminal_reward`, `get_terminate` |
| Diagnostics | `get_trace` |
| Randomization | `save_env_params`, `load_env_params` |
| Execution sync | `sync_env` |

These names are conventions, not a promise of identical signatures. For
example, MJWP synchronization also consumes MuJoCo data, while some external
backends accept an optional viewer state. The entrypoint passes concrete
functions into optimizer factories, so signatures must match that entrypoint.

## State Requirements

`save_state` and `load_state` must round-trip every value that affects future
simulation. Positions and velocities alone may be insufficient; include time,
controls, solver state, randomization state, and backend caches when applicable.

`copy_sample_state` must select the winning parallel environment without
changing tensor rank or dropping backend state. `sync_env` must make the executed
environment agree with the selected optimizer state and any visualization model.

## Tensor and Device Contract

- State, controls, rewards, and termination masks must stay on the configured
  device during optimization.
- The leading dimension represents parallel candidates unless the selected
  entrypoint explicitly documents another layout.
- Reward returns one value per candidate.
- Termination returns a boolean-compatible value per candidate.
- Avoid hidden host transfers inside rollout loops.
- Document coordinate, quaternion, and action conventions at the adapter boundary.

## Configuration

Put external-environment settings in `examples/config/my_backend.yaml`. Fields
shared with `Config` are filtered into the dataclass; backend-only values may
need explicit handling in the entrypoint, as ManipTrans does.

Do not add a backend dependency to the default environment unless the default
workflow imports it. Optional modules should fail with a clear installation
message or be tested only inside their dedicated environment.

## Verification Sequence

1. Import the adapter in its intended external environment.
2. Create the smallest possible batch and step it once.
3. Save state, step, restore, and prove the next step is reproduced.
4. Verify reward and termination shapes and devices.
5. Run one optimizer window and confirm the selected sample is restored.
6. Run two identical seeded processes and compare outputs.
7. Save and replay a short trajectory.
8. Measure compile/warmup separately from steady-state optimization.

A backend is not production-ready merely because it imports or steps. State
restoration, scoring semantics, repeatability, output provenance, and replay all
need explicit evidence.
