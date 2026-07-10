# Configuration and Tuning

SPIDER uses Hydra for entrypoint configuration and a `Config` dataclass for the
processed runtime state. Tune from a dataset/backend override instead of copying
defaults into a new command.

## Configuration Precedence

For `run_mjwp.py` and `run_mjwp_fast.py`, values are resolved in this order:

1. `examples/config/default.yaml`
2. a selected file in `examples/config/override/`
3. explicit CLI overrides
4. derived fields computed by `spider.config.process_config`

For example:

```bash
uv run examples/run_mjwp_fast.py \
  +override=gigahand_fast \
  task=p36-tea \
  data_id=0 \
  num_samples=512
```

The YAML files are the source of truth for defaults. The dataclass also contains
fallback values for programmatic callers, so copying numbers from the class can
produce a different CLI experiment.

## Timing Parameters

| Parameter | Meaning | Main trade-off |
| --- | --- | --- |
| `sim_dt` | Physics timestep | Accuracy and simulator cost |
| `ctrl_dt` | Time executed before replanning | Responsiveness and number of optimizer calls |
| `horizon` | Look-ahead duration | Planning context, memory, and rollout work |
| `knot_dt` | Control parameter spacing | Search dimension and command flexibility |
| `ref_dt` | Input trajectory timestep | Reference interpolation |

`horizon`, `ctrl_dt`, and `knot_dt` must be divisible by `sim_dt`.
Longer horizons create more simulated steps; smaller control intervals invoke
the optimizer more often.

## Optimizer Parameters

| Parameter | Meaning | Main trade-off |
| --- | --- | --- |
| `num_samples` | Parallel candidates | Exploration versus GPU memory and compute |
| `max_num_iterations` | Updates per control window | Optimization depth versus latency |
| `temperature` | Sample-weight softness | Concentration versus robustness |
| `improvement_threshold` | Early-stop gate | Runtime versus additional refinement |
| `first_ctrl_noise_scale` / `last_ctrl_noise_scale` | Noise across the horizon | Local versus broad exploration |
| `final_noise_scale` | Iteration annealing target | Late-stage precision |

Samples, horizon steps, and iterations multiply the amount of rollout work.
A configuration that is fast at one experiment shape may be slow or out of
memory at another.

## Tuning Order

1. Start from the override matching the dataset and entrypoint.
2. Verify the unmodified configuration completes and produces plausible motion.
3. Establish quality with tracking, contact, smoothness, and visual checks.
4. Change one major work dimension at a time: samples, horizon, or iterations.
5. Adjust noise and reward weights only after identifying the failing behavior.
6. Repeat important comparisons across seeds or independent processes.

Do not promote a configuration from aggregate reward or one attractive rollout.
Backend throughput and end-to-end optimizer time are different measurements.

## Runtime Reporting

The MJWP entrypoints print a realtime rate computed as simulated control time
divided by optimizer wall time. Task-specific runners may persist the related
`realtime_factor` field. Use the metric already produced by the runner, report
compile/warmup separately, and compare configurations on the same hardware and
input surface.

## Viewers

Viewer selection is a string, and combinations are supported:

```bash
# Native local viewer
uv run examples/run_mjwp.py viewer=mujoco

# Rerun logging without the native MuJoCo window
uv run examples/run_mjwp.py viewer=rerun

# Combined local and Rerun views
uv run examples/run_mjwp.py viewer=mujoco-rerun

# Viser web view
uv run examples/run_mjwp.py viewer=viser
```

`show_viewer=false` disables interactive viewers. `save_video`, `save_rerun`,
and `save_viser` control saved artifacts. `rerun_spawn` controls whether SPIDER
spawns a local Rerun process; follow Rerun's current upstream documentation for
remote server and client commands.

On a headless machine, use `MUJOCO_GL=egl` when rendering is enabled. Disable
video as well as the viewer when no rendering output is needed.

## Common Failure Patterns

- **Out of memory:** reduce samples or horizon first; inspect the complete
  experiment shape rather than assuming a backend change removes scaling.
- **Slow optimization:** separate compile/warmup from steady state, then measure
  optimizer calls and control duration.
- **Jitter or unstable motion:** inspect control spacing, noise, contact, and
  smoothness together; more samples alone may not fix the mechanism.
- **Good metrics but poor replay:** verify that the scene, commands, reference,
  and replay path come from the same artifact set.
