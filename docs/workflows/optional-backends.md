# Optional Backends

SPIDER includes adapters for DexMachina, HDMI, and ManipTrans. They are
best-effort integrations, not part of the default `uv sync` environment or the
maintained MJWP verification surface.

Each backend depends on an external project with its own Python, CUDA, simulator,
assets, and checkpoint requirements. Use the upstream installation instructions
for those requirements; use this page only for the SPIDER integration points.

## Support Matrix

| Backend | Purpose | Entrypoint | SPIDER config | External runtime |
| --- | --- | --- | --- | --- |
| DexMachina | Dexterous-hand MPC in Genesis | `examples/run_dexmachina.py` | `examples/config/dexmachina.yaml` | DexMachina and its Genesis fork |
| HDMI | Humanoid-object MPC and RL export | `examples/run_hdmi.py` | `examples/config/hdmi.yaml` | HDMI and MjLab |
| ManipTrans | Dexterous-hand MPC in IsaacGym | `examples/run_maniptrans.py` | `examples/config/maniptrans.yaml` | HO-Tracker/ManipTrans and IsaacGym |

These modules import their external simulator packages at runtime. Import errors
in the default SPIDER environment mean that the matching external environment is
not installed; they do not indicate an MJWP installation failure.

## Common Integration Pattern

1. Create and validate the environment required by the upstream project.
2. Install SPIDER into that environment without replacing its dependency set:

   ```bash
   python -m pip install --no-deps -e /path/to/spider
   ```

3. Make the upstream assets and reference trajectories available at the paths
   expected by that backend.
4. Run the SPIDER entrypoint from the repository root.

Do not reuse the main Python 3.12 uv environment when an upstream project pins an
older Python, PyTorch, or CUDA stack.

## DexMachina (Genesis)

- Upstream: [DexMachina](https://github.com/MandiZhao/dexmachina)
- SPIDER simulator adapter: `spider/simulators/dexmachina.py`
- Evaluation: `spider/postprocess/evaluate_dexmachina.py`
- Primary output: `trajectory_dexmachina.npz`

```bash
python examples/run_dexmachina.py
```

Task, robot, sample count, horizon, and output settings come from
`examples/config/dexmachina.yaml` and CLI overrides.

## HDMI (MjLab)

- Upstream: [HDMI](https://github.com/lecar-lab/hdmi)
- SPIDER simulator adapter: `spider/simulators/hdmi.py`
- Export to HDMI: `spider/postprocess/read_to_hdmi.py`

```bash
python examples/run_hdmi.py
```

The active viewer choices are defined by `examples/config/hdmi.yaml` and the
shared SPIDER viewer implementation. Do not rely on backend-specific viewer
names that are absent from that configuration.

## ManipTrans (IsaacGym)

- Upstream: [ManipTrans](https://github.com/ManipTrans/ManipTrans)
- SPIDER simulator adapter: `spider/simulators/maniptrans.py`
- Evaluation: `spider/postprocess/evaluate_maniptrans.py`
- Primary outputs: `rollout_isaac.npz`, `metrics_isaac.json`, and optional MP4

```bash
python examples/run_maniptrans.py
```

ManipTrans uses a legacy, separate IsaacGym environment. Dataset paths, residual
action settings, and evaluation defaults are defined in
`examples/config/maniptrans.yaml` and the entrypoint.

## Maintenance Boundary

SPIDER maintains the adapter code, configuration mapping, and repository
entrypoints. Exact external dependency versions, third-party source patches, and
upstream training procedures are outside this documentation contract. A backend
should only be promoted to the default workflow after it has a reproducible
environment and dedicated verification surface.
