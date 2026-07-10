# Getting Started

This page covers the shortest supported path from a fresh checkout to a SPIDER
trajectory. The default public workflow uses MuJoCo Warp (MJWP) on an NVIDIA GPU.

## Requirements

- Linux with an NVIDIA driver compatible with the versions locked by the project
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Git and Git LFS

Optional backends use separate environments; see
[Optional Backends](/workflows/optional-backends).

## Installation

```bash
git clone https://github.com/facebookresearch/spider.git
cd spider
uv sync

git lfs install
git clone https://huggingface.co/datasets/retarget/retarget_example example_datasets
```

`pyproject.toml` and `uv.lock` are the dependency sources of truth. Avoid mixing
an additional requirements file into the uv environment.

## Run a Fast Example

The example dataset includes processed inputs for the reference tasks. Start
with GigaHands:

```bash
uv run examples/run_mjwp_fast.py \
  +override=gigahand_fast \
  task=p36-tea \
  embodiment_type=bimanual \
  data_id=0 \
  robot_type=xhand
```

The command reads the dataset and robot choices from
`examples/config/override/gigahand_fast.yaml`, then applies the explicit CLI
overrides above.

Outputs are written next to the processed trial:

```text
example_datasets/processed/gigahand/xhand/bimanual/p36-tea/0/
├── trajectory_mjwp_fast.npz
└── visualization_mjwp_fast.mp4
```

Video and trajectory creation can be disabled with `save_video=false` or
`save_info=false`.

## Other Reference Tasks

| Dataset | Command override | Example task | Embodiment |
| --- | --- | --- | --- |
| GigaHands | `gigahand_fast` | `p36-tea` | `bimanual` |
| Arctic v2 | `arcticv2_fast` | `s01-box_use_01` | `bimanual` |
| OakInk v2 | `oakinkv2_fast` | `pick_spoon_bowl` | `right` |

Replace the override and task in the example command. For new OakInk work,
prefer `oakinkv2`; the `oakink` pipeline consumes older preprocessed inputs.

The original, slower optimizer remains available for comparison:

```bash
uv run examples/run_mjwp.py \
  +override=gigahand_origin \
  task=p36-tea \
  embodiment_type=bimanual \
  data_id=0 \
  robot_type=xhand
```

## Headless Machines

MuJoCo rendering needs a headless backend when no display is available:

```bash
MUJOCO_GL=egl uv run examples/run_mjwp_fast.py \
  +override=gigahand_fast task=p36-tea data_id=0
```

For a run without interactive viewing or video rendering, set
`show_viewer=false save_video=false`.

## Where to Go Next

- [MJWP Workflow](/workflows/workflow-mjwp): process a dataset from raw input.
- [Configuration and Tuning](/usage/parameter-tuning): understand overrides,
  performance, and viewers.
- [Data and Outputs](/usage/data-structure): locate and inspect generated files.
- [Development](/development/add-dataset): add a dataset, robot, or backend.
