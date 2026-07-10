# Data and Outputs

SPIDER stores processed data by dataset, robot, embodiment, task, and trial. The
path helper in `spider/io.py` defines the robot-trial directory:

```text
{dataset_dir}/processed/{dataset_name}/{robot_type}/
  {embodiment_type}/{task}/{data_id}/
```

Do not infer paths from examples in experiment notes. `dataset_dir` and the five
identity fields above are the path contract.

## Common Layout

```text
example_datasets/
├── raw/
│   └── {dataset_name}/                     # Processor-specific source data
└── processed/
    └── {dataset_name}/
        ├── assets/
        │   └── objects/{object_name}/      # Visual and collision meshes
        ├── mano/{embodiment}/{task}/
        │   ├── task_info.json
        │   └── {data_id}/
        │       └── trajectory_keypoints.npz
        └── {robot_type}/{embodiment}/{task}/
            ├── task_info.json
            ├── scene.xml
            └── {data_id}/
                ├── trajectory_kinematic.npz
                ├── trajectory_mjwp.npz
                └── visualization_mjwp.mp4
```

Guidance and fast variants use names such as `scene_act.xml`,
`trajectory_kinematic_act.npz`, `trajectory_mjwp_fast.npz`, and
`visualization_mjwp_fast.mp4`.

## Artifact Contracts

### `trajectory_keypoints.npz`

Dataset processors write human and object motion in their common intermediate
form. Hand datasets normally contain arrays such as:

```text
qpos_wrist_right, qpos_finger_right, qpos_obj_right
qpos_wrist_left,  qpos_finger_left,  qpos_obj_left
```

The exact set is processor-specific. Contact detection may add `contact` and
`contact_pos`. Inspect a new processor's `np.savez` call rather than assuming an
external raw-data schema.

### `task_info.json`

Processors and scene-generation steps use this file for task-level facts such as
reference timing, object metadata, first-frame placement, and contact-site
information. Its keys vary with the dataset and preprocessing path.

### `scene.xml`

The scene combines robot assets, objects, actuators, collision geometry, and
contact pairs. It is stored at task level and shared by trial IDs. Keep it with
the trajectory: evaluation, replay, and deployment export may require the exact
model used to produce an output.

### `trajectory_kinematic.npz`

The maintained fast IK path writes `qpos`, `qvel`, and `frequency`. The original
IK path may also include rollout and contact arrays. Consumers must check keys
instead of assuming every IK implementation writes identical extras.

### `trajectory_mjwp*.npz`

MJWP entrypoints write executed states and controls, including `qpos`, `qvel`,
`ctrl`, and `time`, plus optimizer diagnostics produced by the selected path.
Arrays may retain control-window dimensions; flatten only the leading time/window
dimensions and preserve the final state or control dimension.

Inspect any NPZ safely with:

```bash
uv run python -c "import numpy as np; d=np.load('trajectory_mjwp.npz'); print({k: d[k].shape for k in d.files})"
```

## Resolving a Trial in Python

```python
from spider.io import get_processed_data_dir

trial_dir = get_processed_data_dir(
    dataset_dir="example_datasets",
    dataset_name="gigahand",
    robot_type="xhand",
    embodiment_type="bimanual",
    task="p36-tea",
    data_id=0,
)
```

`spider.config.process_config` uses the same helper, selects the appropriate
scene and kinematic input, and fills `model_path`, `data_path`, and `output_dir`.

## Deployment Export

`spider/postprocess/read_to_robot.py` converts a saved Allegro trajectory into
wrist, joint, and object poses for downstream replay:

```bash
uv run spider/postprocess/read_to_robot.py \
  --dataset-name oakink \
  --robot-type allegro \
  --embodiment-type bimanual \
  --task pick_spoon_bowl \
  --data-id 0 \
  --data-type mjwp
```

This is a format export, not a universal hardware controller. Confirm coordinate
frames, joint order, limits, rate, safety checks, and robot-specific control code
before executing a trajectory on hardware.

## Preservation Rules

- Keep the source configuration, scene XML, task metadata, and trajectory
  together when an output must be replayed or compared.
- Generated videos can be recreated; model and trajectory provenance cannot.
- Do not delete intermediate inputs merely because a final NPZ exists.
- Do not document a metrics schema unless the producing entrypoint writes that
  file on the current code path.
