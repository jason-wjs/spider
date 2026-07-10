# Add a Dataset

A dataset processor converts source-specific motion and mesh data into SPIDER's
common keypoint and task-metadata artifacts. Raw source layouts are intentionally
not standardized; each upstream dataset has a different API and coordinate
system.

## Start from the Closest Processor

Use an existing processor with similar source data:

| Source type | Example |
| --- | --- |
| MANO and object poses | `spider/process_datasets/oakinkv2.py` |
| Bimanual hand-object sequences | `spider/process_datasets/arcticv2.py` |
| GigaHands recordings | `spider/process_datasets/gigahand.py` |
| Humanoid motion | `spider/process_datasets/gmr.py` |

Create `spider/process_datasets/<dataset_name>.py` with a CLI that accepts at
least the identity fields needed by the source: dataset root, task, embodiment,
and trial ID.

## Required Outputs

For a hand dataset, resolve the intermediate directory with
`get_processed_data_dir` using `robot_type="mano"`:

```python
from spider.io import get_processed_data_dir

output_dir = get_processed_data_dir(
    dataset_dir=dataset_dir,
    dataset_name="my_dataset",
    robot_type="mano",
    embodiment_type=embodiment_type,
    task=task,
    data_id=data_id,
)
```

Write `trajectory_keypoints.npz`. The maintained hand IK paths expect the
following arrays for the applicable sides:

```text
qpos_wrist_{right,left}: [T, 7]
qpos_finger_{right,left}: [T, num_fingertips, 7]
qpos_obj_{right,left}: [T, 7]
```

Seven-element poses use position followed by a MuJoCo-order quaternion
`[x, y, z, qw, qx, qy, qz]`. For a missing side, follow the closest existing
processor's zero-filled convention so downstream indexing stays stable.

At task level, write `task_info.json` with at least:

- `dataset_name`, `task`, `embodiment_type`, and `data_id`
- `ref_dt`
- object mesh references needed by decomposition and scene generation
- provenance required to reproduce cropping, alignment, and coordinate changes

Object visual meshes belong under:

```text
{dataset_dir}/processed/{dataset_name}/assets/objects/{object_name}/visual.obj
```

Record paths relative to `dataset_dir` when existing processors do so.

## Coordinate and Timing Contract

- Convert positions to the MuJoCo world frame used by the generated scene.
- Store quaternions in `wxyz` order and normalize them.
- Preserve the source frame rate through `ref_dt`; do not silently assume 50 Hz.
- Document frame cropping, global rotation, translation offsets, and object mesh
  transforms in `task_info.json`.
- Ensure the first frame is physically compatible with the generated floor and
  object placement.

## Validate the Processor

Run one short trial, then inspect paths, keys, shapes, finite values, and
quaternion norms:

```bash
uv run spider/process_datasets/my_dataset.py \
  --task example_task \
  --embodiment-type right \
  --data-id 0

uv run python -c "import numpy as np; d=np.load('example_datasets/processed/my_dataset/mano/right/example_task/0/trajectory_keypoints.npz'); print({k: d[k].shape for k in d.files})"
```

Then execute the shared stages in order:

1. `spider/preprocess/decompose_fast.py` or the optional CoACD-based
   `spider/preprocess/decompose.py`
2. optional `spider/preprocess/detect_contact.py`
3. `spider/preprocess/generate_xml.py`
4. `spider/preprocess/ik_fast.py`
5. a short MJWP run

Successful processor execution alone is insufficient; the generated scene and
IK trajectory must agree with the source motion visually and numerically.
