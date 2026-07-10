# MuJoCo Warp Workflow

MuJoCo Warp (MJWP) is SPIDER's maintained default workflow. It covers dataset
conversion, scene generation, inverse kinematics, and batched physics-based
retargeting.

For a preprocessed example, start with [Getting Started](/guide/quick-start).
The steps below are for rebuilding a trial from its source dataset.

## Pipeline

```text
dataset processor
  -> object decomposition
  -> optional contact detection
  -> scene generation
  -> inverse kinematics
  -> MJWP optimization
  -> optional deployment export
```

The processor is dataset-specific. The remaining stages share the same path
contract and CLI parameters.

## Choose an Input

| Dataset | Processor | Recommended override | Example task |
| --- | --- | --- | --- |
| GigaHands | `spider/process_datasets/gigahand.py` | `gigahand_fast` | `p36-tea` |
| Arctic v2 | `spider/process_datasets/arcticv2.py` | `arcticv2_fast` | `s01-box_use_01` |
| OakInk v2 | `spider/process_datasets/oakinkv2.py` | `oakinkv2_fast` | `pick_spoon_bowl` |
| FAIR-MON | `spider/process_datasets/fair_mon.py` | `fair_mon` | dataset-specific |
| FAIR-FRE | `spider/process_datasets/fair_fre.py` | `fair_fre` | dataset-specific |

Use the dataset processor's `--help` output for raw-data-specific arguments.
The following example uses GigaHands.

```bash
export TASK=p36-tea
export DATASET_NAME=gigahand
export ROBOT_TYPE=xhand
export EMBODIMENT_TYPE=bimanual
export DATA_ID=0
```

## 1. Process the Dataset

```bash
uv run spider/process_datasets/gigahand.py \
  --task=${TASK} \
  --embodiment-type=${EMBODIMENT_TYPE} \
  --data-id=${DATA_ID}
```

The common processor output is named `trajectory_keypoints.npz` and lives in
the dataset's `mano` trial directory. Processors also write task metadata and
copy or reference the required object assets.

## 2. Build Collision Meshes

```bash
uv run spider/preprocess/decompose_fast.py \
  --task=${TASK} \
  --dataset-name=${DATASET_NAME} \
  --data-id=${DATA_ID} \
  --embodiment-type=${EMBODIMENT_TYPE}
```

`decompose_fast.py` uses dependencies available in the locked environment.
`decompose.py` is the higher-fidelity CoACD path, but CoACD is optional and must
be installed separately. Prefer it for strongly concave objects when available.

## 3. Detect Contacts When Needed

```bash
uv run spider/preprocess/detect_contact.py \
  --task=${TASK} \
  --dataset-name=${DATASET_NAME} \
  --data-id=${DATA_ID} \
  --embodiment-type=${EMBODIMENT_TYPE}
```

This updates `trajectory_keypoints.npz`. It is optional unless the selected
configuration or downstream workflow consumes contact annotations.

## 4. Generate the Scene

```bash
uv run spider/preprocess/generate_xml.py \
  --task=${TASK} \
  --dataset-name=${DATASET_NAME} \
  --data-id=${DATA_ID} \
  --embodiment-type=${EMBODIMENT_TYPE} \
  --robot-type=${ROBOT_TYPE}
```

The generated `scene.xml` is shared by trial IDs for the same task. Guidance
variants use the script's `--act-scene` option and produce `scene_act.xml`.

## 5. Run Inverse Kinematics

The maintained fast IK path is:

```bash
uv run spider/preprocess/ik_fast.py \
  --task=${TASK} \
  --dataset-name=${DATASET_NAME} \
  --data-id=${DATA_ID} \
  --embodiment-type=${EMBODIMENT_TYPE} \
  --robot-type=${ROBOT_TYPE}
```

It writes `trajectory_kinematic.npz` in the robot trial directory. The original
`ik.py` remains available for experiments that depend on its behavior.

## 6. Run Physics Retargeting

```bash
uv run examples/run_mjwp_fast.py \
  +override=${DATASET_NAME}_fast \
  task=${TASK} \
  data_id=${DATA_ID} \
  robot_type=${ROBOT_TYPE} \
  embodiment_type=${EMBODIMENT_TYPE}
```

Use `examples/run_mjwp.py` with the corresponding non-fast or `_origin`
override when reproducing the original optimizer. The entrypoints write
`trajectory_mjwp_fast.npz` or `trajectory_mjwp.npz` and optional MP4 previews.

## Configuration and Outputs

- [Configuration and Tuning](/usage/parameter-tuning) explains Hydra overrides,
  timing relationships, memory, and viewers.
- [Data and Outputs](/usage/data-structure) documents the current path and
  artifact contracts.
- `examples/config/override/` is the source of truth for dataset-specific
  settings.
