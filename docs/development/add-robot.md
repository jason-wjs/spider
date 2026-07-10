# Add a Robot

Adding a robot means supplying a MuJoCo model that matches SPIDER's scene and IK
conventions. There is no global `ROBOT_TYPES` registry: `robot_type` resolves to
an asset directory by name.

## Asset Layout

Keep the source model in:

```text
spider/assets/robots/{robot_type}/
├── left.xml
├── right.xml
├── bimanual.xml
├── meshes/
└── retarget_config.yaml        # when required by the IK path
```

Only create the embodiment files the robot supports. Existing examples include
`xhand`, `allegro`, `inspire`, `schunk`, and `ability`.

Scene generation currently loads the model from the selected dataset's processed
assets:

```text
{dataset_dir}/processed/{dataset_name}/assets/robots/{robot_type}/
```

Copy or provision the source asset there before running `generate_xml.py`.
Dataset-specific tooling may automate this, but `generate_xml.py` does not copy
the robot automatically.

## Model Contract for Dexterous Hands

The maintained hand pipeline assumes:

- MuJoCo-loadable XML with valid relative mesh paths
- position actuators ordered consistently with the controlled robot joints
- `right_palm` and/or `left_palm` sites
- fingertip sites named `{side}_{finger}_tip`
- `thumb`, `index`, `middle`, `ring`, and normally `pinky` fingers
- collision geoms with stable names and simplified geometry
- explicit joint limits, actuator ranges, inertias, and solver-safe masses

Allegro and MetaHand use four fingertips in the current IK code; other hand
types use five. If a new robot differs, update and test the site-selection logic
in both `spider/preprocess/ik.py` and `spider/preprocess/ik_fast.py`.

Trace/contact sites are optional unless the selected viewer, reward, or contact
workflow consumes them. Follow the naming in the closest existing robot rather
than inventing a second convention.

## Model Contract for Humanoids

Humanoid assets require a scene and reference pipeline that agree on:

- floating-base and actuated-joint order
- feet and other collision geometry
- tracked body/site names
- contact pairs and friction
- controller units and actuator limits
- the reference processor's embodiment name

Use the current `unitree_g1` assets and the matching humanoid configuration as a
reference. Hand-specific assumptions in the generic MJWP path do not
automatically generalize to a humanoid model.

## Integration Sequence

1. Load every XML directly with MuJoCo before using SPIDER.
2. Place the asset under the processed dataset's `assets/robots` directory.
3. Run `generate_xml.py` for one known task.
4. Inspect the generated scene, collision geometry, joint limits, and sites.
5. Run `ik_fast.py` and verify tracking without physics optimization.
6. Run a short MJWP trajectory with conservative settings.
7. Check saved states, control dimensions, contacts, and replay.

Example commands:

```bash
uv run spider/preprocess/generate_xml.py \
  --dataset-name gigahand \
  --task p36-tea \
  --data-id 0 \
  --embodiment-type right \
  --robot-type my_robot

uv run spider/preprocess/ik_fast.py \
  --dataset-name gigahand \
  --task p36-tea \
  --data-id 0 \
  --embodiment-type right \
  --robot-type my_robot
```

## Review Checklist

- XML and all meshes load without path rewriting outside the repository.
- Site names match the selected IK implementation.
- Actuator count and order match the controls written by IK.
- Joint ranges and control ranges are enforced.
- Collision geometry is simple enough for the intended batch size.
- A saved trajectory replays with the same scene and joint ordering.
- Robot-specific assumptions are covered by focused tests or validation scripts.
