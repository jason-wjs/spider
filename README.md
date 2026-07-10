<div align="center">

# 🕸️ SPIDER: Scalable Physics-Informed DExterous Retargeting

<p align="center">

  <a href="https://creativecommons.org/licenses/by-nc/4.0/">
    <img src="https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg" alt="License: CC BY-NC 4.0">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  </a>
  <a href="https://arxiv.org/abs/2511.09484">
    <img src="https://img.shields.io/badge/arXiv-2406.12345-b31b1b.svg" alt="arXiv">
  </a>

</p>

<p align="center">

  <a href="https://jc-bao.github.io/spider-project/"><b>Project Website</b></a> •
  <a href="https://facebookresearch.github.io/spider/"><b>Documentation</b></a> •
  <a href="https://huggingface.co/datasets/retarget/retarget_example"><b>Dataset</b></a>

</p>

![logo](figs/teaser.png)

</div>

## Overview

Scalable Physics-Informed DExterous Retargeting (SPIDER) is a general framework for physics-based retargeting from human to diverse robot embodiments, including both dexterous hand and humanoid robot.
It is designed to be a minimum, flexible and extendable framework for human2robot retargeting.
This code base provides the following pipeline from human video to robot actions:

![pipeline](figs/pipeline_animation.gif)


## Gallery

### Simulation results:

| Inspire Pick Tea Pot (Gigahands Dataset) | Xhand Play Glass (Hot3D dataset) | Schunk Pick Board (Oakink dataset)  | Allegro Pick Cat Toy (Reconstructed from single RGB video) |
| ---------------------------------------- | -------------------------------- | ----------------------------------- | ---------------------------------------------------------- |
| ![](figs/sim/inspire_pick_pot.gif)       | ![](figs/sim/xhand_glass.gif)    | ![](figs/sim/schunk_move_board.gif) | ![](figs/sim/allegro_pick_cat.gif)                         |


| G1 Pick                   | G1 Run                   | H1 Kick                   | T1 skip                   |
| ------------------------- | ------------------------ | ------------------------- | ------------------------- |
| ![](figs/sim/g1_pick.gif) | ![](figs/sim/g1_run.gif) | ![](figs/sim/h1_kick.gif) | ![](figs/sim/t1_skip.gif) |


### Multiple viewer support:
| Mujoco                              | Rerun                              |
| ----------------------------------- | ---------------------------------- |
| ![](figs/viewers/mujoco_viewer.gif) | ![](figs/viewers/rerun_viewer.gif) |


### Multiple simulators support:

| Genesis                      | Mujoco Warp              | IsaacGym              |
| ---------------------------- | ------------------------ | ---------------------- |
| ![](figs/sim/dexmachina.gif) | ![](figs/sim/mjwarp.gif) | ![](figs/sim/maniptrans.gif) |

### Deployment to real-world robots:

| Pick Cup                         | Rotate Bulb                         | Unplug Charger                 | Pick Duck                         |
| -------------------------------- | ----------------------------------- | ------------------------------ | --------------------------------- |
| ![](figs/real/pick_cup_real.gif) | ![](figs/real/rotate_bulb_real.gif) | ![](figs/real/unplug_real.gif) | ![](figs/real/pick_duck_real.gif) |


## Features

- First general **physics-based** retargeting pipeline for both dexterous hand and humanoid robot.
- Supports multiple robot and dataset adapters through explicit configuration.
- Integration points for RL training and BC data augmentation.
- Maintained MuJoCo Warp workflow with optional Genesis, HDMI, and IsaacGym adapters.
- Robot-specific trajectory export for downstream deployment pipelines.

![](figs/embodiment_support.png)

## Quickstart

Install the locked environment and clone the example data:

```bash
uv sync
git lfs install
git clone https://huggingface.co/datasets/retarget/retarget_example example_datasets
```

Run the maintained fast MJWP path on a preprocessed reference task:

```bash
uv run examples/run_mjwp_fast.py \
  +override=gigahand_fast \
  task=p36-tea \
  embodiment_type=bimanual \
  data_id=0 \
  robot_type=xhand
```

See [Getting Started](docs/guide/quick-start.md) for other reference tasks,
headless execution, and output locations.

## Workflow

| Workflow | Support level | Documentation |
| --- | --- | --- |
| MuJoCo Warp (MJWP) | Maintained default | [MJWP workflow](docs/workflows/workflow-mjwp.md) |
| DexMachina (Genesis) | Optional, best effort | [Optional backends](docs/workflows/optional-backends.md#dexmachina-genesis) |
| HDMI (MjLab) | Optional, best effort | [Optional backends](docs/workflows/optional-backends.md#hdmi-mjlab) |
| ManipTrans (IsaacGym) | Optional, best effort | [Optional backends](docs/workflows/optional-backends.md#maniptrans-isaacgym) |

The complete maintained pipeline is documented once in the MJWP workflow.
Configuration, data contracts, viewers, and development guides are available in
the [documentation site](https://facebookresearch.github.io/spider/).

## License

SPIDER is released under the Creative Commons Attribution-NonCommercial 4.0 license. See `LICENSE` for details.

## Code of Conduct

We expect everyone to follow the Contributor Covenant Code of Conduct in `CODE_OF_CONDUCT.md` when participating in this project.

## Acknowledgments

- Thanks Mandi Zhao for the help with the [DexMachina workflow](https://github.com/MandiZhao/dexmachina) for SPIDER + Genesis.
- Thanks Taylor Howell for the help in the early stages of integrating [Mujoco Wrap](https://github.com/google-deepmind/mujoco_warp) for SPIDER + MJWP.
- Thanks Haoyang Weng for the help with the [HDMI workflow](https://github.com/lecar-lab/hdmi) for SPIDER + Sim2real RL.
- Inverse kinematics design is ported from [GMR](https://github.com/YanjieZe/GMR) and [LocoMujoco](https://github.com/robfiras/loco-mujoco).
- Dataset processing is ported from [Hot3D](https://github.com/facebookresearch/hot3d), [Oakinkv2](https://github.com/oakink/OakInk2), [Maniptrans](https://github.com/ManipTrans/ManipTrans), [Gigahands](https://github.com/Gigahands/Gigahands).
- Visualization inspired by other good sampling repo like [Hydrax](https://github.com/vincekurtz/hydrax) and [Judo](https://github.com/bdaiinstitute/judo).


## Citation

```bibtex
@article{pan2025spiderscalablephysicsinformeddexterous,
      title={SPIDER: Scalable Physics-Informed Dexterous Retargeting},
      author={Chaoyi Pan and Changhao Wang and Haozhi Qi and Zixi Liu and Homanga Bharadhwaj and Akash Sharma and Tingfan Wu and Guanya Shi and Jitendra Malik and Francois Hogan},
      year={2025},
      eprint={2511.09484},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.09484},
}
```
