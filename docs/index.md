---
layout: home

hero:
  name: "SPIDER"
  text: "Scalable Physics-Informed DExterous Retargeting"
  tagline: A general framework for physics-based retargeting from human to diverse robot embodiments
  image:
    src: /figs/teaser.png
    alt: SPIDER
  actions:
    - theme: brand
      text: Get Started
      link: /guide/quick-start
    - theme: alt
      text: View on GitHub
      link: https://github.com/facebookresearch/spider

features:
  - icon: 🔬
    title: Physics-Based
    details: First general physics-based retargeting pipeline for both dexterous hand and humanoid robot manipulation

  - icon: ⚡
    title: Fast Simulation
    details: GPU-accelerated batched simulation with MuJoCo Warp

  - icon: 📊
    title: Rich Datasets, Robots and Simulators
    details: Dataset, robot, and simulator adapters for dexterous-hand and humanoid workflows

  - icon: 🔄
    title: Sim2Real Ready
    details: Trajectory export tools for robot-specific deployment pipelines
---

## Quick Example

```bash
# Clone example datasets
git clone https://huggingface.co/datasets/retarget/retarget_example example_datasets

# Install with uv
uv sync

# Run a reference task
uv run examples/run_mjwp_fast.py +override=gigahand_fast task=p36-tea data_id=0
```
