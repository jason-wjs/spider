from __future__ import annotations

from types import SimpleNamespace

import torch

from spider.optimizers.receding import run_receding_horizon


def test_run_receding_horizon_shifts_remaining_controls() -> None:
    config = SimpleNamespace(
        horizon_steps=6,
        ctrl_steps=2,
        max_num_iterations=0,
    )
    controls = torch.arange(6, dtype=torch.float32)[:, None]
    optimize_inputs: list[list[float]] = []
    ref_starts: list[int] = []
    executed: list[list[float]] = []

    def optimize(config, env, controls, ref_slice):
        optimize_inputs.append(controls[:, 0].tolist())
        return controls, {}

    def get_ref_slice(start: int, horizon: int):
        ref_starts.append(start)
        return (torch.tensor([start]), torch.tensor([horizon]))

    def execute_controls(chunk: torch.Tensor, sim_step: int):
        executed.append(chunk[:, 0].tolist())
        return {"sim_step_echo": torch.tensor([sim_step])}

    def make_tail_controls(sim_step: int, steps: int):
        values = torch.arange(steps, dtype=torch.float32) + float(sim_step * 100)
        return values[:, None]

    result = run_receding_horizon(
        config,
        env=None,
        controls=controls,
        total_steps=5,
        optimize=optimize,
        get_ref_slice=get_ref_slice,
        execute_controls=execute_controls,
        make_tail_controls=make_tail_controls,
    )

    assert result.executed_steps == 5
    assert ref_starts == [0, 2, 4]
    assert optimize_inputs == []
    assert executed == [[0.0, 1.0], [2.0, 3.0], [4.0]]
    assert result.controls[:, 0].tolist() == [
        5.0,
        200.0,
        201.0,
        400.0,
        401.0,
        500.0,
    ]


def test_run_receding_horizon_can_disable_warm_start_shift() -> None:
    config = SimpleNamespace(
        horizon_steps=6,
        ctrl_steps=2,
        max_num_iterations=0,
        use_warm_start=False,
    )
    controls = torch.arange(6, dtype=torch.float32)[:, None]
    executed: list[list[float]] = []

    def get_ref_slice(start: int, horizon: int):
        return (torch.tensor([start]), torch.tensor([horizon]))

    def execute_controls(chunk: torch.Tensor, sim_step: int):
        executed.append(chunk[:, 0].tolist())
        return {"sim_step_echo": torch.tensor([sim_step])}

    def make_tail_controls(sim_step: int, steps: int):
        values = torch.arange(steps, dtype=torch.float32) + float(sim_step * 100)
        return values[:, None]

    result = run_receding_horizon(
        config,
        env=None,
        controls=controls,
        total_steps=5,
        optimize=lambda config, env, controls, ref_slice: (controls, {}),
        get_ref_slice=get_ref_slice,
        execute_controls=execute_controls,
        make_tail_controls=make_tail_controls,
    )

    assert executed == [[0.0, 1.0], [200.0, 201.0], [400.0]]
    assert result.controls[:, 0].tolist() == [
        500.0,
        501.0,
        502.0,
        503.0,
        504.0,
        505.0,
    ]
