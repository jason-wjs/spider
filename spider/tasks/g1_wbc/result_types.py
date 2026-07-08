"""Shared result containers for G1 WBC task backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from spider.tasks.g1_wbc.motion import G1CommandBatch


@dataclass
class G1WbcWindowReplayState:
    initial_qpos: torch.Tensor
    initial_qvel: torch.Tensor
    initial_last_action: torch.Tensor | None = None
    initial_history_state: dict[str, Any] | None = None


@dataclass
class G1WbcExecutedCommandChunk:
    start: int
    execute_steps: int
    horizon_steps: int
    command: G1CommandBatch
    replay_state: G1WbcWindowReplayState | None = None


@dataclass
class G1WbcSpiderResult:
    command: G1CommandBatch
    rollout: Any
    refined_qpos: torch.Tensor
    controls: torch.Tensor
    infos: list[dict[str, Any]]
    scores: torch.Tensor
    num_windows: int = 0
    executed_command_chunks: list[G1WbcExecutedCommandChunk] = field(default_factory=list)


@dataclass
class G1WbcMpcRun:
    receding: Any
    result: G1WbcSpiderResult
    metadata: dict[str, Any]


__all__ = [
    "G1WbcExecutedCommandChunk",
    "G1WbcMpcRun",
    "G1WbcSpiderResult",
    "G1WbcWindowReplayState",
]
