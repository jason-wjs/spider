"""Shared result containers for G1 WBC task backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from spider.tasks.g1_wbc.motion import G1CommandBatch


@dataclass
class G1WbcSpiderResult:
    command: G1CommandBatch
    rollout: Any
    refined_qpos: torch.Tensor
    controls: torch.Tensor
    infos: list[dict[str, Any]]
    scores: torch.Tensor
    num_windows: int = 0


@dataclass
class G1WbcMpcRun:
    receding: Any
    result: G1WbcSpiderResult
    metadata: dict[str, Any]


__all__ = ["G1WbcMpcRun", "G1WbcSpiderResult"]
