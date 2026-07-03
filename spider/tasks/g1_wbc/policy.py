"""WBC policy loader for the G1 tracking checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from spider.tasks.g1_wbc.constants import ACTION_DIM, DEFAULT_CKPT_DIRS, OBS_DIM


class WbcActor(nn.Module):
    """MLP actor compatible with the saved WXY checkpoints."""

    def __init__(
        self,
        input_dim: int = OBS_DIM,
        hidden_dims: tuple[int, ...] = (2048, 2048, 1024, 1024, 512, 256, 128),
        output_dim: int = ACTION_DIM,
    ) -> None:
        super().__init__()
        dims = (input_dim, *hidden_dims, output_dim)
        modules: list[nn.Module] = []
        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                modules.append(nn.ELU())
        self.mlp = nn.Sequential(*modules)
        self.register_buffer("obs_mean", torch.zeros(1, input_dim))
        self.register_buffer("obs_std", torch.ones(1, input_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        obs = (obs - self.obs_mean) / (self.obs_std + 1.0e-2)
        return self.mlp(obs)


def resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).expanduser()
    if checkpoint in DEFAULT_CKPT_DIRS:
        directory = DEFAULT_CKPT_DIRS[str(checkpoint)]
        candidates = sorted(directory.glob("model_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No model_*.pt checkpoint found under {directory}")
        return candidates[-1].resolve()
    if path.is_dir():
        candidates = sorted(path.glob("model_*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No model_*.pt checkpoint found under {path}")
        return candidates[-1].resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path.resolve()


def load_wbc_actor(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cuda:0",
) -> WbcActor:
    ckpt_path = resolve_checkpoint_path(checkpoint)
    checkpoint_data = torch.load(ckpt_path, map_location="cpu")
    state_dict = _actor_state_dict(checkpoint_data, ckpt_path)
    _validate_wbc_actor_state_dict(state_dict, ckpt_path)
    actor = WbcActor()

    actor.obs_mean.copy_(state_dict["obs_normalizer._mean"])
    actor.obs_std.copy_(state_dict["obs_normalizer._std"])

    mlp_state = {}
    actor_state = actor.state_dict()
    for key, value in state_dict.items():
        if key.startswith("mlp."):
            mlp_state[key] = value
    missing = [key for key in actor_state if key.startswith("mlp.") and key not in mlp_state]
    if missing:
        raise ValueError(f"Checkpoint {ckpt_path} missing actor weights: {missing[:4]}")
    actor.load_state_dict({**actor.state_dict(), **mlp_state}, strict=True)
    actor.to(device)
    actor.eval()
    return actor


def _actor_state_dict(
    checkpoint_data: object,
    ckpt_path: Path,
) -> Mapping[str, Any]:
    if not isinstance(checkpoint_data, Mapping):
        raise ValueError(f"Unsupported G1 WBC checkpoint format for {ckpt_path}: not a dict.")
    for key in ("actor_state_dict", "model_state_dict", "state_dict"):
        value = checkpoint_data.get(key)
        if isinstance(value, Mapping):
            return value
    return checkpoint_data


def _validate_wbc_actor_state_dict(state_dict: Mapping[str, Any], ckpt_path: Path) -> None:
    keys = {str(key) for key in state_dict}
    has_obs_normalizer = {
        "obs_normalizer._mean",
        "obs_normalizer._std",
    }.issubset(keys)
    has_mlp = any(key.startswith("mlp.") for key in keys)
    if has_obs_normalizer and has_mlp:
        return
    description = _describe_checkpoint_keys(keys)
    raise ValueError(
        "Unsupported G1 WBC checkpoint format for "
        f"{ckpt_path}: {description}. Expected a WBC MLP actor checkpoint with "
        "obs_normalizer._mean, obs_normalizer._std, and mlp.* weights. "
        "SparseTrack transformer checkpoints require a separate policy/observation adapter."
    )


def _describe_checkpoint_keys(keys: set[str]) -> str:
    if any(key.startswith("actor.transformer_blocks.") for key in keys):
        return "looks like a SparseTrack transformer checkpoint"
    if any(key.startswith("actor_task_embedder.") for key in keys):
        return "looks like a SparseTrack transformer checkpoint"
    if any(key.startswith("actor.") for key in keys):
        return "looks like an RSL-RL/SparseTrack actor checkpoint"
    if not keys:
        return "empty state dict"
    sample = ", ".join(sorted(keys)[:4])
    return f"unrecognized keys [{sample}]"
