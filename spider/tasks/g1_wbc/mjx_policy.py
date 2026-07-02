"""JAX-shaped WBC actor helpers for the MJX backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from spider.tasks.g1_wbc.policy import WbcActor


@dataclass(frozen=True)
class JaxActorParams:
    obs_mean: Any
    obs_std: Any
    layers: tuple[tuple[Any, Any], ...]


def convert_wbc_actor_to_jax(actor: WbcActor, *, jnp) -> JaxActorParams:
    """Convert a Torch WBC actor to array params consumable by JAX code."""

    layers: list[tuple[Any, Any]] = []
    for module in actor.mlp:
        if not isinstance(module, nn.Linear):
            continue
        layers.append(
            (
                jnp.asarray(module.weight.detach().cpu().numpy().T),
                jnp.asarray(module.bias.detach().cpu().numpy()),
            )
        )
    return JaxActorParams(
        obs_mean=jnp.asarray(actor.obs_mean.detach().cpu().numpy()),
        obs_std=jnp.asarray(actor.obs_std.detach().cpu().numpy()),
        layers=tuple(layers),
    )


def jax_actor_forward(params: JaxActorParams, obs, *, jnp):
    """Run the WBC actor MLP with JAX-compatible array operations."""

    x = (obs - params.obs_mean) / (params.obs_std + 1.0e-2)
    for index, (weight, bias) in enumerate(params.layers):
        x = x @ weight + bias
        if index < len(params.layers) - 1:
            x = jnp.where(x > 0.0, x, jnp.expm1(x))
    return x
