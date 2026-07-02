"""JAX-shaped WBC actor helpers for the MJX backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from spider.tasks.g1_wbc.policy import WbcActor

try:
    from jax import tree_util as _jax_tree_util
except Exception:
    _jax_tree_util = None


def _register_pytree_node_class(cls):
    if _jax_tree_util is None:
        return cls
    return _jax_tree_util.register_pytree_node_class(cls)


@_register_pytree_node_class
@dataclass(frozen=True)
class JaxActorParams:
    obs_mean: Any
    obs_std: Any
    layers: tuple[tuple[Any, Any], ...]

    def tree_flatten(self):
        return (self.obs_mean, self.obs_std, self.layers), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del aux_data
        obs_mean, obs_std, layers = children
        return cls(obs_mean=obs_mean, obs_std=obs_std, layers=layers)


def convert_wbc_actor_to_jax(actor: WbcActor, *, jnp) -> JaxActorParams:
    """Convert a Torch WBC actor to array params consumable by JAX code."""

    layers: list[tuple[Any, Any]] = []
    modules = list(actor.mlp)
    if not modules or not isinstance(modules[0], nn.Linear):
        raise ValueError("WBC actor MLP must start with a Linear input layer")
    if not isinstance(modules[-1], nn.Linear):
        raise ValueError("WBC actor MLP must end with a Linear output layer")

    expect_linear = True
    for index, module in enumerate(modules):
        if expect_linear:
            if not isinstance(module, nn.Linear):
                raise ValueError(
                    f"Unsupported WBC actor module at index {index}: {type(module).__name__}; "
                    "expected Linear"
                )
            layers.append(
                (
                    jnp.asarray(module.weight.detach().cpu().numpy().T),
                    jnp.asarray(module.bias.detach().cpu().numpy()),
                )
            )
            expect_linear = False
            continue

        if not isinstance(module, nn.ELU):
            raise ValueError(
                f"Unsupported WBC actor module at index {index}: {type(module).__name__}; "
                "expected ELU"
            )
        if module.alpha != 1.0:
            raise ValueError(f"Unsupported WBC actor ELU alpha at index {index}: {module.alpha}")
        expect_linear = True

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
