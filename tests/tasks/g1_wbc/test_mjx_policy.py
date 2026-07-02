import unittest

import numpy as np
import torch

from spider.tasks.g1_wbc.mjx_policy import convert_wbc_actor_to_jax, jax_actor_forward
from spider.tasks.g1_wbc.policy import WbcActor


class _NumpyJnp:
    @staticmethod
    def asarray(value):
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def where(condition, x, y):
        return np.where(condition, x, y)

    @staticmethod
    def expm1(value):
        return np.expm1(value)


class MjxPolicyTest(unittest.TestCase):
    def test_converted_actor_matches_torch_forward(self) -> None:
        torch.manual_seed(7)
        actor = WbcActor(
            input_dim=6,
            hidden_dims=(5, 4),
            output_dim=3,
        )
        actor.obs_mean.copy_(torch.linspace(-0.3, 0.2, 6).view(1, 6))
        actor.obs_std.copy_(torch.linspace(0.5, 1.0, 6).view(1, 6))
        for index, parameter in enumerate(actor.parameters()):
            values = torch.linspace(-0.2, 0.2, parameter.numel()).view_as(parameter)
            parameter.data.copy_(values + index * 0.03)
        actor.eval()
        obs = torch.tensor(
            [
                [-0.4, -0.1, 0.0, 0.1, 0.3, 0.7],
                [0.2, 0.4, -0.5, 0.8, -0.2, 0.0],
            ],
            dtype=torch.float32,
        )

        params = convert_wbc_actor_to_jax(actor, jnp=_NumpyJnp)
        actual = jax_actor_forward(params, obs.numpy(), jnp=_NumpyJnp)

        expected = actor(obs).detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5)

    def test_converter_keeps_linear_layer_order(self) -> None:
        actor = WbcActor(input_dim=2, hidden_dims=(3,), output_dim=1)

        params = convert_wbc_actor_to_jax(actor, jnp=_NumpyJnp)

        self.assertEqual(len(params.layers), 2)
        self.assertEqual(params.layers[0][0].shape, (2, 3))
        self.assertEqual(params.layers[1][0].shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
