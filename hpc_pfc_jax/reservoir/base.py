"""Base reservoir class for reservoir computing."""

import equinox as eqx
import jax
import jax.numpy as jnp


class Reservoir(eqx.Module):
    """Base class for all reservoirs.

    Typically, the only thing differing between reservoirs is their initialization function.
    If inheriting from this class, initialize the hidden size, weight_hh, and weight_ih,
    and then call super().__init__(hidden_size, weight_hh, weight_ih).

    For more details on reservoirs, see https://en.wikipedia.org/wiki/Reservoir_computing.
    """

    hidden_size: int = eqx.field(static=True)
    weight_hh: jax.Array
    weight_ih: jax.Array

    def __call__(self, x: jax.Array, hidden: jax.Array) -> jax.Array:
        """Compute one step of the reservoir.

        Args:
            x: Input vector.
            hidden: Previous hidden state.

        Returns:
            New hidden state.
        """
        return jnp.tanh(self.weight_ih @ x + self.weight_hh @ hidden)
