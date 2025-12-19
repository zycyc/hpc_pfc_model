"""Reservoir with local connectivity pattern."""

import jax
import jax.numpy as jnp
import torch
import torch.utils.dlpack
from loguru import logger

from utils.prng import keygen
from reservoir.base import Reservoir


def _rolling_mask(input_size: int, num_unique: int, num_shared: int):
    """Create a rolling mask for input-to-hidden connectivity."""
    assert input_size > 0, "'input_size' must be a positive integer"
    assert isinstance(num_unique, int), "'num_unique' must be an integer"
    assert isinstance(num_shared, int), "'num_shared' must be an integer"
    assert num_unique + num_shared > 0, "'num_unique' + 'num_shared' must be greater than 0"
    wmask = jnp.kron(jnp.identity(input_size, dtype=jnp.int32), jnp.ones((num_unique + num_shared, 1), dtype=jnp.int32))
    return (wmask | jnp.roll(wmask, num_shared, 0)).astype(jnp.bool)


def _band_2d(shape: tuple[int, int], width: int) -> jax.Array:
    """Returns a 2D matrix of booleans where the diagonal of the requested width is True."""
    i, j = jnp.indices(shape)
    return jnp.abs(i - j) < width


def _random_band_tensor(m: int, band_reach: int, key: jnp.ndarray, uniform_range: tuple[float, float] = (-1.0, 1.0)):
    """Create a random banded matrix."""
    assert m >= 0, "'m' must be a non-negative integer"
    assert 0 <= band_reach <= m, "'band_reach' must be a non-negative integer bounded by 'm'"
    shape = (m, m)
    return jnp.where(
        _band_2d(shape, band_reach),
        jax.random.uniform(key, shape, minval=uniform_range[0], maxval=uniform_range[1]),
        jnp.zeros(shape),
    )


class ReservoirLocalConnectivity(Reservoir):
    """Reservoir with local connectivity plus sparse global connections.

    This creates a reservoir where hidden units have strong local connections
    (forming a banded structure) plus sparse random global connections.
    """

    def __init__(
        self,
        key,
        input_size=279,
        num_unique=20,
        num_shared=10,
        reach=10,
        input_conn_prob=0.5,
        local_conn_prob=0.5,
        global_conn_prob=0.01,
        _random_band_uniform_range=(-1.0, 1.0),
        _hidden_to_hidden_connections_uniform_range=(-1.0, 1.0),
    ):
        kg = keygen(key)
        wmask = _rolling_mask(input_size, num_unique, num_shared)
        hidden_size = input_size * (num_unique + num_shared)
        rnn_weight_hh = _random_band_tensor(hidden_size, reach, next(kg), _random_band_uniform_range)
        rnn_weight_hh = jax.random.bernoulli(p=local_conn_prob, shape=rnn_weight_hh.shape, key=next(kg)) * rnn_weight_hh
        r1, r2 = _hidden_to_hidden_connections_uniform_range
        new_cons_hh = jax.random.uniform(key, rnn_weight_hh.shape, minval=r1, maxval=r2)
        new_cons_hh = jax.random.bernoulli(p=global_conn_prob, shape=new_cons_hh.shape, key=next(kg)) * new_cons_hh
        rnn_weight_hh = rnn_weight_hh + new_cons_hh

        # JAX CPU eigenvalue solver is very slow, and it does not provide a GPU implementation
        # for non-symmetric matrix eigenvalue decomposition, therefore using torch's
        logger.debug("Solving principal eigenvalue of rnn weight hh")
        rnn_weight_hh_torch = torch.utils.dlpack.from_dlpack(rnn_weight_hh)  # Shares underlying storage
        principal_eigvalue = torch.max(torch.real(torch.linalg.eigvals(rnn_weight_hh_torch))).item()
        rnn_weight_hh = rnn_weight_hh / principal_eigvalue
        logger.debug("Solved principal eigenvalue")

        rnn_weight_ih = jax.nn.initializers.glorot_uniform()(key, (hidden_size, input_size), dtype=jnp.float32) * 10
        rnn_weight_ih = jax.random.bernoulli(p=input_conn_prob, shape=rnn_weight_ih.shape, key=next(kg)) * rnn_weight_ih
        rnn_weight_ih = rnn_weight_ih * wmask

        super().__init__(hidden_size=hidden_size, weight_hh=rnn_weight_hh, weight_ih=rnn_weight_ih)
