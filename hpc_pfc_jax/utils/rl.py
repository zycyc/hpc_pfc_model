"""RL helper functions."""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PyTree


def filter_incremental_update(new_tensors: PyTree, old_tensors: PyTree, step_size: float) -> PyTree:
    """Wrapper on top of optax.incremental_update that supports pytrees with non-array leaves."""
    new_tensors, _ = eqx.partition(new_tensors, eqx.is_array)
    old_tensors, static = eqx.partition(old_tensors, eqx.is_array)

    updated = optax.incremental_update(new_tensors, old_tensors, step_size)
    return eqx.combine(updated, static)


def distance_to_kth_nearest_neighbour(new_obs: jnp.ndarray, previous_obs: jnp.ndarray, k: int) -> jnp.ndarray:
    """Computes the distance between new_obs and the kth nearest neighbours in previous observations.

    Both new_obs and previous_obs need a sequence dimension (shape should be (seq, *feat)).
    """

    @eqx.filter_vmap(in_axes=(0, None))
    def _norm(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        diffs = a - b
        return jax.vmap(jnp.linalg.norm)(diffs)

    diffs = _norm(new_obs, previous_obs)
    diffs = jnp.reshape(diffs, shape=(diffs.shape[0], -1))

    match jax.default_backend():
        case "gpu":
            # Two orders of magnitude faster than top_k on GPU.
            closest_k_distances, _ = jax.vmap(jax.lax.approx_min_k, in_axes=(0, None))(diffs, k)
        case "cpu":
            # top_k is much faster than approx_min_k on CPU.
            negative_diffs = -diffs  # top_k on negative diffs is -min_k on positive diffs
            negative_closest_k_distances, _ = jax.vmap(jax.lax.top_k, in_axes=(0, None))(negative_diffs, k)
            closest_k_distances = -negative_closest_k_distances
        case _:
            raise ValueError(f"Unsupported backend: {jax.default_backend()}")
    return closest_k_distances[:, -1]
