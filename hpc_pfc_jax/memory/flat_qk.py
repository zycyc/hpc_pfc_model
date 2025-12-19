"""Flat key-value episodic memory with learned query-key transformations."""

import equinox as eqx
import jax
import jax.numpy as jnp

from memory.base import EpisodicMemory
from utils.buffer import TensorCircularBuffer


def calc_attn(observation: jnp.ndarray, prev_observations: jnp.ndarray, beta: float, num_feature_dims: int):
    """Calculates the attention of the observation to the previous observations.

    Args:
        observation: The observation to calculate the attention for.
        prev_observations: The previous observations to calculate the attention over.
        beta: The beta parameter for the attention calculation.
        num_feature_dims: The number of feature dimensions to calculate the attention over.
    """
    squared_diff = jnp.pow((prev_observations - observation), 2)
    attn = jnp.mean(jnp.exp(-squared_diff * beta), range(-num_feature_dims, 0))
    return attn


def familiarity(
    observation: jnp.ndarray,
    buffer_keys: jnp.ndarray,
    beta: float,
    num_feature_dims: int,
):
    """Calculate the mean and max attention of the observation to the buffer keys.

    Also return the index of the most familiar key.

    Args:
        observation: The observation to calculate the attention for.
        buffer_keys: The keys in the buffer to calculate the attention over.
        beta: The beta parameter for the attention calculation.
        num_feature_dims: The number of feature dimensions to calculate the attention over.
    """
    attn = calc_attn(observation, buffer_keys, beta, num_feature_dims)
    most_familiar = jnp.argmax(attn)
    attn_stats = jnp.concatenate(
        [
            jnp.mean(attn, keepdims=True),
            jnp.max(attn, keepdims=True),
        ]
    )
    return attn_stats, most_familiar


class FlatQKLearnedMemory(EpisodicMemory):
    """Flat episodic memory with learned query-key transformations.

    This memory stores key-value pairs in circular buffers and supports
    learned transformations for query and key matching.
    """

    nkey_features: int = eqx.field(static=True)
    nvalue_features: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    keys: TensorCircularBuffer
    values: TensorCircularBuffer

    def __init__(
        self,
        batch_size: int,
        capacity: int,
        nkey_features: int,
        nvalue_features: int,
        _keys: TensorCircularBuffer | None = None,
        _values: TensorCircularBuffer | None = None,
    ):
        """Initialize the memory.

        Args:
            capacity: The maximum number of keys and values to store.
            nkey_features: Number of features in keys.
            nvalue_features: Number of features in values.
            batch_size: The batch size of the keys and values.
        """
        super().__init__()
        self.batch_size = batch_size
        self.nkey_features = nkey_features
        self.nvalue_features = nvalue_features
        self.capacity = capacity
        if any((_keys is not None, _values is not None)):
            assert _keys is not None
            assert _values is not None
            self.keys = _keys
            self.values = _values
        else:
            self.keys = TensorCircularBuffer(batch_size=batch_size, capacity=capacity, feature_shape=(nkey_features,))
            self.values = TensorCircularBuffer(
                batch_size=batch_size, capacity=capacity, feature_shape=(nvalue_features,)
            )

    def store(self, key: jnp.ndarray, value: jnp.ndarray) -> "FlatQKLearnedMemory":
        """Stores a key-value pair in the next available slot in memory."""
        keys = self.keys.append(key)
        values = self.values.append(value)
        return FlatQKLearnedMemory(
            self.batch_size, self.capacity, self.nkey_features, self.nvalue_features, keys, values
        )

    def learned_recall(
        self, query: jnp.ndarray, key_fn: eqx.Module, gate_fn: eqx.nn.Sequential, n_per_key: int = 1
    ) -> jnp.ndarray:
        """Recalls values using learned key transformation.

        Note: Uses an approximation due to performance reasons.
        """

        def closest_key(query, key_fn, gate_fn, key_buffer, values_buffer):
            # This is used in the experiments where we learn arbitrary transform to the obs as key
            key = eqx.filter_vmap(key_fn)(key_buffer)

            # No gating, just use dot product
            dot_products = jnp.dot(key, query)  # q*M
            weights = jax.nn.softmax(dot_products)
            weighted_values = jnp.einsum("i,ij->j", weights, values_buffer)
            return jnp.expand_dims(weighted_values, axis=0)

        if self.keys.buffer.shape == (self.batch_size, self.capacity, self.nkey_features):
            closest_fn = eqx.filter_vmap(closest_key, in_axes=(0, None, None, 0, 0))
        else:
            closest_fn = closest_key

        return closest_fn(query, key_fn, gate_fn, self.keys.buffer, self.values.buffer)

    def recall(self, key: jnp.ndarray, n_per_key: int = 1) -> jnp.ndarray:
        """Recalls the approximate n_per_key most similar keys and values to the given key.

        Note: Uses an approximation due to performance reasons.
        """

        def closest_key(key, key_buffer, values_buffer):
            # This is used in the experiments where we learn arbitrary transform to the obs as key
            q = key  # we're querying the keys (M, or key_buffer)
            dot_products = jnp.dot(key_buffer, q)  # q*M
            weights = jax.nn.softmax(dot_products)
            weighted_values = jnp.einsum("i,ij->j", weights, values_buffer)
            return jnp.expand_dims(weighted_values, axis=0)

        if self.keys.buffer.shape == (self.batch_size, self.capacity, self.nkey_features):
            closest_fn = eqx.filter_vmap(closest_key)
        else:
            closest_fn = closest_key

        return closest_fn(key, self.keys.buffer, self.values.buffer)

    def top_k_recall(self, key: jnp.ndarray, n_per_key: int = 1) -> jnp.ndarray:
        """Recalls the approximate n_per_key most similar keys and values to the given key.

        Note: Uses an approximation due to performance reasons.
        """

        def closest_key(key, key_buffer, values_buffer):
            q = key
            dot_products = jnp.dot(key_buffer, q)
            _, top_k_indices = jax.lax.approx_max_k(dot_products, n_per_key)
            return values_buffer[top_k_indices]

        if self.keys.buffer.shape == (self.batch_size, self.capacity, self.nkey_features):
            closest_fn = eqx.filter_vmap(closest_key)
        else:
            closest_fn = closest_key

        return closest_fn(key, self.keys.buffer, self.values.buffer)

    def buffers(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Returns a tuple of the keys and values buffers."""
        return (self.keys.buffer, self.values.buffer)
