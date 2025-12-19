"""Abstract base class for episodic memory."""

import abc

import equinox as eqx
import jax.numpy as jnp


class EpisodicMemory(eqx.Module):
    """Abstract class for an episodic memory module.

    An episodic memory is represented as a key-value pair, where the key is some feature
    representation of a data point, and the value is typically a reservoir posterior
    hidden state given that key. During interaction with an environment, the agent can
    query against the keys of this memory, for which the value at the index of the most
    similar key is returned.
    """

    @abc.abstractmethod
    def store(self, key: jnp.ndarray, value: jnp.ndarray) -> "EpisodicMemory":
        """Store key/value pair(s) into memory.

        Args:
            key: The key(s) to store.
            value: The associated value(s) to store.
        """

    @abc.abstractmethod
    def recall(self, key: jnp.ndarray, n_per_key: int = 1) -> jnp.ndarray:
        """Recall the values for n_per_key most similar data in the buffer for the key(s).

        Args:
            key: The query/queries.
            n_per_key: The number of values to recall per key.
        """

    @abc.abstractmethod
    def learned_recall(
        self, query: jnp.ndarray, key_fn: eqx.Module, gate_fn: eqx.nn.Sequential, n_per_key: int = 1
    ) -> jnp.ndarray:
        """Recall the values for n_per_key most similar data in the buffer for the key(s).

        Args:
            query: The query.
            key_fn: The key fn itself to transform the key buffer.
            gate_fn: The gate fn to produce a scalar between 0 and 1 for output.
            n_per_key: The number of values to recall per key.
        """

    @abc.abstractmethod
    def top_k_recall(self, key: jnp.ndarray, n_per_key: int = 1) -> jnp.ndarray:
        """Recall the values for n_per_key most similar data in the buffer for the key(s)."""

    @abc.abstractmethod
    def buffers(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the keys and values in the memory."""
