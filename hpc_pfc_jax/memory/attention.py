"""Attention module for working memory integration."""

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp

from utils.prng import keygen


class Attention(eqx.Module):
    """Attention based working memory for RL agents using multi-head attention."""

    pos_emb: nn.RotaryPositionalEmbedding
    q: nn.Linear
    k: nn.Linear
    v: nn.Linear
    mha: nn.MultiheadAttention

    def __init__(
        self,
        qk_size: int,
        vo_size: int,
        key: jnp.ndarray,
        num_heads: int = 1,
        dropout_p: float = 0,
    ):
        super().__init__()
        kg = keygen(key)
        self.pos_emb = nn.RotaryPositionalEmbedding(vo_size)
        self.q = nn.Linear(vo_size, qk_size, key=next(kg), use_bias=False)
        self.k = nn.Linear(vo_size, qk_size, key=next(kg), use_bias=False)
        self.v = nn.Linear(vo_size, vo_size, key=next(kg), use_bias=False)
        self.mha = nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=qk_size,
            key_size=qk_size,
            value_size=vo_size,
            output_size=vo_size,
            dropout_p=dropout_p,
            key=next(kg),
        )

    def __call__(self, x: jnp.ndarray, keys: jnp.ndarray, states: jnp.ndarray) -> jnp.ndarray:
        """Compute the attention output for the last token.

        Args:
            x: The newest observation (1, embedding size)
            keys: Array of previous observations (sequence length, embedding size)
            states: Array of previous states to use as values (sequence length, embedding size)

        Returns:
            Attention output for the last token (1, embedding size)
        """
        x = jnp.expand_dims(x, 0)  # (1, embedding size)
        keys_pos_emb = self.pos_emb(keys)  # (seq_len, embedding size)
        vals_pos_emb = self.pos_emb(states)  # (seq_len, embedding size)

        query = jax.vmap(self.q)(x)  # (1, qk_size)
        key = jax.vmap(self.k)(keys_pos_emb)  # (seq_len, qk_size)
        value = jax.vmap(self.v)(vals_pos_emb)  # (seq_len, embedding size)
        output = self.mha(query, key, value) + x  # (1, embedding size)
        return output
