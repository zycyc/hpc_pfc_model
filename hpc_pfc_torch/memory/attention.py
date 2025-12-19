"""Attention module with rotary positional embeddings.

This implements multi-head attention for combining working memory
(current hidden state) with episodic memory (retrieved past states).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE).

    Applies rotation matrices to embeddings to encode positional information.
    See: https://arxiv.org/abs/2104.09864

    Args:
        dim: Embedding dimension (must be even).
        max_seq_len: Maximum sequence length.
    """

    def __init__(self, dim: int, max_seq_len: int = 1024):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even for rotary embeddings"

        self.dim = dim
        self.max_seq_len = max_seq_len

        # Precompute rotation frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute sin/cos tables
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings.

        Args:
            x: Tensor of shape (..., seq_len, dim).

        Returns:
            Tensor with rotary embeddings applied.
        """
        seq_len = x.shape[-2]
        cos = self.cos_cached[:seq_len].to(x.dtype)
        sin = self.sin_cached[:seq_len].to(x.dtype)

        # Handle broadcasting for batched inputs
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

        return x * cos + self._rotate_half(x) * sin


class Attention(nn.Module):
    """Multi-head attention for combining working memory and episodic memory.

    This module applies attention between a query (current state) and
    key-value pairs (episodic memories + working memory history).

    Args:
        qk_size: Query/key projection dimension.
        vo_size: Value/output dimension.
        num_heads: Number of attention heads.
        dropout_p: Dropout probability.
    """

    def __init__(
        self,
        qk_size: int,
        vo_size: int,
        num_heads: int = 1,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        self.qk_size = qk_size
        self.vo_size = vo_size
        self.num_heads = num_heads
        self.head_dim = qk_size // num_heads

        assert qk_size % num_heads == 0, "qk_size must be divisible by num_heads"

        # Rotary positional embedding
        self.pos_emb = RotaryPositionalEmbedding(vo_size)

        # Linear projections
        self.q = nn.Linear(vo_size, qk_size, bias=False)
        self.k = nn.Linear(vo_size, qk_size, bias=False)
        self.v = nn.Linear(vo_size, vo_size, bias=False)

        # Output projection
        self.out_proj = nn.Linear(vo_size, vo_size, bias=False)

        self.dropout = nn.Dropout(dropout_p)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        keys: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention output.

        Args:
            x: Current observation embedding of shape (vo_size,) or (batch, vo_size).
            keys: Previous embeddings of shape (seq_len, vo_size) or (batch, seq_len, vo_size).
            states: Previous states as values of shape (seq_len, vo_size) or (batch, seq_len, vo_size).

        Returns:
            Attention output with residual connection, shape (1, vo_size) or (batch, 1, vo_size).
        """
        # Handle unbatched input
        unbatched = x.dim() == 1
        if unbatched:
            x = x.unsqueeze(0)
            keys = keys.unsqueeze(0)
            states = states.unsqueeze(0)

        batch_size = x.shape[0]
        seq_len = keys.shape[-2]

        # Add sequence dimension for query
        x = x.unsqueeze(-2)  # (batch, 1, vo_size)

        # Apply rotary positional embeddings to keys and values
        keys_pos = self.pos_emb(keys)
        vals_pos = self.pos_emb(states)

        # Compute Q, K, V projections
        query = self.q(x)      # (batch, 1, qk_size)
        key = self.k(keys_pos)  # (batch, seq_len, qk_size)
        value = self.v(vals_pos)  # (batch, seq_len, vo_size)

        # Reshape for multi-head attention
        # query: (batch, num_heads, 1, head_dim)
        # key: (batch, num_heads, seq_len, head_dim)
        # value: (batch, num_heads, seq_len, head_dim)
        query = query.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.vo_size // self.num_heads).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value)  # (batch, num_heads, 1, head_dim)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch, 1, num_heads, head_dim)
        attn_output = attn_output.view(batch_size, 1, self.vo_size)  # (batch, 1, vo_size)
        attn_output = self.out_proj(attn_output)

        # Residual connection
        output = attn_output + x

        if unbatched:
            output = output.squeeze(0)

        return output
