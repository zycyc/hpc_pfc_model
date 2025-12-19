"""Episodic Memory with learned query/key transformations.

This implements a key-value memory system for storing and retrieving
past experiences using learned transformations.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from memory.circular_buffer import TensorCircularBuffer


@dataclass
class FlatQKLearnedMemory:
    """Flat episodic memory with learned query/key transformations.

    This memory stores (key, value) pairs where:
    - Key: Observation features (transformed by learned pfc_k)
    - Value: Hidden states from the reservoir

    Retrieval uses softmax attention over transformed keys.

    Attributes:
        batch_size: Number of independent memories (for vectorized envs).
        capacity: Maximum number of memories to store.
        nkey_features: Dimension of keys.
        nvalue_features: Dimension of values.
        keys: Circular buffer storing keys.
        values: Circular buffer storing values.
    """

    batch_size: int
    capacity: int
    nkey_features: int
    nvalue_features: int
    keys: TensorCircularBuffer
    values: TensorCircularBuffer

    @classmethod
    def create(
        cls,
        batch_size: int,
        capacity: int,
        nkey_features: int,
        nvalue_features: int,
        device: torch.device = None,
    ) -> "FlatQKLearnedMemory":
        """Create a new empty episodic memory.

        Args:
            batch_size: Number of independent memories.
            capacity: Maximum number of entries.
            nkey_features: Dimension of keys.
            nvalue_features: Dimension of values.
            device: Device to create tensors on.

        Returns:
            New FlatQKLearnedMemory instance.
        """
        return cls(
            batch_size=batch_size,
            capacity=capacity,
            nkey_features=nkey_features,
            nvalue_features=nvalue_features,
            keys=TensorCircularBuffer.create(
                batch_size=batch_size,
                capacity=capacity,
                feature_shape=(nkey_features,),
                device=device,
            ),
            values=TensorCircularBuffer.create(
                batch_size=batch_size,
                capacity=capacity,
                feature_shape=(nvalue_features,),
                device=device,
            ),
        )

    def store(self, key: torch.Tensor, value: torch.Tensor) -> "FlatQKLearnedMemory":
        """Store a key-value pair in memory.

        Args:
            key: Key tensor of shape (batch_size, nkey_features).
            value: Value tensor of shape (batch_size, nvalue_features).

        Returns:
            New FlatQKLearnedMemory with the pair stored.
        """
        return FlatQKLearnedMemory(
            batch_size=self.batch_size,
            capacity=self.capacity,
            nkey_features=self.nkey_features,
            nvalue_features=self.nvalue_features,
            keys=self.keys.append(key),
            values=self.values.append(value),
        )

    def store_inplace(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store a key-value pair in-place.

        Args:
            key: Key tensor of shape (batch_size, nkey_features).
            value: Value tensor of shape (batch_size, nvalue_features).
        """
        self.keys.append_inplace(key)
        self.values.append_inplace(value)

    def learned_recall(
        self,
        query: torch.Tensor,
        key_fn: nn.Module,
        gate_fn: Optional[nn.Module] = None,
        n_per_key: int = 1,
    ) -> torch.Tensor:
        """Recall values using learned query/key transformations.

        Uses softmax attention over dot products of query and transformed keys.

        Args:
            query: Query tensor of shape (batch_size, nkey_features).
            key_fn: Module that transforms stored keys.
            gate_fn: Optional gating module (not used in main implementation).
            n_per_key: Number of values to return per query (unused, returns weighted sum).

        Returns:
            Retrieved values of shape (batch_size, 1, nvalue_features).
        """
        # Get buffers: (batch_size, capacity, features)
        key_buffer = self.keys.buffer
        values_buffer = self.values.buffer

        # Transform stored keys with the learned key function
        # Apply key_fn to each key in the buffer
        batch_size, capacity, _ = key_buffer.shape

        # Flatten for batch processing
        key_buffer_flat = key_buffer.reshape(-1, self.nkey_features)
        transformed_keys_flat = key_fn(key_buffer_flat)
        transformed_keys = transformed_keys_flat.reshape(batch_size, capacity, -1)

        # Compute dot products: (batch_size, capacity)
        # query: (batch_size, nkey_features)
        # transformed_keys: (batch_size, capacity, nkey_features)
        dot_products = torch.einsum("bf,bcf->bc", query, transformed_keys)

        # Softmax attention weights
        weights = F.softmax(dot_products, dim=-1)  # (batch_size, capacity)

        # Weighted sum of values: (batch_size, nvalue_features)
        weighted_values = torch.einsum("bc,bcf->bf", weights, values_buffer)

        # Add a dimension for consistency with n_per_key
        return weighted_values.unsqueeze(1)  # (batch_size, 1, nvalue_features)

    def recall(
        self,
        key: torch.Tensor,
        n_per_key: int = 1,
    ) -> torch.Tensor:
        """Simple recall using dot product similarity.

        Args:
            key: Query key of shape (batch_size, nkey_features).
            n_per_key: Number of values to return (uses softmax attention).

        Returns:
            Retrieved values of shape (batch_size, 1, nvalue_features).
        """
        key_buffer = self.keys.buffer
        values_buffer = self.values.buffer

        # Compute dot products for similarity
        dot_products = torch.einsum("bf,bcf->bc", key, key_buffer)
        weights = F.softmax(dot_products, dim=-1)

        weighted_values = torch.einsum("bc,bcf->bf", weights, values_buffer)
        return weighted_values.unsqueeze(1)

    def top_k_recall(
        self,
        key: torch.Tensor,
        n_per_key: int = 1,
    ) -> torch.Tensor:
        """Recall top-k most similar values.

        Args:
            key: Query key of shape (batch_size, nkey_features).
            n_per_key: Number of values to return.

        Returns:
            Top-k values of shape (batch_size, n_per_key, nvalue_features).
        """
        key_buffer = self.keys.buffer
        values_buffer = self.values.buffer

        # Compute dot products
        dot_products = torch.einsum("bf,bcf->bc", key, key_buffer)

        # Get top-k indices
        _, top_k_indices = torch.topk(dot_products, n_per_key, dim=-1)

        # Gather top-k values
        batch_indices = torch.arange(self.batch_size, device=values_buffer.device).unsqueeze(1)
        top_k_values = values_buffer[batch_indices, top_k_indices]

        return top_k_values

    def buffers(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the raw key and value buffers.

        Returns:
            Tuple of (keys_buffer, values_buffer).
        """
        return self.keys.buffer, self.values.buffer

    def clear(self) -> "FlatQKLearnedMemory":
        """Clear the memory.

        Returns:
            New empty FlatQKLearnedMemory.
        """
        return FlatQKLearnedMemory(
            batch_size=self.batch_size,
            capacity=self.capacity,
            nkey_features=self.nkey_features,
            nvalue_features=self.nvalue_features,
            keys=self.keys.clear(),
            values=self.values.clear(),
        )

    def to(self, device: torch.device) -> "FlatQKLearnedMemory":
        """Move memory to specified device.

        Args:
            device: Target device.

        Returns:
            New FlatQKLearnedMemory on the specified device.
        """
        return FlatQKLearnedMemory(
            batch_size=self.batch_size,
            capacity=self.capacity,
            nkey_features=self.nkey_features,
            nvalue_features=self.nvalue_features,
            keys=self.keys.to(device),
            values=self.values.to(device),
        )

    def clone(self) -> "FlatQKLearnedMemory":
        """Create a deep copy of the memory.

        Returns:
            New FlatQKLearnedMemory with cloned tensors.
        """
        return FlatQKLearnedMemory(
            batch_size=self.batch_size,
            capacity=self.capacity,
            nkey_features=self.nkey_features,
            nvalue_features=self.nvalue_features,
            keys=self.keys.clone(),
            values=self.values.clone(),
        )
