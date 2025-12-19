"""Tensor Circular Buffer for storing fixed-size history of tensors."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TensorCircularBuffer:
    """A circular buffer for storing tensors.

    This buffer stores a fixed number of tensors and overwrites the oldest
    entries when the capacity is reached. It supports batch operations.

    Attributes:
        batch_size: Number of independent buffers (for vectorized envs).
        capacity: Maximum number of entries per buffer.
        feature_shape: Shape of each stored tensor (excluding batch dim).
        buffer: The underlying storage tensor of shape (batch_size, capacity, *feature_shape).
        index: Current write position for each batch element.
        size: Current number of elements in each buffer.
    """

    batch_size: int
    capacity: int
    feature_shape: tuple[int, ...]
    buffer: torch.Tensor
    index: torch.Tensor  # (batch_size,) long tensor
    size: torch.Tensor   # (batch_size,) long tensor

    @classmethod
    def create(
        cls,
        batch_size: int,
        capacity: int,
        feature_shape: tuple[int, ...],
        device: torch.device = None,
    ) -> "TensorCircularBuffer":
        """Create a new empty circular buffer.

        Args:
            batch_size: Number of independent buffers.
            capacity: Maximum entries per buffer.
            feature_shape: Shape of stored tensors.
            device: Device to create tensors on.

        Returns:
            New TensorCircularBuffer instance.
        """
        shape = (batch_size, capacity, *feature_shape)
        return cls(
            batch_size=batch_size,
            capacity=capacity,
            feature_shape=feature_shape,
            buffer=torch.zeros(shape, device=device),
            index=torch.zeros(batch_size, dtype=torch.long, device=device),
            size=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def append(self, observation: torch.Tensor) -> "TensorCircularBuffer":
        """Append an observation to the buffer.

        Args:
            observation: Tensor of shape (batch_size, *feature_shape).

        Returns:
            New TensorCircularBuffer with the observation appended.
        """
        # Create new buffer with updated values
        new_buffer = self.buffer.clone()

        # Write observation at current index for each batch element
        batch_indices = torch.arange(self.batch_size, device=self.buffer.device)
        new_buffer[batch_indices, self.index] = observation

        # Update index and size
        new_index = (self.index + 1) % self.capacity
        new_size = torch.clamp(self.size + 1, max=self.capacity)

        return TensorCircularBuffer(
            batch_size=self.batch_size,
            capacity=self.capacity,
            feature_shape=self.feature_shape,
            buffer=new_buffer,
            index=new_index,
            size=new_size,
        )

    def append_inplace(self, observation: torch.Tensor) -> None:
        """Append an observation to the buffer in-place.

        This is more efficient when you don't need immutability.

        Args:
            observation: Tensor of shape (batch_size, *feature_shape).
        """
        batch_indices = torch.arange(self.batch_size, device=self.buffer.device)
        self.buffer[batch_indices, self.index] = observation
        self.index = (self.index + 1) % self.capacity
        self.size = torch.clamp(self.size + 1, max=self.capacity)

    def clear(self) -> "TensorCircularBuffer":
        """Clear the buffer.

        Returns:
            New empty TensorCircularBuffer.
        """
        return TensorCircularBuffer(
            batch_size=self.batch_size,
            capacity=self.capacity,
            feature_shape=self.feature_shape,
            buffer=torch.zeros_like(self.buffer),
            index=torch.zeros_like(self.index),
            size=torch.zeros_like(self.size),
        )

    def clear_inplace(self) -> None:
        """Clear the buffer in-place."""
        self.buffer.zero_()
        self.index.zero_()
        self.size.zero_()

    def get_all(self, batch_idx: int = 0) -> torch.Tensor:
        """Get all stored elements for a specific batch index.

        Args:
            batch_idx: Which batch element to retrieve.

        Returns:
            Tensor of shape (size, *feature_shape) containing stored elements.
        """
        return self.buffer[batch_idx, :self.size[batch_idx]]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get element at index for all batch elements.

        Supports negative indexing.

        Args:
            idx: Index to retrieve (can be negative).

        Returns:
            Tensor of shape (batch_size, *feature_shape).
        """
        if idx < 0:
            # Map negative index to correct position
            idx = (self.index + idx) % self.capacity
        else:
            idx = idx % self.capacity

        batch_indices = torch.arange(self.batch_size, device=self.buffer.device)
        return self.buffer[batch_indices, idx]

    def to(self, device: torch.device) -> "TensorCircularBuffer":
        """Move buffer to specified device.

        Args:
            device: Target device.

        Returns:
            New TensorCircularBuffer on the specified device.
        """
        return TensorCircularBuffer(
            batch_size=self.batch_size,
            capacity=self.capacity,
            feature_shape=self.feature_shape,
            buffer=self.buffer.to(device),
            index=self.index.to(device),
            size=self.size.to(device),
        )

    def clone(self) -> "TensorCircularBuffer":
        """Create a deep copy of the buffer.

        Returns:
            New TensorCircularBuffer with cloned tensors.
        """
        return TensorCircularBuffer(
            batch_size=self.batch_size,
            capacity=self.capacity,
            feature_shape=self.feature_shape,
            buffer=self.buffer.clone(),
            index=self.index.clone(),
            size=self.size.clone(),
        )
