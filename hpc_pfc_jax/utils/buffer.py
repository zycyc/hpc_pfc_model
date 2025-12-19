"""Tensor circular buffer implementation."""

import equinox as eqx
import jax
import jax.numpy as jnp


class TensorCircularBuffer(eqx.Module):
    """A circular buffer for storing tensors with JAX compatibility."""

    capacity: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    feature_shape: tuple[int, ...] = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    size: jnp.ndarray
    index: jnp.ndarray
    buffer: jnp.ndarray

    def __init__(
        self,
        *,
        batch_size: int,
        capacity: int,
        feature_shape: tuple[int, ...],
        _size: jnp.ndarray | None = None,
        _index: jnp.ndarray | None = None,
        _buffer: jnp.ndarray | None = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.capacity = capacity
        self.feature_shape = feature_shape

        if _size is None:  # Path we follow when first initializing a tensor circular buffer
            assert _index is None
            self.size = jnp.zeros((batch_size,), dtype=jnp.int32)
            self.index = jnp.zeros((batch_size,), dtype=jnp.int32)
        else:
            assert _index is not None
            self.size = _size  # Useful for vmap, in which setting self.size is not aware of the batch size
            self.index = _index

        self.shape = (self.batch_size, self.capacity, *self.feature_shape)
        if _buffer is None:
            self.buffer = jnp.zeros(self.shape, dtype=jnp.float32)
        else:
            self.buffer = _buffer

    def clear(self) -> "TensorCircularBuffer":
        return TensorCircularBuffer(
            batch_size=self.batch_size,
            capacity=self.capacity,
            feature_shape=self.feature_shape,
            _size=jnp.zeros_like(self.size),
            _index=jnp.zeros_like(self.index),
            _buffer=jnp.zeros_like(self.buffer),
        )

    def _append(
        self, buffer: jnp.ndarray, index: jnp.ndarray, size: jnp.ndarray, observation: jnp.ndarray
    ) -> "TensorCircularBuffer":
        if observation.shape != self.feature_shape:
            raise ValueError(f"The shape of 'observation' must be ({self.feature_shape})")

        buffer = buffer.at[index].set(observation)
        index = (index + 1) % self.capacity
        size = jax.lax.min(size + 1, self.capacity)
        return TensorCircularBuffer(
            batch_size=self.batch_size,
            capacity=self.capacity,
            feature_shape=self.feature_shape,
            _size=size,
            _index=index,
            _buffer=buffer,
        )

    def append(self, observation: jnp.ndarray) -> "TensorCircularBuffer":
        """Append observation to buffer. Handles batched vmapping automatically."""
        if self.buffer.shape == self.shape:
            append_fn = jax.vmap(self._append)
        else:
            # We go down this path when we are vmapping across TensorCircularBuffer
            append_fn = self._append
        return append_fn(self.buffer, self.index, self.size, observation)

    def _append_sequence(
        self, buffer: jnp.ndarray, index: jnp.ndarray, size: jnp.ndarray, observation_sequence: jnp.ndarray
    ) -> "TensorCircularBuffer":
        if len(observation_sequence.shape) != len(self.feature_shape) + 1:
            raise ValueError(
                "The shape of 'observation_sequence' must be {shape}".format(shape=("seq_length",) + self.feature_shape)
            )

        if observation_sequence.shape[0] > self.capacity:
            raise ValueError(
                f"The length of 'observation_sequence' must be less than or equal to the capacity. "
                f"Got sequence length {observation_sequence.shape[0]} and capacity {self.capacity}"
            )

        indices = (jnp.arange(observation_sequence.shape[0]) + index) % self.capacity
        buffer = buffer.at[indices].set(observation_sequence)
        index = (indices[-1] + 1) % self.capacity
        size = jax.lax.min(size + len(observation_sequence), self.capacity)
        return TensorCircularBuffer(
            batch_size=self.batch_size,
            capacity=self.capacity,
            feature_shape=self.feature_shape,
            _size=size,
            _index=index,
            _buffer=buffer,
        )

    def append_sequence(self, observation_sequence: jnp.ndarray) -> "TensorCircularBuffer":
        """Append a sequence of observations. Handles batched vmapping automatically."""
        if len(observation_sequence.shape) == len(self.shape):
            if observation_sequence.shape[0] != self.batch_size:
                raise ValueError("The first dimension of 'observation_sequence' must be equal to the batch size")
            append_fn = jax.vmap(self._append_sequence)
        else:
            # We go down this path when we are vmapping across TensorCircularBuffer
            if observation_sequence.shape[1:] != self.feature_shape:
                raise ValueError(
                    "The shape of 'observation_sequence' must be {shape}".format(
                        shape=("seq_length",) + self.feature_shape
                    )
                )
            append_fn = self._append_sequence
        return append_fn(self.buffer, self.index, self.size, observation_sequence)

    def __len__(self) -> int:
        return self.size[0].item()

    def _getitem(self, buffer: jnp.ndarray, idx: jnp.ndarray | int) -> jnp.ndarray:
        return buffer[idx]

    def __getitem__(self, idx: jnp.ndarray | int) -> jnp.ndarray:
        """Index buffer. Handles batched indices via vmapping automatically."""
        idx_is_ndarray = isinstance(idx, jnp.ndarray)
        is_vmapped = self.buffer.shape != self.shape

        if not idx_is_ndarray:
            idx = jnp.array(idx)

        # If we pass in a negative integer, we map it back to the correct positive value
        if is_vmapped:
            idx = jnp.where(idx < 0, idx + self.size, idx)
        else:
            idx = jnp.where(idx < 0, idx + self.size[0], idx)

        in_axes = (0 if not is_vmapped else None, 0 if idx_is_ndarray else None)
        if in_axes == (None, None):
            return self._getitem(self.buffer, idx)  # Vmap requires at least one axis not 'None'
        else:
            return jax.vmap(self._getitem, in_axes=in_axes)(self.buffer, idx)
