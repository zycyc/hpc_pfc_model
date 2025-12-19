"""PRNG key generator utility."""

from collections.abc import Generator

from jax.numpy import ndarray
from jax.random import split as split_key


def keygen(start_key: ndarray) -> Generator[ndarray, None, None]:
    """Generator that yields new PRNG keys from a starting key.

    Args:
        start_key: Initial JAX PRNG key.

    Yields:
        New PRNG subkeys.
    """
    key = start_key
    while True:
        key, subkey = split_key(key)
        yield subkey
