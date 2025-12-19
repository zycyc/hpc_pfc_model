"""Utility functions for the PyTorch implementation."""

from typing import Any, Callable

import torch


def tree_map(fn: Callable, tree: Any) -> Any:
    """Apply a function to all tensors in a nested structure.

    Args:
        fn: Function to apply to each tensor.
        tree: Nested structure (dict, list, tuple, or tensor).

    Returns:
        New structure with fn applied to all tensors.
    """
    if isinstance(tree, torch.Tensor):
        return fn(tree)
    elif isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    elif isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(fn, x) for x in tree)
    else:
        return tree


def linear_schedule(
    init_value: float,
    end_value: float,
    transition_steps: int,
) -> Callable[[int], float]:
    """Create a linear schedule function.

    Args:
        init_value: Starting value.
        end_value: Final value.
        transition_steps: Number of steps over which to transition.

    Returns:
        A function that takes a step number and returns the scheduled value.
    """
    def schedule(step: int) -> float:
        if step >= transition_steps:
            return end_value
        fraction = step / max(transition_steps, 1)
        return init_value + fraction * (end_value - init_value)

    return schedule


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Soft update of target network parameters.

    target = (1 - tau) * target + tau * source

    Args:
        target: Target network to update.
        source: Source network to copy from.
        tau: Interpolation parameter (0 = no update, 1 = full copy).
    """
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)

        # Also update buffers
        for target_buf, source_buf in zip(target.buffers(), source.buffers()):
            target_buf.data.copy_(source_buf.data)


def clone_module(module: torch.nn.Module) -> torch.nn.Module:
    """Create a deep copy of a PyTorch module.

    Args:
        module: Module to clone.

    Returns:
        A new module with the same architecture and weights.
    """
    import copy
    return copy.deepcopy(module)
