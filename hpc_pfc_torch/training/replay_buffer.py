"""Replay buffer for experience replay."""

from dataclasses import dataclass, field
from typing import List, Optional
import random

import numpy as np
import torch


@dataclass
class Trajectory:
    """A trajectory of environment interactions.

    Attributes:
        obs: Observations of shape (seq_len, obs_dim).
        actions: Actions of shape (seq_len,).
        rewards: Rewards of shape (seq_len,).
        dones: Done flags of shape (seq_len,).
        trial_types: Trial types of shape (seq_len,).
        random_binaries: Random binary indicators of shape (seq_len,).
    """

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    trial_types: Optional[np.ndarray] = None
    random_binaries: Optional[np.ndarray] = None

    def to_tensors(self, device: torch.device = None) -> dict:
        """Convert to PyTorch tensors.

        Args:
            device: Device to put tensors on.

        Returns:
            Dictionary of tensors.
        """
        return {
            "obs": torch.from_numpy(self.obs).float().to(device),
            "actions": torch.from_numpy(self.actions).long().to(device),
            "rewards": torch.from_numpy(self.rewards).float().to(device),
            "dones": torch.from_numpy(self.dones).bool().to(device),
        }


class ReplayBuffer:
    """Simple replay buffer for storing trajectories.

    Args:
        capacity: Maximum number of trajectories to store.
        sequence_length: Length of sequences to sample.
    """

    def __init__(
        self,
        capacity: int = 1000,
        sequence_length: int = 100,
    ):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.buffer: List[Trajectory] = []
        self.priorities: List[float] = []
        self.position = 0

    def add(self, trajectory: Trajectory, priority: float = 1.0) -> None:
        """Add a trajectory to the buffer.

        Args:
            trajectory: Trajectory to add.
            priority: Priority weight for sampling.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(trajectory)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = trajectory
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def add_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        trial_types: Optional[np.ndarray] = None,
        random_binaries: Optional[np.ndarray] = None,
    ) -> None:
        """Add a batch of experience to the buffer.

        Args:
            obs: Observations of shape (batch_size, seq_len, obs_dim).
            actions: Actions of shape (batch_size, seq_len).
            rewards: Rewards of shape (batch_size, seq_len).
            dones: Done flags of shape (batch_size, seq_len).
            trial_types: Optional trial types.
            random_binaries: Optional random binary indicators.
        """
        batch_size = obs.shape[0]

        for i in range(batch_size):
            trajectory = Trajectory(
                obs=obs[i],
                actions=actions[i],
                rewards=rewards[i],
                dones=dones[i],
                trial_types=trial_types[i] if trial_types is not None else None,
                random_binaries=random_binaries[i] if random_binaries is not None else None,
            )

            # Priority based on whether trajectory has episode endings
            priority = 1.0 + float(dones[i].any())
            self.add(trajectory, priority)

    def sample(
        self,
        batch_size: int = 1,
        device: torch.device = None,
    ) -> Optional[dict]:
        """Sample a batch of trajectories.

        Args:
            batch_size: Number of trajectories to sample.
            device: Device to put tensors on.

        Returns:
            Dictionary of batched tensors, or None if buffer is empty.
        """
        if len(self.buffer) == 0:
            return None

        # Sample indices (uniform for now, can add prioritized sampling)
        indices = random.sample(
            range(len(self.buffer)),
            min(batch_size, len(self.buffer)),
        )

        # Gather trajectories
        trajectories = [self.buffer[i] for i in indices]

        # Batch and convert to tensors
        batch = {
            "obs": np.stack([t.obs for t in trajectories]),
            "actions": np.stack([t.actions for t in trajectories]),
            "rewards": np.stack([t.rewards for t in trajectories]),
            "dones": np.stack([t.dones for t in trajectories]),
        }

        # Convert to tensors
        return {
            "obs": torch.from_numpy(batch["obs"]).float().to(device),
            "actions": torch.from_numpy(batch["actions"]).long().to(device),
            "rewards": torch.from_numpy(batch["rewards"]).float().to(device),
            "dones": torch.from_numpy(batch["dones"]).bool().to(device),
        }

    def sample_sequence(
        self,
        device: torch.device = None,
    ) -> Optional[dict]:
        """Sample a single sequence from the buffer.

        For WaterMaze, we want to preserve order within episodes,
        so we sample the entire buffer as a sequence.

        Args:
            device: Device to put tensors on.

        Returns:
            Dictionary of tensors, or None if buffer is empty.
        """
        if len(self.buffer) == 0:
            return None

        # For WaterMaze, use the most recent trajectory
        # (preserves episode ordering)
        trajectory = self.buffer[-1]

        return trajectory.to_tensors(device)

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer = []
        self.priorities = []
        self.position = 0


class SequenceReplayBuffer:
    """Replay buffer that stores sequences and samples full episodes.

    Designed for WaterMaze where episode structure must be preserved.

    Args:
        capacity: Maximum number of steps to store.
        sequence_length: Length of full sequences.
    """

    def __init__(
        self,
        capacity: int = 2000,
        sequence_length: int = 200,
    ):
        self.capacity = capacity
        self.sequence_length = sequence_length

        # Circular buffer for observations
        self._obs: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._dones: List[bool] = []
        self._trial_types: List[int] = []
        self._random_binaries: List[int] = []

        self._position = 0
        self._size = 0

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        trial_type: int = 0,
        random_binary: int = 0,
    ) -> None:
        """Add a single transition.

        Args:
            obs: Observation.
            action: Action taken.
            reward: Reward received.
            done: Whether episode ended.
            trial_type: Type of trial.
            random_binary: Random binary indicator.
        """
        if len(self._obs) < self.capacity:
            self._obs.append(obs)
            self._actions.append(action)
            self._rewards.append(reward)
            self._dones.append(done)
            self._trial_types.append(trial_type)
            self._random_binaries.append(random_binary)
        else:
            self._obs[self._position] = obs
            self._actions[self._position] = action
            self._rewards[self._position] = reward
            self._dones[self._position] = done
            self._trial_types[self._position] = trial_type
            self._random_binaries[self._position] = random_binary

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        device: torch.device = None,
    ) -> Optional[dict]:
        """Sample the full sequence from the buffer.

        For WaterMaze, returns the entire buffer contents to preserve
        episode structure.

        Args:
            device: Device to put tensors on.

        Returns:
            Dictionary of tensors.
        """
        if self._size < self.sequence_length:
            return None

        # Get the most recent sequence_length steps
        if self._position >= self.sequence_length:
            start = self._position - self.sequence_length
            obs = np.stack(self._obs[start:self._position])
            actions = np.array(self._actions[start:self._position])
            rewards = np.array(self._rewards[start:self._position])
            dones = np.array(self._dones[start:self._position])
        else:
            # Handle wrap-around
            first_part = self.capacity - (self.sequence_length - self._position)
            obs = np.stack(
                self._obs[first_part:] + self._obs[:self._position]
            )
            actions = np.array(
                self._actions[first_part:] + self._actions[:self._position]
            )
            rewards = np.array(
                self._rewards[first_part:] + self._rewards[:self._position]
            )
            dones = np.array(
                self._dones[first_part:] + self._dones[:self._position]
            )

        return {
            "obs": torch.from_numpy(obs).float().unsqueeze(0).to(device),  # Add batch dim
            "actions": torch.from_numpy(actions).long().unsqueeze(0).to(device),
            "rewards": torch.from_numpy(rewards).float().unsqueeze(0).to(device),
            "dones": torch.from_numpy(dones).bool().unsqueeze(0).to(device),
        }

    def __len__(self) -> int:
        return self._size
