"""Agent state containers for HPC-PFC Model."""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from memory.episodic_memory import FlatQKLearnedMemory
    from memory.circular_buffer import TensorCircularBuffer


@dataclass
class StepState:
    """State that is updated on each step.

    Attributes:
        episodic_memory: Key-value memory for storing past experiences.
        hidden: Current RNN hidden state of shape (num_envs, reservoir_size).
        init_hidden: Hidden state at the start of the episode.
        t: Current timestep counter.
        last_episode: Last episode index.
        obs_buffer: Buffer for storing observation features.
        h_buffer: Buffer for storing hidden states.
        online_density_buffer: Buffer for online density estimates.
    """

    episodic_memory: "FlatQKLearnedMemory"
    hidden: torch.Tensor
    init_hidden: torch.Tensor
    t: int = 0
    last_episode: int = 0
    obs_buffer: Optional["TensorCircularBuffer"] = None
    h_buffer: Optional["TensorCircularBuffer"] = None
    obs_buffer_exploit: Optional["TensorCircularBuffer"] = None
    online_density_buffer: Optional["TensorCircularBuffer"] = None

    def to(self, device: torch.device) -> "StepState":
        """Move state to device."""
        return StepState(
            episodic_memory=self.episodic_memory.to(device),
            hidden=self.hidden.to(device),
            init_hidden=self.init_hidden.to(device),
            t=self.t,
            last_episode=self.last_episode,
            obs_buffer=self.obs_buffer.to(device) if self.obs_buffer else None,
            h_buffer=self.h_buffer.to(device) if self.h_buffer else None,
            obs_buffer_exploit=self.obs_buffer_exploit.to(device) if self.obs_buffer_exploit else None,
            online_density_buffer=self.online_density_buffer.to(device) if self.online_density_buffer else None,
        )

    def clone(self) -> "StepState":
        """Create a deep copy."""
        return StepState(
            episodic_memory=self.episodic_memory.clone(),
            hidden=self.hidden.clone(),
            init_hidden=self.init_hidden.clone(),
            t=self.t,
            last_episode=self.last_episode,
            obs_buffer=self.obs_buffer.clone() if self.obs_buffer else None,
            h_buffer=self.h_buffer.clone() if self.h_buffer else None,
            obs_buffer_exploit=self.obs_buffer_exploit.clone() if self.obs_buffer_exploit else None,
            online_density_buffer=self.online_density_buffer.clone() if self.online_density_buffer else None,
        )


@dataclass
class OptState:
    """Optimizer state.

    Attributes:
        target_update_count: Counter for target network updates.
    """

    target_update_count: int = 0


@dataclass
class ExperienceState:
    """Experience replay state.

    Attributes:
        replay_buffer: List of trajectories for replay.
        priorities: Priority weights for each trajectory.
    """

    # Simplified: just store trajectories as lists
    # In production, this would be a more sophisticated prioritized buffer
    trajectories: list = field(default_factory=list)
    priorities: list = field(default_factory=list)
    max_capacity: int = 1000

    def add(self, trajectory: dict, priority: float = 1.0) -> None:
        """Add a trajectory to the buffer."""
        if len(self.trajectories) >= self.max_capacity:
            # Remove oldest
            self.trajectories.pop(0)
            self.priorities.pop(0)
        self.trajectories.append(trajectory)
        self.priorities.append(priority)

    def sample(self, batch_size: int = 1) -> Optional[list]:
        """Sample trajectories (simplified uniform sampling)."""
        if not self.trajectories:
            return None
        import random

        indices = random.sample(range(len(self.trajectories)), min(batch_size, len(self.trajectories)))
        return [self.trajectories[i] for i in indices]

    def __len__(self) -> int:
        return len(self.trajectories)


@dataclass
class AgentState:
    """Complete agent state.

    Attributes:
        step: Per-step state (hidden states, episodic memory, etc.).
        opt: Optimizer state.
        experience: Experience replay state.
        inference: Whether in inference mode (no training).
    """

    step: StepState
    opt: OptState
    experience: ExperienceState
    inference: bool = False

    def to(self, device: torch.device) -> "AgentState":
        """Move state to device."""
        return AgentState(
            step=self.step.to(device),
            opt=self.opt,
            experience=self.experience,
            inference=self.inference,
        )
