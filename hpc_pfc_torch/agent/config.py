"""Configuration for HPC-PFC Model agent."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EnvironmentType(Enum):
    """Supported environment types."""

    WATERMAZE = 0
    DISCRETE_MAZE = 1
    CRAFTER = 2


@dataclass(frozen=True)
class Config:
    """Configuration for HPC-PFC Model agent.

    Attributes:
        env_type: Type of environment.
        total_samples: Max samples for density buffers.
        replay_sequence_length: Sequence length for replay sampling.
        replay_buffer_capacity: Max replay buffer size.
        epsilon_transition_steps: Steps for epsilon decay.
        memory_capacity: Episodic memory capacity.
        num_actions: Action space size.
        store_mem_probability: P(store) for non-terminal states.
        n_per_key: Memories to recall per query.
        discount: Gamma for Q-learning.
        filter_loss_coef: Regularization for filter network.
        optimizer_name: Name of optimizer (adam, sgd, etc.).
        optimizer_kwargs: Keyword arguments for optimizer.
        num_envs: Number of parallel environments.
        num_off_policy_updates_per_cycle: Updates per training cycle.
        target_update_interval: Steps between target updates.
        target_update_step_size: Polyak averaging tau.
        epsilon_start: Initial exploration rate.
        epsilon_end: Final exploration rate.
        density_k_nearest_neighbours: K for density estimation.
        env_reward_coef: Weight for environment reward (1.0 = no exploration bonus).
        use_hierarchical_memory: Whether to use hierarchical memory.
        hierarchical_memory_k: K for hierarchical memory clustering.
        hierarchical_memory_depth: Depth of hierarchical memory tree.
        hierarchical_memory_auto_refit_interval: Interval for refitting.
        store_obs_probability: P(store observation in feature buffer).
        use_online_density_in_reward: Use real-time density in reward.
        use_offline_density_in_reward: Use replay density in reward.
    """

    env_type: EnvironmentType
    total_samples: int
    replay_sequence_length: int
    replay_buffer_capacity: int
    epsilon_transition_steps: int
    memory_capacity: int
    num_actions: int

    # Memory storage
    store_mem_probability: float = 0.05
    n_per_key: int = 1

    # Learning
    discount: float = 0.9
    filter_loss_coef: float = 1e-5
    optimizer_name: str = "adam"
    optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 3e-5})  # Match JAX default

    # Training loop
    num_envs: int = 1
    num_off_policy_updates_per_cycle: int = 1
    target_update_interval: int = 1
    target_update_step_size: float = 1.0

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    density_k_nearest_neighbours: int = 15
    env_reward_coef: float = 1.0

    # Hierarchical memory (optional)
    use_hierarchical_memory: bool = False
    hierarchical_memory_k: Optional[int] = None
    hierarchical_memory_depth: Optional[int] = None
    hierarchical_memory_auto_refit_interval: Optional[int] = None

    # Density-based exploration
    store_obs_probability: float = 1.0
    use_online_density_in_reward: bool = False
    use_offline_density_in_reward: bool = False

    def __post_init__(self):
        """Validate configuration values."""
        if not (0 <= self.store_mem_probability <= 1):
            raise ValueError("store_mem_probability must be between 0 and 1")
        if not (0 <= self.store_obs_probability <= 1):
            raise ValueError("store_obs_probability must be between 0 and 1")
        if self.density_k_nearest_neighbours <= 0:
            raise ValueError("density_k_nearest_neighbours must be a positive integer")
        if not (0 <= self.epsilon_start <= 1):
            raise ValueError("epsilon_start must be between 0 and 1")
        if not (0 <= self.epsilon_end <= 1):
            raise ValueError("epsilon_end must be between 0 and 1")
        if not (0 <= self.target_update_step_size <= 1):
            raise ValueError("target_update_step_size must be between 0 and 1")
        if not (0 <= self.env_reward_coef <= 1):
            raise ValueError("env_reward_coef must be between 0 and 1")

    @property
    def exploration_coef(self) -> float:
        """Coefficient for exploration bonus (complement of env_reward_coef)."""
        return 1 - self.env_reward_coef

    def epsilon_schedule(self, step: int) -> float:
        """Get epsilon value at given step.

        Args:
            step: Current training step.

        Returns:
            Epsilon value (linear interpolation).
        """
        if step >= self.epsilon_transition_steps:
            return self.epsilon_end
        fraction = step / max(self.epsilon_transition_steps, 1)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
