"""Episodic Water Maze environment using Gymnasium API.

A gridworld navigation environment with two phases:
- EXPLORE: Agent explores a new maze, learns target location
- EXPLOIT: Agent navigates to a previously seen target location
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class TrialType(IntEnum):
    """Trial types in the water maze."""
    EXPLORE = 0
    EXPLOIT = 1


class Action(IntEnum):
    """Available actions."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


@dataclass
class Params:
    """Environment parameters.

    Attributes:
        dim: Grid size (dim x dim).
        trials_per_episode: Number of trials per episode.
        exploit_probability: Probability of exploit trial.
        tag_length: Length of tag vector.
        noise_rate: Probability of noising tag.
        noise_magnitude: Noise scale.
        forget_after: Memory window for exploit.
        target_visible: Whether target is visible.
        max_steps_in_episode: Max steps per trial.
    """
    dim: int = 4
    trials_per_episode: int = 5
    exploit_probability: float = 0.5
    tag_length: int = 8
    noise_rate: float = 0.5
    noise_magnitude: float = 0.5
    forget_after: int = 200
    target_visible: bool = False
    max_steps_in_episode: int = 1000


class EpisodicWaterMaze(gym.Env):
    """Episodic Water Maze environment.

    A gridworld where the agent must navigate to a target location.
    The target location is identified by a "tag" (random vector).

    Two trial types:
    - EXPLORE: Agent explores a new maze configuration
    - EXPLOIT: Agent must navigate to a previously seen target

    The environment applies rule-based transformations to tags based
    on a random binary indicator.

    Args:
        params: Environment parameters.
        seed: Random seed.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        params: Optional[Params] = None,
        seed: int = 0,
    ):
        super().__init__()

        self.params = params or Params()
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(4)

        # Observation: visible_subgrid (9) + tag (tag_length) + random_binary (1)
        obs_dim = 9 + self.params.tag_length + 1
        self.observation_space = gym.spaces.Box(
            low=-10.0 - 10 * self.params.noise_magnitude,
            high=10.0 + 10.0 * self.params.noise_magnitude,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # State variables
        self._agent_pos = np.array([0, 0])
        self._target_pos = np.array([0, 0])
        self._tag = np.zeros(self.params.tag_length)
        self._random_binary = 0
        self._trial_type = TrialType.EXPLORE
        self._maze_index = 0
        self._current_explore_seed = 0
        self._step_count = 0
        self._initial_distance = 0

        # History of seen mazes for exploit trials
        self._seen_seeds: list = []

        # Create fixed tag transformations (rule-based)
        torch.manual_seed(0)
        self._W0 = nn.Linear(self.params.tag_length, self.params.tag_length, bias=False)
        torch.manual_seed(1)
        self._W1 = nn.Linear(self.params.tag_length, self.params.tag_length, bias=False)

        # Freeze transformation weights
        for p in self._W0.parameters():
            p.requires_grad = False
        for p in self._W1.parameters():
            p.requires_grad = False

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Args:
            seed: Optional random seed.
            options: Optional reset options.

        Returns:
            Tuple of (observation, info).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Generate initial maze
        self._generate_maze(
            seed=0,
            trial_type=TrialType.EXPLORE,
            in_first_half=True,
        )

        self._step_count = 0
        self._maze_index = 0
        self._current_explore_seed = 0
        self._trial_type = TrialType.EXPLORE
        self._seen_seeds = [0]

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action to take (0-3).

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        # Move agent
        old_pos = self._agent_pos.copy()

        if action == Action.UP:
            self._agent_pos[0] = max(0, self._agent_pos[0] - 1)
        elif action == Action.DOWN:
            self._agent_pos[0] = min(self.params.dim - 1, self._agent_pos[0] + 1)
        elif action == Action.LEFT:
            self._agent_pos[1] = max(0, self._agent_pos[1] - 1)
        elif action == Action.RIGHT:
            self._agent_pos[1] = min(self.params.dim - 1, self._agent_pos[1] + 1)

        self._step_count += 1

        # Check if reached target
        reached_target = np.array_equal(self._agent_pos, self._target_pos)

        # Compute reward
        if reached_target:
            reward = 1.0
        else:
            reward = -1.0 / self.params.max_steps_in_episode

        # Check termination
        truncated = self._step_count >= self.params.max_steps_in_episode
        terminated = reached_target

        info = self._get_info()

        # Handle episode reset (new trial within episode)
        if terminated or truncated:
            self._start_new_trial()

        obs = self._get_obs()

        return obs, reward, terminated, truncated, info

    def _start_new_trial(self):
        """Start a new trial within the episode."""
        self._maze_index += 1
        self._step_count = 0

        # Check if we need to generate a new maze
        if self._maze_index >= self.params.trials_per_episode:
            self._maze_index = 0

            # Decide trial type
            if self._rng.random() < self.params.exploit_probability and len(self._seen_seeds) > 0:
                self._trial_type = TrialType.EXPLOIT
            else:
                self._trial_type = TrialType.EXPLORE
                self._current_explore_seed += 1
                self._seen_seeds.append(self._current_explore_seed)

                # Limit memory
                if len(self._seen_seeds) > self.params.forget_after:
                    self._seen_seeds = self._seen_seeds[-self.params.forget_after:]

        # Determine which half of trials we're in
        n_first_half = (self.params.trials_per_episode + 1) // 2
        in_first_half = self._maze_index < n_first_half

        # Generate maze
        if self._trial_type == TrialType.EXPLORE:
            seed = self._current_explore_seed
        else:
            # Pick a random seed from seen mazes
            offset = self._rng.integers(0, min(len(self._seen_seeds), self.params.forget_after))
            seed = self._seen_seeds[-(offset + 1)]

        self._generate_maze(seed, self._trial_type, in_first_half)

    def _generate_maze(
        self,
        seed: int,
        trial_type: TrialType,
        in_first_half: bool,
    ):
        """Generate a new maze configuration.

        Args:
            seed: Seed for maze generation.
            trial_type: Type of trial.
            in_first_half: Whether we're in the first half of trials.
        """
        rng = np.random.default_rng(seed)

        # Determine random binary
        if trial_type == TrialType.EXPLOIT:
            self._random_binary = self._rng.integers(0, 2)
        else:
            self._random_binary = int(in_first_half)

        # Generate target positions for both rules
        # Match JAX behavior: use same seed for A, split for B
        positions = np.arange(self.params.dim ** 2)
        target_pos_A = rng.choice(positions)

        # Simulate JAX's key splitting by using next random value from same rng
        # then creating a new generator from that
        split_seed = rng.integers(0, 2**31)
        rng_B = np.random.default_rng(split_seed)
        target_pos_B = rng_B.choice(positions)

        # Select target based on random binary
        target_pos = target_pos_B if self._random_binary else target_pos_A
        self._target_pos = np.array([
            target_pos // self.params.dim,
            target_pos % self.params.dim,
        ])

        # Generate agent position (not at target)
        available = [p for p in positions if p != target_pos]
        agent_pos = self._rng.choice(available)
        self._agent_pos = np.array([
            agent_pos // self.params.dim,
            agent_pos % self.params.dim,
        ])

        # Generate tag
        self._tag = rng.standard_normal(self.params.tag_length).astype(np.float32)

        # Add noise
        if self.params.noise_rate > 0:
            noise = self.params.noise_magnitude * self._rng.standard_normal(self.params.tag_length)
            noise_mask = self._rng.random(self.params.tag_length) < self.params.noise_rate
            self._tag = np.where(noise_mask, self._tag + noise, self._tag)

        # Apply transformation based on random binary
        tag_tensor = torch.from_numpy(self._tag).float()
        if self._random_binary:
            transformed = self._W1(tag_tensor)
        else:
            transformed = self._W0(tag_tensor)
        self._tag = transformed.detach().numpy()

        # Compute initial distance
        self._initial_distance = abs(self._agent_pos[0] - self._target_pos[0]) + \
                                  abs(self._agent_pos[1] - self._target_pos[1])

    def _get_obs(self) -> np.ndarray:
        """Get current observation.

        Returns:
            Observation array.
        """
        # Build visible subgrid (3x3 around agent)
        grid = np.zeros((self.params.dim, self.params.dim))

        if self.params.target_visible:
            grid[self._target_pos[0], self._target_pos[1]] = 2  # Target

        # Agent position
        grid[self._agent_pos[0], self._agent_pos[1]] = 3

        # Pad grid with walls
        padded = np.pad(grid, 1, mode="constant", constant_values=1)

        # Extract 3x3 subgrid
        y, x = self._agent_pos + 1  # Account for padding
        subgrid = padded[y-1:y+2, x-1:x+2].flatten()

        # Concatenate observation
        obs = np.concatenate([
            subgrid,
            self._tag,
            np.array([self._random_binary], dtype=np.float32),
        ]).astype(np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary.

        Returns:
            Info dictionary.
        """
        excess_steps = self._step_count - self._initial_distance if self._step_count > 0 else -1

        return {
            "trial_type": self._trial_type,
            "random_binary": self._random_binary,
            "maze_index": self._maze_index,
            "excess_steps": excess_steps,
            "distance_to_target": abs(self._agent_pos[0] - self._target_pos[0]) + \
                                   abs(self._agent_pos[1] - self._target_pos[1]),
        }

    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb()

    def _render_human(self):
        """Render to console."""
        grid = np.full((self.params.dim, self.params.dim), ".")
        grid[self._target_pos[0], self._target_pos[1]] = "T"
        grid[self._agent_pos[0], self._agent_pos[1]] = "A"

        print(f"Trial: {self._trial_type.name}, Binary: {self._random_binary}")
        for row in grid:
            print(" ".join(row))
        print()

    def _render_rgb(self) -> np.ndarray:
        """Render to RGB array."""
        cell_size = 32
        img = np.zeros((self.params.dim * cell_size, self.params.dim * cell_size, 3), dtype=np.uint8)

        # Background
        img[:, :] = [200, 200, 200]

        # Target (green)
        ty, tx = self._target_pos * cell_size
        img[ty:ty+cell_size, tx:tx+cell_size] = [0, 200, 0]

        # Agent (blue)
        ay, ax = self._agent_pos * cell_size
        img[ay:ay+cell_size, ax:ax+cell_size] = [0, 0, 200]

        return img
