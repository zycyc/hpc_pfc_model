"""Episodic Water Maze environment with rule-based transformations."""

import functools
from enum import Enum, IntEnum
from typing import Any

import chex
import equinox as eqx
import jax
import jax_dataclasses as jdc
from gymnax.environments import EnvParams, EnvState, spaces
from gymnax.environments.environment import Environment
from jax import numpy as jnp
from jaxtyping import Int, Scalar, UInt32

from utils.transforms import annotate_transform


class _Entity(Enum):
    NONE = 0
    WALL = 1
    TARGET = 2
    AGENT = 3


class _Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class TrialType(IntEnum):
    EXPLORE = 0
    EXPLOIT = 1


VISIBLE_SUBGRID_SIDE_LENGTH = 3


class Coords2D(eqx.Module):
    """2D coordinates for agent/target positions."""

    x: Int[Scalar, "()"]
    y: Int[Scalar, "()"]

    def __init__(self, x: Scalar, y: Scalar):
        super().__init__()
        self.x = jnp.array(x, dtype=jnp.int32)
        self.y = jnp.array(y, dtype=jnp.int32)

    def __eq__(self, other):
        return jnp.logical_and(jnp.array_equal(self.x, other.x), jnp.array_equal(self.y, other.y))

    def __repr__(self):
        return f"Coords2D(x={self.x}, y={self.y})"


@jdc.pytree_dataclass
class State(EnvState):
    """Environment state.

    Attributes:
        current_explore_seed: Counter used to generate new mazes, incremented for new EXPLORE episodes.
        maze_index: Index of the maze currently in progress within a trial.
        trial_type: Internal episode type; "explore" or "exploit".
        random_binary: Rule indicator (0 or 1) for observation transformation.
    """

    tag: jnp.ndarray
    target: Coords2D
    agent: Coords2D
    distance_to_target: jnp.ndarray
    current_explore_seed: jnp.ndarray
    maze_index: jnp.ndarray
    trial_type: UInt32[Scalar, "()"]
    random_binary: jnp.ndarray


@jdc.pytree_dataclass
class Params(EnvParams):
    """Environment parameters.

    Attributes:
        dim: The length (and width) of the square grid.
        trials_per_episode: Number of new grids to try.
        exploit_probability: Probability of running an "exploit" episode.
        tag_length: Length of the "tag" or random vector used to ID a given grid.
        noise_rate: Probability of obscuring any given coordinate in the tag.
        noise_magnitude: Controls size of perturbation applied to obscured coordinates.
        forget_after: Number of most recently seen grids to consider for use in "exploit" episodes.
        target_visible: Whether target is visible when it is within observation box.
        max_steps_in_episode: Maximum episode length.
    """

    dim: jdc.Static[int] = 4
    trials_per_episode: jdc.Static[int] = 5
    exploit_probability: jdc.Static[float] = 0.5
    tag_length: jdc.Static[int] = 8
    noise_rate: jdc.Static[float] = 0.5
    noise_magnitude: jdc.Static[float] = 0.5
    forget_after: jdc.Static[int] = 200
    target_visible: jdc.Static[bool] = False
    max_steps_in_episode: jdc.Static[int] = 1000


class EpisodicWaterMaze(Environment[State, Params]):
    """Episodic Water Maze environment with two-rule variant.

    There are two different types of episodes in this environment:
    - Explore: The agent explores a new maze.
    - Exploit: The agent explores a previously seen maze.

    A maze is defined by a target location and tag (vector identifier of the target location).
    For each type of episode, the agent has a constant number of trials to reach the target.
    """

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: State,
        action: int,
        params: Params,
    ) -> tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, dict[Any, Any]]:
        """Take a step in the environment with auto-reset."""
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action, params)
        obs_re, state_re = self.reset_env(key_reset, params, state)
        # Auto-reset environment based on termination
        state = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), state_re, state_st)
        obs = jax.lax.select(done, obs_re, obs_st)
        return obs, state, reward, done, info

    def _generate_maze_from_seed(
        self, key: jnp.ndarray, seed: jnp.ndarray, params: Params, trial_type: jnp.ndarray, in_first_half: jnp.ndarray
    ) -> tuple[jnp.ndarray, Coords2D, Coords2D, jnp.ndarray, jnp.ndarray]:
        """Generate a maze from a seed with rule-based target location."""
        # Generate random_binary early
        random_binary_exploit = jax.random.randint(key, shape=(), minval=0, maxval=2)
        random_binary_explore = jnp.array(in_first_half).astype(jnp.int32)
        random_binary = jax.lax.select(trial_type == TrialType.EXPLOIT, random_binary_exploit, random_binary_explore)

        # Pick base target locations
        positions = jnp.arange(params.dim**2)
        target_key = jax.random.key(seed)
        target_position_A = jax.random.choice(target_key, positions, shape=())
        target_key_B, _ = jax.random.split(target_key)
        target_position_B = jax.random.choice(target_key_B, positions, shape=())

        target_position = jax.lax.select(random_binary == 1, target_position_B, target_position_A)

        # Remove chosen target pos from positions so agent won't also pick it
        positions = jnp.delete(positions, target_position, assume_unique_indices=True)

        # Define the target's row/col from final target_position
        target = Coords2D(y=target_position // params.dim, x=target_position % params.dim)

        # Pick agent position
        agent_position = jax.random.choice(key, positions, shape=())
        agent = Coords2D(y=agent_position // params.dim, x=agent_position % params.dim)

        # Generate the tag + noise
        tag = jax.random.normal(key=target_key, shape=(params.tag_length,))
        noise = params.noise_magnitude * jax.random.normal(key=key, shape=(params.tag_length,))
        noisy_tag = tag + noise
        uniform_samples = jax.random.uniform(key=key, shape=(params.tag_length,))
        tag = jnp.where(uniform_samples < params.noise_rate, noisy_tag, tag)

        distance_to_target = jnp.abs(agent.x - target.x) + jnp.abs(agent.y - target.y)

        return (
            tag,
            target,
            agent,
            distance_to_target,
            jnp.array([random_binary]),
        )

    def reset_env(self, key: jnp.ndarray, params: Params, state: State | None = None) -> tuple[jnp.ndarray, State]:
        """Reset the environment."""
        if state is None:
            tag, target, agent, distance_to_target, random_binary = self._generate_maze_from_seed(
                key, jnp.array(0), params, trial_type=jnp.array(TrialType.EXPLORE), in_first_half=jnp.array(1)
            )
            # Define the fixed transformation that we apply to the tag during exploration trials
            W0 = eqx.nn.Linear(params.tag_length, params.tag_length, use_bias=False, key=jax.random.PRNGKey(0))
            W1 = eqx.nn.Linear(params.tag_length, params.tag_length, use_bias=False, key=jax.random.PRNGKey(1))
            W = jax.lax.cond(random_binary.squeeze(), lambda _: W1, lambda _: W0, None)
            tag = W(tag)

            reset_state = State(
                time=0,
                tag=tag,
                target=target,
                agent=agent,
                distance_to_target=distance_to_target,
                current_explore_seed=jnp.array(0),
                maze_index=jnp.array(0),
                trial_type=jnp.array(TrialType.EXPLORE),
                random_binary=random_binary,
            )
        else:

            def split_indices(trials_per_episode: int):
                N = trials_per_episode
                indices = (N - 1 + jnp.arange(N)) % N
                n_first = (N + 1) // 2
                first_half = jax.lax.dynamic_slice_in_dim(indices, 0, n_first)
                second_half = jax.lax.dynamic_slice_in_dim(indices, n_first, N - n_first)
                return first_half, second_half

            first, second = split_indices(params.trials_per_episode)
            in_first_half = jnp.any(state.maze_index == first).astype(jnp.int32)

            new_maze_index = state.maze_index + 1
            generate_maze = new_maze_index == params.trials_per_episode

            current_explore_seed = state.current_explore_seed
            next_explore_seed = state.current_explore_seed + 1

            exploit_offset = jax.random.randint(key, (), minval=0, maxval=params.forget_after - 1)
            exploit_seed = (state.current_explore_seed - exploit_offset) % next_explore_seed

            trial_type = jax.lax.select(
                generate_maze,
                jax.lax.select(
                    jax.random.uniform(key) < params.exploit_probability, TrialType.EXPLOIT, TrialType.EXPLORE
                ),
                state.trial_type,
            )
            seed = jax.lax.select(
                generate_maze,
                next_explore_seed,
                current_explore_seed,
            ) * (trial_type == TrialType.EXPLORE) + exploit_seed * (trial_type == TrialType.EXPLOIT)
            tag, target, agent, distance_to_target, random_binary = self._generate_maze_from_seed(
                key, seed, params, trial_type=trial_type, in_first_half=in_first_half
            )

            # Define the fixed transformation that we apply to the tag during exploration trials
            W0 = eqx.nn.Linear(params.tag_length, params.tag_length, use_bias=False, key=jax.random.PRNGKey(0))
            W1 = eqx.nn.Linear(params.tag_length, params.tag_length, use_bias=False, key=jax.random.PRNGKey(1))
            W = jax.lax.cond(random_binary.squeeze(), lambda _: W1, lambda _: W0, None)
            tag = jax.lax.select(
                jnp.logical_or(trial_type == TrialType.EXPLORE, current_explore_seed < params.forget_after),
                W(tag),
                tag,
            )

            reset_state = State(
                time=0,
                tag=tag,
                target=target,
                agent=agent,
                distance_to_target=distance_to_target,
                maze_index=~generate_maze * new_maze_index,
                trial_type=trial_type,
                current_explore_seed=state.current_explore_seed + (trial_type == TrialType.EXPLORE) * generate_maze,
                random_binary=random_binary,
            )

        return self.get_obs(reset_state, params), reset_state

    def get_obs(self, state: State, params: Params) -> jnp.ndarray:
        """Get the observation from the current state.

        Returns:
            Observation = [visible_grid (9), tag (8), random_binary (1)] = 18 dims
        """
        visibility_radius = VISIBLE_SUBGRID_SIDE_LENGTH // 2
        grid = jnp.zeros((params.dim, params.dim))
        grid = grid.at[state.agent.y, state.agent.x].set(_Entity.AGENT.value)
        if params.target_visible:
            grid = grid.at[state.target.y, state.target.x].set(_Entity.TARGET.value)

        padded_grid = annotate_transform(
            jnp.pad,
            f"({params.dim, params.dim}) -> ({params.dim + 2 * visibility_radius, params.dim + 2 * visibility_radius})",
        )(
            grid,
            visibility_radius,
            mode="constant",
            constant_values=_Entity.WALL.value,
        )
        relative_grid = annotate_transform(
            jax.lax.dynamic_slice,
            f"(a, a) -> ({VISIBLE_SUBGRID_SIDE_LENGTH}, {VISIBLE_SUBGRID_SIDE_LENGTH})",
        )(
            padded_grid,
            start_indices=(state.agent.y, state.agent.x),
            slice_sizes=(VISIBLE_SUBGRID_SIDE_LENGTH, VISIBLE_SUBGRID_SIDE_LENGTH),
        )

        return annotate_transform(
            lambda x: jnp.concatenate((jnp.ravel(x), state.tag, state.random_binary)), "(a, a) -> (b,)"
        )(relative_grid)

    def step_env(
        self, key: jnp.ndarray, state: State, action: chex.Array | int | float, params: Params
    ) -> tuple[jnp.ndarray, State, jnp.ndarray, jnp.ndarray, dict]:
        """Internal step method called by step()."""
        current_y = state.agent.y
        current_x = state.agent.x
        agent_next = [
            jnp.array([jnp.maximum(jnp.array(0), current_y - 1), current_x]),  # up
            jnp.array([jnp.minimum(jnp.array(params.dim - 1), current_y + 1), current_x]),  # down
            jnp.array([current_y, jnp.maximum(jnp.array(0), current_x - 1)]),  # left
            jnp.array([current_y, jnp.minimum(jnp.array(params.dim - 1), current_x + 1)]),  # right
        ]

        agent_array = jax.lax.select_n(action, *agent_next)
        agent = Coords2D(y=agent_array[0], x=agent_array[1])
        done = jnp.array(agent == state.target)
        reward = jax.lax.select(done, done.astype(jnp.float32), -1 / params.max_steps_in_episode)

        # truncated if reached max step count
        truncated = jnp.array(state.time + 1 == params.max_steps_in_episode)

        state = State(
            time=state.time + 1,
            tag=state.tag,
            target=state.target,
            agent=agent,
            distance_to_target=state.distance_to_target,
            maze_index=state.maze_index,
            trial_type=state.trial_type,
            current_explore_seed=state.current_explore_seed,
            random_binary=state.random_binary,
        )

        done = jnp.logical_or(done, truncated)
        excess_steps = jnp.where(done, state.time - state.distance_to_target, -1)
        return (
            self.get_obs(state, params),
            state,
            reward,
            done,
            {
                "excess_steps": excess_steps,
                "trial_type": state.trial_type,
                "random_binary": state.random_binary,
            },
        )

    @property
    def num_actions(self) -> int:
        return len(_Action)

    def action_space(self, params: Params) -> spaces.Space:
        """Gymnax action_space specification method."""
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: Params) -> spaces.Space:
        """Gymnax observation_space specification method."""
        dim = (VISIBLE_SUBGRID_SIDE_LENGTH**2) + params.tag_length + 1
        return spaces.Box(
            low=-10.0 - 10 * params.noise_magnitude, high=10.0 + 10.0 * params.noise_magnitude, shape=(dim,)
        )
