"""HPC-PFC Model RL Agent with reservoir computing and episodic memory.

This is the learned version of the HPC-PFC Model, which combines:
- Reservoir Computing (Echo State Networks) for temporal integration
- Episodic Memory with learned query-key transformations
- Multi-head Attention for combining working memory and episodic memory
- Double DQN for off-policy Q-learning
"""

import dataclasses
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, NamedTuple

import equinox as eqx
import equinox.nn as nn
import flashbax as fbx
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import optax
from equinox import filter_vmap
from flashbax.buffers.sum_tree import SumTreeState
from gymnax.environments import EnvState
from gymnax.environments.environment import Environment as GymnaxEnv
from gymnax.environments.environment import EnvParams
from gymnax.environments.spaces import Space
from jax_dataclasses import pytree_dataclass
from jaxtyping import Int, PRNGKeyArray, PyTree, Scalar
from rlax import double_q_learning

# Local imports
from env.water_maze import State as episodic_water_maze_state
from env.water_maze import TrialType
from memory.attention import Attention
from memory.base import EpisodicMemory
from memory.flat_qk import FlatQKLearnedMemory
from reservoir.base import Reservoir
from reservoir.local import ReservoirLocalConnectivity
from utils.jax_utils import filter_cond, filter_scan
from utils.prng import keygen
from utils.rl import distance_to_kth_nearest_neighbour, filter_incremental_update

# =============================================================================
# Inlined EARL types (from earl/core.py)
# =============================================================================


@pytree_dataclass
class EnvStep:
    """The result of taking an action in an environment.

    Note that it may be a batch of timesteps, in which case
    all of the members will have an additional leading batch dimension.
    """

    new_episode: jnp.ndarray  # True if first timestep in an episode
    obs: PyTree
    prev_action: jnp.ndarray  # the action taken in the previous timestep
    reward: jnp.ndarray


class AgentStep(NamedTuple):
    """A batch of actions and updated hidden state."""

    action: jnp.ndarray
    state: Any  # StepState


@dataclass(frozen=True)
class EnvInfo:
    """Environment metadata."""

    num_envs: int
    observation_space: Space
    action_space: Space
    name: str


def env_info_from_gymnax(env: GymnaxEnv, params: EnvParams, num_envs: int) -> EnvInfo:
    """Create EnvInfo from a Gymnax environment."""
    return EnvInfo(num_envs, env.observation_space(params), env.action_space(params), env.name)


# =============================================================================
# Model Components
# =============================================================================


@jdc.pytree_dataclass
class RecallDatum:
    """Data stored in replay buffer."""

    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    new_episode: jnp.ndarray


class EnvironmentType(Enum):
    """Supported environment types."""

    WATERMAZE = 0
    DISCRETE_MAZE = 1
    CRAFTER = 2


class MoELayer(eqx.Module):
    """Mixture of Experts layer with 2 experts."""

    gate: nn.MLP
    expert0: nn.MLP
    expert1: nn.MLP

    def __call__(self, x):
        gate_weights = jax.nn.softmax(self.gate(x))
        expert0_out = self.expert0(x)
        expert1_out = self.expert1(x)
        return gate_weights[0] * expert0_out + gate_weights[1] * expert1_out


class MLPLayer(eqx.Module):
    """Wrapper for MLP to make it compatible with MoE interface."""

    layer: nn.MLP

    def __call__(self, x):
        return self.layer(x)


class RecurrentNetwork(eqx.Module):
    """Main neural network module for the HPC-PFC Model.

    Contains:
    - Preprocessing network
    - Reservoir RNN
    - Filter network for input gating
    - PFC networks for query/key transformations (MLP and MoE variants)
    - Attention module
    - Q-value classifier
    """

    action_size: int = eqx.field(static=True)
    fc_mem: nn.Linear
    filter: nn.Sequential
    hd_bias: jnp.ndarray
    num_feature_dims_mem_keys: int = eqx.field(static=True)
    reservoir_size: int = eqx.field(static=True)

    _classifier: nn.Sequential
    data_size: int = eqx.field(static=True)
    _preprocess: nn.Sequential
    _layer_size: int = eqx.field(static=True)
    mem_size: int = eqx.field(static=True)
    _rnn: Reservoir
    _rnn_input_size: int = eqx.field(static=True)

    _attn1: Attention
    _emb1: nn.MLP

    # MoE variants for PFC
    _pfc_q_moe: MoELayer
    _pfc_k_moe: MoELayer

    # MLP variants for PFC
    _pfc_q_mlp: MLPLayer
    _pfc_k_mlp: MLPLayer

    # Episodic memory retrieval gating
    _em_gate: nn.Sequential

    # Fixed transforms (control experiments)
    _fixed_transform0: nn.Linear
    _fixed_transform1: nn.Linear

    def __init__(
        self,
        env_type: EnvironmentType,
        key: PRNGKeyArray,
        num_feature_dims_mem_keys: int = 1,
    ):
        super().__init__()

        kg = keygen(key)
        match env_type:
            case EnvironmentType.WATERMAZE:
                self._init_watermaze(kg)
            case EnvironmentType.DISCRETE_MAZE:
                self._init_discrete_maze(kg)
            case EnvironmentType.CRAFTER:
                self._init_crafter(kg)
            case _:
                raise ValueError(f"Unsupported environment type: {env_type}")

        reward_size = 2  # task + exploration reward
        self._rnn_input_size = self.action_size + reward_size + self.data_size

        # Initialize reservoir (local connectivity for non-Crafter)
        if env_type != EnvironmentType.CRAFTER:
            self._rnn = ReservoirLocalConnectivity(next(kg), self._rnn_input_size, num_unique=40, num_shared=20)
        else:
            # For Crafter, would use EchoStateNetwork - but keeping simple for this package
            self._rnn = ReservoirLocalConnectivity(next(kg), self._rnn_input_size, num_unique=40, num_shared=20)

        self.reservoir_size = self._rnn.hidden_size

        # Filter network for input gating
        maximum = 5.0
        minimum = 0.25
        self.filter = nn.Sequential(
            [
                nn.Linear(self._rnn_input_size * 2, self._layer_size, key=next(kg)),
                nn.Lambda(jax.nn.relu),
                nn.Linear(self._layer_size, self._layer_size, key=next(kg)),
                nn.Lambda(jax.nn.relu),
                nn.Linear(self._layer_size, self._rnn_input_size, key=next(kg)),
                nn.Lambda(jax.nn.sigmoid),
                nn.Lambda(lambda x: (maximum - minimum) * x + minimum),
            ]
        )

        self.fc_mem = nn.Linear(self.reservoir_size + self.mem_size, self.mem_size, key=next(kg))
        self.hd_bias = jax.random.normal(next(kg), (self._rnn_input_size * 2,))
        self.num_feature_dims_mem_keys = num_feature_dims_mem_keys

        # Q-value classifier
        self._classifier = nn.Sequential(
            [
                nn.Linear(self.mem_size, self._layer_size, key=next(kg)),
                nn.Lambda(jax.nn.relu),
                nn.Linear(self._layer_size, self._layer_size, key=next(kg)),
                nn.Lambda(jax.nn.relu),
                nn.Linear(self._layer_size, self._layer_size, key=next(kg)),
                nn.Lambda(jax.nn.relu),
                nn.Linear(self._layer_size, self.action_size, key=next(kg)),
            ]
        )

        # Attention and embedding
        self._attn1 = Attention(qk_size=32, vo_size=self.mem_size, num_heads=1, key=next(kg))
        self._emb1 = nn.MLP(
            self.reservoir_size,
            out_size=self.mem_size,
            width_size=self.reservoir_size,
            depth=3,
            key=next(kg),
        )

        # Fixed transforms for control experiments
        self._fixed_transform0 = nn.Linear(self.data_size, self.data_size, key=next(kg), use_bias=False)
        self._fixed_transform1 = nn.Linear(self.data_size, self.data_size, key=next(kg), use_bias=False)

        # PFC MLP networks
        self._pfc_q_mlp = MLPLayer(
            nn.MLP(
                self.data_size,
                self.data_size,
                width_size=self.data_size,
                depth=2,
                key=next(kg),
            )
        )
        self._pfc_k_mlp = MLPLayer(
            nn.MLP(
                self.data_size,
                self.data_size,
                width_size=self.data_size,
                depth=2,
                key=next(kg),
            )
        )

        # Episodic memory retrieval gating
        self._em_gate = nn.Sequential(
            [
                nn.MLP(3 + self.data_size, 1, width_size=32, depth=2, key=next(kg)),
                nn.Lambda(jax.nn.sigmoid),
            ]
        )

        # PFC MoE networks
        self._pfc_q_moe = MoELayer(
            gate=nn.MLP(self.data_size, 2, width_size=self.data_size, depth=2, key=next(kg)),
            expert0=nn.MLP(
                self.data_size,
                self.data_size,
                width_size=self.data_size,
                depth=2,
                key=next(kg),
            ),
            expert1=nn.MLP(
                self.data_size,
                self.data_size,
                width_size=self.data_size,
                depth=2,
                key=next(kg),
            ),
        )

        self._pfc_k_moe = MoELayer(
            gate=nn.MLP(self.data_size, 2, width_size=self.data_size, depth=2, key=next(kg)),
            expert0=nn.MLP(
                self.data_size,
                self.data_size,
                width_size=self.data_size,
                depth=2,
                key=next(kg),
            ),
            expert1=nn.MLP(
                self.data_size,
                self.data_size,
                width_size=self.data_size,
                depth=2,
                key=next(kg),
            ),
        )

    def _init_crafter(self, kg: Generator[jnp.ndarray, None, None]):
        """Initialize for Crafter environment."""
        self._layer_size = 256
        self.data_size = 256
        self.mem_size = 256
        self.action_size = 17

        self._preprocess = nn.Sequential(
            [
                nn.Lambda(lambda x: x.transpose(2, 0, 1)),
                nn.Conv2d(3, 16, kernel_size=4, stride=4, padding=0, key=next(kg)),
                nn.Lambda(jax.nn.relu),
                nn.Conv2d(16, 32, kernel_size=4, stride=4, padding=0, key=next(kg)),
                nn.Lambda(jax.nn.relu),
                nn.Lambda(jnp.ravel),
                nn.Linear(288, 288, key=next(kg)),
                nn.Lambda(jax.nn.relu),
                nn.Linear(288, self.data_size, key=next(kg)),
            ]
        )

    def _init_watermaze(self, kg: Generator[jnp.ndarray, None, None]):
        """Initialize for Watermaze environment."""
        self._layer_size = 256
        self.data_size = 17 + 1  # +1 for the random binary dimension
        self.mem_size = 64
        self.action_size = 4

        self._preprocess = nn.Sequential([nn.Lambda(lambda x: x)])

    def _init_discrete_maze(self, kg: Generator[jnp.ndarray, None, None]):
        """Initialize for Discrete Maze environment."""
        self._layer_size = 64
        self.data_size = 2
        self.mem_size = 32
        self.action_size = 4

        self._preprocess = nn.Sequential([nn.Lambda(lambda x: x)])

    def preprocess(self, obs: jnp.ndarray):
        """Preprocess observation."""
        return self._preprocess(obs)

    def rnn(
        self,
        prev_reward: jnp.ndarray,
        prev_density: jnp.ndarray,
        prev_action: jnp.ndarray,
        x_feat: jnp.ndarray,
        m_main: jnp.ndarray,
        hidden: jnp.ndarray,
    ) -> jnp.ndarray:
        """Run one step of the reservoir RNN."""
        x = jnp.concat([prev_reward, prev_density, prev_action, x_feat])
        x = x * m_main
        return self._rnn(x, hidden)

    def classifier(self, hidden: jnp.ndarray) -> jnp.ndarray:
        """Compute Q-values from hidden state."""
        return self._classifier(hidden)

    def attention(self, wm: jnp.ndarray, em: jnp.ndarray) -> jnp.ndarray:
        """Self-attention on working memory and episodic memory.

        Args:
            wm: (elements in working memory, embedding size)
            em: (elements in episodic memory, embedding size)
        """
        em_plus_wm = jnp.concatenate((em, wm[:-1]), 0)
        return self._attn1(wm[-1], em_plus_wm, em_plus_wm)


# =============================================================================
# State Classes
# =============================================================================


@jdc.pytree_dataclass
class Networks:
    """Online and target networks for Double DQN."""

    online: RecurrentNetwork
    target: RecurrentNetwork


@jdc.pytree_dataclass
class StepState:
    """State updated every step."""

    episodic_memory: EpisodicMemory
    init_hidden: jnp.ndarray
    hidden: jnp.ndarray
    key: PRNGKeyArray
    t: Scalar  # Current timestep
    last_episode: jnp.ndarray
    online_density_buffer_state: fbx.trajectory_buffer.TrajectoryBufferState
    obs_buffer_state: fbx.trajectory_buffer.TrajectoryBufferState
    h_buffer_state: fbx.trajectory_buffer.TrajectoryBufferState
    obs_buffer_state_exploit: fbx.trajectory_buffer.TrajectoryBufferState


@jdc.pytree_dataclass
class ExperienceState:
    """Experience replay buffer state."""

    replay: tuple[fbx.prioritised_trajectory_buffer.PrioritisedTrajectoryBufferState]
    goal: tuple[fbx.prioritised_trajectory_buffer.PrioritisedTrajectoryBufferState]


@jdc.pytree_dataclass
class OptState:
    """Optimizer state updated once every cycle."""

    opt_state: optax.OptState
    target_update_count: Int[Scalar, "()"]


class AgentState(eqx.Module):
    """Complete agent state."""

    step: StepState
    nets: Networks
    opt: OptState
    experience: ExperienceState
    inference: bool = False


# =============================================================================
# Configuration
# =============================================================================


@dataclasses.dataclass(eq=True, frozen=True)
class Config:
    """Configuration for the HPC-PFC Model agent."""

    env_type: EnvironmentType
    total_samples: int
    replay_sequence_length: int
    replay_buffer_capacity: int
    epsilon_transition_steps: int
    memory_capacity: int
    num_actions: int

    # Probability of storing a memory for non-terminal states
    store_mem_probability: float = 0.05
    n_per_key: int = 1  # Number of memories to recall per key
    discount: float = 0.9
    filter_loss_coef: float = 1e-5
    optimizer_name: str = "adam"
    optimizer_kwargs: dict[str, Any] = dataclasses.field(default_factory=lambda: {"learning_rate": 1e-4})

    num_envs: int = 1
    num_off_policy_updates_per_cycle: int = 1
    target_update_interval: int = 1
    target_update_step_size: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05

    density_k_nearest_neighbours: int = 15
    env_reward_coef: float = 1.0

    # Hierarchical memory parameters
    use_hierarchical_memory: bool = False
    hierarchical_memory_k: int | None = None
    hierarchical_memory_depth: int | None = None
    hierarchical_memory_auto_refit_interval: int | None = None
    store_obs_probability: float = 1.0

    use_online_density_in_reward: bool = False
    use_offline_density_in_reward: bool = False

    def epsilon_schedule(self) -> optax.Schedule:
        return optax.linear_schedule(
            init_value=self.epsilon_start,
            end_value=self.epsilon_end,
            transition_steps=self.epsilon_transition_steps,
        )

    def __post_init__(self):
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
            raise ValueError("target_update_step_size must be between 0 and 1, inclusive")
        if not (0 <= self.env_reward_coef <= 1):
            raise ValueError("env_reward_coef must be a value between 0 and 1, inclusive")
        # Hierarchical memory validation
        if self.use_hierarchical_memory:
            if not self.hierarchical_memory_k or not self.hierarchical_memory_depth:
                raise ValueError(
                    "hierarchical_memory_k and hierarchical_memory_depth must be set if use_hierarchical_memory is True"
                )
        elif (
            self.hierarchical_memory_k or self.hierarchical_memory_depth or self.hierarchical_memory_auto_refit_interval
        ):
            raise ValueError(
                "hierarchical_memory_k, hierarchical_memory_depth, and hierarchical_memory_auto_refit_interval "
                "must be None if use_hierarchical_memory is False"
            )
        # Density reward validation
        if self.env_reward_coef != 1:
            if not (self.use_online_density_in_reward or self.use_offline_density_in_reward):
                raise ValueError(
                    "If env_reward_coef != 1, then either use_online_density_in_reward or "
                    "use_offline_density_in_reward must be true"
                )
        elif self.use_online_density_in_reward or self.use_offline_density_in_reward:
            raise ValueError(
                "If env_reward_coef == 1, then both use_online_density_in_reward and "
                "use_offline_density_in_reward must be false"
            )
        opt_factory = getattr(optax, self.optimizer_name, None)
        if opt_factory is None:
            raise ValueError(
                f"optimizer_name set to {self.optimizer_name} but optax.{self.optimizer_name} doesn't exist"
            )
        opt_factory(**self.optimizer_kwargs)  # check kwargs are valid

    @property
    def exploration_coef(self) -> float:
        return 1 - self.env_reward_coef

    def optimizer(self) -> optax.GradientTransformation:
        return getattr(optax, self.optimizer_name)(**self.optimizer_kwargs)


# =============================================================================
# Main Agent Class
# =============================================================================


class HPC_PFC:
    def __init__(self, config: Config):
        self.config = config

        self._replay_buffer = fbx.make_prioritised_trajectory_buffer(
            add_batch_size=1,
            sample_batch_size=1,
            sample_sequence_length=self.config.replay_sequence_length,
            period=1,
            min_length_time_axis=1,
            max_length_time_axis=self.config.replay_buffer_capacity,
        )

        self._goal_replay_buffer = fbx.make_prioritised_trajectory_buffer(
            add_batch_size=1,
            sample_batch_size=1,
            sample_sequence_length=self.config.replay_sequence_length,
            period=1,
            min_length_time_axis=1,
            max_length_time_axis=self.config.replay_buffer_capacity,
        )

        self._obs_buffer = fbx.make_flat_buffer(
            max_length=self.config.total_samples,
            min_length=1,
            sample_batch_size=1,
            add_batch_size=self.config.num_envs,
        )

        self._h_buffer = fbx.make_flat_buffer(
            max_length=self.config.total_samples,
            min_length=1,
            sample_batch_size=1,
            add_batch_size=self.config.num_envs,
        )

        self._obs_buffer_exploit = fbx.make_flat_buffer(
            max_length=self.config.total_samples,
            min_length=1,
            sample_batch_size=1,
            add_batch_size=self.config.num_envs,
        )

        self._online_density_buffer = fbx.make_flat_buffer(
            max_length=self.config.total_samples,
            min_length=1,
            sample_batch_size=1,
            add_batch_size=self.config.num_envs,
        )

    def new_state(
        self,
        nets: Networks,
        env_info: EnvInfo,
        key: PRNGKeyArray,
        inference: bool = False,
    ) -> AgentState:
        """Initialize agent state."""

        @eqx.filter_jit(donate="all")
        def _helper(nets: Networks, env_info: EnvInfo, key: PRNGKeyArray):
            hidden_key, opt_key, replay_key = jax.random.split(key, 3)
            if inference:
                return AgentState(
                    step=self._new_step_state(nets, env_info, key),
                    nets=nets,
                    opt=None,
                    experience=None,
                    inference=True,
                )
            else:
                return AgentState(
                    step=self._new_step_state(nets, env_info, hidden_key),
                    nets=nets,
                    opt=self._new_opt_state(nets, env_info, opt_key),
                    experience=self._new_experience_state(nets, env_info, replay_key),
                )

        return _helper(nets, env_info, key)

    def _new_step_state(self, nets: Networks, env_info: EnvInfo, key: PRNGKeyArray) -> StepState:
        kg = keygen(key)

        episodic_memory = FlatQKLearnedMemory(
            capacity=self.config.memory_capacity,
            nkey_features=nets.online.data_size,
            nvalue_features=nets.online.reservoir_size,
            batch_size=env_info.num_envs,
        )

        sample_obs = env_info.observation_space.sample(next(kg))
        assert isinstance(sample_obs, jnp.ndarray)
        sample_features = nets.online.preprocess(sample_obs)
        return StepState(
            episodic_memory=episodic_memory,
            init_hidden=jnp.zeros((env_info.num_envs, nets.online.reservoir_size)),
            hidden=jnp.zeros((env_info.num_envs, nets.online.reservoir_size)),
            key=next(kg),
            t=jnp.array(0, dtype=jnp.uint32),
            last_episode=jnp.zeros(1, dtype=jnp.int32),
            obs_buffer_state=self._obs_buffer.init(jnp.zeros_like(sample_features)),
            obs_buffer_state_exploit=self._obs_buffer_exploit.init(jnp.zeros_like(sample_features)),
            h_buffer_state=self._h_buffer.init(jnp.zeros(nets.online.reservoir_size)),
            online_density_buffer_state=self._online_density_buffer.init(jnp.zeros(1)),
        )

    def _new_experience_state(self, nets: Networks, env_info: EnvInfo, key: PRNGKeyArray) -> ExperienceState:
        sample_obs = env_info.observation_space.sample(key)
        assert isinstance(sample_obs, jax.Array)
        return ExperienceState(
            replay=tuple(
                self._replay_buffer.init(
                    RecallDatum(
                        obs=jnp.zeros_like(sample_obs),
                        action=jnp.array(1, dtype=jnp.int32),
                        reward=jnp.array(1, dtype=jnp.float32),
                        new_episode=jnp.array(True, dtype=jnp.bool),
                    )
                )
                for _ in range(self.config.num_envs)
            ),
            goal=tuple(
                self._replay_buffer.init(
                    RecallDatum(
                        obs=jnp.zeros_like(sample_obs),
                        action=jnp.zeros(1, dtype=jnp.int32),
                        reward=jnp.zeros(1),
                        new_episode=jnp.zeros(1, dtype=jnp.bool),
                    )
                )
                for _ in range(self.config.num_envs)
            ),
        )

    def _new_opt_state(self, nets: Networks, env_info: EnvInfo, key: PRNGKeyArray) -> OptState:
        nets_yes_grad, _ = self.partition_for_grad(nets)
        return OptState(
            opt_state=self.config.optimizer().init(eqx.filter(nets_yes_grad.online, eqx.is_array)),
            target_update_count=jnp.array(0, dtype=jnp.int32),
        )

    def step(self, state: AgentState, env_step: EnvStep, env_state: EnvState) -> AgentStep:
        """Select action and update step state."""

        @eqx.filter_jit(donate="all-except-first")
        def step_jit(env_step, env_state, state):
            return self._step(state, env_step, env_state)

        return step_jit(env_step, env_state, state)

    def _step(self, state: AgentState, env_step: EnvStep, env_state: EnvState) -> AgentStep:
        """Internal step method - selects a batch of actions for a timestep."""
        nets = state.nets
        action_key, store_obs_key, store_mem_key, key = jax.random.split(state.step.key, 4)

        x_feat = eqx.filter_vmap(nets.online.preprocess)(env_step.obs)

        def step_single(
            episodic_memory: EpisodicMemory,
            x_feat: jnp.ndarray,
            reward: jnp.ndarray,
            prev_action: jnp.ndarray,
            density,
            hidden,
            init_hidden,
        ):
            # Get the current reservoir hidden state
            m_main = nets.online.filter(nets.online.hd_bias)
            reward = jnp.expand_dims(reward, 0)
            prev_action_one_hot = jax.nn.one_hot(prev_action, state.nets.online.action_size)
            hidden = nets.online.rnn(reward, density, prev_action_one_hot, x_feat, m_main, hidden)

            # MLP for query
            q = nets.online._pfc_q_mlp(x_feat)

            em_hidden = episodic_memory.learned_recall(
                q, nets.online._pfc_k_mlp, nets.online._em_gate, self.config.n_per_key
            )

            # Form WM hidden state
            wm_hidden = jnp.expand_dims(hidden, 0)

            attn_hidden = nets.online.attention(
                eqx.filter_vmap(nets.online._emb1)(wm_hidden),
                eqx.filter_vmap(nets.online._emb1)(em_hidden),
            )

            action = jnp.argmax(nets.online.classifier(attn_hidden.squeeze(0)))
            rand_action = jax.random.choice(action_key, self.config.num_actions, shape=action.shape)
            take_rand = jax.random.uniform(key) < self.config.epsilon_schedule()(state.step.t)
            chosen_action = jax.lax.select(take_rand, rand_action, action)

            return (chosen_action, jax.lax.stop_gradient(hidden), hidden)

        density = eqx.filter_vmap(distance_to_kth_nearest_neighbour)(
            jnp.reshape(x_feat, (self.config.num_envs, 1, -1)),
            state.step.obs_buffer_state.experience,
            self.config.density_k_nearest_neighbours,
        )

        if self.config.exploration_coef == 0:
            density = jnp.zeros_like(density)

        rnd = jax.random.uniform(store_obs_key, minval=0.0, maxval=1.0)

        assert isinstance(env_state, episodic_water_maze_state)
        obs_buffer_state = jax.lax.cond(
            jnp.logical_and(
                env_step.new_episode.any(),
                env_state.trial_type.squeeze() == TrialType.EXPLORE,
            ),
            lambda: self._obs_buffer.add(state.step.obs_buffer_state, x_feat),
            lambda: state.step.obs_buffer_state,
        )

        obs_buffer_state_exploit = jax.lax.cond(
            jnp.logical_and(
                env_step.new_episode.any(),
                env_state.trial_type.squeeze() == TrialType.EXPLOIT,
            ),
            lambda: self._obs_buffer_exploit.add(state.step.obs_buffer_state_exploit, x_feat),
            lambda: state.step.obs_buffer_state_exploit,
        )

        online_density_buffer_state = self._online_density_buffer.add(state.step.online_density_buffer_state, density)

        action, z_feat, hidden = jax.vmap(step_single)(
            state.step.episodic_memory,
            x_feat,
            env_step.reward,
            env_step.prev_action,
            density,
            state.step.hidden,
            state.step.init_hidden,
        )

        h_buffer_state = jax.lax.cond(
            jnp.logical_and(
                env_step.new_episode.any(),
                env_state.trial_type.squeeze() == TrialType.EXPLORE,
            ),
            lambda: self._h_buffer.add(state.step.h_buffer_state, z_feat),
            lambda: state.step.h_buffer_state,
        )

        rnd = jax.random.uniform(store_mem_key, minval=0.0, maxval=1.0)

        episodic_memory = filter_cond(
            jnp.logical_and(
                env_step.new_episode.any(),
                env_state.trial_type.squeeze() == TrialType.EXPLORE,
            ),
            lambda episodic_memory: episodic_memory.store(
                jax.lax.stop_gradient(x_feat),
                jax.lax.stop_gradient(z_feat),
            ),
            lambda episodic_memory: episodic_memory,
            state.step.episodic_memory,
        )

        init_hidden = filter_cond(
            state.step.t == 0,
            lambda: hidden,
            lambda: state.step.init_hidden,
        )

        step = jdc.replace(
            state.step,
            episodic_memory=episodic_memory,
            init_hidden=init_hidden,
            hidden=hidden,
            key=key,
            t=state.step.t + 1,
            obs_buffer_state=obs_buffer_state,
            h_buffer_state=h_buffer_state,
            obs_buffer_state_exploit=obs_buffer_state_exploit,
            online_density_buffer_state=online_density_buffer_state,
        )
        return AgentStep(action, step)

    def update_experience(self, state: AgentState, trajectory: EnvStep) -> ExperienceState:
        """Update experience replay from trajectory."""

        @eqx.filter_jit(donate="all-except-first")
        def _update_experience_jit(others, experience):
            agent_state_no_exp, trajectory = others
            agent_state = dataclasses.replace(agent_state_no_exp, experience=experience)
            return self._update_experience(agent_state, trajectory)

        exp = state.experience
        agent_state_no_exp = dataclasses.replace(state, experience=None)
        return _update_experience_jit((agent_state_no_exp, trajectory), exp)

    def _update_experience(self, state: AgentState, trajectory: EnvStep) -> ExperienceState:
        num_steps = trajectory.reward.shape[1]

        recall_datum = RecallDatum(
            obs=trajectory.obs,
            action=trajectory.prev_action,
            reward=trajectory.reward,
            new_episode=trajectory.new_episode,
        )

        assert state.experience.replay is not None
        priority_state_tuple: tuple[SumTreeState, ...] = tuple(rb.priority_state for rb in state.experience.replay)
        priority_state: SumTreeState = jax.tree.map(lambda *x: jnp.stack(x), *priority_state_tuple)

        replay_buffer_state = jax.tree.map(lambda *x: jnp.stack(x), *state.experience.replay)
        replay_buffer_state = filter_vmap(self._replay_buffer.add)(
            replay_buffer_state,
            jax.tree.map(lambda x: jnp.expand_dims(x, axis=1), recall_datum),
        )

        def _reset_priority(replay_buffer_state, priority_state):
            return jdc.replace(replay_buffer_state, priority_state=priority_state)

        replay_buffer_state = filter_vmap(_reset_priority)(replay_buffer_state, priority_state)
        indices = jnp.arange(-num_steps, 0)

        indices = (indices + replay_buffer_state.current_index[0]) % self.config.replay_buffer_capacity
        replay_buffer_state = filter_vmap(self._replay_buffer.set_priorities, in_axes=(0, None, 0))(
            replay_buffer_state,
            indices,
            trajectory.new_episode,
        )

        replay_buffer_state = tuple(
            jax.tree.map(lambda x: x[i], replay_buffer_state) for i in range(self.config.num_envs)
        )

        return ExperienceState(
            replay=replay_buffer_state,
            goal=state.experience.goal,
        )

    def partition_for_grad(self, nets: Networks) -> tuple[Networks, Networks]:
        """Partition networks for gradient computation."""
        return eqx.filter_jit(donate="all")(self._partition_for_grad)(nets)

    def _partition_for_grad(self, nets: Networks) -> tuple[Networks, Networks]:
        # All arrays are trainable...
        filter_spec = jax.tree.map(lambda _: eqx.is_array, nets)
        # ... except for target parameters...
        filter_spec = eqx.tree_at(lambda nets: nets.target, filter_spec, replace=False)
        # ... and the reservoir parameters
        filter_spec = eqx.tree_at(
            lambda nets: (nets.online._rnn.weight_hh, nets.online._rnn.weight_ih),
            filter_spec,
            replace=(False, False),
        )
        return eqx.partition(nets, filter_spec)

    def loss(self, state: AgentState) -> tuple[Scalar, dict]:
        """Compute loss. Wrapper with JIT."""
        return eqx.filter_jit(donate="all")(self._loss)(state)

    def _loss(self, state: AgentState):
        replay_buffer_state = jax.tree.map(lambda *x: jnp.stack(x), *state.experience.replay)
        if self.config.env_type == EnvironmentType.WATERMAZE:
            sample = replay_buffer_state
            indices = jnp.repeat(
                jnp.expand_dims(jnp.arange(self.config.replay_buffer_capacity), 0),
                self.config.num_envs,
                0,
            )
        else:
            sample = eqx.filter_vmap(self._recall_random)(
                replay_buffer_state,
                jax.random.split(state.step.key, self.config.num_envs),
            )

            def _sequence_index(index: jnp.ndarray):
                return (index + jnp.arange(self.config.replay_sequence_length)) % self.config.replay_buffer_capacity

            indices = _sequence_index(sample.indices)

        experience = sample.experience
        experience = jax.tree.map(lambda x: x.squeeze(1), experience)

        x_feat_tm1 = filter_vmap(filter_vmap(state.nets.online.preprocess))(experience.obs[:, :-1])
        x_feat_t = filter_vmap(filter_vmap(state.nets.online.preprocess))(experience.obs[:, 1:])
        x_feat_t_selector = filter_vmap(filter_vmap(state.nets.target.preprocess))(experience.obs[:, 1:])

        @eqx.filter_vmap
        def _vmapped_index(arr: jnp.ndarray, index: jnp.ndarray):
            return arr[index]

        online_density = _vmapped_index(state.step.online_density_buffer_state.experience, indices).squeeze(-1)
        if not self.config.use_online_density_in_reward:
            online_density = jnp.zeros_like(online_density)

        offline_density = eqx.filter_vmap(distance_to_kth_nearest_neighbour)(
            jax.vmap(jax.vmap(state.nets.online.preprocess))(experience.obs),
            state.step.obs_buffer_state.experience,
            self.config.density_k_nearest_neighbours,
        )
        if not self.config.use_offline_density_in_reward:
            offline_density = jnp.zeros_like(offline_density)

        replay_density = jax.lax.stop_gradient(online_density + offline_density)

        if self.config.exploration_coef == 0:
            replay_density = jnp.zeros_like(replay_density)

        reward_tm1 = experience.reward[:, :-1]
        reward_t = experience.reward[:, 1:]
        action_tm2 = experience.action[:, :-1]
        action_tm1 = experience.action[:, 1:]

        density_tm1 = replay_density[:, :-1]
        density_t = replay_density[:, 1:]

        combined_reward_t = self.config.exploration_coef * density_t + self.config.env_reward_coef * reward_t

        online_m_main = state.nets.online.filter(state.nets.online.hd_bias)
        target_m_main = state.nets.target.filter(state.nets.target.hd_bias)

        def scan_body(hidden_priors, inputs):
            inputs_tm1, inputs_t, inputs_t_selector = inputs
            x_feat_tm1, reward_tm1, density_tm1, action_tm2 = inputs_tm1
            x_feat_t, reward_t, density_t, action_tm1 = inputs_t
            (x_feat_t_selector,) = inputs_t_selector

            hidden_prior_tm1, hidden_prior_t, hidden_prior_t_selector = hidden_priors

            hidden_post_tm1 = state.nets.online.rnn(
                reward_tm1,
                density_tm1,
                action_tm2,
                x_feat_tm1,
                online_m_main,
                hidden_prior_tm1,
            )

            hidden_post_t = state.nets.online.rnn(
                reward_t, density_t, action_tm1, x_feat_t, online_m_main, hidden_prior_t
            )

            hidden_post_t_selector = state.nets.target.rnn(
                reward_t,
                density_t,
                action_tm1,
                x_feat_t_selector,
                target_m_main,
                hidden_prior_t_selector,
            )

            return (hidden_post_tm1, hidden_post_t, hidden_post_t_selector), (
                hidden_post_tm1,
                hidden_post_t,
                hidden_post_t_selector,
            )

        inputs = (
            (
                x_feat_tm1,
                jnp.expand_dims(reward_tm1, -1),
                jnp.expand_dims(density_tm1, -1),
                jax.nn.one_hot(action_tm2, self.config.num_actions),
            ),
            (
                x_feat_t,
                jnp.expand_dims(reward_t, -1),
                jnp.expand_dims(density_t, -1),
                jax.nn.one_hot(action_tm1, self.config.num_actions),
            ),
            (x_feat_t_selector,),
        )
        hidden = jnp.zeros_like(state.step.hidden)
        _, (hidden_post_tm1, hidden_post_t, hidden_post_t_selector) = filter_vmap(filter_scan, in_axes=(None, 0, 0))(
            scan_body, (hidden, hidden, hidden), inputs
        )

        wm_hidden_post_tm1 = jnp.expand_dims(hidden_post_tm1, -2)
        wm_hidden_post_t = jnp.expand_dims(hidden_post_t, -2)
        wm_hidden_post_t_selector = jnp.expand_dims(hidden_post_t_selector, -2)

        q_post_tm1 = filter_vmap(filter_vmap(state.nets.online._pfc_q_mlp))(x_feat_tm1)
        q_post_t = filter_vmap(filter_vmap(state.nets.online._pfc_q_mlp))(x_feat_t)
        q_post_t_selector = filter_vmap(filter_vmap(state.nets.target._pfc_q_mlp))(x_feat_t_selector)

        z_mem_tm1 = filter_vmap(
            partial(
                state.step.episodic_memory.learned_recall,
                key_fn=state.nets.online._pfc_k_mlp,
                gate_fn=state.nets.online._em_gate,
                n_per_key=self.config.n_per_key,
            ),
            in_axes=1,
            out_axes=1,
        )(q_post_tm1)
        z_mem_t = filter_vmap(
            partial(
                state.step.episodic_memory.learned_recall,
                key_fn=state.nets.online._pfc_k_mlp,
                gate_fn=state.nets.online._em_gate,
                n_per_key=self.config.n_per_key,
            ),
            in_axes=1,
            out_axes=1,
        )(q_post_t)
        z_mem_t_selector = filter_vmap(
            partial(
                state.step.episodic_memory.learned_recall,
                key_fn=state.nets.target._pfc_k_mlp,
                gate_fn=state.nets.target._em_gate,
                n_per_key=self.config.n_per_key,
            ),
            in_axes=1,
            out_axes=1,
        )(q_post_t_selector)

        attn_hidden_post_tm1 = filter_vmap(filter_vmap(state.nets.online.attention))(
            filter_vmap(filter_vmap(filter_vmap(state.nets.online._emb1)))(wm_hidden_post_tm1),
            filter_vmap(filter_vmap(filter_vmap(state.nets.online._emb1)))(z_mem_tm1),
        )

        attn_hidden_post_t = filter_vmap(filter_vmap(state.nets.online.attention))(
            filter_vmap(filter_vmap(filter_vmap(state.nets.online._emb1)))(wm_hidden_post_t),
            filter_vmap(filter_vmap(filter_vmap(state.nets.online._emb1)))(z_mem_t),
        )
        attn_hidden_post_t_selector = filter_vmap(filter_vmap(state.nets.online.attention))(
            filter_vmap(filter_vmap(filter_vmap(state.nets.online._emb1)))(wm_hidden_post_t_selector),
            filter_vmap(filter_vmap(filter_vmap(state.nets.online._emb1)))(z_mem_t_selector),
        )

        flattened_env_and_time_dim = self.config.num_envs * (self.config.replay_sequence_length - 1)
        q_values_tm1 = filter_vmap(filter_vmap(state.nets.online.classifier))(attn_hidden_post_tm1.squeeze(-2)).reshape(
            flattened_env_and_time_dim, self.config.num_actions
        )
        q_values_t = filter_vmap(filter_vmap(state.nets.online.classifier))(attn_hidden_post_t.squeeze(-2)).reshape(
            flattened_env_and_time_dim, self.config.num_actions
        )
        q_values_t_selector = filter_vmap(filter_vmap(state.nets.target.classifier))(
            attn_hidden_post_t_selector.squeeze(-2)
        ).reshape(flattened_env_and_time_dim, self.config.num_actions)

        m_main_loss = self.config.filter_loss_coef * jnp.mean(online_m_main)
        new_episode = jax.lax.stop_gradient(experience.new_episode[:, 1:])
        loss = (
            optax.huber_loss(
                jax.vmap(double_q_learning)(
                    q_tm1=q_values_tm1,
                    a_tm1=jnp.reshape(action_tm1, flattened_env_and_time_dim),
                    r_t=jnp.reshape(combined_reward_t, flattened_env_and_time_dim),
                    discount_t=self.config.discount * ~jnp.reshape(new_episode, flattened_env_and_time_dim),
                    q_t_value=q_values_t,
                    q_t_selector=q_values_t_selector,
                )
            ).mean()
            + m_main_loss
        )

        metrics = {}
        return loss, metrics

    def _recall_random(self, replay_state, key):
        return self._replay_buffer.sample(replay_state, key)

    def optimize_from_grads(self, state: AgentState, nets_grads: PyTree) -> AgentState:
        """Apply gradients to optimize agent."""
        return eqx.filter_jit(donate="all")(self._optimize_from_grads)(state, nets_grads)

    def _optimize_from_grads(self, state: AgentState, nets_grads: PyTree) -> AgentState:
        nets_grads = jax.tree.map(lambda grad: jnp.clip(grad, min=-1.0, max=1.0), nets_grads)
        updates, opt_state = self.config.optimizer().update(nets_grads.online, state.opt.opt_state)
        online = eqx.apply_updates(state.nets.online, updates)

        do_target_update = state.opt.target_update_count % self.config.target_update_interval == 0
        target = filter_cond(
            do_target_update,
            lambda nets: filter_incremental_update(
                new_tensors=nets.online,
                old_tensors=nets.target,
                step_size=self.config.target_update_step_size,
            ),
            lambda nets: nets.target,
            state.nets,
        )

        nets = jdc.replace(state.nets, online=online, target=target)
        opt = jdc.replace(
            state.opt,
            opt_state=opt_state,
            target_update_count=state.opt.target_update_count + 1,
        )
        step = jdc.replace(state.step, key=jax.random.split(state.step.key, num=1)[0])
        return AgentState(nets=nets, opt=opt, step=step, experience=state.experience)

    def num_off_policy_optims_per_cycle(self) -> int:
        """Number of off-policy optimization steps per cycle."""
        return self.config.num_off_policy_updates_per_cycle
