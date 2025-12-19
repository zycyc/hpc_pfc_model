"""Training script for HPC-PFC Model agent with inlined training loop.

This script contains the training loop logic inlined from the EARL framework's GymnaxLoop.
"""

import abc
import collections
import dataclasses
import time
import typing
import weakref
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from gymnax import EnvParams
from gymnax.environments.spaces import Discrete
from jaxtyping import PRNGKeyArray, PyTree, Scalar
from tqdm import tqdm

from agent import (
    AgentState,
    Config,
    EnvironmentType,
    EnvStep,
    HPC_PFC,
    Networks,
    RecurrentNetwork,
    env_info_from_gymnax,
)
from env.water_maze import EpisodicWaterMaze, TrialType
from env.water_maze import Params as EpisodicWaterMazeParams
from utils.jax_utils import filter_scan
from utils.prng import keygen

# =============================================================================
# Metrics
# =============================================================================

Metrics = Mapping[str, Scalar | float | int]


class MetricKey:
    """Standard metric keys."""

    LOSS = "loss"
    TOTAL_REWARD = "total_reward"
    TOTAL_DONES = "total_dones"
    COMPLETE_EPISODE_LENGTH_MEAN = "complete_episode_length_mean"
    NUM_ENVS_THAT_DID_NOT_COMPLETE = "num_envs_that_did_not_complete"
    REWARD_MEAN_SMOOTH = "reward_mean_smooth"
    STEP_NUM = "step_num"
    DURATION_SEC = "duration_sec"
    ACTION_COUNTS = "action_counts"


_ALL_METRIC_KEYS = {
    MetricKey.LOSS,
    MetricKey.TOTAL_REWARD,
    MetricKey.TOTAL_DONES,
    MetricKey.COMPLETE_EPISODE_LENGTH_MEAN,
    MetricKey.NUM_ENVS_THAT_DID_NOT_COMPLETE,
    MetricKey.REWARD_MEAN_SMOOTH,
    MetricKey.STEP_NUM,
    MetricKey.DURATION_SEC,
    MetricKey.ACTION_COUNTS,
}


# =============================================================================
# Logging Infrastructure
# =============================================================================


class Closable(abc.ABC):
    """Base class for closable resources."""

    def __init__(self):
        self._finalizer = weakref.finalize(self, self._close)

    def close(self) -> None:
        """Closes the logger."""
        self._finalizer()

    @abc.abstractmethod
    def _close(self) -> None:
        """Closes the logger. Sub-classes should override this method."""


class MetricLogger(Closable):
    """A logger for metrics."""

    @abc.abstractmethod
    def write(self, metrics: Metrics) -> None:
        """Writes metrics to some destination."""


class MlflowMetricLogger(MetricLogger):
    """Logs metrics to MLflow."""

    def __init__(self, client, run_id: str, label: str = ""):
        super().__init__()
        self._client = client
        self._run_id = run_id
        self._prefix = f"{label}/" if label else ""

    def write(self, metrics: Metrics):
        """Removes MetricKey.STEP_NUM from the metrics and uses it as the step number."""
        import mlflow.entities

        step_num = int(metrics[MetricKey.STEP_NUM]) // 10000
        timestamp = int(time.time() * 1000)
        metrics_list = []
        for k, v in metrics.items():
            if k == MetricKey.STEP_NUM:
                continue
            k = f"{self._prefix}{k}"
            metrics_list.append(mlflow.entities.Metric(k, float(v), timestamp, step_num))

        self._client.log_batch(self._run_id, metrics=metrics_list, synchronous=False)

    def _close(self):
        import mlflow

        self._client.set_terminated(self._run_id)
        mlflow.flush_async_logging()


# =============================================================================
# Step Carry for Scan
# =============================================================================


@jdc.pytree_dataclass()
class _StepCarry:
    """Carry state for the environment step scan."""

    env_step: EnvStep
    env_state: Any  # EnvState
    step_state: Any  # StepState
    key: PRNGKeyArray
    total_reward: Scalar
    total_dones: Scalar
    episode_steps: jnp.ndarray
    complete_episode_length_sum: Scalar
    complete_episode_count: Scalar
    action_counts: jnp.ndarray


@jdc.pytree_dataclass()
class _CycleResult:
    """Result of one training cycle."""

    agent_state: PyTree
    env_state: Any
    env_step: EnvStep
    key: PRNGKeyArray
    metrics: dict[str, jnp.ndarray]
    trajectory: EnvStep
    step_infos: dict[Any, Any]


# =============================================================================
# Observation Functions
# =============================================================================


def watermaze_observe_trajectory_factory(log_to_console: bool = True, log_to_mlflow: bool = False):
    """Factory for creating trajectory observation function for WaterMaze."""
    k = 0

    alpha = 0.99
    reward_moving_average = 0.0

    def observe_trajectory(env_steps, step_infos, step_num) -> Metrics:
        nonlocal k, reward_moving_average

        excess_steps = step_infos["excess_steps"].flatten()
        mask = excess_steps >= 0
        trial_type = step_infos["trial_type"].flatten()
        done_trials = trial_type[mask]
        excess_step = excess_steps[mask]
        explore_excess_steps = excess_step[done_trials == TrialType.EXPLORE]
        exploit_excess_steps = excess_step[done_trials == TrialType.EXPLOIT]

        metrics = {}
        if explore_excess_steps.size > 0:
            metrics["explore_excess_step_means"] = explore_excess_steps.mean()
        if exploit_excess_steps.size > 0:
            metrics["exploit_excess_step_means"] = exploit_excess_steps.mean()

        if "random_binary" in step_infos:
            random_binary = step_infos["random_binary"].flatten()
            explore_random_binary = random_binary[mask][done_trials == TrialType.EXPLORE]
            exploit_random_binary = random_binary[mask][done_trials == TrialType.EXPLOIT]
            explore_excess_steps_rule1 = explore_excess_steps[explore_random_binary == 1]
            exploit_excess_steps_rule1 = exploit_excess_steps[exploit_random_binary == 1]
            explore_excess_steps_rule0 = explore_excess_steps[explore_random_binary == 0]
            exploit_excess_steps_rule0 = exploit_excess_steps[exploit_random_binary == 0]

            if explore_excess_steps_rule1.size > 0:
                metrics["explore_excess_step_means_rule1"] = explore_excess_steps_rule1.mean()
            if exploit_excess_steps_rule1.size > 0:
                metrics["exploit_excess_step_means_rule1"] = exploit_excess_steps_rule1.mean()
            if explore_excess_steps_rule0.size > 0:
                metrics["explore_excess_step_means_rule0"] = explore_excess_steps_rule0.mean()
            if exploit_excess_steps_rule0.size > 0:
                metrics["exploit_excess_step_means_rule0"] = exploit_excess_steps_rule0.mean()

        k += 1
        _LOG_EVERY = 100

        if k % _LOG_EVERY == 0:
            reward_moving_average = alpha * reward_moving_average + (1 - alpha) * env_steps.reward.mean()

            if log_to_mlflow:
                import mlflow

                mlflow.log_metric("reward_mean", reward_moving_average, step=k // _LOG_EVERY)

        return metrics

    return observe_trajectory


# =============================================================================
# Training Loop
# =============================================================================


class TrainingLoop:
    """Training loop for the HPC-PFC Model agent."""

    def __init__(
        self,
        env: EpisodicWaterMaze,
        env_params: EpisodicWaterMazeParams,
        agent: HPC_PFC,
        num_envs: int,
        key: PRNGKeyArray,
        inference: bool = False,
        logger: MetricLogger | None = None,
    ):
        """Initialize the training loop.

        Args:
            env: The Gymnax environment.
            env_params: Environment parameters.
            agent: The Fluid 1.1 agent.
            num_envs: Number of parallel environments.
            key: PRNG key.
            inference: If True, skip agent updates.
            logger: Optional metric logger for logging during training.
        """
        self._env = env
        sample_key, key = jax.random.split(key)
        sample_key = jax.random.split(sample_key, num_envs)
        self._action_space = self._env.action_space(env_params)
        self._example_action = jax.vmap(self._action_space.sample)(sample_key)
        self._env_reset = partial(self._env.reset, params=env_params)
        self._env_step = partial(self._env.step, params=env_params)
        self._agent = agent
        self._num_envs = num_envs
        self._key = key
        self._inference = inference
        self._logger = logger

        # JIT compile the main cycle function
        self._run_cycle_and_update = eqx.filter_jit(self._run_cycle_and_update_impl, donate="warn")

    def run(
        self,
        agent_state: AgentState,
        num_cycles: int,
        steps_per_cycle: int,
        print_progress: bool = True,
        observe_trajectory=None,
    ) -> tuple[AgentState, dict[str, list[float | int]]]:
        """Run training for the specified number of cycles.

        Args:
            agent_state: Initial agent state.
            num_cycles: Number of training cycles.
            steps_per_cycle: Steps per cycle.
            print_progress: Whether to show progress bar.
            observe_trajectory: Optional callback for trajectory observation.

        Returns:
            Final agent state and metrics dictionary.
        """
        import optax

        if num_cycles <= 0:
            raise ValueError("num_cycles must be positive.")
        if steps_per_cycle <= 0:
            raise ValueError("steps_per_cycle must be positive.")

        all_metrics = collections.defaultdict(list)

        # Initialize environment
        env_key, self._key = jax.random.split(self._key, 2)
        env_keys = jax.random.split(env_key, self._num_envs)
        obs, env_state = jax.vmap(self._env_reset)(env_keys)

        env_step = EnvStep(
            new_episode=jnp.ones((self._num_envs,), dtype=jnp.bool),
            obs=obs,
            prev_action=jnp.zeros_like(self._example_action),
            reward=jnp.zeros((self._num_envs,)),
        )

        if observe_trajectory is None:

            def noop(env_steps, step_infos, step_num):
                return {}

            observe_trajectory = noop

        cycles_iter = range(num_cycles)
        if print_progress:
            cycles_iter = tqdm(cycles_iter, desc="cycles", unit="cycle", leave=False)

        for cycle_num in cycles_iter:
            cycle_start = time.time()

            # Strip weak_type to avoid recompilation
            env_step = jax.tree.map(lambda x: x.astype(x.dtype) if isinstance(x, jax.Array) else x, env_step)
            env_state = jax.tree.map(
                lambda x: x.astype(x.dtype) if isinstance(x, jax.Array) else x,
                env_state,
            )

            cycle_result = self._run_cycle_and_update(agent_state, env_state, env_step, self._key, steps_per_cycle)

            trajectory_metrics = observe_trajectory(
                cycle_result.trajectory,
                cycle_result.step_infos,
                cycle_num * steps_per_cycle,
            )

            agent_state, env_state, env_step, self._key = (
                cycle_result.agent_state,
                cycle_result.env_state,
                cycle_result.env_step,
                cycle_result.key,
            )

            # Log metrics
            py_metrics: dict[str, float | int] = {}
            py_metrics[MetricKey.DURATION_SEC] = time.time() - cycle_start
            reward_mean = cycle_result.metrics[MetricKey.TOTAL_REWARD] / self._num_envs

            if MetricKey.REWARD_MEAN_SMOOTH not in all_metrics:
                reward_mean_smooth = reward_mean
            else:
                reward_mean_smooth = optax.incremental_update(
                    jnp.array(reward_mean),
                    all_metrics[MetricKey.REWARD_MEAN_SMOOTH][-1],
                    0.01,
                )
            assert isinstance(reward_mean_smooth, jnp.ndarray)
            py_metrics[MetricKey.REWARD_MEAN_SMOOTH] = float(reward_mean_smooth)

            py_metrics[MetricKey.STEP_NUM] = (cycle_num + 1) * steps_per_cycle

            action_counts = cycle_result.metrics.pop(MetricKey.ACTION_COUNTS)
            if isinstance(action_counts, jnp.ndarray) and action_counts.shape:
                for i in range(action_counts.shape[0]):
                    py_metrics[f"action_counts/{i}"] = int(action_counts[i])

            for k, v in cycle_result.metrics.items():
                if isinstance(v, jnp.ndarray) and v.shape == ():
                    py_metrics[k] = v.item()

            for k, v in trajectory_metrics.items():
                if isinstance(v, jnp.ndarray):
                    v = v.item()
                py_metrics[k] = v

            for k, v in py_metrics.items():
                all_metrics[k].append(v)

            # Log to MLflow every 100 cycles
            if self._logger and (cycle_num + 1) % 100 == 0:
                self._logger.write(py_metrics)

            # Print progress every 1000 cycles
            if print_progress and (cycle_num + 1) % 1000 == 0:
                loss_val = py_metrics.get(MetricKey.LOSS, 0)
                print(f"Cycle {cycle_num + 1}: loss={loss_val:.4f}")

        return agent_state, all_metrics

    def _run_cycle_and_update_impl(
        self,
        agent_state: AgentState,
        env_state: Any,
        env_step: EnvStep,
        key: PRNGKeyArray,
        steps_per_cycle: int,
    ) -> _CycleResult:
        """Run one cycle and update agent (implementation)."""

        @eqx.filter_grad(has_aux=True)
        def _loss_for_cycle_grad(nets_yes_grad, nets_no_grad, other_agent_state):
            agent_state = dataclasses.replace(other_agent_state, nets=eqx.combine(nets_yes_grad, nets_no_grad))
            loss, metrics = self._agent.loss(agent_state)
            mutable_metrics = typing.cast(dict[str, jnp.ndarray], dict(metrics))
            mutable_metrics[MetricKey.LOSS] = loss
            return loss, mutable_metrics

        def _off_policy_update(agent_state, _):
            nets_yes_grad, nets_no_grad = self._agent.partition_for_grad(agent_state.nets)
            grad, metrics = _loss_for_cycle_grad(nets_yes_grad, nets_no_grad, jdc.replace(agent_state, nets=None))
            agent_state = self._agent.optimize_from_grads(agent_state, grad)
            return agent_state, metrics

        # Run cycle
        cycle_result = self._run_cycle(agent_state, env_state, env_step, steps_per_cycle, key)
        agent_state = cycle_result.agent_state

        if not self._inference:
            experience_state = self._agent.update_experience(cycle_result.agent_state, cycle_result.trajectory)
            agent_state = dataclasses.replace(agent_state, experience=experience_state)

        metrics = cycle_result.metrics

        if not self._inference and self._agent.num_off_policy_optims_per_cycle():
            agent_state, off_policy_metrics = filter_scan(
                _off_policy_update,
                init=agent_state,
                xs=None,
                length=self._agent.num_off_policy_optims_per_cycle(),
            )
            metrics.update({k: jnp.mean(v) for k, v in off_policy_metrics.items()})

        return _CycleResult(
            agent_state,
            cycle_result.env_state,
            cycle_result.env_step,
            cycle_result.key,
            metrics,
            cycle_result.trajectory,
            cycle_result.step_infos,
        )

    def _run_cycle(
        self,
        agent_state: AgentState,
        env_state: Any,
        env_step: EnvStep,
        num_steps: int,
        key: PRNGKeyArray,
    ) -> _CycleResult:
        """Run agent in environment for num_steps."""

        def scan_body(inp: _StepCarry, _):
            agent_state_for_step = jdc.replace(agent_state, step=inp.step_state)

            agent_step = self._agent.step(agent_state_for_step, inp.env_step, inp.env_state)
            action = agent_step.action

            if isinstance(self._action_space, Discrete):
                one_hot_actions = jax.nn.one_hot(action, self._env.num_actions, dtype=inp.action_counts.dtype)
                action_counts = inp.action_counts + jnp.sum(one_hot_actions, axis=0)
            else:
                action_counts = inp.action_counts

            env_key, key = jax.random.split(inp.key)
            env_keys = jax.random.split(env_key, self._num_envs)
            obs, env_state, reward, done, info = jax.vmap(self._env_step)(env_keys, inp.env_state, action)
            next_timestep = EnvStep(done, obs, action, reward)

            episode_steps = inp.episode_steps + 1

            completed_episodes = next_timestep.new_episode
            episode_length_sum = inp.complete_episode_length_sum + jnp.sum(episode_steps * completed_episodes)
            episode_count = inp.complete_episode_count + jnp.sum(completed_episodes, dtype=jnp.uint32)

            episode_steps = jnp.where(completed_episodes, jnp.zeros_like(episode_steps), episode_steps)

            total_reward = inp.total_reward + jnp.sum(next_timestep.reward)
            total_dones = inp.total_dones + jnp.sum(next_timestep.new_episode, dtype=jnp.uint32)

            return (
                _StepCarry(
                    next_timestep,
                    env_state,
                    agent_step.state,
                    key,
                    total_reward,
                    total_dones,
                    episode_steps,
                    episode_length_sum,
                    episode_count,
                    action_counts,
                ),
                (inp.env_step, info),
            )

        if isinstance(self._action_space, Discrete):
            action_counts = jnp.zeros(self._env.num_actions, dtype=jnp.uint32)
        else:
            action_counts = jnp.array(0, dtype=jnp.uint32)

        final_carry, (trajectory, step_infos) = filter_scan(
            scan_body,
            init=_StepCarry(
                env_step,
                env_state,
                agent_state.step,
                key,
                jnp.array(0.0),
                jnp.array(0, dtype=jnp.uint32),
                jnp.zeros(self._num_envs, dtype=jnp.uint32),
                jnp.array(0, dtype=jnp.uint32),
                jnp.array(0, dtype=jnp.uint32),
                action_counts,
            ),
            xs=None,
            length=num_steps,
        )

        agent_state = dataclasses.replace(agent_state, step=final_carry.step_state)

        complete_episode_length_mean = jnp.where(
            final_carry.complete_episode_count > 0,
            final_carry.complete_episode_length_sum / final_carry.complete_episode_count,
            0,
        )

        metrics = {}
        metrics[MetricKey.COMPLETE_EPISODE_LENGTH_MEAN] = jnp.mean(complete_episode_length_mean)
        metrics[MetricKey.NUM_ENVS_THAT_DID_NOT_COMPLETE] = jnp.sum(final_carry.complete_episode_count == 0)
        metrics[MetricKey.TOTAL_DONES] = final_carry.total_dones
        metrics[MetricKey.TOTAL_REWARD] = final_carry.total_reward
        metrics[MetricKey.ACTION_COUNTS] = final_carry.action_counts

        # Transpose to (num_envs, num_steps, ...)
        def to_num_envs_first(x):
            if isinstance(x, jnp.ndarray) and x.ndim > 1:
                return jnp.transpose(x, (1, 0, *range(2, x.ndim)))
            return x

        trajectory = jax.tree.map(to_num_envs_first, trajectory)

        return _CycleResult(
            agent_state,
            final_carry.env_state,
            final_carry.env_step,
            final_carry.key,
            metrics,
            trajectory,
            step_infos,
        )


# =============================================================================
# Experiment Configuration
# =============================================================================


@dataclass
class ExperimentConfig:
    """Configuration for a Fluid 1.1 experiment."""

    env_type: EnvironmentType
    agent: Config
    env: EnvParams
    num_envs: int
    num_cycles: int
    steps_per_cycle: int
    seed: int = 0
    mlflow: bool = False
    mlflow_run_name: str | None = None
    mlflow_experiment_name: str | None = None
    inference: bool = False
    checkpoint: bool = False

    def __post_init__(self):
        self.agent = replace(self.agent, num_envs=self.num_envs, env_type=self.env_type)
        if self.agent.epsilon_transition_steps < 0:
            self.agent = replace(
                self.agent,
                epsilon_transition_steps=self.num_cycles * self.steps_per_cycle,
            )

    def new_env(self):
        match self.env_type:
            case EnvironmentType.WATERMAZE:
                return EpisodicWaterMaze()
            case _:
                raise ValueError(f"Unsupported environment: {self.env_type}")

    def new_agent(self):
        return HPC_PFC(self.agent)

    def new_networks(self):
        online = RecurrentNetwork(self.env_type, jax.random.PRNGKey(self.seed))
        return Networks(online=online, target=deepcopy(online))

    def new_observe_trajectory(self, log_to_mlflow: bool = False):
        match self.env_type:
            case EnvironmentType.WATERMAZE:
                return watermaze_observe_trajectory_factory(log_to_console=True, log_to_mlflow=log_to_mlflow)
            case _:
                raise ValueError(f"Unsupported environment: {self.env_type}")


def run_experiment(
    cfg: ExperimentConfig,
) -> tuple[AgentState, dict[str, list[float | int]]]:
    """Run a Fluid 1.1 experiment.

    Args:
        cfg: Experiment configuration.

    Returns:
        Final agent state and metrics dictionary.
    """
    logger = None

    if cfg.mlflow:
        import mlflow
        import mlflow.entities

        mlflow.set_tracking_uri("databricks")
        experiment_name = cfg.mlflow_experiment_name or f"/hpc_pfc_{cfg.env_type.name.lower()}"
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=cfg.mlflow_run_name)
        current_run = mlflow.active_run()
        mlflow_run_id = ""
        if current_run:
            mlflow_run_id = current_run.info.run_id
        logger = MlflowMetricLogger(client=mlflow.MlflowClient(), run_id=mlflow_run_id)

        # Log config parameters
        config_dict = dataclasses.asdict(cfg)
        env_params = config_dict.pop("env")
        for k, v in env_params.items():
            config_dict[f"env/{k}"] = v
        agent_params = config_dict.pop("agent")
        for k, v in agent_params.items():
            config_dict[f"agent/{k}"] = v
        mlflow.log_params(config_dict)

    env = cfg.new_env()
    agent = cfg.new_agent()
    kg = keygen(jax.random.PRNGKey(cfg.seed))

    loop = TrainingLoop(
        env=env,
        env_params=cfg.env,
        agent=agent,
        num_envs=cfg.num_envs,
        key=next(kg),
        inference=cfg.inference,
        logger=logger,
    )

    nets = cfg.new_networks()
    env_info = env_info_from_gymnax(env, cfg.env, cfg.num_envs)
    agent_state = agent.new_state(nets, env_info, next(kg), inference=cfg.inference)

    agent_state, metrics = loop.run(
        agent_state,
        cfg.num_cycles,
        cfg.steps_per_cycle,
        observe_trajectory=cfg.new_observe_trajectory(log_to_mlflow=cfg.mlflow),
    )

    if cfg.mlflow:
        import mlflow

        mlflow.flush_async_logging()
        mlflow.end_run()

    return agent_state, metrics


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for the Fluid 1.1 experiment."""
    # Watermaze experiment configuration
    # These hyperparameters match run_experiment_learned.py exactly
    env_params = EpisodicWaterMazeParams(
        trials_per_episode=10,
        noise_rate=0.0,
        noise_magnitude=0.0,
        max_steps_in_episode=20,
        target_visible=False,
        exploit_probability=0.5,
        forget_after=5,
    )

    steps_per_cycle = env_params.max_steps_in_episode * 5

    agent_config = Config(
        # Note, we are still storing long term memories for terminal states.
        # See the comment in the config for more details.
        store_mem_probability=0,
        optimizer_name="adam",
        optimizer_kwargs={"learning_rate": 3e-5},
        env_type=EnvironmentType.WATERMAZE,
        num_envs=1,
        num_actions=4,
        # Needs to be larger than the max steps in an episode because in Watermaze, there is
        # environment state that is preserved across episodes. Additionally, in _loss
        # we start with a hidden state initialized to 0, whereas during inference
        # we preserve the hidden state across all steps without reset. This results
        # in the initial hidden state for training being biased, and it requires
        # a large amount of steps to correct for that initial bias. Therefore, the results
        # on the first episode in the sequence won't be as accurate as
        # subsequent episodes.
        replay_sequence_length=env_params.max_steps_in_episode * 10,
        num_off_policy_updates_per_cycle=2,
        target_update_interval=100,
        replay_buffer_capacity=env_params.max_steps_in_episode * 10,
        epsilon_transition_steps=env_params.max_steps_in_episode * 50,
        total_samples=1000,  # This should be larger once you make exploration_coef non-0
        store_obs_probability=1.0,
        memory_capacity=env_params.max_steps_in_episode * env_params.forget_after,
        use_hierarchical_memory=False,
        hierarchical_memory_depth=None,
        hierarchical_memory_k=None,
        hierarchical_memory_auto_refit_interval=None,
        use_online_density_in_reward=False,
        use_offline_density_in_reward=False,
    )

    cfg = ExperimentConfig(
        agent=agent_config,
        env=env_params,
        num_envs=1,
        env_type=EnvironmentType.WATERMAZE,
        num_cycles=800000,
        steps_per_cycle=steps_per_cycle,
        seed=0,
        mlflow=True,  # Set to True to enable MLflow logging
        mlflow_run_name=None,
        mlflow_experiment_name="/hpc_pfc_jax",
        inference=False,
        checkpoint=False,
    )

    print("Starting Fluid 1.1 experiment...")
    print(f"Environment: {cfg.env_type.name}")
    print(f"Cycles: {cfg.num_cycles}")
    print(f"Steps per cycle: {cfg.steps_per_cycle}")

    agent_state, metrics = run_experiment(cfg)

    print("\nExperiment complete!")
    if "explore_excess_step_means" in metrics:
        final_explore = metrics["explore_excess_step_means"][-1] if metrics["explore_excess_step_means"] else "N/A"
        print(f"Final explore excess steps: {final_explore}")
    if "exploit_excess_step_means" in metrics:
        final_exploit = metrics["exploit_excess_step_means"][-1] if metrics["exploit_excess_step_means"] else "N/A"
        print(f"Final exploit excess steps: {final_exploit}")


if __name__ == "__main__":
    main()
