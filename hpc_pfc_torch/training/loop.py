"""Training loop for HPC-PFC Model agent."""

from typing import Callable, Dict, List, Optional, Any
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from agent import HPC_PFC, AgentState, Config, RecurrentNetwork
from env import EpisodicWaterMaze, TrialType
from training.replay_buffer import SequenceReplayBuffer


class TrainingLoop:
    """Training loop for HPC-PFC Model agent on WaterMaze.

    Args:
        env: The WaterMaze environment.
        agent: The HPC-PFC Model agent.
        device: Device to run on.
    """

    def __init__(
        self,
        env: EpisodicWaterMaze,
        agent: HPC_PFC,
        device: torch.device = None,
    ):
        self.env = env
        self.agent = agent
        self.device = device or agent.device

        # Replay buffer
        self.replay_buffer = SequenceReplayBuffer(
            capacity=agent.config.replay_buffer_capacity,
            sequence_length=agent.config.replay_sequence_length,
        )

    def run(
        self,
        agent_state: AgentState,
        num_cycles: int,
        steps_per_cycle: int,
        observe_trajectory: Optional[Callable] = None,
        log_interval: int = 100,
        mlflow_logging: bool = False,
    ) -> tuple[AgentState, Dict[str, List[float]]]:
        """Run training loop.

        Args:
            agent_state: Initial agent state.
            num_cycles: Number of training cycles.
            steps_per_cycle: Environment steps per cycle.
            observe_trajectory: Optional callback for trajectory observation.
            log_interval: How often to log metrics.
            mlflow_logging: Whether to log to MLflow.

        Returns:
            Tuple of (final_agent_state, metrics_dict).
        """
        # Metrics tracking
        metrics: Dict[str, List[float]] = defaultdict(list)

        # Episode tracking
        explore_excess_steps: List[float] = []
        exploit_excess_steps: List[float] = []

        # Reset environment
        obs, info = self.env.reset()
        prev_action = torch.zeros(1, dtype=torch.long, device=self.device)
        prev_reward = torch.zeros(1, device=self.device)

        # Training progress bar
        pbar = tqdm(range(num_cycles), desc="Training")

        for cycle in pbar:
            # Collect trajectory for this cycle
            cycle_obs = []
            cycle_actions = []
            cycle_rewards = []
            cycle_dones = []
            cycle_trial_types = []
            cycle_random_binaries = []

            for step in range(steps_per_cycle):
                # Convert observation to tensor
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                reward_tensor = prev_reward
                done_tensor = torch.zeros(1, dtype=torch.bool, device=self.device)

                # Get trial type
                trial_type = info.get("trial_type", TrialType.EXPLORE)
                is_explore = trial_type == TrialType.EXPLORE

                # Select action
                action, agent_state = self.agent.step(
                    agent_state,
                    obs_tensor,
                    reward_tensor,
                    prev_action,
                    done_tensor,
                    is_explore=is_explore,
                )

                # Take environment step
                next_obs, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated

                # Store transition
                cycle_obs.append(obs)
                cycle_actions.append(action.item())
                cycle_rewards.append(reward)
                cycle_dones.append(done)
                cycle_trial_types.append(int(info.get("trial_type", 0)))
                cycle_random_binaries.append(info.get("random_binary", 0))

                # Add to replay buffer
                self.replay_buffer.add(
                    obs=obs,
                    action=action.item(),
                    reward=reward,
                    done=done,
                    trial_type=int(info.get("trial_type", 0)),
                    random_binary=info.get("random_binary", 0),
                )

                # Track excess steps for completed trials
                if done:
                    excess_steps = info.get("excess_steps", -1)
                    if excess_steps >= 0:
                        if info.get("trial_type") == TrialType.EXPLORE:
                            explore_excess_steps.append(excess_steps)
                        else:
                            exploit_excess_steps.append(excess_steps)

                    # Update episodic memory for explore trials
                    if is_explore:
                        x_feat = self.agent.online.preprocess(obs_tensor)
                        agent_state.step.episodic_memory.store_inplace(
                            x_feat.detach(),
                            agent_state.step.hidden.detach(),
                        )

                # Update state
                obs = next_obs
                prev_action = action
                prev_reward = torch.tensor([reward], device=self.device)

            # Call observe_trajectory callback if provided
            if observe_trajectory is not None:
                traj_metrics = observe_trajectory(
                    {"obs": np.array(cycle_obs), "rewards": np.array(cycle_rewards)},
                    {
                        "trial_type": np.array(cycle_trial_types),
                        "excess_steps": np.zeros(len(cycle_dones)) - 1,
                    },
                    cycle * steps_per_cycle,
                )
                if traj_metrics:
                    for k, v in traj_metrics.items():
                        metrics[k].append(v)

            # Off-policy updates
            for _ in range(self.agent.num_off_policy_optims_per_cycle()):
                # Sample from replay buffer
                batch = self.replay_buffer.sample(device=self.device)

                if batch is not None:
                    # Compute loss
                    loss, loss_metrics = self._compute_loss_from_batch(agent_state, batch)
                    # Optimize
                    self.agent.optimize(loss)

                    # Update target network
                    agent_state = AgentState(
                        step=agent_state.step,
                        opt=self.agent.update_target(agent_state),
                        experience=agent_state.experience,
                    )

                    # Track metrics
                    for k, v in loss_metrics.items():
                        metrics[k].append(v)

            # Log progress
            if (cycle + 1) % log_interval == 0:
                avg_loss = np.mean(metrics.get("dqn_loss", [0])[-log_interval:])

                explore_mean = np.mean(explore_excess_steps[-100:]) if explore_excess_steps else 0
                exploit_mean = np.mean(exploit_excess_steps[-100:]) if exploit_excess_steps else 0

                pbar.set_postfix(
                    {
                        "loss": f"{avg_loss:.4f}",
                        "explore_excess": f"{explore_mean:.1f}",
                        "exploit_excess": f"{exploit_mean:.1f}",
                        "eps": f"{self.agent.config.epsilon_schedule(agent_state.step.t):.3f}",
                    }
                )

                # Store summary metrics
                metrics["explore_excess_step_means"].append(explore_mean)
                metrics["exploit_excess_step_means"].append(exploit_mean)

                if mlflow_logging:
                    try:
                        import mlflow

                        mlflow.log_metric("loss", avg_loss, step=cycle)
                        mlflow.log_metric("explore_excess_step_mean", explore_mean, step=cycle)
                        mlflow.log_metric("exploit_excess_step_mean", exploit_mean, step=cycle)
                    except Exception:
                        pass

        return agent_state, dict(metrics)

    def _compute_loss_from_batch(
        self,
        agent_state: AgentState,
        batch: dict,
    ) -> tuple[torch.Tensor, dict]:
        """Compute loss from a batch of data.

        This optimized version batches operations across the sequence dimension
        where possible, minimizing Python overhead.

        Args:
            agent_state: Current agent state.
            batch: Dictionary containing batch data.

        Returns:
            Tuple of (loss, metrics).
        """
        self.agent.online.train()

        obs = batch["obs"]  # (batch, seq_len, obs_dim)
        actions = batch["actions"]  # (batch, seq_len)
        rewards = batch["rewards"]  # (batch, seq_len)
        dones = batch["dones"]  # (batch, seq_len)

        batch_size, seq_len = obs.shape[:2]

        # ========== VECTORIZED PREPROCESSING ==========
        # Flatten batch and sequence dimensions for parallel processing
        obs_flat = obs.reshape(batch_size * seq_len, -1)  # (batch*seq, obs_dim)

        # Preprocess all observations at once
        x_feat_flat = self.agent.online.preprocess(obs_flat)
        x_feat_target_flat = self.agent.target.preprocess(obs_flat)

        # Reshape back to (batch, seq_len, features)
        x_feat_all = x_feat_flat.reshape(batch_size, seq_len, -1)
        x_feat_target_all = x_feat_target_flat.reshape(batch_size, seq_len, -1)

        # ========== VECTORIZED QUERY COMPUTATION ==========
        # Compute all queries at once
        query_online_flat = self.agent.online._pfc_q_mlp(x_feat_flat)
        query_target_flat = self.agent.target._pfc_q_mlp(x_feat_target_flat)

        query_online_all = query_online_flat.reshape(batch_size, seq_len, -1)
        query_target_all = query_target_flat.reshape(batch_size, seq_len, -1)

        # ========== VECTORIZED EPISODIC MEMORY RECALL ==========
        # The episodic memory is fixed during loss computation, so we can batch all recalls
        # Process each timestep's queries in a batched way
        em_hidden_online_list = []
        em_hidden_target_list = []
        for t in range(seq_len):
            em_hidden_online = agent_state.step.episodic_memory.learned_recall(
                query_online_all[:, t],
                self.agent.online._pfc_k_mlp,
                None,
                self.agent.config.n_per_key,
            )
            em_hidden_target = agent_state.step.episodic_memory.learned_recall(
                query_target_all[:, t],
                self.agent.target._pfc_k_mlp,
                None,
                self.agent.config.n_per_key,
            )
            em_hidden_online_list.append(em_hidden_online.squeeze(1))  # (batch, reservoir_size)
            em_hidden_target_list.append(em_hidden_target.squeeze(1))

        # Stack: (batch, seq_len, reservoir_size)
        em_hidden_online_all = torch.stack(em_hidden_online_list, dim=1)
        em_hidden_target_all = torch.stack(em_hidden_target_list, dim=1)

        # ========== VECTORIZED EM EMBEDDING ==========
        # Embed all episodic memory retrievals at once
        em_flat = em_hidden_online_all.reshape(batch_size * seq_len, -1)
        em_target_flat = em_hidden_target_all.reshape(batch_size * seq_len, -1)

        em_online_emb_flat = self.agent.online._emb1(em_flat)
        em_target_emb_flat = self.agent.online._emb1(em_target_flat)

        em_online_emb_all = em_online_emb_flat.reshape(batch_size, seq_len, -1)
        em_target_emb_all = em_target_emb_flat.reshape(batch_size, seq_len, -1)

        # ========== SEQUENTIAL RNN (cannot be vectorized) ==========
        # Compute filter modulations
        m_main_online = self.agent.online.compute_filter()
        m_main_target = self.agent.target.compute_filter()

        # Initialize hidden states
        hidden = torch.zeros(batch_size, self.agent.online.reservoir_size, device=self.device)
        hidden_target = torch.zeros(batch_size, self.agent.target.reservoir_size, device=self.device)

        hiddens_online = []
        hiddens_target = []

        for t in range(seq_len):
            # Prepare inputs
            prev_reward = rewards[:, t].unsqueeze(-1)
            prev_density = torch.zeros_like(prev_reward)

            if t > 0:
                prev_action = actions[:, t - 1]
            else:
                prev_action = torch.zeros(batch_size, dtype=torch.long, device=self.device)

            prev_action_onehot = torch.nn.functional.one_hot(prev_action, self.agent.config.num_actions).float()

            # Run RNNs with pre-computed x_feat
            hidden = self.agent.online.rnn(
                prev_reward,
                prev_density,
                prev_action_onehot,
                x_feat_all[:, t],
                m_main_online,
                hidden,
            )
            hidden_target = self.agent.target.rnn(
                prev_reward,
                prev_density,
                prev_action_onehot,
                x_feat_target_all[:, t],
                m_main_target,
                hidden_target,
            )

            hiddens_online.append(hidden)
            hiddens_target.append(hidden_target)

        # Stack hidden states: (batch, seq_len, reservoir_size)
        hiddens_online = torch.stack(hiddens_online, dim=1)
        hiddens_target = torch.stack(hiddens_target, dim=1)

        # ========== VECTORIZED WM EMBEDDING ==========
        # Embed all hidden states at once
        hiddens_flat = hiddens_online.reshape(batch_size * seq_len, -1)
        hiddens_target_flat = hiddens_target.reshape(batch_size * seq_len, -1)

        wm_online_emb_flat = self.agent.online._emb1(hiddens_flat)
        wm_target_emb_flat = self.agent.online._emb1(hiddens_target_flat)

        wm_online_emb_all = wm_online_emb_flat.reshape(batch_size, seq_len, -1)
        wm_target_emb_all = wm_target_emb_flat.reshape(batch_size, seq_len, -1)

        # ========== VECTORIZED ATTENTION AND CLASSIFIER ==========
        # Process attention for all timesteps
        # Note: attention expects (batch, len, features) inputs
        q_values_online = []
        q_values_target = []

        for t in range(seq_len):
            wm_online = wm_online_emb_all[:, t : t + 1]  # (batch, 1, mem_size)
            em_online = em_online_emb_all[:, t : t + 1]  # (batch, 1, mem_size)
            attn_online = self.agent.online.attention(wm_online, em_online)

            wm_target = wm_target_emb_all[:, t : t + 1]
            em_target = em_target_emb_all[:, t : t + 1]
            attn_target = self.agent.online.attention(wm_target, em_target)

            q_online = self.agent.online.classifier(attn_online.squeeze(1))
            q_target = self.agent.target.classifier(attn_target.squeeze(1))

            q_values_online.append(q_online)
            q_values_target.append(q_target)

        # Stack Q-values: (batch, seq_len, num_actions)
        q_values_online = torch.stack(q_values_online, dim=1)
        q_values_target = torch.stack(q_values_target, dim=1)

        # ========== DOUBLE DQN LOSS ==========
        q_tm1 = q_values_online[:, :-1]
        q_t_value = q_values_online[:, 1:]
        q_t_selector = q_values_target[:, 1:]

        a_tm1 = actions[:, :-1]
        r_t = rewards[:, 1:]
        done_t = dones[:, 1:].float()

        # Select best actions using target network
        best_actions = q_t_selector.argmax(dim=-1)

        # Get values from online network for selected actions
        q_t_best = q_t_value.gather(-1, best_actions.unsqueeze(-1)).squeeze(-1)

        # Compute targets
        targets = r_t + self.agent.config.discount * (1 - done_t) * q_t_best.detach()

        # Get Q-values for taken actions
        q_tm1_a = q_tm1.gather(-1, a_tm1.unsqueeze(-1)).squeeze(-1)

        # Huber loss
        loss = torch.nn.functional.smooth_l1_loss(q_tm1_a, targets)

        # Add filter loss
        filter_loss = self.agent.config.filter_loss_coef * m_main_online.mean()
        total_loss = loss + filter_loss

        metrics = {
            "dqn_loss": loss.item(),
            "filter_loss": filter_loss.item(),
            "q_values_mean": q_tm1_a.mean().item(),
        }

        return total_loss, metrics


def create_training_loop(
    config: Config,
    env_params: dict,
    seed: int = 0,
    device: torch.device = None,
) -> tuple[TrainingLoop, AgentState]:
    """Create training loop with initialized components.

    Args:
        config: Agent configuration.
        env_params: Environment parameters.
        seed: Random seed.
        device: Device to run on.

    Returns:
        Tuple of (training_loop, initial_agent_state).
    """
    from env.episodic_water_maze import Params as EnvParams

    # Create environment
    params = EnvParams(**env_params)
    env = EpisodicWaterMaze(params=params, seed=seed)

    # Create agent
    agent = HPC_PFC(config, device=device)

    # Create network
    network = RecurrentNetwork(config.env_type, seed=seed)

    # Initialize agent state
    agent_state = agent.new_state(network, num_envs=config.num_envs, seed=seed)

    # Create training loop
    loop = TrainingLoop(env, agent, device=device)

    return loop, agent_state
