"""HPC-PFC Model Agent implementation in PyTorch.

This agent combines:
- Reservoir computing (fixed random RNN)
- Episodic memory with learned Q/K transformations
- Multi-head attention for memory integration
- Double DQN for learning
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.config import Config, EnvironmentType
from agent.state import AgentState, StepState, OptState, ExperienceState
from agent.networks import RecurrentNetwork
from memory.episodic_memory import FlatQKLearnedMemory
from memory.circular_buffer import TensorCircularBuffer
from utils.helpers import soft_update


class HPC_PFC:
    """HPC-PFC Model reinforcement learning agent.

    This agent combines reservoir computing with episodic memory and
    Double DQN for off-policy learning.

    Args:
        config: Agent configuration.
        device: Device to run on.
    """

    def __init__(self, config: Config, device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks will be set via new_state
        self.online: Optional[RecurrentNetwork] = None
        self.target: Optional[RecurrentNetwork] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def new_state(
        self,
        online: RecurrentNetwork,
        num_envs: int,
        seed: int = 0,
    ) -> AgentState:
        """Initialize agent state.

        Args:
            online: The online network.
            num_envs: Number of parallel environments.
            seed: Random seed.

        Returns:
            Initialized AgentState.
        """
        # Store networks
        self.online = online.to(self.device)
        self.target = copy.deepcopy(online).to(self.device)

        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

        # Create optimizer
        optimizer_cls = getattr(torch.optim, self.config.optimizer_name.capitalize(), torch.optim.Adam)
        self.optimizer = optimizer_cls(
            self.online.parameters(),
            **self.config.optimizer_kwargs,
        )

        # Initialize step state
        episodic_memory = FlatQKLearnedMemory.create(
            batch_size=num_envs,
            capacity=self.config.memory_capacity,
            nkey_features=online.data_size,
            nvalue_features=online.reservoir_size,
            device=self.device,
        )

        step_state = StepState(
            episodic_memory=episodic_memory,
            hidden=torch.zeros(num_envs, online.reservoir_size, device=self.device),
            init_hidden=torch.zeros(num_envs, online.reservoir_size, device=self.device),
            t=0,
            obs_buffer=TensorCircularBuffer.create(
                batch_size=num_envs,
                capacity=self.config.total_samples,
                feature_shape=(online.data_size,),
                device=self.device,
            ),
            h_buffer=TensorCircularBuffer.create(
                batch_size=num_envs,
                capacity=self.config.total_samples,
                feature_shape=(online.reservoir_size,),
                device=self.device,
            ),
            obs_buffer_exploit=TensorCircularBuffer.create(
                batch_size=num_envs,
                capacity=self.config.total_samples,
                feature_shape=(online.data_size,),
                device=self.device,
            ),
            online_density_buffer=TensorCircularBuffer.create(
                batch_size=num_envs,
                capacity=self.config.total_samples,
                feature_shape=(1,),
                device=self.device,
            ),
        )

        return AgentState(
            step=step_state,
            opt=OptState(),
            experience=ExperienceState(max_capacity=self.config.replay_buffer_capacity),
        )

    @torch.no_grad()
    def step(
        self,
        state: AgentState,
        obs: torch.Tensor,
        reward: torch.Tensor,
        prev_action: torch.Tensor,
        done: torch.Tensor,
        trial_type: Optional[torch.Tensor] = None,
        is_explore: bool = True,
    ) -> Tuple[torch.Tensor, AgentState]:
        """Select actions for a batch of observations.

        Args:
            state: Current agent state.
            obs: Observations of shape (batch_size, obs_dim).
            reward: Previous rewards of shape (batch_size,).
            prev_action: Previous actions of shape (batch_size,).
            done: Episode done flags of shape (batch_size,).
            trial_type: Trial type (EXPLORE=0, EXPLOIT=1).
            is_explore: Whether this is an exploration trial.

        Returns:
            Tuple of (actions, updated_state).
        """
        self.online.eval()

        batch_size = obs.shape[0]
        obs = obs.to(self.device)
        reward = reward.to(self.device)

        # Preprocess observation
        x_feat = self.online.preprocess(obs)

        # Compute filter modulation
        m_main = self.online.compute_filter()

        # Prepare inputs
        prev_reward = reward.unsqueeze(-1)
        prev_density = torch.zeros_like(prev_reward)  # Simplified: no density exploration
        prev_action_onehot = F.one_hot(prev_action.long().to(self.device), self.config.num_actions).float()

        # Run RNN
        hidden = self.online.rnn(
            prev_reward,
            prev_density,
            prev_action_onehot,
            x_feat,
            m_main,
            state.step.hidden,
        )

        # Query episodic memory using learned PFC query
        query = self.online._pfc_q_mlp(x_feat)
        em_hidden = state.step.episodic_memory.learned_recall(
            query,
            self.online._pfc_k_mlp,
            gate_fn=None,
            n_per_key=self.config.n_per_key,
        )

        # Embed hidden states
        wm_hidden = self.online._emb1(hidden).unsqueeze(1)  # (batch, 1, mem_size)
        em_hidden_emb = self.online._emb1(em_hidden.squeeze(1)).unsqueeze(1)  # (batch, 1, mem_size)

        # Apply attention
        attn_hidden = self.online.attention(wm_hidden, em_hidden_emb)

        # Compute Q-values
        q_values = self.online.classifier(attn_hidden.squeeze(1))

        # Epsilon-greedy action selection
        epsilon = self.config.epsilon_schedule(state.step.t)
        rand_actions = torch.randint(0, self.config.num_actions, (batch_size,), device=self.device)
        greedy_actions = q_values.argmax(dim=-1)

        take_random = torch.rand(batch_size, device=self.device) < epsilon
        actions = torch.where(take_random, rand_actions, greedy_actions)

        # Update episodic memory if episode is done (explore trials only)
        new_episodic_memory = state.step.episodic_memory
        if done.any() and is_explore:
            # Store observation features as key, hidden state as value
            new_episodic_memory = new_episodic_memory.store(
                x_feat.detach(),
                hidden.detach(),
            )

        # Update init_hidden at start of episode
        new_init_hidden = state.step.init_hidden
        if state.step.t == 0:
            new_init_hidden = hidden.detach()

        # Create updated step state
        new_step = StepState(
            episodic_memory=new_episodic_memory,
            hidden=hidden.detach(),
            init_hidden=new_init_hidden,
            t=state.step.t + 1,
            obs_buffer=state.step.obs_buffer,
            h_buffer=state.step.h_buffer,
            obs_buffer_exploit=state.step.obs_buffer_exploit,
            online_density_buffer=state.step.online_density_buffer,
        )

        new_state = AgentState(
            step=new_step,
            opt=state.opt,
            experience=state.experience,
        )

        return actions, new_state

    def update_experience(
        self,
        state: AgentState,
        trajectory: dict,
    ) -> ExperienceState:
        """Update experience replay buffer.

        Args:
            state: Current agent state.
            trajectory: Dictionary containing trajectory data.

        Returns:
            Updated ExperienceState.
        """
        state.experience.add(trajectory, priority=1.0)
        return state.experience

    def compute_loss(
        self,
        state: AgentState,
        trajectory: dict,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute Double DQN loss.

        Args:
            state: Current agent state.
            trajectory: Dictionary containing trajectory data.

        Returns:
            Tuple of (loss, metrics_dict).
        """
        self.online.train()

        # Extract data from trajectory
        obs = trajectory["obs"].to(self.device)  # (seq_len, batch, obs_dim)
        actions = trajectory["actions"].to(self.device)  # (seq_len, batch)
        rewards = trajectory["rewards"].to(self.device)  # (seq_len, batch)
        dones = trajectory["dones"].to(self.device)  # (seq_len, batch)

        seq_len, batch_size = obs.shape[:2]

        # Initialize hidden states
        hidden = torch.zeros(batch_size, self.online.reservoir_size, device=self.device)
        hidden_target = torch.zeros(batch_size, self.target.reservoir_size, device=self.device)

        # Compute filter modulations
        m_main_online = self.online.compute_filter()
        m_main_target = self.target.compute_filter()

        # Lists to collect Q-values
        q_values_online = []
        q_values_target = []

        # Process sequence
        for t in range(seq_len):
            x_feat = self.online.preprocess(obs[t])
            x_feat_target = self.target.preprocess(obs[t])

            # Prepare inputs
            prev_reward = rewards[t].unsqueeze(-1)
            prev_density = torch.zeros_like(prev_reward)

            if t > 0:
                prev_action = actions[t - 1]
            else:
                prev_action = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            prev_action_onehot = F.one_hot(prev_action, self.config.num_actions).float()

            # Run RNNs
            hidden = self.online.rnn(
                prev_reward,
                prev_density,
                prev_action_onehot,
                x_feat,
                m_main_online,
                hidden,
            )
            hidden_target = self.target.rnn(
                prev_reward,
                prev_density,
                prev_action_onehot,
                x_feat_target,
                m_main_target,
                hidden_target,
            )

            # Query episodic memory
            query_online = self.online._pfc_q_mlp(x_feat)
            query_target = self.target._pfc_q_mlp(x_feat_target)

            em_hidden_online = state.step.episodic_memory.learned_recall(
                query_online, self.online._pfc_k_mlp, None, self.config.n_per_key
            )
            em_hidden_target = state.step.episodic_memory.learned_recall(
                query_target, self.target._pfc_k_mlp, None, self.config.n_per_key
            )

            # Embed and attend
            wm_online = self.online._emb1(hidden).unsqueeze(1)
            em_online = self.online._emb1(em_hidden_online.squeeze(1)).unsqueeze(1)
            attn_online = self.online.attention(wm_online, em_online)

            # NOTE: JAX uses ONLINE's embedding and attention for selector, but TARGET's classifier
            # This is intentional per the comment: "using online is actually better"
            wm_target_emb = self.online._emb1(hidden_target).unsqueeze(1)
            em_target_emb = self.online._emb1(em_hidden_target.squeeze(1)).unsqueeze(1)
            attn_target = self.online.attention(wm_target_emb, em_target_emb)

            # Compute Q-values
            q_online = self.online.classifier(attn_online.squeeze(1))
            # Selector uses TARGET classifier but ONLINE embedding/attention
            q_target = self.target.classifier(attn_target.squeeze(1))

            q_values_online.append(q_online)
            q_values_target.append(q_target)

        # Stack Q-values: (seq_len, batch, num_actions)
        q_values_online = torch.stack(q_values_online)
        q_values_target = torch.stack(q_values_target)

        # Double DQN loss computation
        # Q(s_t, a_t) using online network
        q_tm1 = q_values_online[:-1]  # (seq_len-1, batch, num_actions)
        a_tm1 = actions[:-1]  # (seq_len-1, batch)

        # Gather Q-values for taken actions
        q_tm1_a = q_tm1.gather(-1, a_tm1.unsqueeze(-1)).squeeze(-1)  # (seq_len-1, batch)

        # Target: r + gamma * Q_target(s_{t+1}, argmax_a Q_online(s_{t+1}, a))
        q_t_online = q_values_online[1:]  # For action selection
        q_t_target = q_values_target[1:]  # For value estimation

        # Select best actions using online network
        best_actions = q_t_online.argmax(dim=-1)  # (seq_len-1, batch)

        # Evaluate using target network
        q_t_best = q_t_target.gather(-1, best_actions.unsqueeze(-1)).squeeze(-1)

        # Compute targets
        r_t = rewards[1:]  # (seq_len-1, batch)
        done_t = dones[1:].float()  # (seq_len-1, batch)
        targets = r_t + self.config.discount * (1 - done_t) * q_t_best.detach()

        # TD error with Huber loss
        td_errors = targets - q_tm1_a
        loss = F.smooth_l1_loss(q_tm1_a, targets.detach())

        # Add filter loss regularization
        filter_loss = self.config.filter_loss_coef * m_main_online.mean()
        total_loss = loss + filter_loss

        metrics = {
            "dqn_loss": loss.item(),
            "filter_loss": filter_loss.item(),
            "td_error_mean": td_errors.mean().item(),
            "q_values_mean": q_tm1_a.mean().item(),
        }

        return total_loss, metrics

    def optimize(self, loss: torch.Tensor) -> None:
        """Perform optimization step.

        Args:
            loss: Loss to backpropagate.
        """
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_value_(self.online.parameters(), 1.0)

        self.optimizer.step()

    def update_target(self, state: AgentState) -> OptState:
        """Update target network using Polyak averaging.

        Args:
            state: Current agent state.

        Returns:
            Updated OptState.
        """
        new_count = state.opt.target_update_count + 1

        if new_count % self.config.target_update_interval == 0:
            soft_update(
                self.target,
                self.online,
                self.config.target_update_step_size,
            )

        return OptState(target_update_count=new_count)

    def num_off_policy_optims_per_cycle(self) -> int:
        """Number of off-policy updates per training cycle."""
        return self.config.num_off_policy_updates_per_cycle

    def save(self, path: str) -> None:
        """Save agent state to file.

        Args:
            path: Path to save file.
        """
        torch.save(
            {
                "online_state_dict": self.online.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent state from file.

        Args:
            path: Path to load file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.online.load_state_dict(checkpoint["online_state_dict"])
        self.target.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
