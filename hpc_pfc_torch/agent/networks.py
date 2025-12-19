"""Neural network components for HPC-PFC Model agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.config import EnvironmentType
from memory.attention import Attention
from reservoir import ReservoirLocalConnectivity


def make_mlp(
    input_size: int,
    output_size: int,
    hidden_size: int,
    depth: int,
    activation: str = "relu",
) -> nn.Sequential:
    """Create an MLP with specified architecture.

    Args:
        input_size: Input dimension.
        output_size: Output dimension.
        hidden_size: Hidden layer dimension.
        depth: Number of hidden layers.
        activation: Activation function name.

    Returns:
        nn.Sequential containing the MLP.
    """
    if activation == "relu":
        act_fn = nn.ReLU
    elif activation == "tanh":
        act_fn = nn.Tanh
    elif activation == "sigmoid":
        act_fn = nn.Sigmoid
    else:
        raise ValueError(f"Unknown activation: {activation}")

    layers = []

    # First layer
    layers.append(nn.Linear(input_size, hidden_size))
    layers.append(act_fn())

    # Hidden layers
    for _ in range(depth - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(act_fn())

    # Output layer
    layers.append(nn.Linear(hidden_size, output_size))

    return nn.Sequential(*layers)


class MLPLayer(nn.Module):
    """Wrapper for MLP to make it compatible with MoE interface."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        depth: int = 2,
    ):
        super().__init__()
        self.layer = make_mlp(input_size, output_size, hidden_size, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class MoELayer(nn.Module):
    """Mixture of Experts layer with learned gating.

    Two-expert MoE where the output is a weighted combination of
    expert outputs based on a learned gating network.

    Args:
        input_size: Input dimension.
        output_size: Output dimension.
        hidden_size: Hidden dimension for experts and gate.
        depth: Depth of expert MLPs.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        depth: int = 2,
    ):
        super().__init__()

        self.gate = make_mlp(input_size, 2, hidden_size, depth)
        self.expert0 = make_mlp(input_size, output_size, hidden_size, depth)
        self.expert1 = make_mlp(input_size, output_size, hidden_size, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., input_size).

        Returns:
            Output tensor of shape (..., output_size).
        """
        gate_weights = F.softmax(self.gate(x), dim=-1)
        expert0_out = self.expert0(x)
        expert1_out = self.expert1(x)

        # Weighted combination
        return gate_weights[..., 0:1] * expert0_out + gate_weights[..., 1:2] * expert1_out


class RecurrentNetwork(nn.Module):
    """Main recurrent network for HPC-PFC Model agent.

    This network combines:
    - Reservoir computing (fixed-weight RNN)
    - Filter network (input modulation)
    - PFC query/key transformations (MLP or MoE)
    - Multi-head attention for memory integration
    - Q-value classifier

    Args:
        env_type: Type of environment (determines architecture sizes).
        seed: Random seed for initialization.
    """

    def __init__(
        self,
        env_type: EnvironmentType = EnvironmentType.WATERMAZE,
        seed: int = 0,
    ):
        super().__init__()

        self.env_type = env_type
        self._cached_filter = None

        # Initialize environment-specific parameters
        if env_type == EnvironmentType.WATERMAZE:
            self._init_watermaze()
        elif env_type == EnvironmentType.DISCRETE_MAZE:
            self._init_discrete_maze()
        elif env_type == EnvironmentType.CRAFTER:
            self._init_crafter()
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")

        # Compute RNN input size
        reward_size = 2  # task + exploration reward
        self._rnn_input_size = self.action_size + reward_size + self.data_size

        # Initialize reservoir (fixed weights)
        if env_type != EnvironmentType.CRAFTER:
            self._rnn = ReservoirLocalConnectivity(
                input_size=self._rnn_input_size,
                num_unique=40,
                num_shared=20,
                seed=seed,
            )
        else:
            # Crafter uses echo state network - simplified for PyTorch version
            self._rnn = ReservoirLocalConnectivity(
                input_size=self._rnn_input_size,
                num_unique=40,
                num_shared=20,
                seed=seed,
            )

        self.reservoir_size = self._rnn.hidden_size

        # Filter network: modulates RNN input
        # Outputs values between minimum and maximum
        self.filter = nn.Sequential(
            nn.Linear(self._rnn_input_size * 2, self._layer_size),
            nn.ReLU(),
            nn.Linear(self._layer_size, self._layer_size),
            nn.ReLU(),
            nn.Linear(self._layer_size, self._rnn_input_size),
            nn.Sigmoid(),
        )
        self._filter_min = 0.25
        self._filter_max = 5.0

        # HD bias vector (learned)
        self.hd_bias = nn.Parameter(torch.randn(self._rnn_input_size * 2))

        # Classifier: Q-value prediction
        self._classifier = nn.Sequential(
            nn.Linear(self.mem_size, self._layer_size),
            nn.ReLU(),
            nn.Linear(self._layer_size, self._layer_size),
            nn.ReLU(),
            nn.Linear(self._layer_size, self._layer_size),
            nn.ReLU(),
            nn.Linear(self._layer_size, self.action_size),
        )

        # Attention for combining working memory and episodic memory
        self._attn1 = Attention(
            qk_size=32,
            vo_size=self.mem_size,
            num_heads=1,
        )

        # Embedding from reservoir to memory size
        self._emb1 = make_mlp(
            self.reservoir_size,
            self.mem_size,
            self.reservoir_size,
            depth=3,
        )

        # PFC query/key MLPs (for learned memory retrieval)
        self._pfc_q_mlp = MLPLayer(
            self.data_size,
            self.data_size,
            self.data_size,
            depth=2,
        )
        self._pfc_k_mlp = MLPLayer(
            self.data_size,
            self.data_size,
            self.data_size,
            depth=2,
        )

        # PFC query/key MoEs (alternative to MLPs)
        self._pfc_q_moe = MoELayer(
            self.data_size,
            self.data_size,
            self.data_size,
            depth=2,
        )
        self._pfc_k_moe = MoELayer(
            self.data_size,
            self.data_size,
            self.data_size,
            depth=2,
        )

        # Episodic memory retrieval gating
        self._em_gate = nn.Sequential(
            make_mlp(3 + self.data_size, 1, 32, depth=2),
            nn.Sigmoid(),
        )

        # Fixed transformations for tag (rule-based)
        torch.manual_seed(0)
        self._fixed_transform0 = nn.Linear(self.data_size, self.data_size, bias=False)
        torch.manual_seed(1)
        self._fixed_transform1 = nn.Linear(self.data_size, self.data_size, bias=False)

        # Memory projection (optional)
        self.fc_mem = nn.Linear(self.reservoir_size + self.mem_size, self.mem_size)

    def _init_watermaze(self):
        """Initialize for WaterMaze environment."""
        self._layer_size = 256
        self.data_size = 17 + 1  # +1 for random binary dimension
        self.mem_size = 64
        self.action_size = 4
        self._preprocess = nn.Identity()

    def _init_discrete_maze(self):
        """Initialize for DiscreteMaze environment."""
        self._layer_size = 64
        self.data_size = 2
        self.mem_size = 32
        self.action_size = 4
        self._preprocess = nn.Identity()

    def _init_crafter(self):
        """Initialize for Crafter environment."""
        self._layer_size = 256
        self.data_size = 256
        self.mem_size = 256
        self.action_size = 17

        # CNN preprocessing for Crafter images
        self._preprocess = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=4, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(288, 288),
            nn.ReLU(),
            nn.Linear(288, self.data_size),
        )

    def train(self, mode: bool = True):
        """Set training mode and invalidate cached values."""
        super().train(mode)
        # Invalidate filter cache when switching modes
        self._cached_filter = None
        return self

    def preprocess(self, obs: torch.Tensor) -> torch.Tensor:
        """Preprocess observation.

        Args:
            obs: Raw observation tensor.

        Returns:
            Preprocessed features.
        """
        if self.env_type == EnvironmentType.CRAFTER:
            # For Crafter, need to transpose to (C, H, W)
            if obs.dim() == 3:
                obs = obs.permute(2, 0, 1)
            elif obs.dim() == 4:
                obs = obs.permute(0, 3, 1, 2)
        return self._preprocess(obs)

    def compute_filter(self) -> torch.Tensor:
        """Compute filter modulation values.

        Returns:
            Filter values scaled to [_filter_min, _filter_max].

        Note:
            During eval mode, the filter is cached since hd_bias doesn't change.
            During training, we recompute to support gradient flow.
        """
        if not self.training and self._cached_filter is not None:
            return self._cached_filter

        raw_filter = self.filter(self.hd_bias)
        result = (self._filter_max - self._filter_min) * raw_filter + self._filter_min

        if not self.training:
            self._cached_filter = result

        return result

    def rnn(
        self,
        prev_reward: torch.Tensor,
        prev_density: torch.Tensor,
        prev_action: torch.Tensor,
        x_feat: torch.Tensor,
        m_main: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Run one step of the reservoir RNN.

        Args:
            prev_reward: Previous reward of shape (batch_size, 1).
            prev_density: Previous density of shape (batch_size, 1).
            prev_action: Previous action one-hot of shape (batch_size, action_size).
            x_feat: Observation features of shape (batch_size, data_size).
            m_main: Filter modulation of shape (rnn_input_size,).
            hidden: Previous hidden state of shape (batch_size, reservoir_size).

        Returns:
            New hidden state of shape (batch_size, reservoir_size).
        """
        # Concatenate inputs
        x = torch.cat([prev_reward, prev_density, prev_action, x_feat], dim=-1)

        # Apply filter modulation
        x = x * m_main

        # Run RNN
        return self._rnn(x, hidden)

    def classifier(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute Q-values from hidden state.

        Args:
            hidden: Hidden state of shape (..., mem_size).

        Returns:
            Q-values of shape (..., action_size).
        """
        return self._classifier(hidden)

    def attention(
        self,
        wm: torch.Tensor,
        em: torch.Tensor,
    ) -> torch.Tensor:
        """Apply attention between working memory and episodic memory.

        Args:
            wm: Working memory of shape (batch, wm_len, mem_size).
            em: Episodic memory of shape (batch, em_len, mem_size).

        Returns:
            Attention output of shape (batch, 1, mem_size).
        """
        # Combine episodic and working memory (excluding last working memory element)
        if wm.dim() == 2:
            wm = wm.unsqueeze(0)
        if em.dim() == 2:
            em = em.unsqueeze(0)

        # em_plus_wm: concatenate EM and all but the last WM element
        if wm.shape[1] > 1:
            em_plus_wm = torch.cat([em, wm[:, :-1]], dim=1)
        else:
            em_plus_wm = em

        # Query with the last WM element
        return self._attn1(wm[:, -1], em_plus_wm, em_plus_wm)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor,
        prev_reward: torch.Tensor,
        prev_density: torch.Tensor,
        prev_action: torch.Tensor,
        episodic_memory_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            obs: Observation of shape (batch, obs_dim).
            hidden: Hidden state of shape (batch, reservoir_size).
            prev_reward: Previous reward of shape (batch, 1).
            prev_density: Previous density of shape (batch, 1).
            prev_action: Previous action (one-hot) of shape (batch, action_size).
            episodic_memory_hidden: Retrieved episodic memories of shape (batch, n_per_key, reservoir_size).

        Returns:
            Tuple of (q_values, new_hidden) where:
            - q_values: Q-values of shape (batch, action_size)
            - new_hidden: New hidden state of shape (batch, reservoir_size)
        """
        # Preprocess observation
        x_feat = self.preprocess(obs)

        # Compute filter modulation
        m_main = self.compute_filter()

        # Run RNN
        hidden = self.rnn(prev_reward, prev_density, prev_action, x_feat, m_main, hidden)

        # Embed hidden state and episodic memories
        wm_hidden = self._emb1(hidden).unsqueeze(1)  # (batch, 1, mem_size)
        em_hidden = self._emb1(episodic_memory_hidden.view(-1, self.reservoir_size))
        em_hidden = em_hidden.view(hidden.shape[0], -1, self.mem_size)  # (batch, n_per_key, mem_size)

        # Apply attention
        attn_hidden = self.attention(wm_hidden, em_hidden)  # (batch, 1, mem_size)

        # Compute Q-values
        q_values = self.classifier(attn_hidden.squeeze(1))

        return q_values, hidden
