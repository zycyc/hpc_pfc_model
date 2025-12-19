"""Reservoir with local connectivity pattern.

This implements a fixed-weight Echo State Network with local connectivity
structure, which provides temporal dynamics for the HPC-PFC Model agent.
"""

import torch
import torch.nn as nn


def _rolling_mask(input_size: int, num_unique: int, num_shared: int) -> torch.Tensor:
    """Create a rolling mask for input-to-hidden connectivity.

    Args:
        input_size: Size of input.
        num_unique: Number of unique connections per input.
        num_shared: Number of shared connections between adjacent inputs.

    Returns:
        Boolean mask tensor of shape (hidden_size, input_size).
    """
    assert input_size > 0, "'input_size' must be a positive integer"
    assert isinstance(num_unique, int), "'num_unique' must be an integer"
    assert isinstance(num_shared, int), "'num_shared' must be an integer"
    assert num_unique + num_shared > 0, "'num_unique' + 'num_shared' must be greater than 0"

    identity = torch.eye(input_size, dtype=torch.float32)
    ones = torch.ones((num_unique + num_shared, 1), dtype=torch.float32)
    wmask = torch.kron(identity, ones)
    wmask = wmask.bool() | torch.roll(wmask, num_shared, dims=0).bool()
    return wmask


def _band_2d(shape: tuple[int, int], width: int) -> torch.Tensor:
    """Create a 2D banded matrix mask.

    Args:
        shape: Shape of the matrix (rows, cols).
        width: Band width around diagonal.

    Returns:
        Boolean tensor where entries within `width` of diagonal are True.
    """
    i = torch.arange(shape[0]).unsqueeze(1)
    j = torch.arange(shape[1]).unsqueeze(0)
    return torch.abs(i - j) < width


def _random_band_tensor(
    m: int,
    band_reach: int,
    generator: torch.Generator,
    uniform_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """Create a random banded matrix.

    Args:
        m: Matrix size (m x m).
        band_reach: Band width.
        generator: Random number generator.
        uniform_range: Range for uniform sampling.

    Returns:
        Banded random matrix of shape (m, m).
    """
    assert m >= 0, "'m' must be a non-negative integer"
    assert 0 <= band_reach <= m, "'band_reach' must be a non-negative integer bounded by 'm'"

    shape = (m, m)
    band_mask = _band_2d(shape, band_reach)
    random_values = torch.empty(shape).uniform_(uniform_range[0], uniform_range[1], generator=generator)
    return torch.where(band_mask, random_values, torch.zeros(shape))


class ReservoirLocalConnectivity(nn.Module):
    """Reservoir (Echo State Network) with local connectivity structure.

    The reservoir is a FIXED random recurrent network. Its weights are
    initialized once and never updated during training. We use register_buffer
    to store the weights so they are part of the module state but not trainable.

    Key properties:
    - Local banded connectivity for hidden-to-hidden connections
    - Sparse global connections added on top
    - Spectral normalization: divide weight_hh by principal eigenvalue for stability
    - Rolling mask pattern for input-to-hidden connections

    Args:
        input_size: Size of input features.
        num_unique: Number of unique hidden units per input.
        num_shared: Number of shared units between adjacent inputs.
        reach: Band width for local hidden-to-hidden connections.
        input_conn_prob: Probability of input-to-hidden connection.
        local_conn_prob: Probability of local hidden-to-hidden connection.
        global_conn_prob: Probability of global hidden-to-hidden connection.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        input_size: int = 24,
        num_unique: int = 40,
        num_shared: int = 20,
        reach: int = 10,
        input_conn_prob: float = 0.5,
        local_conn_prob: float = 0.5,
        global_conn_prob: float = 0.01,
        seed: int = 0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = input_size * (num_unique + num_shared)

        # Use a generator for reproducibility
        generator = torch.Generator().manual_seed(seed)

        # Create rolling mask for input connectivity
        wmask = _rolling_mask(input_size, num_unique, num_shared)

        # Create banded hidden-to-hidden weights
        hidden_size = self.hidden_size
        rnn_weight_hh = _random_band_tensor(hidden_size, reach, generator)

        # Apply local connectivity probability
        local_mask = torch.bernoulli(torch.full((hidden_size, hidden_size), local_conn_prob), generator=generator)
        rnn_weight_hh = local_mask * rnn_weight_hh

        # Add sparse global connections
        global_weights = torch.empty((hidden_size, hidden_size)).uniform_(-1.0, 1.0, generator=generator)
        global_mask = torch.bernoulli(torch.full((hidden_size, hidden_size), global_conn_prob), generator=generator)
        rnn_weight_hh = rnn_weight_hh + global_mask * global_weights

        # Spectral normalization: divide by principal eigenvalue for stability
        # This ensures the spectral radius is <= 1
        eigvals = torch.linalg.eigvals(rnn_weight_hh)
        principal_eigvalue = torch.max(torch.real(eigvals)).item()
        if abs(principal_eigvalue) > 1e-10:
            rnn_weight_hh = rnn_weight_hh / principal_eigvalue

        # Create input-to-hidden weights with Glorot initialization
        rnn_weight_ih = (
            torch.nn.init.xavier_uniform_(
                torch.empty((hidden_size, input_size)),
                generator=generator,
            )
            * 10
        )  # Scale factor from original implementation

        # Apply input connectivity probability
        input_mask = torch.bernoulli(torch.full((hidden_size, input_size), input_conn_prob), generator=generator)
        rnn_weight_ih = input_mask * rnn_weight_ih

        # Apply rolling mask
        rnn_weight_ih = rnn_weight_ih * wmask.float()

        # Register as buffers (not parameters!) since they are FIXED
        self.register_buffer("weight_hh", rnn_weight_hh)
        self.register_buffer("weight_ih", rnn_weight_ih)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Single step of reservoir dynamics.

        Args:
            x: Input tensor of shape (batch_size, input_size).
            hidden: Hidden state of shape (batch_size, hidden_size).

        Returns:
            New hidden state of shape (batch_size, hidden_size).
        """
        # h_{t+1} = tanh(W_ih @ x + W_hh @ h_t)
        return torch.tanh(torch.matmul(x, self.weight_ih.T) + torch.matmul(hidden, self.weight_hh.T))

    def init_hidden(self, batch_size: int, device: torch.device = None) -> torch.Tensor:
        """Initialize hidden state to zeros.

        Args:
            batch_size: Batch size.
            device: Device to create tensor on.

        Returns:
            Zero tensor of shape (batch_size, hidden_size).
        """
        if device is None:
            device = self.weight_hh.device
        return torch.zeros(batch_size, self.hidden_size, device=device)
