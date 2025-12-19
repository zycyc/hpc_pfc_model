# HPC-PFC Model PyTorch Implementation

A standalone PyTorch implementation of the HPC-PFC Model for reinforcement learning with episodic memory. We recommend using the jax implementation for replication of the results in the paper. The PyTorch version is provided for conceptual clarity and ease of modification. For the watermaze RL environment, it takes too long to be run practically (>10 days on a single H100 GPU).

## Overview

The HPC-PFC Model is a reinforcement learning agent that combines:

1. **Reservoir Computing**: A fixed random recurrent network (Echo State Network / Local Connectivity Reservoir) that provides temporal dynamics
2. **Episodic Memory**: A key-value memory system for storing and retrieving past experiences
3. **Learned Query/Key Transformations**: MLPs that learn how to query the episodic memory
4. **Multi-Head Attention**: Combines working memory (current hidden state) with retrieved episodic memories
5. **Double DQN**: Off-policy learning with target network updates

Note: Readers do not need to be concerned with `hierarchical memory` or `density` in the codebase; these features were included for other purposes and were not used in the paper.

## Installation

```bash
uv sync --all-extras
```

## Usage

### Quick Start

```bash
# Replace 0 with the GPU number you want to use
CUDA_VISIBLE_DEVICES=0 uv run run_experiment.py --mlflow
```

## Project Structure

```
torch/
├── pyproject.toml          # Dependencies
├── README.md               # This file
├── run_experiment.py       # Main entry point
├── agent/
│   ├── __init__.py
│   ├── hpc_pfc.py          # HPC_PFC agent class
│   ├── networks.py         # RecurrentNetwork, MoELayer, MLPLayer
│   ├── config.py           # Config dataclass
│   └── state.py            # AgentState, StepState containers
├── memory/
│   ├── __init__.py
│   ├── episodic_memory.py  # FlatQKLearnedMemory
│   ├── circular_buffer.py  # TensorCircularBuffer
│   └── attention.py        # Attention with rotary embeddings
├── reservoir/
│   ├── __init__.py
│   └── local_connectivity.py  # ReservoirLocalConnectivity
├── env/
│   ├── __init__.py
│   └── episodic_water_maze.py  # EpisodicWaterMaze (Gymnasium-based)
├── training/
│   ├── __init__.py
│   ├── loop.py             # Training loop
│   ├── replay_buffer.py    # Sequence replay buffer
│   └── losses.py           # double_q_learning, huber_loss
└── utils/
    ├── __init__.py
    └── helpers.py          # Utility functions
```

## Environment: EpisodicWaterMaze

A gridworld navigation environment with two phases:
- **EXPLORE**: Agent explores a new maze, learns target location
- **EXPLOIT**: Agent navigates to a previously seen target location

The environment applies a rule-based linear transformation to the observation tag based on a random binary indicator, which the agent must learn to handle.

### Observation Space

- Visible subgrid (3x3 around agent): 9 values
- Tag (random vector identifying the maze): 8 values
- Random binary (rule indicator): 1 value
- **Total: 18 dimensions**

### Action Space

4 discrete actions: UP (0), DOWN (1), LEFT (2), RIGHT (3)

### Reward

- +1 for reaching target
- -1/max_steps_in_episode per step otherwise

## Configuration Parameters

Key parameters from the original implementation:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `data_size` | 18 | 9 (grid) + 8 (tag) + 1 (binary rule) |
| `action_size` | 4 | UP, DOWN, LEFT, RIGHT |
| `mem_size` | 64 | Working memory embedding size |
| `reservoir_size` | ~1440 | 24 * (40 + 20) for LocalConnectivity |
| `discount` | 0.9 | γ for Q-learning |
| `learning_rate` | 3e-5 | Adam optimizer |
| `epsilon_start` | 1.0 | Initial exploration |
| `epsilon_end` | 0.05 | Final exploration |
| `target_update_interval` | 100 | Steps between target updates |

## Verification

The implementation should show:
- Decreasing `exploit_excess_step_means` over training
- Agent learns to navigate to remembered target locations
- Exploit trials should show faster performance than explore trials after training
