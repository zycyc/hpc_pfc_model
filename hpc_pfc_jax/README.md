# HPC-PFC Model JAX Implementation

A standalone JAX-based reinforcement learning agent that combines:

## Overview

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

## Running an Experiment

```bash
# Replace 0 with the GPU number you want to use
CUDA_VISIBLE_DEVICES=0 uv run training
```

## Architecture

```
jax/
├── __init__.py             # Package exports
├── agent.py                # Main HPC_PFC agent class
├── run_experiment.py       # Training script with inlined loop
├── env/
│   ├── __init__.py
│   └── water_maze.py       # Episodic Water Maze environment
├── memory/
│   ├── __init__.py
│   ├── base.py             # EpisodicMemory base class
│   ├── flat_qk.py          # FlatQKLearnedMemory implementation
│   └── attention.py        # Multi-head attention module
├── reservoir/
│   ├── __init__.py
│   ├── base.py             # Reservoir base class
│   └── local.py            # ReservoirLocalConnectivity
├── utils/
│   ├── __init__.py
│   ├── prng.py             # PRNG key generator
│   ├── jax_utils.py        # filter_scan, filter_cond utilities
│   ├── rl.py               # RL helper functions
│   ├── buffer.py           # TensorCircularBuffer
│   └── transforms.py       # Shape annotation utilities
├── pyproject.toml
└── uv.lock
```