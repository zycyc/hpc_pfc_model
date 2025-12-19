"""HPC-PFC Model JAX - Standalone RL agent with reservoir computing and episodic memory."""

from agent import (
    AgentState,
    AgentStep,
    Config,
    EnvInfo,
    EnvironmentType,
    EnvStep,
    ExperienceState,
    HPC_PFC,
    MLPLayer,
    MoELayer,
    Networks,
    OptState,
    RecurrentNetwork,
    StepState,
    env_info_from_gymnax,
)
from run_experiment import (
    ExperimentConfig,
    MetricLogger,
    MlflowMetricLogger,
    TrainingLoop,
    run_experiment,
)

__all__ = [
    # Agent
    "HPC_PFC",
    "Config",
    "EnvironmentType",
    "RecurrentNetwork",
    "Networks",
    "MoELayer",
    "MLPLayer",
    # State classes
    "AgentState",
    "AgentStep",
    "StepState",
    "OptState",
    "ExperienceState",
    # Types
    "EnvStep",
    "EnvInfo",
    "env_info_from_gymnax",
    # Training
    "TrainingLoop",
    "ExperimentConfig",
    "run_experiment",
    # Logging
    "MetricLogger",
    "MlflowMetricLogger",
]

__version__ = "0.1.0"
