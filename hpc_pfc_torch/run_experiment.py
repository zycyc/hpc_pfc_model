#!/usr/bin/env python3
"""Main entry point for running HPC-PFC experiments.

Usage:
    python run_experiment.py [--num_cycles N] [--mlflow] [--seed S]

Example:
    python run_experiment.py --num_cycles 10000 --seed 42
"""

import argparse
from dataclasses import dataclass
from typing import Optional

import torch

from agent import Config, RecurrentNetwork, HPC_PFC
from agent.config import EnvironmentType
from env import EpisodicWaterMaze, TrialType
from env.episodic_water_maze import Params as EnvParams
from training import TrainingLoop


@dataclass
class ExperimentConfig:
    """Configuration for a HPC-PFC experiment."""

    # Environment
    env_type: EnvironmentType = EnvironmentType.WATERMAZE
    max_steps_in_episode: int = 20
    trials_per_episode: int = 10
    exploit_probability: float = 0.5
    forget_after: int = 5
    noise_rate: float = 0.0
    noise_magnitude: float = 0.0
    target_visible: bool = False

    # Agent
    num_envs: int = 1
    memory_capacity: int = 100  # max_steps * forget_after
    replay_buffer_capacity: int = 200  # max_steps * 10
    replay_sequence_length: int = 200  # max_steps * 10

    # Training
    num_cycles: int = 800000  # Match JAX default
    steps_per_cycle: int = 100  # max_steps * 5
    epsilon_transition_steps: int = 1000  # max_steps * 50

    # Learning
    learning_rate: float = 3e-5
    discount: float = 0.9
    num_off_policy_updates_per_cycle: int = 2
    target_update_interval: int = 100

    # Misc
    seed: int = 0
    mlflow: bool = False
    mlflow_run_name: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_experiment(cfg: ExperimentConfig):
    """Run a HPC-PFC experiment.

    Args:
        cfg: Experiment configuration.

    Returns:
        Tuple of (final_agent_state, metrics).
    """
    # Enable TF32 for faster matmuls on Ampere+ GPUs (A100, H100)
    torch.set_float32_matmul_precision("high")

    print(f"Running experiment on {cfg.device}")
    print(f"Num cycles: {cfg.num_cycles}")
    print(f"Steps per cycle: {cfg.steps_per_cycle}")

    device = torch.device(cfg.device)

    # Create environment
    env_params = EnvParams(
        trials_per_episode=cfg.trials_per_episode,
        noise_rate=cfg.noise_rate,
        noise_magnitude=cfg.noise_magnitude,
        max_steps_in_episode=cfg.max_steps_in_episode,
        target_visible=cfg.target_visible,
        exploit_probability=cfg.exploit_probability,
        forget_after=cfg.forget_after,
    )
    env = EpisodicWaterMaze(params=env_params, seed=cfg.seed)

    # Create agent configuration
    agent_config = Config(
        env_type=cfg.env_type,
        num_envs=cfg.num_envs,
        num_actions=4,
        memory_capacity=cfg.memory_capacity,
        replay_buffer_capacity=cfg.replay_buffer_capacity,
        replay_sequence_length=cfg.replay_sequence_length,
        epsilon_transition_steps=cfg.epsilon_transition_steps,
        total_samples=1000,
        store_mem_probability=0.0,  # Only store at episode end
        discount=cfg.discount,
        num_off_policy_updates_per_cycle=cfg.num_off_policy_updates_per_cycle,
        target_update_interval=cfg.target_update_interval,
        optimizer_name="adam",
        optimizer_kwargs={"lr": cfg.learning_rate},
    )

    # Create agent
    agent = HPC_PFC(agent_config, device=device)

    # Create network
    network = RecurrentNetwork(cfg.env_type, seed=cfg.seed)

    # Initialize agent state
    agent_state = agent.new_state(network, num_envs=cfg.num_envs, seed=cfg.seed)

    # Create training loop
    loop = TrainingLoop(env, agent, device=device)

    # MLflow setup
    if cfg.mlflow:
        try:
            import mlflow

            mlflow.set_tracking_uri("databricks")
            mlflow.set_experiment(f"/hpc_pfc_torch")
            mlflow.start_run(run_name=cfg.mlflow_run_name)

            # Log config
            mlflow.log_params(
                {
                    "env_type": cfg.env_type.name,
                    "num_cycles": cfg.num_cycles,
                    "steps_per_cycle": cfg.steps_per_cycle,
                    "learning_rate": cfg.learning_rate,
                    "discount": cfg.discount,
                    "seed": cfg.seed,
                }
            )
        except Exception as e:
            print(f"MLflow setup failed: {e}")
            cfg.mlflow = False

    # Define trajectory observer
    def observe_trajectory(env_steps, step_infos, step_num):
        """Observe trajectory and compute metrics."""
        metrics = {}

        # This is a simplified version - full implementation would track
        # explore/exploit excess steps properly
        return metrics

    # Run training
    agent_state, metrics = loop.run(
        agent_state=agent_state,
        num_cycles=cfg.num_cycles,
        steps_per_cycle=cfg.steps_per_cycle,
        observe_trajectory=observe_trajectory,
        log_interval=100,
        mlflow_logging=cfg.mlflow,
    )

    # End MLflow run
    if cfg.mlflow:
        try:
            import mlflow

            mlflow.end_run()
        except Exception:
            pass

    print("\nTraining complete!")

    # Print final metrics
    if "exploit_excess_step_means" in metrics:
        final_exploit = metrics["exploit_excess_step_means"][-10:] if metrics["exploit_excess_step_means"] else []
        if final_exploit:
            print(f"Final exploit excess steps (last 10): {sum(final_exploit) / len(final_exploit):.2f}")

    if "explore_excess_step_means" in metrics:
        final_explore = metrics["explore_excess_step_means"][-10:] if metrics["explore_excess_step_means"] else []
        if final_explore:
            print(f"Final explore excess steps (last 10): {sum(final_explore) / len(final_explore):.2f}")

    return agent_state, metrics


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run HPC-PFC experiment")

    parser.add_argument(
        "--num_cycles",
        type=int,
        default=800000,
        help="Number of training cycles (default: 800000)",
    )
    parser.add_argument(
        "--steps_per_cycle",
        type=int,
        default=100,
        help="Environment steps per cycle (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging")
    parser.add_argument("--run_name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda/cpu)")
    parser.add_argument("--max_steps", type=int, default=20, help="Max steps per episode (default: 20)")
    parser.add_argument(
        "--trials_per_episode",
        type=int,
        default=10,
        help="Trials per episode (default: 10)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate (default: 3e-5)",
    )

    args = parser.parse_args()

    # Create configuration
    cfg = ExperimentConfig(
        num_cycles=args.num_cycles,
        steps_per_cycle=args.steps_per_cycle,
        seed=args.seed,
        mlflow=args.mlflow,
        mlflow_run_name=args.run_name,
        max_steps_in_episode=args.max_steps,
        trials_per_episode=args.trials_per_episode,
        learning_rate=args.learning_rate,
    )

    if args.device:
        cfg.device = args.device

    # Update dependent parameters (match JAX run_experiment_learned.py)
    cfg.memory_capacity = args.max_steps * cfg.forget_after  # max_steps * forget_after
    cfg.replay_buffer_capacity = args.max_steps * 10
    cfg.replay_sequence_length = args.max_steps * 10
    cfg.epsilon_transition_steps = args.max_steps * 50

    # Run experiment
    run_experiment(cfg)


if __name__ == "__main__":
    main()
