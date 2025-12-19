"""Loss functions for Double DQN training."""

import torch
import torch.nn.functional as F


def double_q_learning(
    q_tm1: torch.Tensor,
    a_tm1: torch.Tensor,
    r_t: torch.Tensor,
    discount_t: torch.Tensor,
    q_t_value: torch.Tensor,
    q_t_selector: torch.Tensor,
) -> torch.Tensor:
    """Compute Double Q-learning TD error.

    In Double Q-learning:
    - Action selection is done using one network (selector/target)
    - Value estimation is done using another network (value/online)

    TD target: r + gamma * Q_value(s', argmax_a Q_selector(s', a))

    Args:
        q_tm1: Q-values at time t-1 from online network, shape (..., num_actions).
        a_tm1: Actions taken at t-1, shape (...,).
        r_t: Rewards at time t, shape (...,).
        discount_t: Discount factors (gamma * (1 - done)), shape (...,).
        q_t_value: Q-values at time t from online network for value estimation.
        q_t_selector: Q-values at time t from target network for action selection.

    Returns:
        TD errors of shape (...,).
    """
    # Select best actions using selector (target) network
    best_actions = q_t_selector.argmax(dim=-1)

    # Get Q-values for best actions from value (online) network
    q_t_best = q_t_value.gather(-1, best_actions.unsqueeze(-1)).squeeze(-1)

    # Compute target
    target = r_t + discount_t * q_t_best

    # Get Q-value for taken action
    q_tm1_a = q_tm1.gather(-1, a_tm1.unsqueeze(-1).long()).squeeze(-1)

    # TD error
    td_error = target - q_tm1_a

    return td_error


def huber_loss(
    td_errors: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Compute Huber loss (smooth L1 loss).

    Huber loss is quadratic for small errors and linear for large errors,
    making it more robust to outliers than MSE.

    Args:
        td_errors: TD errors of shape (...,).
        delta: Threshold where loss transitions from quadratic to linear.

    Returns:
        Mean Huber loss (scalar).
    """
    return F.smooth_l1_loss(
        td_errors,
        torch.zeros_like(td_errors),
        beta=delta,
    )


def compute_dqn_loss(
    q_values_online: torch.Tensor,
    q_values_target: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    discount: float = 0.99,
) -> tuple[torch.Tensor, dict]:
    """Compute Double DQN loss for a batch of transitions.

    Args:
        q_values_online: Q-values from online network, shape (batch, seq_len, num_actions).
        q_values_target: Q-values from target network, shape (batch, seq_len, num_actions).
        actions: Actions taken, shape (batch, seq_len).
        rewards: Rewards received, shape (batch, seq_len).
        dones: Done flags, shape (batch, seq_len).
        discount: Discount factor gamma.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    # Compute TD errors using Double Q-learning
    # Q(s_t, a_t) from online network
    q_tm1 = q_values_online[:, :-1]  # (batch, seq_len-1, num_actions)
    a_tm1 = actions[:, :-1]  # (batch, seq_len-1)
    r_t = rewards[:, 1:]  # (batch, seq_len-1)
    done_t = dones[:, 1:].float()  # (batch, seq_len-1)

    # Next state Q-values
    q_t_value = q_values_online[:, 1:]  # For value estimation
    q_t_selector = q_values_target[:, 1:]  # For action selection

    # Discount factor (0 if done)
    discount_t = discount * (1 - done_t)

    # Compute TD errors
    td_errors = double_q_learning(
        q_tm1=q_tm1,
        a_tm1=a_tm1,
        r_t=r_t,
        discount_t=discount_t,
        q_t_value=q_t_value,
        q_t_selector=q_t_selector,
    )

    # Huber loss
    loss = huber_loss(td_errors)

    metrics = {
        "td_error_mean": td_errors.mean().item(),
        "td_error_std": td_errors.std().item(),
        "q_values_mean": q_tm1.mean().item(),
    }

    return loss, metrics
