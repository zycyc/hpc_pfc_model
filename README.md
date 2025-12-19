# Flexible Prefrontal Control over Hippocampal Episodic Memory for Goal-Directed Generalization

Official implementation for the paper:

> **Flexible Prefrontal Control over Hippocampal Episodic Memory for Goal-Directed Generalization**
> Yicong Zheng, Nora Wolf, Charan Ranganath, Randall C. O'Reilly, Kevin L. McKee
> 2025 Conference on Cognitive Computational Neuroscience (CCN 2025)
> [[OpenReview]](https://openreview.net/forum?id=7hhz5ToJnM) [[arXiv]](https://arxiv.org/abs/2503.02303)

## Overview

This repository contains a reinforcement learning model simulating prefrontal cortex (PFC) and hippocampus (HPC) interactions. The model enables flexible modification of perception and behavior based on current goals by allowing the prefrontal region to generate query-key representations that selectively encode and retrieve task-relevant episodic memories.

- PFC-HPC interaction mechanism for goal-directed generalization
- Top-down prefrontal control for learning arbitrary associations


## Repository Structure

```
hpc_pfc_model/
├── jax/    # JAX/Equinox implementation
└── torch/  # PyTorch implementation
```

Both implementations replicate **Experiment 3 (blocked condition)** from the paper. See the paper for full experimental details.

Instructions for running the experiments are provided in the README files of each implementation directory.

