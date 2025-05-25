# Source Directory

This directory contains all the Safe RL ISO Agent implementation files that were added to the original energy-net-zoo repository.

## Files in this directory:

### Core Training Scripts
- **`train_safe_iso.py`** - Main training script for safe RL ISO agents
- **`evaluate_safe_iso.py`** - Evaluation script for trained safe RL agents  
- **`run_safe_iso.sh`** - Shell script wrapper for training and evaluation

### Environment Wrapper
- **`safe_iso_wrapper.py`** - SafeISOWrapper that adds safety constraints to ISO-RLZoo-v0

## Purpose

This directory separates all the safe RL implementation files from the original repository structure, making it easy to:

1. **Track additions**: Everything in `source/` was added for the safe RL implementation
2. **Maintain separation**: Original repo files remain unchanged in their locations
3. **Easy migration**: Can move these files to a separate repository if needed
4. **Clear organization**: All safe RL code is contained in one place

## Usage

The main entry point is through the benchmark script in the root directory:

```bash
# From the root directory:
./benchmark_all_algorithms.sh
```

Or you can run individual training sessions:

```bash
# From the root directory:
./source/run_safe_iso.sh --algo PPOLag --steps 100000
```

## Dependencies

These scripts depend on:
- OmniSafe (for safe RL algorithms)
- stable-baselines3 (for fallback PPO)
- The original energy-net package (for ISO-RLZoo-v0 environment)

## File Relationships

```
root/
├── benchmark_all_algorithms.sh  # Entry point (original repo + modifications)
├── source/                      # All our safe RL additions
│   ├── run_safe_iso.sh         # Calls train_safe_iso.py and evaluate_safe_iso.py
│   ├── train_safe_iso.py       # Uses safe_iso_wrapper.py
│   ├── evaluate_safe_iso.py    # Uses safe_iso_wrapper.py  
│   └── safe_iso_wrapper.py     # Wraps ISO-RLZoo-v0 with safety constraints
└── logs/                        # Generated during training (created by scripts)
    ├── safe_iso/               # Training logs and models
    └── eval/                   # Evaluation results
```

## Safe RL Implementation Summary

The key innovation is the **SafeISOWrapper** which:

1. **Wraps ISO-RLZoo-v0** to add safety constraints
2. **Calculates constraint costs** for voltage, frequency, battery, and supply-demand violations
3. **Provides cost signals** to safe RL algorithms for constraint-based learning
4. **Maintains grid safety** while optimizing economic performance

The implementation supports 5 safe RL algorithms:
- PPOLag (with safe fallback)
- CPO  
- FOCOPS
- CUP
- SautRL 