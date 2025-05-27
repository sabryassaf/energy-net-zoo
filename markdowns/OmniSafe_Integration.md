# OmniSafe Integration for Safe RL Training

## Overview

This document explains how the ISO environment is integrated with OmniSafe 0.5.0 for training Safe RL agents.

## Key Components

### 1. SafeISOWrapper (`safe_iso_wrapper.py`)
- Adds safety constraints to the base ISO environment
- Implements 4 constraint types: voltage, frequency, battery, supply-demand
- Calculates constraint violation costs
- Standard Gymnasium wrapper

### 2. SafeISOCMDP (`omnisafe_iso_env.py`) 
- OmniSafe-compatible wrapper that inherits from CMDP
- Converts SafeISOWrapper to work with OmniSafe algorithms
- Returns 6 values from step(): obs, reward, cost, terminated, truncated, info
- Registered with OmniSafe using `@env_register` decorator

### 3. Training Script (`train_safe_iso.py`)
- Creates OmniSafe agents for algorithms: PPOLag, CPO, FOCOPS, CUP, PPOSaute
- Uses SafeISO-v0 environment ID
- Handles model saving and logging

## How It Works

1. **Environment Chain**: ISO-RLZoo-v0 → SafeISOWrapper → SafeISOCMDP → OmniSafe
2. **Training**: OmniSafe algorithms train on the CMDP-wrapped environment
3. **Constraints**: Safety costs from constraint violations guide the learning process
4. **Output**: Trained models saved with proper constraint enforcement

## Usage

```bash
# Train a single algorithm
./source/run_safe_iso.sh --algo PPOLag --steps 300000

# Train all algorithms
sbatch submit_full_benchmark.sbatch
```

The integration ensures that all trained agents respect safety constraints while learning optimal energy management policies. 