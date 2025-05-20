# Safe RL ISO Agent for Smart Grid Control

This component implements a robust and safe reinforcement learning (RL) agent for Independent System Operator (ISO) control in smart grids using OmniSafe.

## Overview

The Safe ISO agent is designed to:

1. **Maintain supply-demand balance** in the grid while adapting to uncertain demand
2. **Adhere to critical safety constraints** such as voltage limits, frequency stability, and battery safety
3. **Withstand stress tests** including unusual demand patterns, generator outages, and sensor noise

## Features

- **Safety-Constrained Learning**: Uses constrained RL algorithms from OmniSafe to respect physical grid limits
- **Multiple Algorithms**: Implements PPO-Lagrangian, CPO, FOCOPS, CUP, and SautRL
- **Robustness Testing**: Evaluates agent performance under various adversarial conditions
- **Detailed Metrics**: Tracks safety violations, reliability, and economic performance

## Installation

OmniSafe is already added to the requirements.txt in the rl-baselines3-zoo directory. If not already installed:

```bash
cd rl-baselines3-zoo
pip install -r requirements.txt
```

## Usage

### Basic Training and Evaluation

The easiest way to train and evaluate a safe ISO agent is using the provided script:

```bash
./run_safe_iso.sh --algo PPOLag --steps 100000
```

This trains a PPO-Lagrangian agent for 100,000 timesteps with default settings and runs a basic evaluation.

### Running Stress Tests

To evaluate robustness under challenging conditions:

```bash
./run_safe_iso.sh --algo PPOLag --steps 100000 --stress-test --demand-noise 0.2 --sensor-noise 0.1 --outage-prob 0.05 --spike-prob 0.1
```

This runs additional evaluations with demand noise, sensor noise, generator outages, and demand spikes.

### Training Options

The training script supports multiple algorithms and configurations:

```bash
./run_safe_iso.sh --algo CPO --steps 200000 --cost-threshold 20.0 --pricing ONLINE --demand SINUSOIDAL --cost-type CONSTANT --use-dispatch
```

### Evaluation Only

To evaluate a previously trained model:

```bash
./run_safe_iso.sh --eval-only --model-path logs/safe_iso/PPOLag/TIMESTAMP/final_model.pt --algo PPOLag --stress-test
```

## Available Safe RL Algorithms

1. **PPO-Lagrangian (PPOLag)**: Integrates Lagrangian multipliers with PPO to balance rewards and constraints
2. **Constrained Policy Optimization (CPO)**: Uses trust regions to optimize while satisfying constraints
3. **First Order Constrained Optimization in Policy Space (FOCOPS)**: Projects policies into safe policy spaces
4. **Constrained Update Projection (CUP)**: Handles uncertainty by projecting updates into a safe region
5. **Sauté RL (SautRL)**: Augments states with a "safety budget" for safe exploration

## Configuration Parameters

### Environment Options
- `--pricing`: Pricing policy (ONLINE, QUADRATIC, CONSTANT)
- `--demand`: Demand pattern (SINUSOIDAL, RANDOM, PERIODIC, SPIKES)
- `--cost-type`: Grid cost calculation (CONSTANT, VARIABLE, TIME_OF_USE)
- `--use-dispatch`: Enable dispatch commands 

### Safety Parameters
- `--cost-threshold`: Threshold for constraint violations (default: 25.0)

### Robustness Parameters
- `--demand-noise`: Noise level for demand forecasts
- `--sensor-noise`: Noise level for sensor readings
- `--outage-prob`: Probability of generator outages
- `--spike-prob`: Probability of demand spikes
- `--spike-mag`: Magnitude of demand spikes (multiplier)

## File Structure

- `safe_iso_wrapper.py`: Wrapper that adds safety constraints to the ISO environment
- `train_safe_iso.py`: Script for training safe ISO agents
- `evaluate_safe_iso.py`: Script for evaluating trained agents under regular and adversarial conditions
- `run_safe_iso.sh`: Convenience script for training and evaluation

## Evaluation Metrics

The evaluation results include:

1. **Reward**: Economic performance of the agent
2. **Cost**: Total safety constraint violations
3. **Constraint Violations**: Breakdown of violations by type:
   - Voltage violations
   - Frequency violations
   - Battery state-of-charge violations
   - Supply-demand imbalance violations

## Visualization

Evaluation results are automatically plotted and saved to the logs directory:

- `reward_cost_comparison.png`: Comparison of rewards and costs across evaluation scenarios
- `constraint_violations.png`: Breakdown of constraint violations by type and scenario

## Safety Constraint Implementation

The safety wrapper defines costs for four primary constraints:

1. **Voltage Limits**: Costs for operating outside voltage ranges (0.95-1.05 p.u.)
2. **Frequency Stability**: Costs for deviation from nominal frequency (49.8-50.2 Hz)
3. **Battery Safety**: Costs for battery SOC outside safe operation (10%-90%)
4. **Supply-Demand Balance**: Costs for significant imbalances (threshold: 10 MW)

These constraints are weighted and combined to form the overall safety cost function.

## Extending the Framework

You can easily extend this framework by:

1. Modifying constraint limits in `safe_iso_wrapper.py`
2. Adding new stress test scenarios in `evaluate_safe_iso.py`
3. Implementing additional safe RL algorithms in `train_safe_iso.py`
4. Customizing the environment configuration in `run_safe_iso.sh`

## Detailed Usage Guide

### Command-line Options

#### Core Parameters

```bash
./run_safe_iso.sh --algo PPOLag --steps 500000 --cost-threshold 20.0
```

| Option | Description | Default |
|--------|-------------|---------|
| `--algo` | Algorithm to use | `PPOLag` |
| `--steps` | Training timesteps | `5000` |
| `--cost-threshold` | Safety constraint threshold (lower = stricter) | `25.0` |
| `--seed` | Random seed | `42` |

#### Environment Configuration

```bash
./run_safe_iso.sh --pricing ONLINE --demand SINUSOIDAL --cost-type CONSTANT --use-dispatch
```

| Option | Description | Choices | Default |
|--------|-------------|---------|---------|
| `--pricing` | Pricing policy | `ONLINE`, `QUADRATIC`, `CONSTANT` | `ONLINE` |
| `--demand` | Demand pattern | `SINUSOIDAL`, `RANDOM`, `PERIODIC`, `SPIKES` | `SINUSOIDAL` |
| `--cost-type` | Grid cost model | `CONSTANT`, `VARIABLE`, `TIME_OF_USE` | `CONSTANT` |
| `--use-dispatch` | Enable dispatch actions | flag | disabled |

#### Evaluation Options

```bash
./run_safe_iso.sh --eval-only --model-path logs/safe_iso/PPOLag/2023-06-01_12-34-56/final_model.pt
```

| Option | Description |
|--------|-------------|
| `--eval-only` | Skip training, only evaluate |
| `--model-path` | Path to saved model (required with `--eval-only`) |

#### Stress Testing

```bash
./run_safe_iso.sh --stress-test --demand-noise 0.2 --sensor-noise 0.1 --outage-prob 0.05 --spike-prob 0.1
```

| Option | Description | Range | Default |
|--------|-------------|-------|---------|
| `--stress-test` | Enable stress testing | flag | disabled |
| `--demand-noise` | Noise level for demand forecasts | 0.0-1.0 | `0.0` |
| `--sensor-noise` | Noise level for sensor readings | 0.0-1.0 | `0.0` |
| `--outage-prob` | Probability of generator outages | 0.0-1.0 | `0.0` |
| `--spike-prob` | Probability of demand spikes | 0.0-1.0 | `0.0` |
| `--spike-mag` | Magnitude of demand spikes (multiplier) | > 1.0 | `2.0` |

### Recommended Workflows

#### 1. Initial Training & Testing

```bash
# Quick test run
./run_safe_iso.sh --algo PPOLag --steps 5000

# Full training
./run_safe_iso.sh --algo PPOLag --steps 1000000
```

#### 2. Algorithm Comparison

Train and evaluate multiple algorithms with the same settings:

```bash
for algo in PPOLag CPO FOCOPS CUP; do
  ./run_safe_iso.sh --algo $algo --steps 100000
done
```

#### 3. Robustness Evaluation

Test trained models under various stress conditions:

```bash
# Test against demand noise
./run_safe_iso.sh --eval-only --model-path PATH/TO/MODEL --algo PPOLag --stress-test --demand-noise 0.2

# Test against multiple adverse conditions
./run_safe_iso.sh --eval-only --model-path PATH/TO/MODEL --algo PPOLag --stress-test \
  --demand-noise 0.2 --sensor-noise 0.1 --outage-prob 0.05 --spike-prob 0.1
```

### Output Files

After running the script, you'll find results in:

| Path | Contents |
|------|----------|
| `logs/safe_iso/[ALGO]/[TIMESTAMP]/` | Training logs and saved models |
| `logs/safe_iso/[ALGO]/[TIMESTAMP]/final_model.pt` | Trained model |
| `logs/safe_iso/[ALGO]/[TIMESTAMP]/args.json` | Training configuration |
| `logs/eval/[ALGO]/` | Evaluation results |
| `logs/eval/[ALGO]/reward_cost_comparison.png` | Performance visualization |
| `logs/eval/[ALGO]/constraint_violations.png` | Safety violation breakdown |
| `logs/eval/[ALGO]/evaluation_results.json` | Detailed results in JSON format |

### Example Scenarios

#### Grid Stability Training

Train an agent focused on grid stability with strict safety constraints:

```bash
./run_safe_iso.sh --algo CPO --steps 500000 --cost-threshold 10.0
```

#### Economic Optimization

Train an agent for economic dispatch with demand spikes:

```bash
./run_safe_iso.sh --algo PPOLag --steps 500000 --demand SPIKES --pricing ONLINE --use-dispatch
```

#### Extreme Condition Testing

Test a trained agent against extreme grid conditions:

```bash
./run_safe_iso.sh --eval-only --model-path PATH/TO/MODEL --algo PPOLag \
  --stress-test --demand-noise 0.3 --outage-prob 0.1 --spike-prob 0.2 --spike-mag 3.0
``` 