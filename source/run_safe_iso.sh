#!/bin/bash
# Script for training and evaluating safe ISO agents with OmniSafe

# Exit on error
set -e

# Ensure our environment is on PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Default values
ALGO="PPOLag"
NUM_STEPS=5000
COST_THRESHOLD=25.0
SEED=42
EVAL_ONLY=false
STRESS_TEST=false
PRICING_POLICY="ONLINE"
DEMAND_PATTERN="SINUSOIDAL"
COST_TYPE="CONSTANT"
USE_DISPATCH=false
DEMAND_NOISE=0.0
SENSOR_NOISE=0.0
OUTAGE_PROB=0.0
DEMAND_SPIKE_PROB=0.0
SPIKE_MAGNITUDE=2.0

# Function to display usage information
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --algo ALGO           Algorithm to use: PPOLag, CPO, FOCOPS, CUP, PPOSaute (default: $ALGO)"
  echo "  --steps NUM           Number of timesteps to train (default: $NUM_STEPS)"
  echo "  --cost-threshold VAL  Cost threshold for constraints (default: $COST_THRESHOLD)"
  echo "  --seed NUM            Random seed (default: $SEED)"
  echo "  --eval-only           Skip training and only evaluate"
  echo "  --model-path PATH     Path to model for evaluation (required with --eval-only)"
  echo "  --stress-test         Run stress tests during evaluation"
  echo "  --pricing POLICY      Pricing policy: ONLINE, QUADRATIC, CONSTANT (default: $PRICING_POLICY)"
  echo "  --demand PATTERN      Demand pattern: SINUSOIDAL, RANDOM, PERIODIC, SPIKES (default: $DEMAND_PATTERN)"
  echo "  --cost-type TYPE      Cost type: CONSTANT, VARIABLE, TIME_OF_USE (default: $COST_TYPE)"
  echo "  --use-dispatch        Enable dispatch action"
  echo "  --demand-noise VAL    Noise level for demand (default: $DEMAND_NOISE)"
  echo "  --sensor-noise VAL    Noise level for sensor readings (default: $SENSOR_NOISE)"
  echo "  --outage-prob VAL     Probability of generator outage (default: $OUTAGE_PROB)"
  echo "  --spike-prob VAL      Probability of demand spike (default: $DEMAND_SPIKE_PROB)"
  echo "  --spike-mag VAL       Magnitude of demand spikes (default: $SPIKE_MAGNITUDE)"
  echo "  --help                Display this help message"
  exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --algo)
      ALGO="$2"
      shift 2
      ;;
    --steps)
      NUM_STEPS="$2"
      shift 2
      ;;
    --cost-threshold)
      COST_THRESHOLD="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --eval-only)
      EVAL_ONLY=true
      shift
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --stress-test)
      STRESS_TEST=true
      shift
      ;;
    --pricing)
      PRICING_POLICY="$2"
      shift 2
      ;;
    --demand)
      DEMAND_PATTERN="$2"
      shift 2
      ;;
    --cost-type)
      COST_TYPE="$2"
      shift 2
      ;;
    --use-dispatch)
      USE_DISPATCH=true
      shift
      ;;
    --demand-noise)
      DEMAND_NOISE="$2"
      shift 2
      ;;
    --sensor-noise)
      SENSOR_NOISE="$2"
      shift 2
      ;;
    --outage-prob)
      OUTAGE_PROB="$2"
      shift 2
      ;;
    --spike-prob)
      DEMAND_SPIKE_PROB="$2"
      shift 2
      ;;
    --spike-mag)
      SPIKE_MAGNITUDE="$2"
      shift 2
      ;;
    --help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Check if model path is provided when eval-only is set
if [ "$EVAL_ONLY" = true ] && [ -z "$MODEL_PATH" ]; then
  echo "ERROR: --model-path is required when using --eval-only"
  usage
fi

# Prepare dispatch flag
DISPATCH_FLAG=""
if [ "$USE_DISPATCH" = true ]; then
  DISPATCH_FLAG="--use-dispatch"
fi

# Prepare stress test flag
STRESS_FLAG=""
if [ "$STRESS_TEST" = true ]; then
  STRESS_FLAG="--stress-test"
fi

# Create log directories
mkdir -p logs/safe_iso/$ALGO

# Initialize model path if not in eval-only mode
if [ "$EVAL_ONLY" = false ]; then
  # Train the agent
  echo "=== Training Safe ISO Agent ==="
  echo "Algorithm: $ALGO"
  echo "Timesteps: $NUM_STEPS"
  echo "Cost threshold: $COST_THRESHOLD"
  echo "Seed: $SEED"
  echo "Pricing policy: $PRICING_POLICY"
  echo "Demand pattern: $DEMAND_PATTERN"
  echo "Cost type: $COST_TYPE"
  echo "Use dispatch: $USE_DISPATCH"
  
  # Run training
  python source/train_safe_iso.py \
    --algo $ALGO \
    --num-steps $NUM_STEPS \
    --cost-threshold $COST_THRESHOLD \
    --seed $SEED \
    --pricing-policy $PRICING_POLICY \
    --demand-pattern $DEMAND_PATTERN \
    --cost-type $COST_TYPE \
    $DISPATCH_FLAG
  
  # Find the latest model directory
  LATEST_DIR=$(find logs/safe_iso/$ALGO -maxdepth 1 -type d | sort | tail -n 1)
  
  # Set model path to the final model
  MODEL_PATH="$LATEST_DIR/final_model.pt"
  
  echo "Training completed. Model saved to $MODEL_PATH"
fi

# Evaluate the agent
echo "=== Evaluating Safe ISO Agent ==="
echo "Model path: $MODEL_PATH"
echo "Algorithm: $ALGO"
echo "Pricing policy: $PRICING_POLICY"
echo "Demand pattern: $DEMAND_PATTERN"
echo "Cost type: $COST_TYPE"
echo "Use dispatch: $USE_DISPATCH"
echo "Stress test: $STRESS_TEST"

# Run evaluation
python source/evaluate_safe_iso.py \
  --model-path $MODEL_PATH \
  --algo $ALGO \
  --cost-threshold $COST_THRESHOLD \
  --pricing-policy $PRICING_POLICY \
  --demand-pattern $DEMAND_PATTERN \
  --cost-type $COST_TYPE \
  $DISPATCH_FLAG \
  $STRESS_FLAG \
  --demand-noise $DEMAND_NOISE \
  --sensor-noise $SENSOR_NOISE \
  --outage-prob $OUTAGE_PROB \
  --demand-spike-prob $DEMAND_SPIKE_PROB \
  --spike-mag $SPIKE_MAGNITUDE \
  --run-name "$(basename $(dirname $MODEL_PATH))"

echo "Evaluation completed. Results saved to logs/eval/$ALGO/"

# If we ran stress tests, highlight this
if [ "$STRESS_TEST" = true ]; then
  echo "Stress test results are available in the evaluation log"
  echo "For visualizations, see logs/eval/$ALGO/constraint_violations.png"
fi

echo "=== Done ===" 