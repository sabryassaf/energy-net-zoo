#!/bin/bash
# Script for training safe ISO agents with OmniSafe

# Exit on error
set -e

# Ensure our environment is on PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Default values
ALGO="PPOLag"
NUM_STEPS=5000
COST_THRESHOLD=25.0
SEED=42
PRICING_POLICY="ONLINE"
DEMAND_PATTERN="SINUSOIDAL"
COST_TYPE="CONSTANT"
USE_DISPATCH=false

# Function to display usage information
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --algo ALGO           Algorithm to use: PPOLag, CPO, FOCOPS, CUP, PPOSaute (default: $ALGO)"
  echo "  --steps NUM           Number of timesteps to train (default: $NUM_STEPS)"
  echo "  --cost-threshold VAL  Cost threshold for constraints (default: $COST_THRESHOLD)"
  echo "  --seed NUM            Random seed (default: $SEED)"
  echo "  --pricing POLICY      Pricing policy: ONLINE, QUADRATIC, CONSTANT (default: $PRICING_POLICY)"
  echo "  --demand PATTERN      Demand pattern: SINUSOIDAL, RANDOM, PERIODIC, SPIKES (default: $DEMAND_PATTERN)"
  echo "  --cost-type TYPE      Cost type: CONSTANT, VARIABLE, TIME_OF_USE (default: $COST_TYPE)"
  echo "  --use-dispatch        Enable dispatch action"
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
    --help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Prepare dispatch flag
DISPATCH_FLAG=""
if [ "$USE_DISPATCH" = true ]; then
  DISPATCH_FLAG="--use-dispatch"
fi

# Create log directories
mkdir -p logs/safe_iso/$ALGO

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
echo ""

# Run training
python source/omniSafe_utils/train_safe_iso.py \
  --algo $ALGO \
  --num-steps $NUM_STEPS \
  --cost-threshold $COST_THRESHOLD \
  --seed $SEED \
  --pricing-policy $PRICING_POLICY \
  --demand-pattern $DEMAND_PATTERN \
  --cost-type $COST_TYPE \
  $DISPATCH_FLAG

# Find the latest model directory
LATEST_DIR=$(find logs/safe_iso/$ALGO -maxdepth 1 -type d -name "20*" | sort | tail -n 1)

if [ -n "$LATEST_DIR" ]; then
  MODEL_PATH="$LATEST_DIR/final_model.pt"
  echo ""
  echo "=== Training Complete ==="
  echo "Model saved to: $MODEL_PATH"
  echo ""
  echo "To evaluate this model:"
  echo "  ./source/evaluate_model.sh --model-path $MODEL_PATH --algo $ALGO"
  echo ""
else
  echo "ERROR: Could not find trained model directory"
  exit 1
fi

echo "=== Done ===" 