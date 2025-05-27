#!/bin/bash
#SBATCH --job-name=safeiso_eval_latest
#SBATCH --output=logs/evaluation_latest_%j.out
#SBATCH --error=logs/evaluation_latest_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=all

# SafeISO Latest Model Evaluation with Responsive PCS Policy
# This script evaluates only the latest model from each Safe RL algorithm

echo "=== SafeISO Latest Model Evaluation Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load environment
echo "Loading conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate energy-net-zoo1

# Verify environment
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo ""

# Default parameters (can be overridden via environment variables)
MODELS_DIR=${MODELS_DIR:-"logs/safe_iso"}
OUTPUT_DIR=${OUTPUT_DIR:-"evaluation_latest_results_$(date +%Y%m%d_%H%M%S)"}
NUM_EPISODES=${NUM_EPISODES:-10}
SEED=${SEED:-42}
COST_THRESHOLD=${COST_THRESHOLD:-25.0}
MAX_EPISODE_STEPS=${MAX_EPISODE_STEPS:-500}
ALGORITHMS=${ALGORITHMS:-"PPOLag,CPO,FOCOPS,CUP,PPOSaute"}

echo "=== Evaluation Parameters ==="
echo "Models directory: $MODELS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of episodes: $NUM_EPISODES"
echo "Algorithms: $ALGORITHMS"
echo "Cost threshold: $COST_THRESHOLD"
echo "Max episode steps: $MAX_EPISODE_STEPS"
echo "Random seed: $SEED"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if models directory exists
if [ ! -d "$MODELS_DIR" ]; then
    echo "ERROR: Models directory not found: $MODELS_DIR"
    echo "Please ensure you have trained models in this directory"
    exit 1
fi

# Run the evaluation script
echo "Starting latest model evaluation..."
echo "Command: ./source/evaluate_latest_models.sh --models-dir $MODELS_DIR --output-dir $OUTPUT_DIR --num-episodes $NUM_EPISODES --seed $SEED --cost-threshold $COST_THRESHOLD --max-episode-steps $MAX_EPISODE_STEPS --algorithms $ALGORITHMS"
echo ""

./source/evaluate_latest_models.sh \
    --models-dir "$MODELS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --num-episodes "$NUM_EPISODES" \
    --seed "$SEED" \
    --cost-threshold "$COST_THRESHOLD" \
    --max-episode-steps "$MAX_EPISODE_STEPS" \
    --algorithms "$ALGORITHMS"

exit_code=$?

echo ""
echo "=== Latest Model Evaluation Job Complete ==="
echo "Exit code: $exit_code"
echo "End time: $(date)"

if [ $exit_code -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
    echo "Results saved in: $OUTPUT_DIR"
    
    # Display summary if available
    if [ -f "$OUTPUT_DIR/evaluation_summary.json" ]; then
        echo ""
        echo "=== Quick Summary ==="
        python -c "
import json
try:
    with open('$OUTPUT_DIR/evaluation_summary.json', 'r') as f:
        data = json.load(f)
    
    print(f'Evaluated {len(data)} algorithms (latest models only):')
    for item in data:
        print(f'  {item[\"algorithm\"]}: reward={item[\"mean_reward\"]:.2f}±{item[\"std_reward\"]:.2f}, cost={item[\"mean_cost\"]:.3f}±{item[\"std_cost\"]:.3f}')
        
    # Check if fix worked
    rewards = [d['mean_reward'] for d in data]
    costs = [d['mean_cost'] for d in data]
    if len(set(rewards)) > 1 or len(set(costs)) > 1:
        print('\\n✅ Action-invariance fix verified: Different algorithms show different results!')
    else:
        print('\\n⚠️  Results still identical - may need to retrain models')
        
except Exception as e:
    print(f'Could not read summary: {e}')
"
    fi
else
    echo "❌ Evaluation failed with exit code: $exit_code"
    echo "Check the logs for details"
fi

exit $exit_code 