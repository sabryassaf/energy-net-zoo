#!/bin/bash
#SBATCH --job-name=safe_rl_benchmark
#SBATCH --output=logs/slurm_benchmark_%j.out
#SBATCH --error=logs/slurm_benchmark_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

echo "Starting Safe RL ISO Agent Full Benchmark..."
echo "=============================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate energy-net-zoo1

# Ensure we're in the right directory
cd /home/sleemann/energy-net-zoo
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Array of algorithms to test
ALGORITHMS=("PPOLag" "CPO" "FOCOPS" "CUP" "PPOSaute")

# Training parameters
STEPS=500000
COST_THRESHOLD=25.0
SEED=42

echo "Training parameters:"
echo "  - Steps: $STEPS"
echo "  - Cost threshold: $COST_THRESHOLD"
echo "  - Seed: $SEED"
echo "  - Algorithms: ${ALGORITHMS[*]}"
echo ""

# Phase 1: Training all algorithms
echo "Phase 1: Training all algorithms"
echo "-----------------------------------"

for i in "${!ALGORITHMS[@]}"; do
    algo="${ALGORITHMS[$i]}"
    echo ""
    echo "[$((i+1))/5] Training $algo..."
    echo "Started at: $(date)"
    
    # Run the training directly (no srun needed inside batch job)
    ./source/omniSafe_utils/run_safe_iso.sh --algo $algo --steps $STEPS --cost-threshold $COST_THRESHOLD --seed $SEED
    
    if [ $? -eq 0 ]; then
        echo "SUCCESS: $algo training completed at $(date)"
    else
        echo "FAILED: $algo training failed at $(date)"
        echo "Check logs in logs/safe_iso/$algo/ for details"
    fi
    
    echo "Waiting 10 seconds before next training..."
    sleep 10
done

echo ""
echo "All training completed!"
echo "=========================="
echo "Finished at: $(date)"
echo ""
echo "Results saved in: logs/safe_iso/"
echo ""
echo "To find your trained models:"
for algo in "${ALGORITHMS[@]}"; do
    echo "  $algo: ls logs/safe_iso/$algo/"
done
