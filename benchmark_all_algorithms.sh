#!/bin/bash

# Safe RL ISO Agent Benchmark - Automated Training Script
# This script trains all 5 safe RL algorithms sequentially

# Activate environment
conda activate energy-net-zoo1
cd /home/sleemann/energy-net-zoo

# Array of algorithms to test
ALGORITHMS=("PPOLag" "CPO" "FOCOPS" "CUP" "PPOSaute")

# Training parameters
STEPS=500000
COST_THRESHOLD=25.0
SEED=42

echo "🚀 Starting Safe RL ISO Agent Benchmark..."
echo "=========================================="
echo "Training parameters:"
echo "  - Steps: $STEPS"
echo "  - Cost threshold: $COST_THRESHOLD"
echo "  - Seed: $SEED"
echo "  - Algorithms: ${ALGORITHMS[*]}"
echo ""

# Phase 1: Training all algorithms
echo "📚 Phase 1: Training all algorithms"
echo "-----------------------------------"

for i in "${!ALGORITHMS[@]}"; do
    algo="${ALGORITHMS[$i]}"
    echo ""
    echo "[$((i+1))/5] 🏋️  Training $algo..."
    echo "Command: srun -c 2 --gres=gpu:1 bash -c \"conda activate energy-net-zoo1 && cd /home/sleemann/energy-net-zoo && ./source/run_safe_iso.sh --algo $algo --steps $STEPS --cost-threshold $COST_THRESHOLD --seed $SEED\""
    
    # Run the training
    srun -c 2 --gres=gpu:1 bash -c "conda activate energy-net-zoo1 && cd /home/sleemann/energy-net-zoo && ./source/run_safe_iso.sh --algo $algo --steps $STEPS --cost-threshold $COST_THRESHOLD --seed $SEED"
    
    if [ $? -eq 0 ]; then
        echo "✅ $algo training completed successfully!"
    else
        echo "❌ $algo training failed!"
        echo "Check logs in logs/safe_iso/$algo/ for details"
    fi
    
    echo "Waiting 30 seconds before next training..."
    sleep 30
done

echo ""
echo "🎉 All training completed!"
echo "=========================="
echo ""
echo "📁 Results saved in: logs/safe_iso/"
echo ""
echo "📊 Next steps:"
echo "1. Check training results: ls logs/safe_iso/*/latest/"
echo "2. Run stress tests using commands from SAFE_RL_BENCHMARK_GUIDE.md"
echo "3. Compare evaluation results in logs/eval/"
echo ""
echo "🔍 To find your trained models:"
for algo in "${ALGORITHMS[@]}"; do
    echo "  $algo: ls logs/safe_iso/$algo/"
done
echo ""
echo "📈 Run evaluation for a specific model:"
echo "  ./source/run_safe_iso.sh --eval-only --model-path logs/safe_iso/ALGO/TIMESTAMP/final_model.pt --algo ALGO --stress-test"
echo ""
echo "🏁 Benchmark complete! Happy analyzing! 🎯" 