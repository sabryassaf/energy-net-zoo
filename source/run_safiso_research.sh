#!/bin/bash
#SBATCH --job-name=safiso_research
#SBATCH --output=safiso_research_%j.out
#SBATCH --error=safiso_research_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=all

echo "=== SafeISO Research Run Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"

# Set up environment (matching your working scripts)
echo "Loading conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate energy-net-zoo1

# Change to project directory and set Python path
cd ~/energy-net-zoo
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Verify environment
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo ""

# Navigate to source directory
cd source

# Full baseline study
echo "Running comprehensive baseline study..."
python omniSafe_research_evaluation.py --config baseline --episodes 30 --seeds 10 --slurm

if [ $? -eq 0 ]; then
    echo "✅ Baseline study completed successfully"
else
    echo "❌ Baseline study failed"
    exit 1
fi

# Stress testing study  
echo "Running stress testing study..."
python omniSafe_research_evaluation.py --config stress --episodes 25 --seeds 8 --slurm

if [ $? -eq 0 ]; then
    echo "✅ Stress testing study completed successfully"
else
    echo "❌ Stress testing study failed"
    exit 1
fi

# Algorithm-specific optimization
echo "Running algorithm-specific optimization..."
python omniSafe_research_evaluation.py --config algorithm_specific --episodes 20 --seeds 5 --slurm

if [ $? -eq 0 ]; then
    echo "✅ Algorithm-specific optimization completed successfully"
else
    echo "❌ Algorithm-specific optimization failed"
    exit 1
fi

# Find and report results
echo -e "\n=== Research Results Summary ==="
LATEST_DIR=$(ls -td research-results/experiment_* | head -1 2>/dev/null)
if [ -d "$LATEST_DIR" ]; then
    echo "Latest results directory: $LATEST_DIR"
    echo "Contents:"
    ls -la "$LATEST_DIR/"
    
    if [ -f "$LATEST_DIR/research_report.md" ]; then
        echo -e "\nResearch report preview:"
        head -20 "$LATEST_DIR/research_report.md"
    fi
else
    echo "⚠️  No results directory found"
fi

echo -e "\n=== Research Complete ==="
echo "End time: $(date)"
echo "All studies completed successfully!"
echo "Check the latest research-results directory for detailed findings."
