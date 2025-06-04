#!/bin/bash
#SBATCH --job-name=safiso_test
#SBATCH --output=safiso_test_%j.out
#SBATCH --error=safiso_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=all

# Set up environment (matching your working scripts)
echo "Loading conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate energy-net-zoo1

# Change to project directory and set Python path
cd ~/energy-net-zoo
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "=== SafeISO Framework SLURM Testing Started ===" 
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Date: $(date)"
echo "=================================="

# Navigate to source directory
cd source

# Test 1: Check configuration system
echo -e "\n1. Testing configuration system..."
python experamint_config.py --list
CONFIG_EXIT=$?

if [ $CONFIG_EXIT -eq 0 ]; then
    echo "✅ Configuration system working"
else
    echo "❌ Configuration system failed"
    exit 1
fi

# Test 2: Quick smoke test with SLURM integration
echo -e "\n2. Running quick smoke test with SLURM..."
python omniSafe_research_evaluation.py --config quick --episodes 2 --seeds 1 --slurm
QUICK_EXIT=$?

# Test 3: Test constraint configurations with SLURM
echo -e "\n3. Testing strict safety constraints with SLURM..."
python omniSafe_research_evaluation.py --config strict_safety --episodes 3 --seeds 1 --slurm
STRICT_EXIT=$?

# Test 4: Test pricing configurations with SLURM
echo -e "\n4. Testing high volatility pricing with SLURM..."
python omniSafe_research_evaluation.py --config high_volatility --episodes 3 --seeds 1 --slurm
VOLATILITY_EXIT=$?

# Test 5: Baseline experiment with SLURM (more comprehensive)
echo -e "\n5. Running baseline experiment with SLURM..."
python omniSafe_research_evaluation.py --config baseline --episodes 5 --seeds 2 --slurm
BASELINE_EXIT=$?

# Check all results
echo -e "\n=== Test Results Summary ==="
echo "Configuration test: $([ $CONFIG_EXIT -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo "Quick test: $([ $QUICK_EXIT -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo "Strict safety: $([ $STRICT_EXIT -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo "High volatility: $([ $VOLATILITY_EXIT -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo "Baseline experiment: $([ $BASELINE_EXIT -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"

# Check results structure
echo -e "\n=== Results Validation ==="
LATEST_DIR=$(ls -td research-results/experiment_* | head -1 2>/dev/null)
if [ -d "$LATEST_DIR" ]; then
    echo "✅ Results directory found: $LATEST_DIR"
    echo "Contents:"
    ls -la "$LATEST_DIR/"
    
    # Check constraint and pricing configurations
    if [ -f "$LATEST_DIR/configuration.json" ]; then
        echo -e "\n📋 Configuration validation:"
        
        if grep -q "constraint_config" "$LATEST_DIR/configuration.json"; then
            echo "✅ Constraint configuration present"
            echo "Constraint details:"
            grep -A 10 "constraint_config" "$LATEST_DIR/configuration.json"
        else
            echo "❌ Constraint configuration missing"
        fi
        
        if grep -q "pricing_config" "$LATEST_DIR/configuration.json"; then
            echo "✅ Pricing configuration present"
            echo "Pricing details:"
            grep -A 5 "pricing_config" "$LATEST_DIR/configuration.json"
        else
            echo "❌ Pricing configuration missing"
        fi
    fi
else
    echo "❌ No results directory found"
fi

# Final status
TOTAL_FAILURES=$((($CONFIG_EXIT != 0) + ($QUICK_EXIT != 0) + ($STRICT_EXIT != 0) + ($VOLATILITY_EXIT != 0) + ($BASELINE_EXIT != 0)))

echo -e "\n=== Final Status ==="
if [ $TOTAL_FAILURES -eq 0 ]; then
    echo "🎉 ALL TESTS PASSED! Framework is ready for research."
    echo "SLURM integration working correctly."
    echo "Constraint and pricing configurations validated."
else
    echo "⚠️  $TOTAL_FAILURES test(s) failed. Check logs above."
    exit 1
fi

echo "=== Testing Complete ==="
echo "Total runtime: $(date)" 