# Evaluation System Guide - No More Overwrites! 🎉

## 🚨 **Problem Solved**
Previously, each evaluation run would overwrite the previous results. Now, every evaluation is timestamped and preserved!

## 📁 **New Directory Structure**

```
logs/eval/
├── PPOLag/
│   ├── 2025-05-20_19-23-21/          # Timestamped run 1
│   │   ├── evaluation_results.json
│   │   ├── reward_cost_comparison.png
│   │   └── constraint_violations.png
│   ├── 2025-05-20_20-15-30/          # Timestamped run 2
│   │   ├── evaluation_results.json
│   │   └── ...
│   ├── baseline_run/                 # Custom named run
│   │   └── ...
│   └── latest -> 2025-05-20_20-15-30 # Symlink to latest
├── CPO/
│   ├── 2025-05-20_21-00-00/
│   └── latest -> 2025-05-20_21-00-00
└── ...
```

## 🔧 **How It Works**

### **1. Automatic Timestamping**
Every evaluation gets a unique timestamp directory:
```bash
./run_safe_iso.sh --eval-only --model-path MODEL_PATH --algo PPOLag
# Creates: logs/eval/PPOLag/2025-05-20_19-23-21/
```

### **2. Custom Run Names**
You can specify custom names for important runs:
```bash
python evaluate_safe_iso.py --model-path MODEL_PATH --algo PPOLag --run-name "baseline_test"
# Creates: logs/eval/PPOLag/baseline_test/
```

### **3. Latest Symlink**
Always points to the most recent evaluation:
```bash
# This always works, regardless of timestamps:
cat logs/eval/PPOLag/latest/evaluation_results.json
```

## 📊 **Analysis System**

### **View All Available Runs**
```bash
python analyze_benchmark_results.py --list
```
Output:
```
📂 Available Evaluation Runs:
==================================================

PPOLag:
  - 2025-05-20_20-15-30 (latest)
  - 2025-05-20_19-23-21
  - baseline_test

CPO:
  - 2025-05-20_21-00-00 (latest)
```

### **Analyze Latest Results**
```bash
python analyze_benchmark_results.py
```
- Automatically uses the latest evaluation for each algorithm
- Shows which runs are being analyzed
- Creates timestamped output files

## 🎯 **Benchmark Workflow Examples**

### **1. Quick Test & Compare**
```bash
# Test each algorithm quickly
srun -c 2 --gres=gpu:1 bash -c "conda activate energy-net-zoo1 && cd /home/sleemann/energy-net-zoo && ./run_safe_iso.sh --algo PPOLag --steps 1000"
srun -c 2 --gres=gpu:1 bash -c "conda activate energy-net-zoo1 && cd /home/sleemann/energy-net-zoo && ./run_safe_iso.sh --algo CPO --steps 1000"

# Compare results
python analyze_benchmark_results.py
```

### **2. Full Benchmark with Multiple Evaluations**
```bash
# Train all algorithms
./benchmark_all_algorithms.sh

# Run baseline evaluations
for algo in PPOLag CPO FOCOPS CUP SautRL; do
    latest_model=$(ls logs/safe_iso/$algo/ | sort | tail -1)
    srun -c 2 --gres=gpu:1 bash -c "conda activate energy-net-zoo1 && cd /home/sleemann/energy-net-zoo && ./run_safe_iso.sh --eval-only --model-path logs/safe_iso/$algo/$latest_model/final_model.pt --algo $algo --run-name baseline"
done

# Run stress tests
for algo in PPOLag CPO FOCOPS CUP SautRL; do
    latest_model=$(ls logs/safe_iso/$algo/ | sort | tail -1)
    srun -c 2 --gres=gpu:1 bash -c "conda activate energy-net-zoo1 && cd /home/sleemann/energy-net-zoo && ./run_safe_iso.sh --eval-only --model-path logs/safe_iso/$algo/$latest_model/final_model.pt --algo $algo --stress-test --run-name stress_test"
done

# Analyze latest results
python analyze_benchmark_results.py
```

### **3. Compare Different Training Runs**
```bash
# Train same algorithm with different settings
srun -c 2 --gres=gpu:1 bash -c "conda activate energy-net-zoo1 && cd /home/sleemann/energy-net-zoo && ./run_safe_iso.sh --algo PPOLag --steps 100000 --cost-threshold 25.0"
srun -c 2 --gres=gpu:1 bash -c "conda activate energy-net-zoo1 && cd /home/sleemann/energy-net-zoo && ./run_safe_iso.sh --algo PPOLag --steps 100000 --cost-threshold 15.0"

# Evaluate both models with custom names
python evaluate_safe_iso.py --model-path logs/safe_iso/PPOLag/RUN1/final_model.pt --algo PPOLag --run-name "threshold_25"
python evaluate_safe_iso.py --model-path logs/safe_iso/PPOLag/RUN2/final_model.pt --algo PPOLag --run-name "threshold_15"

# See all runs
python analyze_benchmark_results.py --list
```

## 📈 **Output Files (No More Overwrites!)**

### **Evaluation Results**
- `logs/eval/ALGO/TIMESTAMP/evaluation_results.json` (never overwritten)
- `logs/eval/ALGO/latest/` (symlink to most recent)

### **Analysis Results**
- `logs/benchmark_comparison.png` (overwritten each analysis)
- `logs/benchmark_report.txt` (overwritten each analysis)  
- `logs/benchmark_metrics_TIMESTAMP.csv` (timestamped, never overwritten)

## 🔍 **Useful Commands**

### **Find Latest Model**
```bash
# Get the latest trained model for an algorithm
latest_model=$(ls logs/safe_iso/PPOLag/ | sort | tail -1)
echo "Latest PPOLag model: logs/safe_iso/PPOLag/$latest_model/final_model.pt"
```

### **Evaluate Specific Model**
```bash
# Evaluate a specific model with custom name
python evaluate_safe_iso.py \
  --model-path logs/safe_iso/PPOLag/2025-05-20_19-23-21/final_model.pt \
  --algo PPOLag \
  --run-name "experiment_1" \
  --stress-test
```

### **View Specific Evaluation**
```bash
# View results from a specific evaluation run
cat logs/eval/PPOLag/baseline_test/evaluation_results.json
```

### **Clean Old Evaluations (Optional)**
```bash
# Remove old evaluation runs (keep only last 5)
for algo in PPOLag CPO FOCOPS CUP SautRL; do
    cd logs/eval/$algo/
    ls -t | grep -E '^[0-9]{4}-[0-9]{2}-[0-9]{2}' | tail -n +6 | xargs rm -rf
    cd ../../..
done
```

## 🎉 **Benefits**

✅ **No more overwrites** - Every evaluation is preserved  
✅ **Easy comparison** - Compare different runs side by side  
✅ **Organized results** - Clear timestamps and custom names  
✅ **Latest access** - Always know which is the most recent  
✅ **Backward compatible** - Works with existing evaluation files  

## 🚀 **Ready to Benchmark!**

Now you can run as many evaluations as you want without losing any data:

```bash
# Quick start
conda activate energy-net-zoo1 && srun -c 2 --gres=gpu:1 --pty bash
./run_safe_iso.sh --algo PPOLag --steps 1000
python analyze_benchmark_results.py
```

Happy benchmarking! 🎯 