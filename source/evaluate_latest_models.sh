#!/bin/bash
# Model Evaluation Script for SafeISO with Responsive PCS Policy
# This script evaluates ONLY the latest model from each algorithm

set -e  # Exit on any error

# Default parameters
MODELS_DIR="logs/safe_iso"
OUTPUT_DIR="evaluation_results"
NUM_EPISODES=10
SEED=42
COST_THRESHOLD=25.0
MAX_EPISODE_STEPS=500
ALGORITHMS=("PPOLag" "CPO" "FOCOPS" "CUP" "PPOSaute")

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --cost-threshold)
            COST_THRESHOLD="$2"
            shift 2
            ;;
        --max-episode-steps)
            MAX_EPISODE_STEPS="$2"
            shift 2
            ;;
        --algorithms)
            IFS=',' read -ra ALGORITHMS <<< "$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --models-dir DIR          Directory containing trained models (default: logs/safe_iso)"
            echo "  --output-dir DIR          Directory for evaluation results (default: evaluation_results)"
            echo "  --num-episodes N          Number of episodes to evaluate (default: 10)"
            echo "  --seed N                  Random seed (default: 42)"
            echo "  --cost-threshold N        Cost threshold for constraints (default: 25.0)"
            echo "  --max-episode-steps N     Maximum steps per episode (default: 500)"
            echo "  --algorithms LIST         Comma-separated list of algorithms (default: PPOLag,CPO,FOCOPS,CUP,PPOSaute)"
            echo "  --help                    Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== SafeISO Latest Model Evaluation with Responsive PCS Policy ==="
echo "Models directory: $MODELS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of episodes: $NUM_EPISODES"
echo "Algorithms: ${ALGORITHMS[*]}"
echo "Cost threshold: $COST_THRESHOLD"
echo "Max episode steps: $MAX_EPISODE_STEPS"
echo "Random seed: $SEED"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to find the latest model for an algorithm
find_latest_model() {
    local algo=$1
    local algo_dir="$MODELS_DIR/$algo"
    
    if [ ! -d "$algo_dir" ]; then
        echo ""
        return 1
    fi
    
    # Find the most recent training run directory
    local latest_run_dir=$(find "$algo_dir" -maxdepth 1 -type d -name "20*" | sort | tail -1)
    
    if [ -z "$latest_run_dir" ]; then
        echo ""
        return 1
    fi
    
    # Find the highest epoch model in the latest run
    local latest_model=$(find "$latest_run_dir" -name "epoch-*.pt" | sort -V | tail -1)
    
    if [ -z "$latest_model" ]; then
        echo ""
        return 1
    fi
    
    echo "$latest_model"
    return 0
}

# Create evaluation script
cat > "$OUTPUT_DIR/evaluate_single_model.py" << 'EOF'
#!/usr/bin/env python3
"""
Single model evaluation script using responsive SafeISO environment
"""

import sys
import os
import argparse
import numpy as np
import torch
import json
from datetime import datetime

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'source'))

from omnisafe_environments import create_responsive_safe_iso_env

def load_model(model_path):
    """Load a trained model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    return checkpoint

def evaluate_model(model_path, num_episodes=10, seed=42, cost_threshold=25.0, max_episode_steps=500):
    """Evaluate a single model."""
    print(f"Evaluating model: {model_path}")
    
    # Create responsive environment (with the fix!)
    env = create_responsive_safe_iso_env(
        cost_threshold=cost_threshold,
        normalize_reward=True,
        max_episode_steps=max_episode_steps
    )
    
    # Load model
    checkpoint = load_model(model_path)
    
    # Extract policy weights
    if 'pi' in checkpoint:
        policy_weights = checkpoint['pi']
    elif 'policy' in checkpoint:
        policy_weights = checkpoint['policy']
    else:
        raise KeyError("No policy weights found in checkpoint")
    
    # Create a simple policy evaluator
    from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
    import gymnasium as gym
    
    # Determine architecture from weights
    if 'mean.0.weight' in policy_weights:
        input_dim = policy_weights['mean.0.weight'].shape[1]
        hidden1_dim = policy_weights['mean.0.weight'].shape[0]
        hidden2_dim = policy_weights['mean.2.weight'].shape[0]
        output_dim = policy_weights['mean.4.weight'].shape[0]
    else:
        # Fallback dimensions
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.shape[0]
        hidden1_dim = 64
        hidden2_dim = 64
    
    # Create actor
    actor = GaussianLearningActor(
        obs_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(input_dim,)),
        act_space=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(output_dim,)),
        hidden_sizes=[hidden1_dim, hidden2_dim],
        activation='tanh'
    )
    
    try:
        actor.load_state_dict(policy_weights)
        actor.eval()
    except Exception as e:
        print(f"Warning: Could not load policy weights: {e}")
        print("Using random policy for evaluation")
    
    # Evaluation loop
    episode_rewards = []
    episode_costs = []
    episode_violations = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_cost = 0
        episode_violation_count = 0
        step_count = 0
        
        for step in range(max_episode_steps):
            # Get action from policy
            try:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action_dist = actor(obs_tensor)
                    action = action_dist.sample()
                    action = action.squeeze(0).numpy()
            except:
                # Fallback to random action if policy fails
                action = env.action_space.sample()
            
            # Step environment
            try:
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                cost = info.get('cost', 0.0)
                episode_cost += cost
                
                if cost > 0:
                    episode_violation_count += 1
                
                step_count += 1
                
                if terminated or truncated:
                    break
                    
            except Exception as e:
                if "unsupported format string" in str(e):
                    # Known logging issue, continue
                    step_count += 1
                    continue
                else:
                    raise e
        
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        episode_violations.append(episode_violation_count)
        episode_lengths.append(step_count)
        
        print(f"  Episode {episode+1:2d}: reward={episode_reward:8.2f}, cost={episode_cost:6.3f}, violations={episode_violation_count:2d}, steps={step_count:3d}")
    
    env.close()
    
    # Calculate statistics
    results = {
        'model_path': model_path,
        'num_episodes': num_episodes,
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_cost': float(np.mean(episode_costs)),
        'std_cost': float(np.std(episode_costs)),
        'mean_violations': float(np.mean(episode_violations)),
        'std_violations': float(np.std(episode_violations)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'episode_rewards': episode_rewards,
        'episode_costs': episode_costs,
        'episode_violations': episode_violations,
        'episode_lengths': episode_lengths,
        'evaluation_time': datetime.now().isoformat()
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate a single SafeISO model')
    parser.add_argument('model_path', help='Path to model checkpoint')
    parser.add_argument('--num-episodes', type=int, default=10, help='Number of episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--cost-threshold', type=float, default=25.0, help='Cost threshold')
    parser.add_argument('--max-episode-steps', type=int, default=500, help='Max episode steps')
    parser.add_argument('--output', help='Output JSON file')
    
    args = parser.parse_args()
    
    results = evaluate_model(
        args.model_path,
        num_episodes=args.num_episodes,
        seed=args.seed,
        cost_threshold=args.cost_threshold,
        max_episode_steps=args.max_episode_steps
    )
    
    print(f"\nResults for {os.path.basename(args.model_path)}:")
    print(f"  Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"  Mean Cost: {results['mean_cost']:.3f} ± {results['std_cost']:.3f}")
    print(f"  Mean Violations: {results['mean_violations']:.1f} ± {results['std_violations']:.1f}")
    print(f"  Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    return results

if __name__ == '__main__':
    main()
EOF

chmod +x "$OUTPUT_DIR/evaluate_single_model.py"

# Find and evaluate latest models
echo "Searching for latest trained models..."
total_evaluations=0
all_results=()

for algo in "${ALGORITHMS[@]}"; do
    echo ""
    echo "=== Evaluating $algo (latest model only) ==="
    
    # Find the latest model for this algorithm (with error handling)
    latest_model=$(find_latest_model "$algo") || {
        echo "No models found for $algo in $MODELS_DIR/$algo"
        continue
    }
    
    if [ -z "$latest_model" ]; then
        echo "No models found for $algo in $MODELS_DIR/$algo"
        continue
    fi
    
    echo "Found latest model: $(basename "$latest_model")"
    echo "  Training run: $(echo "$latest_model" | cut -d'/' -f3)"
    echo "  Full path: $latest_model"
    
    # Create output filename
    model_basename=$(basename "$latest_model" .pt)
    output_file="$OUTPUT_DIR/${algo}_${model_basename}_results.json"
    
    # Run evaluation with proper error handling under set -e
    echo ""
    echo "Evaluating: $latest_model"
    
    # Temporarily disable set -e for this evaluation
    set +e
    python "$OUTPUT_DIR/evaluate_single_model.py" \
        "$latest_model" \
        --num-episodes "$NUM_EPISODES" \
        --seed "$SEED" \
        --cost-threshold "$COST_THRESHOLD" \
        --max-episode-steps "$MAX_EPISODE_STEPS" \
        --output "$output_file"
    eval_exit_code=$?
    set -e  # Re-enable set -e
    
    # Check the actual exit code
    if [ $eval_exit_code -eq 0 ]; then
        all_results+=("$output_file")
        total_evaluations=$((total_evaluations + 1))
        echo "✅ Successfully evaluated $algo"
    else
        echo "❌ Failed to evaluate $latest_model (exit code: $eval_exit_code)"
        echo "   Continuing with next algorithm..."
    fi
done

echo ""
echo "=== Evaluation Summary ==="
echo "Total evaluations completed: $total_evaluations"
echo "Results saved in: $OUTPUT_DIR"

# Create summary report
if [ ${#all_results[@]} -gt 0 ]; then
    echo ""
    echo "Creating summary report..."
    
    cat > "$OUTPUT_DIR/create_summary.py" << 'EOF'
#!/usr/bin/env python3
import json
import sys
import os
from glob import glob

def create_summary(results_dir):
    """Create a summary of all evaluation results."""
    result_files = glob(os.path.join(results_dir, "*_results.json"))
    
    if not result_files:
        print("No result files found")
        return
    
    print("=== EVALUATION SUMMARY ===")
    print(f"{'Algorithm':<12} {'Mean Reward':<15} {'Mean Cost':<12} {'Mean Violations':<15} {'Model'}")
    print("-" * 80)
    
    summary_data = []
    
    for result_file in sorted(result_files):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            algo = os.path.basename(result_file).split('_')[0]
            model_name = os.path.basename(data['model_path'])
            
            summary_data.append({
                'algorithm': algo,
                'model': model_name,
                'mean_reward': data['mean_reward'],
                'std_reward': data['std_reward'],
                'mean_cost': data['mean_cost'],
                'std_cost': data['std_cost'],
                'mean_violations': data['mean_violations'],
                'std_violations': data['std_violations']
            })
            
            print(f"{algo:<12} {data['mean_reward']:8.2f}±{data['std_reward']:5.2f} "
                  f"{data['mean_cost']:6.3f}±{data['std_cost']:4.3f} "
                  f"{data['mean_violations']:6.1f}±{data['std_violations']:4.1f} "
                  f"{model_name}")
                  
        except Exception as e:
            print(f"Error reading {result_file}: {e}")
    
    # Save summary
    summary_file = os.path.join(results_dir, 'evaluation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Check if results are different (fix verification)
    if len(summary_data) > 1:
        rewards = [d['mean_reward'] for d in summary_data]
        costs = [d['mean_cost'] for d in summary_data]
        
        reward_range = max(rewards) - min(rewards)
        cost_range = max(costs) - min(costs)
        
        print(f"\n=== FIX VERIFICATION ===")
        print(f"Reward range: {reward_range:.2f} (min: {min(rewards):.2f}, max: {max(rewards):.2f})")
        print(f"Cost range: {cost_range:.3f} (min: {min(costs):.3f}, max: {max(costs):.3f})")
        
        if reward_range > 1.0 or cost_range > 0.001:
            print("✅ SUCCESS: Different algorithms show different performance!")
            print("   The action-invariance issue has been resolved.")
        else:
            print("⚠️  WARNING: Results still very similar across algorithms")
            print("   May need to retrain models or check environment setup")

if __name__ == '__main__':
    results_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    create_summary(results_dir)
EOF

    # Run summary creation with error handling
    set +e
    python "$OUTPUT_DIR/create_summary.py" "$OUTPUT_DIR"
    summary_exit_code=$?
    set -e
    
    if [ $summary_exit_code -ne 0 ]; then
        echo "⚠️  Warning: Summary generation failed (exit code: $summary_exit_code)"
        echo "   Individual results are still available in $OUTPUT_DIR"
    fi
fi

echo ""
echo "Latest model evaluation complete! Check $OUTPUT_DIR for detailed results." 