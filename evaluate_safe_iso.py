#!/usr/bin/env python
"""
Evaluation script for Safe ISO Agent

This script evaluates a trained safe ISO agent under regular and adversarial conditions
to assess its robustness and safety compliance.
"""

import os
import argparse
import numpy as np
import torch
import omnisafe
import gymnasium as gym
import json
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Any, List, Optional, Tuple

# Import our safe ISO wrapper
from safe_iso_wrapper import SafeISOWrapper, make_safe_iso_env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("evaluate_safe_iso")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained safe ISO agent")
    
    # Model parameters
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--algo", type=str, default="PPOLag", 
                        choices=["PPOLag", "CPO", "FOCOPS", "CUP", "SautRL"],
                        help="Algorithm used to train the model")
    
    # Environment parameters
    parser.add_argument("--env-id", type=str, default="ISO-RLZoo-v0", 
                        help="Environment ID (default: ISO-RLZoo-v0)")
    parser.add_argument("--cost-threshold", type=float, default=25.0,
                        help="Cost threshold for constraints (default: 25.0)")
    
    # Environment configuration
    parser.add_argument("--pricing-policy", type=str, default="ONLINE",
                        choices=["ONLINE", "QUADRATIC", "CONSTANT"],
                        help="Pricing policy (default: ONLINE)")
    parser.add_argument("--demand-pattern", type=str, default="SINUSOIDAL",
                        choices=["SINUSOIDAL", "RANDOM", "PERIODIC", "SPIKES"],
                        help="Demand pattern (default: SINUSOIDAL)")
    parser.add_argument("--cost-type", type=str, default="CONSTANT",
                        choices=["CONSTANT", "VARIABLE", "TIME_OF_USE"],
                        help="Cost type (default: CONSTANT)")
    parser.add_argument("--use-dispatch", action="store_true",
                        help="Enable dispatch action")
    
    # Evaluation parameters
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of episodes to evaluate (default: 10)")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default="logs/eval",
                        help="Directory for saving evaluation results")
    
    # Stress test parameters
    parser.add_argument("--stress-test", action="store_true",
                        help="Run stress tests in addition to regular evaluation")
    parser.add_argument("--demand-noise", type=float, default=0.0,
                        help="Noise level for demand (default: 0.0)")
    parser.add_argument("--sensor-noise", type=float, default=0.0,
                        help="Noise level for sensor readings (default: 0.0)")
    parser.add_argument("--outage-prob", type=float, default=0.0,
                        help="Probability of generator outage (default: 0.0)")
    parser.add_argument("--demand-spike-prob", type=float, default=0.0,
                        help="Probability of demand spike (default: 0.0)")
    parser.add_argument("--spike-magnitude", type=float, default=2.0,
                        help="Magnitude of demand spikes as multiplier (default: 2.0)")
    
    return parser.parse_args()

def create_env_kwargs(args, scenario="regular"):
    """Create environment kwargs from args based on evaluation scenario."""
    # Basic kwargs common to all evaluation runs
    env_kwargs = {
        "pricing_policy": args.pricing_policy,
        "demand_pattern": args.demand_pattern,
        "cost_type": args.cost_type,
        "dispatch_config": {
            "use_dispatch_action": args.use_dispatch,
            "default_strategy": "PROPORTIONAL"
        }
    }
    
    # Apply stress test parameters for adversarial scenarios
    if scenario == "demand_noise" and args.demand_noise > 0:
        env_kwargs["demand_noise"] = args.demand_noise
    
    elif scenario == "sensor_noise" and args.sensor_noise > 0:
        env_kwargs["sensor_noise"] = args.sensor_noise
    
    elif scenario == "generator_outage" and args.outage_prob > 0:
        env_kwargs["generator_outage_prob"] = args.outage_prob
    
    elif scenario == "demand_spike" and args.demand_spike_prob > 0:
        env_kwargs["demand_spike_prob"] = args.demand_spike_prob
        env_kwargs["spike_magnitude"] = args.spike_magnitude
    
    elif scenario == "combined" and args.stress_test:
        # Combine all stress factors for a worst-case scenario
        if args.demand_noise > 0:
            env_kwargs["demand_noise"] = args.demand_noise
        if args.sensor_noise > 0:
            env_kwargs["sensor_noise"] = args.sensor_noise
        if args.outage_prob > 0:
            env_kwargs["generator_outage_prob"] = args.outage_prob
        if args.demand_spike_prob > 0:
            env_kwargs["demand_spike_prob"] = args.demand_spike_prob
            env_kwargs["spike_magnitude"] = args.spike_magnitude
    
    return env_kwargs

def evaluate_agent(agent, env, num_episodes=10, render=False):
    """Evaluate an agent in the given environment for multiple episodes."""
    episode_rewards = []
    episode_costs = []
    episode_constraint_violations = []
    
    for i in tqdm(range(num_episodes), desc="Evaluating"):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        episode_cost = 0
        episode_violation = {"voltage": 0, "frequency": 0, "battery": 0, "supply_demand": 0}
        
        while not (done or truncated):
            # Get action from the agent
            action = agent.predict(obs)[0]
            
            # Take step in the environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Update metrics
            episode_reward += reward
            episode_cost += info.get("cost", 0)
            
            # Track constraint violations
            violations = info.get("constraint_violations", {})
            for constraint in episode_violation:
                episode_violation[constraint] = violations.get(constraint, 0)
            
            # Render if requested
            if render:
                env.render()
        
        # Store episode results
        episode_rewards.append(episode_reward)
        episode_costs.append(episode_cost)
        episode_constraint_violations.append(episode_violation)
    
    # Calculate summary statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_cost = np.mean(episode_costs)
    std_cost = np.std(episode_costs)
    
    # Calculate total violations by constraint type
    total_violations = {}
    for constraint in ["voltage", "frequency", "battery", "supply_demand"]:
        total_violations[constraint] = sum(ep[constraint] for ep in episode_constraint_violations)
    
    return {
        "rewards": episode_rewards,
        "costs": episode_costs,
        "constraint_violations": episode_constraint_violations,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_cost": mean_cost,
        "std_cost": std_cost,
        "total_violations": total_violations
    }

def create_agent(algo, model_path, env, seed=42):
    """Create and load an agent - supports both OmniSafe and stable-baselines3."""
    # Basic configuration
    config = {
        "train_cfgs": {
            "device": "auto",
            "seed": seed
        },
        "algo_cfgs": {
            "steps_per_epoch": 2048,
            "use_cost": True
        }
    }
    
    # First try to determine if the model is from stable-baselines3
    if os.path.exists(model_path) and (model_path.endswith('.zip') or 
                                       os.path.isdir(model_path) and 
                                       any(f.endswith('.zip') for f in os.listdir(model_path))):
        try:
            # Try to load as a stable-baselines3 model
            logger.info("Attempting to load model as stable-baselines3 model")
            from stable_baselines3 import PPO
            agent = PPO.load(model_path, env=env.wrapped_env)
            logger.info("Successfully loaded stable-baselines3 model")
            return agent
        except Exception as e:
            logger.warning(f"Failed to load as stable-baselines3 model: {e}")
    
    # If not stable-baselines3 or loading failed, try OmniSafe
    try:
        # Create the appropriate OmniSafe agent
        logger.info("Attempting to load model as OmniSafe model")
        if algo == "PPOLag":
            agent = omnisafe.PPOLag(env=env, **config)
        elif algo == "CPO":
            agent = omnisafe.CPO(env=env, **config)
        elif algo == "FOCOPS":
            agent = omnisafe.FOCOPS(env=env, **config)
        elif algo == "CUP":
            agent = omnisafe.CUP(env=env, **config)
        elif algo == "SautRL":
            agent = omnisafe.SautRL(env=env, **config)
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")
        
        # Load the trained model
        agent.load(model_path)
        logger.info("Successfully loaded OmniSafe model")
        return agent
    except Exception as e:
        logger.error(f"Failed to load OmniSafe model: {e}")
        
        # Final fallback - try again with stable-baselines3 but with a different approach
        logger.info("Falling back to stable-baselines3 PPO")
        try:
            from stable_baselines3 import PPO
            agent = PPO.load(model_path)
            logger.info("Successfully loaded stable-baselines3 model in fallback mode")
            return agent
        except Exception as e2:
            logger.error(f"All loading methods failed. Last error: {e2}")
            raise ValueError(f"Could not load model from {model_path} with algorithm {algo}")

def plot_evaluation_results(results, output_dir):
    """Plot and save evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot reward distribution
    plt.figure(figsize=(10, 6))
    scenarios = list(results.keys())
    rewards = [results[scenario]["mean_reward"] for scenario in scenarios]
    costs = [results[scenario]["mean_cost"] for scenario in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot rewards on primary y-axis
    bars1 = ax1.bar(x - width/2, rewards, width, label='Reward', color='blue', alpha=0.7)
    ax1.set_ylabel('Mean Reward', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot costs on secondary y-axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, costs, width, label='Cost', color='red', alpha=0.7)
    ax2.set_ylabel('Mean Cost', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add labels and title
    ax1.set_xlabel('Evaluation Scenario')
    ax1.set_title('Mean Reward and Cost by Scenario')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reward_cost_comparison.png'))
    plt.close()
    
    # Plot constraint violations
    plt.figure(figsize=(12, 6))
    constraint_types = ["voltage", "frequency", "battery", "supply_demand"]
    
    # Create a dictionary to store violations by constraint type and scenario
    violations_by_constraint = {constraint: [] for constraint in constraint_types}
    
    for scenario in scenarios:
        for constraint in constraint_types:
            violations_by_constraint[constraint].append(
                results[scenario]["total_violations"][constraint]
            )
    
    # Create a grouped bar chart
    x = np.arange(len(scenarios))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot bars for each constraint type
    for i, constraint in enumerate(constraint_types):
        offset = (i - 1.5) * width
        ax.bar(x + offset, violations_by_constraint[constraint], width, 
               label=constraint.capitalize())
    
    # Add labels and title
    ax.set_xlabel('Evaluation Scenario')
    ax.set_ylabel('Number of Violations')
    ax.set_title('Constraint Violations by Scenario and Type')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'constraint_violations.png'))
    plt.close()
    
    # Save raw results as JSON
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        json_results = {}
        for scenario, scenario_results in results.items():
            json_results[scenario] = {}
            for key, value in scenario_results.items():
                if isinstance(value, np.ndarray):
                    json_results[scenario][key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    json_results[scenario][key] = [v.tolist() for v in value]
                elif isinstance(value, (np.float32, np.float64)):
                    json_results[scenario][key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    json_results[scenario][key] = int(value)
                else:
                    json_results[scenario][key] = value
        
        json.dump(json_results, f, indent=4)

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.algo)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(output_dir, "eval_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Define evaluation scenarios
    scenarios = ["regular"]
    if args.stress_test:
        scenarios.extend([
            "demand_noise",
            "sensor_noise",
            "generator_outage",
            "demand_spike",
            "combined"
        ])
    
    # Run evaluations
    results = {}
    
    for scenario in scenarios:
        logger.info(f"Evaluating scenario: {scenario}")
        
        # Create environment with scenario-specific settings
        env_kwargs = create_env_kwargs(args, scenario)
        
        # Log environment settings
        logger.info(f"Environment kwargs for {scenario}: {env_kwargs}")
        
        # Create environment
        env = make_safe_iso_env(
            env_id=args.env_id,
            env_kwargs=env_kwargs,
            cost_threshold=args.cost_threshold
        )
        
        # Create and load agent
        agent = create_agent(args.algo, args.model_path, env, args.seed)
        
        # Evaluate agent
        scenario_results = evaluate_agent(
            agent=agent,
            env=env,
            num_episodes=args.num_episodes,
            render=args.render
        )
        
        # Store results
        results[scenario] = scenario_results
        
        # Close environment
        env.close()
        
        # Log summary
        logger.info(f"Scenario {scenario} - Mean reward: {scenario_results['mean_reward']:.2f}, "
                   f"Mean cost: {scenario_results['mean_cost']:.2f}")
        logger.info(f"Total violations: {scenario_results['total_violations']}")
    
    # Plot and save results
    plot_evaluation_results(results, output_dir)
    logger.info(f"Evaluation results saved to {output_dir}")

if __name__ == "__main__":
    main() 