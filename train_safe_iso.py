#!/usr/bin/env python
"""
Training script for Safe ISO Agent using OmniSafe

This script trains a safe ISO agent using constraint-based RL algorithms from OmniSafe.
"""

import os
import time
import argparse
import numpy as np
import torch
import gymnasium as gym
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Import our safe ISO wrapper
from safe_iso_wrapper import SafeISOWrapper, make_safe_iso_env

# Try different OmniSafe imports based on the installed version
try:
    import omnisafe
    # Check if omnisafe has the expected algorithm interface
    HAS_OMNISAFE_ALGO = hasattr(omnisafe, 'PPOLag')
except ImportError:
    HAS_OMNISAFE_ALGO = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_safe_iso")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a safe ISO agent with OmniSafe")
    
    # Environment parameters
    parser.add_argument("--env-id", type=str, default="ISO-RLZoo-v0", 
                        help="Environment ID (default: ISO-RLZoo-v0)")
    parser.add_argument("--cost-threshold", type=float, default=25.0,
                        help="Cost threshold for constraints (default: 25.0)")
    
    # Algorithm choice
    parser.add_argument("--algo", type=str, default="PPOLag", 
                        choices=["PPOLag", "CPO", "FOCOPS", "CUP", "SautRL"],
                        help="Safe RL algorithm to use (default: PPOLag)")
    
    # Training parameters
    parser.add_argument("--num-steps", type=int, default=1_000_000,
                        help="Total number of timesteps (default: 1M)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run on (auto, cpu, cuda, cuda:0, etc.)")
    
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
    
    # Logging and saving
    parser.add_argument("--log-dir", type=str, default="logs/safe_iso",
                        help="Directory for saving logs (default: logs/safe_iso)")
    parser.add_argument("--eval-interval", type=int, default=10000,
                        help="Evaluation interval in timesteps (default: 10000)")
    parser.add_argument("--save-interval", type=int, default=50000,
                        help="Model saving interval in timesteps (default: 50000)")
    
    # Robustness parameters
    parser.add_argument("--demand-noise", type=float, default=0.0,
                        help="Noise level for demand (default: 0.0)")
    parser.add_argument("--sensor-noise", type=float, default=0.0,
                        help="Noise level for sensor readings (default: 0.0)")
    
    return parser.parse_args()

def create_env_kwargs(args):
    """Create environment kwargs from args."""
    # Basic kwargs common to all training runs
    env_kwargs = {
        "pricing_policy": args.pricing_policy,
        "demand_pattern": args.demand_pattern,
        "cost_type": args.cost_type,
        "dispatch_config": {
            "use_dispatch_action": args.use_dispatch,
            "default_strategy": "PROPORTIONAL"
        }
    }
    
    # Add robustness parameters if specified
    if args.demand_noise > 0:
        env_kwargs["demand_noise"] = args.demand_noise
    
    if args.sensor_noise > 0:
        env_kwargs["sensor_noise"] = args.sensor_noise
    
    return env_kwargs

def configure_algorithm(algo_name):
    """Configure the algorithm hyperparameters based on the algorithm name."""
    # Common parameters
    config = {
        "train_cfgs": {
            "device": "auto",
            "epochs": 500,
            "steps_per_epoch": 2048,
        },
        "algo_cfgs": {
            "steps_per_epoch": 2048,
            "update_iters": 10,
            "batch_size": 64,
            "target_kl": 0.01,
            "entropy_coef": 0.01,
            "reward_normalize": True,
            "cost_normalize": True, 
            "obs_normalize": True,
            "hidden_sizes": (64, 64),
            "activation": "tanh",
            "lr": 3e-4,
        }
    }
    
    # Algorithm-specific parameters
    if algo_name == "PPOLag":
        config["algo_cfgs"].update({
            "use_cost": True,
            "cost_limit": 25.0,
            "lagrangian_multiplier_init": 0.1,
            "lambda_lr": 0.035,
            "lambda_optimizer": "Adam",
        })
    elif algo_name == "CPO":
        config["algo_cfgs"].update({
            "cost_limit": 25.0,
            "damping_coef": 0.1,
            "cg_iter": 10,
        })
    elif algo_name == "FOCOPS":
        config["algo_cfgs"].update({
            "cost_limit": 25.0,
            "lam_c": 1.5,
            "xi": 0.01,
        })
    elif algo_name == "CUP":
        config["algo_cfgs"].update({
            "cost_limit": 25.0,
            "cup_type": "ratio",
            "projection_type": "direct", 
        })
    elif algo_name == "SautRL":
        config["algo_cfgs"].update({
            "cost_limit": 25.0,
            "eta": 0.1,
            "lam": 0.1,
            "eps": 0.2,
        })
    
    return config

def create_agent(algo_name, env, config):
    """Create an OmniSafe agent with fallback to alternative implementation."""
    # Extract train_cfgs and algo_cfgs from config
    train_cfgs = config.get("train_cfgs", {})
    algo_cfgs = config.get("algo_cfgs", {})
    
    # Check which OmniSafe API version we have
    if HAS_OMNISAFE_ALGO:
        # Direct algorithm access (newer versions)
        try:
            if algo_name == "PPOLag":
                return omnisafe.PPOLag(env=env, train_cfgs=train_cfgs, algo_cfgs=algo_cfgs)
            elif algo_name == "CPO":
                return omnisafe.CPO(env=env, train_cfgs=train_cfgs, algo_cfgs=algo_cfgs)
            elif algo_name == "FOCOPS":
                return omnisafe.FOCOPS(env=env, train_cfgs=train_cfgs, algo_cfgs=algo_cfgs)
            elif algo_name == "CUP":
                return omnisafe.CUP(env=env, train_cfgs=train_cfgs, algo_cfgs=algo_cfgs)
            elif algo_name == "SautRL":
                return omnisafe.SautRL(env=env, train_cfgs=train_cfgs, algo_cfgs=algo_cfgs)
        except TypeError as e:
            # If there's a TypeError, try with the config structure directly
            if algo_name == "PPOLag":
                return omnisafe.PPOLag(env_id=env.env_id, **config)
            elif algo_name == "CPO":
                return omnisafe.CPO(env_id=env.env_id, **config)
            elif algo_name == "FOCOPS":
                return omnisafe.FOCOPS(env_id=env.env_id, **config)
            elif algo_name == "CUP":
                return omnisafe.CUP(env_id=env.env_id, **config)
            elif algo_name == "SautRL":
                return omnisafe.SautRL(env_id=env.env_id, **config)
    else:
        # Try alternative import paths for older versions
        try:
            # Import algorithms through submodules
            if algo_name == "PPOLag":
                from omnisafe.algorithms.on_policy import PPOLag
                return PPOLag(env_id=env.env_id, train_cfgs=train_cfgs, algo_cfgs=algo_cfgs)
            elif algo_name == "CPO":
                from omnisafe.algorithms.on_policy import CPO
                return CPO(env_id=env.env_id, train_cfgs=train_cfgs, algo_cfgs=algo_cfgs)
            elif algo_name == "FOCOPS":
                from omnisafe.algorithms.on_policy import FOCOPS
                return FOCOPS(env_id=env.env_id, train_cfgs=train_cfgs, algo_cfgs=algo_cfgs)
            elif algo_name == "CUP":
                from omnisafe.algorithms.on_policy import CUP
                return CUP(env_id=env.env_id, train_cfgs=train_cfgs, algo_cfgs=algo_cfgs)
            elif algo_name == "SautRL":
                from omnisafe.algorithms.on_policy import SautRL
                return SautRL(env_id=env.env_id, train_cfgs=train_cfgs, algo_cfgs=algo_cfgs)
        except (ImportError, TypeError) as e:
            logger.error(f"Failed to import algorithm {algo_name} from omnisafe: {e}")
            logger.error("You may need to reinstall omnisafe with the correct version.")
            raise ImportError(f"Could not import {algo_name} from omnisafe.")

def main():
    """Main training function."""
    args = parse_args()
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.algo, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    # Save arguments to log directory
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment kwargs
    env_kwargs = create_env_kwargs(args)
    
    # Create a safe environment
    env = make_safe_iso_env(
        env_id=args.env_id,
        env_kwargs=env_kwargs,
        cost_threshold=args.cost_threshold
    )
    
    # Configure the algorithm
    algo_config = configure_algorithm(args.algo)
    
    # Update config with command line arguments
    algo_config["train_cfgs"]["device"] = args.device
    algo_config["train_cfgs"]["epochs"] = args.num_steps // algo_config["train_cfgs"]["steps_per_epoch"]
    algo_config["train_cfgs"]["save_dir"] = log_dir
    algo_config["algo_cfgs"]["cost_limit"] = args.cost_threshold
    
    # Log the environment and algorithm details
    logger.info(f"Training safe ISO agent with {args.algo} algorithm")
    logger.info(f"Environment: {args.env_id}")
    logger.info(f"Environment kwargs: {env_kwargs}")
    logger.info(f"Cost threshold: {args.cost_threshold}")
    logger.info(f"Algorithm config: {algo_config}")
    
    # Create the algorithm
    try:
        agent = create_agent(args.algo, env, algo_config)
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        logger.error("Due to compatibility issues with OmniSafe, we'll fall back to using stable-baselines3 PPO.")
        logger.error("Please reinstall omnisafe properly if you want to use safe RL algorithms.")
        
        # Fall back to using stable-baselines3 PPO
        try:
            from stable_baselines3 import PPO
            
            # Configure policy network architecture
            policy_kwargs = dict(
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
                activation_fn=torch.nn.Tanh
            )
            
            agent = PPO(
                "MlpPolicy", 
                env.wrapped_env,  # Use the unwrapped env for standard PPO
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=os.path.join(log_dir, "tensorboard"),
                seed=args.seed
            )
            
            logger.info("Successfully created fallback PPO agent from stable-baselines3.")
        except Exception as fallback_error:
            logger.error(f"Failed to create fallback agent: {fallback_error}")
            raise
    
    # Train the agent
    start_time = time.time()
    logger.info("Starting training...")
    
    try:
        # Check if it's a stable-baselines3 agent (which requires total_timesteps)
        if 'stable_baselines3' in str(type(agent)):
            agent.learn(total_timesteps=args.num_steps)
        else:
            # OmniSafe agent doesn't require total_timesteps
            agent.learn()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise
    finally:
        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Training took {training_time:.2f} seconds")
        
        # Save final model
        final_model_path = os.path.join(log_dir, "final_model.pt")
        try:
            agent.save(final_model_path)
            logger.info(f"Final model saved to {final_model_path}")
        except Exception as save_error:
            logger.error(f"Failed to save model: {save_error}")
            # Try alternate save method
            try:
                if hasattr(agent, "save_checkpoint"):
                    agent.save_checkpoint(os.path.join(log_dir, "checkpoint"))
                    logger.info(f"Model checkpoint saved to {os.path.join(log_dir, 'checkpoint')}")
            except:
                logger.error("All save methods failed. Model will not be preserved.")
        
        # Close the environment
        env.close()

if __name__ == "__main__":
    main() 