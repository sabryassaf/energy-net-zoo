#!/usr/bin/env python
"""
Training script for Safe ISO Agent using OmniSafe 0.5.0

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

# Import our safe ISO wrapper
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from safe_iso_wrapper import SafeISOWrapper, make_safe_iso_env

# Import OmniSafe
import omnisafe

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
                        choices=["PPOLag", "CPO", "FOCOPS", "CUP", "PPOSaute"],
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
    
    return parser.parse_args()

def create_env_kwargs(args):
    """Create environment kwargs from args."""
    env_kwargs = {
        "pricing_policy": args.pricing_policy,
        "demand_pattern": args.demand_pattern,
        "cost_type": args.cost_type,
        "dispatch_config": {
            "use_dispatch_action": args.use_dispatch,
            "default_strategy": "PROPORTIONAL"
        }
    }
    return env_kwargs

def create_custom_cfgs(algo_name, cost_threshold, num_steps, log_dir, device):
    """Create custom configuration for OmniSafe 0.5.0 API."""
    
    # Base configuration following OmniSafe 0.5.0 structure
    custom_cfgs = {
        # Training configuration
        "train_cfgs": {
            "device": device,
            "total_steps": num_steps,
        },
        
        # Algorithm configuration
        "algo_cfgs": {
            "steps_per_epoch": 2048,
            "update_iters": 10,
            "batch_size": 64,
            "target_kl": 0.01,
            "entropy_coef": 0.01,
            "reward_normalize": True,
            "cost_normalize": True,
            "obs_normalize": True,
            "use_cost": True,
        },
        
        # Logger configuration
        "logger_cfgs": {
            "log_dir": log_dir,
            "use_tensorboard": True,
            "save_model_freq": 10,
        },
        
        # Model configuration
        "model_cfgs": {
            "actor": {
                "hidden_sizes": [64, 64],
                "activation": "tanh",
                "lr": 3e-4,
            },
            "critic": {
                "hidden_sizes": [64, 64],
                "activation": "tanh",
                "lr": 3e-4,
            }
        },
        
        # Lagrangian configuration (for constrained algorithms)
        "lagrange_cfgs": {
            "cost_limit": cost_threshold,
        }
    }
    
    # Algorithm-specific parameters
    if algo_name == "PPOLag":
        custom_cfgs["lagrange_cfgs"].update({
            "lagrangian_multiplier_init": 0.1,
            "lambda_lr": 0.035,
            "lambda_optimizer": "Adam",
        })
    elif algo_name == "CPO":
        custom_cfgs["algo_cfgs"].update({
            "damping_coef": 0.1,
            "cg_iters": 10,
        })
    elif algo_name == "FOCOPS":
        custom_cfgs["algo_cfgs"].update({
            "lam_c": 1.5,
            "xi": 0.01,
        })
    elif algo_name == "CUP":
        custom_cfgs["algo_cfgs"].update({
            "cup_type": "ratio",
            "projection_type": "direct",
        })
    elif algo_name == "PPOSaute":
        custom_cfgs["algo_cfgs"].update({
            "eta": 0.1,
            "lam": 0.1,
            "eps": 0.2,
        })
    
    return custom_cfgs

def main():
    """Main training function."""
    args = parse_args()
    
    # Handle algorithm name mapping (SautRL -> PPOSaute for OmniSafe)
    if args.algo == "SautRL":
        omnisafe_algo = "PPOSaute"
        logger.info("Mapping SautRL to PPOSaute for OmniSafe compatibility")
    else:
        omnisafe_algo = args.algo
    
    # Verify algorithm is supported
    if omnisafe_algo not in omnisafe.ALGORITHMS['on-policy']:
        raise ValueError(f"Algorithm {omnisafe_algo} not found in OmniSafe. Available: {omnisafe.ALGORITHMS['on-policy']}")
    
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
    
    # Register our custom environment with gymnasium
    safe_env_id = f"SafeISO-{args.env_id}-v0"
    
    # Register the environment factory with gymnasium
    try:
        gym.spec(safe_env_id)
        logger.info(f"Environment {safe_env_id} already registered")
    except gym.error.UnregisteredEnv:
        gym.register(
            id=safe_env_id,
            entry_point=lambda: make_safe_iso_env(
                env_id=args.env_id,
                env_kwargs=env_kwargs,
                cost_threshold=args.cost_threshold
            )
        )
        logger.info(f"Registered environment {safe_env_id}")
    
    # Log the environment and algorithm details
    logger.info(f"Training safe ISO agent with {omnisafe_algo} algorithm")
    logger.info(f"Base environment: {args.env_id}")
    logger.info(f"Safe environment ID: {safe_env_id}")
    logger.info(f"Environment kwargs: {env_kwargs}")
    logger.info(f"Cost threshold: {args.cost_threshold}")
    
    # Create custom configuration
    custom_cfgs = create_custom_cfgs(omnisafe_algo, args.cost_threshold, args.num_steps, log_dir, args.device)
    
    # Add seed to custom configuration
    custom_cfgs["seed"] = args.seed
    
    # Terminal configuration (empty for now)
    train_terminal_cfgs = None
    
    logger.info(f"Creating OmniSafe agent with algorithm: {omnisafe_algo}")
    logger.info(f"Custom configuration: {custom_cfgs}")
    
    # Create the agent using OmniSafe 0.5.0 API
    agent = omnisafe.Agent(
        algo=omnisafe_algo,
        env_id=safe_env_id,
        train_terminal_cfgs=train_terminal_cfgs,
        custom_cfgs=custom_cfgs
    )
    
    logger.info(f"Successfully created {omnisafe_algo} agent using OmniSafe")
    
    # Train the agent
    start_time = time.time()
    logger.info("Starting training...")
    
    try:
        # Train using OmniSafe's learn method
        agent.learn()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error("Training failed. Check the configuration and environment setup.")
        raise
    finally:
        # Calculate training time
        training_time = time.time() - start_time
        logger.info(f"Training took {training_time:.2f} seconds")
        
        # Save final model (OmniSafe automatically saves during training)
        final_model_path = os.path.join(log_dir, "final_model.pt")
        if os.path.exists(os.path.join(log_dir, "torch_save")):
            # Copy the saved model to a standard location
            import shutil
            saved_models = os.listdir(os.path.join(log_dir, "torch_save"))
            if saved_models:
                latest_model = sorted(saved_models)[-1]
                shutil.copy(
                    os.path.join(log_dir, "torch_save", latest_model),
                    final_model_path
                )
                logger.info(f"Final model copied to {final_model_path}")
            else:
                logger.warning("No saved model found in torch_save directory")
        else:
            logger.warning("No torch_save directory found - model may not have been saved")

if __name__ == "__main__":
    main() 