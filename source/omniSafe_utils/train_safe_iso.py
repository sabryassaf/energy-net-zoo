#!/usr/bin/env python
"""
Training script for Safe ISO Agent using OmniSafe 0.5.0

Trains safe ISO agents using constraint-based RL algorithms from OmniSafe.
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

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import OmniSafe first
import omnisafe

# Register base environments first
from base_env_register import register_base_environments
register_base_environments(max_episode_steps=500)

# Import OmniSafe environments - registration handled by @env_register decorator
from omniSafe_env_register import SafeISOCMDP, make_omnisafe_iso_env

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_safe_iso")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a safe ISO agent with OmniSafe")
    
    parser.add_argument("--env-id", type=str, default="SafeISO-v0", 
                        help="Environment ID (default: SafeISO-v0)")
    parser.add_argument("--cost-threshold", type=float, default=25.0,
                        help="Cost threshold for constraints (default: 25.0)")
    parser.add_argument("--algo", type=str, default="PPOLag", 
                        choices=["PPOLag", "CPO", "FOCOPS", "CUP", "PPOSaute"],
                        help="Safe RL algorithm to use (default: PPOLag)")
    parser.add_argument("--num-steps", type=int, default=1000000,
                        help="Total number of timesteps (default: 1M)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run on (auto, cpu, cuda, cuda:0, etc.)")
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
    parser.add_argument("--log-dir", type=str, default="logs/safe_iso",
                        help="Directory for saving logs (default: logs/safe_iso)")
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    if args.algo == "SautRL":
        omnisafe_algo = "PPOSaute"
        logger.info("Mapping SautRL to PPOSaute for OmniSafe compatibility")
    else:
        omnisafe_algo = args.algo
    
    if omnisafe_algo not in omnisafe.ALGORITHMS['on-policy']:
        raise ValueError("Algorithm {} not found in OmniSafe. Available: {}".format(omnisafe_algo, omnisafe.ALGORITHMS['on-policy']))
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.algo, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    with open(os.path.join(log_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    logger.info(f"Training safe ISO agent with {omnisafe_algo} algorithm")
    logger.info(f"Environment ID: {args.env_id}")
    logger.info(f"Cost threshold: {args.cost_threshold}")
    logger.info(f"Pricing policy: {args.pricing_policy}")
    logger.info(f"Demand pattern: {args.demand_pattern}")
    logger.info(f"Cost type: {args.cost_type}")
    logger.info(f"Use dispatch: {args.use_dispatch}")
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Auto-detected device: {device}")
    else:
        device = args.device
    
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': args.num_steps,
            'device': device,
        },
        'logger_cfgs': {
            'log_dir': log_dir,
        },
        'seed': args.seed,
    }
    
    logger.info(f"Creating OmniSafe agent with algorithm: {omnisafe_algo}")
    logger.info(f"Custom configuration: {custom_cfgs}")
    
    start_time = time.time()
    try:
        logger.info("Using SafeISO CMDP environment registered with OmniSafe...")
        
        agent = omnisafe.Agent(
            algo=omnisafe_algo,
            env_id=args.env_id,
            train_terminal_cfgs=None,
            custom_cfgs=custom_cfgs
        )
        
        logger.info(f"Successfully created {omnisafe_algo} agent with SafeISO environment")
        logger.info("Starting training...")
        
        agent.learn()
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error("Training failed. Check the configuration and environment setup.")
        logger.error("Make sure the SafeISO CMDP environment is properly registered.")
        raise
    finally:
        training_time = time.time() - start_time
        logger.info(f"Training took {training_time:.2f} seconds")

if __name__ == "__main__":
    main() 