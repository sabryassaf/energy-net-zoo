#!/usr/bin/env python
"""
Register base Gymnasium environments with proper episode length
"""

import gymnasium as gym
from gymnasium.envs.registration import register, registry

def register_base_environments(max_episode_steps=500):
    """Register the base ISO and PCS environments with configurable episode length.
    
    Args:
        max_episode_steps (int): Maximum number of steps per episode (default: 500)
    """
    
    # First, unregister the existing environment if it exists
    if 'ISO-RLZoo-v0' in registry:
        del registry['ISO-RLZoo-v0']
    
    # Re-register with configurable episode length
    register(
        id='ISO-RLZoo-v0',
        entry_point='energy_net.env.iso_env:make_iso_env_zoo',
        max_episode_steps=max_episode_steps,
    )
    
    # Also register PCS environment
    if 'PCS-RLZoo-v0' in registry:
        del registry['PCS-RLZoo-v0']
    
    register(
        id='PCS-RLZoo-v0',
        entry_point='energy_net.env.pcs_env:make_pcs_env_zoo',
        max_episode_steps=max_episode_steps,
    )

if __name__ == "__main__":
    register_base_environments() 