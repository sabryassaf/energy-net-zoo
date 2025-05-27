#!/usr/bin/env python
"""
Fix environment registration to use proper episode length
"""

import gymnasium as gym
from gymnasium.envs.registration import register, registry

def fix_iso_env_registration():
    """Fix the ISO-RLZoo-v0 environment registration to use proper episode length."""
    
    # First, unregister the existing environment if it exists
    if 'ISO-RLZoo-v0' in registry:
        del registry['ISO-RLZoo-v0']
        print("Unregistered existing ISO-RLZoo-v0")
    
    # Re-register with proper episode length
    print("Re-registering ISO-RLZoo-v0 with max_episode_steps=500")
    register(
        id='ISO-RLZoo-v0',
        entry_point='energy_net.env.iso_env:make_iso_env_zoo',
        max_episode_steps=500,  # Proper episode length
    )
    
    # Also fix PCS environment
    if 'PCS-RLZoo-v0' in registry:
        del registry['PCS-RLZoo-v0']
        print("Unregistered existing PCS-RLZoo-v0")
    
    print("Re-registering PCS-RLZoo-v0 with max_episode_steps=500")
    register(
        id='PCS-RLZoo-v0',
        entry_point='energy_net.env.pcs_env:make_pcs_env_zoo',
        max_episode_steps=500,  # Proper episode length
    )
    
    print("Environment registration fixed!")

if __name__ == "__main__":
    fix_iso_env_registration() 