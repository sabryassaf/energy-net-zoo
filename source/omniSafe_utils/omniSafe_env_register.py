#!/usr/bin/env python
"""
OmniSafe Environment Registration for SafeISO

This module contains all OmniSafe-related environment wrappers and utilities
for the SafeISO smart grid energy management environment.

Includes:
- SafeISOCMDP: OmniSafe-compatible CMDP wrapper
- Explicit environment registration functions
"""

from __future__ import annotations

from typing import Any, ClassVar
import numpy as np
import torch
import gymnasium as gym
import sys
import os

# Add source directories to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omnisafe.common.logger import Logger
from omnisafe.envs.core import CMDP, env_register

from safe_iso_wrapper import SafeISOWrapper, make_safe_iso_env
from base_env_register import register_base_environments
from responsive_pcs_policy import ResponsivePCSPolicy

# Register base environments with Gymnasium
register_base_environments(max_episode_steps=500)


def _patch_energy_net_logging():
    """
    Simple patch to fix energy_net numpy array logging TypeError.
    """
    try:
        import energy_net.alternating_wrappers as aw
        import logging
        
        # Just suppress the problematic logger entirely
        aw_logger = logging.getLogger('alternating_wrappers')
        aw_logger.setLevel(logging.WARNING)  # Only show warnings and above
        
        print("Applied energy_net logging patch")
        
    except Exception as e:
        print(f"Warning: Could not patch energy_net logging: {e}")


# Register SafeISO environments with OmniSafe using decorator
@env_register
class SafeISOCMDP(CMDP):
    """OmniSafe-compatible CMDP wrapper for ISO-RLZoo environment with standard responsive PCS policy."""
    
    _support_envs: ClassVar[list[str]] = [
        'SafeISO-v0',
        'SafeISO-ONLINE-v0', 
        'SafeISO-QUADRATIC-v0',
        'SafeISO-CONSTANT-v0'
    ]
    
    metadata: ClassVar[dict[str, int]] = {'render_fps': 30}
    
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False
    _num_envs = 1
    
    def __init__(
        self,
        env_id: str = 'SafeISO-v0',
        cost_threshold: float = 25.0,
        pricing_policy: str = "ONLINE",
        demand_pattern: str = "SINUSOIDAL", 
        cost_type: str = "CONSTANT",
        use_dispatch: bool = False,
        max_episode_steps: int = 500,
        constraint_config: dict = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_id, **kwargs)
        
        self.cost_threshold = cost_threshold
        self.pricing_policy = pricing_policy
        self.demand_pattern = demand_pattern
        self.cost_type = cost_type
        self.use_dispatch = use_dispatch
        
        # Prepare environment kwargs with constraint config
        env_kwargs = {
            'pricing_policy': pricing_policy,
            'demand_pattern': demand_pattern,
            'cost_type': cost_type,
            'max_episode_steps': max_episode_steps,
        }
        
        # Add constraint config if provided
        if constraint_config:
            env_kwargs['constraint_config'] = constraint_config
            
        env_kwargs.update(kwargs)
        
        # Create the Safe ISO wrapper with standard responsive PCS policy
        # All algorithms use the same PCS policy for fair comparison
        pcs_policy = ResponsivePCSPolicy()
        
        self._env = make_safe_iso_env(
            env_id='ISO-RLZoo-v0',
            cost_threshold=cost_threshold,
            pcs_policy=pcs_policy,
            env_kwargs=env_kwargs,
        )
        
        self._observation_space = self._env.observation_space
        self._action_space = self._env.action_space
        
        self._episode_step = 0
        self._max_episode_steps = max_episode_steps
        
        self.env_spec_log = {
            'Env/VoltageViolations': 0,
            'Env/FrequencyViolations': 0,
            'Env/BatteryViolations': 0,
            'Env/SupplyDemandViolations': 0,
            'Env/TotalViolations': 0,
            'Env/TotalCost': 0.0,
            'Env/EpisodeSteps': 0,
        }
        
    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run one timestep returning obs, reward, cost, terminated, truncated, info."""
        action_np = action.detach().cpu().numpy()
        obs, reward, terminated, truncated, info = self._env.step(action_np)
        
        cost = info.get('cost', 0.0)
        self._episode_step += 1
        
        violations = info.get('violations', {})
        self.env_spec_log['Env/VoltageViolations'] += violations.get('voltage', 0)
        self.env_spec_log['Env/FrequencyViolations'] += violations.get('frequency', 0) 
        self.env_spec_log['Env/BatteryViolations'] += violations.get('battery', 0)
        self.env_spec_log['Env/SupplyDemandViolations'] += violations.get('supply_demand', 0)
        self.env_spec_log['Env/TotalCost'] += cost
        self.env_spec_log['Env/EpisodeSteps'] = self._episode_step
        
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32)
        cost_tensor = torch.as_tensor(cost, dtype=torch.float32)
        terminated_tensor = torch.as_tensor(terminated, dtype=torch.bool)
        truncated_tensor = torch.as_tensor(truncated, dtype=torch.bool)
        
        # Convert final_observation to tensor if present
        if 'final_observation' in info:
            info['final_observation'] = torch.as_tensor(info['final_observation'], dtype=torch.float32)
        
        return obs_tensor, reward_tensor, cost_tensor, terminated_tensor, truncated_tensor, info
        
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Reset environment and return initial observation."""
        if seed is not None:
            self.set_seed(seed)
            
        obs, info = self._env.reset(seed=seed, options=options)
        self._episode_step = 0
        
        self.env_spec_log['Env/VoltageViolations'] = 0
        self.env_spec_log['Env/FrequencyViolations'] = 0
        self.env_spec_log['Env/BatteryViolations'] = 0
        self.env_spec_log['Env/SupplyDemandViolations'] = 0
        self.env_spec_log['Env/TotalCost'] = 0.0
        self.env_spec_log['Env/EpisodeSteps'] = 0
        
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        return obs_tensor, info
        
    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps
        
    def spec_log(self, logger: Logger) -> None:
        """Log environment-specific metrics."""
        total_violations = sum([
            self.env_spec_log['Env/VoltageViolations'],
            self.env_spec_log['Env/FrequencyViolations'],
            self.env_spec_log['Env/BatteryViolations'],
            self.env_spec_log['Env/SupplyDemandViolations']
        ])
        self.env_spec_log['Env/TotalViolations'] = total_violations
        logger.store(self.env_spec_log)
        
    def set_seed(self, seed: int) -> None:
        """Set random seed for the environment."""
        if hasattr(self._env, 'set_seed'):
            self._env.set_seed(seed)
        elif hasattr(self._env, 'seed'):
            self._env.seed(seed)
        else:
            if hasattr(self._env, 'wrapped_env') and hasattr(self._env.wrapped_env, 'seed'):
                self._env.wrapped_env.seed(seed)
        
    def render(self) -> Any:
        """Render the environment."""
        return self._env.render()
        
    def close(self) -> None:
        """Close the environment."""
        if hasattr(self._env, 'close'):
            self._env.close()

def make_omnisafe_iso_env(
    env_id: str = 'SafeISO-v0',
    cost_threshold: float = 25.0,
    pricing_policy: str = "ONLINE",
    demand_pattern: str = "SINUSOIDAL",
    cost_type: str = "CONSTANT", 
    use_dispatch: bool = False,
    max_episode_steps: int = 500,
    constraint_config: dict = None,
    **kwargs: Any
) -> SafeISOCMDP:
    """Factory function to create OmniSafe-compatible ISO environment with standard PCS policy."""
    
    return SafeISOCMDP(
        env_id=env_id,
        cost_threshold=cost_threshold,
        pricing_policy=pricing_policy,
        demand_pattern=demand_pattern,
        cost_type=cost_type,
        use_dispatch=use_dispatch,
        max_episode_steps=max_episode_steps,
        constraint_config=constraint_config,
        **kwargs
    )


def test_omnisafe_environment():
    """
    Test function to verify the OmniSafe environment works correctly.
    """
    print("=== Testing Standard OmniSafe SafeISO Environment ===")
       
    # Create environment with standard PCS policy
    try:
        env = make_omnisafe_iso_env(
            env_id='SafeISO-v0',
            cost_threshold=25.0,
            pricing_policy="ONLINE",
            demand_pattern="SINUSOIDAL"
        )
        
        print("Successfully created SafeISO CMDP environment")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        print("   All algorithms will train on the same environment for fair comparison")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print(f"Reset successful, obs shape: {obs.shape}")
        
        # Test a few steps
        for i in range(5):
            action = torch.tensor([0.1 * i - 0.2])  # Simple test actions
            obs, reward, cost, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: reward={reward:.4f}, cost={cost:.4f}")
            
            if terminated or truncated:
                break
                
        env.close()
        print("Environment test completed successfully")
        
    except Exception as e:
        print(f"Environment test failed: {e}")
        raise


if __name__ == "__main__":
    test_omnisafe_environment() 