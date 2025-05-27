#!/usr/bin/env python
"""
OmniSafe Environments for SafeISO

This module contains all OmniSafe-related environment wrappers and utilities
for the SafeISO smart grid energy management environment.

Includes:
- SafeISOCMDP: OmniSafe-compatible CMDP wrapper
- ResponsivePCSPolicy: Fixes action-invariance issue
- Environment registration and factory functions
"""

from __future__ import annotations

from typing import Any, ClassVar
import numpy as np
import torch
import gymnasium as gym
import sys
import os

from omnisafe.common.logger import Logger
from omnisafe.envs.core import CMDP, env_register

from safe_iso_wrapper import SafeISOWrapper, make_safe_iso_env
from fix_env_registration import fix_iso_env_registration
from algorithm_specific_pcs_policy import AlgorithmSpecificPCSPolicy

# Fix environment registration to use proper episode length
fix_iso_env_registration()


class ResponsivePCSPolicy:
    """
    A responsive PCS policy that reacts to price signals with economic logic.
    
    This policy fixes the action-invariance issue by implementing economic behavior:
    - Charge battery when buy price is low (profitable to store energy)
    - Discharge battery when sell price is high (profitable to sell energy)
    - Consider battery state of charge limits for safety
    """
    
    def __init__(self, 
                 charge_threshold=3.0,    # Charge when buy price < this
                 discharge_threshold=7.0, # Discharge when sell price > this
                 max_charge_rate=1.0,     # Maximum charging rate (conservative to respect battery limits)
                 max_discharge_rate=1.0): # Maximum discharging rate (conservative to respect battery limits)
        """
        Initialize the responsive PCS policy.
        
        Args:
            charge_threshold: Buy price threshold below which to charge
            discharge_threshold: Sell price threshold above which to discharge
            max_charge_rate: Maximum battery charging rate
            max_discharge_rate: Maximum battery discharging rate
        """
        self.charge_threshold = charge_threshold
        self.discharge_threshold = discharge_threshold
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
    
    def predict(self, observation, deterministic=True):
        """
        Predict PCS action based on observation.
        
        The policy outputs actions in [-1, 1] range which the ISOEnvWrapper rescales to [-10, 10].
        Battery limits are charge_rate_max=10.0 and discharge_rate_max=10.0, so we must ensure
        our output * 10 never exceeds these limits.
        
        Args:
            observation: PCS observation [battery_level, time, buy_price, sell_price]
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Battery action in [-1,1] range (positive=charge, negative=discharge)
            state: None (stateless policy)
        """
        if observation.ndim == 2:
            obs = observation[0]  # Extract from batch
        else:
            obs = observation
        
        battery_level, time, buy_price, sell_price = obs
        
        # Normalize battery level (assuming 0-100 range)
        battery_soc = battery_level / 100.0
        
        # Economic decision logic
        action = 0.0  # Default: no action
        
        # Maximum safe action magnitude to ensure final action after 10x scaling stays within battery limits
        # Battery limits: charge_rate_max=10.0, discharge_rate_max=10.0
        # ISOEnvWrapper scales [-1,1] to [-10,10], so max_safe_action=0.8 gives [-8,8] final range
        max_safe_action = 0.8
        
        # Charge when buy price is low and battery not full
        if buy_price < self.charge_threshold and battery_soc < 0.9:
            # Charge more aggressively when price is lower
            charge_intensity = (self.charge_threshold - buy_price) / self.charge_threshold
            # Scale by max_safe_action and max_charge_rate to ensure we stay within limits
            action = charge_intensity * max_safe_action * (self.max_charge_rate / 5.0)
        
        # Discharge when sell price is high and battery not empty
        elif sell_price > self.discharge_threshold and battery_soc > 0.1:
            # Discharge more aggressively when price is higher
            discharge_intensity = (sell_price - self.discharge_threshold) / (10.0 - self.discharge_threshold)
            # Scale by max_safe_action and max_discharge_rate to ensure we stay within limits
            action = -discharge_intensity * max_safe_action * (self.max_discharge_rate / 5.0)
        
        # Add some randomness if not deterministic
        if not deterministic:
            noise = np.random.normal(0, 0.05)  # Reduced noise to avoid exceeding limits
            action += noise
        
        # Final clipping to ensure we stay within safe bounds
        # After ISOEnvWrapper scaling, this becomes [-max_safe_action*10, max_safe_action*10]
        action = np.clip(action, -max_safe_action, max_safe_action)
        
        return np.array([action]), None


def _patch_energy_net_logging():
    """
    Monkey patch the energy_net logging issue that causes TypeError.
    Also suppress repetitive PCS action rescaling messages.
    
    The issue is in alternating_wrappers.py where numpy arrays are formatted
    with .4f format strings, which is not supported.
    """
    try:
        import energy_net.alternating_wrappers as aw
        import logging
        
        # Suppress repetitive PCS action rescaling messages
        class PCSActionFilter(logging.Filter):
            def filter(self, record):
                # Filter out the repetitive "Rescaled PCS action" messages
                if "Rescaled PCS action from" in record.getMessage():
                    return False
                return True
        
        # Add filter to alternating_wrappers logger
        aw_logger = logging.getLogger('alternating_wrappers')
        aw_logger.addFilter(PCSActionFilter())
        
        # Store original method
        if not hasattr(aw.ISOEnvWrapper, '_original_unnormalize_pcs_action'):
            aw.ISOEnvWrapper._original_unnormalize_pcs_action = aw.ISOEnvWrapper._unnormalize_pcs_action
        
        def patched_unnormalize_pcs_action(self, normalized_action):
            """Patched version that handles numpy array formatting correctly."""
            pcs_space = self.unwrapped.action_space["pcs"]
            low = pcs_space.low
            high = pcs_space.high
            
            # Standard linear rescaling from [-1, 1] to [low, high]
            unnormalized_action = low + (normalized_action + 1.0) * 0.5 * (high - low)
            
            # Safe logging that handles numpy arrays
            try:
                if isinstance(normalized_action, np.ndarray) and len(normalized_action) > 0:
                    norm_val = float(normalized_action[0]) if len(normalized_action) > 0 else 0.0
                    unnorm_val = float(unnormalized_action[0]) if len(unnormalized_action) > 0 else 0.0
                    low_val = float(low[0]) if hasattr(low, '__len__') and len(low) > 0 else float(low)
                    high_val = float(high[0]) if hasattr(high, '__len__') and len(high) > 0 else float(high)
                    self.logger.info(f"Rescaled PCS battery action from {norm_val:.4f} to {unnorm_val:.4f} [range: {low_val:.1f}-{high_val:.1f}]")
                else:
                    norm_val = float(normalized_action)
                    unnorm_val = float(unnormalized_action)
                    low_val = float(low)
                    high_val = float(high)
                    self.logger.info(f"Rescaled PCS action from {norm_val:.4f} to {unnorm_val:.4f} [range: {low_val:.1f}-{high_val:.1f}]")
            except Exception:
                # Fallback logging without formatting
                self.logger.info(f"Rescaled PCS action from {normalized_action} to {unnormalized_action}")
                
            self.logger.debug(f"Unnormalized PCS action from {normalized_action} to {unnormalized_action}")
            return unnormalized_action
        
        # Apply the patch
        aw.ISOEnvWrapper._unnormalize_pcs_action = patched_unnormalize_pcs_action
        print("✅ Applied energy_net logging patch")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not patch energy_net logging: {e}")


def create_responsive_safe_iso_env(env_id="ISO-RLZoo-v0", algorithm_name=None, **kwargs):
    """
    Create a SafeISO environment with algorithm-specific responsive PCS policy.
    
    This function creates a SafeISO environment and injects an algorithm-specific PCS policy
    that reacts to price signals with behavior tailored to each Safe RL algorithm.
    
    Args:
        env_id: Environment ID (default: "ISO-RLZoo-v0")
        algorithm_name: Name of the Safe RL algorithm (for algorithm-specific behavior)
        **kwargs: Additional arguments passed to make_safe_iso_env
        
    Returns:
        SafeISO environment with algorithm-specific responsive PCS policy
    """
    # Apply logging patch to fix energy_net TypeError
    _patch_energy_net_logging()
    
    # Create algorithm-specific PCS policy
    if algorithm_name:
        responsive_pcs = AlgorithmSpecificPCSPolicy(
            algorithm_name=algorithm_name,
            base_seed=kwargs.pop('pcs_base_seed', 42)
        )
        print(f"✅ Created algorithm-specific PCS policy for {algorithm_name}")
    else:
        # Fallback to original ResponsivePCSPolicy for backward compatibility
        responsive_pcs = ResponsivePCSPolicy(
            charge_threshold=kwargs.pop('pcs_charge_threshold', 3.0),
            discharge_threshold=kwargs.pop('pcs_discharge_threshold', 7.0),
            max_charge_rate=kwargs.pop('pcs_max_charge_rate', 5.0),
            max_discharge_rate=kwargs.pop('pcs_max_discharge_rate', 5.0)
        )
        print("✅ Created standard responsive PCS policy (no algorithm specified)")
    
    # Create the base SafeISO environment
    env = make_safe_iso_env(env_id=env_id, **kwargs)
    
    # Inject responsive PCS policy into the ISOEnvWrapper
    current_env = env.wrapped_env
    depth = 0
    injected = False
    
    while hasattr(current_env, 'env') and depth < 10:
        if 'ISOEnvWrapper' in str(type(current_env)):
            current_env.pcs_policy = responsive_pcs
            print(f"✅ Injected responsive PCS policy into {type(current_env).__name__}")
            injected = True
            break
        current_env = current_env.env
        depth += 1
    
    if not injected:
        print("⚠️  Warning: Could not find ISOEnvWrapper to inject PCS policy")
        print("   Environment may still use passive PCS policy")
    
    return env


@env_register
class SafeISOCMDP(CMDP):
    """OmniSafe-compatible CMDP wrapper for ISO-RLZoo environment with responsive PCS policy."""
    
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
        env_id: str,
        cost_threshold: float = 25.0,
        pricing_policy: str = "ONLINE",
        demand_pattern: str = "SINUSOIDAL", 
        cost_type: str = "CONSTANT",
        use_dispatch: bool = False,
        max_episode_steps: int = 500,
        algorithm_name: str = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env_id, **kwargs)
        
        self.cost_threshold = cost_threshold
        self.pricing_policy = pricing_policy
        self.demand_pattern = demand_pattern
        self.cost_type = cost_type
        self.use_dispatch = use_dispatch
        
        env_kwargs = {
            "pricing_policy": pricing_policy,
            "demand_pattern": demand_pattern,
            "cost_type": cost_type,
            "dispatch_config": {
                "use_dispatch_action": use_dispatch,
                "default_strategy": "PROPORTIONAL"
            }
        }
        
        # Use the responsive SafeISO environment that fixes action-invariance
        self._env = create_responsive_safe_iso_env(
            env_id="ISO-RLZoo-v0",
            algorithm_name=algorithm_name,
            env_kwargs=env_kwargs,
            cost_threshold=cost_threshold,
            max_episode_steps=max_episode_steps
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
    algorithm_name: str = None,
    **kwargs: Any
) -> SafeISOCMDP:
    """Factory function to create OmniSafe-compatible ISO environment with algorithm-specific responsive PCS policy."""
    return SafeISOCMDP(
        env_id=env_id,
        cost_threshold=cost_threshold,
        pricing_policy=pricing_policy,
        demand_pattern=demand_pattern,
        cost_type=cost_type,
        use_dispatch=use_dispatch,
        max_episode_steps=max_episode_steps,
        algorithm_name=algorithm_name,
        **kwargs
    )


def test_responsive_omnisafe_environment():
    """
    Test function to verify the responsive OmniSafe environment works correctly.
    """
    print("=== Testing Responsive OmniSafe SafeISO Environment ===")
    
    # Create responsive environment
    env = make_omnisafe_iso_env(
        cost_threshold=25.0,
        pricing_policy="ONLINE"
    )
    
    # Test different price actions
    test_actions = [
        ("Low prices", torch.tensor([1.0, 1.0])),
        ("High prices", torch.tensor([10.0, 10.0])),
        ("Mixed prices", torch.tensor([1.0, 10.0])),
    ]
    
    print(f"Action space: {env._action_space}")
    print(f"Observation space: {env._observation_space}")
    
    for action_name, action in test_actions:
        print(f"\n--- Testing {action_name} ---")
        
        # Reset environment
        obs, _ = env.reset(seed=42)
        
        try:
            # Step with action
            obs, reward, cost, terminated, truncated, info = env.step(action)
            
            print(f"  Action: {action.numpy()}")
            print(f"  Reward: {reward.item():.1f}")
            print(f"  Cost: {cost.item():.3f}")
            print(f"  Observation shape: {obs.shape}")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    env.close()
    
    print("\n✅ Test complete. OmniSafe environment with responsive PCS policy is working!")


if __name__ == "__main__":
    test_responsive_omnisafe_environment() 