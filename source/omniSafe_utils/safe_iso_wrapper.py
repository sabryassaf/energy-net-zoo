#!/usr/bin/env python
"""
Safe ISO Wrapper for Energy Net environment
This wrapper adapts the ISO-RLZoo-v0 environment to work with OmniSafe's constraint-based algorithms
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import omnisafe
import logging
from typing import Dict, Tuple, Any, Optional, Union, List

# Try different import paths based on OmniSafe versions
try:
    # For newer OmniSafe versions
    from omnisafe.common.safe_env import SafeEnv
    from omnisafe.common.episode import EpisodeState
except ImportError:
    try:
        # For OmniSafe 0.5.0
        from omnisafe.envs.core import CMDP as SafeEnv
        
        # Create a simpler EpisodeState implementation
        class EpisodeState:
            def __init__(self):
                self.rewards = []
                self.costs = []
                self.terminated = False
            
            def reset(self):
                self.rewards = []
                self.costs = []
                self.terminated = False
            
            def step(self, obs, reward, cost, done):
                self.rewards.append(reward)
                self.costs.append(cost)
                self.terminated = done
    except ImportError:
        print("ERROR: Could not find appropriate OmniSafe classes. Try reinstalling omnisafe.")
        raise

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("safe_iso_wrapper")

class SafeISOWrapper(SafeEnv):
    """
    This wrapper adapts the ISO-RLZoo-v0 environment to work with OmniSafe
    by adding safety constraints related to:
    - Grid voltage stability
    - Frequency stability
    - Battery damage prevention
    - Supply-demand balance
    """
    
    # Define supported environments
    _support_envs = ["ISO-RLZoo-v0"]
    
    def __init__(
        self,
        env_id: str = "ISO-RLZoo-v0",
        env_kwargs: Optional[Dict[str, Any]] = None,
        cost_threshold: float = 25.0,
        normalize_reward: bool = True,
        **kwargs,
    ):
        """
        Initialize the SafeISOWrapper.
        
        Args:
            env_id: The ID of the ISO environment to wrap
            env_kwargs: Additional keyword arguments for the environment
            cost_threshold: The threshold for considering a cost violation
            normalize_reward: Whether to normalize rewards
        """
        # Extract constraint config from env_kwargs
        env_kwargs = env_kwargs or {}
        constraint_config = env_kwargs.pop('constraint_config', {})
        
        # Store the environment ID for reference
        self.env_id = env_id
        
        # First register the environments if needed
        try:
            import energy_net.env.register_envs
        except ImportError:
            logger.error("Could not import energy_net.env.register_envs")
            raise
        
        # Create the actual environment
        self.wrapped_env = gym.make(env_id, **env_kwargs)
        
        # Override max_episode_steps if provided in env_kwargs
        if 'max_episode_steps' in env_kwargs:
            self._override_max_episode_steps(env_kwargs['max_episode_steps'])
        
        # Set up cost threshold
        self.cost_threshold = cost_threshold
        
        # Initialize the parent class first to handle observation and action spaces
        super().__init__(env_id)
        
        # Initialize constraints with configurable parameters
        self.cost_function_kwargs = {
            "voltage_cost_weight": constraint_config.get("voltage_cost_weight", 1.0),
            "frequency_cost_weight": constraint_config.get("frequency_cost_weight", 1.0),
            "battery_cost_weight": constraint_config.get("battery_cost_weight", 1.0),
            "supply_demand_cost_weight": constraint_config.get("supply_demand_cost_weight", 2.0),
            "voltage_limit_min": constraint_config.get("voltage_limit_min", 0.95),
            "voltage_limit_max": constraint_config.get("voltage_limit_max", 1.05),
            "frequency_limit_min": constraint_config.get("frequency_limit_min", 49.8),
            "frequency_limit_max": constraint_config.get("frequency_limit_max", 50.2),
            "battery_soc_min": constraint_config.get("battery_soc_min", 0.1),
            "battery_soc_max": constraint_config.get("battery_soc_max", 0.9),
            "supply_demand_imbalance_threshold": constraint_config.get("supply_demand_imbalance_threshold", 10.0),
        }
        
        # Additional tracking variables
        self.episode_costs = []
        self.current_episode_cost = 0.0
        self.episode_constraint_violations = {
            "voltage": 0,
            "frequency": 0,
            "battery": 0,
            "supply_demand": 0
        }
        
        # Normalize reward settings
        self.normalize_reward = normalize_reward
        self.reward_scaling = 0.1  # Scale rewards for stability
        
        # Initialize the episode state
        self.episode_state = EpisodeState()
        
        # Set up action and observation spaces from wrapped environment
        self._action_space = self.wrapped_env.action_space
        self._observation_space = self.wrapped_env.observation_space
        
        # Fix Monitor wrapper reset state issue
        self._fix_monitor_wrapper_state()
        
        logger.info(f"SafeISOWrapper initialized with cost threshold: {self.cost_threshold}")
    
    @property
    def action_space(self):
        """Action space of the environment."""
        return self._action_space
    
    @property
    def observation_space(self):
        """Observation space of the environment."""
        return self._observation_space
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and return the initial observation."""
        obs, info = self.wrapped_env.reset(**kwargs)
        
        # Ensure info is a dictionary
        if not isinstance(info, dict):
            info = {}
        
        # Reset cost tracking
        self.current_episode_cost = 0.0
        self.episode_constraint_violations = {
            "voltage": 0,
            "frequency": 0,
            "battery": 0,
            "supply_demand": 0
        }
        
        # Reset episode state
        self.episode_state.reset()
        
        # Add safety info to the info dict
        info["constraint_values"] = np.zeros(4)  # 4 constraints
        info["cost"] = 0.0
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment using the given action.
        Computes safety costs based on the state after the step.
        """
        # Ensure Monitor wrapper is in correct state before stepping
        self._fix_monitor_wrapper_state()
        
        # Take a step in the wrapped environment
        obs, reward, terminated, truncated, info = self.wrapped_env.step(action)
        
        # Get the constraint-related values from the info dict
        voltage = info.get("voltage", 1.0)  # Default to nominal if not provided
        frequency = info.get("frequency", 50.0)  # Default to nominal if not provided
        battery_soc = info.get("battery_soc", 0.5)  # Default to middle if not provided
        supply_demand_imbalance = info.get("supply_demand_imbalance", 0.0)
        
        # Calculate costs for each constraint
        voltage_cost = self._calculate_voltage_cost(voltage)
        frequency_cost = self._calculate_frequency_cost(frequency)
        battery_cost = self._calculate_battery_cost(battery_soc)
        supply_demand_cost = self._calculate_supply_demand_cost(supply_demand_imbalance)
        
        # Combine costs
        cost = voltage_cost + frequency_cost + battery_cost + supply_demand_cost
        
        # Track violations
        if voltage_cost > 0:
            self.episode_constraint_violations["voltage"] += 1
        if frequency_cost > 0:
            self.episode_constraint_violations["frequency"] += 1
        if battery_cost > 0:
            self.episode_constraint_violations["battery"] += 1
        if supply_demand_cost > 0:
            self.episode_constraint_violations["supply_demand"] += 1
        
        # Update episode cost
        self.current_episode_cost += cost
        
        # Update episode state
        self.episode_state.step(obs, reward, cost, terminated or truncated)
        
        # Scale reward if needed
        if self.normalize_reward:
            reward = reward * self.reward_scaling
        
        # Add constraint information to info dict
        info["constraint_values"] = np.array([
            voltage_cost, 
            frequency_cost,
            battery_cost,
            supply_demand_cost
        ])
        info["cost"] = cost
        info["constraint_violations"] = self.episode_constraint_violations.copy()
        
        # Add final_observation for OmniSafe when episode ends
        if terminated or truncated:
            info["final_observation"] = obs
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_voltage_cost(self, voltage: float) -> float:
        """Calculate cost for voltage constraint violations."""
        min_limit = self.cost_function_kwargs["voltage_limit_min"]
        max_limit = self.cost_function_kwargs["voltage_limit_max"]
        weight = self.cost_function_kwargs["voltage_cost_weight"]
        
        if voltage < min_limit:
            return weight * (min_limit - voltage)**2
        elif voltage > max_limit:
            return weight * (voltage - max_limit)**2
        return 0.0
    
    def _calculate_frequency_cost(self, frequency: float) -> float:
        """Calculate cost for frequency constraint violations."""
        min_limit = self.cost_function_kwargs["frequency_limit_min"]
        max_limit = self.cost_function_kwargs["frequency_limit_max"]
        weight = self.cost_function_kwargs["frequency_cost_weight"]
        
        if frequency < min_limit:
            return weight * (min_limit - frequency)**2
        elif frequency > max_limit:
            return weight * (frequency - max_limit)**2
        return 0.0
    
    def _calculate_battery_cost(self, battery_soc: float) -> float:
        """Calculate cost for battery SoC constraint violations."""
        min_limit = self.cost_function_kwargs["battery_soc_min"]
        max_limit = self.cost_function_kwargs["battery_soc_max"]
        weight = self.cost_function_kwargs["battery_cost_weight"]
        
        if battery_soc < min_limit:
            return weight * (min_limit - battery_soc)**2
        elif battery_soc > max_limit:
            return weight * (battery_soc - max_limit)**2
        return 0.0
    
    def _calculate_supply_demand_cost(self, imbalance: float) -> float:
        """Calculate cost for supply-demand imbalance."""
        threshold = self.cost_function_kwargs["supply_demand_imbalance_threshold"]
        weight = self.cost_function_kwargs["supply_demand_cost_weight"]
        
        if abs(imbalance) > threshold:
            return weight * (abs(imbalance) - threshold)**2
        return 0.0
    
    def get_episode_costs(self) -> List[float]:
        """Returns the costs for the current episode."""
        return self.episode_state.costs
    
    def get_episode_constraint_violations(self) -> Dict[str, int]:
        """Returns the constraint violations for the current episode."""
        return self.episode_constraint_violations.copy()
    
    def render(self, **kwargs) -> Optional[np.ndarray]:
        """Render the environment."""
        if hasattr(self.wrapped_env, 'render'):
            return self.wrapped_env.render(**kwargs)
        logger.warning("Render method called but not implemented in wrapped environment")
        return None
    
    def set_seed(self, seed: int) -> None:
        """Set the seed for this environment's random number generator."""
        if hasattr(self.wrapped_env, 'set_seed'):
            self.wrapped_env.set_seed(seed)
        elif hasattr(self.wrapped_env, 'seed'):
            self.wrapped_env.seed(seed)
        else:
            logger.warning("Cannot set seed: no seed method found in wrapped environment")
    
    def close(self) -> None:
        """Close the environment."""
        return self.wrapped_env.close()
    
    def _override_max_episode_steps(self, max_episode_steps: int) -> None:
        """Override the max_episode_steps in the underlying environment controller."""
        # Walk through the wrapper chain to find the base environment
        env = self.wrapped_env
        while hasattr(env, 'env'):
            env = env.env
        
        # Check if this is an EnergyNetV0 environment with a controller
        if hasattr(env, 'controller') and hasattr(env.controller, 'max_steps_per_episode'):
            old_value = env.controller.max_steps_per_episode
            env.controller.max_steps_per_episode = max_episode_steps
            logger.info(f"Overrode max_steps_per_episode: {old_value} -> {max_episode_steps}")
        else:
            logger.warning("Could not find controller.max_steps_per_episode to override")
    
    def _fix_monitor_wrapper_state(self) -> None:
        """Fix Monitor wrapper reset state to prevent 'needs reset' errors."""
        # Walk through the wrapper chain to find the Monitor wrapper specifically
        env = self.wrapped_env
        monitor_found = False
        while hasattr(env, 'env'):
            # Check if this is specifically the Monitor wrapper from stable_baselines3
            if (hasattr(env, 'needs_reset') and 
                type(env).__name__ == 'Monitor' and 
                'stable_baselines3' in str(type(env).__module__)):
                # Found the Monitor wrapper, set it to not need reset
                env.needs_reset = False
                monitor_found = True
                logger.debug(f"Fixed Monitor wrapper reset state: {type(env).__name__}")
                break
            env = env.env
        
        if not monitor_found:
            logger.warning("Monitor wrapper not found in environment chain")


def make_safe_iso_env(
    env_id: str = "ISO-RLZoo-v0",
    env_kwargs: Optional[Dict[str, Any]] = None,
    cost_threshold: float = 25.0,
    normalize_reward: bool = True,
    max_episode_steps: int = 500,
    **kwargs
) -> SafeISOWrapper:
    """
    Factory function to create a safe ISO environment.
    
    Args:
        env_id: The ID of the ISO environment to wrap
        env_kwargs: Additional keyword arguments for the environment
        cost_threshold: The threshold for considering a cost violation
        normalize_reward: Whether to normalize rewards
        max_episode_steps: Maximum number of steps per episode
        
    Returns:
        A SafeISOWrapper instance
    """
    # Ensure env_kwargs is a dictionary
    if env_kwargs is None:
        env_kwargs = {}
    
    # Add max_episode_steps to env_kwargs to pass to the underlying environment
    env_kwargs['max_episode_steps'] = max_episode_steps
    
    return SafeISOWrapper(
        env_id=env_id,
        env_kwargs=env_kwargs,
        cost_threshold=cost_threshold,
        normalize_reward=normalize_reward,
        **kwargs
    ) 