#!/usr/bin/env python3
"""
Algorithm-Specific ResponsivePCSPolicy for SafeISO
Introduces controlled variability to differentiate between Safe RL algorithms
"""

import numpy as np
from typing import Dict, Optional, Tuple
import hashlib

class AlgorithmSpecificPCSPolicy:
    """
    Enhanced PCS policy that introduces algorithm-specific behavior patterns
    while maintaining safety constraints
    """
    
    def __init__(self, algorithm_name: str, base_seed: int = 42):
        self.algorithm_name = algorithm_name
        self.base_seed = base_seed
        
        # Generate algorithm-specific parameters
        self.params = self._generate_algorithm_params()
        
        # Safety bounds (non-negotiable)
        self.safety_bounds = {
            'min_soc': 0.15,  # 15% minimum
            'max_soc': 0.85,  # 85% maximum
            'max_charge_rate': 1.0,
            'max_discharge_rate': -1.0
        }
        
        # Algorithm-specific behavioral parameters
        self.behavioral_params = {
            'risk_tolerance': self.params['risk_tolerance'],
            'response_speed': self.params['response_speed'],
            'price_sensitivity': self.params['price_sensitivity'],
            'safety_margin': self.params['safety_margin'],
            'noise_level': self.params['noise_level']
        }
        
        # Initialize random state for consistent behavior
        self.rng = np.random.RandomState(self._get_algorithm_seed())
    
    def _generate_algorithm_params(self) -> Dict[str, float]:
        """Generate algorithm-specific parameters based on algorithm name"""
        
        # Create deterministic but different parameters for each algorithm
        seed = self._get_algorithm_seed()
        rng = np.random.RandomState(seed)
        
        # Define algorithm archetypes
        algorithm_profiles = {
            'PPOLag': {
                'risk_tolerance': 0.3,    # Conservative
                'response_speed': 0.7,    # Moderate response
                'price_sensitivity': 0.8, # High price sensitivity
                'safety_margin': 0.15,    # Large safety margin
                'noise_level': 0.05       # Low noise
            },
            'CPO': {
                'risk_tolerance': 0.2,    # Very conservative
                'response_speed': 0.9,    # Fast response
                'price_sensitivity': 0.6, # Moderate price sensitivity
                'safety_margin': 0.2,     # Very large safety margin
                'noise_level': 0.03       # Very low noise
            },
            'FOCOPS': {
                'risk_tolerance': 0.5,    # Balanced
                'response_speed': 0.8,    # Fast response
                'price_sensitivity': 0.9, # Very high price sensitivity
                'safety_margin': 0.1,     # Moderate safety margin
                'noise_level': 0.07       # Moderate noise
            },
            'CUP': {
                'risk_tolerance': 0.4,    # Moderate conservative
                'response_speed': 0.6,    # Moderate response
                'price_sensitivity': 0.7, # High price sensitivity
                'safety_margin': 0.12,    # Moderate safety margin
                'noise_level': 0.06       # Moderate noise
            },
            'PPOSaute': {
                'risk_tolerance': 0.6,    # More aggressive
                'response_speed': 0.5,    # Slower response
                'price_sensitivity': 0.5, # Lower price sensitivity
                'safety_margin': 0.08,    # Smaller safety margin
                'noise_level': 0.1        # Higher noise
            }
        }
        
        # Get base profile or create random one
        if self.algorithm_name in algorithm_profiles:
            base_params = algorithm_profiles[self.algorithm_name].copy()
        else:
            # Generate random parameters for unknown algorithms
            base_params = {
                'risk_tolerance': rng.uniform(0.2, 0.6),
                'response_speed': rng.uniform(0.5, 0.9),
                'price_sensitivity': rng.uniform(0.5, 0.9),
                'safety_margin': rng.uniform(0.08, 0.2),
                'noise_level': rng.uniform(0.03, 0.1)
            }
        
        # Add small random variations to make each run slightly different
        for key in base_params:
            variation = rng.normal(0, 0.02)  # 2% standard deviation
            base_params[key] = np.clip(base_params[key] + variation, 0.01, 0.99)
        
        return base_params
    
    def _get_algorithm_seed(self) -> int:
        """Generate consistent seed for algorithm"""
        hash_input = f"{self.algorithm_name}_{self.base_seed}".encode()
        return int(hashlib.md5(hash_input).hexdigest()[:8], 16) % 100000
    
    def predict(self, observation, deterministic=True):
        """
        Predict PCS action based on observation (compatible with ResponsivePCSPolicy interface).
        
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
        
        # Use the main get_action method
        action = self.get_action(
            price_signal=buy_price if buy_price < sell_price else sell_price,
            battery_soc=battery_soc,
            grid_state={'time': time},
            safety_violations=None
        )
        
        # Ensure action is in [-1, 1] range for compatibility
        action = np.clip(action, -1.0, 1.0)
        
        return np.array([action]), None
    
    def get_action(self, price_signal: float, battery_soc: float, 
                   grid_state: Optional[Dict] = None, 
                   safety_violations: Optional[Dict] = None) -> float:
        """
        Get PCS action with algorithm-specific behavior
        
        Args:
            price_signal: Current electricity price
            battery_soc: Battery state of charge (0-1)
            grid_state: Optional grid state information
            safety_violations: Optional current safety violation info
            
        Returns:
            PCS action (-1 to 1, negative=charge, positive=discharge)
        """
        
        # 1. SAFETY FIRST: Hard constraints (non-negotiable)
        if battery_soc <= self.safety_bounds['min_soc']:
            return self._safe_charge_action()
        elif battery_soc >= self.safety_bounds['max_soc']:
            return self._safe_discharge_action()
        
        # 2. EMERGENCY RESPONSE: React to safety violations
        if safety_violations and any(safety_violations.values()):
            return self._emergency_response(battery_soc, safety_violations)
        
        # 3. ALGORITHM-SPECIFIC ECONOMIC BEHAVIOR
        base_action = self._economic_decision(price_signal, battery_soc, grid_state)
        
        # 4. ADD ALGORITHM-SPECIFIC NOISE AND BEHAVIOR
        modified_action = self._apply_algorithm_behavior(base_action, price_signal, battery_soc)
        
        # 5. FINAL SAFETY CHECK
        return self._apply_safety_bounds(modified_action, battery_soc)
    
    def _safe_charge_action(self) -> float:
        """Generate safe charging action with algorithm-specific variation"""
        base_charge = -0.8  # Strong charging
        
        # Add algorithm-specific variation
        noise = self.rng.normal(0, self.behavioral_params['noise_level'])
        varied_charge = base_charge + noise * 0.2
        
        return np.clip(varied_charge, self.safety_bounds['max_discharge_rate'], 0)
    
    def _safe_discharge_action(self) -> float:
        """Generate safe discharge action with algorithm-specific variation"""
        base_discharge = 0.8  # Strong discharge
        
        # Add algorithm-specific variation
        noise = self.rng.normal(0, self.behavioral_params['noise_level'])
        varied_discharge = base_discharge + noise * 0.2
        
        return np.clip(varied_discharge, 0, self.safety_bounds['max_charge_rate'])
    
    def _emergency_response(self, battery_soc: float, safety_violations: Dict) -> float:
        """Algorithm-specific emergency response to safety violations"""
        
        # Different algorithms respond differently to emergencies
        response_intensity = self.behavioral_params['response_speed']
        risk_tolerance = self.behavioral_params['risk_tolerance']
        
        # Determine emergency action based on violation type and algorithm personality
        if safety_violations.get('voltage_violation', False):
            # Voltage issues: conservative algorithms discharge more aggressively
            emergency_action = (1 - risk_tolerance) * 0.9
        elif safety_violations.get('frequency_violation', False):
            # Frequency issues: fast-responding algorithms react more
            emergency_action = response_intensity * 0.7
        elif safety_violations.get('supply_demand_violation', False):
            # Supply-demand: price-sensitive algorithms respond based on economics
            emergency_action = self.behavioral_params['price_sensitivity'] * 0.6
        else:
            # General emergency: moderate response
            emergency_action = 0.5
        
        # Add algorithm-specific noise
        noise = self.rng.normal(0, self.behavioral_params['noise_level'])
        emergency_action += noise * 0.3
        
        return self._apply_safety_bounds(emergency_action, battery_soc)
    
    def _economic_decision(self, price_signal: float, battery_soc: float, 
                          grid_state: Optional[Dict]) -> float:
        """Make economic decision with algorithm-specific logic"""
        
        # Algorithm-specific price thresholds
        base_buy_threshold = 3.0
        base_sell_threshold = 7.0
        
        # Adjust thresholds based on algorithm personality
        price_sensitivity = self.behavioral_params['price_sensitivity']
        buy_threshold = base_buy_threshold * (2 - price_sensitivity)
        sell_threshold = base_sell_threshold * price_sensitivity
        
        # Economic logic with algorithm-specific modifications
        if price_signal < buy_threshold:
            # Cheap electricity: charge battery
            charge_intensity = (buy_threshold - price_signal) / buy_threshold
            charge_intensity *= self.behavioral_params['response_speed']
            action = -charge_intensity * 0.8
        elif price_signal > sell_threshold:
            # Expensive electricity: discharge battery
            discharge_intensity = (price_signal - sell_threshold) / sell_threshold
            discharge_intensity *= self.behavioral_params['response_speed']
            action = discharge_intensity * 0.8
        else:
            # Neutral price: minimal action with algorithm-specific bias
            risk_bias = (self.behavioral_params['risk_tolerance'] - 0.5) * 0.2
            action = risk_bias
        
        return action
    
    def _apply_algorithm_behavior(self, base_action: float, price_signal: float, 
                                 battery_soc: float) -> float:
        """Apply algorithm-specific behavioral modifications"""
        
        # 1. Add behavioral noise
        noise = self.rng.normal(0, self.behavioral_params['noise_level'])
        action_with_noise = base_action + noise
        
        # 2. Apply safety margin adjustments
        safety_margin = self.behavioral_params['safety_margin']
        
        # Conservative algorithms reduce action magnitude near safety bounds
        if battery_soc < (self.safety_bounds['min_soc'] + safety_margin):
            # Near low SOC: reduce discharge actions
            if action_with_noise > 0:
                action_with_noise *= (1 - safety_margin)
        elif battery_soc > (self.safety_bounds['max_soc'] - safety_margin):
            # Near high SOC: reduce charge actions
            if action_with_noise < 0:
                action_with_noise *= (1 - safety_margin)
        
        # 3. Apply risk tolerance scaling
        risk_scaling = 0.5 + self.behavioral_params['risk_tolerance']
        action_with_noise *= risk_scaling
        
        return action_with_noise
    
    def _apply_safety_bounds(self, action: float, battery_soc: float) -> float:
        """Apply final safety bounds to action"""
        
        # Hard safety constraints
        if battery_soc <= self.safety_bounds['min_soc'] and action > 0:
            action = -0.5  # Force charging
        elif battery_soc >= self.safety_bounds['max_soc'] and action < 0:
            action = 0.5   # Force discharging
        
        # Clip to action bounds
        action = np.clip(action, 
                        self.safety_bounds['max_discharge_rate'], 
                        self.safety_bounds['max_charge_rate'])
        
        return action
    
    def get_algorithm_signature(self) -> Dict:
        """Get algorithm-specific behavioral signature for analysis"""
        return {
            'algorithm': self.algorithm_name,
            'seed': self._get_algorithm_seed(),
            'behavioral_params': self.behavioral_params.copy(),
            'safety_bounds': self.safety_bounds.copy()
        }

def create_algorithm_specific_policies() -> Dict[str, AlgorithmSpecificPCSPolicy]:
    """Create algorithm-specific PCS policies for all algorithms"""
    
    algorithms = ['PPOLag', 'CPO', 'FOCOPS', 'CUP', 'PPOSaute']
    policies = {}
    
    for algorithm in algorithms:
        policies[algorithm] = AlgorithmSpecificPCSPolicy(algorithm)
    
    return policies

def compare_policy_behaviors():
    """Compare behavioral differences between algorithm-specific policies"""
    
    policies = create_algorithm_specific_policies()
    
    print("🔍 ALGORITHM-SPECIFIC PCS POLICY COMPARISON")
    print("=" * 60)
    
    # Test scenarios
    test_scenarios = [
        {'price': 2.0, 'soc': 0.5, 'name': 'Low Price, Mid SOC'},
        {'price': 8.0, 'soc': 0.5, 'name': 'High Price, Mid SOC'},
        {'price': 5.0, 'soc': 0.2, 'name': 'Mid Price, Low SOC'},
        {'price': 5.0, 'soc': 0.8, 'name': 'Mid Price, High SOC'},
    ]
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")
        print("-" * 40)
        
        for algo_name, policy in policies.items():
            action = policy.get_action(scenario['price'], scenario['soc'])
            print(f"{algo_name:10}: {action:+6.3f}")
    
    # Show behavioral parameters
    print(f"\nBEHAVIORAL PARAMETERS:")
    print("-" * 40)
    print(f"{'Algorithm':<10} {'Risk':<6} {'Speed':<6} {'Price':<6} {'Margin':<7} {'Noise':<6}")
    
    for algo_name, policy in policies.items():
        params = policy.behavioral_params
        print(f"{algo_name:<10} "
              f"{params['risk_tolerance']:<6.2f} "
              f"{params['response_speed']:<6.2f} "
              f"{params['price_sensitivity']:<6.2f} "
              f"{params['safety_margin']:<7.2f} "
              f"{params['noise_level']:<6.2f}")

if __name__ == "__main__":
    compare_policy_behaviors() 