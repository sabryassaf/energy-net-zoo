#!/usr/bin/env python
import numpy as np


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
            obs = observation[0] 
        else:
            obs = observation
        
        battery_level, time, buy_price, sell_price = obs
        
        battery_soc = battery_level / 100.0
        
        action = 0.0 
        max_safe_action = 0.8
        
        # Charge when buy price is low and battery not full
        if buy_price < self.charge_threshold and battery_soc < 0.9:
            charge_intensity = (self.charge_threshold - buy_price) / self.charge_threshold
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


def create_responsive_pcs_policy(charge_threshold=3.0, discharge_threshold=7.0, 
                               max_charge_rate=1.0, max_discharge_rate=1.0):
    """
    Factory function to create a responsive PCS policy with custom parameters.
    
    Args:
        charge_threshold: Buy price threshold below which to charge
        discharge_threshold: Sell price threshold above which to discharge
        max_charge_rate: Maximum battery charging rate
        max_discharge_rate: Maximum battery discharging rate
        
    Returns:
        ResponsivePCSPolicy: Configured policy instance
    """
    return ResponsivePCSPolicy(
        charge_threshold=charge_threshold,
        discharge_threshold=discharge_threshold,
        max_charge_rate=max_charge_rate,
        max_discharge_rate=max_discharge_rate
    )


def test_responsive_pcs_policy():
    """Test function to verify the responsive PCS policy works correctly."""
    print("=== Testing Responsive PCS Policy ===")
    
    # Create policy
    policy = ResponsivePCSPolicy()
    
    # Test scenarios
    test_scenarios = [
        # [battery_level, time, buy_price, sell_price] -> expected behavior
        ([50.0, 12.0, 2.0, 5.0], "Should CHARGE (low buy price)"),
        ([50.0, 12.0, 5.0, 8.0], "Should DISCHARGE (high sell price)"),
        ([50.0, 12.0, 4.0, 6.0], "Should DO NOTHING (prices in middle range)"),
        ([90.0, 12.0, 1.0, 5.0], "Should NOT CHARGE (battery almost full)"),
        ([10.0, 12.0, 5.0, 9.0], "Should NOT DISCHARGE (battery almost empty)"),
    ]
    
    print(f"Policy settings: charge_threshold={policy.charge_threshold}, "
          f"discharge_threshold={policy.discharge_threshold}")
    print()
    
    for observation, expected in test_scenarios:
        action, _ = policy.predict(np.array(observation))
        action_value = action[0]
        
        if action_value > 0.1:
            behavior = "CHARGE"
        elif action_value < -0.1:
            behavior = "DISCHARGE"
        else:
            behavior = "NO ACTION"
        
        print(f"Observation: {observation}")
        print(f"  Expected: {expected}")
        print(f"  Action: {action_value:.4f} -> {behavior}")
        print()
    
    print("Responsive PCS Policy test completed")


if __name__ == "__main__":
    test_responsive_pcs_policy() 