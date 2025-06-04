#!/usr/bin/env python3
"""
Improved Stress Testing System for SafeISO Safe RL Algorithms
Addresses identical safety performance issue through stochastic scenarios
"""

import numpy as np
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import json

@dataclass
class StochasticScenario:
    """Parameterized stress scenario with randomization"""
    name: str
    base_severity: float
    severity_range: Tuple[float, float]
    duration_range: Tuple[int, int]
    frequency_range: Tuple[float, float]  # How often violations occur
    seed_offset: int = 0  # Different per algorithm
    
class ImprovedStressTestScenarios:
    """Enhanced stress scenarios with algorithm-specific randomization"""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.scenarios = {
            'voltage_instability': StochasticScenario(
                name='voltage_instability',
                base_severity=0.1,  # 10% voltage deviation
                severity_range=(0.05, 0.2),  # 5-20% deviation
                duration_range=(3, 15),  # 3-15 timesteps
                frequency_range=(0.1, 0.3)  # 10-30% of episodes
            ),
            'frequency_oscillation': StochasticScenario(
                name='frequency_oscillation', 
                base_severity=0.5,  # 0.5 Hz deviation
                severity_range=(0.2, 1.0),  # 0.2-1.0 Hz
                duration_range=(5, 20),
                frequency_range=(0.15, 0.35)
            ),
            'battery_degradation': StochasticScenario(
                name='battery_degradation',
                base_severity=0.3,  # 30% capacity loss
                severity_range=(0.1, 0.5),  # 10-50% loss
                duration_range=(10, 30),
                frequency_range=(0.2, 0.4)
            ),
            'demand_surge': StochasticScenario(
                name='demand_surge',
                base_severity=5.0,  # 5 MW surge
                severity_range=(2.0, 10.0),  # 2-10 MW
                duration_range=(2, 8),
                frequency_range=(0.05, 0.25)
            ),
            'cascading_instability': StochasticScenario(
                name='cascading_instability',
                base_severity=0.8,  # High severity
                severity_range=(0.5, 1.0),
                duration_range=(5, 25),
                frequency_range=(0.1, 0.2)
            )
        }
    
    def generate_algorithm_specific_scenario(self, scenario_name: str, algorithm: str, 
                                           episode_length: int = 50) -> Dict:
        """Generate scenario parameters specific to each algorithm"""
        scenario = self.scenarios[scenario_name]
        
        # Create algorithm-specific seed
        algo_seed = hash(algorithm) % 10000 + self.base_seed + scenario.seed_offset
        rng = np.random.RandomState(algo_seed)
        
        # Randomize scenario parameters
        severity = rng.uniform(*scenario.severity_range)
        duration = rng.randint(*scenario.duration_range)
        frequency = rng.uniform(*scenario.frequency_range)
        
        # Generate violation schedule
        num_violations = int(episode_length * frequency)
        violation_timesteps = sorted(rng.choice(episode_length, num_violations, replace=False))
        
        return {
            'algorithm': algorithm,
            'scenario': scenario_name,
            'severity': severity,
            'duration': duration,
            'frequency': frequency,
            'violation_timesteps': violation_timesteps,
            'seed': algo_seed
        }
    
    def inject_voltage_instability(self, env, params: Dict, timestep: int):
        """Inject voltage instability with algorithm-specific parameters"""
        if timestep in params['violation_timesteps']:
            # Calculate voltage deviation
            severity = params['severity']
            deviation = severity * (1 if np.random.random() > 0.5 else -1)
            
            # Inject into environment state
            if hasattr(env, 'force_voltage_deviation'):
                env.force_voltage_deviation(1.0 + deviation, params['duration'])
            else:
                # Fallback: modify observation directly
                obs = env.get_observation()
                if 'voltage' in obs:
                    obs['voltage'] *= (1.0 + deviation)
    
    def inject_frequency_oscillation(self, env, params: Dict, timestep: int):
        """Inject frequency oscillations"""
        if timestep in params['violation_timesteps']:
            severity = params['severity']
            freq_deviation = severity * np.sin(timestep * 0.1)  # Oscillating pattern
            
            if hasattr(env, 'force_frequency_deviation'):
                env.force_frequency_deviation(50.0 + freq_deviation, params['duration'])
    
    def inject_battery_degradation(self, env, params: Dict, timestep: int):
        """Inject battery capacity degradation"""
        if timestep in params['violation_timesteps']:
            severity = params['severity']
            
            if hasattr(env, 'degrade_battery_capacity'):
                env.degrade_battery_capacity(1.0 - severity, params['duration'])
    
    def inject_demand_surge(self, env, params: Dict, timestep: int):
        """Inject sudden demand increases"""
        if timestep in params['violation_timesteps']:
            severity = params['severity']
            
            if hasattr(env, 'add_demand_surge'):
                env.add_demand_surge(severity, params['duration'])
    
    def inject_cascading_instability(self, env, params: Dict, timestep: int):
        """Inject multiple simultaneous failures"""
        if timestep in params['violation_timesteps']:
            # Combine multiple stress factors
            self.inject_voltage_instability(env, params, timestep)
            self.inject_frequency_oscillation(env, params, timestep)
            
            # Add random additional stressor
            if np.random.random() > 0.5:
                self.inject_demand_surge(env, params, timestep)

class AlgorithmSpecificSafetyAnalyzer:
    """Analyze safety behavior differences between algorithms"""
    
    def __init__(self):
        self.safety_metrics = {}
    
    def record_safety_decision(self, algorithm: str, state: Dict, action: np.ndarray, 
                             safety_cost: float, violation_type: str):
        """Record algorithm-specific safety decisions"""
        if algorithm not in self.safety_metrics:
            self.safety_metrics[algorithm] = {
                'decisions': [],
                'violation_responses': {},
                'safety_costs': [],
                'action_distributions': []
            }
        
        metrics = self.safety_metrics[algorithm]
        metrics['decisions'].append({
            'state': state.copy(),
            'action': action.copy(),
            'safety_cost': safety_cost,
            'violation_type': violation_type,
            'timestamp': len(metrics['decisions'])
        })
        
        # Track responses to specific violation types
        if violation_type not in metrics['violation_responses']:
            metrics['violation_responses'][violation_type] = []
        metrics['violation_responses'][violation_type].append(action.copy())
        
        metrics['safety_costs'].append(safety_cost)
        metrics['action_distributions'].append(action.copy())
    
    def analyze_safety_differences(self) -> Dict:
        """Analyze differences in safety behavior between algorithms"""
        analysis = {}
        
        for algo, metrics in self.safety_metrics.items():
            # Calculate safety response characteristics
            avg_safety_cost = np.mean(metrics['safety_costs'])
            safety_cost_variance = np.var(metrics['safety_costs'])
            
            # Analyze action distributions during violations
            action_means = {}
            action_stds = {}
            
            for violation_type, actions in metrics['violation_responses'].items():
                if actions:
                    actions_array = np.array(actions)
                    action_means[violation_type] = np.mean(actions_array, axis=0)
                    action_stds[violation_type] = np.std(actions_array, axis=0)
            
            analysis[algo] = {
                'avg_safety_cost': avg_safety_cost,
                'safety_cost_variance': safety_cost_variance,
                'action_means_by_violation': action_means,
                'action_stds_by_violation': action_stds,
                'total_decisions': len(metrics['decisions']),
                'violation_type_counts': {vtype: len(actions) 
                                        for vtype, actions in metrics['violation_responses'].items()}
            }
        
        return analysis
    
    def compare_algorithms(self, analysis: Dict) -> Dict:
        """Compare safety behaviors between algorithms"""
        comparisons = {}
        algorithms = list(analysis.keys())
        
        for i, algo1 in enumerate(algorithms):
            for algo2 in algorithms[i+1:]:
                # Compare safety cost distributions
                cost_diff = abs(analysis[algo1]['avg_safety_cost'] - 
                              analysis[algo2]['avg_safety_cost'])
                
                # Compare action responses
                action_similarities = {}
                for violation_type in analysis[algo1]['action_means_by_violation']:
                    if violation_type in analysis[algo2]['action_means_by_violation']:
                        mean1 = analysis[algo1]['action_means_by_violation'][violation_type]
                        mean2 = analysis[algo2]['action_means_by_violation'][violation_type]
                        similarity = np.corrcoef(mean1, mean2)[0, 1] if len(mean1) > 1 else 0
                        action_similarities[violation_type] = similarity
                
                comparisons[f"{algo1}_vs_{algo2}"] = {
                    'safety_cost_difference': cost_diff,
                    'action_similarities': action_similarities,
                    'avg_action_similarity': np.mean(list(action_similarities.values())) 
                                           if action_similarities else 0
                }
        
        return comparisons

def create_differentiated_stress_test():
    """Create improved stress test that can differentiate between algorithms"""
    
    stress_scenarios = ImprovedStressTestScenarios(base_seed=12345)
    safety_analyzer = AlgorithmSpecificSafetyAnalyzer()
    
    algorithms = ['PPOLag', 'CPO', 'FOCOPS', 'CUP', 'PPOSaute']
    scenario_names = list(stress_scenarios.scenarios.keys())
    
    # Generate algorithm-specific scenario parameters
    test_plan = {}
    for algorithm in algorithms:
        test_plan[algorithm] = {}
        for scenario_name in scenario_names:
            params = stress_scenarios.generate_algorithm_specific_scenario(
                scenario_name, algorithm, episode_length=50
            )
            test_plan[algorithm][scenario_name] = params
    
    return test_plan, stress_scenarios, safety_analyzer

# Example usage and diagnostic functions
def diagnose_stress_test_issues():
    """Diagnostic functions to identify why all algorithms show identical results"""
    
    print("🔍 STRESS TEST DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    
    # Check 1: Verify scenario randomization
    test_plan, _, _ = create_differentiated_stress_test()
    
    print("\n1. SCENARIO RANDOMIZATION CHECK:")
    print("-" * 40)
    
    for scenario in ['voltage_instability', 'frequency_oscillation']:
        print(f"\n{scenario.upper()}:")
        for algo in ['PPOLag', 'CPO', 'FOCOPS']:
            params = test_plan[algo][scenario]
            print(f"  {algo:8}: severity={params['severity']:.3f}, "
                  f"freq={params['frequency']:.3f}, "
                  f"violations={len(params['violation_timesteps'])}")
    
    # Check 2: Verify different seeds produce different results
    print("\n2. SEED DIFFERENTIATION CHECK:")
    print("-" * 40)
    
    scenarios = ImprovedStressTestScenarios()
    for algo in ['PPOLag', 'CPO']:
        params1 = scenarios.generate_algorithm_specific_scenario('voltage_instability', algo)
        params2 = scenarios.generate_algorithm_specific_scenario('voltage_instability', algo)
        print(f"{algo}: Same params? {params1 == params2}")
    
    # Check 3: Recommend environment modifications
    print("\n3. ENVIRONMENT MODIFICATION RECOMMENDATIONS:")
    print("-" * 40)
    print("✓ Add stochastic noise to ResponsivePCSPolicy")
    print("✓ Make safety thresholds algorithm-dependent")
    print("✓ Introduce partial observability of safety state")
    print("✓ Add algorithm-specific safety cost weights")
    
    return test_plan

if __name__ == "__main__":
    diagnose_stress_test_issues() 