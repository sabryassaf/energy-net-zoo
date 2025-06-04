#!/usr/bin/env python3
"""
SafeISO Research Configuration File
All experiment configurations organized in one place
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import json
from pathlib import Path

@dataclass 
class PCSPolicyConfig:
    mode: str = "uniform"  # "uniform", "algorithm_specific", "custom"
    risk_tolerance: float = 0.4
    response_speed: float = 0.7
    price_sensitivity: float = 0.7
    safety_margin: float = 0.12
    noise_level: float = 0.05
    custom_params: Optional[Dict] = None

@dataclass
class ConstraintConfig:
    """Safety constraint parameters for the ISO environment"""
    # Constraint weights (how much each violation costs)
    voltage_cost_weight: float = 1.0
    frequency_cost_weight: float = 1.0
    battery_cost_weight: float = 1.0
    supply_demand_cost_weight: float = 2.0
    
    # Voltage limits (per unit)
    voltage_limit_min: float = 0.95  # 95% of nominal
    voltage_limit_max: float = 1.05  # 105% of nominal
    
    # Frequency limits (Hz)
    frequency_limit_min: float = 49.8  # Hz
    frequency_limit_max: float = 50.2  # Hz
    
    # Battery State of Charge limits
    battery_soc_min: float = 0.1     # 10%
    battery_soc_max: float = 0.9     # 90%
    
    # Supply-demand imbalance threshold (MW)
    supply_demand_imbalance_threshold: float = 10.0

@dataclass
class PricingConfig:
    """Pricing parameters for the environment"""
    pricing_policy: str = "ONLINE"  # "ONLINE", "OFFLINE", "CONSTANT"
    demand_pattern: str = "SINUSOIDAL"  # "SINUSOIDAL", "STEP", "RANDOM"
    cost_type: str = "CONSTANT"  # "CONSTANT", "VARIABLE"

@dataclass
class EnvironmentConfig:
    cost_threshold: float = 25.0
    max_episode_steps: int = 1000
    stress_testing: bool = True
    stress_scenarios: List[str] = None
    base_seed: int = 42
    scenario_types: List[str] = None
    output_dir: str = "research-results"
    
    # New constraint and pricing configurations
    constraint_config: ConstraintConfig = None
    pricing_config: PricingConfig = None
    
    def __post_init__(self):
        if self.stress_scenarios is None:
            self.stress_scenarios = [
                'voltage_instability',
                'frequency_oscillation', 
                'cascading_instability',
                'renewable_variability',
                'load_spike'
            ]
        
        if self.scenario_types is None:
            self.scenario_types = self.stress_scenarios
            
        # Initialize sub-configurations with defaults if not provided
        if self.constraint_config is None:
            self.constraint_config = ConstraintConfig()
            
        if self.pricing_config is None:
            self.pricing_config = PricingConfig()

@dataclass
class AlgorithmConfig:
    algorithms: List[str] = None
    
    def __post_init__(self):
        if self.algorithms is None:
            self.algorithms = ['PPOLag', 'CPO', 'FOCOPS', 'CUP', 'PPOSaute']

@dataclass
class ExperimentConfig:
    name: str
    description: str = ""
    num_episodes: int = 10
    num_seeds: int = 3
    output_dir: str = "research-results"
    
    # Sub-configurations
    pcs_config: PCSPolicyConfig = None
    env_config: EnvironmentConfig = None 
    algo_config: AlgorithmConfig = None
    
    def __post_init__(self):
        if self.pcs_config is None:
            self.pcs_config = PCSPolicyConfig()
        if self.env_config is None:
            self.env_config = EnvironmentConfig()
        if self.algo_config is None:
            self.algo_config = AlgorithmConfig()

# =============================================================================
# PREDEFINED EXPERIMENT CONFIGURATIONS
# =============================================================================

def get_baseline_config() -> ExperimentConfig:
    """Fair comparison with identical parameters"""
    return ExperimentConfig(
        name="SafeISO_Controlled_Baseline",
        description="Fair comparison with identical PCS parameters for all algorithms",
        num_episodes=20,
        num_seeds=5,
        pcs_config=PCSPolicyConfig(
            mode="uniform",
            risk_tolerance=0.4,
            response_speed=0.7,
            price_sensitivity=0.7,
            safety_margin=0.12,
            noise_level=0.05
        ),
        env_config=EnvironmentConfig(
            stress_testing=True,
            cost_threshold=25.0
        )
    )

def get_stress_test_config() -> ExperimentConfig:
    """Intensive stress testing"""
    return ExperimentConfig(
        name="SafeISO_Comprehensive_Stress_Test", 
        description="Intensive stress testing across all scenarios",
        num_episodes=25,
        num_seeds=5,
        pcs_config=PCSPolicyConfig(mode="uniform"),
        env_config=EnvironmentConfig(
            stress_testing=True,
            cost_threshold=20.0,  # Stricter for stress testing
        )
    )

def get_algorithm_specific_config() -> ExperimentConfig:
    """Algorithm-specific parameters (your original approach)"""
    
    custom_params = {
        'PPOLag': {
            'risk_tolerance': 0.3,
            'response_speed': 0.7, 
            'price_sensitivity': 0.8,
            'safety_margin': 0.15,
            'noise_level': 0.05
        },
        'CPO': {
            'risk_tolerance': 0.2,
            'response_speed': 0.9,
            'price_sensitivity': 0.6, 
            'safety_margin': 0.20,
            'noise_level': 0.03
        },
        'FOCOPS': {
            'risk_tolerance': 0.5,
            'response_speed': 0.8,
            'price_sensitivity': 0.9,
            'safety_margin': 0.10, 
            'noise_level': 0.07
        },
        'CUP': {
            'risk_tolerance': 0.4,
            'response_speed': 0.6,
            'price_sensitivity': 0.7,
            'safety_margin': 0.12,
            'noise_level': 0.06
        },
        'PPOSaute': {
            'risk_tolerance': 0.6,
            'response_speed': 0.5,
            'price_sensitivity': 0.5,
            'safety_margin': 0.08,
            'noise_level': 0.10
        }
    }
    
    return ExperimentConfig(
        name="SafeISO_Algorithm_Specific_Parameters",
        description="Test with algorithm-specific PCS parameters",
        num_episodes=15,
        num_seeds=5,
        pcs_config=PCSPolicyConfig(
            mode="custom",
            custom_params=custom_params
        ),
        env_config=EnvironmentConfig(stress_testing=True)
    )

def get_quick_test_config() -> ExperimentConfig:
    """Quick test for debugging"""
    return ExperimentConfig(
        name="SafeISO_Quick_Test",
        description="Fast configuration for testing and debugging", 
        num_episodes=5,
        num_seeds=2,
        pcs_config=PCSPolicyConfig(mode="uniform"),
        env_config=EnvironmentConfig(stress_testing=False),
        algo_config=AlgorithmConfig(algorithms=['PPOLag', 'CPO'])  # Just 2 algorithms
    )

def get_strict_safety_config() -> ExperimentConfig:
    """Configuration with stricter safety constraints"""
    return ExperimentConfig(
        name="SafeISO_Strict_Safety",
        description="Stricter safety constraints for conservative operation",
        num_episodes=20,
        num_seeds=5,
        pcs_config=PCSPolicyConfig(mode="uniform"),
        env_config=EnvironmentConfig(
            constraint_config=ConstraintConfig(
                voltage_cost_weight=3.0,      # Higher penalty
                frequency_cost_weight=3.0,    # Higher penalty
                battery_cost_weight=2.0,      # Higher penalty
                voltage_limit_min=0.97,       # Tighter limits
                voltage_limit_max=1.03,       # Tighter limits
                frequency_limit_min=49.9,     # Tighter limits
                frequency_limit_max=50.1,     # Tighter limits
            ),
            pricing_config=PricingConfig(
                pricing_policy="ONLINE",
                demand_pattern="SINUSOIDAL"
            )
        )
    )

def get_high_volatility_config() -> ExperimentConfig:
    """Configuration with high price volatility"""
    return ExperimentConfig(
        name="SafeISO_High_Volatility",
        description="High volatility pricing for market stress testing",
        num_episodes=25,
        num_seeds=8,
        pcs_config=PCSPolicyConfig(mode="algorithm_specific"),
        env_config=EnvironmentConfig(
            pricing_config=PricingConfig(
                pricing_policy="ONLINE",
                demand_pattern="RANDOM",
                cost_type="VARIABLE"
            )
        )
    )

# =============================================================================
# CONFIGURATION REGISTRY
# =============================================================================

AVAILABLE_CONFIGS = {
    'baseline': get_baseline_config,
    'stress': get_stress_test_config, 
    'algorithm_specific': get_algorithm_specific_config,
    'quick': get_quick_test_config,
    'strict_safety': get_strict_safety_config,
    'high_volatility': get_high_volatility_config
}

def get_config(config_name: str, config_dir: str = "saved_configs") -> ExperimentConfig:
    """Get a configuration by name (predefined or saved)"""
    
    # First, check predefined configurations
    if config_name in AVAILABLE_CONFIGS:
        return AVAILABLE_CONFIGS[config_name]()
    
    # Then, check saved configurations
    try:
        return load_config(config_name, config_dir)
    except FileNotFoundError:
        pass
    
    # Configuration not found
    available_predefined = list(AVAILABLE_CONFIGS.keys())
    
    # List saved configs
    config_path = Path(config_dir)
    if config_path.exists():
        saved_configs = [f.stem for f in config_path.glob("*.json")]
    else:
        saved_configs = []
    
    raise ValueError(
        f"Unknown config '{config_name}'. "
        f"Available predefined: {available_predefined}. "
        f"Available saved: {saved_configs}"
    )

def list_available_configs():
    """Print all predefined configurations"""
    print("Available Experiment Configurations:")
    print("="*50)
    
    for name, func in AVAILABLE_CONFIGS.items():
        config = func()
        print(f"{name:20} - {config.description}")
        print(f"   Episodes: {config.num_episodes}, Seeds: {config.num_seeds}")
        print(f"   PCS Mode: {config.pcs_config.mode}")
        print(f"   Algorithms: {len(config.algo_config.algorithms)}")
        print()

def list_all_configs():
    """List both predefined and saved configurations"""
    print("ALL AVAILABLE CONFIGURATIONS")
    print("="*60)
    
    print("\nPREDEFINED CONFIGURATIONS:")
    list_available_configs()
    
    print("\nSAVED CONFIGURATIONS:")
    list_saved_configs()

# =============================================================================
# CONFIGURATION SAVE/LOAD FUNCTIONALITY
# =============================================================================

def save_config(config: ExperimentConfig, config_name: str, config_dir: str = "saved_configs"):
    """Save a configuration for later reuse"""
    
    # Create saved configs directory
    config_path = Path(config_dir)
    config_path.mkdir(exist_ok=True)
    
    # Save configuration as JSON
    config_file = config_path / f"{config_name}.json"
    
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2, default=str)
    
    print(f"Configuration saved: {config_file}")
    return str(config_file)

def load_config(config_name: str, config_dir: str = "saved_configs") -> ExperimentConfig:
    """Load a previously saved configuration"""
    
    config_path = Path(config_dir)
    config_file = config_path / f"{config_name}.json"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration '{config_name}' not found at {config_file}")
    
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    # Reconstruct the configuration objects
    pcs_config = PCSPolicyConfig(**config_data['pcs_config'])
    env_config = EnvironmentConfig(**config_data['env_config'])
    algo_config = AlgorithmConfig(**config_data['algo_config'])
    
    # Create main config
    config = ExperimentConfig(
        name=config_data['name'],
        description=config_data.get('description', ''),
        num_episodes=config_data.get('num_episodes', 10),
        num_seeds=config_data.get('num_seeds', 3),
        output_dir=config_data.get('output_dir', 'research-results'),
        pcs_config=pcs_config,
        env_config=env_config,
        algo_config=algo_config
    )
    
    print(f"Configuration loaded: {config.name}")
    return config

def list_saved_configs(config_dir: str = "saved_configs"):
    """List all saved configurations"""
    config_path = Path(config_dir)
    
    if not config_path.exists():
        print("No saved configurations found.")
        return
    
    config_files = list(config_path.glob("*.json"))
    
    if not config_files:
        print("No saved configurations found.")
        return
    
    print("Saved Configurations:")
    print("-" * 30)
    
    for config_file in config_files:
        name = config_file.stem
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            description = config_data.get('description', 'No description')
            episodes = config_data.get('num_episodes', 'Unknown')
            seeds = config_data.get('num_seeds', 'Unknown')
            
            print(f"{name:20} - {description}")
            print(f"   Episodes: {episodes}, Seeds: {seeds}")
            print(f"   File: {config_file}")
            print()
            
        except Exception as e:
            print(f"Error reading {config_file}: {e}")

def create_custom_config() -> ExperimentConfig:
    """Interactive configuration creator"""
    print("Creating Custom Configuration")
    print("="*40)
    
    name = input("Experiment name: ")
    description = input("Description (optional): ")
    num_episodes = int(input("Number of episodes (default 10): ") or "10")
    num_seeds = int(input("Number of seeds (default 3): ") or "3")
    
    # PCS Policy Configuration
    print("\nPCS Policy Configuration:")
    mode = input("Mode [uniform/algorithm_specific/custom] (default uniform): ") or "uniform"
    risk_tolerance = float(input("Risk tolerance (default 0.4): ") or "0.4")
    response_speed = float(input("Response speed (default 0.7): ") or "0.7")
    
    pcs_config = PCSPolicyConfig(
        mode=mode,
        risk_tolerance=risk_tolerance,
        response_speed=response_speed
    )
    
    # Environment Configuration  
    print("\nEnvironment Configuration:")
    stress_testing = input("Enable stress testing? [y/n] (default y): ").lower() != 'n'
    cost_threshold = float(input("Cost threshold (default 25.0): ") or "25.0")
    
    env_config = EnvironmentConfig(
        stress_testing=stress_testing,
        cost_threshold=cost_threshold
    )
    
    # Algorithm Configuration
    print(f"\nAlgorithm Configuration:")
    print("Available algorithms: PPOLag, CPO, FOCOPS, CUP, PPOSaute")
    algo_input = input("Algorithms (comma-separated, default all): ")
    
    if algo_input.strip():
        algorithms = [alg.strip() for alg in algo_input.split(',')]
        algo_config = AlgorithmConfig(algorithms=algorithms)
    else:
        algo_config = AlgorithmConfig()  # Uses defaults
    
    # Create configuration
    config = ExperimentConfig(
        name=name,
        description=description,
        num_episodes=num_episodes,
        num_seeds=num_seeds,
        pcs_config=pcs_config,
        env_config=env_config,
        algo_config=algo_config
    )
    
    return config

# Update main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Configuration Management")
    parser.add_argument("--list", action="store_true", help="List all configurations")
    parser.add_argument("--save", type=str, help="Save current config with given name")
    parser.add_argument("--load", type=str, help="Load saved configuration")
    parser.add_argument("--create", action="store_true", help="Create custom configuration")
    
    args = parser.parse_args()
    
    if args.list:
        list_all_configs()
    elif args.create:
        config = create_custom_config()
        save_name = input("\nSave this configuration as (optional): ")
        if save_name:
            save_config(config, save_name)
    elif args.load:
        config = load_config(args.load)
        print(f"Loaded: {config.name}")
    elif args.save:
        # Example of saving a predefined config
        config = get_baseline_config()
        save_config(config, args.save)
    else:
        list_all_configs()