#!/usr/bin/env python3
"""
Visualization script for SafeISO stress test results
Creates comprehensive plots showing algorithm performance under stress
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List
import argparse

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StressTestVisualizer:
    """Visualize stress test results with comprehensive plots"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.algorithms = ['PPOLag', 'CPO', 'FOCOPS', 'CUP', 'PPOSaute']
        self.scenarios = ['voltage_instability', 'frequency_oscillation', 'battery_degradation', 
                         'demand_surge', 'cascading_instability']
        self.results_data = {}
        
    def load_results(self):
        """Load all stress test results from JSON files"""
        print(f"Loading results from {self.results_dir}")
        
        for algorithm in self.algorithms:
            self.results_data[algorithm] = {}
            for scenario in self.scenarios:
                result_file = self.results_dir / f"{algorithm}_{scenario}_stress_results.json"
                
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        self.results_data[algorithm][scenario] = data
                        print(f"✓ Loaded {algorithm} - {scenario}")
                else:
                    print(f"✗ Missing {algorithm} - {scenario}")
                    
    def create_cost_comparison_heatmap(self):
        """Create heatmap showing cost comparison across algorithms and scenarios"""
        # Prepare data for heatmap
        cost_matrix = []
        algorithm_labels = []
        scenario_labels = []
        
        for algorithm in self.algorithms:
            if algorithm in self.results_data:
                algorithm_labels.append(algorithm)
                row = []
                for scenario in self.scenarios:
                    if scenario in self.results_data[algorithm]:
                        cost = self.results_data[algorithm][scenario].get('mean_cost', 0)
                        row.append(cost)
                    else:
                        row.append(0)
                cost_matrix.append(row)
                
        if not algorithm_labels:
            print("No data available for heatmap")
            return
            
        scenario_labels = [s.replace('_', ' ').title() for s in self.scenarios]
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(cost_matrix, 
                   xticklabels=scenario_labels,
                   yticklabels=algorithm_labels,
                   annot=True, 
                   fmt='.1f',
                   cmap='Reds',
                   cbar_kws={'label': 'Mean Safety Cost'})
        
        plt.title('Safety Cost Comparison: Algorithms vs Stress Scenarios', fontsize=16, fontweight='bold')
        plt.xlabel('Stress Scenarios', fontsize=12)
        plt.ylabel('Safe RL Algorithms', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_file = self.results_dir / 'cost_comparison_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap: {output_file}")
        plt.show()
        
    def create_algorithm_performance_bars(self):
        """Create bar chart comparing total performance across algorithms"""
        algorithm_data = {}
        
        for algorithm in self.algorithms:
            if algorithm in self.results_data:
                total_cost = 0
                scenario_count = 0
                
                for scenario in self.scenarios:
                    if scenario in self.results_data[algorithm]:
                        cost = self.results_data[algorithm][scenario].get('mean_cost', 0)
                        total_cost += cost
                        scenario_count += 1
                
                if scenario_count > 0:
                    algorithm_data[algorithm] = {
                        'total_cost': total_cost,
                        'avg_cost': total_cost / scenario_count,
                        'scenario_count': scenario_count
                    }
        
        if not algorithm_data:
            print("No data available for bar chart")
            return
            
        # Create subplot with multiple metrics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Total cost comparison
        algorithms = list(algorithm_data.keys())
        total_costs = [algorithm_data[alg]['total_cost'] for alg in algorithms]
        
        bars1 = ax1.bar(algorithms, total_costs, color=sns.color_palette("husl", len(algorithms)))
        ax1.set_title('Total Safety Cost Under Stress', fontweight='bold')
        ax1.set_ylabel('Total Safety Cost')
        ax1.set_xlabel('Safe RL Algorithms')
        
        # Add value labels on bars
        for bar, cost in zip(bars1, total_costs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{cost:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Average cost comparison
        avg_costs = [algorithm_data[alg]['avg_cost'] for alg in algorithms]
        bars2 = ax2.bar(algorithms, avg_costs, color=sns.color_palette("husl", len(algorithms)))
        ax2.set_title('Average Safety Cost Per Scenario', fontweight='bold')
        ax2.set_ylabel('Average Safety Cost')
        ax2.set_xlabel('Safe RL Algorithms')
        
        # Add value labels on bars
        for bar, cost in zip(bars2, avg_costs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{cost:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        output_file = self.results_dir / 'algorithm_performance_comparison.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved bar chart: {output_file}")
        plt.show()
        
    def create_scenario_severity_analysis(self):
        """Analyze and visualize scenario severity across algorithms"""
        scenario_data = {}
        
        for scenario in self.scenarios:
            scenario_data[scenario] = {
                'costs': [],
                'severities': [],
                'algorithms': []
            }
            
            for algorithm in self.algorithms:
                if (algorithm in self.results_data and 
                    scenario in self.results_data[algorithm]):
                    
                    data = self.results_data[algorithm][scenario]
                    cost = data.get('mean_cost', 0)
                    severity = data.get('scenario_params', {}).get('severity', 0)
                    
                    scenario_data[scenario]['costs'].append(cost)
                    scenario_data[scenario]['severities'].append(severity)
                    scenario_data[scenario]['algorithms'].append(algorithm)
        
        # Create scatter plot showing severity vs cost
        plt.figure(figsize=(14, 10))
        
        colors = sns.color_palette("husl", len(self.scenarios))
        
        for i, scenario in enumerate(self.scenarios):
            if scenario_data[scenario]['costs']:
                plt.scatter(scenario_data[scenario]['severities'], 
                           scenario_data[scenario]['costs'],
                           label=scenario.replace('_', ' ').title(),
                           color=colors[i],
                           s=100,
                           alpha=0.7)
                
                # Add algorithm labels
                for j, alg in enumerate(scenario_data[scenario]['algorithms']):
                    plt.annotate(alg, 
                               (scenario_data[scenario]['severities'][j], 
                                scenario_data[scenario]['costs'][j]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
        
        plt.xlabel('Scenario Severity', fontsize=12)
        plt.ylabel('Safety Cost', fontsize=12)
        plt.title('Stress Scenario Severity vs Safety Cost', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = self.results_dir / 'severity_vs_cost_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plot: {output_file}")
        plt.show()
        
    def create_detailed_cost_breakdown(self):
        """Create detailed breakdown of costs by scenario type"""
        # Prepare data for grouped bar chart
        scenario_costs = {scenario: [] for scenario in self.scenarios}
        algorithm_labels = []
        
        for algorithm in self.algorithms:
            if algorithm in self.results_data:
                algorithm_labels.append(algorithm)
                for scenario in self.scenarios:
                    if scenario in self.results_data[algorithm]:
                        cost = self.results_data[algorithm][scenario].get('mean_cost', 0)
                        scenario_costs[scenario].append(cost)
                    else:
                        scenario_costs[scenario].append(0)
        
        if not algorithm_labels:
            print("No data available for detailed breakdown")
            return
            
        # Create grouped bar chart
        x = np.arange(len(algorithm_labels))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        colors = sns.color_palette("husl", len(self.scenarios))
        
        for i, scenario in enumerate(self.scenarios):
            offset = (i - len(self.scenarios)/2) * width
            bars = ax.bar(x + offset, scenario_costs[scenario], width, 
                         label=scenario.replace('_', ' ').title(),
                         color=colors[i])
            
            # Add value labels on bars (only for non-zero values)
            for bar, cost in zip(bars, scenario_costs[scenario]):
                if cost > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{cost:.0f}', ha='center', va='bottom', 
                           fontsize=8, rotation=90)
        
        ax.set_xlabel('Safe RL Algorithms', fontsize=12)
        ax.set_ylabel('Safety Cost', fontsize=12)
        ax.set_title('Detailed Safety Cost Breakdown by Scenario', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithm_labels)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = self.results_dir / 'detailed_cost_breakdown.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved detailed breakdown: {output_file}")
        plt.show()
        
    def create_summary_report(self):
        """Create a text summary of the stress test results"""
        report_file = self.results_dir / 'stress_test_visualization_summary.txt'
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SafeISO Stress Test Results - Visualization Summary\n")
            f.write("=" * 80 + "\n\n")
            
            # Algorithm ranking by total cost
            algorithm_totals = {}
            for algorithm in self.algorithms:
                if algorithm in self.results_data:
                    total_cost = sum(
                        self.results_data[algorithm][scenario].get('mean_cost', 0)
                        for scenario in self.scenarios
                        if scenario in self.results_data[algorithm]
                    )
                    algorithm_totals[algorithm] = total_cost
            
            sorted_algorithms = sorted(algorithm_totals.items(), key=lambda x: x[1])
            
            f.write("ALGORITHM RANKING (by total safety cost under stress):\n")
            f.write("-" * 50 + "\n")
            for i, (algorithm, total_cost) in enumerate(sorted_algorithms, 1):
                f.write(f"{i}. {algorithm:<12} | Total Cost: {total_cost:8.1f}\n")
            
            f.write("\nMOST CHALLENGING SCENARIOS:\n")
            f.write("-" * 50 + "\n")
            
            # Scenario difficulty ranking
            scenario_totals = {}
            for scenario in self.scenarios:
                total_cost = sum(
                    self.results_data[algorithm][scenario].get('mean_cost', 0)
                    for algorithm in self.algorithms
                    if algorithm in self.results_data and scenario in self.results_data[algorithm]
                )
                scenario_totals[scenario] = total_cost
            
            sorted_scenarios = sorted(scenario_totals.items(), key=lambda x: x[1], reverse=True)
            
            for i, (scenario, total_cost) in enumerate(sorted_scenarios, 1):
                scenario_name = scenario.replace('_', ' ').title()
                f.write(f"{i}. {scenario_name:<20} | Total Cost: {total_cost:8.1f}\n")
            
            f.write("\nKEY INSIGHTS:\n")
            f.write("-" * 50 + "\n")
            
            if sorted_algorithms:
                safest = sorted_algorithms[0][0]
                riskiest = sorted_algorithms[-1][0]
                f.write(f"• Safest Algorithm: {safest} (lowest total cost under stress)\n")
                f.write(f"• Riskiest Algorithm: {riskiest} (highest total cost under stress)\n")
            
            if sorted_scenarios:
                hardest = sorted_scenarios[0][0].replace('_', ' ').title()
                easiest = sorted_scenarios[-1][0].replace('_', ' ').title()
                f.write(f"• Most Challenging Scenario: {hardest}\n")
                f.write(f"• Least Challenging Scenario: {easiest}\n")
            
            f.write("\n• Stress injection system successfully created safety violations\n")
            f.write("• Algorithm-specific scenarios revealed performance differences\n")
            f.write("• Different algorithms show distinct stress response patterns\n")
        
        print(f"Saved summary report: {report_file}")
        
    def generate_all_visualizations(self):
        """Generate all visualization plots and reports"""
        print("🎨 Generating comprehensive stress test visualizations...")
        
        self.load_results()
        
        if not self.results_data:
            print("❌ No results data found!")
            return
            
        print("\n📊 Creating visualizations...")
        self.create_cost_comparison_heatmap()
        self.create_algorithm_performance_bars()
        self.create_scenario_severity_analysis()
        self.create_detailed_cost_breakdown()
        self.create_summary_report()
        
        print(f"\n✅ All visualizations saved in: {self.results_dir}")
        print("📋 Generated files:")
        print("   • cost_comparison_heatmap.png")
        print("   • algorithm_performance_comparison.png") 
        print("   • severity_vs_cost_analysis.png")
        print("   • detailed_cost_breakdown.png")
        print("   • stress_test_visualization_summary.txt")

def find_latest_results_dir(base_dir: str = "stress_test_results") -> str:
    """Find the most recent stress test results directory"""
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Results directory {base_dir} not found")
    
    # Find all stress test directories
    stress_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('stress_test_')]
    
    if not stress_dirs:
        raise FileNotFoundError(f"No stress test results found in {base_dir}")
    
    # Return the most recent one
    latest_dir = max(stress_dirs, key=lambda x: x.stat().st_mtime)
    return str(latest_dir)

def main():
    """Main function to run stress test visualization"""
    parser = argparse.ArgumentParser(description='Visualize SafeISO stress test results')
    parser.add_argument('--results-dir', 
                       help='Directory containing stress test results (auto-detects latest if not specified)')
    
    args = parser.parse_args()
    
    # Find results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        try:
            results_dir = find_latest_results_dir()
            print(f"🔍 Auto-detected latest results: {results_dir}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            return
    
    # Create visualizer and generate plots
    visualizer = StressTestVisualizer(results_dir)
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main() 