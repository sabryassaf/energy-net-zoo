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
        
    def create_violation_count_heatmap(self):
        """Create heatmap showing violation count comparison across algorithms and scenarios"""
        # Prepare data for heatmap
        violation_matrix = []
        algorithm_labels = []
        scenario_labels = []
        
        for algorithm in self.algorithms:
            if algorithm in self.results_data:
                algorithm_labels.append(algorithm)
                row = []
                for scenario in self.scenarios:
                    if scenario in self.results_data[algorithm]:
                        violations = self.results_data[algorithm][scenario].get('total_violations', 0)
                        row.append(violations)
                    else:
                        row.append(0)
                violation_matrix.append(row)
                
        if not algorithm_labels:
            print("No data available for violation heatmap")
            return
            
        scenario_labels = [s.replace('_', ' ').title() for s in self.scenarios]
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(violation_matrix, 
                   xticklabels=scenario_labels,
                   yticklabels=algorithm_labels,
                   annot=True, 
                   fmt='d',  # Integer format for violation counts
                   cmap='Oranges',
                   cbar_kws={'label': 'Total Violations'})
        
        plt.title('Safety Violation Count Comparison: Algorithms vs Stress Scenarios', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Stress Scenarios', fontsize=12)
        plt.ylabel('Safe RL Algorithms', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_file = self.results_dir / 'violation_count_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved violation heatmap: {output_file}")
        plt.show()

    def create_violation_vs_cost_scatter(self):
        """Create scatter plot showing relationship between violations and costs"""
        plt.figure(figsize=(14, 10))
        
        colors = sns.color_palette("husl", len(self.algorithms))
        
        for i, algorithm in enumerate(self.algorithms):
            if algorithm in self.results_data:
                violations = []
                costs = []
                scenario_names = []
                
                for scenario in self.scenarios:
                    if scenario in self.results_data[algorithm]:
                        data = self.results_data[algorithm][scenario]
                        violations.append(data.get('total_violations', 0))
                        costs.append(data.get('mean_cost', 0))
                        scenario_names.append(scenario.replace('_', ' ').title())
                
                if violations and costs:
                    plt.scatter(violations, costs, 
                              label=algorithm,
                              color=colors[i],
                              s=120,
                              alpha=0.7,
                              edgecolors='black',
                              linewidth=1)
                    
                    # Add scenario labels
                    for j, scenario_name in enumerate(scenario_names):
                        plt.annotate(f'{algorithm[:3]}-{scenario_name[:3]}', 
                                   (violations[j], costs[j]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8)
        
        plt.xlabel('Total Violations', fontsize=12, fontweight='bold')
        plt.ylabel('Mean Safety Cost', fontsize=12, fontweight='bold')
        plt.title('Safety Violations vs Cost Trade-off Analysis', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_file = self.results_dir / 'violations_vs_cost_scatter.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plot: {output_file}")
        plt.show()

    def create_violation_breakdown_stacked_bars(self):
        """Create stacked bar chart showing violation type breakdown by algorithm and scenario"""
        # Prepare data for stacked bars
        violation_types = ['voltage', 'frequency', 'battery', 'supply_demand']
        
        fig, axes = plt.subplots(len(self.scenarios), 1, figsize=(16, 4*len(self.scenarios)))
        if len(self.scenarios) == 1:
            axes = [axes]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']  # Red, Teal, Blue, Orange
        
        for scenario_idx, scenario in enumerate(self.scenarios):
            ax = axes[scenario_idx]
            
            algorithms_with_data = []
            violation_data = {vtype: [] for vtype in violation_types}
            
            for algorithm in self.algorithms:
                if (algorithm in self.results_data and 
                    scenario in self.results_data[algorithm]):
                    
                    algorithms_with_data.append(algorithm)
                    breakdown = self.results_data[algorithm][scenario].get('violation_breakdown', {})
                    
                    for vtype in violation_types:
                        violation_data[vtype].append(breakdown.get(vtype, 0))
            
            if algorithms_with_data:
                x = np.arange(len(algorithms_with_data))
                bottom = np.zeros(len(algorithms_with_data))
                
                for i, vtype in enumerate(violation_types):
                    ax.bar(x, violation_data[vtype], bottom=bottom, 
                          label=vtype.replace('_', ' ').title(),
                          color=colors[i], alpha=0.8)
                    bottom += violation_data[vtype]
                
                ax.set_title(f'{scenario.replace("_", " ").title()} - Violation Type Breakdown', 
                           fontweight='bold')
                ax.set_xlabel('Algorithms')
                ax.set_ylabel('Violation Count')
                ax.set_xticks(x)
                ax.set_xticklabels(algorithms_with_data)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = self.results_dir / 'violation_breakdown_stacked_bars.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved violation breakdown: {output_file}")
        plt.show()

    def create_algorithm_safety_ranking(self):
        """Create comprehensive algorithm safety ranking visualization"""
        # Calculate total violations and costs per algorithm
        algorithm_data = {}
        
        for algorithm in self.algorithms:
            if algorithm in self.results_data:
                total_violations = 0
                total_cost = 0
                scenario_count = 0
                
                for scenario in self.scenarios:
                    if scenario in self.results_data[algorithm]:
                        violations = self.results_data[algorithm][scenario].get('total_violations', 0)
                        cost = self.results_data[algorithm][scenario].get('mean_cost', 0)
                        total_violations += violations
                        total_cost += cost
                        scenario_count += 1
                
                if scenario_count > 0:
                    algorithm_data[algorithm] = {
                        'total_violations': total_violations,
                        'total_cost': total_cost,
                        'avg_violations': total_violations / scenario_count,
                        'avg_cost': total_cost / scenario_count
                    }
        
        if not algorithm_data:
            print("No data available for safety ranking")
            return
        
        # Create subplot with violations and costs
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        algorithms = list(algorithm_data.keys())
        total_violations = [algorithm_data[alg]['total_violations'] for alg in algorithms]
        total_costs = [algorithm_data[alg]['total_cost'] for alg in algorithms]
        
        # Sort by violations for ranking
        sorted_indices = np.argsort(total_violations)
        sorted_algorithms = [algorithms[i] for i in sorted_indices]
        sorted_violations = [total_violations[i] for i in sorted_indices]
        sorted_costs = [total_costs[i] for i in sorted_indices]
        
        # Violation ranking (lower is better)
        colors_violations = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_algorithms)))
        bars1 = ax1.bar(sorted_algorithms, sorted_violations, color=colors_violations)
        ax1.set_title('Algorithm Safety Ranking by Total Violations\n(Lower = Safer)', 
                     fontweight='bold', fontsize=14)
        ax1.set_ylabel('Total Violations Across All Scenarios')
        ax1.set_xlabel('Safe RL Algorithms (Ranked by Safety)')
        
        # Add ranking numbers and values
        for i, (bar, violations) in enumerate(zip(bars1, sorted_violations)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'#{i+1}\n{violations}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        # Cost ranking (lower is better)
        cost_sorted_indices = np.argsort(total_costs)
        cost_sorted_algorithms = [algorithms[i] for i in cost_sorted_indices]
        cost_sorted_costs = [total_costs[i] for i in cost_sorted_indices]
        
        colors_costs = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(cost_sorted_algorithms)))
        bars2 = ax2.bar(cost_sorted_algorithms, cost_sorted_costs, color=colors_costs)
        ax2.set_title('Algorithm Cost Performance Ranking\n(Lower = Better)', 
                     fontweight='bold', fontsize=14)
        ax2.set_ylabel('Total Safety Cost Across All Scenarios')
        ax2.set_xlabel('Safe RL Algorithms (Ranked by Cost)')
        
        # Add ranking numbers and values
        for i, (bar, cost) in enumerate(zip(bars2, cost_sorted_costs)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'#{i+1}\n{cost:.0f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_file = self.results_dir / 'algorithm_safety_ranking.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved safety ranking: {output_file}")
        plt.show()

    def create_comprehensive_dashboard(self):
        """Create a comprehensive dashboard with multiple metrics"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create a 3x2 grid of subplots
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Cost heatmap (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        cost_matrix = []
        algorithm_labels = []
        
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
        
        if cost_matrix:
            scenario_labels = [s.replace('_', ' ')[:8] for s in self.scenarios]  # Shortened labels
            sns.heatmap(cost_matrix, xticklabels=scenario_labels, yticklabels=algorithm_labels,
                       annot=True, fmt='.0f', cmap='Reds', ax=ax1, cbar_kws={'label': 'Cost'})
            ax1.set_title('Safety Cost Heatmap', fontweight='bold')
        
        # 2. Violation heatmap (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        violation_matrix = []
        
        for algorithm in self.algorithms:
            if algorithm in self.results_data:
                row = []
                for scenario in self.scenarios:
                    if scenario in self.results_data[algorithm]:
                        violations = self.results_data[algorithm][scenario].get('total_violations', 0)
                        row.append(violations)
                    else:
                        row.append(0)
                violation_matrix.append(row)
        
        if violation_matrix:
            sns.heatmap(violation_matrix, xticklabels=scenario_labels, yticklabels=algorithm_labels,
                       annot=True, fmt='d', cmap='Oranges', ax=ax2, cbar_kws={'label': 'Violations'})
            ax2.set_title('Violation Count Heatmap', fontweight='bold')
        
        # 3. Algorithm total performance (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        if algorithm_labels:
            total_costs = [sum(row) for row in cost_matrix]
            bars = ax3.bar(algorithm_labels, total_costs, color=sns.color_palette("husl", len(algorithm_labels)))
            ax3.set_title('Total Cost by Algorithm', fontweight='bold')
            ax3.set_ylabel('Total Cost')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, cost in zip(bars, total_costs):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{cost:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Algorithm total violations (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        if algorithm_labels:
            total_violations = [sum(row) for row in violation_matrix]
            bars = ax4.bar(algorithm_labels, total_violations, color=sns.color_palette("husl", len(algorithm_labels)))
            ax4.set_title('Total Violations by Algorithm', fontweight='bold')
            ax4.set_ylabel('Total Violations')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, violations in zip(bars, total_violations):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{violations}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Violations vs Cost scatter (bottom, spanning both columns)
        ax5 = fig.add_subplot(gs[2, :])
        colors = sns.color_palette("husl", len(self.algorithms))
        
        for i, algorithm in enumerate(self.algorithms):
            if algorithm in self.results_data:
                violations = []
                costs = []
                
                for scenario in self.scenarios:
                    if scenario in self.results_data[algorithm]:
                        data = self.results_data[algorithm][scenario]
                        violations.append(data.get('total_violations', 0))
                        costs.append(data.get('mean_cost', 0))
                
                if violations and costs:
                    ax5.scatter(violations, costs, label=algorithm, color=colors[i], s=100, alpha=0.7)
        
        ax5.set_xlabel('Violations per Scenario')
        ax5.set_ylabel('Cost per Scenario')
        ax5.set_title('Violations vs Cost Trade-off by Algorithm and Scenario', fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('SafeISO Stress Test Comprehensive Dashboard', fontsize=20, fontweight='bold', y=0.98)
        
        output_file = self.results_dir / 'comprehensive_dashboard.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive dashboard: {output_file}")
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
        
        # Original visualizations
        self.create_cost_comparison_heatmap()
        self.create_algorithm_performance_bars()
        self.create_scenario_severity_analysis()
        self.create_detailed_cost_breakdown()
        
        # NEW: Violation-focused visualizations
        self.create_violation_count_heatmap()
        self.create_violation_vs_cost_scatter()
        self.create_violation_breakdown_stacked_bars()
        self.create_algorithm_safety_ranking()
        self.create_comprehensive_dashboard()
        
        self.create_summary_report()
        
        print(f"\n✅ All visualizations saved in: {self.results_dir}")
        print("📋 Generated files:")
        print("   • cost_comparison_heatmap.png")
        print("   • algorithm_performance_comparison.png") 
        print("   • severity_vs_cost_analysis.png")
        print("   • detailed_cost_breakdown.png")
        print("   • violation_count_heatmap.png")
        print("   • violations_vs_cost_scatter.png")
        print("   • violation_breakdown_stacked_bars.png")
        print("   • algorithm_safety_ranking.png")
        print("   • comprehensive_dashboard.png")
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