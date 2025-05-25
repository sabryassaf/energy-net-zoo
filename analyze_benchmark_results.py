#!/usr/bin/env python3
"""
Benchmark Results Analysis Script for Safe RL ISO Agents

This script analyzes and compares the performance of all trained safe RL algorithms.
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

def load_evaluation_results():
    """Load evaluation results from all algorithms."""
    results = {}
    algorithms = ["PPOLag", "CPO", "FOCOPS", "CUP", "SautRL"]
    
    for algo in algorithms:
        # First try to load from the 'latest' symlink
        latest_path = f"logs/eval/{algo}/latest/evaluation_results.json"
        if os.path.exists(latest_path):
            try:
                with open(latest_path, 'r') as f:
                    results[algo] = json.load(f)
                print(f"✅ Loaded {algo} results from latest run")
                continue
            except Exception as e:
                print(f"⚠️  Error loading {algo} from latest: {e}")
        
        # If no latest, try to find the most recent timestamped directory
        algo_dir = f"logs/eval/{algo}/"
        if os.path.exists(algo_dir):
            # Get all subdirectories with timestamps
            subdirs = [d for d in os.listdir(algo_dir) 
                      if os.path.isdir(os.path.join(algo_dir, d)) 
                      and d != "latest"]
            
            if subdirs:
                # Sort by modification time and take the most recent
                subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(algo_dir, x)), reverse=True)
                most_recent = subdirs[0]
                eval_path = f"logs/eval/{algo}/{most_recent}/evaluation_results.json"
                
                if os.path.exists(eval_path):
                    try:
                        with open(eval_path, 'r') as f:
                            results[algo] = json.load(f)
                        print(f"✅ Loaded {algo} results from {most_recent}")
                        continue
                    except Exception as e:
                        print(f"⚠️  Error loading {algo} from {most_recent}: {e}")
        
        # Fallback to old location (backward compatibility)
        old_path = f"logs/eval/{algo}/evaluation_results.json"
        if os.path.exists(old_path):
            try:
                with open(old_path, 'r') as f:
                    results[algo] = json.load(f)
                print(f"✅ Loaded {algo} results from old location")
            except Exception as e:
                print(f"⚠️  Error loading {algo} from old location: {e}")
        else:
            print(f"⚠️  No results found for {algo}")
    
    return results

def extract_metrics(results):
    """Extract key metrics from evaluation results."""
    metrics = []
    
    for algo, data in results.items():
        # Regular scenario metrics
        regular = data.get('regular', {})
        
        metric = {
            'Algorithm': algo,
            'Mean_Reward': regular.get('mean_reward', 0),
            'Std_Reward': regular.get('std_reward', 0),
            'Mean_Cost': regular.get('mean_cost', 0),
            'Std_Cost': regular.get('std_cost', 0),
            'Total_Voltage_Violations': regular.get('total_violations', {}).get('voltage', 0),
            'Total_Frequency_Violations': regular.get('total_violations', {}).get('frequency', 0),
            'Total_Battery_Violations': regular.get('total_violations', {}).get('battery', 0),
            'Total_Supply_Demand_Violations': regular.get('total_violations', {}).get('supply_demand', 0),
        }
        
        # Calculate total violations
        metric['Total_Violations'] = sum([
            metric['Total_Voltage_Violations'],
            metric['Total_Frequency_Violations'], 
            metric['Total_Battery_Violations'],
            metric['Total_Supply_Demand_Violations']
        ])
        
        # Safety score (0 = perfect safety, higher = more violations)
        metric['Safety_Score'] = metric['Total_Violations']
        
        # Economic efficiency (reward per unit cost, handle zero cost)
        if metric['Mean_Cost'] > 0:
            metric['Economic_Efficiency'] = abs(metric['Mean_Reward']) / metric['Mean_Cost']
        else:
            metric['Economic_Efficiency'] = abs(metric['Mean_Reward'])  # Perfect safety case
            
        metrics.append(metric)
    
    return pd.DataFrame(metrics)

def create_comparison_plots(df):
    """Create comprehensive comparison plots."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Safe RL ISO Agent Benchmark Comparison', fontsize=16, fontweight='bold')
    
    # 1. Mean Reward Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['Algorithm'], df['Mean_Reward'], color='skyblue', alpha=0.7)
    ax1.set_title('Mean Reward (Higher = Better)', fontweight='bold')
    ax1.set_ylabel('Mean Reward')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}', ha='center', va='bottom')
    
    # 2. Safety Performance (Total Violations)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['Algorithm'], df['Total_Violations'], color='lightcoral', alpha=0.7)
    ax2.set_title('Total Safety Violations (Lower = Better)', fontweight='bold')
    ax2.set_ylabel('Total Violations')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 3. Cost Performance
    ax3 = axes[0, 2]
    bars3 = ax3.bar(df['Algorithm'], df['Mean_Cost'], color='lightgreen', alpha=0.7)
    ax3.set_title('Mean Safety Cost (Lower = Better)', fontweight='bold')
    ax3.set_ylabel('Mean Cost')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    # 4. Violation Breakdown
    ax4 = axes[1, 0]
    violation_types = ['Voltage', 'Frequency', 'Battery', 'Supply_Demand']
    violation_cols = ['Total_Voltage_Violations', 'Total_Frequency_Violations', 
                     'Total_Battery_Violations', 'Total_Supply_Demand_Violations']
    
    bottom = np.zeros(len(df))
    colors = ['red', 'orange', 'yellow', 'purple']
    
    for i, (vtype, vcol) in enumerate(zip(violation_types, violation_cols)):
        ax4.bar(df['Algorithm'], df[vcol], bottom=bottom, label=vtype, color=colors[i], alpha=0.7)
        bottom += df[vcol]
    
    ax4.set_title('Constraint Violation Breakdown', fontweight='bold')
    ax4.set_ylabel('Number of Violations')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Reward vs Cost Scatter
    ax5 = axes[1, 1]
    scatter = ax5.scatter(df['Mean_Cost'], df['Mean_Reward'], 
                         s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
    
    for i, algo in enumerate(df['Algorithm']):
        ax5.annotate(algo, (df.iloc[i]['Mean_Cost'], df.iloc[i]['Mean_Reward']),
                    xytext=(5, 5), textcoords='offset points')
    
    ax5.set_xlabel('Mean Cost')
    ax5.set_ylabel('Mean Reward')
    ax5.set_title('Risk-Return Trade-off', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Overall Performance Radar (normalized metrics)
    ax6 = axes[1, 2]
    
    # Normalize metrics for radar chart (0-1 scale, 1 = best)
    df_norm = df.copy()
    df_norm['Norm_Reward'] = (df['Mean_Reward'] - df['Mean_Reward'].min()) / (df['Mean_Reward'].max() - df['Mean_Reward'].min())
    df_norm['Norm_Safety'] = 1 - (df['Total_Violations'] / df['Total_Violations'].max()) if df['Total_Violations'].max() > 0 else 1
    df_norm['Norm_Cost'] = 1 - (df['Mean_Cost'] / df['Mean_Cost'].max()) if df['Mean_Cost'].max() > 0 else 1
    
    # Simple bar chart instead of radar for clarity
    metrics_names = ['Reward', 'Safety', 'Low Cost']
    x_pos = np.arange(len(df))
    width = 0.15
    
    for i, algo in enumerate(df['Algorithm']):
        values = [df_norm.iloc[i]['Norm_Reward'], 
                 df_norm.iloc[i]['Norm_Safety'],
                 df_norm.iloc[i]['Norm_Cost']]
        ax6.bar(x_pos + i*width, values, width, label=algo, alpha=0.7)
    
    ax6.set_xlabel('Performance Metrics')
    ax6.set_ylabel('Normalized Score (0-1)')
    ax6.set_title('Overall Performance Comparison', fontweight='bold')
    ax6.set_xticks(x_pos + width * 2)
    ax6.set_xticklabels(metrics_names)
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('logs/benchmark_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Comparison plots saved to: logs/benchmark_comparison.png")

def generate_report(df):
    """Generate a comprehensive text report."""
    
    report = []
    report.append("=" * 60)
    report.append("SAFE RL ISO AGENT BENCHMARK RESULTS")
    report.append("=" * 60)
    report.append("")
    
    # Overall rankings
    report.append("🏆 ALGORITHM RANKINGS:")
    report.append("-" * 30)
    
    # Best reward (least negative)
    best_reward = df.loc[df['Mean_Reward'].idxmax()]
    report.append(f"🥇 Best Economic Performance: {best_reward['Algorithm']} (Reward: {best_reward['Mean_Reward']:.0f})")
    
    # Best safety (fewest violations)
    best_safety = df.loc[df['Total_Violations'].idxmin()]
    report.append(f"🛡️  Best Safety Performance: {best_safety['Algorithm']} (Violations: {best_safety['Total_Violations']})")
    
    # Best cost (lowest)
    best_cost = df.loc[df['Mean_Cost'].idxmin()]
    report.append(f"💰 Lowest Safety Cost: {best_cost['Algorithm']} (Cost: {best_cost['Mean_Cost']:.3f})")
    
    report.append("")
    report.append("📊 DETAILED RESULTS:")
    report.append("-" * 30)
    
    for _, row in df.iterrows():
        report.append(f"\n{row['Algorithm']}:")
        report.append(f"  Economic Performance: {row['Mean_Reward']:.0f} ± {row['Std_Reward']:.0f}")
        report.append(f"  Safety Cost: {row['Mean_Cost']:.3f} ± {row['Std_Cost']:.3f}")
        report.append(f"  Total Violations: {row['Total_Violations']}")
        report.append(f"    - Voltage: {row['Total_Voltage_Violations']}")
        report.append(f"    - Frequency: {row['Total_Frequency_Violations']}")
        report.append(f"    - Battery: {row['Total_Battery_Violations']}")
        report.append(f"    - Supply-Demand: {row['Total_Supply_Demand_Violations']}")
    
    report.append("")
    report.append("🎯 RECOMMENDATIONS:")
    report.append("-" * 30)
    
    # Find the best overall algorithm
    perfect_safety = df[df['Total_Violations'] == 0]
    if len(perfect_safety) > 0:
        best_overall = perfect_safety.loc[perfect_safety['Mean_Reward'].idxmax()]
        report.append(f"✅ Recommended Algorithm: {best_overall['Algorithm']}")
        report.append(f"   Reason: Perfect safety with best economic performance ({best_overall['Mean_Reward']:.0f})")
    else:
        # Trade-off analysis
        df['Score'] = df['Mean_Reward'] - (df['Total_Violations'] * 1000)  # Penalty for violations
        best_overall = df.loc[df['Score'].idxmax()]
        report.append(f"⚖️  Best Trade-off: {best_overall['Algorithm']}")
        report.append(f"   Reason: Best balance of performance and safety")
    
    report.append("")
    report.append("=" * 60)
    
    # Save report
    report_text = "\n".join(report)
    with open('logs/benchmark_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n📄 Full report saved to: logs/benchmark_report.txt")

def list_all_evaluation_runs():
    """List all available evaluation runs for each algorithm."""
    algorithms = ["PPOLag", "CPO", "FOCOPS", "CUP", "SautRL"]
    
    print("\n📂 Available Evaluation Runs:")
    print("=" * 50)
    
    for algo in algorithms:
        algo_dir = f"logs/eval/{algo}/"
        if os.path.exists(algo_dir):
            subdirs = [d for d in os.listdir(algo_dir) 
                      if os.path.isdir(os.path.join(algo_dir, d)) 
                      and d != "latest"]
            
            if subdirs:
                subdirs.sort(reverse=True)  # Most recent first
                print(f"\n{algo}:")
                for i, subdir in enumerate(subdirs):
                    marker = " (latest)" if i == 0 else ""
                    print(f"  - {subdir}{marker}")
            else:
                print(f"\n{algo}: No runs found")
        else:
            print(f"\n{algo}: No evaluation directory")

def main():
    """Main analysis function."""
    # Check if user wants to list available runs
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        list_all_evaluation_runs()
        return
    
    print("🔍 Safe RL ISO Agent Benchmark Analysis")
    print("=" * 50)
    
    # Show available runs
    list_all_evaluation_runs()
    
    # Load results
    print("\n📂 Loading latest evaluation results...")
    results = load_evaluation_results()
    
    if not results:
        print("❌ No evaluation results found!")
        print("💡 Make sure you have run evaluations for your trained models.")
        print("   Use: ./run_safe_iso.sh --eval-only --model-path MODEL_PATH --algo ALGO")
        print("\n📋 To see all available evaluation runs:")
        print("   python analyze_benchmark_results.py --list")
        return
    
    print(f"✅ Found results for {len(results)} algorithms")
    
    # Extract metrics
    print("\n📊 Extracting metrics...")
    df = extract_metrics(results)
    print(df.to_string(index=False))
    
    # Save detailed metrics with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metrics_file = f'logs/benchmark_metrics_{timestamp}.csv'
    df.to_csv(metrics_file, index=False)
    print(f"\n💾 Detailed metrics saved to: {metrics_file}")
    
    # Create plots
    print("\n📈 Creating comparison plots...")
    create_comparison_plots(df)
    
    # Generate report
    print("\n📋 Generating report...")
    generate_report(df)
    
    print("\n🎉 Benchmark analysis complete!")
    print("\nGenerated files:")
    print("  - logs/benchmark_comparison.png (Comparison plots)")
    print("  - logs/benchmark_report.txt (Text report)")
    print(f"  - {metrics_file} (Raw metrics)")
    print("\n💡 Use 'python analyze_benchmark_results.py --list' to see all available evaluation runs")

if __name__ == "__main__":
    main() 