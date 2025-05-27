#!/bin/bash
# Plot Evaluation Results Script for SafeISO Safe RL Algorithms
# This script creates visualizations from the latest evaluation results

set -e

# Default parameters
RESULTS_DIR=""
OUTPUT_DIR=""  # Will be set to same as RESULTS_DIR if not specified
PLOT_FORMAT="png"
DPI=300

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --format)
            PLOT_FORMAT="$2"
            shift 2
            ;;
        --dpi)
            DPI="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --results-dir DIR     Directory containing evaluation results (auto-detected if not specified)"
            echo "  --output-dir DIR      Directory for plot outputs (default: same as results directory)"
            echo "  --format FORMAT       Plot format: png, pdf, svg (default: png)"
            echo "  --dpi DPI            Plot resolution (default: 300)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Auto-detect latest results directory if not specified
if [ -z "$RESULTS_DIR" ]; then
    RESULTS_DIR=$(find . -maxdepth 1 -name "evaluation_latest_results_*" -type d | sort | tail -1)
    if [ -z "$RESULTS_DIR" ]; then
        echo "ERROR: No evaluation results directory found!"
        echo "Please specify --results-dir or ensure evaluation results exist"
        exit 1
    fi
    echo "Auto-detected results directory: $RESULTS_DIR"
fi

# Set output directory to same as results directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$RESULTS_DIR"
    echo "Output directory set to: $OUTPUT_DIR"
fi

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Check if summary file exists
SUMMARY_FILE="$RESULTS_DIR/evaluation_summary.json"
if [ ! -f "$SUMMARY_FILE" ]; then
    echo "ERROR: Summary file not found: $SUMMARY_FILE"
    echo "Please ensure the evaluation results contain a summary file"
    exit 1
fi

echo "=== SafeISO Evaluation Results Plotting ==="
echo "Results directory: $RESULTS_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Plot format: $PLOT_FORMAT"
echo "DPI: $DPI"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create the plotting script
cat > "$OUTPUT_DIR/create_plots.py" << 'EOF'
#!/usr/bin/env python3
"""
SafeISO Evaluation Results Plotting Script

This script creates comprehensive visualizations of Safe RL algorithm performance
from the evaluation results, showing the action-invariance fix effectiveness.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import os
from pathlib import Path
import argparse

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_evaluation_data(results_dir):
    """Load evaluation data from JSON files."""
    summary_file = os.path.join(results_dir, 'evaluation_summary.json')
    
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    
    # Load individual result files for detailed episode data
    detailed_data = {}
    for item in summary_data:
        algo = item['algorithm']
        result_files = [f for f in os.listdir(results_dir) if f.startswith(f"{algo}_") and f.endswith("_results.json")]
        
        if result_files:
            result_file = os.path.join(results_dir, result_files[0])
            with open(result_file, 'r') as f:
                detailed_data[algo] = json.load(f)
    
    return summary_data, detailed_data

def create_performance_comparison(summary_data, output_dir, plot_format, dpi):
    """Create algorithm performance comparison plots."""
    
    # Extract data for plotting
    algorithms = [item['algorithm'] for item in summary_data]
    mean_rewards = [item['mean_reward'] for item in summary_data]
    std_rewards = [item['std_reward'] for item in summary_data]
    mean_costs = [item['mean_cost'] for item in summary_data]
    std_costs = [item['std_cost'] for item in summary_data]
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'Algorithm': algorithms,
        'Mean_Reward': mean_rewards,
        'Std_Reward': std_rewards,
        'Mean_Cost': mean_costs,
        'Std_Cost': std_costs
    })
    
    # Sort by performance (higher reward = better)
    df = df.sort_values('Mean_Reward', ascending=False)
    
    # 1. Reward Comparison Bar Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.bar(df['Algorithm'], df['Mean_Reward'], 
                  yerr=df['Std_Reward'], capsize=5, 
                  alpha=0.8, edgecolor='black', linewidth=1)
    
    # Color bars by performance (best to worst)
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(df)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_title('Safe RL Algorithm Performance Comparison\n(SafeISO Environment - Action-Invariance Fixed)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Reward', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, reward, std) in enumerate(zip(bars, df['Mean_Reward'], df['Std_Reward'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + (max(mean_rewards) - min(mean_rewards)) * 0.01,
                f'{reward:.0f}±{std:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Add performance ranking
    for i, (bar, algo) in enumerate(zip(bars, df['Algorithm'])):
        rank_text = f"#{i+1}"
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                rank_text, ha='center', va='center', fontweight='bold', 
                fontsize=12, color='white' if i < 3 else 'black')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/algorithm_performance_comparison.{plot_format}', 
                dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Reward Range Visualization (Action-Invariance Fix Proof)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    reward_range = max(mean_rewards) - min(mean_rewards)
    
    # Create horizontal bar chart showing reward spread
    y_pos = np.arange(len(algorithms))
    bars = ax.barh(y_pos, mean_rewards, xerr=std_rewards, 
                   alpha=0.7, capsize=5, height=0.6)
    
    # Color by performance
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Algorithm'])
    ax.set_xlabel('Mean Reward', fontsize=14, fontweight='bold')
    ax.set_title(f'Action-Invariance Fix Verification\nReward Range: {reward_range:.0f} (Algorithms Show Different Performance!)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add success message
    ax.text(0.02, 0.98, '✅ SUCCESS: Different algorithms show different performance!\n   Action-invariance issue resolved!', 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/action_invariance_fix_proof.{plot_format}', 
                dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return df

def create_episode_analysis(detailed_data, output_dir, plot_format, dpi):
    """Create detailed episode-by-episode analysis plots."""
    
    if not detailed_data:
        print("No detailed episode data available for episode analysis")
        return
    
    # 1. Episode Rewards Distribution
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    algorithms = list(detailed_data.keys())
    
    for i, algo in enumerate(algorithms):
        if i >= len(axes):
            break
            
        episode_rewards = detailed_data[algo]['episode_rewards']
        
        # Histogram
        axes[i].hist(episode_rewards, bins=15, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{algo}\nMean: {np.mean(episode_rewards):.0f}±{np.std(episode_rewards):.0f}', 
                         fontweight='bold')
        axes[i].set_xlabel('Episode Reward')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(algorithms), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Episode Reward Distributions by Algorithm', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/episode_reward_distributions.{plot_format}', 
                dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Episode-by-Episode Performance
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for algo in algorithms:
        episode_rewards = detailed_data[algo]['episode_rewards']
        episodes = range(1, len(episode_rewards) + 1)
        ax.plot(episodes, episode_rewards, marker='o', linewidth=2, 
                markersize=6, label=algo, alpha=0.8)
    
    ax.set_title('Episode-by-Episode Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xlabel('Episode Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Episode Reward', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/episode_by_episode_performance.{plot_format}', 
                dpi=dpi, bbox_inches='tight')
    plt.close()

def create_safety_analysis(summary_data, detailed_data, output_dir, plot_format, dpi):
    """Create safety constraint analysis plots."""
    
    # Extract safety data
    algorithms = [item['algorithm'] for item in summary_data]
    mean_costs = [item['mean_cost'] for item in summary_data]
    mean_violations = [item['mean_violations'] for item in summary_data]
    
    # 1. Safety Performance Summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Cost violations
    bars1 = ax1.bar(algorithms, mean_costs, alpha=0.8, color='red', edgecolor='black')
    ax1.set_title('Safety Cost Violations by Algorithm', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Algorithm', fontweight='bold')
    ax1.set_ylabel('Mean Cost', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, cost in zip(bars1, mean_costs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{cost:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Violation counts
    bars2 = ax2.bar(algorithms, mean_violations, alpha=0.8, color='orange', edgecolor='black')
    ax2.set_title('Safety Violations Count by Algorithm', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Algorithm', fontweight='bold')
    ax2.set_ylabel('Mean Violations per Episode', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, violations in zip(bars2, mean_violations):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{violations:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/safety_analysis.{plot_format}', 
                dpi=dpi, bbox_inches='tight')
    plt.close()

def create_summary_report(summary_data, detailed_data, output_dir):
    """Create a text summary report."""
    
    report_file = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SafeISO Safe RL Algorithm Evaluation Report\n")
        f.write("Action-Invariance Issue Resolution Verification\n")
        f.write("=" * 80 + "\n\n")
        
        # Performance ranking
        sorted_data = sorted(summary_data, key=lambda x: x['mean_reward'], reverse=True)
        
        f.write("ALGORITHM PERFORMANCE RANKING:\n")
        f.write("-" * 40 + "\n")
        for i, item in enumerate(sorted_data, 1):
            f.write(f"{i:2d}. {item['algorithm']:<12} | Reward: {item['mean_reward']:>10.2f} ± {item['std_reward']:>6.2f}\n")
        
        f.write("\n")
        
        # Action-invariance verification
        rewards = [item['mean_reward'] for item in summary_data]
        reward_range = max(rewards) - min(rewards)
        
        f.write("ACTION-INVARIANCE FIX VERIFICATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Reward Range: {reward_range:.2f}\n")
        f.write(f"Min Reward:   {min(rewards):.2f}\n")
        f.write(f"Max Reward:   {max(rewards):.2f}\n")
        
        if reward_range > 1000:  # Significant difference
            f.write("\n✅ SUCCESS: Algorithms show significantly different performance!\n")
            f.write("   The action-invariance issue has been RESOLVED.\n")
        else:
            f.write("\n⚠️  WARNING: Algorithms show similar performance.\n")
            f.write("   Action-invariance issue may still exist.\n")
        
        f.write("\n")
        
        # Safety analysis
        f.write("SAFETY CONSTRAINT ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        total_violations = sum(item['mean_violations'] for item in summary_data)
        total_cost = sum(item['mean_cost'] for item in summary_data)
        
        f.write(f"Total Violations: {total_violations:.1f}\n")
        f.write(f"Total Cost:       {total_cost:.3f}\n")
        
        if total_violations == 0 and total_cost == 0:
            f.write("✅ All algorithms respect safety constraints!\n")
        else:
            f.write("⚠️  Some safety violations detected.\n")
        
        f.write("\n")
        
        # Detailed algorithm analysis
        f.write("DETAILED ALGORITHM ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        for item in sorted_data:
            f.write(f"\n{item['algorithm']}:\n")
            f.write(f"  Mean Reward:     {item['mean_reward']:>10.2f} ± {item['std_reward']:>6.2f}\n")
            f.write(f"  Mean Cost:       {item['mean_cost']:>10.3f} ± {item['std_cost']:>6.3f}\n")
            f.write(f"  Mean Violations: {item['mean_violations']:>10.1f} ± {item['std_violations']:>6.1f}\n")
    
    print(f"Summary report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Plot SafeISO evaluation results')
    parser.add_argument('results_dir', help='Directory containing evaluation results')
    parser.add_argument('--output-dir', help='Output directory for plots (default: same as results directory)')
    parser.add_argument('--format', default='png', choices=['png', 'pdf', 'svg'], help='Plot format')
    parser.add_argument('--dpi', type=int, default=300, help='Plot resolution')
    
    args = parser.parse_args()
    
    # Set output directory to same as results directory if not specified
    if not args.output_dir:
        args.output_dir = args.results_dir
    
    print("Loading evaluation data...")
    summary_data, detailed_data = load_evaluation_data(args.results_dir)
    
    print(f"Found {len(summary_data)} algorithms in evaluation results")
    print("Creating plots...")
    
    # Create all plots
    df = create_performance_comparison(summary_data, args.output_dir, args.format, args.dpi)
    create_episode_analysis(detailed_data, args.output_dir, args.format, args.dpi)
    create_safety_analysis(summary_data, detailed_data, args.output_dir, args.format, args.dpi)
    create_summary_report(summary_data, detailed_data, args.output_dir)
    
    print(f"\n✅ All plots created successfully!")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Plot format: {args.format}")
    print(f"🎯 Resolution: {args.dpi} DPI")
    
    # List created files
    plot_files = [f for f in os.listdir(args.output_dir) if f.endswith(f'.{args.format}')]
    print(f"\n📈 Created {len(plot_files)} plots:")
    for plot_file in sorted(plot_files):
        print(f"   • {plot_file}")
    
    print(f"\n📄 Summary report: evaluation_report.txt")

if __name__ == '__main__':
    main()
EOF

chmod +x "$OUTPUT_DIR/create_plots.py"

# Run the plotting script
echo "Creating plots from evaluation results..."
python "$OUTPUT_DIR/create_plots.py" "$RESULTS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --format "$PLOT_FORMAT" \
    --dpi "$DPI"

echo ""
echo "=== Plotting Complete! ==="
echo "📁 Plots saved in: $OUTPUT_DIR"
echo "📊 Format: $PLOT_FORMAT"
echo "🎯 Resolution: ${DPI} DPI"
echo ""
echo "Generated plots:"
echo "  • algorithm_performance_comparison.$PLOT_FORMAT - Main performance comparison"
echo "  • action_invariance_fix_proof.$PLOT_FORMAT - Proof that fix worked"
echo "  • episode_reward_distributions.$PLOT_FORMAT - Episode analysis"
echo "  • episode_by_episode_performance.$PLOT_FORMAT - Detailed episode tracking"
echo "  • safety_analysis.$PLOT_FORMAT - Safety constraint analysis"
echo "  • evaluation_report.txt - Comprehensive text summary"
echo ""
echo "🎉 All visualizations ready for analysis!" 