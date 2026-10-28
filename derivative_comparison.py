#!/usr/bin/env python3
"""
Formant Ratio Derivative Comparison Tool
----------------------------------------
Computes and compares the rate of change (derivative) of formant ratios
between two speech samples.

Usage: (default name: Real Speech for file1 and Synthetic Speech for file2; use label1 and label2 args to change)
    python derivative_comparison.py --file1 real.csv --file2 synthetic.csv --ratio F2/F1
    python derivative_comparison.py --file1 real.csv --file2 synthetic.csv --all-ratios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from datetime import datetime

# ======================== CONFIGURATION ========================

# Color scheme
COLOR_FILE1 = "#1f77b4"  # Blue
COLOR_FILE2 = "#d62728"  # Red

AVAILABLE_RATIOS = ["F2/F1", "F3/F2", "F4/F3", "F3/F1", "F4/F2"]

# ======================== FUNCTIONS ========================

def load_and_compute_ratio(filepath, ratio_name):
    """
    Load CSV, compute ratio, and its derivative.
    
    Args:
        filepath: Path to CSV file
        ratio_name: Name of ratio (e.g., "F2/F1")
        
    Returns:
        DataFrame with ratio and derivative columns
    """
    print(f"\n[INFO] Loading: {filepath}")
    df = pd.read_csv(filepath)
    
    # Convert formants to numeric
    for col in ["F1(Hz)", "F2(Hz)", "F3(Hz)", "F4(Hz)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Compute ratio
    ratio_map = {
        "F2/F1": ("F2(Hz)", "F1(Hz)"),
        "F3/F2": ("F3(Hz)", "F2(Hz)"),
        "F4/F3": ("F4(Hz)", "F3(Hz)"),
        "F3/F1": ("F3(Hz)", "F1(Hz)"),
        "F4/F2": ("F4(Hz)", "F2(Hz)")
    }
    
    num, den = ratio_map[ratio_name]
    df[ratio_name] = df[num] / df[den]
    
    # Compute derivative
    df["time_diff"] = df["time(s)"].diff()
    df[f"d({ratio_name})/dt"] = df[ratio_name].diff() / df["time_diff"]
    
    valid_count = df[f"d({ratio_name})/dt"].notna().sum()
    print(f"  {ratio_name}: {valid_count} valid derivative values")
    
    return df

def plot_derivative_comparison(df1, df2, ratio_name, label1="Real Speech", label2="Synthetic Speech", output_dir="./output"):
    """
    Plot derivative comparison for a ratio.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        ratio_name: Ratio name
        label1: Label for first dataset
        label2: Label for second dataset
        output_dir: Directory to save plot
    """
    derivative_col = f"d({ratio_name})/dt"
    
    plt.figure(figsize=(12, 6))
    
    # Plot derivatives
    plt.plot(df1["time(s)"], df1[derivative_col], 
             color=COLOR_FILE1, linewidth=2, alpha=0.85, label=label1)
    plt.plot(df2["time(s)"], df2[derivative_col], 
             color=COLOR_FILE2, linewidth=2, alpha=0.85, label=label2)
    
    # Zero line
    plt.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    
    plt.title(f"Derivative of {ratio_name} Over Time", fontsize=16, fontweight='bold')
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel(f"d({ratio_name})/dt", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    # Save with timestamp
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = Path(output_dir) / f"derivative_{ratio_name.replace('/', '_')}_{timestamp}.png"
    plt.savefig(output_name, dpi=300)
    print(f"\n[✅] Derivative plot saved: {output_name}")
    
    plt.show()

def plot_derivative_distribution(df1, df2, ratio_name, label1="Real Speech", label2="Synthetic Speech", output_dir="./output"):
    """
    Plot histogram of derivative values.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        ratio_name: Ratio name
        label1: Label for first dataset
        label2: Label for second dataset
        output_dir: Directory to save plot
    """
    derivative_col = f"d({ratio_name})/dt"
    
    plt.figure(figsize=(10, 6))
    
    # Get data, remove NaN and extreme outliers for better visualization
    data1 = df1[derivative_col].dropna()
    data2 = df2[derivative_col].dropna()
    
    # Remove extreme outliers (beyond 99th percentile) for better visualization
    p99_1 = data1.quantile(0.99)
    p01_1 = data1.quantile(0.01)
    p99_2 = data2.quantile(0.99)
    p01_2 = data2.quantile(0.01)
    
    data1_filtered = data1[(data1 >= p01_1) & (data1 <= p99_1)]
    data2_filtered = data2[(data2 >= p01_2) & (data2 <= p99_2)]
    
    # Plot histograms
    plt.hist(data1_filtered, bins=50, alpha=0.6, color=COLOR_FILE1, 
             label=label1, density=True, edgecolor='black')
    plt.hist(data2_filtered, bins=50, alpha=0.6, color=COLOR_FILE2, 
             label=label2, density=True, edgecolor='black')
    
    plt.axvline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    
    plt.title(f"Distribution of d({ratio_name})/dt (Outliers Removed)", fontsize=16, fontweight='bold')
    plt.xlabel(f"d({ratio_name})/dt", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y', linestyle="--", alpha=0.3)
    plt.tight_layout()
    
    # Save with timestamp
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = Path(output_dir) / f"derivative_dist_{ratio_name.replace('/', '_')}_{timestamp}.png"
    plt.savefig(output_name, dpi=300)
    print(f"[✅] Distribution plot saved: {output_name}")
    
    plt.show()

def print_derivative_statistics(df, ratio_name, label=""):
    """
    Print statistics for derivative values.
    
    Args:
        df: DataFrame with derivative data
        ratio_name: Ratio name
        label: Label for output
    """
    derivative_col = f"d({ratio_name})/dt"
    data = df[derivative_col].dropna()
    
    if len(data) == 0:
        print(f"\n[{label}] No valid derivative data for {ratio_name}")
        return
    
    print(f"\n{'='*50}")
    print(f"Derivative Statistics: {label} - {ratio_name}")
    print(f"{'='*50}")
    print(f"  Count:       {len(data)}")
    print(f"  Mean:        {data.mean():.3f}")
    print(f"  Median:      {data.median():.3f}")
    print(f"  Std Dev:     {data.std():.3f}")
    print(f"  Min:         {data.min():.3f}")
    print(f"  Max:         {data.max():.3f}")
    print(f"  Max |value|: {data.abs().max():.3f}")
    print(f"  Range:       {data.max() - data.min():.3f}")

# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser(
        description="Formant Ratio Derivative Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--file1", type=str, required=True,
                       help="Path to first CSV file (e.g., real speech)")
    parser.add_argument("--file2", type=str, required=True,
                       help="Path to second CSV file (e.g., synthetic speech)")
    parser.add_argument("--label1", type=str, default="Real Speech",
                       help="Label for first file")
    parser.add_argument("--label2", type=str, default="Synthetic Speech",
                       help="Label for second file")
    parser.add_argument("--ratio", type=str, default="F2/F1",
                       choices=AVAILABLE_RATIOS,
                       help="Formant ratio to analyze")
    parser.add_argument("--all-ratios", action='store_true',
                       help="Generate plots for all available ratios")
    parser.add_argument("--no-distribution", action='store_true',
                       help="Skip distribution histogram plots")
    parser.add_argument("--output-dir", type=str, default="./formant_output",
                       help="Directory to save plots")
    
    args = parser.parse_args()
    
    print("="*60)
    print("FORMANT RATIO DERIVATIVE COMPARISON TOOL")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    
    # Determine which ratios to analyze
    ratios_to_analyze = AVAILABLE_RATIOS if args.all_ratios else [args.ratio]
    
    for ratio in ratios_to_analyze:
        print(f"\n{'#'*60}")
        print(f"# Analyzing: {ratio}")
        print(f"{'#'*60}")
        
        # Load and compute
        df1 = load_and_compute_ratio(args.file1, ratio)
        df2 = load_and_compute_ratio(args.file2, ratio)
        
        # Print statistics
        print_derivative_statistics(df1, ratio, args.label1)
        print_derivative_statistics(df2, ratio, args.label2)
        
        # Generate plots
        plot_derivative_comparison(df1, df2, ratio, args.label1, args.label2, args.output_dir)
        
        if not args.no_distribution:
            plot_derivative_distribution(df1, df2, ratio, args.label1, args.label2, args.output_dir)
    
    print("\n" + "="*60)
    print("DERIVATIVE ANALYSIS COMPLETE")
    print("="*60)
    print(f"All plots saved in: {args.output_dir}")

if __name__ == "__main__":
    main()