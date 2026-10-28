#!/usr/bin/env python3
"""
Simple Formant Ratio Comparison Tool
------------------------------------
Loads two Praat-exported formant CSVs and plots them side-by-side
for easy comparison.

Usage: (default name: Real Speech for file1 and Synthetic Speech for file2; use label1 and label2 args to change)
    python formant_comparison.py --file1 real.csv --file2 synthetic.csv --ratio F2/F1
    python formant_comparison.py --file1 real.csv --file2 synthetic.csv --all-ratios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from datetime import datetime

# ======================== CONFIGURATION ========================

# Color scheme for comparison
COLOR_FILE1 = "#1f77b4"  # Blue for first file (typically real speech)
COLOR_FILE2 = "#d62728"  # Red for second file (typically synthetic)

# Available ratios
AVAILABLE_RATIOS = ["F2/F1", "F3/F2", "F4/F3", "F3/F1", "F4/F2"]

# ======================== FUNCTIONS ========================

def load_csv(filepath):
    """
    Load Praat CSV and handle undefined values.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with cleaned formant data
    """
    print(f"\n[INFO] Loading: {filepath}")
    df = pd.read_csv(filepath)
    
    # Convert formant columns to numeric, replacing '--undefined--' with NaN
    for col in ["F1(Hz)", "F2(Hz)", "F3(Hz)", "F4(Hz)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"  Loaded {len(df)} frames")
    return df

def compute_ratio(df, ratio_name):
    """
    Calculate specified formant ratio.
    
    Args:
        df: DataFrame with formant data
        ratio_name: Name of ratio (e.g., "F2/F1")
        
    Returns:
        DataFrame with ratio column added
    """
    ratio_map = {
        "F2/F1": ("F2(Hz)", "F1(Hz)"),
        "F3/F2": ("F3(Hz)", "F2(Hz)"),
        "F4/F3": ("F4(Hz)", "F3(Hz)"),
        "F3/F1": ("F3(Hz)", "F1(Hz)"),
        "F4/F2": ("F4(Hz)", "F2(Hz)")
    }
    
    if ratio_name not in ratio_map:
        raise ValueError(f"Unknown ratio: {ratio_name}")
    
    numerator, denominator = ratio_map[ratio_name]
    df[ratio_name] = df[numerator] / df[denominator]
    
    # Count valid ratios
    valid_count = df[ratio_name].notna().sum()
    print(f"  {ratio_name}: {valid_count} valid values")
    
    return df

def plot_comparison(df1, df2, ratio_name, label1="File 1", label2="File 2", output_dir="./output"):
    """
    Create comparison plot for specified ratio.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        ratio_name: Ratio to plot
        label1: Label for first dataset
        label2: Label for second dataset
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 6))
    
    # Plot first file
    plt.plot(df1["time(s)"], df1[ratio_name], 
             linewidth=2, color=COLOR_FILE1, alpha=0.85, label=label1)
    
    # Plot second file
    plt.plot(df2["time(s)"], df2[ratio_name], 
             linewidth=2, color=COLOR_FILE2, alpha=0.85, label=label2)
    
    plt.title(f"Formant Ratio Comparison: {ratio_name}", fontsize=16, fontweight='bold')
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel(f"{ratio_name} Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save plot with timestamp
    output_name = Path(output_dir) / f"comparison_{ratio_name.replace('/', '_')}_{timestamp}.png"
    plt.savefig(output_name, dpi=300)
    print(f"\n[✅] Plot saved as {output_name}")
    
    plt.show()

def plot_distribution_comparison(df1, df2, ratio_name, label1="File 1", label2="File 2", output_dir="./output"):
    """
    Create histogram comparison for ratio distribution.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        ratio_name: Ratio to plot
        label1: Label for first dataset
        label2: Label for second dataset
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(10, 6))
    
    # Plot histograms
    plt.hist(df1[ratio_name].dropna(), bins=50, alpha=0.6, 
             color=COLOR_FILE1, label=label1, density=True, edgecolor='black')
    plt.hist(df2[ratio_name].dropna(), bins=50, alpha=0.6, 
             color=COLOR_FILE2, label=label2, density=True, edgecolor='black')
    
    plt.title(f"Distribution Comparison: {ratio_name}", fontsize=16, fontweight='bold')
    plt.xlabel(f"{ratio_name} Value", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, axis='y', linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save plot with timestamp
    output_name = Path(output_dir) / f"distribution_{ratio_name.replace('/', '_')}_{timestamp}.png"
    plt.savefig(output_name, dpi=300)
    print(f"[✅] Distribution saved as {output_name}")
    
    plt.show()

def print_statistics(df, ratio_name, label=""):
    """
    Print basic statistics for a ratio.
    
    Args:
        df: DataFrame with ratio data
        ratio_name: Name of ratio
        label: Label for output
    """
    data = df[ratio_name].dropna()
    
    if len(data) == 0:
        print(f"\n[{label}] No valid data for {ratio_name}")
        return
    
    print(f"\n{'='*50}")
    print(f"Statistics: {label} - {ratio_name}")
    print(f"{'='*50}")
    print(f"  Count:  {len(data)}")
    print(f"  Mean:   {data.mean():.3f}")
    print(f"  Median: {data.median():.3f}")
    print(f"  Std:    {data.std():.3f}")
    print(f"  Min:    {data.min():.3f}")
    print(f"  Max:    {data.max():.3f}")

# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser(
        description="Simple Formant Ratio Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--file1", type=str, required=True,
                       help="Path to first CSV file (e.g., real speech)")
    parser.add_argument("--file2", type=str, required=True,
                       help="Path to second CSV file (e.g., synthetic speech)")
    parser.add_argument("--label1", type=str, default="Real Speech",
                       help="Label for first file (default: Real Speech)")
    parser.add_argument("--label2", type=str, default="Synthetic Speech",
                       help="Label for second file (default: Synthetic Speech)")
    parser.add_argument("--ratio", type=str, default="F2/F1",
                       choices=AVAILABLE_RATIOS,
                       help="Formant ratio to analyze")
    parser.add_argument("--all-ratios", action='store_true',
                       help="Generate plots for all available ratios")
    parser.add_argument("--no-distribution", action='store_true',
                       help="Skip distribution histogram plots")
    parser.add_argument("--output-dir", type=str, default="./formant_output",
                       help="Directory to save output plots (default: ./formant_output)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SIMPLE FORMANT RATIO COMPARISON TOOL")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    
    # Determine which ratios to analyze
    ratios_to_analyze = AVAILABLE_RATIOS if args.all_ratios else [args.ratio]
    
    # Load data
    df1 = load_csv(args.file1)
    df2 = load_csv(args.file2)
    
    # Process each ratio
    for ratio in ratios_to_analyze:
        print(f"\n{'#'*60}")
        print(f"# Analyzing: {ratio}")
        print(f"{'#'*60}")
        
        # Compute ratios
        df1 = compute_ratio(df1, ratio)
        df2 = compute_ratio(df2, ratio)
        
        # Print statistics
        print_statistics(df1, ratio, args.label1)
        print_statistics(df2, ratio, args.label2)
        
        # Generate plots
        plot_comparison(df1, df2, ratio, args.label1, args.label2, args.output_dir)
        
        if not args.no_distribution:
            plot_distribution_comparison(df1, df2, ratio, args.label1, args.label2, args.output_dir)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"All plots saved in: {args.output_dir}")

if __name__ == "__main__":
    main()