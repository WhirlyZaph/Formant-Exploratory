#!/usr/bin/env python3
"""
F1-F2 Vowel Space Comparison Tool
----------------------------------
Loads two Praat-exported formant CSVs and plots their F1-F2 vowel spaces
for side-by-side comparison.

Usage: (default name: Real Speech for file1 and Synthetic Speech for file2; use label1 and label2 args to change)
    python vowel_space_comparison.py --file1 real.csv --file2 synthetic.csv
    python vowel_space_comparison.py --file1 real.csv --file2 synthetic.csv --separate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from datetime import datetime

# ======================== CONFIGURATION ========================

# Color scheme for comparison
COLOR_FILE1 = "#1f77b4"  # Blue for first file
COLOR_FILE2 = "#d62728"  # Red for second file

# ======================== FUNCTIONS ========================

def load_formants(filepath):
    """
    Load formant data from Praat CSV.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with F1 and F2 data
    """
    print(f"\n[INFO] Loading: {filepath}")
    df = pd.read_csv(filepath)
    
    # Convert formant columns to numeric
    for col in ["F1(Hz)", "F2(Hz)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with missing formants
    df = df.dropna(subset=["F1(Hz)", "F2(Hz)"])
    
    print(f"  Loaded {len(df)} valid frames")
    return df

def plot_vowel_space_overlay(df1, df2, label1="Real Speech", label2="Synthetic Speech", output_dir="./output"):
    """
    Create overlaid vowel space plot.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        label1: Label for first dataset
        label2: Label for second dataset
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot first file
    plt.scatter(
        df1["F2(Hz)"],
        df1["F1(Hz)"],
        s=30,
        alpha=0.5,
        color=COLOR_FILE1,
        edgecolor="black",
        linewidth=0.2,
        label=label1
    )
    
    # Plot second file
    plt.scatter(
        df2["F2(Hz)"],
        df2["F1(Hz)"],
        s=30,
        alpha=0.5,
        color=COLOR_FILE2,
        edgecolor="black",
        linewidth=0.2,
        label=label2
    )
    
    # Invert y-axis (phonetic convention: high vowels at top)
    plt.gca().invert_yaxis()
    
    plt.title("F1-F2 Vowel Space Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("F2 (Hz) → frontness", fontsize=14)
    plt.ylabel("F1 (Hz) → height", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    # Save with timestamp
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = Path(output_dir) / f"vowel_space_overlay_{timestamp}.png"
    plt.savefig(output_name, dpi=300)
    print(f"\n[✅] Overlay plot saved: {output_name}")
    
    plt.show()

def plot_vowel_space_separate(df1, df2, label1="Real Speech", label2="Synthetic Speech", output_dir="./output"):
    """
    Create side-by-side vowel space plots.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        label1: Label for first dataset
        label2: Label for second dataset
        output_dir: Directory to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot first file
    ax1.scatter(
        df1["F2(Hz)"],
        df1["F1(Hz)"],
        s=30,
        alpha=0.5,
        color=COLOR_FILE1,
        edgecolor="black",
        linewidth=0.2
    )
    ax1.invert_yaxis()
    ax1.set_title(label1, fontsize=14, fontweight='bold')
    ax1.set_xlabel("F2 (Hz) → frontness", fontsize=12)
    ax1.set_ylabel("F1 (Hz) → height", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.3)
    
    # Plot second file
    ax2.scatter(
        df2["F2(Hz)"],
        df2["F1(Hz)"],
        s=30,
        alpha=0.5,
        color=COLOR_FILE2,
        edgecolor="black",
        linewidth=0.2
    )
    ax2.invert_yaxis()
    ax2.set_title(label2, fontsize=14, fontweight='bold')
    ax2.set_xlabel("F2 (Hz) → frontness", fontsize=12)
    ax2.set_ylabel("F1 (Hz) → height", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.3)
    
    # Match axis limits for fair comparison
    f1_min = min(df1["F1(Hz)"].min(), df2["F1(Hz)"].min())
    f1_max = max(df1["F1(Hz)"].max(), df2["F1(Hz)"].max())
    f2_min = min(df2["F2(Hz)"].min(), df2["F2(Hz)"].min())
    f2_max = max(df1["F2(Hz)"].max(), df2["F2(Hz)"].max())
    
    ax1.set_xlim(f2_min, f2_max)
    ax1.set_ylim(f1_max, f1_min)  # Inverted
    ax2.set_xlim(f2_min, f2_max)
    ax2.set_ylim(f1_max, f1_min)  # Inverted
    
    fig.suptitle("F1-F2 Vowel Space: Side-by-Side Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save with timestamp
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = Path(output_dir) / f"vowel_space_separate_{timestamp}.png"
    plt.savefig(output_name, dpi=300)
    print(f"[✅] Side-by-side plot saved: {output_name}")
    
    plt.show()

def print_statistics(df, label=""):
    """
    Print vowel space statistics.
    
    Args:
        df: DataFrame with formant data
        label: Label for output
    """
    print(f"\n{'='*50}")
    print(f"Vowel Space Statistics: {label}")
    print(f"{'='*50}")
    print(f"  F1 range: {df['F1(Hz)'].min():.0f} - {df['F1(Hz)'].max():.0f} Hz")
    print(f"  F1 mean:  {df['F1(Hz)'].mean():.0f} Hz")
    print(f"  F2 range: {df['F2(Hz)'].min():.0f} - {df['F2(Hz)'].max():.0f} Hz")
    print(f"  F2 mean:  {df['F2(Hz)'].mean():.0f} Hz")
    print(f"  Total frames: {len(df)}")

# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser(
        description="F1-F2 Vowel Space Comparison Tool",
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
    parser.add_argument("--separate", action='store_true',
                       help="Create side-by-side plots instead of overlay")
    parser.add_argument("--both", action='store_true',
                       help="Create both overlay and side-by-side plots")
    parser.add_argument("--output-dir", type=str, default="./formant_output",
                       help="Directory to save plots")
    
    args = parser.parse_args()
    
    print("="*60)
    print("F1-F2 VOWEL SPACE COMPARISON TOOL")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    df1 = load_formants(args.file1)
    df2 = load_formants(args.file2)
    
    # Print statistics
    print_statistics(df1, args.label1)
    print_statistics(df2, args.label2)
    
    # Generate plots
    if args.both:
        plot_vowel_space_overlay(df1, df2, args.label1, args.label2, args.output_dir)
        plot_vowel_space_separate(df1, df2, args.label1, args.label2, args.output_dir)
    elif args.separate:
        plot_vowel_space_separate(df1, df2, args.label1, args.label2, args.output_dir)
    else:
        plot_vowel_space_overlay(df1, df2, args.label1, args.label2, args.output_dir)
    
    print("\n" + "="*60)
    print("VOWEL SPACE ANALYSIS COMPLETE")
    print("="*60)
    print(f"All plots saved in: {args.output_dir}")

if __name__ == "__main__":
    main()