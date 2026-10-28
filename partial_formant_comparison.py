#!/usr/bin/env python3
"""
Partial Formant Contributions Comparison Tool
---------------------------------------------
Shows individual formant contributions (F1-F5) to the spectral envelope
for real vs synthetic speech comparison.

Usage: (default name: Real Speech for file1 and Synthetic Speech for file2; use label1 and label2 args to change)
    python partial_formant_comparison.py --file1 real.csv --file2 synthetic.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from datetime import datetime

# ======================== CONFIGURATION ========================

COLOR_FILE1 = "#1f77b4"  # Blue
COLOR_FILE2 = "#d62728"  # Red

FORMANT_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']  # Distinct colors for F1-F5

# ======================== FUNCTIONS ========================

def load_and_compute_partials(filepath, freq_max=6000, npoints=2000):
    """
    Load CSV and compute partial formant envelopes.
    
    Args:
        filepath: Path to CSV file
        freq_max: Maximum frequency (Hz)
        npoints: Resolution
        
    Returns:
        tuple: (freqs array, partials dict)
    """
    print(f"\n[INFO] Loading: {filepath}")
    df = pd.read_csv(filepath, sep=None, engine='python')
    
    # Clean undefined entries
    for c in ["F1(Hz)", "B1(Hz)", "F2(Hz)", "B2(Hz)", 
              "F3(Hz)", "B3(Hz)", "F4(Hz)", "B4(Hz)", 
              "F5(Hz)", "B5(Hz)"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace("--undefined--", np.nan), errors="coerce")
        else:
            df[c] = np.nan
    
    # Filter vowel frames
    df_vowel = df.dropna(subset=["F1(Hz)", "F2(Hz)"])
    print(f"  Valid vowel frames: {len(df_vowel)}")
    
    # Compute partial envelopes
    freqs = np.linspace(0, freq_max, npoints)
    partials = {}
    
    for i in range(1, 6):
        fcol, bcol = f"F{i}(Hz)", f"B{i}(Hz)"
        env = np.zeros_like(freqs)
        valid = df_vowel[[fcol, bcol]].dropna()
        
        for f, b in zip(valid[fcol], valid[bcol]):
            if np.isnan(f) or np.isnan(b) or b <= 0:
                continue
            sigma = b / 2.355  # Convert bandwidth to Gaussian std dev
            env += np.exp(-0.5 * ((freqs - f) / sigma) ** 2)
        
        if env.max() != 0:
            env /= env.max()  # Normalize each formant individually
        
        partials[f"F{i}"] = env
        print(f"  F{i}: {len(valid)} contributions")
    
    return freqs, partials

def plot_partials_overlay(freqs1, partials1, freqs2, partials2,
                          label1="Real Speech", label2="Synthetic Speech",
                          output_dir="./output"):
    """
    Plot partial formant contributions overlaid for comparison.
    
    Args:
        freqs1, partials1: Data for first file
        freqs2, partials2: Data for second file
        label1, label2: Labels
        output_dir: Save directory
    """
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    
    for i, (ax, color) in enumerate(zip(axes, FORMANT_COLORS), 1):
        formant_key = f"F{i}"
        
        # Plot both datasets
        ax.plot(freqs1, partials1[formant_key], 
                color=COLOR_FILE1, linewidth=2, alpha=0.8, label=label1)
        ax.plot(freqs2, partials2[formant_key], 
                color=COLOR_FILE2, linewidth=2, alpha=0.8, label=label2)
        
        ax.set_ylabel(f"{formant_key}\nAmplitude", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
        ax.set_xlim(0, freqs1[-1])
    
    axes[-1].set_xlabel("Frequency (Hz)", fontsize=12)
    fig.suptitle("Partial Formant Contributions Comparison (F1-F5)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save with timestamp
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = Path(output_dir) / f"partial_formants_{timestamp}.png"
    plt.savefig(output_name, dpi=300)
    print(f"\n[✅] Partial formants plot saved: {output_name}")
    
    plt.show()

def plot_combined_comparison(freqs1, partials1, freqs2, partials2,
                            label1="Real Speech", label2="Synthetic Speech",
                            output_dir="./output"):
    """
    Plot all partials on same axes for each dataset.
    
    Args:
        freqs1, partials1: Data for first file
        freqs2, partials2: Data for second file
        label1, label2: Labels
        output_dir: Save directory
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot first file
    for i, color in enumerate(FORMANT_COLORS, 1):
        formant_key = f"F{i}"
        ax1.plot(freqs1, partials1[formant_key], 
                color=color, linewidth=2, label=formant_key)
    
    ax1.set_xlabel("Frequency (Hz)", fontsize=12)
    ax1.set_ylabel("Normalized Amplitude", fontsize=12)
    ax1.set_title(label1, fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, freqs1[-1])
    
    # Plot second file
    for i, color in enumerate(FORMANT_COLORS, 1):
        formant_key = f"F{i}"
        ax2.plot(freqs2, partials2[formant_key], 
                color=color, linewidth=2, label=formant_key)
    
    ax2.set_xlabel("Frequency (Hz)", fontsize=12)
    ax2.set_ylabel("Normalized Amplitude", fontsize=12)
    ax2.set_title(label2, fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, freqs2[-1])
    
    fig.suptitle("Partial Formant Contributions: Side-by-Side", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save with timestamp
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = Path(output_dir) / f"partial_formants_sidebyside_{timestamp}.png"
    plt.savefig(output_name, dpi=300)
    print(f"[✅] Side-by-side plot saved: {output_name}")
    
    plt.show()

# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser(
        description="Partial Formant Contributions Comparison Tool",
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
    parser.add_argument("--freq-max", type=int, default=6000,
                       help="Maximum frequency (Hz)")
    parser.add_argument("--output-dir", type=str, default="./formant_output",
                       help="Directory to save plots")
    parser.add_argument("--overlay-only", action='store_true',
                       help="Only generate overlay plot")
    parser.add_argument("--sidebyside-only", action='store_true',
                       help="Only generate side-by-side plot")
    
    args = parser.parse_args()
    
    print("="*60)
    print("PARTIAL FORMANT CONTRIBUTIONS COMPARISON TOOL")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    
    # Load and compute
    freqs1, partials1 = load_and_compute_partials(args.file1, args.freq_max)
    freqs2, partials2 = load_and_compute_partials(args.file2, args.freq_max)
    
    # Generate plots
    if args.sidebyside_only:
        plot_combined_comparison(freqs1, partials1, freqs2, partials2,
                                args.label1, args.label2, args.output_dir)
    elif args.overlay_only:
        plot_partials_overlay(freqs1, partials1, freqs2, partials2,
                             args.label1, args.label2, args.output_dir)
    else:
        # Generate both by default
        plot_partials_overlay(freqs1, partials1, freqs2, partials2,
                             args.label1, args.label2, args.output_dir)
        plot_combined_comparison(freqs1, partials1, freqs2, partials2,
                                args.label1, args.label2, args.output_dir)
    
    print("\n" + "="*60)
    print("PARTIAL FORMANT ANALYSIS COMPLETE")
    print("="*60)
    print(f"All plots saved in: {args.output_dir}")

if __name__ == "__main__":
    main()