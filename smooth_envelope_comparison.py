#!/usr/bin/env python3
"""
Smooth Spectral Envelope Comparison Tool
----------------------------------------
Generates smooth, global average spectral envelopes from formant tracks
using Gaussian smoothing over all formant observations.

Usage: (default name: Real Speech for file1 and Synthetic Speech for file2; use label1 and label2 args to change)
    python smooth_envelope_comparison.py --file1 real.csv --file2 synthetic.csv
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

# ======================== FUNCTIONS ========================

def load_and_compute_smooth_envelope(filepath):
    """
    Load formant data and compute smooth spectral envelope.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        tuple: (freqs array, envelope array, stats dict)
    """
    print(f"\n[INFO] Loading: {filepath}")
    df = pd.read_csv(filepath)
    
    # Convert to numeric
    for col in ["F1(Hz)", "B1(Hz)", "F2(Hz)", "B2(Hz)", "F3(Hz)", "B3(Hz)", "F4(Hz)", "B4(Hz)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Collect all formants into one pool
    formant_freqs = []
    bandwidths = []
    
    for i in range(1, 5):
        f_col = f"F{i}(Hz)"
        b_col = f"B{i}(Hz)"
        valid = df[[f_col, b_col]].dropna()
        formant_freqs.extend(valid[f_col].values)
        bandwidths.extend(valid[b_col].values)
    
    formant_freqs = np.array(formant_freqs)
    bandwidths = np.array(bandwidths)
    
    if len(formant_freqs) == 0:
        raise ValueError(f"No valid formant data in {filepath}")
    
    # Compute global averages
    f_mean = np.mean(formant_freqs)
    b_mean = np.mean(bandwidths)
    
    stats = {
        'f_mean': f_mean,
        'b_mean': b_mean,
        'n_formants': len(formant_freqs)
    }
    
    print(f"  Global average formant frequency: {f_mean:.0f} Hz")
    print(f"  Global average bandwidth: {b_mean:.0f} Hz")
    print(f"  Total formant observations: {len(formant_freqs)}")
    
    # Build frequency axis
    freqs = np.linspace(0, 6000, 2000)
    
    # Compute smooth envelope using Gaussian peaks
    envelope = np.zeros_like(freqs)
    for f, b in zip(formant_freqs, bandwidths):
        # Convert bandwidth (FWHM) to standard deviation
        sigma = b / 2.355
        envelope += np.exp(-0.5 * ((freqs - f) / sigma) ** 2)
    
    # Normalize
    envelope /= envelope.max()
    
    return freqs, envelope, stats

def plot_smooth_envelope_comparison(freqs1, envelope1, stats1,
                                   freqs2, envelope2, stats2,
                                   label1="Real Speech", label2="Synthetic Speech",
                                   output_dir="./output"):
    """
    Plot smooth spectral envelope comparison.
    
    Args:
        freqs1, envelope1, stats1: Data for first file
        freqs2, envelope2, stats2: Data for second file
        label1, label2: Labels
        output_dir: Save directory
    """
    plt.figure(figsize=(12, 7))
    
    # Plot envelopes
    plt.plot(freqs1, envelope1, color=COLOR_FILE1, linewidth=2.5, 
             alpha=0.9, label=label1)
    plt.plot(freqs2, envelope2, color=COLOR_FILE2, linewidth=2.5, 
             alpha=0.9, label=label2)
    
    plt.title("Smooth Average Spectral Envelope Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Normalized Amplitude", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0, 6000)
    plt.tight_layout()
    
    # Save with timestamp
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = Path(output_dir) / f"smooth_envelope_{timestamp}.png"
    plt.savefig(output_name, dpi=300)
    print(f"\n[✅] Smooth envelope plot saved: {output_name}")
    
    plt.show()

def plot_envelope_difference(freqs1, envelope1, envelope2, 
                            label1="Real Speech", label2="Synthetic Speech",
                            output_dir="./output"):
    """
    Plot the difference between two envelopes.
    
    Args:
        freqs1: Frequency array
        envelope1, envelope2: Envelope arrays
        label1, label2: Labels
        output_dir: Save directory
    """
    difference = envelope1 - envelope2
    
    plt.figure(figsize=(12, 6))
    
    # Plot difference
    plt.plot(freqs1, difference, color='#2ca02c', linewidth=2)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    plt.fill_between(freqs1, 0, difference, where=(difference > 0), 
                     alpha=0.3, color=COLOR_FILE1, label=f'{label1} higher')
    plt.fill_between(freqs1, 0, difference, where=(difference < 0), 
                     alpha=0.3, color=COLOR_FILE2, label=f'{label2} higher')
    
    plt.title("Spectral Envelope Difference", fontsize=16, fontweight='bold')
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Amplitude Difference", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0, 6000)
    plt.tight_layout()
    
    # Save with timestamp
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = Path(output_dir) / f"envelope_difference_{timestamp}.png"
    plt.savefig(output_name, dpi=300)
    print(f"[✅] Difference plot saved: {output_name}")
    
    plt.show()

def print_comparison_stats(stats1, stats2, label1="Real Speech", label2="Synthetic Speech"):
    """
    Print comparison statistics.
    
    Args:
        stats1, stats2: Statistics dictionaries
        label1, label2: Labels
    """
    print(f"\n{'='*60}")
    print(f"SPECTRAL ENVELOPE COMPARISON")
    print(f"{'='*60}")
    print(f"\n{label1}:")
    print(f"  Average formant frequency: {stats1['f_mean']:.0f} Hz")
    print(f"  Average bandwidth: {stats1['b_mean']:.0f} Hz")
    print(f"  Total observations: {stats1['n_formants']}")
    
    print(f"\n{label2}:")
    print(f"  Average formant frequency: {stats2['f_mean']:.0f} Hz")
    print(f"  Average bandwidth: {stats2['b_mean']:.0f} Hz")
    print(f"  Total observations: {stats2['n_formants']}")
    
    print(f"\nDifferences:")
    print(f"  Formant frequency: {stats2['f_mean'] - stats1['f_mean']:+.0f} Hz")
    print(f"  Bandwidth: {stats2['b_mean'] - stats1['b_mean']:+.0f} Hz")

# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser(
        description="Smooth Spectral Envelope Comparison Tool",
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
    parser.add_argument("--output-dir", type=str, default="./formant_output",
                       help="Directory to save plots")
    parser.add_argument("--show-difference", action='store_true',
                       help="Also plot the envelope difference")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SMOOTH SPECTRAL ENVELOPE COMPARISON TOOL")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    
    # Load and compute envelopes
    freqs1, envelope1, stats1 = load_and_compute_smooth_envelope(args.file1)
    freqs2, envelope2, stats2 = load_and_compute_smooth_envelope(args.file2)
    
    # Print comparison
    print_comparison_stats(stats1, stats2, args.label1, args.label2)
    
    # Generate plots
    plot_smooth_envelope_comparison(freqs1, envelope1, stats1,
                                   freqs2, envelope2, stats2,
                                   args.label1, args.label2, args.output_dir)
    
    if args.show_difference:
        plot_envelope_difference(freqs1, envelope1, envelope2,
                                args.label1, args.label2, args.output_dir)
    
    print("\n" + "="*60)
    print("SMOOTH ENVELOPE ANALYSIS COMPLETE")
    print("="*60)
    print(f"All plots saved in: {args.output_dir}")

if __name__ == "__main__":
    main()