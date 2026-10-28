#!/usr/bin/env python3
"""
Spectral Envelope Comparison Tool
---------------------------------
Computes and compares average spectral envelopes between two speech samples
based on formant frequencies and bandwidths.

Usage: (default name: Real Speech for file1 and Synthetic Speech for file2; use label1 and label2 args to change)
    python spectral_envelope_comparison.py --file1 real.csv --file2 synthetic.csv
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

def load_and_compute_envelope(filepath):
    """
    Load formant data and compute spectral envelope.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        tuple: (freqs array, envelope array, formants dict)
    """
    print(f"\n[INFO] Loading: {filepath}")
    df = pd.read_csv(filepath)
    
    # Convert to numeric
    for col in ["F1(Hz)", "B1(Hz)", "F2(Hz)", "B2(Hz)", "F3(Hz)", "B3(Hz)", "F4(Hz)", "B4(Hz)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Compute average formants and bandwidths
    formants = {}
    for i in range(1, 5):
        f_col = f"F{i}(Hz)"
        b_col = f"B{i}(Hz)"
        f_mean = df[f_col].mean()
        b_mean = df[b_col].mean()
        if not np.isnan(f_mean) and not np.isnan(b_mean):
            formants[f"F{i}"] = (f_mean, b_mean)
    
    if not formants:
        raise ValueError(f"No valid formant data in {filepath}")
    
    print(f"  Average formants:")
    for k, (f, b) in formants.items():
        print(f"    {k}: {f:.0f} Hz (bandwidth: {b:.0f} Hz)")
    
    # Build frequency axis
    freqs = np.linspace(0, 6000, 2000)
    
    # Compute spectral envelope using Lorentzian model
    envelope = np.zeros_like(freqs)
    for f, b in formants.values():
        envelope += 1 / (1 + ((freqs - f) / (b / 2)) ** 2)
    
    # Normalize
    envelope /= envelope.max()
    
    return freqs, envelope, formants

def plot_envelope_comparison(freqs1, envelope1, formants1, 
                            freqs2, envelope2, formants2,
                            label1="Real Speech", label2="Synthetic Speech",
                            output_dir="./output"):
    """
    Plot spectral envelope comparison.
    
    Args:
        freqs1, envelope1, formants1: Data for first file
        freqs2, envelope2, formants2: Data for second file
        label1, label2: Labels
        output_dir: Save directory
    """
    plt.figure(figsize=(12, 7))
    
    # Plot envelopes
    plt.plot(freqs1, envelope1, color=COLOR_FILE1, linewidth=2.5, 
             alpha=0.9, label=label1)
    plt.plot(freqs2, envelope2, color=COLOR_FILE2, linewidth=2.5, 
             alpha=0.9, label=label2)
    
    # Mark formant peaks for first file
    for label, (f, _) in formants1.items():
        plt.axvline(f, color=COLOR_FILE1, linestyle=':', alpha=0.3, linewidth=1)
    
    # Mark formant peaks for second file
    for label, (f, _) in formants2.items():
        plt.axvline(f, color=COLOR_FILE2, linestyle=':', alpha=0.3, linewidth=1)
    
    plt.title("Average Spectral Envelope Comparison", fontsize=16, fontweight='bold')
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Normalized Amplitude", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.xlim(0, 6000)
    plt.tight_layout()
    
    # Save with timestamp
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = Path(output_dir) / f"spectral_envelope_{timestamp}.png"
    plt.savefig(output_name, dpi=300)
    print(f"\n[✅] Spectral envelope plot saved: {output_name}")
    
    plt.show()

def plot_formant_comparison_table(formants1, formants2, label1="Real Speech", label2="Synthetic Speech", output_dir="./output"):
    """
    Create a comparison table of formant frequencies.
    
    Args:
        formants1, formants2: Formant dictionaries
        label1, label2: Labels
        output_dir: Save directory
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('tight')
    ax.axis('off')
    
    # Build table data
    table_data = [["Formant", f"{label1} (Hz)", f"{label2} (Hz)", "Difference (Hz)"]]
    
    for i in range(1, 5):
        key = f"F{i}"
        if key in formants1 and key in formants2:
            f1_freq = formants1[key][0]
            f2_freq = formants2[key][0]
            diff = f2_freq - f1_freq
            table_data.append([
                key,
                f"{f1_freq:.0f}",
                f"{f2_freq:.0f}",
                f"{diff:+.0f}"
            ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2, 0.3, 0.3, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title("Formant Frequency Comparison", fontsize=14, fontweight='bold', pad=20)
    
    # Save with timestamp
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = Path(output_dir) / f"formant_table_{timestamp}.png"
    plt.savefig(output_name, dpi=300, bbox_inches='tight')
    print(f"[✅] Formant table saved: {output_name}")
    
    plt.show()

def print_comparison_stats(formants1, formants2, label1="Real Speech", label2="Synthetic Speech"):
    """
    Print comparison statistics.
    
    Args:
        formants1, formants2: Formant dictionaries
        label1, label2: Labels
    """
    print(f"\n{'='*60}")
    print(f"FORMANT COMPARISON: {label1} vs {label2}")
    print(f"{'='*60}")
    
    for i in range(1, 5):
        key = f"F{i}"
        if key in formants1 and key in formants2:
            f1_freq, f1_bw = formants1[key]
            f2_freq, f2_bw = formants2[key]
            freq_diff = f2_freq - f1_freq
            bw_diff = f2_bw - f1_bw
            
            print(f"\n{key}:")
            print(f"  {label1}: {f1_freq:.0f} Hz (BW: {f1_bw:.0f} Hz)")
            print(f"  {label2}: {f2_freq:.0f} Hz (BW: {f2_bw:.0f} Hz)")
            print(f"  Difference: {freq_diff:+.0f} Hz (BW: {bw_diff:+.0f} Hz)")

# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser(
        description="Spectral Envelope Comparison Tool",
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
    parser.add_argument("--no-table", action='store_true',
                       help="Skip formant comparison table")
    
    args = parser.parse_args()
    
    print("="*60)
    print("SPECTRAL ENVELOPE COMPARISON TOOL")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    
    # Load and compute envelopes
    freqs1, envelope1, formants1 = load_and_compute_envelope(args.file1)
    freqs2, envelope2, formants2 = load_and_compute_envelope(args.file2)
    
    # Print comparison
    print_comparison_stats(formants1, formants2, args.label1, args.label2)
    
    # Generate plots
    plot_envelope_comparison(freqs1, envelope1, formants1,
                            freqs2, envelope2, formants2,
                            args.label1, args.label2, args.output_dir)
    
    if not args.no_table:
        plot_formant_comparison_table(formants1, formants2, 
                                     args.label1, args.label2, args.output_dir)
    
    print("\n" + "="*60)
    print("SPECTRAL ENVELOPE ANALYSIS COMPLETE")
    print("="*60)
    print(f"All plots saved in: {args.output_dir}")

if __name__ == "__main__":
    main()