#!/usr/bin/env python3
"""
Formant Ratio Analysis Tool
---------------------------
This script loads a Praat-exported formant CSV (comma-separated),
computes a user-selected formant ratio (e.g., F2/F1), and plots
it over time with clean, publication-ready visuals.

Usage:
    python formant_ratio_plot.py --file path/to/file.csv --ratio F2/F1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# ----------------- CLI arguments -----------------
parser = argparse.ArgumentParser(description="Formant Ratio Plotter")
parser.add_argument("--file", type=str, required=True, help="Path to Praat CSV file")
parser.add_argument(
    "--ratio",
    type=str,
    default="F2/F1",
    choices=["F2/F1", "F3/F2", "F4/F3", "F3/F1", "F4/F2"],
    help="Choose formant ratio to compute",
)
args = parser.parse_args()

# ----------------- Load data -----------------
df = pd.read_csv(args.file)

# Replace '--undefined--' with NaN and convert numeric columns
for col in ["F1(Hz)", "F2(Hz)", "F3(Hz)", "F4(Hz)"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------- Compute ratio -----------------
ratio_map = {
    "F2/F1": ("F2(Hz)", "F1(Hz)"),
    "F3/F2": ("F3(Hz)", "F2(Hz)"),
    "F4/F3": ("F4(Hz)", "F3(Hz)"),
    "F3/F1": ("F3(Hz)", "F1(Hz)"),
    "F4/F2": ("F4(Hz)", "F2(Hz)")
}

numerator, denominator = ratio_map[args.ratio]
df[args.ratio] = df[numerator] / df[denominator]

# ----------------- Plot -----------------
plt.figure(figsize=(10, 6))
plt.plot(df["time(s)"], df[args.ratio], linewidth=2, color="#1f77b4", alpha=0.85)
plt.title(f"Formant Ratio Over Time: {args.ratio}", fontsize=16)
plt.xlabel("Time (s)", fontsize=14)
plt.ylabel(f"{args.ratio} Value", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# ----------------- Save & Show -----------------
output_name = f"formant_ratio_{args.ratio.replace('/', '_')}.png"
plt.savefig(output_name, dpi=300)
print(f"[✅] Plot saved as {output_name}")
plt.show()
