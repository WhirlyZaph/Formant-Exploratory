#!/usr/bin/env python3
"""
Enhanced Formant Ratio Analysis Tool
-------------------------------------
This script loads Praat-exported formant CSV files, applies quality filtering,
computes formant ratios, and generates publication-ready visualizations.

Features:
- Quality filtering based on physiological constraints
- Formant relationship validation
- Statistical analysis and outlier detection
- Comparison mode for real vs synthetic speech
- Multiple visualization options

Usage:
    Single file: python formant_ratio_analyzer.py --file sample.csv --ratio F2/F1
    Comparison:  python formant_ratio_analyzer.py --file real.csv --compare synthetic.csv --ratio F2/F1
    All ratios:  python formant_ratio_analyzer.py --file sample.csv --all-ratios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
from pathlib import Path

# ======================== CONFIGURATION ========================

# Formant frequency valid ranges (Hz) - based on adult American English
FORMANT_RANGES = {
    'F1': (200, 1200),   # First formant: jaw opening/tongue height
    'F2': (600, 3500),   # Second formant: tongue front/back position
    'F3': (1400, 4500),  # Third formant: lip rounding + tongue tip
    'F4': (2500, 5500)   # Fourth formant: fine articulatory details
}

# Formant ratio valid ranges - based on vowel space characteristics
RATIO_RANGES = {
    'F2/F1': (1.2, 5.0),   # Back vowels ~1.5-2.0, front vowels ~3.5-4.5
    'F3/F2': (1.1, 2.5),   # Typical formant spacing
    'F4/F3': (1.05, 2.0),  # Higher formants more closely spaced
    'F3/F1': (2.0, 8.0),   # Extended ratio for analysis
    'F4/F2': (1.5, 4.0)    # Extended ratio for analysis
}

# Statistical outlier threshold (number of standard deviations)
OUTLIER_THRESHOLD = 3.0

# ======================== UTILITY FUNCTIONS ========================

def load_and_clean_csv(filepath):
    """
    Load Praat CSV and convert undefined values to NaN.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with cleaned data
    """
    print(f"\n[INFO] Loading CSV: {filepath}")
    df = pd.read_csv(filepath)
    
    # Expected formant columns
    formant_cols = ['F1(Hz)', 'F2(Hz)', 'F3(Hz)', 'F4(Hz)']
    
    # Check if required columns exist
    missing_cols = [col for col in formant_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert '--undefined--' strings to NaN and ensure numeric type
    for col in formant_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Rename columns for easier access
    df.rename(columns={
        'F1(Hz)': 'F1',
        'F2(Hz)': 'F2',
        'F3(Hz)': 'F3',
        'F4(Hz)': 'F4'
    }, inplace=True)
    
    total_frames = len(df)
    print(f"[INFO] Total frames loaded: {total_frames}")
    
    return df

def filter_formant_frequencies(df):
    """
    Apply physiological constraints to filter unrealistic formant values.
    
    Args:
        df: DataFrame with formant columns
        
    Returns:
        DataFrame with 'valid_formants' boolean column
    """
    print("\n[INFO] Applying formant frequency filters...")
    
    # Initialize validity flags
    df['valid_F1'] = True
    df['valid_F2'] = True
    df['valid_F3'] = True
    df['valid_F4'] = True
    
    # Apply range filters for each formant
    for formant, (min_hz, max_hz) in FORMANT_RANGES.items():
        invalid_low = df[formant] < min_hz
        invalid_high = df[formant] > max_hz
        invalid_count = invalid_low.sum() + invalid_high.sum()
        
        df[f'valid_{formant}'] = ~(invalid_low | invalid_high | df[formant].isna())
        
        print(f"  {formant}: Valid range [{min_hz}-{max_hz} Hz] - Removed {invalid_count} frames")
    
    return df

def validate_formant_relationships(df):
    """
    Ensure formants maintain proper ordering: F1 < F2 < F3 < F4.
    
    Args:
        df: DataFrame with formant columns and validity flags
        
    Returns:
        DataFrame with updated validity flags
    """
    print("\n[INFO] Validating formant relationships (F1 < F2 < F3 < F4)...")
    
    # Check ordering for frames where all formants are individually valid
    all_valid_mask = df['valid_F1'] & df['valid_F2'] & df['valid_F3'] & df['valid_F4']
    
    # Check proper ordering
    proper_order = (
        (df['F1'] < df['F2']) &
        (df['F2'] < df['F3']) &
        (df['F3'] < df['F4'])
    )
    
    # Update validity: must have proper ordering AND individual validity
    ordering_violations = all_valid_mask & ~proper_order
    violation_count = ordering_violations.sum()
    
    print(f"  Found {violation_count} frames with ordering violations")
    
    # Mark frames with ordering violations as invalid
    df.loc[ordering_violations, ['valid_F1', 'valid_F2', 'valid_F3', 'valid_F4']] = False
    
    # Create overall validity flag
    df['valid_formants'] = df['valid_F1'] & df['valid_F2'] & df['valid_F3'] & df['valid_F4']
    
    valid_count = df['valid_formants'].sum()
    valid_percent = (valid_count / len(df)) * 100
    print(f"  Overall valid frames: {valid_count}/{len(df)} ({valid_percent:.1f}%)")
    
    if valid_percent < 50:
        print("  [WARNING] Less than 50% valid frames - check audio quality or Praat settings")
    
    return df

def compute_ratios(df):
    """
    Calculate formant ratios for valid frames.
    
    Args:
        df: DataFrame with validated formant data
        
    Returns:
        DataFrame with ratio columns
    """
    print("\n[INFO] Computing formant ratios...")
    
    # Initialize ratio columns
    for ratio_name in RATIO_RANGES.keys():
        df[ratio_name] = np.nan
    
    # Only compute ratios for valid frames
    valid_mask = df['valid_formants']
    
    # Compute each ratio
    df.loc[valid_mask, 'F2/F1'] = df.loc[valid_mask, 'F2'] / df.loc[valid_mask, 'F1']
    df.loc[valid_mask, 'F3/F2'] = df.loc[valid_mask, 'F3'] / df.loc[valid_mask, 'F2']
    df.loc[valid_mask, 'F4/F3'] = df.loc[valid_mask, 'F4'] / df.loc[valid_mask, 'F3']
    df.loc[valid_mask, 'F3/F1'] = df.loc[valid_mask, 'F3'] / df.loc[valid_mask, 'F1']
    df.loc[valid_mask, 'F4/F2'] = df.loc[valid_mask, 'F4'] / df.loc[valid_mask, 'F2']
    
    return df

def filter_ratio_ranges(df, ratio_name):
    """
    Filter ratios to physiologically plausible ranges.
    
    Args:
        df: DataFrame with ratio column
        ratio_name: Name of the ratio to filter
        
    Returns:
        DataFrame with valid_ratio boolean column
    """
    if ratio_name not in RATIO_RANGES:
        raise ValueError(f"Unknown ratio: {ratio_name}")
    
    min_val, max_val = RATIO_RANGES[ratio_name]
    
    # Create validity mask
    valid_mask = (
        (df[ratio_name] >= min_val) & 
        (df[ratio_name] <= max_val) &
        ~df[ratio_name].isna()
    )
    
    df['valid_ratio'] = valid_mask
    
    removed = (~valid_mask).sum()
    print(f"  {ratio_name}: Valid range [{min_val:.1f}-{max_val:.1f}] - Removed {removed} outliers")
    
    return df

def detect_statistical_outliers(df, ratio_name):
    """
    Detect statistical outliers using mean ± 3σ threshold.
    
    Args:
        df: DataFrame with ratio column
        ratio_name: Name of the ratio
        
    Returns:
        DataFrame with statistical_outlier boolean column
    """
    valid_ratios = df.loc[df['valid_ratio'], ratio_name]
    
    if len(valid_ratios) < 10:
        print(f"  [WARNING] Too few valid points for outlier detection")
        df['statistical_outlier'] = False
        return df
    
    mean = valid_ratios.mean()
    std = valid_ratios.std()
    
    lower_bound = mean - (OUTLIER_THRESHOLD * std)
    upper_bound = mean + (OUTLIER_THRESHOLD * std)
    
    df['statistical_outlier'] = (
        (df[ratio_name] < lower_bound) | 
        (df[ratio_name] > upper_bound)
    ) & df['valid_ratio']
    
    outlier_count = df['statistical_outlier'].sum()
    print(f"  Statistical outliers (±{OUTLIER_THRESHOLD}σ): {outlier_count} frames")
    print(f"  Mean: {mean:.2f}, Std: {std:.2f}, Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    return df

def print_statistics(df, ratio_name, label=""):
    """
    Print descriptive statistics for a ratio.
    
    Args:
        df: DataFrame with ratio data
        ratio_name: Name of the ratio
        label: Optional label for output
    """
    valid_data = df.loc[df['valid_ratio'] & ~df['statistical_outlier'], ratio_name]
    
    if len(valid_data) == 0:
        print(f"\n[{label}] No valid data for {ratio_name}")
        return
    
    print(f"\n{'='*60}")
    print(f"Statistics for {ratio_name} {label}")
    print(f"{'='*60}")
    print(f"  Count:        {len(valid_data)}")
    print(f"  Mean:         {valid_data.mean():.3f}")
    print(f"  Median:       {valid_data.median():.3f}")
    print(f"  Std Dev:      {valid_data.std():.3f}")
    print(f"  Min:          {valid_data.min():.3f}")
    print(f"  Max:          {valid_data.max():.3f}")
    print(f"  Range:        {valid_data.max() - valid_data.min():.3f}")
    print(f"  25th %ile:    {valid_data.quantile(0.25):.3f}")
    print(f"  75th %ile:    {valid_data.quantile(0.75):.3f}")

# ======================== VISUALIZATION FUNCTIONS ========================

def plot_ratio_over_time(df, ratio_name, output_dir, label="", color='#1f77b4', style='-'):
    """
    Create line plot of ratio over time.
    
    Args:
        df: DataFrame with ratio data
        ratio_name: Ratio to plot
        output_dir: Directory for saving plots
        label: Label for legend
        color: Line color
        style: Line style
        
    Returns:
        Figure and axis objects
    """
    valid_data = df[df['valid_ratio'] & ~df['statistical_outlier']]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(valid_data['time(s)'], valid_data[ratio_name], 
            linewidth=2, color=color, linestyle=style, alpha=0.85, label=label)
    
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel(f'{ratio_name} Value', fontsize=14)
    ax.set_title(f'Formant Ratio Over Time: {ratio_name}', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if label:
        ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    return fig, ax

def plot_ratio_distribution(df, ratio_name, output_dir, label="", color='#1f77b4', alpha=0.7):
    """
    Create histogram of ratio distribution.
    
    Args:
        df: DataFrame with ratio data
        ratio_name: Ratio to plot
        output_dir: Directory for saving plots
        label: Label for legend
        color: Bar color
        alpha: Transparency
        
    Returns:
        Figure and axis objects
    """
    valid_data = df.loc[df['valid_ratio'] & ~df['statistical_outlier'], ratio_name]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(valid_data, bins=50, color=color, alpha=alpha, edgecolor='black', label=label, density=True)
    
    ax.set_xlabel(f'{ratio_name} Value', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title(f'Distribution of {ratio_name}', fontsize=16, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    if label:
        ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    return fig, ax

def plot_comparison(df1, df2, ratio_name, output_dir, label1="Real", label2="Synthetic"):
    """
    Create comparison plots for two datasets.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        ratio_name: Ratio to compare
        output_dir: Directory for saving plots
        label1: Label for first dataset
        label2: Label for second dataset
    """
    print(f"\n[INFO] Creating comparison plots for {ratio_name}...")
    
    # Time series comparison
    fig, ax = plot_ratio_over_time(df1, ratio_name, output_dir, label=label1, color='#1f77b4', style='-')
    
    valid_data2 = df2[df2['valid_ratio'] & ~df2['statistical_outlier']]
    ax.plot(valid_data2['time(s)'], valid_data2[ratio_name], 
            linewidth=2, color='#d62728', linestyle='--', alpha=0.85, label=label2)
    
    ax.legend(fontsize=12)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"comparison_timeseries_{ratio_name.replace('/', '_')}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()
    
    # Distribution comparison
    fig, ax = plot_ratio_distribution(df1, ratio_name, output_dir, label=label1, color='#1f77b4', alpha=0.6)
    
    valid_data2 = df2.loc[df2['valid_ratio'] & ~df2['statistical_outlier'], ratio_name]
    ax.hist(valid_data2, bins=50, color='#d62728', alpha=0.6, edgecolor='black', label=label2, density=True)
    
    ax.legend(fontsize=12)
    
    filename = output_dir / f"comparison_distribution_{ratio_name.replace('/', '_')}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Saved: {filename}")
    plt.close()

# ======================== MAIN PROCESSING ========================

def process_file(filepath, ratio_name):
    """
    Complete processing pipeline for a single file.
    
    Args:
        filepath: Path to CSV file
        ratio_name: Ratio to analyze
        
    Returns:
        Processed DataFrame
    """
    # Load and clean
    df = load_and_clean_csv(filepath)
    
    # Filter formants
    df = filter_formant_frequencies(df)
    df = validate_formant_relationships(df)
    
    # Compute ratios
    df = compute_ratios(df)
    
    # Filter specific ratio
    df = filter_ratio_ranges(df, ratio_name)
    df = detect_statistical_outliers(df, ratio_name)
    
    # Print statistics
    print_statistics(df, ratio_name)
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Formant Ratio Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--file", type=str, required=True, 
                       help="Path to primary Praat CSV file")
    parser.add_argument("--compare", type=str, default=None,
                       help="Path to second CSV file for comparison")
    parser.add_argument("--ratio", type=str, default="F2/F1",
                       choices=list(RATIO_RANGES.keys()),
                       help="Formant ratio to analyze")
    parser.add_argument("--all-ratios", action='store_true',
                       help="Generate plots for all available ratios")
    parser.add_argument("--output-dir", type=str, default="./formant_plots",
                       help="Directory for output plots")
    parser.add_argument("--export-csv", action='store_true',
                       help="Export processed data to CSV")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("ENHANCED FORMANT RATIO ANALYSIS TOOL")
    print("="*60)
    
    # Determine which ratios to analyze
    ratios_to_analyze = list(RATIO_RANGES.keys()) if args.all_ratios else [args.ratio]
    
    # Process primary file
    for ratio in ratios_to_analyze:
        print(f"\n{'#'*60}")
        print(f"# Processing Ratio: {ratio}")
        print(f"{'#'*60}")
        
        df1 = process_file(args.file, ratio)
        
        # Generate plots
        fig, ax = plot_ratio_over_time(df1, ratio, output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"ratio_timeseries_{ratio.replace('/', '_')}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n[✅] Plot saved: {filename}")
        plt.close()
        
        fig, ax = plot_ratio_distribution(df1, ratio, output_dir)
        filename = output_dir / f"ratio_distribution_{ratio.replace('/', '_')}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[✅] Plot saved: {filename}")
        plt.close()
        
        # Comparison mode
        if args.compare:
            print(f"\n{'='*60}")
            print(f"COMPARISON MODE: {ratio}")
            print(f"{'='*60}")
            df2 = process_file(args.compare, ratio)
            plot_comparison(df1, df2, ratio, output_dir)
            
            # Print comparative statistics
            print("\n" + "="*60)
            print("COMPARATIVE ANALYSIS")
            print("="*60)
            
            mean1 = df1.loc[df1['valid_ratio'] & ~df1['statistical_outlier'], ratio].mean()
            mean2 = df2.loc[df2['valid_ratio'] & ~df2['statistical_outlier'], ratio].mean()
            
            print(f"  Mean difference: {abs(mean1 - mean2):.3f}")
            print(f"  Real speech mean: {mean1:.3f}")
            print(f"  Synthetic speech mean: {mean2:.3f}")
        
        # Export processed CSV
        if args.export_csv:
            csv_filename = output_dir / f"processed_{ratio.replace('/', '_')}_{timestamp}.csv"
            df1.to_csv(csv_filename, index=False)
            print(f"\n[✅] Processed data exported: {csv_filename}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"All outputs saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()