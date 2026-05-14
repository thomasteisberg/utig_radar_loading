#!/usr/bin/env python3
"""
Analyze UTIG data directories and file statistics.

This script searches for directories matching the pattern:
/kucresis/scratch/data/UTIG/UTIG2/targ/pcor/<prj>/<set>/<trn>/<stream>/

For each directory, it collects statistics on ct.gz, ct, xds.gz, and bxds files.
"""

import pandas as pd
import subprocess
import os
from pathlib import Path
import glob
import sys

def run_command(cmd):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return None
    except Exception:
        return None

def count_lines(file_path):
    """Count lines in a file (handles .gz files automatically)."""
    if not os.path.exists(file_path):
        return None

    if file_path.endswith('.gz'):
        cmd = f"zcat '{file_path}' | wc -l"
    else:
        cmd = f"cat '{file_path}' | wc -l"

    result = run_command(cmd)
    if result:
        try:
            return int(result)
        except ValueError:
            return None
    return None

def get_file_size(file_path):
    """Get file size in bytes."""
    try:
        return os.path.getsize(file_path)
    except OSError:
        return None

def analyze_directory(dir_path):
    """Analyze a single directory for the required files."""
    dir_path = Path(dir_path)

    # Extract components from path
    parts = dir_path.parts
    if len(parts) < 4:
        return None

    # Find indices for the key path components
    try:
        pcor_idx = parts.index('pcor')
        if pcor_idx + 4 >= len(parts):
            return None
        prj = parts[pcor_idx + 1]
        set_name = parts[pcor_idx + 2]
        trn = parts[pcor_idx + 3]
        stream = parts[pcor_idx + 4]
    except (ValueError, IndexError):
        return None

    # Check for each file type and collect statistics
    stats = {
        'directory': str(dir_path),
        'prj': prj,
        'set': set_name,
        'trn': trn,
        'stream': stream,
        'ct_gz_lines': None,
        'ct_lines': None,
        'xds_gz_lines': None,
        'bxds_bytes': None
    }

    # Check each file
    ct_gz_path = dir_path / 'ct.gz'
    if ct_gz_path.exists():
        stats['ct_gz_lines'] = count_lines(str(ct_gz_path))

    ct_path = dir_path / 'ct'
    if ct_path.exists():
        stats['ct_lines'] = count_lines(str(ct_path))

    xds_gz_path = dir_path / 'xds.gz'
    if xds_gz_path.exists():
        stats['xds_gz_lines'] = count_lines(str(xds_gz_path))

    bxds_path = dir_path / 'bxds'
    if bxds_path.exists():
        stats['bxds_bytes'] = get_file_size(str(bxds_path))

    return stats

def main():
    print("Searching for UTIG data directories...")

    # Find all directories matching the pattern
    pattern = "/kucresis/scratch/data/UTIG/UTIG2/targ/pcor/*/*/*/*"
    directories = glob.glob(pattern)

    # Filter to only include directories (not files)
    directories = [d for d in directories if os.path.isdir(d)]

    print(f"Found {len(directories)} directories to analyze")

    # Debug: show first few directories
    if directories:
        print("Sample directories:")
        for i, d in enumerate(directories[:5]):
            print(f"  {i+1}: {d}")
        if len(directories) > 5:
            print(f"  ... and {len(directories) - 5} more")

    # Analyze each directory
    results = []
    for i, directory in enumerate(directories):
        if i % 50 == 0:
            print(f"Processing directory {i+1}/{len(directories)}: {Path(directory).name}")

        stats = analyze_directory(directory)
        if stats:
            results.append(stats)

        # Optional: limit for quick testing
        # if i >= 100:
        #     print(f"Stopping early after {i+1} directories for testing...")
        #     break

    if not results:
        print("No valid directories found")
        return

    # Create DataFrame
    df = pd.DataFrame(results)

    print(f"\nAnalyzed {len(df)} directories")
    print(f"Unique streams: {df['stream'].nunique()}")

    # Calculate bytes per line ratio
    # Use ct.gz lines if available, otherwise ct lines
    df['ct_total_lines'] = df['ct_gz_lines'].fillna(df['ct_lines'])

    # Calculate ratio where both bxds_bytes and ct_total_lines exist
    mask = (df['bxds_bytes'].notna()) & (df['ct_total_lines'].notna()) & (df['ct_total_lines'] > 0)
    df.loc[mask, 'bytes_per_line'] = df.loc[mask, 'bxds_bytes'] / df.loc[mask, 'ct_total_lines']

    print(f"\nCalculated bytes_per_line for {mask.sum()} directories")

    # Report median and extreme values by stream
    if 'bytes_per_line' in df.columns and df['bytes_per_line'].notna().any():
        print("\n" + "="*60)
        print("BYTES PER LINE STATISTICS BY STREAM")
        print("="*60)

        stream_stats = df.groupby('stream')['bytes_per_line'].agg([
            'count', 'median', 'min', 'max'
        ]).round(2)

        stream_stats.columns = ['Count', 'Median', 'Min', 'Max']
        print(stream_stats)

    # Identify mismatched line counts
    print("\n" + "="*60)
    print("CHECKING FOR MISMATCHED LINE COUNTS")
    print("="*60)

    mismatches = []
    for idx, row in df.iterrows():
        line_counts = []
        file_types = []

        if pd.notna(row['ct_gz_lines']):
            line_counts.append(row['ct_gz_lines'])
            file_types.append('ct.gz')

        if pd.notna(row['ct_lines']):
            line_counts.append(row['ct_lines'])
            file_types.append('ct')

        if pd.notna(row['xds_gz_lines']):
            line_counts.append(row['xds_gz_lines'])
            file_types.append('xds.gz')

        # Check if all line counts are the same
        if len(line_counts) > 1 and len(set(line_counts)) > 1:
            mismatch_info = {
                'directory': row['directory'],
                'stream': row['stream'],
                'file_types': ', '.join(file_types),
                'line_counts': ', '.join(map(str, line_counts))
            }
            mismatches.append(mismatch_info)

    if mismatches:
        print(f"Found {len(mismatches)} directories with mismatched line counts:")
        mismatch_df = pd.DataFrame(mismatches)
        print(mismatch_df.to_string(index=False))
    else:
        print("No directories with mismatched line counts found")

    # Save results to CSV
    output_file = "utig_file_analysis.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    # Display summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total directories analyzed: {len(df)}")
    print(f"Directories with ct.gz: {df['ct_gz_lines'].notna().sum()}")
    print(f"Directories with ct: {df['ct_lines'].notna().sum()}")
    print(f"Directories with xds.gz: {df['xds_gz_lines'].notna().sum()}")
    print(f"Directories with bxds: {df['bxds_bytes'].notna().sum()}")
    print(f"Directories with calculable bytes_per_line: {df['bytes_per_line'].notna().sum()}")
    print(f"Directories with mismatched line counts: {len(mismatches)}")

if __name__ == "__main__":
    main()