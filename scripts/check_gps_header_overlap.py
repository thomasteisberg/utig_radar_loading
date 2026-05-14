"""
Check time overlap between header files and GPS support files using segment information.

Usage:
    uv run python check_gps_header_overlap.py <header_dir> <gps_dir> <records_csv>

Example:
    uv run python check_gps_header_overlap.py \
        /kucresis/scratch/tteisberg_sta/scripts/opr_user_tmp/headers/rds/2008_Antarctica_BaslerJKB \
        /kucresis/scratch/tteisberg_sta/scripts/python/utig_radar_loading/outputs/gps/2008_Antarctica_BaslerJKB \
        /kucresis/scratch/tteisberg_sta/scripts/python/utig_radar_loading/outputs/params/2008_Antarctica_BaslerJKB/records.csv
"""

import sys
import ast
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import hdf5storage
import numpy as np
import pandas as pd


def load_time_range_radar(mat_file: Path) -> Tuple[Optional[float], Optional[float]]:
    """Load radar_time range from a .mat file."""
    try:
        data = hdf5storage.loadmat(str(mat_file))
        if 'radar_time' not in data:
            return None, None
        time_array = np.array(data['radar_time']).flatten()
        if len(time_array) == 0:
            return None, None
        return float(time_array.min()), float(time_array.max())
    except Exception as e:
        print(f"Warning: Could not load {mat_file}: {e}", file=sys.stderr)
        return None, None


def load_radar_time_array(mat_file: Path) -> Optional[np.ndarray]:
    """Load full radar_time array from a .mat file."""
    try:
        data = hdf5storage.loadmat(str(mat_file))
        if 'radar_time' not in data:
            return None
        time_array = np.array(data['radar_time']).flatten()
        return time_array
    except Exception as e:
        print(f"Warning: Could not load {mat_file}: {e}", file=sys.stderr)
        return None


def find_valid_trace_ranges(radar_times: np.ndarray, gps_start: float, gps_end: float) -> List[Tuple[int, int]]:
    """Find ranges of trace indices that fall within GPS coverage."""
    # Find indices where radar_time is within GPS range
    valid_mask = (radar_times >= gps_start) & (radar_times <= gps_end)
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return []

    # Group consecutive indices into ranges
    ranges = []
    start_idx = valid_indices[0]
    prev_idx = valid_indices[0]

    for idx in valid_indices[1:]:
        if idx != prev_idx + 1:
            # Gap found, close current range
            ranges.append((int(start_idx), int(prev_idx)))
            start_idx = idx
        prev_idx = idx

    # Close final range
    ranges.append((int(start_idx), int(prev_idx)))

    return ranges


def format_radar_time(radar_time: float) -> str:
    """Format radar_time (10us ticks) as seconds."""
    seconds = radar_time / 100000  # Convert 10us ticks to seconds
    return f"{seconds:.2f}s"


def find_gaps(hdr_ranges: List[Tuple[float, float]], gps_range: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Find gaps in header ranges not covered by GPS range."""
    gps_start, gps_end = gps_range

    # Collect all header time points covered by GPS
    covered_ranges = []
    for hdr_start, hdr_end in hdr_ranges:
        # Clip header range to GPS range
        overlap_start = max(hdr_start, gps_start)
        overlap_end = min(hdr_end, gps_end)
        if overlap_start < overlap_end:
            covered_ranges.append((overlap_start, overlap_end))

    if not covered_ranges:
        # No coverage at all - entire range is a gap
        all_hdr_start = min(start for start, _ in hdr_ranges)
        all_hdr_end = max(end for _, end in hdr_ranges)
        return [(all_hdr_start, all_hdr_end)]

    # Sort covered ranges
    covered_ranges.sort()

    # Find gaps between header ranges that are not covered
    gaps = []

    # Check for gaps before first covered range
    all_hdr_start = min(start for start, _ in hdr_ranges)
    if all_hdr_start < covered_ranges[0][0]:
        gaps.append((all_hdr_start, covered_ranges[0][0]))

    # Check for gaps between covered ranges
    for i in range(len(covered_ranges) - 1):
        gap_start = covered_ranges[i][1]
        gap_end = covered_ranges[i + 1][0]

        # Check if any header range spans this gap
        has_header = any(start <= gap_start and end >= gap_end for start, end in hdr_ranges)
        if has_header and gap_end > gap_start:
            gaps.append((gap_start, gap_end))

    # Check for gaps after last covered range
    all_hdr_end = max(end for _, end in hdr_ranges)
    if all_hdr_end > covered_ranges[-1][1]:
        gaps.append((covered_ranges[-1][1], all_hdr_end))

    return gaps


def main():
    if len(sys.argv) != 4:
        print("Usage: check_gps_header_overlap.py <header_dir> <gps_dir> <records_csv>")
        sys.exit(1)

    header_dir = Path(sys.argv[1])
    gps_dir = Path(sys.argv[2])
    records_csv = Path(sys.argv[3])

    if not header_dir.exists():
        print(f"Error: Header directory not found: {header_dir}")
        sys.exit(1)

    if not gps_dir.exists():
        print(f"Error: GPS directory not found: {gps_dir}")
        sys.exit(1)

    if not records_csv.exists():
        print(f"Error: Records CSV not found: {records_csv}")
        sys.exit(1)

    # Load records CSV
    print(f"Loading segment mappings from {records_csv}...")
    df_records = pd.read_csv(records_csv)

    print(f"Found {len(df_records)} segments")
    print()

    # Process each segment
    print("=" * 120)
    total_gaps = 0
    total_segments = 0

    # Store valid ranges for each segment (for final output)
    segment_ranges = {}  # key: (segment_date_str, segment_number), value: dict of file -> ranges

    for idx, row in df_records.iterrows():
        segment_name = f"{row['segment_date_str']}_{row['segment_number']:02d}"

        # Parse board folder names (comes as string representation of set)
        board_folders_str = row['file.board_folder_name']
        try:
            board_folders = ast.literal_eval(board_folders_str)
            if isinstance(board_folders, str):
                board_folders = {board_folders}
        except:
            print(f"\nSegment {segment_name}: ERROR parsing board folders: {board_folders_str}")
            continue

        # Get GPS file path
        gps_file = Path(row['gps.fn'])

        if not gps_file.exists():
            print(f"\nSegment {segment_name}: GPS file not found: {gps_file}")
            continue

        # Load GPS time range
        gps_start, gps_end = load_time_range_radar(gps_file)
        if gps_start is None:
            print(f"\nSegment {segment_name}: Could not read GPS file: {gps_file}")
            continue

        # Find all header files for this segment
        header_files = []
        for board_folder in board_folders:
            # board_folder is like "MCM/JKB1a/WLKX01a/RADjh1"
            board_path = header_dir / board_folder

            # Find bxds*.mat files in this directory
            if board_path.exists():
                for hdr_file in board_path.glob("bxds*.mat"):
                    header_files.append(hdr_file)
            else:
                print(f"\nSegment {segment_name}: Board folder not found: {board_path}")

        if not header_files:
            print(f"\nSegment {segment_name}: No header files found")
            continue

        # Load time ranges for all header files
        header_ranges = []
        header_info = []
        header_files_map = {}  # Map file path to full file object for later use
        for hdr_file in sorted(header_files):
            hdr_start, hdr_end = load_time_range_radar(hdr_file)
            if hdr_start is not None:
                header_ranges.append((hdr_start, hdr_end))
                rel_path = hdr_file.relative_to(header_dir)
                header_info.append((rel_path, hdr_start, hdr_end))
                header_files_map[rel_path] = hdr_file

        if not header_ranges:
            print(f"\nSegment {segment_name}: Could not read any header files")
            continue

        total_segments += 1

        # Check for gaps
        gaps = find_gaps(header_ranges, (gps_start, gps_end))

        # Print report
        print(f"\nSegment: {segment_name}")
        print(f"  GPS file: {gps_file.name}")
        print(f"  GPS radar_time range: {format_radar_time(gps_start)} to {format_radar_time(gps_end)}")
        print(f"  Header files: {len(header_files)}")

        for hdr_path, hdr_start, hdr_end in header_info:
            print(f"    - {hdr_path}: {format_radar_time(hdr_start)} to {format_radar_time(hdr_end)}")

        # Calculate valid trace ranges for each header file
        segment_key = (row['segment_date_str'], row['segment_number'])
        segment_ranges[segment_key] = {}

        for hdr_path, hdr_start, hdr_end in header_info:
            radar_times = load_radar_time_array(header_files_map[hdr_path])
            if radar_times is not None:
                valid_ranges = find_valid_trace_ranges(radar_times, gps_start, gps_end)
                max_trace_idx = len(radar_times) - 1  # Zero-indexed max
                segment_ranges[segment_key][str(hdr_path)] = (valid_ranges, max_trace_idx)
            else:
                segment_ranges[segment_key][str(hdr_path)] = None

        if gaps:
            total_gaps += len(gaps)
            print(f"  GAPS: {len(gaps)} time gap(s) in header coverage")
            for gap_start, gap_end in gaps:
                print(f"    - {format_radar_time(gap_start)} to {format_radar_time(gap_end)} (duration: {format_radar_time(gap_end - gap_start)})")

            # Show valid trace ranges for each header file
            print(f"  Valid trace ranges (within GPS coverage):")
            for hdr_path, hdr_start, hdr_end in header_info:
                range_data = segment_ranges[segment_key][str(hdr_path)]
                if range_data is not None:
                    valid_ranges, max_trace_idx = range_data
                    if valid_ranges:
                        ranges_str = ", ".join([f"[{start}:{end+1}]" for start, end in valid_ranges])
                        total_traces = sum(end - start + 1 for start, end in valid_ranges)
                        print(f"    - {hdr_path}: {ranges_str} ({total_traces} traces)")
                    else:
                        print(f"    - {hdr_path}: NO VALID TRACES")
                else:
                    print(f"    - {hdr_path}: Could not load radar_time array")
        else:
            print(f"  ✓ All header times covered by GPS")

    print()
    print("=" * 120)
    print(f"Summary: Processed {total_segments} segments, found {total_gaps} total gaps")

    # Output start and stop records in one-indexed format
    print()
    print("=" * 120)
    print("START AND STOP RECORDS (one-indexed, comma-separated)")
    print("=" * 120)

    for idx, row in df_records.iterrows():
        segment_key = (row['segment_date_str'], row['segment_number'])

        if segment_key not in segment_ranges:
            print(f"{row['segment_date_str']},{row['segment_number']},NO_DATA")
            continue

        ranges_by_file = segment_ranges[segment_key]

        # Get all unique filenames (should be bxds1.mat and bxds2.mat typically)
        filenames = sorted(ranges_by_file.keys())

        if not filenames:
            print(f"{row['segment_date_str']},{row['segment_number']},NO_FILES")
            continue

        # Check if all files have the same ranges
        all_range_data = [ranges_by_file[fn] for fn in filenames if ranges_by_file[fn] is not None]

        if len(all_range_data) == 0:
            print(f"{row['segment_date_str']},{row['segment_number']},COULD_NOT_LOAD")
            continue

        # Extract ranges and max_trace_idx from first file
        first_ranges, first_max_trace = all_range_data[0]

        # Compare ranges between files
        ranges_differ = False
        for other_data in all_range_data[1:]:
            other_ranges, other_max_trace = other_data
            if first_ranges != other_ranges or first_max_trace != other_max_trace:
                ranges_differ = True
                break

        if ranges_differ:
            print(f"WARNING: Segment {row['segment_date_str']}_{row['segment_number']:02d} has differing ranges between header files!", file=sys.stderr)
            for fn in filenames:
                range_data = ranges_by_file[fn]
                if range_data is not None:
                    ranges, max_trace = range_data
                    if ranges:
                        ranges_str = ", ".join([f"[{start+1}:{end+1}]" for start, end in ranges])
                        print(f"  {fn}: {ranges_str}", file=sys.stderr)

        # Output the ranges (using first file's ranges, one-indexed)
        if not first_ranges:
            print(f"{row['segment_date_str']},{row['segment_number']},NO_VALID_TRACES")
        else:
            # Convert to one-indexed and format as comma-separated start,stop pairs
            # Leave blank if at the limits (no cropping needed)
            range_parts = []
            for start, end in first_ranges:
                # Check if this range covers the entire file
                if start == 0 and end == first_max_trace:
                    # No cropping needed - leave both blank
                    range_parts.append(",")
                elif start == 0:
                    # Only need to specify end
                    range_parts.append(f",{end+1}")
                elif end == first_max_trace:
                    # Only need to specify start
                    range_parts.append(f"{start+1},")
                else:
                    # Need both start and end
                    range_parts.append(f"{start+1},{end+1}")
            ranges_output = ",".join(range_parts)
            print(f"{row['segment_date_str']},{row['segment_number']},{ranges_output}")


if __name__ == "__main__":
    main()
