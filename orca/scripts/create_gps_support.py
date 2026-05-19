"""Stage 2 (ORCA): Create GPS support files from parameter spreadsheet.

Usage (from repo root):
    uv run orca/scripts/create_gps_support.py path/to/season_params.xlsx [--overwrite]
"""

import argparse
from pathlib import Path

import pandas as pd

from opr_ingest.core import params
from opr_ingest.orca import gps_pipeline


def seg_label(row):
    return f"{int(row['day_seg_date'])}_{int(row['day_seg_num']):02d}"


def main():
    parser = argparse.ArgumentParser(description="Create ORCA GPS support files from parameter spreadsheet")
    parser.add_argument("spreadsheet", help="Path to xlsx parameter spreadsheet")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing GPS files")
    args = parser.parse_args()

    spreadsheet_path = Path(args.spreadsheet)
    if spreadsheet_path.suffix == ".xlsx":
        sheets = params.read_xlsx(spreadsheet_path)
    else:
        sheets = params.read_csvs(spreadsheet_path)

    processable = params.segments_to_process(sheets)
    records = sheets["records"]
    print(f"Found {len(processable)} segments to process (out of {len(records)})")

    n_generated = 0
    n_skipped = 0
    n_errors = 0

    for seg_idx in processable:
        row = records.loc[seg_idx]
        label = seg_label(row)

        gps_fn = row.get("gps.fn")
        if pd.isna(gps_fn):
            print(f"[SKIP] {label}: no gps.fn defined")
            n_skipped += 1
            continue

        output_path = Path(gps_fn)
        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] {label}: {output_path} already exists")
            n_skipped += 1
            continue

        gpspipe_str = row.get("gps.field_fn", "")
        gpspipe_paths = (
            params.parse_matlab_cell_string(str(gpspipe_str)) if pd.notna(gpspipe_str) else []
        )

        if not gpspipe_paths:
            print(f"[SKIP] {label}: no gpspipe log paths")
            n_skipped += 1
            continue

        print(f"\nProcessing {label}")
        try:
            gps_pipeline.generate_gps_file(
                gpspipe_paths=gpspipe_paths,
                output_path=output_path,
                gps_pad_time_s=3,
            )
            n_generated += 1
        except Exception as e:
            print(f"[ERROR] {label}: {e}")
            n_errors += 1

    print(f"\nDone. Generated: {n_generated}, Skipped: {n_skipped}, Errors: {n_errors}")


if __name__ == "__main__":
    main()
