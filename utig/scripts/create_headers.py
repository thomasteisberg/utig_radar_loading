"""Stage 3: Create temporary header files from parameter spreadsheet.

Usage:
    uv run scripts/create_headers.py path/to/season_params.xlsx [--overwrite]
"""

import argparse
from pathlib import Path

import hdf5storage
import numpy as np
import pandas as pd
import yaml

from utig_radar_loading import opr_header_generation, param_spreadsheet


def load_user_config() -> dict:
    """Load user_config.yaml from project root."""
    # Try a few locations
    for candidate in [Path("user_config.yaml"), Path(__file__).parent.parent / "user_config.yaml"]:
        if candidate.exists():
            with open(candidate) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("user_config.yaml not found")


def get_and_save_header(path: str, fn: str):
    """Get header information and save to .mat file."""
    header = opr_header_generation.get_header_information(path)
    fn_path = Path(fn)
    fn_path.parent.mkdir(parents=True, exist_ok=True)
    hdf5storage.savemat(str(fn_path), header, format="7.3", truncate_existing=True)
    print(f"Saved header to {fn}")
    return header


def get_season_name(sheets: dict) -> str:
    """Extract season name from the cmd sheet metadata or records paths."""
    # Try to find it from gps.fn paths in records (e.g., .../2015_Antarctica_BaslerJKB/...)
    records = sheets.get("records", pd.DataFrame())
    if "gps.fn" in records.columns:
        sample = records["gps.fn"].dropna().iloc[0] if len(records["gps.fn"].dropna()) > 0 else ""
        parts = Path(sample).parts
        for part in parts:
            if "_Antarctica_" in part or "_Greenland_" in part or "_Arctic_" in part:
                return part
    return "unknown_season"


def collect_radar_files(sheets: dict, processable, user_config: dict):
    """Collect all radar files to process, expanding RADjh1 to both channels."""
    records = sheets["records"]
    season_name = get_season_name(sheets)
    header_base_dir = str(Path(user_config["header_base_dir"]) / season_name)
    print(f"Header output dir: {header_base_dir}")

    radar_paths = []
    header_locations = []

    for seg_idx in processable:
        row = records.loc[seg_idx]
        base_dir = row.get("file.base_dir", "")
        board_folder_str = row.get("file.board_folder_name", "")

        if pd.isna(board_folder_str) or pd.isna(base_dir):
            continue

        folder_names = param_spreadsheet.parse_matlab_cell_string(str(board_folder_str))

        for folder_name in folder_names:
            # Determine the full radar file path
            prefix = row.get("file.prefix", "bxds")
            if pd.isna(prefix):
                prefix = "bxds"
            full_path = Path(base_dir) / folder_name / prefix

            # Check for RADjh1 - expand to both channels
            if "RADjh1" in folder_name:
                for channel_file in ["bxds1", "bxds2"]:
                    channel_path = Path(base_dir) / folder_name / channel_file
                    if channel_path.exists():
                        header_fn = opr_header_generation.get_header_file_location(
                            str(channel_path), str(header_base_dir)
                        )
                        radar_paths.append(str(channel_path))
                        header_locations.append(header_fn)
                    else:
                        print(f"[WARNING] RADjh1 channel file not found: {channel_path}")
            else:
                if full_path.exists():
                    header_fn = opr_header_generation.get_header_file_location(
                        str(full_path), str(header_base_dir)
                    )
                    radar_paths.append(str(full_path))
                    header_locations.append(header_fn)
                else:
                    print(f"[WARNING] Radar file not found: {full_path}")

    return radar_paths, header_locations


def main():
    parser = argparse.ArgumentParser(description="Create temporary header files from parameter spreadsheet")
    parser.add_argument("spreadsheet", help="Path to xlsx parameter spreadsheet")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing header files")
    args = parser.parse_args()

    user_config = load_user_config()

    spreadsheet_path = Path(args.spreadsheet)
    if spreadsheet_path.suffix == ".xlsx":
        sheets = param_spreadsheet.read_xlsx(spreadsheet_path)
    else:
        sheets = param_spreadsheet.read_csvs(spreadsheet_path)

    processable = param_spreadsheet.segments_to_process(sheets)
    print(f"Found {len(processable)} segments to process")

    radar_paths, header_locations = collect_radar_files(sheets, processable, user_config)
    print(f"Found {len(radar_paths)} radar files to process")

    # Filter out existing files unless --overwrite
    files_to_process = []
    for path, fn in zip(radar_paths, header_locations):
        if Path(fn).exists() and not args.overwrite:
            print(f"[SKIP] {fn} already exists")
        else:
            files_to_process.append((path, fn))

    if not files_to_process:
        print("No header files need to be generated.")
        return

    print(f"Processing {len(files_to_process)} header files...")

    # Use Dask for parallel processing
    n_workers = user_config.get("dask_workers", 10)
    try:
        from dask import delayed
        from dask.distributed import Client

        print(f"Starting Dask with {n_workers} workers...")
        client = Client(n_workers=n_workers)
        print(f"Dashboard: {client.dashboard_link}")

        delayed_tasks = [delayed(get_and_save_header)(path, fn) for path, fn in files_to_process]
        futures = client.compute(delayed_tasks)
        results = client.gather(futures)
        client.close()
        print(f"\nSuccessfully generated {len(results)} header files.")
    except ImportError:
        print("Dask not available, processing sequentially...")
        for path, fn in files_to_process:
            try:
                get_and_save_header(path, fn)
            except Exception as e:
                print(f"[ERROR] {path}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
