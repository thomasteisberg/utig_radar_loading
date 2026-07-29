"""Stage 3 (ORCA): Create temporary header files from parameter spreadsheet.

Usage (from repo root):
    uv run orca/scripts/create_headers.py path/to/season_params.xlsx [--overwrite]
"""

import argparse
from pathlib import Path

import hdf5storage
import pandas as pd
import yaml

from opr_ingest.core import params
from opr_ingest.orca import headers as orca_headers


def load_user_config() -> dict:
    """Load user_config.yaml from project root."""
    for candidate in [Path("user_config.yaml"), Path(__file__).resolve().parents[2] / "user_config.yaml"]:
        if candidate.exists():
            with open(candidate) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("user_config.yaml not found")


def get_and_save_header(path: str, fn: str):
    """Get header information and save to .mat file."""
    header = orca_headers.get_header_information(path)
    fn_path = Path(fn)
    fn_path.parent.mkdir(parents=True, exist_ok=True)
    hdf5storage.savemat(str(fn_path), header, format="7.3", truncate_existing=True)
    print(f"Saved header to {fn}")
    return header


def get_season_name(sheets: dict) -> str:
    """Extract season name from gps.fn paths in the records sheet."""
    records = sheets.get("records", pd.DataFrame())
    if "gps.fn" in records.columns:
        sample = records["gps.fn"].dropna().iloc[0] if len(records["gps.fn"].dropna()) > 0 else ""
        parts = Path(sample).parts
        for part in parts:
            if "_Antarctica_" in part or "_Greenland_" in part or "_Arctic_" in part:
                return part
    return "unknown_season"


def collect_radar_files(sheets: dict, processable, user_config: dict):
    """Collect ORCA recordings to process and their destination header paths.

    ORCA data is flat under `file.base_dir`, and `file.prefix` carries the full
    rx_samps filename (recording timestamp + `_rx_samps.bin`), with
    `file.board_folder_name` empty:
        <base_dir>/<file.prefix>      (e.g. .../20251204_224207_rx_samps.bin)

    OPR's records_create reads the per-file header at
        opr_filename_opr_tmp(...,'headers', fullfile(board_folder_name, [fn_name '.mat']))
    With board_folder_name empty and fn_name the rx_samps stem, that is a flat
        <opr_tmp>/headers/rds/<season>/<rx_stem>.mat
    so we write the header to the matching flat location under header_base_dir
    (point header_base_dir at <opr_tmp>/headers/rds to avoid a deploy hard-link).
    """
    records = sheets["records"]
    season_name = get_season_name(sheets)
    header_base_dir = Path(user_config["header_base_dir"]) / season_name
    print(f"Header output dir: {header_base_dir}")

    radar_paths = []
    header_locations = []

    for seg_idx in processable:
        row = records.loc[seg_idx]
        base_dir = row.get("file.base_dir", "")
        file_prefix = row.get("file.prefix", "")

        if pd.isna(base_dir) or pd.isna(file_prefix) or not str(file_prefix).strip():
            continue

        rx_path = Path(base_dir) / str(file_prefix)
        if not rx_path.exists():
            print(f"[WARNING] Radar file not found: {rx_path}")
            continue
        header_fn = header_base_dir / f"{rx_path.stem}.mat"
        radar_paths.append(str(rx_path))
        header_locations.append(str(header_fn))

    return radar_paths, header_locations


def main():
    parser = argparse.ArgumentParser(description="Create ORCA temporary header files from parameter spreadsheet")
    parser.add_argument("spreadsheet", help="Path to xlsx parameter spreadsheet")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing header files")
    args = parser.parse_args()

    user_config = load_user_config()

    spreadsheet_path = Path(args.spreadsheet)
    if spreadsheet_path.suffix == ".xlsx":
        sheets = params.read_xlsx(spreadsheet_path)
    else:
        sheets = params.read_csvs(spreadsheet_path)

    processable = params.segments_to_process(sheets)
    print(f"Found {len(processable)} segments to process")

    radar_paths, header_locations = collect_radar_files(sheets, processable, user_config)
    print(f"Found {len(radar_paths)} radar files to process")

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
