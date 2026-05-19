"""Stage 1 (ORCA): Define segments from raw ORCA recordings.

Usage (from repo root):
    uv run orca/scripts/define_segments.py orca/seasons_config/<season>.yaml
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

from opr_ingest.orca import file_index, segment_splits


def load_configs(season_config_path: str):
    """Load season config and user config."""
    with open(season_config_path) as f:
        season_config = yaml.safe_load(f)

    user_config_path = Path("user_config.yaml")
    with open(user_config_path) as f:
        user_config = yaml.safe_load(f)

    return season_config, user_config


def build_recording_index(user_config: dict) -> pd.DataFrame:
    """Index ORCA recordings under the configured base path."""
    df_recordings = file_index.load_file_index_df(
        user_config["orca_raw_data_base_path"],
        user_config["file_index_cache"],
        read_cache=True,
    )
    return file_index.arrange_by_transect(df_recordings)


def generate_csvs(df_season: pd.DataFrame, season_config: dict, user_config: dict):
    """Generate parameter spreadsheet CSVs from the segment-assigned recordings DataFrame."""
    # TODO: adapt utig/scripts/define_segments.py:generate_csvs to ORCA. Key
    # differences: file.board_folder_name is built from ORCA prefix folders,
    # gps.field_fn lists <prefix>_gpspipe_stdout.log paths, gps.postprocessed_fn
    # is dropped (no separate post-processed GPS source for ORCA).
    raise NotImplementedError(
        "ORCA define_segments.generate_csvs: build cmd/records/qlook/sar/array/"
        "radar/post/analysis_noise CSVs under <params_output_base_dir>/<season>/"
        " using the same schema as utig/scripts/define_segments.py:generate_csvs."
    )


def generate_map(df_season, season_name, output_dir):
    """Generate an interactive HTML map of segments."""
    # TODO: mirror utig/scripts/define_segments.py:generate_map. Needs a per-
    # transect GPS loader that returns LAT/LON/TIMESTAMP for ORCA recordings
    # (likely a thin wrapper around load_and_parse_gpspipe_file).
    raise NotImplementedError(
        "ORCA define_segments.generate_map: render an Antarctica/Arctic basemap "
        "with each segment's track. Reuses opr_ingest.core.basemap and "
        "opr_ingest.core.geo.create_path."
    )


def print_match_report(df_season):
    print("\n" + "=" * 60)
    print("ORCA SEGMENT REPORT")
    print("=" * 60)
    print(f"Recordings: {len(df_season)}")
    n_segments = (
        df_season["segment_path"].nunique() if "segment_path" in df_season.columns else "N/A"
    )
    print(f"Total segments: {n_segments}")
    print("=" * 60)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <season_config.yaml>")
        sys.exit(1)

    season_config_path = sys.argv[1]
    season_config, user_config = load_configs(season_config_path)
    season_name = season_config["season_name"]
    print(f"Processing ORCA season: {season_name}")

    df_recordings = build_recording_index(user_config)
    print(f"Indexed {len(df_recordings)} ORCA recordings")

    df_season = segment_splits.assign_segments(df_recordings)

    print_match_report(df_season)

    generate_csvs(df_season, season_config, user_config)

    map_output_dir = Path(user_config.get("maps_output_base_dir", "outputs/maps"))
    generate_map(df_season, season_name, map_output_dir)


if __name__ == "__main__":
    main()
