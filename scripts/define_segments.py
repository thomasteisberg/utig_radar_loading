"""Stage 1: Define segments from post-processed GPS + raw radar data.

Usage:
    uv run scripts/define_segments.py seasons_config/2015_Antarctica_BaslerJKB.yaml
"""

import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from utig_radar_loading import file_util, segment_splits, opr_gps_file_generation
from utig_radar_loading.preprocessing import load_defaults


def load_configs(season_config_path: str):
    """Load season config and user config."""
    with open(season_config_path) as f:
        season_config = yaml.safe_load(f)

    user_config_path = Path(season_config_path).parent.parent / "user_config.yaml"
    with open(user_config_path) as f:
        user_config = yaml.safe_load(f)

    return season_config, user_config


def enumerate_postprocessed_gps(gps_dir: str) -> pd.DataFrame:
    """Find all post-processed GPS files and parse their (prj, set, trn) from filenames."""
    gps_files = glob.glob(f"{gps_dir}/**/*PUT[GN]1B*_position.txt", recursive=True)
    records = []
    for f in gps_files:
        parts = Path(f).stem.split("_")
        if len(parts) < 6:
            continue
        records.append({
            "postprocessed_gps_path": f,
            "prj": parts[-4],
            "set": parts[-3],
            "trn": parts[-2],
        })
    return pd.DataFrame(records)


def build_transect_index(user_config: dict, datasets: list[str]):
    """Load file index and build transect DataFrame."""
    df_files = file_util.load_file_index_df(
        user_config["raw_data_base_path"],
        user_config["file_index_cache"],
        read_cache=True,
    )
    df_artifacts = file_util.create_artifacts_df(df_files, datasets=datasets)

    usable_artifact_types = {
        "gps": {"stream_types": ["GPSnc1", "GPStp2", "GPSap3", "GPSkc1"], "file_names": ["xds.gz"]},
        "radar": {"stream_types": ["RADnh5", "RADnh3", "RADnh2", "RADnh4", "RADjh1"], "file_names": ["bxds", "bxds1"]},
    }
    df_transects = file_util.arrange_by_transect(df_artifacts, usable_artifact_types)
    return df_transects


def match_gps_to_transects(df_gps: pd.DataFrame, df_transects: pd.DataFrame):
    """Match post-processed GPS files to transects by (prj, trn), ignoring set.

    Returns (df_matched, unmatched_gps) where df_matched is df_transects with
    postprocessed_gps_path added, and unmatched_gps is a list of unmatched GPS records.
    """
    df_transects = df_transects.copy()
    df_transects["postprocessed_gps_path"] = pd.Series(dtype="object", index=df_transects.index)
    df_transects["postprocessed_gps_type"] = pd.Series(dtype="object", index=df_transects.index)

    unmatched = []
    transect_idx = df_transects.reset_index()

    for _, gps_row in df_gps.iterrows():
        prj, trn = gps_row["prj"], gps_row["trn"]
        matching_keys = [
            k for k in df_transects.index if k[0] == prj and k[2] == trn
        ]
        if not matching_keys:
            unmatched.append(gps_row)
            continue
        for key in matching_keys:
            df_transects.loc[key, "postprocessed_gps_path"] = gps_row["postprocessed_gps_path"]
            df_transects.loc[key, "postprocessed_gps_type"] = "EPUTG1B"

    return df_transects, unmatched


def filter_season(df_transects: pd.DataFrame):
    """Filter to transects with radar data and post-processed GPS, update timestamps.

    Returns (df_season, df_missing_radar, df_missing_gps).
    """
    # Update start timestamps using post-processed GPS where available
    df_transects["start_timestamp"] = df_transects.apply(file_util.get_start_timestamp, axis=1)

    # Filter out timing-only GPS without post-processed data
    timing_only = (df_transects["gps_stream_type"] == "GPSkc1") & df_transects["postprocessed_gps_path"].isna()
    if timing_only.any():
        print(f"[WARNING] Dropping {timing_only.sum()} transects with only GPSkc1 timing GPS and no post-processed GPS")
    df_transects = df_transects[~timing_only]

    # Separate missing radar
    df_missing_radar = df_transects[df_transects["radar_path"].isna()]
    df_season = df_transects[df_transects["radar_path"].notna()]
    if len(df_missing_radar) > 0:
        print(f"[WARNING] {len(df_missing_radar)} transects missing radar data (out of {len(df_transects)})")

    # Handle duplicate prj/trn (different sets) - keep the one with radar data
    df_reset = df_season.reset_index()
    grouped = df_reset.groupby(["prj", "trn"])
    rows_to_keep = []
    for (prj, trn), group in grouped:
        if len(group) == 1:
            rows_to_keep.append(group.iloc[0])
        else:
            with_radar = group[group["radar_path"].notna()]
            if len(with_radar) == 0:
                rows_to_keep.append(group.iloc[0])
            elif len(with_radar) == 1:
                rows_to_keep.append(with_radar.iloc[0])
            else:
                print(f"[WARNING] Multiple rows with radar for prj={prj}, trn={trn}: sets={with_radar['set'].tolist()}")
                rows_to_keep.append(with_radar.iloc[0])

    df_season = pd.DataFrame(rows_to_keep).set_index(["prj", "set", "trn"]).sort_index()

    # Separate missing post-processed GPS
    df_missing_gps = df_season[df_season["postprocessed_gps_path"].isna()]
    if len(df_missing_gps) > 0:
        print(f"[WARNING] {len(df_missing_gps)} transects missing post-processed GPS")
    df_season = df_season[df_season["postprocessed_gps_path"].notna()]

    return df_season, df_missing_radar, df_missing_gps


def generate_csvs(df_season: pd.DataFrame, season_config: dict, user_config: dict):
    """Generate parameter spreadsheet CSVs from the segment-assigned season DataFrame."""
    season_name = season_config["season_name"]
    season_config_path = f"seasons_config/{season_name}.yaml"
    defaults = load_defaults(season_config_path)

    base_params_dir = Path(user_config["params_output_base_dir"]) / season_name
    base_params_dir.mkdir(parents=True, exist_ok=True)

    gps_support_base_dir = user_config["gps_support_base_dir"]

    # Helper to build parameter sheet with optional per-segment overrides
    def make_parameter_sheet(default_values, segments, overrides=None):
        df = pd.DataFrame(default_values, index=segments)
        if overrides:
            for key, value in overrides.items():
                df[key] = value
        return df

    segment_index = df_season.groupby(["segment_date_str", "segment_number"]).ngroup()
    segments = df_season.groupby(["segment_date_str", "segment_number"]).first().index

    # Compute per-segment aggregations
    def radar_paths_ordered(x):
        l = x.sort_values("start_timestamp")["radar_path"].tolist()
        l = [str(Path(*Path(p).parts[-5:-1])) for p in l]
        return "{'" + "', '".join(l) + "'}"

    radar_paths = df_season.groupby(["segment_date_str", "segment_number"])[
        ["radar_path", "start_timestamp"]
    ].apply(radar_paths_ordered)

    def gps_fn(x):
        date_str = x["segment_date_str"].iloc[0]
        seg_num = x["segment_number"].iloc[0]
        return f"{gps_support_base_dir}/{season_name}/gps_{date_str}_{seg_num}.mat"

    gps_paths = df_season.groupby(["segment_date_str", "segment_number"])[
        ["segment_date_str", "segment_number"]
    ].apply(gps_fn, include_groups=False)

    def postprocessed_gps_fns(x):
        paths = x.sort_values("start_timestamp")["postprocessed_gps_path"].unique().tolist()
        return "{'" + "', '".join(str(p) for p in paths) + "'}"

    postprocessed_gps = df_season.groupby(["segment_date_str", "segment_number"])[
        ["postprocessed_gps_path", "start_timestamp"]
    ].apply(postprocessed_gps_fns)

    def field_gps_fns(x):
        paths = x.sort_values("start_timestamp")["gps_path"].dropna().unique().tolist()
        if not paths:
            return ""
        return "{'" + "', '".join(str(p) for p in paths) + "'}"

    field_gps = df_season.groupby(["segment_date_str", "segment_number"])[
        ["gps_path", "start_timestamp"]
    ].apply(field_gps_fns)

    def first_prj_list(x):
        return list(x["prj"].unique())

    def transect_name_list(x):
        return list(x.sort_values("start_timestamp")["trn"])

    mission_names = df_season.reset_index().groupby(["segment_date_str", "segment_number"])[
        ["prj", "set"]
    ].apply(first_prj_list)

    transect_names = df_season.reset_index().groupby(["segment_date_str", "segment_number"])[
        ["start_timestamp", "trn"]
    ].apply(transect_name_list)

    # Write CSVs
    make_parameter_sheet(defaults["params"]["cmd"], segments, overrides={
        "mission_names": mission_names,
        "notes": transect_names,
    }).to_csv(base_params_dir / "cmd.csv")

    make_parameter_sheet(defaults["params"]["records"], segments, overrides={
        "file.board_folder_name": radar_paths,
        "gps.fn": gps_paths,
        "gps.postprocessed_fn": postprocessed_gps,
        "gps.field_fn": field_gps,
    }).to_csv(base_params_dir / "records.csv")

    sheets_with_defaults_only = ["qlook", "sar", "array", "radar", "post", "analysis_noise"]
    for sheet_name in sheets_with_defaults_only:
        if sheet_name in defaults["params"]:
            make_parameter_sheet(defaults["params"][sheet_name], segments).to_csv(
                base_params_dir / f"{sheet_name}.csv"
            )

    print(f"\nCSV files written to {base_params_dir}/")


def generate_map(df_season, df_missing_radar, df_missing_gps, season_name, output_dir):
    """Generate an interactive HTML map of segments."""
    import holoviews as hv
    from utig_radar_loading import geo_util, stream_util

    hv.extension("bokeh")

    paths = []

    def try_add_path(df, source_type, color, width, label):
        try:
            if source_type is None:
                dfs = geo_util.load_gps_data(df)
            else:
                dfs = geo_util.load_gps_data(df, source_type=source_type)
            if dfs:
                _, p = geo_util.create_path(dfs)
                paths.append(p.opts(color=color, line_width=width).relabel(label))
        except Exception as e:
            print(f"[WARNING] Could not create map layer '{label}': {e}")

    # Missing radar data (red) — only transects that have post-processed GPS
    # (i.e. they belong to this season) but no matching radar data
    df_missing_with_postproc = df_missing_radar.dropna(subset=["postprocessed_gps_path"])
    if len(df_missing_with_postproc) > 0:
        print(f"Map: {len(df_missing_with_postproc)} transects with post-processed GPS but no radar")
        try_add_path(df_missing_with_postproc, "postprocessed", "red", 2, "Missing Radar Data")

    # Matched segments (blue)
    try_add_path(df_season, "postprocessed", "blue", 1, "Radar + Post-processed GPS Data")

    basemap = stream_util.create_antarctica_basemap()
    plot = basemap * hv.Overlay(paths)
    plot = plot.opts(aspect="equal", frame_width=800, frame_height=800, tools=["hover"])
    plot = plot.opts(title=season_name, legend_position="right")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / f"{season_name}.html"
    hv.save(plot, str(map_path))
    print(f"Map saved to {map_path}")


def print_match_report(df_season, df_missing_radar, df_missing_gps, unmatched_gps):
    """Print a summary of matching results."""
    print("\n" + "=" * 60)
    print("MATCH REPORT")
    print("=" * 60)
    print(f"Matched transects with radar + GPS: {len(df_season)}")
    print(f"Missing radar data:                 {len(df_missing_radar)}")
    print(f"Missing post-processed GPS:         {len(df_missing_gps)}")
    print(f"Unmatched GPS files (no radar):     {len(unmatched_gps)}")

    if unmatched_gps:
        print("\nUnmatched GPS files:")
        for row in unmatched_gps:
            print(f"  {Path(row['postprocessed_gps_path']).name}  (prj={row['prj']}, trn={row['trn']})")

    n_segments = df_season["segment_path"].nunique() if "segment_path" in df_season.columns else "N/A"
    print(f"\nTotal segments: {n_segments}")

    # Stream type summary
    print(f"GPS stream types:   {df_season['gps_stream_type'].unique().tolist()}")
    print(f"Radar stream types: {df_season['radar_stream_type'].unique().tolist()}")
    print(f"Sets:               {df_season.reset_index()['set'].unique().tolist()}")
    print(f"Projects:           {df_season.reset_index()['prj'].unique().tolist()}")
    print("=" * 60)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <season_config.yaml>")
        sys.exit(1)

    season_config_path = sys.argv[1]
    season_config, user_config = load_configs(season_config_path)
    season_name = season_config["season_name"]
    print(f"Processing season: {season_name}")

    # Step 1: Enumerate post-processed GPS files
    gps_dir = season_config["postprocessed_gps_dir"]
    df_gps = enumerate_postprocessed_gps(gps_dir)
    print(f"Found {len(df_gps)} post-processed GPS files in {gps_dir}")

    # Step 2: Build transect index from raw data
    datasets = season_config.get("datasets", ["UTIG1", "UTIG2"])
    df_transects = build_transect_index(user_config, datasets)
    print(f"Indexed {len(df_transects)} transects from raw data")

    # Step 3: Match GPS to transects
    df_transects, unmatched_gps = match_gps_to_transects(df_gps, df_transects)

    # Step 4: Filter to usable season data
    df_season, df_missing_radar, df_missing_gps = filter_season(df_transects)

    # Step 5: Assign segments
    df_season = segment_splits.assign_segments(
        df_season, timestamp_field="tim", parse_ct=False, timestamp_split_threshold=2000
    )

    # Step 6: Report
    print_match_report(df_season, df_missing_radar, df_missing_gps, unmatched_gps)

    # Step 7: Generate CSVs
    generate_csvs(df_season, season_config, user_config)

    # Step 8: Generate map
    map_output_dir = Path(user_config.get("maps_output_base_dir", "outputs/maps"))
    generate_map(df_season, df_missing_radar, df_missing_gps, season_name, map_output_dir)


if __name__ == "__main__":
    main()
