"""Stage 1 (ORCA): Define segments from raw ORCA recordings.

Usage (from repo root):
    uv run orca/scripts/define_segments.py orca/seasons_config/<season>.yaml
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

from opr_ingest.orca import file_index, radar_config, segment_splits


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
        user_config["orca_file_index_cache"],
        read_cache=True,
    )
    return file_index.arrange_by_transect(df_recordings)


def generate_csvs(df_season: pd.DataFrame, season_config: dict, user_config: dict):
    """Generate parameter spreadsheet CSVs from the segment-assigned recordings DataFrame.

    Mirrors `utig/scripts/define_segments.py:generate_csvs` against the UTIG schema
    (file.version 425 doubles as the ORCA schema for now). ORCA-specific overrides:

      * `file.base_dir` → user_config["orca_raw_data_base_path"]
      * `file.board_folder_name` → cell-string list of one element per segment, the
        recording prefix (so `base_dir + board_folder_name + file.prefix` resolves
        to `<base_dir>/<prefix>_rx_samps.bin` with `file.prefix: "_rx_samps.bin"`
        set in the season yaml).
      * `gps.fn` → absolute `<orca_gps_base_dir>/<season>/gps_<segment_path>.mat`.
        Must be absolute: OPR's `opr_filename_support` returns absolute paths
        as-is, but prepends `gRadar.support_path` (a shared, often read-only
        tree) to relative ones. Pointing `orca_gps_base_dir` at your own
        writable scratch keeps GPS out of the shared tree. Stage 2
        (`create_gps_support.py`) writes the .mat to this same path.
      * `gps.field_fn` → cell-string list of gpspipe log paths
      * No `gps.postprocessed_fn` (ORCA has no separate post-processed GPS).
    """
    season_name = season_config["season_name"]
    defaults = season_config

    base_params_dir = Path(user_config["params_output_base_dir"]) / season_name
    base_params_dir.mkdir(parents=True, exist_ok=True)

    orca_gps_base_dir = user_config["orca_gps_base_dir"]
    orca_raw_data_base_path = user_config["orca_raw_data_base_path"]

    def make_parameter_sheet(default_values, segments, overrides=None):
        df = pd.DataFrame(default_values, index=segments)
        if overrides:
            for key, value in overrides.items():
                df[key] = value
        return df

    grouped = df_season.groupby(["segment_date_str", "segment_number"])
    segments = grouped.first().index

    # Per-segment radar params from each segment's first recording's _config.yaml.
    # ORCA configs vary across recordings (rx_duration, num_presums, possibly
    # sample rate and RF center), so these can't live in the yaml as scalars.
    radar_keys = ("fs", "prf", "f0", "f1", "Tpd", "presums")
    per_segment_radar = {k: [] for k in radar_keys}
    for (date_str, seg_num) in segments:
        grp = grouped.get_group((date_str, seg_num)).sort_values("start_timestamp")
        params = radar_config.load_radar_params(grp.iloc[0]["config_path"])
        for k in radar_keys:
            per_segment_radar[k].append(params[k])
    radar_overrides = {
        k: pd.Series(v, index=segments) for k, v in per_segment_radar.items()
    }
    file_clk = radar_overrides["fs"]

    def board_folder_per_segment(x):
        prefixes = list(x.sort_values("start_timestamp").index)
        return "{'" + "', '".join(prefixes) + "'}"

    def field_gps_per_segment(x):
        paths = x.sort_values("start_timestamp")["gps_path"].dropna().unique().tolist()
        if not paths:
            return ""
        return "{'" + "', '".join(str(p) for p in paths) + "'}"

    def notes_per_segment(x):
        return list(x.sort_values("start_timestamp").index)

    board_folder_name = grouped.apply(board_folder_per_segment, include_groups=False)
    field_gps = grouped.apply(field_gps_per_segment, include_groups=False)
    notes = grouped.apply(notes_per_segment, include_groups=False)

    # Absolute path so OPR loads it directly (a relative path would be
    # resolved against the shared, read-only gRadar.support_path).
    gps_fn = pd.Series(
        [
            f"{orca_gps_base_dir}/{season_name}/gps_{date_str}_{seg_num:02d}.mat"
            for (date_str, seg_num) in segments
        ],
        index=segments,
    )

    mission_names = pd.Series([["ORCA"]] * len(segments), index=segments)

    make_parameter_sheet(defaults["params"]["cmd"], segments, overrides={
        "mission_names": mission_names,
        "notes": notes,
    }).to_csv(base_params_dir / "cmd.csv")

    make_parameter_sheet(defaults["params"]["records"], segments, overrides={
        "file.base_dir": orca_raw_data_base_path,
        "file.board_folder_name": board_folder_name,
        "gps.fn": gps_fn,
        "gps.field_fn": field_gps,
        "file.clk": file_clk,
    }).to_csv(base_params_dir / "records.csv")

    radar_defaults = defaults["params"].get("radar") or {}
    if radar_defaults:
        make_parameter_sheet(
            radar_defaults, segments, overrides=radar_overrides
        ).to_csv(base_params_dir / "radar.csv")

    sheets_with_defaults_only = ["qlook", "sar", "array", "post", "analysis_noise"]
    for sheet_name in sheets_with_defaults_only:
        sheet_defaults = defaults["params"].get(sheet_name) or {}
        if sheet_defaults:
            make_parameter_sheet(sheet_defaults, segments).to_csv(
                base_params_dir / f"{sheet_name}.csv"
            )

    print(f"\nCSV files written to {base_params_dir}/")


def generate_map(df_season, season_name, output_dir):
    """Render an interactive HTML map of ORCA segments over the Antarctica basemap."""
    import holoviews as hv

    from opr_ingest.core import basemap, geo
    from opr_ingest.orca.gpspipe_gps import load_and_parse_gpspipe_file

    hv.extension("bokeh")

    segment_dfs = []
    for prefix, row in df_season.iterrows():
        try:
            gps = load_and_parse_gpspipe_file(row["gps_path"])
        except Exception as e:
            print(f"[WARNING] Failed to load gpspipe for {prefix}: {e}")
            continue
        if gps.empty:
            print(f"[WARNING] No usable TPV records in {row['gps_path']}; skipping {prefix}")
            continue
        segment_dfs.append(pd.DataFrame({
            "LAT": gps["LAT"].values,
            "LON": gps["LON"].values,
            # Fake prj/set/trn so the instrument-agnostic core.geo.create_path
            # contract is satisfied; these surface as hover tooltips on the map.
            "prj": "ORCA",
            "set": row["segment_date_str"],
            "trn": prefix,
            "segment_path": row["segment_path"],
        }))

    if not segment_dfs:
        print("[WARNING] No segments with usable GPS data; skipping map generation")
        return

    _, path = geo.create_path(segment_dfs)
    path = path.opts(color="blue", line_width=1).relabel("ORCA Recordings")

    basemap_plot = basemap.create_antarctica_basemap()
    plot = basemap_plot * hv.Overlay([path])
    plot = plot.opts(aspect="equal", frame_width=800, frame_height=800, tools=["hover"])
    plot = plot.opts(title=season_name, legend_position="right")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_path = output_dir / f"{season_name}.html"
    hv.save(plot, str(map_path))
    print(f"Map saved to {map_path}")


def print_match_report(df_season):
    print("\n" + "=" * 60)
    print("ORCA SEGMENT REPORT")
    print("=" * 60)
    print(f"Recordings: {len(df_season)}")
    n_segments = (
        df_season["segment_path"].nunique() if "segment_path" in df_season.columns else "N/A"
    )
    print(f"Total segments: {n_segments}")
    if "segment_date_str" in df_season.columns:
        per_day = df_season.groupby("segment_date_str").size()
        print(f"Date range: {per_day.index.min()} .. {per_day.index.max()}  ({len(per_day)} days)")
    missing_rx = df_season["rx_samps_path"].apply(lambda p: not Path(p).exists()).sum()
    if missing_rx:
        print(f"Recordings with no merged _rx_samps.bin yet: {missing_rx}")
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
