"""Stage 1 (ORCA): Define segments from raw ORCA recordings.

Usage (from repo root):
    uv run orca/scripts/define_segments.py orca/seasons_config/<season>.yaml
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

from opr_ingest.orca import file_index, gps_pipeline, headers, radar_config, segment_splits

import holoviews as hv

from opr_ingest.core import basemap, geo
from opr_ingest.orca.gpspipe_gps import load_and_parse_gpspipe_file

# Minimum usable GPS fixes for a segment to be processable. OPR's
# records_create_sync_gps interp1 needs >= 2 points inside the radar window;
# segments below this (e.g. early ground/calibration runs with no 3D fix) are
# marked 'do not process' so the run scripts skip them.
MIN_GPS_POINTS = 2

# Minimum along-track motion (bbox diagonal in meters) for SAR / along-track
# stages to have anything to work with. Stationary recordings (start/end-of-day
# tests) have ~0 m extent and crash sar_coord_task ("Interpolation requires at
# least two sample points" with diagnostic "In 0 - 0 m === Out 0 - 0 m"). 100 m
# is conservative — clearly catches stationary, keeps any real transect.
MIN_PATH_METERS = 100


def gps_coverage_dnp_reason(gps_path, rx_samps_path) -> str:
    """Empty string if OPR can sync GPS for this recording, else a DNP reason.

    Pre-empts the failures we hit downstream in records_create_sync_gps and
    SAR's sar_coord_task by replicating their requirements against the GPS log
    and the radar comp_time window (from headers.get_header_information):

      * < MIN_GPS_POINTS usable GPS fixes total (empty / no-3D-fix recordings).
      * Stationary recording (bbox extent < MIN_PATH_METERS) -> SAR's interp1
        on along_track errors with "two sample points" + the diagnostic
        "In 0 - 0 m === Out 0 - 0 m".
      * < MIN_GPS_POINTS usable fixes inside the radar window -> OPR's
        records_create_sync_gps interp1 'requires at least two sample points'.
      * radar window not bracketed by GPS coverage -> OPR interpolates with
        interp1(...,NaN) (no extrapolation), producing NaN lat/lon/attitude and
        a `keyboard` halt.

    Also catches recordings whose radar header is unreadable (e.g. the merged
    _rx_samps.bin not present yet for unmerged-chunk prefixes).
    """
    comp = gps_pipeline.valid_gps_comp_times(gps_path)
    if len(comp) < MIN_GPS_POINTS:
        return f"only {len(comp)} valid GPS fix(es) < {MIN_GPS_POINTS}"
    extent_m = gps_pipeline.gps_path_extent_meters(gps_path)
    if extent_m < MIN_PATH_METERS:
        return f"stationary recording: {extent_m:.1f} m extent < {MIN_PATH_METERS} m"
    try:
        radar_comp = headers.get_header_information(rx_samps_path)["comp_time"]
    except Exception as e:
        return f"radar header unreadable ({type(e).__name__})"
    r0, r1 = float(radar_comp[0]), float(radar_comp[-1])
    n_in = int(((comp >= r0) & (comp <= r1)).sum())
    if n_in < MIN_GPS_POINTS:
        return f"only {n_in} GPS fix(es) in radar window < {MIN_GPS_POINTS}"
    if comp.min() > r0 or comp.max() < r1:
        return "GPS coverage does not bracket radar comp_time window"
    return ""


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
      * `file.prefix` → per-segment `<recording_prefix>_rx_samps.bin` (the
        recording timestamp + the `_rx_samps.bin` suffix from the season yaml).
        `file.board_folder_name` is left empty. ORCA data is flat under
        `file.base_dir`, but OPR's get_segment_file_list joins
        `base_dir + board_folder_name` as a *directory* and globs `file.prefix*`
        inside it. Putting the timestamp in file.prefix makes that glob match
        `<base_dir>/<recording_prefix>_rx_samps.bin` directly — no per-recording
        subdirectory (or hard-link) needed. (1 recording = 1 segment, so each
        segment has exactly one prefix.)
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
    radar_keys = ("fs", "prf", "f0", "f1", "Tpd", "presums", "DDC_freq")
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

    rx_suffix = defaults["params"]["records"].get("file.prefix") or "_rx_samps.bin"

    def file_prefix_per_segment(x):
        prefixes = list(x.sort_values("start_timestamp").index)
        if len(prefixes) != 1:
            raise ValueError(
                "Expected exactly one recording per segment (1:1 transect design); "
                f"got {len(prefixes)} for this segment: {prefixes}"
            )
        return f"{prefixes[0]}{rx_suffix}"

    def field_gps_per_segment(x):
        paths = x.sort_values("start_timestamp")["gps_path"].dropna().unique().tolist()
        if not paths:
            return ""
        return "{'" + "', '".join(str(p) for p in paths) + "'}"

    def notes_per_segment(x):
        x = x.sort_values("start_timestamp")
        prefixes = list(x.index)
        reasons = [r for r in x["dnp_reason"] if r] if "dnp_reason" in x else []
        if reasons:
            # Marked DNP: OPR's run scripts filter cmd.notes on 'do not process'
            # (case-insensitive regexp), so these are skipped automatically.
            return f"do not process ({reasons[0]}): {prefixes}"
        return prefixes

    file_prefix = grouped.apply(file_prefix_per_segment, include_groups=False)
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
        "file.prefix": file_prefix,
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

    # Pre-flight each recording's GPS↔radar overlap (same checks OPR's
    # records_create_sync_gps does) so generate_csvs can mark unprocessable
    # segments 'do not process' instead of halting the MATLAB run.
    df_season["dnp_reason"] = df_season.apply(
        lambda r: gps_coverage_dnp_reason(r["gps_path"], r["rx_samps_path"]), axis=1
    )

    print_match_report(df_season)

    generate_csvs(df_season, season_config, user_config)

    map_output_dir = Path(user_config.get("maps_output_base_dir", "outputs/maps"))
    generate_map(df_season, season_name, map_output_dir)


if __name__ == "__main__":
    main()
