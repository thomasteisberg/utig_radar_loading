"""Geographic projection, distance, and HoloViews path helpers."""

import numpy as np
import pandas as pd
import pyproj
from shapely import LineString
import holoviews as hv


def calculate_track_distance_km(df, lat_col='LAT', lon_col='LON'):
    """Total length of a track in km, projecting to EPSG:3031."""
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Columns '{lat_col}' and/or '{lon_col}' not found in DataFrame")

    valid = df.dropna(subset=[lat_col, lon_col])
    if len(valid) < 2:
        return 0.0

    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    x, y = transformer.transform(valid[lon_col].values, valid[lat_col].values)
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sum(np.sqrt(dx**2 + dy**2)) / 1000.0


def project_split_and_simplify(lon, lat, projection='EPSG:3031', simplify_tolerance=1000,
                               split_dist=2000, calc_length=False):
    """Project lon/lat to `projection`, split at gaps >= split_dist meters, simplify."""
    transformer = pyproj.Transformer.from_crs('EPSG:4326', projection, always_xy=True)
    x_proj, y_proj = transformer.transform(lon, lat)

    dist_deltas = np.sqrt(np.diff(x_proj)**2 + np.diff(y_proj)**2)
    segment_indices = np.array(np.where(dist_deltas > split_dist)) + 1
    segment_indices = np.insert(segment_indices, 0, 0)
    segment_indices = np.append(segment_indices, len(x_proj))

    x_simplified = []
    y_simplified = []
    length = 0

    for start_idx, end_idx in zip(segment_indices[:-1], segment_indices[1:]):
        if end_idx - start_idx < 5:
            continue
        x_segment = x_proj[start_idx:end_idx]
        y_segment = y_proj[start_idx:end_idx]
        if np.isnan(x_segment).any() or np.isnan(y_segment).any():
            print(f"Warning: NaN values found in segment {start_idx}:{end_idx}")
            continue

        line = LineString(zip(x_segment, y_segment))
        if calc_length:
            length += line.length
        if simplify_tolerance:
            line = line.simplify(tolerance=simplify_tolerance)
        coords = list(line.coords)

        x_simplified.extend([c[0] for c in coords])
        y_simplified.extend([c[1] for c in coords])
        x_simplified.append(np.nan)
        y_simplified.append(np.nan)

    if calc_length:
        return x_simplified, y_simplified, length
    return x_simplified, y_simplified


def create_path(segment_dfs, path_opts_kwargs={}):
    """Build a HoloViews `Path` from a list of per-segment GPS DataFrames.

    Each input df must have `prj`, `set`, `trn`, `LAT`, `LON`. Optional fields
    `segment_path`, `radar_stream_type` are propagated as path attributes.
    """
    dfs = []

    for idx, df_sub in enumerate(segment_dfs):
        df_tmp = df_sub.copy()
        df_tmp = df_tmp[df_tmp['LAT'] <= -50]
        if len(df_tmp) < 3:
            continue

        try:
            x_proj, y_proj = project_split_and_simplify(df_tmp['LON'].values, df_tmp['LAT'].values)
        except Exception as e:
            print(f"Error processing segment {idx}: {e}")
            continue

        x_proj.append(np.nan)
        y_proj.append(np.nan)

        df_simplified = pd.DataFrame({'x': x_proj, 'y': y_proj})

        required_fields = ['prj', 'set', 'trn']
        optional_fields = ['segment_path', 'radar_stream_type']
        display_fields = required_fields.copy()
        for k in optional_fields:
            if k in df_tmp:
                display_fields.append(k)

        for k in display_fields:
            df_simplified[k] = df_tmp[k].iloc[0]
            if len(df_tmp[k].unique()) > 1:
                print(f"segment_dfs[{idx}]['{k}'].unique(): {df_tmp[k].unique()}")

        dfs.append(df_simplified)

    df_combined = pd.concat(dfs, ignore_index=True)
    path = hv.Path(df_combined, ['x', 'y'], display_fields).opts(
        tools=['hover'],
        line_width=0.5,
        show_legend=True,
        **path_opts_kwargs,
    )
    return dfs, path
