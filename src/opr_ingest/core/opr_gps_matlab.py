"""Build and write OPR-format GPS .mat files.

These helpers operate on standardized DataFrames keyed by OPR column names
(GPS_TIME, RADAR_TIME, LAT, LON, ELEV, ROLL, PITCH, HEADING, COMP_TIME).
Instrument-specific loaders live in their respective subpackages.
"""

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import hdf5storage
import numpy as np
import pandas as pd


def merge_position_files(
    file_paths: List[Union[str, Path]],
    load_function: Callable,
    time_sort_key: str = 'GPS_TIME',
    load_kwargs: Optional[Dict] = None,
    remove_duplicates: bool = True,
) -> pd.DataFrame:
    """Load multiple position/navigation files with `load_function` and merge sorted by time."""
    if load_kwargs is None:
        load_kwargs = {}

    all_dfs = []
    for file_path in file_paths:
        df = load_function(file_path, **load_kwargs)
        if not df.empty:
            if time_sort_key in df.columns:
                first_time = df[time_sort_key].iloc[0] if len(df) > 0 else float('inf')
            else:
                warnings.warn(f"Time sort key '{time_sort_key}' not found in {file_path}")
                first_time = float('inf')
            all_dfs.append((first_time, df))

    if not all_dfs:
        raise ValueError("No valid data loaded from provided files")

    all_dfs.sort(key=lambda x: x[0])
    merged_df = pd.concat([df for _, df in all_dfs], ignore_index=True)

    if time_sort_key in merged_df.columns:
        merged_df = merged_df.sort_values(time_sort_key).reset_index(drop=True)
        if remove_duplicates:
            time_diff = np.diff(merged_df[time_sort_key])
            if np.any(time_diff <= 0):
                merged_df = merged_df.drop_duplicates(subset=time_sort_key, keep='first').reset_index(drop=True)
                warnings.warn(f"Duplicate or non-increasing {time_sort_key} entries found and removed")

    return merged_df


def merge_df(target_df, source_df, interp_x_key, interp_y_keys=None,
             extrapolation_distance=2, other_keys_suffix=""):
    """Merge `source_df` into `target_df` by interpolating against `interp_x_key`.

    Keys in `interp_y_keys` overwrite same-named columns in target_df. Other
    columns from source_df get `other_keys_suffix` appended, or are skipped if
    `other_keys_suffix is None`.
    """
    if interp_y_keys is None:
        interp_y_keys = []

    if (target_df[interp_x_key].min() > source_df[interp_x_key].max()) or \
       (target_df[interp_x_key].max() < source_df[interp_x_key].min()):
        raise ValueError(f"Ranges of '{interp_x_key}' do not overlap between target_df and source_df")

    merged_df = target_df.copy()

    if interp_x_key not in target_df.columns or interp_x_key not in source_df.columns:
        raise ValueError(f"interp_x_key '{interp_x_key}' must be present in both dataframes")

    x_target = target_df[interp_x_key].values
    x_source = source_df[interp_x_key].values

    for key in source_df.columns:
        if (key not in interp_y_keys) and (other_keys_suffix is None):
            continue

        try:
            y_source = source_df[key].values
            if pd.api.types.is_datetime64_any_dtype(source_df[key]):
                y_source = source_df[key].astype('int64') / 1e9

            interp_y = np.interp(x_target, x_source, y_source, left=np.nan, right=np.nan)

            if key not in interp_y_keys:
                key = key + other_keys_suffix

            merged_df[key] = interp_y
        except Exception as e:
            print(f"Warning: Failed to merge key '{key}' ({e}): "
                  f"x_target dtype: {x_target.dtype}, x_source dtype: {x_source.dtype}, "
                  f"source_df[{key}] dtype: {source_df[key].dtype}")

    return merged_df


def create_gps_matlab_structure(df: pd.DataFrame, source: str) -> Dict[str, Any]:
    """Build the dict that gets written as an OPR GPS .mat file."""
    if 'ELEV' in df.columns:
        elev = df['ELEV'].values
    elif 'vert_cor' in df.columns:
        elev = df['vert_cor'].values
    else:
        elev = np.zeros(len(df))

    roll = df['ROLL'].values if 'ROLL' in df.columns else np.zeros(len(df))
    pitch = df['PITCH'].values if 'PITCH' in df.columns else np.zeros(len(df))
    heading = df['HEADING'].values if 'HEADING' in df.columns else np.zeros(len(df))

    gps_struct = {
        'gps_time': df['GPS_TIME'].values.astype(np.float64),
        'radar_time': df['RADAR_TIME'].values.astype(np.float64),
        'lat': df['LAT'].values.astype(np.float64),
        'lon': df['LON'].values.astype(np.float64),
        'elev': elev.astype(np.float64),
        'roll': roll.astype(np.float64),
        'pitch': pitch.astype(np.float64),
        'heading': heading.astype(np.float64),
        'gps_source': source,
        'file_type': 'gps',
    }

    if 'COMP_TIME' in df.columns:
        gps_struct['comp_time'] = df['COMP_TIME'].values.astype(np.float64)

    n_records = len(df)
    for key in ['gps_time', 'comp_time', 'radar_time', 'lat', 'lon', 'elev', 'roll', 'pitch', 'heading']:
        if key in gps_struct:
            gps_struct[key] = gps_struct[key].reshape(1, -1)
            assert np.shape(gps_struct[key]) == (1, n_records), \
                f"Shape mismatch for {key}, expected (1, {n_records}), got {np.shape(gps_struct[key])}"

    return gps_struct


def save_gps_matlab_file(gps_struct: Dict[str, Any], output_path: Union[str, Path]):
    """Save the GPS structure as a MATLAB v7.3 .mat file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hdf5storage.savemat(
        output_path, gps_struct,
        format='7.3', store_python_metadata=False, matlab_compatible=True, truncate_existing=True,
    )
