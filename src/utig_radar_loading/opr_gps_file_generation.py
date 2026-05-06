"""
GPS file generation for OPR (Open Polar Radar) compatible format.

This module provides functions to generate MATLAB-compatible GPS files
from UTIG raw GPS data streams, matching the format used by CReSIS 
gps_create functions.
"""

import numpy as np
import pandas as pd
import scipy.io
import h5py
import hdf5storage
import astropy.time
from pathlib import Path
from datetime import datetime
import warnings
from typing import List, Optional, Union, Dict, Any
from . import stream_util
from utig_radar_loading.time_util import UNIX_EPOCH, GPS_EPOCH, NI_EPOCH

def load_and_parse_gps_file(gps_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single GPS file and parse it to standard format.
    
    Parameters:
    -----------
    gps_path : str or Path
        Path to GPS data file (e.g., GPSnc1/xds.gz)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with parsed GPS data including GPS_TIME, RADAR_TIME
    """
    gps_path = Path(gps_path)
    
    # Load the GPS data using stream_util
    try:
        df = stream_util.load_xds_stream_file(gps_path, parse=True)
    except Exception as e:
        warnings.warn(f"Failed to load {gps_path}: {e}")
        return pd.DataFrame()
    
    # Ensure we have the required columns
    required_cols = ['GPS_TIME', 'RADAR_TIME']
    if not all(col in df.columns for col in required_cols):
        warnings.warn(f"Missing required columns in {gps_path}. Required columns: {required_cols}, found columns: {df.columns.tolist()}")
        return pd.DataFrame()

    return df


def merge_position_files(
    file_paths: List[Union[str, Path]],
    load_function: callable,
    time_sort_key: str = 'GPS_TIME',
    load_kwargs: Optional[Dict] = None,
    remove_duplicates: bool = True
) -> pd.DataFrame:
    """
    Generic function to load and merge multiple position/navigation files, sorted by time.

    Parameters:
    -----------
    file_paths : list of str or Path
        List of paths to data files
    load_function : callable
        Function to load and parse individual files (e.g., load_and_parse_gps_file)
    time_sort_key : str
        Column name to sort by, typically `GPS_TIME` or `RADAR_TIME` (default: `GPS_TIME`)
    load_kwargs : dict, optional
        Additional keyword arguments to pass to the load function
    remove_duplicates : bool
        Whether to remove duplicate timestamps (default: True)

    Returns:
    --------
    pd.DataFrame
        Merged and sorted data
    """
    if load_kwargs is None:
        load_kwargs = {}

    all_dfs = []

    for file_path in file_paths:
        df = load_function(file_path, **load_kwargs)
        if not df.empty:
            # Get first timestamp for sorting files
            if time_sort_key in df.columns:
                first_time = df[time_sort_key].iloc[0] if len(df) > 0 else float('inf')
            else:
                warnings.warn(f"Time sort key '{time_sort_key}' not found in {file_path}")
                first_time = float('inf')
            all_dfs.append((first_time, df))

    if not all_dfs:
        raise ValueError("No valid data loaded from provided files")

    # Sort by first timestamp of each file
    all_dfs.sort(key=lambda x: x[0])

    # Concatenate all dataframes
    merged_df = pd.concat([df for _, df in all_dfs], ignore_index=True)

    # Sort by timestamp (in case files have overlapping times)
    if time_sort_key in merged_df.columns:
        merged_df = merged_df.sort_values(time_sort_key).reset_index(drop=True)

        if remove_duplicates:
            time_diff = np.diff(merged_df[time_sort_key])
            if np.any(time_diff <= 0):
                merged_df = merged_df.drop_duplicates(subset=time_sort_key, keep='first').reset_index(drop=True)
                warnings.warn(f"Duplicate or non-increasing {time_sort_key} entries found and removed")

    return merged_df


def load_and_parse_postprocessed_gps_file(gps_path: Union[str, Path]) -> pd.DataFrame:
    columns = "YEAR DOY SOD SEQ LON LAT AC_ELEVATION ROLL PITCH HEADING STDDEV NS_ACCELERATION EW_ACCELERATION VERT_ACCELERATION".split(" ")

    # Read first line to check for "EPUTG1B" format
    with open(gps_path, 'r') as f:
        first_line = f.readline().strip()
    if not (("EPUTG1B" in first_line) or ("SPUTG1B" in first_line) or ("IPUTG1B" in first_line)):
        warnings.warn(f"File {gps_path} does not appear to be in expected post-processed GPS format (missing 'EPUTG1B' or 'SPUTG1B' in first line)")
        #return pd.DataFrame()

    df = pd.read_csv(gps_path, sep=r'\s+', names=columns, comment='#', header=None)

    # Create GPS_TIME column from YEAR, DOY, SOD
    # Note: The input YEAR, DOY, and SOD are in UTC time

    dt = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str).str.zfill(3), format='%Y%j') + pd.to_timedelta(df['SOD'], unit='s')

    # Use astropy to convert from UTC time to CReSIS "GPS_time", which is ANSI-C time (seconds since Jan 1, 1970)
    t_utc = astropy.time.Time(dt, format='datetime64', scale='utc')
    df['GPS_TIME'] = t_utc.unix

    # Convert PITCH, ROLL, and HEADING from degrees to radians
    df['PITCH'] = np.deg2rad(df['PITCH'])
    df['ROLL'] = np.deg2rad(df['ROLL'])
    df['HEADING'] = np.deg2rad(df['HEADING'])

    # Rename columns to match expected format
    df = df.rename(columns={
        'AC_ELEVATION': 'ELEV'
    })

    return df

def load_and_parse_imu_file(imu_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a single IMU file and parse it to standard format.
    
    Parameters:
    -----------
    imu_path : str or Path
        Path to IMU data file (e.g., IMUnc1/xds.gz)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with parsed IMU data including ROLL, PITCH, HEADING, and either TIMESTAMP or gps_time
    """
    imu_path = Path(imu_path)
    
    stream_type = imu_path.parent.name  # e.g., 'AVNnp1

    if stream_type == 'AVNnp1':
        imu_df = stream_util.parse_binary_AVNnp1(imu_path)
    else:
        warnings.warn(f"Unknown IMU stream type '{stream_type}' in {imu_path}")
        return pd.DataFrame()

    # Ensure we have the required columns
    required_cols = ['ROLL', 'PITCH', 'HEADING']
    if not all(col in imu_df.columns for col in required_cols):
        warnings.warn(f"Missing required columns in {imu_path}")
        return pd.DataFrame()

    if 'TIMESTAMP' not in imu_df.columns and 'GPS_TIME' not in imu_df.columns:
        warnings.warn(f"Missing TIMESTAMP or GPS_TIME in {imu_path}")
        return pd.DataFrame()
    
    # Add source file information
    imu_df['source_file'] = str(imu_path)

    return imu_df




def create_gps_matlab_structure(df: pd.DataFrame, source:str = "UTIG_GPSnc1-field") -> Dict[str, Any]:
    """
    Create a dictionary structure matching MATLAB GPS format.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Merged GPS dataframe with LAT, LON, unix_time columns
        
    Returns:
    --------
    dict
        Dictionary with GPS data in MATLAB-compatible format
    """

    # Extract elevation if available, otherwise use zeros
    if 'ELEV' in df.columns:
        elev = df['ELEV'].values
    elif 'vert_cor' in df.columns:
        elev = df['vert_cor'].values
    else:
        elev = np.zeros(len(df))
    
    # Extract roll, pitch, heading if available
    if 'ROLL' in df.columns:
        roll = df['ROLL'].values
    else:
        roll = np.zeros(len(df))
    
    if 'PITCH' in df.columns:
        pitch = df['PITCH'].values
    else:
        pitch = np.zeros(len(df))
        
    if 'HEADING' in df.columns:
        heading = df['HEADING'].values  
    else:
        heading = np.zeros(len(df))
    
    # Create the GPS structure matching MATLAB format
    gps_struct = {
        'gps_time': df['GPS_TIME'].values.astype(np.float64),  # Unix epoch seconds
        'radar_time': df['RADAR_TIME'].values.astype(np.float64), # 10 us ticks
        'lat': df['LAT'].values.astype(np.float64),  # Latitude in degrees
        'lon': df['LON'].values.astype(np.float64),  # Longitude in degrees
        'elev': elev.astype(np.float64),  # Elevation in meters
        'roll': roll.astype(np.float64),  # Roll in radians (or zeros)
        'pitch': pitch.astype(np.float64),  # Pitch in radians (or zeros)
        'heading': heading.astype(np.float64),  # Heading in radians (or zeros)
        'gps_source': source,  # GPS source identifier
        'file_type': 'gps'  # File type identifier
    }

    if 'COMP_TIME' in df.columns:
        gps_struct['comp_time'] = df['COMP_TIME'].values.astype(np.float64)  # Timestamps from context files
    
    # Ensure all arrays are 1D and same length
    n_records = len(df)
    for key in ['gps_time', 'comp_time', 'radar_time', 'lat', 'lon', 'elev', 'roll', 'pitch', 'heading']:
        if key in gps_struct:
            gps_struct[key] = gps_struct[key].reshape(1, -1)
            assert np.shape(gps_struct[key]) == (1, n_records), f"Shape mismatch for {key}, expected (1, {n_records}), got {np.shape(gps_struct[key])}"
    
    return gps_struct


def save_gps_matlab_file(gps_struct: Dict[str, Any], output_path: Union[str, Path]):
    """
    Save GPS structure to MATLAB-compatible .mat file.
    
    Parameters:
    -----------
    gps_struct : dict
        GPS data structure
    output_path : str or Path
        Output file path (.mat extension recommended)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hdf5storage.savemat(output_path, gps_struct, format='7.3', store_python_metadata=False, matlab_compatible=True, truncate_existing=True)


def generate_gps_file(gps_paths: List[Union[str, Path]], 
                     output_path: Union[str, Path],
                     output_path_temporary_df: Optional[Union[str, Path]] = None,
                     imu_paths: Optional[List[Union[str, Path]]] = None,
                     postprocessed_gps_paths: Optional[List[Union[str, Path]]] = None,
                     gps_pad_time_s: float = 2) -> None:
    """
    Generate a MATLAB-compatible GPS file from UTIG raw GPS data.
    
    This is the main function that orchestrates the entire process:
    1. Load multiple GPS files
    2. Merge them sorted by time
    3. Convert to MATLAB format
    4. Save as .mat file
    
    Parameters:
    -----------
    gps_paths : list of str or Path
        List of paths to raw GPS data files
    output_path : str or Path
        Output file path for MATLAB GPS file
        
    Example:
    --------
    >>> gps_paths = [
    ...     '/data/UTIG/ASB/JKB2s/GL0107b/GPSnc1/xds.gz',
    ...     '/data/UTIG/ASB/JKB2s/GL0107c/GPSnc1/xds.gz'
    ... ]
    >>> generate_gps_file(gps_paths, 'output/gps_20180105_01.mat')
    """

    postproc_df = None
    field_gps_df = None
    field_imu_df = None

    # Load and merge post-processed GPS files
    if postprocessed_gps_paths and len(postprocessed_gps_paths) > 0:
        postproc_df = merge_position_files(
            file_paths=postprocessed_gps_paths,
            load_function=load_and_parse_postprocessed_gps_file,
            time_sort_key='GPS_TIME'
        )
        print(f"Merged {len(postproc_df)} post-processed GPS records from {postprocessed_gps_paths}")

    # Load and merge GPS files
    field_gps_df = merge_position_files(
        file_paths=gps_paths,
        load_function=load_and_parse_gps_file,
        time_sort_key='GPS_TIME',
        remove_duplicates=True
    )
    print(f"Merged {len(field_gps_df)} GPS records")
    print(f"field_gps_df columns: {field_gps_df.columns.tolist()}")

    if imu_paths and len(imu_paths) > 0:
        field_imu_df = merge_position_files(
            file_paths=imu_paths,
            load_function=load_and_parse_imu_file,
            time_sort_key='GPS_TIME',
            load_kwargs={},
            remove_duplicates=False
        )
        print(f"Merged {len(field_imu_df)} IMU records")

    # If we have post-processed GPS data, start with that as our base. If not, use field GPS data.
    if postproc_df is not None and not postproc_df.empty:
        merged_df = postproc_df
        source = "UTIG_EPUTG1B-postproc"

        # # TODO DEBUG
        # print(f"GPS_TIME range in post-processed GPS: {merged_df['GPS_TIME'].min()} to {merged_df['GPS_TIME'].max()}")
        # print(f"GPS_TIME range in field GPS: {field_gps_df['GPS_TIME'].min()} to {field_gps_df['GPS_TIME'].max()}")

        merged_df = merge_df(merged_df, field_gps_df, interp_x_key='GPS_TIME', interp_y_keys=['RADAR_TIME', 'COMP_TIME'], other_keys_suffix = "_field_gps")
        
        # if field_imu_df is not None and not field_imu_df.empty:
        #     merged_df = merge_df(merged_df, field_imu_df, interp_x_key='GPS_TIME', other_keys_suffix = "_field_imu")
    else:
        # No post-processed GPS, just use field GPS data
        merged_df = field_gps_df
        source = "UTIG_GPSnc1-field"

        if field_imu_df is not None and not field_imu_df.empty:
            merged_df = merge_df(merged_df, field_imu_df, interp_x_key='GPS_TIME', interp_y_keys=['HEADING', 'PITCH', 'ROLL'], other_keys_suffix = "_field_imu")
    
    merged_df = merged_df.dropna(subset=['GPS_TIME', 'COMP_TIME', 'RADAR_TIME', 'LAT', 'LON'])
    print(f"Merged dataframe columns: {merged_df.columns.tolist()} (len: {len(merged_df)}) (source: {source})")
    
    # Add extrapolated values to cover small time gaps between GPS and radar
    if gps_pad_time_s and gps_pad_time_s > 0:
        # Add one entry before and after, extrapolating the time values
        entry_before = merged_df.iloc[0]
        entry_after = merged_df.iloc[-1]
        for t_key in ['GPS_TIME', 'COMP_TIME', 'RADAR_TIME']:
            t = merged_df[t_key]
            dt_per_gps_time = (t.iloc[-1] - t.iloc[0]) / (merged_df['GPS_TIME'].iloc[-1] - merged_df['GPS_TIME'].iloc[0])
            entry_before[t_key] -= dt_per_gps_time * gps_pad_time_s
            entry_after[t_key] += dt_per_gps_time * gps_pad_time_s

        merged_df = pd.concat([pd.DataFrame([entry_before]), merged_df, pd.DataFrame([entry_after])], ignore_index=True)
        
    # Optionally, save the merged dataframe to a CSV for debugging
    if output_path_temporary_df is not None:
        output_path_temporary_df = Path(output_path_temporary_df)
        output_path_temporary_df.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path_temporary_df, index=False)
        print(f"Saved merged dataframe to: {output_path_temporary_df}")

    # Convert to MATLAB structure
    gps_struct = create_gps_matlab_structure(merged_df, source=source)
    
    # Save to MATLAB file
    save_gps_matlab_file(gps_struct, output_path)
    print(f"Saved GPS file to: {output_path}")

    # Print summary statistics
    print(f"GPS time range: {gps_struct['gps_time'][0,0]:.2f} to {gps_struct['gps_time'][0,-1]:.2f}")
    print(f"Lat range: {gps_struct['lat'].min():.6f} to {gps_struct['lat'].max():.6f}")
    print(f"Lon range: {gps_struct['lon'].min():.6f} to {gps_struct['lon'].max():.6f}")
    print(f"Elev range: {gps_struct['elev'].min():.2f} to {gps_struct['elev'].max():.2f} meters")

def merge_df(target_df, source_df, interp_x_key, interp_y_keys=None,
             extrapolation_distance=2, other_keys_suffix=""):
    """
    Merge source_df into target_df by interpolating interp_y_keys based on interp_x_key.
    Any other columns in source_df will be merged with a suffix.

    Parameters:
    -----------
    target_df : pd.DataFrame
        The target dataframe to merge into
    source_df : pd.DataFrame
        The source dataframe to merge from
    interp_x_key : str
        The column name to use as the x-axis for interpolation (e.g., 'GPS_TIME')
    interp_y_keys : list of str, optional
        List of column names to interpolate and merge without renaming (default: None)
    extrapolation_distance : float
        Maximum distance to allow for extrapolation (default: 2)
    other_keys_suffix : str
        Suffix to append to other keys when merging. Or None to skip merging other keys (default: "")

    Returns:
    --------
    pd.DataFrame
        Merged dataframe
    """
    if interp_y_keys is None:
        interp_y_keys = []

    # Check that the ranges of the x key overlaps
    if (target_df[interp_x_key].min() > source_df[interp_x_key].max()) or (target_df[interp_x_key].max() < source_df[interp_x_key].min()):
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
            # Get the source values
            y_source = source_df[key].values

            # Check if this is a datetime column and convert to numeric if needed
            if pd.api.types.is_datetime64_any_dtype(source_df[key]):
                # Convert datetime to seconds since epoch for interpolation
                y_source = source_df[key].astype('int64') / 1e9  # nanoseconds to seconds

            # Interpolate this key
            interp_y = np.interp(x_target, x_source, y_source, left=np.nan, right=np.nan)

            # Fill extrapolated values beyond limits with NaN
            # mask = (x_target < x_source[0] - extrapolation_distance) | (x_target > x_source[-1] + extrapolation_distance)
            # interp_y[mask] = np.nan

            if (key not in interp_y_keys):
                key = key + other_keys_suffix

            merged_df[key] = interp_y
        except Exception as e:
            print(f"Warning: Failed to merge key '{key}' ({e}): x_target dtype: {x_target.dtype}, x_source dtype: {x_source.dtype}, source_df[{key}] dtype: {source_df[key].dtype}")

    return merged_df

def make_segment_gps_file(x, output_base_dir, overwrite=False):
    # Designed to be called by .apply on df_season grouped by segment
    # Example usage: gps_paths = df_season.groupby(['segment_date_str', 'segment_number'])[['segment_date_str', 'segment_number', 'gps_path']].apply(opr_gps_file_generation.make_segment_gps_file, include_groups=False, overwrite=False)
    x = x.reset_index()
    print(f"{x['segment_date_str'].iloc[0]}_{x['segment_number'].iloc[0]}")
    gps_paths = list(x['gps_path'].unique())
    
    # Check for imu_path column
    if 'imu_path' in x:
        if x['imu_path'].isnull().any():
            warnings.warn(f"IMU paths contain null values for segment {x['segment_date_str'].iloc[0]}_{x['segment_number'].iloc[0]}")
            imu_paths = None
        else:
            imu_paths = list(x['imu_path'].unique())
    else:
        imu_paths = None

    # Check for postprocessed_gps_path column
    if 'postprocessed_gps_path' in x:
        if x['postprocessed_gps_path'].isnull().any():
            warnings.warn(f"Post-processed GPS paths contain null values for segment {x['segment_date_str'].iloc[0]}_{x['segment_number'].iloc[0]}")
            postprocessed_gps_paths = None
        else:
            postprocessed_gps_paths = list(x['postprocessed_gps_path'].unique())
    else:
        print("[WARNING] No post-processed GPS paths provided.")
        postprocessed_gps_paths = None

    output_path = output_base_dir / Path(f"gps_{x['segment_date_str'].iloc[0]}_{x['segment_number'].iloc[0]}.mat")

    # Only generate if the file does not exist

    if (not output_path.exists()) or overwrite:
        try:
            generate_gps_file(gps_paths, output_path, imu_paths=imu_paths,
                            postprocessed_gps_paths=postprocessed_gps_paths,
                            gps_pad_time_s=3)
        except ValueError as e:
            print(f"Failed to generate GPS file for segment {x['segment_date_str'].iloc[0]}_{x['segment_number'].iloc[0]}: {e}")
            return None
    else:
        print(f"File {output_path} already exists. Skipping generation. If you want to regenerate, delete the file or set overwrite=True.")

    return output_path.resolve()