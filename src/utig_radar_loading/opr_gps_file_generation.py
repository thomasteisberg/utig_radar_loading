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
from pathlib import Path
from datetime import datetime
import warnings
from typing import List, Union, Dict, Any
from . import stream_util


def unix_epoch_to_datenum(unix_time):
    """
    Convert Unix epoch time to MATLAB datenum.
    
    MATLAB datenum is days since January 0, 0000.
    Unix epoch is seconds since January 1, 1970.
    
    Parameters:
    -----------
    unix_time : float or array-like
        Unix epoch time in seconds
        
    Returns:
    --------
    float or numpy array
        MATLAB datenum
    """
    # MATLAB datenum for Unix epoch (Jan 1, 1970)
    matlab_unix_epoch = 719529.0
    
    # Convert seconds to days and add to epoch
    return matlab_unix_epoch + np.asarray(unix_time) / 86400.0


def load_and_parse_gps_file(gps_path: Union[str, Path], use_ct: bool = True) -> pd.DataFrame:
    """
    Load a single GPS file and parse it to standard format.
    
    Parameters:
    -----------
    gps_path : str or Path
        Path to GPS data file (e.g., GPSnc1/xds.gz)
    use_ct : bool
        Whether to use CT time if available (default: True)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with parsed GPS data including LAT, LON, TIMESTAMP
    """
    gps_path = Path(gps_path)
    
    # Load the GPS data using stream_util
    try:
        df = stream_util.load_gzipped_stream_file(
            gps_path, 
            debug=False, 
            parse=True, 
            parse_kwargs={'use_ct': use_ct}
        )
    except Exception as e:
        warnings.warn(f"Failed to load {gps_path}: {e}")
        return pd.DataFrame()
    
    # Ensure we have the required columns
    required_cols = ['LAT', 'LON', 'TIMESTAMP']
    if not all(col in df.columns for col in required_cols):
        warnings.warn(f"Missing required columns in {gps_path}")
        return pd.DataFrame()
    
    # Add source file information
    df['source_file'] = str(gps_path)
    
    # Convert TIMESTAMP to Unix epoch if it's a datetime
    if pd.api.types.is_datetime64_any_dtype(df['TIMESTAMP']):
        df['unix_time'] = df['TIMESTAMP'].astype(np.int64) / 1e9  # Convert nanoseconds to seconds
    else:
        df['unix_time'] = df['TIMESTAMP']
    
    return df


def merge_gps_files(gps_paths: List[Union[str, Path]], use_ct: bool = True) -> pd.DataFrame:
    """
    Load and merge multiple GPS files, sorted by time.
    
    Parameters:
    -----------
    gps_paths : list of str or Path
        List of paths to GPS data files
    use_ct : bool
        Whether to use CT time if available (default: True)
        
    Returns:
    --------
    pd.DataFrame
        Merged and sorted GPS data
    """
    all_dfs = []
    
    for gps_path in gps_paths:
        df = load_and_parse_gps_file(gps_path, use_ct=use_ct)
        if not df.empty:
            # Get first timestamp for sorting files
            first_time = df['unix_time'].iloc[0] if len(df) > 0 else float('inf')
            all_dfs.append((first_time, df))
    
    if not all_dfs:
        raise ValueError("No valid GPS data loaded from provided files")
    
    # Sort by first timestamp of each file
    all_dfs.sort(key=lambda x: x[0])
    
    # Concatenate all dataframes
    merged_df = pd.concat([df for _, df in all_dfs], ignore_index=True)
    
    # Sort by timestamp (in case files have overlapping times)
    merged_df = merged_df.sort_values('unix_time').reset_index(drop=True)
    
    return merged_df


def create_gps_matlab_structure(df: pd.DataFrame) -> Dict[str, Any]:
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
        'gps_time': df['unix_time'].values.astype(np.float64),  # Unix epoch seconds
        'lat': df['LAT'].values.astype(np.float64),  # Latitude in degrees
        'lon': df['LON'].values.astype(np.float64),  # Longitude in degrees
        'elev': elev.astype(np.float64),  # Elevation in meters
        'roll': roll.astype(np.float64),  # Roll in radians (or zeros)
        'pitch': pitch.astype(np.float64),  # Pitch in radians (or zeros)
        'heading': heading.astype(np.float64),  # Heading in radians (or zeros)
        'gps_source': 'UTIG_GPSNC1'  # GPS source identifier
    }
    
    # Ensure all arrays are 1D and same length
    n_records = len(df)
    for key in ['gps_time', 'lat', 'lon', 'elev', 'roll', 'pitch', 'heading']:
        if key in gps_struct:
            gps_struct[key] = gps_struct[key].reshape(-1)[:n_records]
    
    return gps_struct


def save_gps_matlab_file(gps_struct: Dict[str, Any], output_path: Union[str, Path], 
                        format: str = 'scipy'):
    """
    Save GPS structure to MATLAB-compatible .mat file.
    
    Parameters:
    -----------
    gps_struct : dict
        GPS data structure
    output_path : str or Path
        Output file path (.mat extension recommended)
    format : str
        Format to save in: 'scipy' for old-style .mat (default), 'hdf5' for v7.3
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'scipy':
        # Save using scipy.io.savemat for standard MATLAB compatibility
        matlab_data = {'gps': gps_struct}
        scipy.io.savemat(
            str(output_path), 
            matlab_data, 
            do_compression=True
        )
    elif format == 'hdf5':
        # Save as HDF5 file (MATLAB v7.3 format) using hdf5storage
        # This library handles MATLAB compatibility automatically
        
        # Convert data to proper format for hdf5storage
        matlab_data = {}
        for key, value in gps_struct.items():
            if isinstance(value, str):
                matlab_data[key] = value
            else:
                # Ensure column vector format (Nx1) for MATLAB compatibility
                data = np.asarray(value, dtype=np.float64)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                matlab_data[key] = data
        
        # Save using hdf5storage for full MATLAB compatibility
        hdf5storage.savemat(
            str(output_path), 
            matlab_data, 
            format='7.3',
            store_python_metadata=False,
            matlab_compatible=True
        )
    else:
        raise ValueError(f"Unknown format '{format}'. Use 'scipy' or 'hdf5'.")


def generate_gps_file(gps_paths: List[Union[str, Path]], 
                     output_path: Union[str, Path],
                     use_ct: bool = True,
                     format: str = 'hdf5') -> None:
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
    use_ct : bool
        Whether to use CT time if available (default: True)
    format : str
        Format to save in: 'scipy' for old-style .mat (default), 'hdf5' for v7.3
        
    Example:
    --------
    >>> gps_paths = [
    ...     '/data/UTIG/ASB/JKB2s/GL0107b/GPSnc1/xds.gz',
    ...     '/data/UTIG/ASB/JKB2s/GL0107c/GPSnc1/xds.gz'
    ... ]
    >>> generate_gps_file(gps_paths, 'output/gps_20180105_01.mat')
    >>> generate_gps_file(gps_paths, 'output/gps_20180105_01.mat', format='hdf5')
    """
    print(f"Processing {len(gps_paths)} GPS files...")
    
    # Load and merge GPS files
    merged_df = merge_gps_files(gps_paths, use_ct=use_ct)
    print(f"Merged {len(merged_df)} GPS records")
    
    # Convert to MATLAB structure
    gps_struct = create_gps_matlab_structure(merged_df)
    
    # Save to MATLAB file
    save_gps_matlab_file(gps_struct, output_path, format=format)
    print(f"Saved GPS file to: {output_path} (format: {format})")
    
    # Print summary statistics
    print(f"GPS time range: {gps_struct['gps_time'][0]:.2f} to {gps_struct['gps_time'][-1]:.2f}")
    print(f"Lat range: {gps_struct['lat'].min():.6f} to {gps_struct['lat'].max():.6f}")
    print(f"Lon range: {gps_struct['lon'].min():.6f} to {gps_struct['lon'].max():.6f}")
    print(f"Elev range: {gps_struct['elev'].min():.2f} to {gps_struct['elev'].max():.2f} meters")
