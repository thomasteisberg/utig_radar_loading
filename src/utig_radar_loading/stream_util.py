import pandas as pd
import gzip
import re
import os
from pathlib import Path
import warnings
import holoviews as hv
import geoviews as gv
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geoviews.feature as gf
import numpy as np
from datetime import datetime, timedelta
import sys

#streams_definitions = "/resfs/GROUPS/CRESIS/dataproducts/metadata/2022_Antarctica_BaslerMKB/UTIG_documentation/streams"


def load_gzipped_stream_file(file_path, debug=False, parse=True, parse_kwargs={'use_ct': True},):
    """
    Load a gzipped stream file as a pandas DataFrame with appropriate column names.
    
    Parameters:
    -----------
    file_path : str
        Path to the gzipped stream file (e.g., "/path/to/GPSnc1/xds.gz")
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with appropriate column names based on stream definition
    """
    file_path = Path(file_path)
    
    # Extract stream type from folder name
    stream_type = file_path.parent.name

    if debug:
        # Print all other files in the same directory
        print(f"Other files in {file_path.parent}:")
        for f in file_path.parent.glob('*'):
            print(f" -> {f.name} {"(this file)" if f == file_path else ""}")

    # Get stream definition
    column_names = get_stream_headers(stream_type)
    if not column_names:
        raise ValueError(f"No column names found for stream type: {stream_type}")

    if debug:
        print(f"Column names: {column_names}")
    
    # Load the data file
    if file_path.suffix == '.gz':
        file = gzip.open(file_path, 'rt')
    else:
        file = open(file_path, 'r')

    # Catch ParserWarning and print the list of expected columns
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        df = pd.read_csv(file, sep=r'\s+', names=column_names, index_col=False)

        for warning in w:
            print(f"Warning: {warning}")
            if issubclass(warning.category, pd.errors.ParserWarning):
                print(f"ParserWarning: {warning.message}")
                print("Data file did not match expected columns.")
                print(f"Expected {len(column_names)} columns for {stream_type}.")
                print(f"Expected column names were: {column_names}")

    # Check if a ct.gz file exists in the same directory
    ct_path = file_path.parent / "ct.gz"
    if ct_path.exists():
        ct_df = load_ct_file(ct_path)
        # ct_file = gzip.open(ct_path, 'rt')
        # ct_columns = ['prj', 'set', 'trn', 'seq', 'clk_y', 'clk_n', 'clk_d', 'clk_h', 'clk_m', 'clk_s', 'clk_f', 'tim']
        # ct_df = pd.read_csv(ct_file, sep=r'\s+', names=ct_columns, index_col=False)

        if debug:
            print(f"Found ct.gz file: {ct_path}")
            print(f"len(ct_df): {len(ct_df)}, len(df): {len(df)}")
        
        if len(ct_df) == len(df):
            # Merge columns of the two dataframes, joining by column number ignoring any index
            df = pd.concat([df, ct_df], axis=1)

    if parse:
        if stream_type == 'GPSnc1':
            df = parse_GPSnc1(df, **parse_kwargs)
        elif stream_type == 'GPStp2':
            df = parse_GPStp2(df, **parse_kwargs)
        elif stream_type == 'GPSap1':
            df = parse_GPSap1(df, **parse_kwargs)
        else:
            print(f"Warning: Unsupported stream type '{stream_type}' for parsing.")

    return df

def load_ct_file(file_path : str):
    path = Path(file_path)
    if path.is_file():
        path = path.parent
    
    path = path / 'ct.gz'
    if not path.exists():
        raise FileNotFoundError(f"ct.gz file not found at {file_path}")

    ct_file = gzip.open(path, 'rt')
    ct_columns = ['prj', 'set', 'trn', 'seq', 'clk_y', 'clk_n', 'clk_d', 'clk_h', 'clk_m', 'clk_s', 'clk_f', 'tim']
    return pd.read_csv(ct_file, sep=r'\s+', names=ct_columns, index_col=False)

def get_stream_headers(stream_type):
    if stream_type == 'GPSap3':
        return 'rtime ecefx ecefy ecefz rcoff vx vy vz rcdrft pdop'.split(' ')
    elif stream_type == 'GPSnc2':
        return 'ppsct tickcount timei timef id timebase'.split(' ')
    elif stream_type == 'GPSap1':
        # Based on GPSap1 stream definition - fields from IDS and XDS sections
        return [
            'len', 'sta', 'id',  # IDS section
            'utc_h', 'utc_m', 'utc_s',  # UTC time
            'lat_d', 'lat_m', 'lth',  # Latitude (degrees, minutes, hemisphere)
            'lon_d', 'lon_m', 'lnh',  # Longitude (degrees, minutes, hemisphere)  
            'nsv', 'hdp',  # Number of satellites, horizontal dilution
            'aht', 'ght',  # Antenna height, geoidal height
            'cog', 'sog_n', 'sog_k',  # Course over ground, speed (knots & km/hr)
            'gxt', 'gxd',  # Crosstrack error and direction
            'tpf', 'osn',  # Time of position fix, site name
            'efx', 'efy', 'efz',  # Earth-fixed coordinates (ECEF)
            'nco', 'vx', 'vy', 'vz',  # Navigation clock offset, velocities
            'ncd', 'pdp'  # Navigation clock drift, position dilution
        ]
    elif stream_type == 'GPSnc1':
        # Based on GPSnc1 stream definition - National Instruments PXI-6682/6683 timing
        return [
            'gps_time', 'gps_subsecs',  # GPS time (seconds, fraction)
            'pps_time', 'pps_subsecs',  # PPS time (seconds, fraction)
            'query_time', 'query_subsecs',  # Query time (seconds, fraction)
            'time_source',  # Time source (0=unknown, 1=system, 2=GPS, etc.)
            'pps_ct', 'query_ct',  # PPS counter, query counter
            'ct_flags',  # Counter flags (validity)
            'EW_vel', 'NS_vel', 'vert_vel',  # Velocities (East, North, Vertical)
            'lat_ang', 'lon_ang', 'vert_cor',  # Position (lat, lon, elevation)
            'gps_state',  # GPS state (0=uninitialized, etc.)
            'state',  # Bitfield state (135 nominal)
            'self_survey',  # Self survey percentage complete
            'time_offset', 'time_corr',  # Time offset and correction
            'utc_offset',  # UTC to TAI offset
            'nsv', 'sv_time',  # Number of satellites, SV observation time
            'sw_state'  # Software state (0=unknown, 1=init, 2=operating, 3=shutdown)
        ]
    elif stream_type == 'GPStp2':
        # Based on GPStp2 stream definition - Trimble Trimflite differential GPS (ASCII format)
        return [
            #'lat_hemisphere',   # +/- for latitude hemisphere
            'latitude',         # latitude in decimal degrees (dd.fffffffff)
            #'lon_hemisphere',   # +/- for longitude hemisphere
            'longitude',        # longitude in decimal degrees (ddd.fffffffff)
            'track',           # course over ground in decimal degrees (dddd.f)
            'ground_speed',    # ground speed in knots (dddd.f)
            'offline_distance', # offline distance in meters (ddddddd.f)
            'PDOP',            # position dilution of precision (ddddd.f)
            'gps_height',      # GPS height in meters (ddddd.ff)
            'easting',         # easting in meters (dddddddd.ffff)
            'northing',        # northing in meters (dddddddd.ffff)
            'dos_time',         # DOS time (hh:mm:ss.s)
        ]
    else:
        return None



def calculate_track_distance_km(df, lat_col='LAT', lon_col='LON'):
    """
    Calculate the total distance of a GPS track in kilometers after projecting to EPSG:3031.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing latitude and longitude columns
    lat_col : str
        Name of latitude column (default: 'LAT')
    lon_col : str
        Name of longitude column (default: 'LON')
        
    Returns:
    --------
    float
        Total distance of the track in kilometers
    """
    from pyproj import Transformer
    
    if lat_col not in df.columns or lon_col not in df.columns:
        raise ValueError(f"Columns '{lat_col}' and/or '{lon_col}' not found in DataFrame")
    
    # Filter out any NaN values
    valid_coords = df.dropna(subset=[lat_col, lon_col])
    
    if len(valid_coords) < 2:
        return 0.0  # Need at least 2 points to calculate distance
    
    # Create transformer from lat/lon (WGS84) to Antarctic Polar Stereographic (EPSG:3031)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    
    # Transform coordinates to meters in EPSG:3031
    x, y = transformer.transform(valid_coords[lon_col].values, valid_coords[lat_col].values)
    
    # Calculate distances between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)
    distances_m = np.sqrt(dx**2 + dy**2)
    
    # Sum all distances and convert to kilometers
    total_distance_km = np.sum(distances_m) / 1000.0
    
    return total_distance_km

def parse_CT(df):
    """
    Parse CT time headers and create TIMESTAMP column.
    
    CT headers: clk_y, clk_n, clk_d, clk_h, clk_m, clk_s, clk_f
    where:
    - clk_y: year
    - clk_n: day of year (1-366)
    - clk_d: day (possibly redundant with clk_n)
    - clk_h: hour
    - clk_m: minute
    - clk_s: second
    - clk_f: fractional seconds
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame potentially containing CT time columns
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame with added TIMESTAMP column if CT columns exist, None otherwise
    """
    ct_columns = ['clk_y', 'clk_n', 'clk_d', 'clk_h', 'clk_m', 'clk_s', 'clk_f']
    
    # Check if all CT columns exist
    if not all(col in df.columns for col in ct_columns):
        return None
    
    df = df.copy()
    
    # Create datetime from CT components
    # Using clk_y (year) and clk_n (day of year) as primary date info
    timestamps = []
    for idx, row in df.iterrows():
        year = int(row['clk_y'])
        day_of_year = int(row['clk_n'])
        hour = int(row['clk_h'])
        minute = int(row['clk_m'])
        second = int(row['clk_s'])
        microsecond = int(row['clk_f'] * 1e6) if row['clk_f'] < 1 else int((row['clk_f'] % 1) * 1e6)
        
        # Create datetime from year and day of year
        dt = datetime(year, 1, 1) + timedelta(days=day_of_year - 1, 
                                              hours=hour, 
                                              minutes=minute, 
                                              seconds=second,
                                              microseconds=microsecond)
        timestamps.append(dt)
    
    df['TIMESTAMP'] = pd.to_datetime(timestamps)
    
    return df


def parse_GPSnc1(df, use_ct=False):
    """
    Parse GPSnc1 format dataframe and add LAT, LON, TIMESTAMP columns.
    
    GPSnc1 contains:
    - gps_time, gps_subsecs: GPS time information
    - lat_ang: Latitude in decimal degrees (WGS-84)
    - lon_ang: Longitude in decimal degrees (WGS-84)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame from load_gzipped_stream_file with GPSnc1 data
    use_ct : bool
        If True, attempt to use CT time headers for TIMESTAMP
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added LAT, LON, TIMESTAMP columns
    """
    df = df.copy()
    
    # Map latitude and longitude directly (already in decimal degrees)
    if 'lat_ang' in df.columns:
        df['LAT'] = df['lat_ang']
    else:
        raise ValueError("Column 'lat_ang' not found in GPSnc1 data")
        
    if 'lon_ang' in df.columns:
        df['LON'] = df['lon_ang']
    else:
        raise ValueError("Column 'lon_ang' not found in GPSnc1 data")
    
    # Try CT time first if requested
    if use_ct:
        df_with_ct = parse_CT(df)
        if df_with_ct is not None:
            # CT parsing successful, use it
            df = df_with_ct
        else:
            # CT parsing failed, fall back to GPS time
            use_ct = False
    
    # Create TIMESTAMP from GPS time if not using CT
    if not use_ct:
        if 'gps_time' in df.columns and 'gps_subsecs' in df.columns:
            # GPS epoch starts at January 6, 1980 00:00:00 UTC
            gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
            
            # Convert GPS time (seconds since GPS epoch) to datetime
            gps_seconds = df['gps_time'].astype('int64')
            
            # Convert subseconds from fraction * 2^64 to actual fraction
            gps_fractions = df['gps_subsecs'].astype('uint64') / (2**64)
            
            # Combine seconds and fractions
            total_seconds = gps_seconds + gps_fractions
            
            # Create datetime by adding total seconds to GPS epoch
            df['TIMESTAMP'] = pd.to_datetime(gps_epoch) + pd.to_timedelta(total_seconds, unit='s')
        else:
            raise ValueError("Columns 'gps_time' and/or 'gps_subsecs' not found in GPSnc1 data")
    
    return df


def parse_GPStp2(df, use_ct=False):
    """
    Parse GPStp2 format dataframe and add LAT, LON, TIMESTAMP columns.
    
    GPStp2 is ASCII format from Trimble Trimflite differential GPS with:
    - Latitude and longitude already in signed decimal degrees
    - DOS Time (hh:mm:ss.s)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame from load_gzipped_stream_file with GPStp2 data
    use_ct : bool
        If True, attempt to use CT time headers for TIMESTAMP
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added LAT, LON, TIMESTAMP columns
    """
    df = df.copy()
    
    # Map latitude directly (already in signed decimal degrees)
    if 'latitude' in df.columns:
        df['LAT'] = pd.to_numeric(df['latitude'], errors='coerce')
    else:
        raise ValueError("Column 'latitude' not found in GPStp2 data")
    
    # Map longitude directly (already in signed decimal degrees)
    if 'longitude' in df.columns:
        df['LON'] = pd.to_numeric(df['longitude'], errors='coerce')
    else:
        raise ValueError("Column 'longitude' not found in GPStp2 data")
    
    # Handle TIMESTAMP
    if use_ct:
        df_with_ct = parse_CT(df)
        if df_with_ct is not None:
            df = df_with_ct
    
    # If no CT timestamp or CT failed, try DOS time
    if 'TIMESTAMP' not in df.columns and 'dos_time' in df.columns:
        # Parse DOS time (hh:mm:ss.s format)
        timestamps = []
        for dos_time_str in df['dos_time']:
            try:
                if pd.notna(dos_time_str) and ':' in str(dos_time_str):
                    time_parts = str(dos_time_str).strip().split(':')
                    if len(time_parts) >= 3:
                        hour = int(time_parts[0])
                        minute = int(time_parts[1])
                        sec_parts = time_parts[2].split('.')
                        second = int(sec_parts[0])
                        microsecond = int(float('0.' + sec_parts[1]) * 1e6) if len(sec_parts) > 1 else 0
                        
                        # If we have date from CT, use it; otherwise use today as placeholder
                        if 'clk_y' in df.columns and 'clk_n' in df.columns:
                            # Use the date from CT columns
                            year = int(df.loc[df.index[0], 'clk_y'])
                            day_of_year = int(df.loc[df.index[0], 'clk_n'])
                            base_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
                        else:
                            # Use a placeholder date - user will need to provide actual date
                            base_date = datetime.now().date()
                        
                        dt = datetime.combine(base_date, datetime.min.time()).replace(
                            hour=hour, minute=minute, second=second, microsecond=microsecond
                        )
                        timestamps.append(dt)
                    else:
                        timestamps.append(pd.NaT)
                else:
                    timestamps.append(pd.NaT)
            except (ValueError, IndexError):
                timestamps.append(pd.NaT)
        
        df['TIMESTAMP'] = pd.to_datetime(timestamps)
        
        if 'clk_y' not in df.columns:
            print("Warning: DOS time converted without date information. Timestamps use placeholder date.")
    
    return df


def parse_GPSap1(df, use_ct=False):
    """
    Parse GPSap1 format dataframe and add LAT, LON, TIMESTAMP columns.
    
    GPSap1 is from Ashtech M12 GPS Navigation System with:
    - Latitude in degrees and minutes with hemisphere
    - Longitude in degrees and minutes with hemisphere
    - UTC time components (hours, minutes, seconds)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame from load_gzipped_stream_file with GPSap1 data
    use_ct : bool
        If True, attempt to use CT time headers for TIMESTAMP
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added LAT, LON, TIMESTAMP columns
    """
    df = df.copy()
    
    # Convert latitude from degrees and minutes to decimal degrees
    if 'lat_d' in df.columns and 'lat_m' in df.columns and 'lth' in df.columns:
        lat_degrees = pd.to_numeric(df['lat_d'], errors='coerce')
        lat_minutes = pd.to_numeric(df['lat_m'], errors='coerce')
        lat_hemisphere = df['lth'].astype(str).str.strip()
        
        # Convert to decimal degrees: degrees + minutes/60
        lat_decimal = lat_degrees + lat_minutes / 60.0
        
        # Apply hemisphere (S = negative, N = positive)
        lat_sign = lat_hemisphere.apply(lambda x: -1 if x == 'S' else 1)
        df['LAT'] = lat_decimal * lat_sign
    else:
        raise ValueError("Columns 'lat_d', 'lat_m', and/or 'lth' not found in GPSap1 data")
    
    # Convert longitude from degrees and minutes to decimal degrees
    if 'lon_d' in df.columns and 'lon_m' in df.columns and 'lnh' in df.columns:
        lon_degrees = pd.to_numeric(df['lon_d'], errors='coerce')
        lon_minutes = pd.to_numeric(df['lon_m'], errors='coerce')
        lon_hemisphere = df['lnh'].astype(str).str.strip()
        
        # Convert to decimal degrees: degrees + minutes/60
        lon_decimal = lon_degrees + lon_minutes / 60.0
        
        # Apply hemisphere (W = negative, E = positive)
        lon_sign = lon_hemisphere.apply(lambda x: -1 if x == 'W' else 1)
        df['LON'] = lon_decimal * lon_sign
    else:
        raise ValueError("Columns 'lon_d', 'lon_m', and/or 'lnh' not found in GPSap1 data")
    
    # Handle TIMESTAMP
    if use_ct:
        df_with_ct = parse_CT(df)
        if df_with_ct is not None:
            df = df_with_ct
    
    # If no CT timestamp or CT failed, try UTC time
    if 'TIMESTAMP' not in df.columns and all(col in df.columns for col in ['utc_h', 'utc_m', 'utc_s']):
        # Parse UTC time components
        timestamps = []
        for idx, row in df.iterrows():
            try:
                hour = int(row['utc_h'])
                minute = int(row['utc_m'])
                second_float = float(row['utc_s'])
                second = int(second_float)
                microsecond = int((second_float % 1) * 1e6)
                
                # If we have date from CT, use it; otherwise use today as placeholder
                if 'clk_y' in df.columns and 'clk_n' in df.columns:
                    # Use the date from CT columns
                    year = int(row['clk_y'])
                    day_of_year = int(row['clk_n'])
                    base_date = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
                else:
                    # Use a placeholder date - user will need to provide actual date
                    base_date = datetime.now().date()
                
                dt = datetime.combine(base_date, datetime.min.time()).replace(
                    hour=hour, minute=minute, second=second, microsecond=microsecond
                )
                timestamps.append(dt)
            except (ValueError, TypeError):
                timestamps.append(pd.NaT)
        
        df['TIMESTAMP'] = pd.to_datetime(timestamps)
        
        if 'clk_y' not in df.columns:
            print("Warning: UTC time converted without date information. Timestamps use placeholder date.")
    
    return df


# MAP HELPERS

def create_antarctica_basemap():
    """
    Create a basemap of Antarctica with ocean and land features.
    
    Returns:
    --------
    geoviews.Overlay
        Basemap with ocean and land features in EPSG:3031 projection
    """
    epsg_3031 = ccrs.Stereographic(central_latitude=-90, true_scale_latitude=-71)
    return gf.ocean.options(scale='50m').opts(projection=epsg_3031) * gf.coastline.options(scale='50m').opts(projection=epsg_3031)

