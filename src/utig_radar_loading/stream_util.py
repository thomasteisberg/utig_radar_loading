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
from utig_radar_loading.time_util import UNIX_EPOCH, GPS_EPOCH, NI_EPOCH

#
# This file contains methods for parsing data from UTIG data files of various kinds.
# There are four main categories of data files, each with different starting points for
# loading and parsing:
# 1. CT ("context") files (ct.gz or ct)
#    |-> load_ct_file() accepts a file path to a ct or ct.gz file and returns a dataframe.
#        Context files are located in the same directories as stream files and are supposed
#        to have the same number of rows as the stream file in that directory.
# 2. XDS stream files (e.g. GPSnc1, GPStp2, GPSap1, GPSap3)
#    |-> load_xds_stream_file() accepts a file path to an xds or xds.gz file. The stream
#        type is inferred from the parent folder name. The
#        load_xds_stream_file() function will automatically look for a ct.gz or ct file
#        in the same directory as the stream file and merge the two dataframes if found.
# 3. Binary files (e.g. AVNnp1, RADnh1)
#    |-> parse_binary_AVNnp1() accepts a file path to a binary AVNnp1 file and returns
#        a dataframe.
#        For loading any RAD*** binary files, use the UTIG unfoc library:
#        https://github.com/UTIG/unfoc/
# 4. Post-processed positioning files
#    |-> These are space-separated value .txt files made from post-processed GPS and
#        IMU data. They are loaded with load_postprocessed_position_file().
#
# All of these functions will return a dataframe with appropriate column names.
# The convention here is that loaded file columns are lowercase, while any parsed
# or derived columns are uppercase. Parsed columns that this library tries to add
# correspond to the definitions for OPR GPS files or OPR records files:
# https://gitlab.com/openpolarradar/opr/-/wikis/GPS-File-Guide
# https://gitlab.com/openpolarradar/opr/-/wikis/Record-File-Guide
#
# Parsed columns are:
# - GPS_TIME: gps time of each record in ANSI-C time (seconds since Jan 1, 1970).
# - GPS_TIME_DT: The above, converted to a pandas datetime object for convenience. Recommended for reference only.
# - HEADING: heading of each record in radians. Positive is clockwise. Zero at true north.
# - LAT: latitude of each record in degrees. Positive toward north. Zero at equator.
# - LON: longitude of each record in degrees. Positive toward east. Zero at prime meridian.
# - PITCH: pitch of each record in radians. Positive nose up. Zero is level flight.
# - ROLL: roll of each record in radians. Positive means right wing tip down. Zero is level flight.
# - COMP_TIME: Parsed time from context (CT) files in ANSI-C time (seconds since Jan 1, 1970).
# - COMP_TIME_DT: The above, converted to a pandas datetime object for convenience. Recommended for reference only.
# - RADAR_TIME: `tim` counter field from context (CT) files. Measures time in microseconds. No absolute reference.
# 
# Notes about `GPS_TIME`:
# 1. `GPS_TIME` is NOT UTC time and leap seconds need to be subtracted to find UTC time.
# 2. `GPS_TIME` is seconds since Jan 1, 1970, NOT seconds since Jan 6, 1980 (the GPS epoch).
#    This is to follow the conventions set by OPR.
#
# Some documentation about UTIG stream formats can be found on the OPR servers:
# > /resfs/GROUPS/CRESIS/dataproducts/metadata/2022_Antarctica_BaslerMKB/UTIG_documentation/streams
# Note, however, that these documents are not 100% accurate.
#

def load_xds_stream_file(file_path, parse=True, debug=False, parse_kwargs={}):
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

    file.close()

    # Check if a ct.gz file exists in the same directory
    ct_path = file_path.parent / "ct.gz"
    if not ct_path.exists():
        ct_path = file_path.parent / "ct" # Check for an uncompressed version
    
    if ct_path.exists():
        ct_df = load_ct_file(str(ct_path))

        if debug:
            print(f"Found ct().gz) file: {ct_path}")
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
        elif stream_type == 'GPSap3':
            df = parse_GPSap3(df, **parse_kwargs)
        elif stream_type == 'GPSkc1':
            df = parse_GPSkc1(df, **parse_kwargs)
        else:
            print(f"Warning: Unsupported stream type '{stream_type}' for parsing.")

    return df

def load_ct_file(file_path : str, read_csv_kwargs = {}, parse=True):
    path = Path(file_path)
    if path.is_file():
        path = path.parent
    
    path = path / 'ct.gz'
    if not path.exists():
        path = path.parent / 'ct'  # Check for uncompressed version
    if not path.exists():
        raise FileNotFoundError(f"Neither a ct.gz nor a ct file could be found at {file_path}")

    if path.suffix == '.gz':
        ct_file = gzip.open(path, 'rt')
    else:
        ct_file = open(path, 'r')

    ct_columns = ['prj', 'set', 'trn', 'seq', 'clk_y', 'clk_n', 'clk_d', 'clk_h', 'clk_m', 'clk_s', 'clk_f', 'tim']
    df = pd.read_csv(ct_file, sep=r'\s+', names=ct_columns, index_col=False, **read_csv_kwargs)

    ct_file.close()

    if parse:
        try:
            df = parse_CT(df) # Parse date/time and add COMP_TIME column
            df['RADAR_TIME'] = df['tim'] # Add RADAR_TIME column for consistency with stream files
        except ValueError as e:
            print(f"Error parsing CT file at {file_path}: {e}")
            raise e

    return df

def parse_CT(df):
    """
    Parse CT time headers and create COMP_TIME parsed column.
    
    CT headers: clk_y, clk_n, clk_d, clk_h, clk_m, clk_s, clk_f
    where:
    - clk_y: year
    - clk_n: month
    - clk_d: day
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
        DataFrame with added COMP_TIME column if CT columns exist, None otherwise
    """
    ct_columns = ['clk_y', 'clk_n', 'clk_d', 'clk_h', 'clk_m', 'clk_s', 'clk_f']
    
    # Check if all CT columns exist
    if not all(col in df.columns for col in ct_columns):
        return None
    
    df = df.copy()
    
    # Vectorized approach - much faster than iterrows
    # Convert fractional seconds to microseconds
    # Handle both cases: clk_f < 1 (already fractional) and clk_f >= 1 (needs modulo)
    microseconds = np.where(
        df['clk_f'] < 1,
        (df['clk_f'] * 1e6).astype('int64'),
        ((df['clk_f'] % 1) * 1e6).astype('int64')
    )
    
    # Create datetime strings in ISO format for pd.to_datetime to parse
    # This is much faster than creating datetime objects individually
    # Convert microseconds to Series for string operations
    microseconds_series = pd.Series(microseconds, index=df.index)

    if np.any(df['clk_n'] == 0):
        print("Note: Adding 1 to clk_n (month)")
        df['clk_n'] += 1 # TODO: I suspect that the month was zero-indexed in early data

    if np.any(df['clk_n'] < 1) or np.any(df['clk_n'] > 12):
        raise ValueError(f"Invalid month value in CT data. Found clk_n values: {df['clk_n'].unique()} and clk_y values: {df['clk_y'].unique()}")

    
    datetime_strings = (
        df['clk_y'].astype('int').astype('str') + '-' +
        df['clk_n'].astype('int').astype('str').str.zfill(2) + '-' +
        df['clk_d'].astype('int').astype('str').str.zfill(2) + ' ' +
        df['clk_h'].astype('int').astype('str').str.zfill(2) + ':' +
        df['clk_m'].astype('int').astype('str').str.zfill(2) + ':' +
        df['clk_s'].astype('int').astype('str').str.zfill(2) + '.' +
        microseconds_series.astype('str').str.zfill(6)
    )
    
    # Parse all datetime strings at once
    df['COMP_TIME_DT'] = pd.to_datetime(datetime_strings, format='%Y-%m-%d %H:%M:%S.%f')
    df['COMP_TIME'] = df['COMP_TIME_DT'].astype('int64') / 1e9
    
    return df

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
    elif stream_type == 'GPSkc1':
        return ['gps_clk', 'gps_tqc', 'ignored']
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


def parse_GPSnc1(df):
    """
    Parse GPSnc1 format dataframe and add LAT, LON, GPS_TIME columns.

    GPSnc1 is from the National Instruments PXI-6682/6683 timing card.
    When locked to GPS, the card reports time in the GPS time scale
    (continuous SI seconds with no leap second adjustments) as seconds
    since the NI/LabVIEW epoch (1904-01-01 00:00:00).

    The card also reports ``utc_offset``, which is TAI-UTC (the cumulative
    number of leap seconds). Since GPS = TAI - 19s (a fixed constant),
    the GPS-UTC offset is ``(utc_offset - 19)`` seconds.

    This function converts from GPS time to ANSI-C time (POSIX seconds
    since 1970-01-01 00:00:00 UTC) by subtracting the GPS-UTC offset.
    """
    df = df.copy()

    if 'lat_ang' in df.columns:
        df['LAT'] = df['lat_ang']
    else:
        raise ValueError("Column 'lat_ang' not found in GPSnc1 data")

    if 'lon_ang' in df.columns:
        df['LON'] = df['lon_ang']
    else:
        raise ValueError("Column 'lon_ang' not found in GPSnc1 data")

    # Step 1: NI/LabVIEW seconds since 1904-01-01, in GPS time scale
    ni_seconds = df['gps_time'].astype('int64') + df['gps_subsecs'].astype('uint64') / (2**64)

    # Step 2: Convert to datetime in GPS time scale
    # (pandas treats every day as 86400s, matching the GPS convention)
    gps_datetime = NI_EPOCH + pd.to_timedelta(ni_seconds, unit='s')

    # Step 3: Convert GPS datetime to UTC datetime
    # utc_offset is TAI-UTC (leap seconds); GPS-UTC = utc_offset - 19
    gps_utc_offset_s = df['utc_offset'] - 19
    utc_datetime = gps_datetime - pd.to_timedelta(gps_utc_offset_s, unit='s')

    # Step 4: ANSI-C time (seconds since 1970-01-01 00:00:00 UTC)
    df['GPS_TIME'] = (utc_datetime - UNIX_EPOCH) / pd.Timedelta('1s')

    return df


def parse_GPStp2(df):
    """
    Parse GPStp2 format dataframe and add LAT, LON, TIMESTAMP columns.
    
    GPStp2 is ASCII format from Trimble Trimflite differential GPS with:
    - Latitude and longitude already in signed decimal degrees
    - DOS Time (hh:mm:ss.s)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame from load_xds_stream_file with GPStp2 data
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
                    
                    # If we have date from CT, use it
                    if 'clk_y' in df.columns and 'clk_n' in df.columns and 'clk_d' in df.columns:
                        # Use the date from CT columns (clk_n is month, clk_d is day)
                        year = int(df.loc[df.index[0], 'clk_y'])
                        month = int(df.loc[df.index[0], 'clk_n'])
                        day = int(df.loc[df.index[0], 'clk_d'])
                        base_date = datetime(year, month, day).date()

                        dt = datetime.combine(base_date, datetime.min.time()).replace(
                            hour=hour, minute=minute, second=second, microsecond=microsecond
                        )
                        timestamps.append(dt)
                    else:
                        # Cannot create timestamp without date information
                        timestamps.append(pd.NaT)
                else:
                    timestamps.append(pd.NaT)
            else:
                timestamps.append(pd.NaT)
        except (ValueError, IndexError):
            timestamps.append(pd.NaT)

    # Only create GPS_TIME columns if we have date information
    if 'clk_y' in df.columns and 'clk_n' in df.columns and 'clk_d' in df.columns:
        df['GPS_TIME_DT'] = pd.to_datetime(timestamps)
        df['GPS_TIME'] = (df['GPS_TIME_DT'] - UNIX_EPOCH) / pd.Timedelta('1s')
    else:
        print("Warning: Cannot create GPS_TIME without date information from CT columns (clk_y, clk_n, clk_d).")
    
    return df


def parse_GPSap3(df):
    """
    Parse GPSap3 format dataframe and add LAT, LON, TIMESTAMP columns.
    
    GPSap3 is from Ashtech GG24 GPS and Glonass Navigation System with:
    - ECEF coordinates (Earth Centered Earth Fixed) in meters
    - GPS time in milliseconds of GPS week
    - Velocities in m/s
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame from load_xds_stream_file with GPSap3 data
    use_ct : bool
        If True, attempt to use CT time headers for TIMESTAMP
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added LAT, LON, TIMESTAMP columns
    """
    df = df.copy()
    
    # Convert ECEF coordinates to lat/lon using pyproj
    if all(col in df.columns for col in ['ecefx', 'ecefy', 'ecefz']):
        from pyproj import Transformer
        
        # Create transformer from ECEF (EPSG:4978) to WGS84 (EPSG:4326)
        transformer = Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
        
        # Convert ECEF to lat/lon
        # Note: pyproj expects X, Y, Z order for ECEF
        lon, lat, alt = transformer.transform(
            df['ecefx'].values,
            df['ecefy'].values, 
            df['ecefz'].values
        )
        
        df['LAT'] = lat
        df['LON'] = lon
        df['ALT'] = alt  # altitude in meters above ellipsoid
    else:
        raise ValueError("Columns 'ecefx', 'ecefy', and/or 'ecefz' not found in GPSap3 data")
    
    
    # GPS time is in milliseconds of GPS week
    # GPS epoch starts at January 6, 1980 00:00:00 UTC
    # GPS weeks start on Sunday
    
    # Note: We need to know which GPS week we're in to get absolute time
    # Without that info, we can only get time within the week
    # For now, we'll need to use CT time or warn the user
    
    if 'clk_y' in df.columns and 'clk_n' in df.columns and 'clk_d' in df.columns:
        # We have date info from CT, use it to determine GPS week
        timestamps = []
        for idx, row in df.iterrows():
            try:
                # Get the date from CT columns
                year = int(row['clk_y'])
                month = int(row['clk_n'])
                day = int(row['clk_d'])
                
                # Calculate GPS week from this date
                GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
                current_date = datetime(year, month, day)
                days_since_epoch = (current_date - GPS_EPOCH).days
                gps_week = days_since_epoch // 7
                
                # Calculate timestamp from GPS week and milliseconds
                ms_in_week = int(row['rtime'])
                total_ms_since_epoch = gps_week * 7 * 24 * 60 * 60 * 1000 + ms_in_week
                
                # Convert to datetime
                dt = GPS_EPOCH + timedelta(milliseconds=total_ms_since_epoch)
                timestamps.append(dt)
            except (ValueError, TypeError):
                timestamps.append(pd.NaT)
        
        df['GPS_TIME_DT'] = pd.to_datetime(timestamps)
    else:
        print("Warning: GPSap3 GPS time requires GPS week number or CT date info for absolute timestamps.")
        print("Using relative time within GPS week starting from Sunday 00:00:00.")
        
        # Convert milliseconds to time within the week
        ms_in_week = df['rtime'].astype('int64')
        # Assume current GPS week starts at a recent Sunday
        # This is a placeholder - user should provide actual GPS week or date
        week_start = datetime.now()
        week_start = week_start - timedelta(days=week_start.weekday() + 1)  # Previous Sunday
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

        df['GPS_TIME_DT'] = pd.to_datetime(week_start) + pd.to_timedelta(ms_in_week, unit='ms')

    df['GPS_TIME'] = (df['GPS_TIME_DT'] - UNIX_EPOCH) / pd.Timedelta('1s') # For consistency with OPR, GPS_TIME is seconds since Unix epoch

    return df


def parse_binary_AVNnp1(file_path):
    """
    Parse binary AVNnp1 IMU data from Novatel INSPVAS packets.

    Searches for Novatel OEM sync pattern (0xAA4413) and parses the complete
    packet structure including header and INSPVAS payload.

    This function screens for the Novatel Message ID and will ONLY parse Message ID 508.
    All other messages are ignored.

    Be aware that the documentation in /resfs/GROUPS/CRESIS/dataproducts/metadata/2022_Antarctica_BaslerMKB/UTIG_documentation/streams/AVNnp1
    is not correct.

    Refer to the official Novatel documentation instead:
    https://docs.novatel.com/OEM7/Content/SPAN_Logs/INSPVAS.htm
    https://docs.novatel.com/OEM7/Content/Messages/Description_of_Short_Headers.htm

    Novatel OEM Short Header (12 bytes total):
    - Sync: 0xAA4413 (3 bytes)
    - Message Length: varies (1 byte) - payload length only
    - Message ID: varies (2 bytes) - identifies the message type
    - Week Number: varies (2 bytes) - GPS week number
    - Milliseconds of Week: varies (4 bytes) - time within GPS week

    Parameters:
    -----------
    file_path : str or Path
        Path to binary AVNnp1 bxds file

    Returns:
    --------
    pd.DataFrame
        DataFrame with parsed IMU data and GPS timestamps
    """
    import struct

    file_path = Path(file_path)

    # Read binary data
    with open(file_path, 'rb') as f:
        binary_data = f.read()

    # Search for Novatel OEM sync pattern
    sync_pattern = bytes([0xAA, 0x44, 0x13])
    pos = 0
    records = []

    while True:
        # Find next sync pattern
        pos = binary_data.find(sync_pattern, pos)
        if pos == -1:
            break

        try:
            # Parse 12-byte header starting from sync pattern
            header_bytes = binary_data[pos:pos+12]
            if len(header_bytes) < 12:
                break

            # Parse Novatel OEM short header fields correctly:
            # Bytes 0-2: Sync (0xAA4413) - already found
            # Byte 3: Message Length (1 byte)
            # Bytes 4-5: Message ID (2 bytes, little-endian)
            # Bytes 6-7: Week Number (2 bytes, little-endian)
            # Bytes 8-11: Milliseconds of Week (4 bytes, little-endian)
            message_length = header_bytes[3]
            message_id = struct.unpack('<H', header_bytes[4:6])[0]
            hdr_week_number = struct.unpack('<H', header_bytes[6:8])[0]
            milliseconds_of_week = struct.unpack('<I', header_bytes[8:12])[0]

            # Validate header - reasonable message length and GPS week
            if message_length > 100000 or not (1000 <= hdr_week_number <= 3000):
                pos += 1
                continue

            # For now, only parse message type 508 (INSPVAS)
            # https://docs.novatel.com/OEM7/Content/SPAN_Logs/INSPVAS.htm
            if message_id != 508:
                pos += 1
                continue

            # Parse payload
            payload_format = '<IdddddddddddII'
            payload_size = struct.calcsize(payload_format)
            payload_start = pos + 12
            payload_bytes = binary_data[payload_start:payload_start + payload_size]
            if len(payload_bytes) < payload_size:
                break

            # Unpack INSPVAS data (Message ID: 508)
            inspvas_data = struct.unpack(payload_format, payload_bytes)

            gps_week = inspvas_data[0]
            seconds_into_week = inspvas_data[1]
            latitude_deg = inspvas_data[2]
            longitude_deg = inspvas_data[3]
            altitude_m = inspvas_data[4]
            north_velocity_ms = inspvas_data[5]
            east_velocity_ms = inspvas_data[6]
            up_velocity_ms = inspvas_data[7]
            roll_deg = inspvas_data[8]
            pitch_deg = inspvas_data[9]
            azimuth_deg = inspvas_data[10]
            imu_status = inspvas_data[11]
            # TODO: CRC ignored for now

            if hdr_week_number != gps_week:
                print(f"Warning: Header week number {hdr_week_number} does not match payload GPS week {gps_week}")
                pos += 1
                continue

            # Calculate GPS timestamp
            gps_time_from_GPS_EPOCH = (pd.Timedelta(weeks=gps_week) + pd.Timedelta(seconds=seconds_into_week))/pd.Timedelta('1s')
            gps_time = (GPS_EPOCH + pd.Timedelta(seconds=gps_time_from_GPS_EPOCH) - UNIX_EPOCH) / pd.Timedelta('1s') # For consistency with OPR, GPS_TIME is seconds since Unix epoch

            # Store record
            records.append({
                # Header fields
                'position': pos,
                'message_id': message_id,
                'message_length': message_length,

                # Time fields
                'gps_week': gps_week,
                'seconds_into_week': seconds_into_week,
                'GPS_TIME': gps_time,

                # Status
                'imu_status': imu_status,

                # Position
                'latitude_deg': latitude_deg,
                'longitude_deg': longitude_deg,
                'altitude_m': altitude_m,

                # Velocities
                'north_velocity_ms': north_velocity_ms,
                'east_velocity_ms': east_velocity_ms,
                'up_velocity_ms': up_velocity_ms,

                # Attitude
                'roll_deg': roll_deg,
                'pitch_deg': pitch_deg,
                'azimuth_deg': azimuth_deg,

                # Attitude (rad)
                'ROLL': np.deg2rad(roll_deg) if not np.isnan(roll_deg) else np.nan,
                'PITCH': np.deg2rad(pitch_deg) if not np.isnan(pitch_deg) else np.nan,
                'HEADING': np.deg2rad(azimuth_deg) if not np.isnan(azimuth_deg) else np.nan,
            })

        except (struct.error, ValueError):
            pass

        # Move to next potential sync pattern
        pos += 1

    df = pd.DataFrame(records)

    if len(df) == 0:
        print("Warning: No valid packets found")
        return df

    # Sort by GPS time
    df = df.sort_values('GPS_TIME').reset_index(drop=True)

    return df


def parse_GPSap1(df):
    """
    Parse GPSap1 format dataframe and add LAT, LON, GPS_TIME columns.
    
    GPSap1 is from Ashtech M12 GPS Navigation System with:
    - Latitude in degrees and minutes with hemisphere
    - Longitude in degrees and minutes with hemisphere
    - UTC time components (hours, minutes, seconds)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame from load_xds_stream_file with GPSap1 data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added LAT, LON, GPS_TIME columns
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
    
    # Parse UTC time components
    timestamps = []
    for idx, row in df.iterrows():
        try:
            hour = int(row['utc_h'])
            minute = int(row['utc_m'])
            second_float = float(row['utc_s'])
            second = int(second_float)
            microsecond = int((second_float % 1) * 1e6)
            
            # If we have date from CT, use it
            if 'clk_y' in df.columns and 'clk_n' in df.columns and 'clk_d' in df.columns:
                # Use the date from CT columns (clk_n is month, clk_d is day)
                year = int(row['clk_y'])
                month = int(row['clk_n'])
                day = int(row['clk_d'])
                base_date = datetime(year, month, day).date()

                dt = datetime.combine(base_date, datetime.min.time()).replace(
                    hour=hour, minute=minute, second=second, microsecond=microsecond
                )
                timestamps.append(dt)
            else:
                # Cannot create timestamp without date information
                timestamps.append(pd.NaT)
        except (ValueError, TypeError):
            timestamps.append(pd.NaT)

    # Only create GPS_TIME columns if we have date information
    if 'clk_y' in df.columns and 'clk_n' in df.columns and 'clk_d' in df.columns:
        df['GPS_TIME_DT'] = pd.to_datetime(timestamps)
        df['GPS_TIME'] = (df['GPS_TIME_DT'] - UNIX_EPOCH) / pd.Timedelta('1s')
    else:
        print("Warning: Cannot create GPS_TIME without date information from CT columns (clk_y, clk_n, clk_d).")
    
    return df


def parse_GPSkc1(df):
    """
    Parse GPSkc1 format dataframe and add GPS_TIME columns.

    GPSkc1 is from Kinemetrics TrueTime 705-101 GPS Time Code Generator with:
    - gps_clk: GPS time in jjj:hh:mm:ss format (Julian day:hours:minutes:seconds)
    - gps_tqc: Time quality character (SP, ., *, #, ?)

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame from load_xds_stream_file with GPSkc1 data

    Returns:
    --------
    pandas.DataFrame
        DataFrame with added GPS_TIME and GPS_TIME_DT columns
    """
    df = df.copy()

    if 'gps_clk' not in df.columns:
        raise ValueError("Column 'gps_clk' not found in GPSkc1 data")

    # Parse gps_clk field (jjj:hh:mm:ss format) using vectorized string operations
    # Strip any leading or trailing whitespace or underscores
    df['gps_clk'] = df['gps_clk'].astype(str).str.strip().str.replace('_', '')
    # Split the clock string into components
    clk_parts = df['gps_clk'].astype(str).str.split(':', expand=True)

    # Extract each component as integers
    day_of_year = pd.to_numeric(clk_parts[0], errors='coerce')
    hour = pd.to_numeric(clk_parts[1], errors='coerce')
    minute = pd.to_numeric(clk_parts[2], errors='coerce')
    second = pd.to_numeric(clk_parts[3], errors='coerce')

    # Calculate timedelta from start of year in seconds (vectorized)
    timedelta_seconds = (
        (day_of_year - 1) * 24 * 60 * 60 +
        hour * 60 * 60 +
        minute * 60 +
        second
    )

    # Check if clk_y field exists (calendar year from CT file)
    if 'clk_y' in df.columns:
        # Create datetime strings for each year start
        year_start_strings = df['clk_y'].astype('int').astype('str') + '-01-01'
        year_starts = pd.to_datetime(year_start_strings, format='%Y-%m-%d')

        # Add timedelta to year start (vectorized)
        df['GPS_TIME_DT'] = year_starts + pd.to_timedelta(timedelta_seconds, unit='s')
        df['GPS_TIME'] = (df['GPS_TIME_DT'] - UNIX_EPOCH) / pd.Timedelta('1s')
    else:
        print("Warning: 'clk_y' field not found. Cannot create GPS_TIME without calendar year information.")

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

