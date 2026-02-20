import pandas as pd
import glob
import os
import numpy as np
import gzip
from pathlib import Path

import utig_radar_loading.stream_util as stream_util
import utig_radar_loading.opr_gps_file_generation as opr_gps_file_generation

#
# df_files
# --------------------------------
# Functions for loading raw dataframe of all files
#

def load_file_index_df(base_path : str, cache_file : str, read_cache : bool = True) -> pd.DataFrame:
    cache_path = Path(cache_file)
    if read_cache and cache_path.exists():
        # Read from cache
        print(f"Reading from cache file {cache_path}")
        df_files = pd.read_csv(cache_path, index_col=0)
        df_files.columns = df_files.columns.astype(int)
    else:
        # Load and process files to create DataFrame
        print(f"Generating file index")
        print(f"Looking for xds.gz")
        xds_files = glob.glob(f"{base_path}/**/xds.gz", recursive=True)
        print(f"Looking for bxds")
        bxds_files = glob.glob(f"{base_path}/**/bxds*", recursive=True)
        df_files = pd.DataFrame([Path(f).parts for f in (xds_files + bxds_files)])
        
        if cache_file is not None:
            print(f"Saving new cache to {cache_file}")
            df_files.to_csv(cache_path)
    
    return df_files

#
# df_artifacts
# --------------------------------
# Functions for processing df_files into df_artifacts
#

def create_artifacts_df(file_index_df : pd.DataFrame, datasets=['UTIG1', 'UTIG2']) -> pd.DataFrame:
    column_mapping = {
        5: "dataset",
        6: "processing_level",
        7: "processing_type",
        8: "prj",
        9: "set",
        10: "trn",
        11: "stream",
        12: "file_name",
        "full_path": "full_path"
        }

    df_tmp = file_index_df.copy()

    df_tmp = df_tmp.dropna(axis='columns')

    df_tmp["full_path"] = df_tmp.apply(lambda row: Path(*row).as_posix(), axis=1)

    df_tmp = df_tmp[list(column_mapping.keys())]
    df_artifacts = df_tmp.rename(columns=column_mapping)

    if datasets is not None:
        df_artifacts = df_artifacts[df_artifacts['dataset'].isin(datasets)]

    #df_artifacts = df_tmp.drop(columns=[0,1,2,3,4,12,13])

    df_artifacts['artifact'] = df_artifacts.apply(lambda row: tuple(row[['processing_level', 'processing_type', 'stream']]), axis='columns')

    return df_artifacts

#
# df_transects
# --------------------------------
# Functions for grouping df_artifacts by transect and selecting desired streams
#

def arrange_by_transect(df_artifacts, streams, ignore_set=False):
    """
    Group by transects (unique combinations of (prj, set, trn)) and pull out paths
    to the desired data streams.

    If ignore_set is True, then transects will be grouped by only prj and trn.

    streams is a dictionary mapping names of data categories to a list of acceptable
    stream types. For example:
    { "gps": ["GPSnc1", "GPSnc2"],
      "radar": ["RADnh5", "RADnh6"] }

    The resulting dataframe will have two columns per entry in the streams dictionary:
    <data category>_stream_type will contain the matched stream type and
    <data category>_path will contain the path to the data file.

    If multiple matching stream types are available, preference will be given to the
    first stream type in the list. If no matching stream types are available, columns
    will be filled with NaN.
    """
    
    def agg_fn(group):
        df = pd.DataFrame(index=[0])
        
        # Look for requested data streams
        for data_category in streams.keys():
            df[f"{data_category}_stream_type"] = np.nan
            df[f"{data_category}_path"] = np.nan
            
            matching_entry = group[(group['stream'].isin(streams[data_category]['stream_types'])) & \
                (group['file_name'].isin(streams[data_category]['file_names']))]
            if not matching_entry.empty:
                # Sort matching_entry by preferred stream type order
                preferred_order = {stype: i for i, stype in enumerate(streams[data_category]['stream_types'])}
                matching_entry['preferred_order'] = matching_entry['stream'].map(preferred_order)
                matching_entry = matching_entry.sort_values('preferred_order')

                df[f"{data_category}_stream_type"] = matching_entry['stream'].values[0]
                df[f"{data_category}_path"] = matching_entry['full_path'].values[0]

        # Add in any other unique keys
        for k in group:
            if k in ['full_path', 'stream', 'processing_level', 'processing_type']:
                continue
            
            if len(group[k].unique()) == 1:
                df[k] = str(group[k].values[0]) # TODO

        return df

    if ignore_set:
        groupby_keys = ['prj', 'trn']
    else:
        groupby_keys = ['prj', 'set', 'trn']
    
    df = df_artifacts.groupby(groupby_keys).apply(agg_fn, include_groups=False)
    df.index = df.index.droplevel(-1)
    return df

def get_start_timestamp(transect):
    """
    Get the start timestamp for a transect.

    Priority:
    1. If postprocessed_gps_path is available, use GPS_TIME from that (most accurate)
    2. Otherwise, use COMP_TIME_DT from field GPS context file (radar computer time)
    """

    # Try postprocessed GPS first (most accurate)
    if 'postprocessed_gps_path' in transect and pd.notna(transect['postprocessed_gps_path']):
        try:
            gps_df = opr_gps_file_generation.load_and_parse_postprocessed_gps_file(transect['postprocessed_gps_path'])
            if len(gps_df) > 0:
                # Convert GPS_TIME (Unix epoch seconds) to datetime
                return pd.Timestamp.fromtimestamp(gps_df.iloc[0]['GPS_TIME'])
        except Exception as e:
            print(f"Warning: Could not load postprocessed GPS for {transect.name}, falling back to field GPS: {e}")
            # Fall through to field GPS

    # Fall back to field GPS context file (radar computer time)
    fp = transect['gps_path']
    if isinstance(fp, float) and np.isnan(fp):
        return None

    ct_df = stream_util.load_ct_file(fp, read_csv_kwargs={'nrows': 1}, parse=True)

    return ct_df.iloc[0]['COMP_TIME_DT']

def get_end_timestamp(transect):
    fp = transect['gps_path']
    
    # Read last few bytes and extract last line
    with gzip.open(fp, 'rb') as f:
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR)
        last_line = f.readline().decode()
    
    # Load and parse just the last line
    from io import StringIO
    ct_columns = ['prj', 'set', 'trn', 'seq', 'clk_y', 'clk_n', 'clk_d', 'clk_h', 'clk_m', 'clk_s', 'clk_f', 'tim']
    ct_df = pd.read_csv(StringIO(last_line), sep=r'\s+', names=ct_columns, index_col=False)
    ct_df = stream_util.parse_CT(ct_df)
    return ct_df.iloc[0]['COMP_TIME']

def season_from_datetime(d):
    if d.month >= 6:
        return d.year
    else:
        return d.year - 1

def assign_seasons(df_transects):
    df_all_seasons = df_transects.copy()
    df_all_seasons['start_timestamp'] = df_all_seasons.apply(get_start_timestamp, axis=1)
    df_all_seasons['season'] = df_all_seasons['start_timestamp'].apply(season_from_datetime)
    df_all_seasons['season'] = df_all_seasons['season'].astype('Int32')
    df_all_seasons = df_all_seasons.sort_values('prj')
    return df_all_seasons

def add_postprocessed_gps_paths(df_season, season_gps_postprocessed_dir, ignore_set : bool = False):
    # Add post-processed GPS paths if available
    # Files should be named as EPUTG1B_XXXX_PRJ_SET_TRN_position.txt

    df_season = df_season.copy()
    df_season['postprocessed_gps_path'] = np.nan
    df_season['postprocessed_gps_type'] = np.nan

    gps_files = glob.glob(f"{season_gps_postprocessed_dir}/*PUTG1B_*_position.txt")
    for f in gps_files:
        parts = Path(f).stem.split('_')
        if len(parts) < 6:
            continue
        prj = parts[-4]
        set_ = parts[-3]
        trn = parts[-2]
        key = (prj, set_, trn)

        if ignore_set:
            matching_keys = [k for k in df_season.index if k[0] == prj and k[2] == trn]
            if len(matching_keys) == 0:
                continue
            elif len(matching_keys) > 1:
                print(f"Warning: Multiple matching keys found for prj={prj}, trn={trn} when ignoring set")
                for mk in matching_keys:
                    df_season.loc[mk, 'postprocessed_gps_path'] = f
                    df_season.loc[mk, 'postprocessed_gps_type'] = 'EPUTG1B'
                continue
            else:
                key = matching_keys[0]

        df_season.loc[key, 'postprocessed_gps_path'] = f
        df_season.loc[key, 'postprocessed_gps_type'] = 'EPUTG1B'

    return df_season