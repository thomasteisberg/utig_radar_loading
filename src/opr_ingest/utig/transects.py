"""Per-transect helpers: start/end timestamps, post-processed GPS path attachment, GPS loading."""

import glob
import gzip
import os
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from opr_ingest.utig import stream_util
from opr_ingest.utig.postprocessed_gps import load_and_parse_postprocessed_gps_file


def get_start_timestamp(transect):
    """Start timestamp for a transect.

    Priority:
      1. Post-processed GPS (GPS_TIME, most accurate)
      2. Field GPS context file (COMP_TIME_DT, radar computer time)
    """
    if 'postprocessed_gps_path' in transect and pd.notna(transect['postprocessed_gps_path']):
        try:
            gps_df = load_and_parse_postprocessed_gps_file(transect['postprocessed_gps_path'])
            if len(gps_df) > 0:
                return pd.Timestamp.fromtimestamp(gps_df.iloc[0]['GPS_TIME'], tz='UTC')
        except Exception as e:
            print(f"Warning: Could not load postprocessed GPS for {transect.name}, falling back to field GPS: {e}")

    fp = transect['gps_path']
    if isinstance(fp, float) and np.isnan(fp):
        return None

    ct_df = stream_util.load_ct_file(fp, read_csv_kwargs={'nrows': 1}, parse=True)
    return ct_df.iloc[0]['COMP_TIME_DT']


def get_end_timestamp(transect):
    fp = transect['gps_path']

    with gzip.open(fp, 'rb') as f:
        f.seek(-2, os.SEEK_END)
        while f.read(1) != b'\n':
            f.seek(-2, os.SEEK_CUR)
        last_line = f.readline().decode()

    ct_columns = ['prj', 'set', 'trn', 'seq', 'clk_y', 'clk_n', 'clk_d', 'clk_h', 'clk_m', 'clk_s', 'clk_f', 'tim']
    ct_df = pd.read_csv(StringIO(last_line), sep=r'\s+', names=ct_columns, index_col=False)
    ct_df = stream_util.parse_CT(ct_df)
    return ct_df.iloc[0]['COMP_TIME']


def add_postprocessed_gps_paths(df_season, season_gps_postprocessed_dir, ignore_set: bool = False):
    """Attach `postprocessed_gps_path`/`postprocessed_gps_type` columns to a transect frame.

    Looks under `season_gps_postprocessed_dir` for `*PUT[GN]1B_*_PRJ_SET_TRN_position.txt`.
    """
    df_season = df_season.copy()
    df_season['postprocessed_gps_path'] = np.nan
    df_season['postprocessed_gps_type'] = np.nan

    gps_files = glob.glob(f"{season_gps_postprocessed_dir}/**/*PUT[GN]1B_*_position.txt")
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


def load_gps_data(transects_df, source_type=None):
    """Load per-transect GPS data into a list of DataFrames suitable for plotting / track length.

    Each row picks postprocessed GPS if a path is present, else field GPS via stream_util.
    Override per-call with `source_type='field'` or `'postprocessed'`.
    """
    segment_dfs = []

    for idx, row in transects_df.iterrows():

        if source_type is None:
            tmp_source_type = 'postprocessed' if pd.notna(row['postprocessed_gps_path']) else 'field'
        else:
            tmp_source_type = source_type

        if tmp_source_type == 'field':
            df = stream_util.load_xds_stream_file(row['gps_path'], parse=True)
        elif tmp_source_type == 'postprocessed':
            df = load_and_parse_postprocessed_gps_file(row['postprocessed_gps_path'])
            df['prj'] = idx[0]
            df['set'] = idx[1]
            df['trn'] = idx[2]
        else:
            raise ValueError(f"Unknown source_type {tmp_source_type}")

        necessary_keys = ['prj', 'set', 'trn', 'clk_y', 'LAT', 'LON', 'TIMESTAMP']
        for k in necessary_keys:
            if k not in df:
                df[k] = np.nan

        df_sub = df[['prj', 'set', 'trn', 'clk_y', 'LAT', 'LON', 'TIMESTAMP']]

        if 'segment_path' in row:
            df_sub['segment_path'] = row['segment_path']

        segment_dfs.append(df_sub)
    return segment_dfs
