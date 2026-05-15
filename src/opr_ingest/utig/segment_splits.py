"""Segment-assignment for UTIG transects.

Uses gaps in a CT timestamp field (default: `tim`, the radar clock) to detect
segment boundaries and assign a `<YYYYMMDD>_NN` segment label to each row.
"""

import numpy as np
from tqdm import tqdm

from opr_ingest.utig import stream_util


def assign_segments(df_season, timestamp_field='tim', timestamp_split_threshold=1000, parse_ct=False, progress=False):
    """Assign segment paths to each row of `df_season` based on gaps in the CT timestamp field.

    Parameters
    ----------
    df_season : DataFrame with `start_timestamp` and `radar_path` columns
    timestamp_field : CT column to inspect for gaps (default `tim`)
    timestamp_split_threshold : gap (in `tim` units) that triggers a new segment
    parse_ct : pass through to load_ct_file
    progress : show tqdm

    Returns df with added `segment_path`, `segment_date_str`, `segment_number`.
    """
    df_season = df_season.sort_values(by='start_timestamp')

    last_segment_ct = stream_util.load_ct_file(df_season.iloc[0]['radar_path'], parse=parse_ct)

    df_season['segment_path'] = ""
    df_season['segment_date_str'] = ""
    df_season['segment_number'] = -1
    current_segment_datestring = df_season.iloc[0]['start_timestamp'].strftime("%Y%m%d")
    current_segment_idx = 1

    df_season.iloc[0, df_season.columns.get_loc('segment_date_str')] = current_segment_datestring
    df_season.iloc[0, df_season.columns.get_loc('segment_path')] = f"{current_segment_datestring}_{current_segment_idx:02d}"
    df_season.iloc[0, df_season.columns.get_loc('segment_number')] = current_segment_idx

    print(f"Initial segment path is: {df_season.iloc[0]['segment_path']}")

    iterator = range(1, len(df_season))
    if progress:
        iterator = tqdm(iterator)

    for row_iloc in iterator:
        try:
            curr_segment_ct = stream_util.load_ct_file(df_season.iloc[row_iloc]['radar_path'], parse=parse_ct)

            delta_from_last = curr_segment_ct[timestamp_field].iloc[0] - last_segment_ct[timestamp_field].iloc[-1]

            if np.abs(delta_from_last) > timestamp_split_threshold:
                new_datestring = df_season.iloc[row_iloc]['start_timestamp'].strftime("%Y%m%d")
                if new_datestring == current_segment_datestring:
                    current_segment_idx += 1
                else:
                    current_segment_idx = 1
                    current_segment_datestring = new_datestring

                print(f"Segment path changed to {current_segment_datestring}_{current_segment_idx:02d}. Delta in '{timestamp_field}' was {delta_from_last}")

            df_season.iloc[row_iloc, df_season.columns.get_loc('segment_date_str')] = current_segment_datestring
            df_season.iloc[row_iloc, df_season.columns.get_loc('segment_path')] = f"{current_segment_datestring}_{current_segment_idx:02d}"
            df_season.iloc[row_iloc, df_season.columns.get_loc('segment_number')] = current_segment_idx

            last_segment_ct = curr_segment_ct
        except Exception as e:
            print(f"Could not load index {row_iloc}")
            print(df_season.iloc[row_iloc]['radar_path'])
            print(e)

    return df_season
