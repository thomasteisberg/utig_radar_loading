"""Segment-assignment for ORCA recordings.

Each `<prefix>_rx_samps.bin` is one continuous capture (operator stops and
restarts produce a new prefix with a new timestamp), so every ORCA recording
is its own segment by definition — no gap detection needed, unlike UTIG where
many tiny streaming files have to be regrouped by clock gaps.

This function just sorts by timestamp and assigns the `segment_date_str` /
`segment_number` / `segment_path` columns downstream stages rely on, matching
the schema produced by `opr_ingest.utig.segment_splits.assign_segments`.
"""

import pandas as pd


def assign_segments(df_recordings: pd.DataFrame, timestamp_field: str = "timestamp") -> pd.DataFrame:
    """Assign one segment per recording, numbered within day.

    Returns a copy of `df_recordings` sorted by `timestamp_field` with three
    new columns:
        segment_date_str  YYYYMMDD
        segment_number    1-based, resets each new date
        segment_path      "<segment_date_str>_<NN>"
    """
    df = df_recordings.sort_values(timestamp_field).copy()
    df["segment_date_str"] = df[timestamp_field].dt.strftime("%Y%m%d")
    df["segment_number"] = df.groupby("segment_date_str").cumcount() + 1
    df["segment_path"] = (
        df["segment_date_str"] + "_" + df["segment_number"].astype(str).str.zfill(2)
    )
    return df
