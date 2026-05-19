"""Segment-assignment for ORCA recordings.

ORCA has no CT-style timing files; segments are inferred from gaps in the
prefix-derived timestamp series.
"""

import pandas as pd


def assign_segments(
    df_recordings: pd.DataFrame,
    gap_threshold_s: float = 3600.0,
    timestamp_field: str = "timestamp",
):
    """Assign segment columns based on gaps in `timestamp_field`.

    Adds `segment_date_str`, `segment_number`, `segment_path` columns to match
    the UTIG schema produced by `opr_ingest.utig.segment_splits.assign_segments`.
    """
    raise NotImplementedError(
        "ORCA segment_splits.assign_segments: sort by timestamp, declare a new "
        "segment when delta > gap_threshold_s, fill segment_date_str / "
        "segment_number / segment_path. TODO: confirm the right gap threshold "
        "for ORCA flights."
    )
