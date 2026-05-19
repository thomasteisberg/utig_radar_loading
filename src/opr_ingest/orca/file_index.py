"""Discover ORCA recordings on disk and build a DataFrame indexed by prefix.

ORCA recordings are flat groups of files sharing a `YYYYMMDD_HHMMSS` prefix:
    <prefix>_rx_samps.bin     # raw RX samples (binary)
    <prefix>_config.yaml      # capture config (CHIRP / DEVICE / GENERATE sections)
    <prefix>_uhd_stdout.log   # UHD radar stdout log
    <prefix>_gpspipe_stdout.log  # gpsd / gpspipe JSON log
"""

from pathlib import Path

import pandas as pd


PREFIX_PATTERN = r"\d{8}_\d{6}"


def load_file_index_df(base_path: str, cache_file: str, read_cache: bool = True) -> pd.DataFrame:
    """Build (or load from cache) a DataFrame of every ORCA recording under `base_path`.

    Each row corresponds to one recording prefix and carries the paths of its
    associated files (rx_samps, config, uhd_log, gpspipe_log) plus a parsed
    `timestamp` derived from the prefix.
    """
    raise NotImplementedError(
        "ORCA file_index.load_file_index_df: glob <base_path>/**/*_rx_samps.bin, derive "
        "prefix from basename, collect sibling files. See reference/processing_dask.py "
        "for the prefix/file-naming convention."
    )


def arrange_by_transect(df_recordings: pd.DataFrame) -> pd.DataFrame:
    """For API parity with the UTIG pipeline. ORCA's transect unit is the prefix itself.

    Should return a DataFrame indexed by `prefix` with the columns that downstream
    stages need (radar_path, gps_path, start_timestamp).
    """
    raise NotImplementedError(
        "ORCA file_index.arrange_by_transect: confirm grouping semantics. Likely "
        "this is identity (one prefix == one transect)."
    )
