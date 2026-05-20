"""Discover ORCA recordings on disk and build a DataFrame indexed by prefix.

ORCA recordings live as flat groups of files sharing a `YYYYMMDD_HHMMSS` prefix.
Discovery globs `*_config.yaml` (every recording has exactly one) and derives
the prefix from that. Sibling files per prefix:

Required:
    <prefix>_rx_samps.bin        # raw RX samples (singleton; produced by merging
                                 # `_pN_rx_samps.bin` / `_rx_samps.bin.N` parts)
    <prefix>_uhd_stdout.log      # UHD radar stdout log
    <prefix>_gpspipe_stdout.log  # gpsd / gpspipe JSON log

Optional:
    <prefix>_gpsrinex.obs        # RINEX-format GNSS observations
    <prefix>_track.gpx           # Precomputed GPX track log

The multi-part `_pN_rx_samps.bin` and `_rx_samps.bin.N` files are intermediate
artifacts merged into the singleton `_rx_samps.bin` by a separate pre-processing
step, so they're ignored here.

If a prefix doesn't parse as `YYYYMMDD_HHMMSS` (e.g., `20251210_999999`), the
timestamp is recovered from the `[START]` line in the UHD log.
"""

import glob
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from opr_ingest.orca.uhd_log import parse_start_timestamp


PREFIX_PATTERN = r"\d{8}_\d{6}"
_PREFIX_RE = re.compile(rf"^{PREFIX_PATTERN}$")

_REQUIRED_SIBLINGS = {
    "rx_samps_path": "_rx_samps.bin",
    "uhd_log_path": "_uhd_stdout.log",
    "gpspipe_log_path": "_gpspipe_stdout.log",
}
_OPTIONAL_SIBLINGS = {
    "gpsrinex_path": "_gpsrinex.obs",
    "track_gpx_path": "_track.gpx",
}
_ALL_SIBLINGS = {**_REQUIRED_SIBLINGS, **_OPTIONAL_SIBLINGS}


def _recover_timestamp_from_uhd_log(uhd_log_path):
    """Return a naive-UTC datetime from the `[START]` line of a UHD stdout log, or None."""
    start_t = parse_start_timestamp(uhd_log_path)
    if start_t is None:
        return None
    return datetime.fromtimestamp(start_t, tz=timezone.utc).replace(tzinfo=None)


def load_file_index_df(base_path: str, cache_file: str, read_cache: bool = True) -> pd.DataFrame:
    """Build (or load from cache) a DataFrame of every ORCA recording under `base_path`."""
    cache_path = Path(cache_file) if cache_file is not None else None
    if read_cache and cache_path is not None and cache_path.exists():
        print(f"Reading from cache file {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, low_memory=False)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    print(f"Generating ORCA file index under {base_path}")
    config_files = sorted(glob.glob(f"{base_path}/**/*_config.yaml", recursive=True))
    print(f"Found {len(config_files)} *_config.yaml files")

    rows = []
    for cfg_path in config_files:
        p = Path(cfg_path)
        prefix = p.name[: -len("_config.yaml")]
        if not _PREFIX_RE.match(prefix):
            continue

        row = {"prefix": prefix, "config_path": str(p)}
        for col, suf in _ALL_SIBLINGS.items():
            row[col] = str(p.with_name(prefix + suf))

        empty = []
        for col in (*_REQUIRED_SIBLINGS, "config_path"):
            sib = Path(row[col])
            if not sib.exists():
                print(f"  warning: {prefix} missing {col} ({row[col]})")
            elif sib.stat().st_size == 0:
                empty.append(col)
        if empty:
            # Zero-byte required siblings mean the capture never produced data
            # (rx_samps never written, UHD log empty so no [START], or config
            # truncated). Drop these — no downstream stage can use them.
            print(f"  skipping {prefix}: empty {', '.join(empty)}")
            continue

        try:
            ts = datetime.strptime(prefix, "%Y%m%d_%H%M%S")
        except ValueError:
            ts = _recover_timestamp_from_uhd_log(row["uhd_log_path"])
            if ts is None:
                print(f"  skipping {prefix}: bad prefix and no [START] in UHD log")
                continue
            print(f"  {prefix}: recovered timestamp {ts.isoformat()} from UHD log")

        row["timestamp"] = ts
        rows.append(row)

    df = pd.DataFrame(
        rows,
        columns=[
            "prefix",
            "rx_samps_path",
            "config_path",
            "uhd_log_path",
            "gpspipe_log_path",
            "gpsrinex_path",
            "track_gpx_path",
            "timestamp",
        ],
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving new cache to {cache_path}")
        df.to_csv(cache_path)

    return df


def arrange_by_transect(df_recordings: pd.DataFrame) -> pd.DataFrame:
    """For API parity with the UTIG pipeline. ORCA's transect unit is the prefix itself.

    Returns a DataFrame indexed by `prefix` with aliased columns
    (`radar_path`, `gps_path`, `start_timestamp`) the downstream stages expect,
    while keeping the original sibling-path columns for stages that need them.
    """
    df = df_recordings.copy().set_index("prefix")
    df["radar_path"] = df["rx_samps_path"]
    df["gps_path"] = df["gpspipe_log_path"]
    df["start_timestamp"] = df["timestamp"]
    return df
