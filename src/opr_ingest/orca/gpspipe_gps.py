"""Loader for ORCA `<prefix>_gpspipe_stdout.log` files.

Each line of the log is gpsd output with a leading UNIX timestamp and an
embedded JSON TPV record (lat/lon/alt/altMSL/ecef*). See
reference/gps_processing.py for the original parser.
"""

from pathlib import Path
from typing import Union

import pandas as pd


def load_and_parse_gpspipe_file(path: Union[str, Path]) -> pd.DataFrame:
    """Return a DataFrame with OPR-standard columns from one gpspipe log.

    Expected output columns:
        GPS_TIME   : seconds since Unix epoch (from the leading UNIX timestamp)
        LAT, LON   : decimal degrees
        ELEV       : meters (prefer altMSL over alt)
        ROLL/PITCH/HEADING: zeros — gpspipe TPV records carry no attitude
        (pass-through) ecefx, ecefy, ecefz from the TPV record
    """
    raise NotImplementedError(
        "ORCA gpspipe_gps.load_and_parse_gpspipe_file: port the regex-based "
        "TPV extraction from reference/gps_processing.py:parse_line, then map "
        "to OPR columns. Open questions: leap-second handling vs UTIG GPS_TIME, "
        "and how to source RADAR_TIME (likely needs the UHD log)."
    )
