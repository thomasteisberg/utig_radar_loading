"""Loader for ORCA `<prefix>_gpspipe_stdout.log` files.

Each TPV line is shaped like::

    2025-12-04 22:14:15 1764886455.314251: {"class":"TPV", ..., "time":"2025-12-04T22:14:24.000Z", "lat":..., ...}

The leading float is the host's wall-clock Unix time when gpsd delivered the
line (COMP_TIME). The TPV JSON ``time`` field is the GPS receiver's UTC fix
time (GPS_TIME). Both are emitted so downstream alignment can interpolate
radar pulse times (in host-clock space) onto GPS positions despite any host
clock drift.

Ported from the regex/JSON extractor in `reference/gps_processing.py` and
extended to also surface the TPV ``time``, ``mode``, ``speed``, and ``track``
fields.
"""

import json
import re
from pathlib import Path
from typing import Union

import pandas as pd


_LINE_RE = re.compile(r"(\d+\.\d+):\s*(\{.*\})\s*$")

_OUTPUT_COLUMNS = [
    "COMP_TIME",
    "GPS_TIME",
    "LAT",
    "LON",
    "ELEV",
    "ecefx",
    "ecefy",
    "ecefz",
    "speed",
    "track",
    "mode",
]


def load_and_parse_gpspipe_file(path: Union[str, Path]) -> pd.DataFrame:
    """Parse a `_gpspipe_stdout.log` file into a DataFrame of TPV records."""
    rows = []
    with open(path) as f:
        for line in f:
            if '"class":"TPV"' not in line:
                continue
            m = _LINE_RE.search(line)
            if not m:
                continue
            try:
                tpv = json.loads(m.group(2))
            except json.JSONDecodeError:
                continue
            if tpv.get("class") != "TPV":
                continue

            gps_iso = tpv.get("time")
            if gps_iso is None:
                continue
            try:
                gps_time = pd.Timestamp(gps_iso).timestamp()
            except (ValueError, TypeError):
                continue

            elev = tpv.get("altMSL")
            if elev is None:
                elev = tpv.get("alt")

            rows.append({
                "COMP_TIME": float(m.group(1)),
                "GPS_TIME": gps_time,
                "LAT": tpv.get("lat"),
                "LON": tpv.get("lon"),
                "ELEV": elev,
                "ecefx": tpv.get("ecefx"),
                "ecefy": tpv.get("ecefy"),
                "ecefz": tpv.get("ecefz"),
                "speed": tpv.get("speed"),
                "track": tpv.get("track"),
                "mode": tpv.get("mode"),
            })

    return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
