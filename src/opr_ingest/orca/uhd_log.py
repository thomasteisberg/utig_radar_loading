"""Helpers for parsing ORCA `<prefix>_uhd_stdout.log` files.

Every UHD stdout line is prefixed with the host's Unix-time wall clock when
the line was logged, e.g. ``[1764886465.196]\\t[START] Beginning main loop``.
The ``[START]`` marker (or ``Scheduling chirp 0 RX``, for older variants)
gives the host-clock time at which the radar began transmitting / receiving.
"""

import re
from pathlib import Path
from typing import Optional, Union


_START_LINE_RE = re.compile(r"^\[(\d+\.\d+)\].*(?:\[START\]|Scheduling chirp 0 RX)")


def parse_start_timestamp(uhd_log_path: Union[str, Path]) -> Optional[float]:
    """Return the host Unix timestamp from the `[START]` line of a UHD stdout log.

    Returns None if the file can't be read or contains no `[START]` marker.
    """
    try:
        with open(uhd_log_path) as f:
            for line in f:
                m = _START_LINE_RE.match(line)
                if m:
                    return float(m.group(1))
    except OSError:
        return None
    return None
