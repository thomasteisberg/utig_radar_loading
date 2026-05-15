"""Epoch constants and timestamp-epoch detection utility.

`GPS_TIME` in OPR is ANSI-C time (seconds since the Unix epoch), not seconds
since the GPS epoch — see the OPR GPS-File-Guide.
"""

import pandas as pd
from typing import Optional, Tuple

UNIX_EPOCH = pd.to_datetime('1970-01-01 00:00:00')
GPS_EPOCH = pd.to_datetime('1980-01-06 00:00:00')
NI_EPOCH = pd.to_datetime('1904-01-01 00:00:00')
# See https://www.ni.com/en/support/documentation/supplemental/08/labview-timestamp-overview.html


def parse_timestamp_with_epoch(
    timestamp: float,
    expected_year: int
) -> Tuple[Optional[str], Optional[pd.Timestamp]]:
    """Convert a timestamp to a datetime under several epoch assumptions.

    Returns (epoch_name, parsed_datetime) for whichever epoch produces
    the expected year, else (None, None).
    """
    epochs = {
        'unix': pd.Timestamp('1970-01-01 00:00:00', tz='UTC'),
        'gps': pd.Timestamp('1980-01-06 00:00:00', tz='UTC'),
        'mac': pd.Timestamp('1904-01-01 00:00:00', tz='UTC'),
    }

    td = pd.Timedelta(seconds=timestamp)

    print(f"\nParsing timestamp {timestamp} (expecting year {expected_year}):")
    print("-" * 60)

    matched_epoch = None
    matched_datetime = None

    for epoch_name, epoch_start in epochs.items():
        try:
            dt = epoch_start + td
            print(f"{epoch_name.upper():8} epoch: {dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC")
            if dt.year == expected_year:
                print(f"   Matches expected year {expected_year}")
                matched_epoch = epoch_name
                matched_datetime = dt
        except (ValueError, OverflowError) as e:
            print(f"{epoch_name.upper():8} epoch: Error - {str(e)}")

    if matched_epoch:
        print(f"\nReturning: {matched_epoch} epoch")
    else:
        print(f"\nNo epoch yielded year {expected_year}")

    return matched_epoch, matched_datetime
