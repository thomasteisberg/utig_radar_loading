"""Loader for UTIG post-processed GPS position files (EPUTG1B / IPUTG1B / IPUTN1B / SPUTG1B)."""

import warnings
from pathlib import Path
from typing import Union

import astropy.time
import numpy as np
import pandas as pd


def load_and_parse_postprocessed_gps_file(gps_path: Union[str, Path]) -> pd.DataFrame:
    columns = "YEAR DOY SOD SEQ LON LAT AC_ELEVATION ROLL PITCH HEADING STDDEV NS_ACCELERATION EW_ACCELERATION VERT_ACCELERATION".split(" ")

    # 2008/2015 files have a "# ... <PREFIX> data format" header; 2009 has no
    # header so the filename is the only signal.
    known_prefixes = ("EPUTG1B", "SPUTG1B", "IPUTG1B", "IPUTN1B")
    name = str(gps_path)
    with open(gps_path, 'r') as f:
        first_line = f.readline().strip()
    if not any(p in name or p in first_line for p in known_prefixes):
        warnings.warn(f"File {gps_path} does not appear to be in expected post-processed GPS format (no known prefix in filename or header)")

    df = pd.read_csv(gps_path, sep=r'\s+', names=columns, comment='#', header=None)

    # Build GPS_TIME from YEAR/DOY/SOD (input is UTC); convert to ANSI-C
    # (Unix-epoch seconds) via astropy.
    dt = pd.to_datetime(df['YEAR'].astype(str) + df['DOY'].astype(str).str.zfill(3), format='%Y%j') + pd.to_timedelta(df['SOD'], unit='s')
    t_utc = astropy.time.Time(dt, format='datetime64', scale='utc')
    df['GPS_TIME'] = t_utc.unix

    df['PITCH'] = np.deg2rad(df['PITCH'])
    df['ROLL'] = np.deg2rad(df['ROLL'])
    df['HEADING'] = np.deg2rad(df['HEADING'])

    df = df.rename(columns={'AC_ELEVATION': 'ELEV'})

    return df
