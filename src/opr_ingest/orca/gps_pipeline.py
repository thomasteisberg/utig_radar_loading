"""End-to-end GPS .mat generation pipeline for ORCA segments.

Wraps the ORCA gpspipe loader with the shared OPR GPS .mat builder. ORCA
has no separate radar hardware clock — UHD drives the B-Series radio off
the host wall clock — so `RADAR_TIME` is set equal to `COMP_TIME` (the
host's Unix-time stamps from the gpspipe line prefixes). Per-pulse radar
times in the header file are derived in the same timebase from the UHD
log's `[START]` + PRI math, so the GPS and header files share one radar
clock.
"""

import re
import warnings
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from opr_ingest.core.opr_gps_matlab import (
    create_gps_matlab_structure,
    merge_position_files,
    save_gps_matlab_file,
)
from opr_ingest.orca.gpspipe_gps import load_and_parse_gpspipe_file


_UHD_START_RE = re.compile(r"^\[(\d+\.\d+)\].*\[START\]")


def _parse_uhd_start_timestamp(uhd_log_path: Union[str, Path]) -> Optional[float]:
    """Return the host Unix timestamp from a UHD stdout log's `[START]` line."""
    try:
        with open(uhd_log_path) as f:
            for line in f:
                m = _UHD_START_RE.match(line)
                if m:
                    return float(m.group(1))
    except OSError:
        return None
    return None


def generate_gps_file(
    gpspipe_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    output_path_temporary_df: Optional[Union[str, Path]] = None,
    uhd_log_paths: Optional[List[Union[str, Path]]] = None,
    gps_pad_time_s: float = 2,
) -> pd.DataFrame:
    """Build and save one OPR GPS .mat file from a set of ORCA gpspipe logs."""
    merged_df = merge_position_files(
        file_paths=gpspipe_paths,
        load_function=load_and_parse_gpspipe_file,
        time_sort_key="COMP_TIME",
        remove_duplicates=True,
    )
    print(f"Merged {len(merged_df)} TPV records from {len(gpspipe_paths)} gpspipe file(s)")

    merged_df["RADAR_TIME"] = merged_df["COMP_TIME"]

    if uhd_log_paths:
        comp_min = merged_df["COMP_TIME"].min()
        comp_max = merged_df["COMP_TIME"].max()
        for uhd_path in uhd_log_paths:
            start_t = _parse_uhd_start_timestamp(uhd_path)
            if start_t is None:
                warnings.warn(f"Could not find [START] in {uhd_path}")
            elif not (comp_min <= start_t <= comp_max):
                warnings.warn(
                    f"UHD [START] {start_t:.3f} from {uhd_path} outside gpspipe "
                    f"COMP_TIME range [{comp_min:.3f}, {comp_max:.3f}]"
                )
            else:
                print(
                    f"UHD [START] {start_t:.3f} from {Path(uhd_path).name} sits "
                    f"{start_t - comp_min:.1f}s into gpspipe coverage "
                    f"(span {comp_max - comp_min:.1f}s)"
                )

    if "mode" in merged_df.columns:
        n_before = len(merged_df)
        merged_df = merged_df[merged_df["mode"] >= 3].reset_index(drop=True)
        if len(merged_df) < n_before:
            print(f"Filtered out {n_before - len(merged_df)} rows with mode < 3 (no 3D fix)")

    merged_df = merged_df.dropna(
        subset=["GPS_TIME", "COMP_TIME", "RADAR_TIME", "LAT", "LON"]
    ).reset_index(drop=True)
    print(
        f"Merged dataframe columns: {merged_df.columns.tolist()} "
        f"(len: {len(merged_df)})"
    )

    if gps_pad_time_s and gps_pad_time_s > 0 and len(merged_df) >= 2:
        entry_before = merged_df.iloc[0].copy()
        entry_after = merged_df.iloc[-1].copy()
        gps_span = merged_df["GPS_TIME"].iloc[-1] - merged_df["GPS_TIME"].iloc[0]
        for t_key in ["GPS_TIME", "COMP_TIME", "RADAR_TIME"]:
            t = merged_df[t_key]
            dt_per_gps_time = (t.iloc[-1] - t.iloc[0]) / gps_span if gps_span > 0 else 1.0
            entry_before[t_key] -= dt_per_gps_time * gps_pad_time_s
            entry_after[t_key] += dt_per_gps_time * gps_pad_time_s
        merged_df = pd.concat(
            [pd.DataFrame([entry_before]), merged_df, pd.DataFrame([entry_after])],
            ignore_index=True,
        )

    if output_path_temporary_df is not None:
        output_path_temporary_df = Path(output_path_temporary_df)
        output_path_temporary_df.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_path_temporary_df, index=False)
        print(f"Saved merged dataframe to: {output_path_temporary_df}")

    gps_struct = create_gps_matlab_structure(merged_df, source="ORCA_gpspipe")
    save_gps_matlab_file(gps_struct, output_path)
    print(f"Saved GPS file to: {output_path}")

    print(f"GPS time range: {gps_struct['gps_time'][0, 0]:.2f} to {gps_struct['gps_time'][0, -1]:.2f}")
    print(f"Lat range: {gps_struct['lat'].min():.6f} to {gps_struct['lat'].max():.6f}")
    print(f"Lon range: {gps_struct['lon'].min():.6f} to {gps_struct['lon'].max():.6f}")
    print(f"Elev range: {gps_struct['elev'].min():.2f} to {gps_struct['elev'].max():.2f} meters")

    return merged_df
