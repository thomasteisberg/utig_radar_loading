"""End-to-end GPS .mat generation pipeline for ORCA segments.

Wraps the ORCA gpspipe loader with the shared OPR GPS .mat builder.
"""

from pathlib import Path
from typing import List, Optional, Union

from opr_ingest.core.opr_gps_matlab import (
    create_gps_matlab_structure,
    merge_position_files,
    save_gps_matlab_file,
)
from opr_ingest.orca.gpspipe_gps import load_and_parse_gpspipe_file


def generate_gps_file(
    gpspipe_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    output_path_temporary_df: Optional[Union[str, Path]] = None,
    gps_pad_time_s: float = 2,
) -> None:
    """Build and save one OPR GPS .mat file from a set of ORCA gpspipe logs."""
    raise NotImplementedError(
        "ORCA gps_pipeline.generate_gps_file: call merge_position_files with "
        "load_and_parse_gpspipe_file, derive RADAR_TIME (TODO: confirm source — "
        "GPS_TIME copy vs UHD log alignment), optionally pad endpoints, then "
        "create_gps_matlab_structure + save_gps_matlab_file. See "
        "src/opr_ingest/utig/gps_pipeline.py:generate_gps_file for the template."
    )
