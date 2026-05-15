"""OPR temporary-headers conventions.

The OPR records_create MATLAB stage expects per-bxds-file `.mat` header files
under `<opr_tmp>/headers/<season>/<board_folder_name>/<bxds_stem>.mat`, each
containing at least:
    - comp_time:  (N,) computer time per record (seconds, Unix epoch)
    - radar_time: (N,) radar clock per record
    - offset:     (Nb, N, Nc) int64 byte offsets into the bxds file, where
                  Nb = boards, Nc = channels. Missing data: -2**31.

Instrument-specific code is responsible for building those arrays; only the
path/dispatch conventions live here.
"""

from pathlib import Path
from typing import Union


def get_header_file_location(bxds_path: Union[str, Path], base_dir: Union[str, Path]) -> str:
    """Return the OPR-tmp .mat path for a bxds file under `base_dir`.

    Mirrors the 4-level board-folder hierarchy of the source path.
    """
    p = Path(bxds_path)
    fn_name = p.stem
    board_folder_name = Path(*p.parts[-5:-1])
    return str(Path(base_dir) / board_folder_name / (fn_name + '.mat'))
