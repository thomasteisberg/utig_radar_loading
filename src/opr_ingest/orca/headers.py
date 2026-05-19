"""ORCA radar header extraction.

For ORCA there are no inline per-record headers — every recording is a single
`<prefix>_rx_samps.bin` whose layout is described by `<prefix>_config.yaml`
(CHIRP.rx_duration, CHIRP.pulse_rep_int, GENERATE.sample_rate, etc.) and whose
absolute start time is logged in `<prefix>_uhd_stdout.log`.
"""

from pathlib import Path
from typing import Union


def get_header_information(rx_samps_path: Union[str, Path]) -> dict:
    """Return the dict shape that OPR `records_create` expects.

    Required keys:
        comp_time  : (N,) computer time per record (seconds, Unix epoch)
        radar_time : (N,) radar clock per record (matches UTIG `tim` semantics)
        offset     : (Nb, N, Nc) int64 byte offsets into the bin file
    """
    raise NotImplementedError(
        "ORCA headers.get_header_information: parse <prefix>_config.yaml for "
        "rx_duration, sample_rate, pulse_rep_int, n_rxs to compute record "
        "stride and offsets; parse <prefix>_uhd_stdout.log for [START] timestamp "
        "and per-pulse timing. See reference/processing_dask.py:save_radar_data_to_zarr "
        "for the bin/config decoding logic and "
        "src/opr_ingest/utig/headers.py:get_header_information for the return shape."
    )
