"""ORCA radar header extraction.

For ORCA there are no inline per-record headers — every recording is a single
`<prefix>_rx_samps.bin` whose layout is fully described by `<prefix>_config.yaml`
(rx_duration, sample_rate, cpu_format, pulse_rep_int, num_presums) and whose
host-clock start time is logged in `<prefix>_uhd_stdout.log` as `[START]`.

Per record (one record = `num_presums` averaged RX windows = `num_presums *
pulse_rep_int` seconds of acquisition):

    record_stride_bytes = int(rx_duration * sample_rate) * sample_bytes_per_complex
    N                   = file_size // record_stride_bytes
    radar_time[i]       = [START] + i * pulse_rep_int * num_presums
    comp_time[i]        = radar_time[i]              # ORCA has no separate hardware clock
    offset[0, i, 0]     = i * record_stride_bytes    # single-board, single-channel

Output shape and key set match `opr_ingest.utig.headers.get_header_information`
so OPR `records_create` sees the same contract regardless of instrument.
"""

import os
from pathlib import Path
from typing import Union

import numpy as np
import yaml

from opr_ingest.orca.uhd_log import parse_start_timestamp


# fc32 = pair of float32 (8 bytes per complex sample);
# sc16 = pair of int16  (4 bytes);
# sc8  = pair of int8   (2 bytes).
_SAMPLE_BYTES_PER_COMPLEX = {"fc32": 8, "sc16": 4, "sc8": 2}


def get_header_information(rx_samps_path: Union[str, Path]) -> dict:
    """Build the comp_time / radar_time / offset arrays for one ORCA recording.

    Returns the same dict shape as `opr_ingest.utig.headers.get_header_information`:
        comp_time  : (N,)            float64 host Unix seconds per record
        radar_time : (N,)            float64 host Unix seconds per record (== comp_time)
        offset     : (Nb=1, N, Nc=1) int64   byte offsets into the bin file
    """
    rx_path = Path(rx_samps_path)
    if not rx_path.exists():
        raise FileNotFoundError(
            f"ORCA rx_samps file not found: {rx_path}. The recording's `_pN_rx_samps.bin` "
            f"chunks may not have been merged yet."
        )
    if not rx_path.name.endswith("_rx_samps.bin"):
        raise ValueError(f"Expected a *_rx_samps.bin path, got {rx_path}")
    stem_prefix = rx_path.name[: -len("_rx_samps.bin")]
    config_path = rx_path.with_name(stem_prefix + "_config.yaml")
    uhd_log_path = rx_path.with_name(stem_prefix + "_uhd_stdout.log")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    cpu_format = str(config["DEVICE"]["cpu_format"])
    if cpu_format not in _SAMPLE_BYTES_PER_COMPLEX:
        raise ValueError(
            f"Unsupported DEVICE.cpu_format '{cpu_format}' in {config_path}; "
            f"expected one of {list(_SAMPLE_BYTES_PER_COMPLEX)}"
        )
    sample_bytes = _SAMPLE_BYTES_PER_COMPLEX[cpu_format]

    rx_duration = float(config["CHIRP"]["rx_duration"])
    sample_rate = float(config["GENERATE"]["sample_rate"])
    pulse_rep_int = float(config["CHIRP"]["pulse_rep_int"])
    num_presums = int(config["CHIRP"].get("num_presums", 1))

    rx_len_samples = int(rx_duration * sample_rate)
    record_stride_bytes = rx_len_samples * sample_bytes

    # ORCA on SORA is single RX channel. The config carries
    # `DEVICE.rx_channels` as a comma-separated string (e.g. "0" or "0,1");
    # we assert it here so a future multi-channel recording fails loudly
    # instead of silently mis-computing offsets. Multi-channel support also
    # needs the on-disk interleaving convention checked (per-sample vs.
    # per-chirp) before record_stride_bytes / offsets can be generalized.
    rx_channels_raw = config["DEVICE"].get("rx_channels", "0")
    n_rx_channels = len(
        [c for c in str(rx_channels_raw).split(",") if c.strip() != ""]
    )
    if n_rx_channels != 1:
        raise NotImplementedError(
            f"ORCA get_header_information currently supports only single-channel "
            f"recordings; {config_path} declares rx_channels='{rx_channels_raw}' "
            f"({n_rx_channels} channels). Generalizing requires confirming the "
            f"on-disk channel interleaving convention."
        )

    file_size = os.path.getsize(rx_path)
    N = file_size // record_stride_bytes
    remainder = file_size - N * record_stride_bytes
    if remainder != 0:
        print(
            f"[WARNING] {rx_path.name}: file size {file_size} is not an integer "
            f"multiple of record stride {record_stride_bytes} (remainder {remainder} "
            f"bytes); truncating to {N} records."
        )

    start_t = parse_start_timestamp(uhd_log_path)
    if start_t is None:
        raise ValueError(
            f"No [START] marker found in {uhd_log_path}; cannot derive radar_time. "
            f"Add manual override or fall back to filename timestamp if needed."
        )

    record_dt = pulse_rep_int * num_presums
    radar_time = start_t + np.arange(N, dtype=np.float64) * record_dt
    comp_time = radar_time.copy()

    offsets = (np.arange(N, dtype=np.int64) * record_stride_bytes).reshape(1, N, 1)

    return {
        "comp_time": comp_time,
        "radar_time": radar_time,
        "offset": offsets,
    }
