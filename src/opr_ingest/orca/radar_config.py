"""Per-recording ORCA radar parameters derived from `_config.yaml`.

The ORCA per-recording `_config.yaml` is the source of truth for fields that
vary across captures (sample rate, chirp band, pulse duration, presum count,
PRI). This module reads one such file and emits the subset of OPR `radar`-
sheet fields that need per-segment values, plus `file.clk` for the records
sheet.

For the single-channel SORA hardware, `DEVICE.rx_channels` is `"0"` (or
`"1"`) and the corresponding `RF<n>` block carries the actual sample rate /
LO settings. Multi-channel recordings raise `NotImplementedError`, matching
`opr_ingest.orca.headers.get_header_information`.
"""

from pathlib import Path
from typing import Union

import yaml


def load_radar_params(config_path: Union[str, Path]) -> dict:
    """Read one ORCA _config.yaml and derive the per-recording radar params.

    Returns dict with keys:
        fs        : float  RX sample rate [Hz]  (radar.fs and records.file.clk)
        prf       : float  raw pulse repetition frequency [Hz] = 1/pulse_rep_int
        Tpd       : float  pulse / chirp duration [s] = GENERATE.chirp_length
        presums   : int    CHIRP.num_presums
        f0, f1    : float  Baseband chirp band edges [Hz]:
                           GENERATE.lo_offset_sw  -/+  chirp_bandwidth/2
                           These describe the band of frequencies present
                           in the digitized stream, not the RF carrier.
                           ORCA's USRP B205mini DDCs to baseband in
                           hardware before we get the samples, so the
                           chirp lives near 0 Hz (offset by lo_offset_sw)
                           and OPR's matched filter must be built there.
                           Using RF f0/f1 here yields a matched filter
                           at hundreds of MHz against a baseband signal
                           → no pulse compression, fast-time smearing.
        num_sam   : int    samples per record = round(fs * CHIRP.rx_duration)
    """
    cfg_path = Path(config_path)
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    rx_channels_raw = config["DEVICE"].get("rx_channels", "0")
    rx_channels = [c.strip() for c in str(rx_channels_raw).split(",") if c.strip() != ""]
    if len(rx_channels) != 1:
        raise NotImplementedError(
            f"ORCA load_radar_params currently supports only single-channel "
            f"recordings; {cfg_path} declares rx_channels='{rx_channels_raw}' "
            f"({len(rx_channels)} channels)."
        )
    rf_block = f"RF{rx_channels[0]}"
    rf = config[rf_block]

    generate = config["GENERATE"]
    chirp = config["CHIRP"]

    fs = float(rf["rx_rate"])
    prf = 1.0 / float(chirp["pulse_rep_int"])
    Tpd = float(generate["chirp_length"])
    presums = int(chirp.get("num_presums", 1))

    # OPR's wfs.f0/f1 describe the band that's actually present in the
    # digitized samples (used to build the matched filter). For ORCA the
    # data is complex baseband (USRP DDCs to ~0 around RF.freq+RF.lo_offset
    # before writing), so the chirp lives at lo_offset_sw +/- chirp_bw/2,
    # NOT at the RF carrier. Verified against the diagnostic script's
    # clean pulse compression on 20251204_224207 (see CLAUDE.md note).
    lo_offset_sw = float(generate["lo_offset_sw"])
    half_bw = float(generate["chirp_bandwidth"]) / 2.0
    f0 = lo_offset_sw - half_bw
    f1 = lo_offset_sw + half_bw

    num_sam = int(round(fs * float(chirp["rx_duration"])))

    return {
        "fs": fs,
        "prf": prf,
        "Tpd": Tpd,
        "presums": presums,
        "f0": f0,
        "f1": f1,
        "num_sam": num_sam,
    }
