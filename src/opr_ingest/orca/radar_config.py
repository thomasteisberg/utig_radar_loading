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
        f0, f1    : float  RF chirp band edges [Hz]:
                           RF<n>.freq + RF<n>.lo_offset + lo_offset_sw
                             -/+ chirp_bandwidth/2
        DDC_freq  : float  Hardware-DDC center frequency [Hz]:
                           RF<n>.freq + RF<n>.lo_offset
                           (the SDR's LO; we tell OPR the data was
                            mixed down from this RF). fc=(f0+f1)/2 then
                            equals lo_offset_sw + DDC_freq, OPR's
                            pulse compression mask aligns correctly,
                            and lambda=c/fc is finite — needed by SAR
                            for synthetic-aperture math (sar.m:577,
                            sar_task.m:139, etc.). Setting fc=0 makes
                            lambda=Inf which propagates through SAR's
                            cluster cpu_time/mem estimates.
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

    # OPR's convention: f0/f1 describe the original RF chirp band; DDC_freq
    # describes the hardware-DDC center the SDR mixed the signal down by.
    # OPR uses (f0+f1)/2 = fc for wavelength = c/fc (needed by SAR's
    # synthetic-aperture math and cluster resource estimates -- fc=0 gives
    # lambda=Inf -> Inf cpu_time/mem estimates). The matched-filter band-pass
    # mask uses BW_window derived from f0/f1, and OPR's pulse_compress shifts
    # the data's frequency axis by DDC_freq so the baseband-sampled chirp
    # lands in the correct RF-labeled bins. Both conditions met if we emit
    # RF f0/f1 AND DDC_freq=hardware-LO=RF.freq+RF.lo_offset. lo_offset_sw
    # is the residual baseband offset inside the chirp after HW DDC, so
    # fc=DDC_freq+lo_offset_sw.
    ddc_freq = float(rf["freq"]) + float(rf["lo_offset"])
    rf_center = ddc_freq + float(generate["lo_offset_sw"])
    half_bw = float(generate["chirp_bandwidth"]) / 2.0
    f0 = rf_center - half_bw
    f1 = rf_center + half_bw

    num_sam = int(round(fs * float(chirp["rx_duration"])))

    return {
        "fs": fs,
        "prf": prf,
        "Tpd": Tpd,
        "presums": presums,
        "f0": f0,
        "f1": f1,
        "DDC_freq": ddc_freq,
        "num_sam": num_sam,
    }
