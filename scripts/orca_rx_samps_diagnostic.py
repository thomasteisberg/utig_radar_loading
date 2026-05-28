"""Diagnose ORCA rx_samps.bin interpretation vs OPR's qlook.

Working theory (see CLAUDE.md item under qlook debugging): OPR loads ORCA
data with `file.version=425`, which is UTIG's ICECAP file format expecting
*real-valued int16* samples from a direct-IF digitizer. But ORCA's B205mini
records *complex float32* (`fc32`) baseband — every (I, Q) pair is a single
8-byte complex sample, per `reference/processing.py:extractSig`:

    # The format of the bin file is <1st real><1st imag><2nd real><2nd imag>
    # The real and imaginary parts of the signal are of type np.float32

If OPR's v425 reader interprets these bytes as int16 real samples, every
8 bytes of one complex float32 pair become 4 garbage int16 reals — the
energy distribution is still vaguely chirp-like, hence "fuzzy" qlook rather
than pure noise.

This script makes that visible:

  1. Reads one recording two ways: as fc32 (correct, per reference) and as
     int16 real (what v425 probably does).
  2. Plots PSDs of both: the fc32 PSD should show a clean chirp band; the
     int16 PSD should be dramatically different (aliased / spread).
  3. Synthesizes the TX chirp from the config and pulse-compresses N
     records, plotting a single range line and a small echogram strip —
     this is the "what qlook *should* look like" baseline.

Usage:
    uv run scripts/orca_rx_samps_diagnostic.py <prefix-or-rx-samps-path>
       [--n-records 500] [--save-dir outputs/diagnostics] [--no-show]

`<prefix-or-rx-samps-path>` can be either a recording prefix
(`/path/20251204_224207`) or the full bin path
(`/path/20251204_224207_rx_samps.bin`). The script looks up the
sibling `_config.yaml` either way.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

import numpy as np
import yaml


def _prefix_from_arg(arg: str) -> Path:
    """Accept either a recording prefix or the rx_samps.bin path."""
    p = Path(arg)
    if p.name.endswith("_rx_samps.bin"):
        return p.with_name(p.name[: -len("_rx_samps.bin")])
    return p


def _load_config(prefix: Path) -> dict:
    cfg_path = prefix.with_name(prefix.name + "_config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config beside prefix: {cfg_path}")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _rf_block(config: dict) -> dict:
    """Pick the active RF<n> block referenced by GENERATE/DEVICE."""
    rf_blocks = {k: v for k, v in config.items() if k.startswith("RF")}
    if not rf_blocks:
        raise ValueError("No RF<n> block in config")
    # ORCA's single-channel SORA configs only have one RF block; if multiple,
    # this picks the first deterministically. Generalize if multi-RF ships.
    return rf_blocks[sorted(rf_blocks)[0]]


def _get_first(config: dict, candidates: list[tuple[str, str]], description: str) -> float:
    """Look up a value by trying several (block, key) paths in order.

    Surfaces the actual config block contents in the error message so the
    user (or Claude) can see what keys are available without re-reading the
    file manually.
    """
    for block, key in candidates:
        if block in config and key in config[block]:
            return float(config[block][key])
    seen = {b: list(config.get(b, {})) for b, _ in candidates}
    tried = ", ".join(f"{b}.{k}" for b, k in candidates)
    raise KeyError(
        f"Could not find {description}; tried {tried}. "
        f"Available keys: {seen}"
    )


def synthesize_tx_chirp(config: dict) -> tuple[np.ndarray, float, float, float]:
    """Build the transmitted chirp from the config.

    Returns
    -------
    chirp : np.ndarray, complex64
        Baseband chirp samples — what the receiver, post-DDC, should see at
        zero range. Use this as the matched-filter reference.
    fs : float
        Sample rate, Hz (`GENERATE.sample_rate`).
    f0, f1 : float
        Chirp start and stop frequencies *relative to baseband*, Hz.
    """
    gen = config["GENERATE"]
    fs = float(gen["sample_rate"])

    # Tpd: CLAUDE.md notes ORCA stores chirp duration in GENERATE.chirp_length,
    # but allow CHIRP.chirp_length as a fallback for older/variant configs.
    tpd = _get_first(
        config,
        [("GENERATE", "chirp_length"), ("CHIRP", "chirp_length")],
        "chirp duration (Tpd)",
    )

    # chirp_bandwidth could plausibly live in either block.
    try:
        bw = _get_first(
            config,
            [("GENERATE", "chirp_bandwidth"), ("CHIRP", "chirp_bandwidth")],
            "chirp bandwidth",
        )
    except KeyError as e:
        print(f"[WARN] {e}\n       Falling back to bw = sample_rate ({fs/1e6:.2f} MHz)")
        bw = fs

    # At baseband (post-DDC), the chirp sweeps -BW/2 .. +BW/2 if it's
    # symmetric around the LO. lo_offset_sw shifts that.
    lo_offset_sw = float(gen.get("lo_offset_sw", 0.0))
    f0 = -bw / 2.0 + lo_offset_sw
    f1 = +bw / 2.0 + lo_offset_sw

    n = int(round(tpd * fs))
    t = np.arange(n) / fs
    # Linear FM up-chirp at baseband. Phase = 2*pi*(f0*t + 0.5*k*t^2).
    k = (f1 - f0) / tpd
    phase = 2.0 * np.pi * (f0 * t + 0.5 * k * t * t)
    return np.exp(1j * phase).astype(np.complex64), fs, f0, f1


def load_records_fc32(rx_path: Path, n_records: int, rx_len: int) -> np.ndarray:
    """Load the first n_records as complex float32 (the correct interpretation).

    Returns array shape (rx_len, n_records), dtype=complex64.
    """
    n_floats = n_records * rx_len * 2  # 2 floats per complex sample
    raw = np.fromfile(rx_path, dtype=np.float32, count=n_floats)
    actual_records = raw.size // (rx_len * 2)
    if actual_records < n_records:
        print(f"[fc32] Wanted {n_records} records, file only has {actual_records}")
        n_records = actual_records
    raw = raw[: n_records * rx_len * 2]
    complex_samples = raw[0::2] + 1j * raw[1::2]
    return complex_samples.reshape(n_records, rx_len).T.astype(np.complex64)


def load_records_int16(rx_path: Path, n_records: int, rx_len: int) -> np.ndarray:
    """Load the first n_records *as if* it were int16 real samples.

    This is the wrong interpretation but mimics what OPR's v425 reader is
    suspected to do. The 'record stride in samples' under this misreading
    is 4 int16 samples per 1 true complex float32 — we keep the byte stride
    fixed (record_stride_bytes) and reinterpret.
    """
    bytes_per_complex_record = rx_len * 8  # 8 bytes/complex (fc32)
    n_int16_per_record = bytes_per_complex_record // 2  # 4 int16 per 1 fc32
    n_int16_total = n_records * n_int16_per_record
    raw = np.fromfile(rx_path, dtype=np.int16, count=n_int16_total)
    actual_records = raw.size // n_int16_per_record
    if actual_records < n_records:
        n_records = actual_records
    raw = raw[: n_records * n_int16_per_record]
    return raw.reshape(n_records, n_int16_per_record).T.astype(np.float32)


def pulse_compress_one(record: np.ndarray, chirp: np.ndarray) -> np.ndarray:
    """Matched-filter one complex record against the synthesized chirp."""
    from scipy.signal import correlate

    out = correlate(record, chirp, mode="valid", method="auto")
    return out / np.sum(np.abs(chirp) ** 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefix_or_path", help="Recording prefix or _rx_samps.bin path")
    parser.add_argument("--n-records", type=int, default=500,
                        help="Number of records to load for the diagnostic (default 500)")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save PNGs to (also good for headless cresis hosts).")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not call plt.show() (useful with --save-dir on headless machines).")
    parser.add_argument("--dump-config", action="store_true",
                        help="Print the config blocks/keys and exit (no plotting). Useful for "
                             "inventory before chasing a KeyError.")
    args = parser.parse_args()

    if args.dump_config:
        prefix = _prefix_from_arg(args.prefix_or_path)
        cfg = _load_config(prefix)
        print(f"# Config for {prefix.name}")
        for block, vals in cfg.items():
            if isinstance(vals, dict):
                print(f"[{block}]")
                for k, v in vals.items():
                    print(f"  {k} = {v!r}")
            else:
                print(f"{block} = {vals!r}")
        return 0

    if args.no_show or args.save_dir:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prefix = _prefix_from_arg(args.prefix_or_path)
    rx_path = prefix.with_name(prefix.name + "_rx_samps.bin")
    if not rx_path.exists():
        raise FileNotFoundError(f"No rx_samps.bin beside prefix: {rx_path}")
    config = _load_config(prefix)

    fs = float(config["GENERATE"]["sample_rate"])
    rx_duration = float(config["CHIRP"]["rx_duration"])
    rx_len = int(round(rx_duration * fs))
    cpu_format = str(config["DEVICE"].get("cpu_format", "?"))
    num_presums = int(config["CHIRP"].get("num_presums", 1))

    print("=" * 72)
    print(f"Recording: {prefix.name}")
    print(f"  fs              : {fs/1e6:.3f} MHz")
    print(f"  rx_duration     : {rx_duration*1e6:.2f} us  -> rx_len = {rx_len} samples")
    print(f"  cpu_format      : {cpu_format}  (expect 'fc32' = complex float32, 8 B/cplx)")
    print(f"  num_presums     : {num_presums}")
    file_size = rx_path.stat().st_size
    n_records_fc32 = file_size // (rx_len * 8)
    n_records_int16_misread = file_size // (rx_len * 2)  # if read as int16 real
    print(f"  file size       : {file_size:,} bytes")
    print(f"  records (fc32)  : {n_records_fc32:,}")
    print(f"  records (i16, misread): {n_records_int16_misread:,}  "
          f"[4x more — first clue something's off in v425]")
    print("=" * 72)

    n = min(args.n_records, n_records_fc32)
    print(f"\nLoading {n} records both ways...")
    data_fc32 = load_records_fc32(rx_path, n, rx_len)
    data_i16 = load_records_int16(rx_path, n, rx_len)
    print(f"  fc32 shape: {data_fc32.shape}  dtype={data_fc32.dtype}")
    print(f"  i16  shape: {data_i16.shape}   dtype={data_i16.dtype}")

    chirp, _, f0, f1 = synthesize_tx_chirp(config)
    print(f"\nSynthesized TX chirp: {len(chirp)} samples, "
          f"f0={f0/1e6:.3f} MHz, f1={f1/1e6:.3f} MHz at baseband")

    one = data_fc32[:, 0]
    rc = pulse_compress_one(one, chirp)

    # ---- plots ----
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(f"ORCA rx_samps diagnostic: {prefix.name}", fontsize=13)

    axs[0, 0].psd(data_fc32[:, 0], NFFT=2048, Fs=fs / 1e6)
    axs[0, 0].set_title("PSD: record loaded as fc32 (CORRECT)\nshould show clean chirp band")
    axs[0, 0].set_xlabel("Freq (MHz, baseband)")

    axs[0, 1].psd(data_i16[:, 0], NFFT=2048, Fs=fs / 1e6)
    axs[0, 1].set_title("PSD: record loaded as int16 (WRONG, v425-style)\nbroken/aliased = what OPR sees")
    axs[0, 1].set_xlabel("Freq (MHz, mis-rate)")

    axs[0, 2].plot(np.real(chirp), label="Re", alpha=0.7)
    axs[0, 2].plot(np.imag(chirp), label="Im", alpha=0.7)
    axs[0, 2].set_title("Synthesized TX chirp (matched-filter ref)")
    axs[0, 2].set_xlabel("Sample"); axs[0, 2].legend()

    axs[1, 0].plot(np.real(one), label="Re", alpha=0.6)
    axs[1, 0].plot(np.imag(one), label="Im", alpha=0.6)
    axs[1, 0].set_title("One RX record (fc32) — time domain")
    axs[1, 0].set_xlabel("Sample"); axs[1, 0].legend()

    fast_time_us = 1e6 * np.arange(len(rc)) / fs
    axs[1, 1].plot(fast_time_us, 20 * np.log10(np.abs(rc) + 1e-12))
    axs[1, 1].set_title("Pulse-compressed range line (one record)")
    axs[1, 1].set_xlabel("Fast time (us)"); axs[1, 1].set_ylabel("|rc| (dB)")

    # Tiny echogram: pulse-compress every record we have, log magnitude, plot.
    echo = np.zeros((max(rx_len - len(chirp) + 1, 1), n), dtype=np.float32)
    for i in range(n):
        echo[:, i] = np.abs(pulse_compress_one(data_fc32[:, i], chirp))
    db = 20 * np.log10(echo + 1e-12)
    vmin = np.percentile(db, 5); vmax = np.percentile(db, 99.5)
    im = axs[1, 2].imshow(db, aspect="auto", cmap="gray", origin="upper",
                          vmin=vmin, vmax=vmax,
                          extent=(0, n, fast_time_us[-1], fast_time_us[0]))
    axs[1, 2].set_title(f"Mini echogram ({n} records, fc32)\nbaseline for qlook comparison")
    axs[1, 2].set_xlabel("Record idx"); axs[1, 2].set_ylabel("Fast time (us)")
    fig.colorbar(im, ax=axs[1, 2], shrink=0.8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if args.save_dir:
        save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        out = save_dir / f"rx_samps_diagnostic_{prefix.name}.png"
        fig.savefig(out, dpi=130)
        print(f"\nSaved figure to {out}")
    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
