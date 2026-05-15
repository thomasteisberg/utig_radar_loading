"""UTIG radar header extraction (RADjh1 / RADnh3 / RADnh5)."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import unfoc

from opr_ingest.utig import stream_util


def create_header_df(input_filename, waveform_record_length=6400):
    """Build the header DataFrame for one bxds file.

    Auto-detects stream type:
      - RADjh1 (HiCARS1): no per-record headers; offsets derived from file size.
      - RADnh3 / RADnh5:  use unfoc to index headers.

    Columns: `tim`, `COMP_TIME`, and one `chN_offset` column per channel.
    Index: `rseq` (radar sequence number).
    """
    path = Path(input_filename)
    if path.parent.name == 'RADjh1':
        return _create_header_df_radjh1(input_filename, waveform_record_length)

    stream = unfoc.get_radar_stream(input_filename)

    ct_data = stream_util.load_ct_file(input_filename)
    ct_data = stream_util.parse_CT(ct_data)

    fpos, header_len, header = zip(*unfoc.index_RADnhx_bxds(input_filename, full_header=True))

    if stream == 'RADnh5':
        rseq = np.array([h.rseq for h in header])
    elif stream == 'RADnh3':
        rseq = ct_data['seq'].values
    else:
        raise ValueError(f"Unsupported stream type: {stream}")

    choff = np.array([h.choff for h in header])
    # 0xff is a sentinel for "no channel" and is treated as 0
    unique_choff = np.unique(choff[choff != 0xff])
    num_channels = len(unique_choff) * 2  # each choff value carries 2 channels

    df_dict = {
        'rseq': rseq,
        'choff': choff,
        'start_fpos': fpos,
        'header_len': header_len,
    }
    for i in range(num_channels):
        df_dict[f'ch{i}_offset'] = pd.NA

    df = pd.DataFrame(df_dict)
    df = df.join(ct_data[['tim', 'COMP_TIME']])

    for choff_val in unique_choff:
        choff_idx = np.where(df['choff'] == choff_val)[0]
        base_ch = choff_val

        df.iloc[choff_idx, df.columns.get_loc(f'ch{base_ch}_offset')] = \
            df.iloc[choff_idx]['start_fpos'] + df.iloc[choff_idx]['header_len']
        df.iloc[choff_idx, df.columns.get_loc(f'ch{base_ch+1}_offset')] = \
            df.iloc[choff_idx]['start_fpos'] + df.iloc[choff_idx]['header_len'] + waveform_record_length

    ch_cols = [f'ch{i}_offset' for i in range(num_channels)]
    df = df[['tim', 'COMP_TIME', 'rseq'] + ch_cols]

    df = df.groupby('rseq').first()

    with pd.option_context('future.no_silent_downcasting', True):
        df = df.fillna(-2**31)

    file_size = os.path.getsize(input_filename)
    max_offset_allowed = file_size - waveform_record_length
    max_offset_df = df[ch_cols].max().max()
    if max_offset_df > max_offset_allowed:
        print(f"[WARNING] Header has offset of {max_offset_df}, which is too large for a file of {file_size} bytes. Violating rows will be dropped.")
        df = df[df[ch_cols].max(axis=1) <= max_offset_allowed]

    return df


def _create_header_df_radjh1(input_filename, waveform_record_length=6400):
    """Header DataFrame for one RADjh1 (HiCARS1) bxds file.

    RADjh1 files have no per-record headers — just raw trace data of fixed length.
    """
    ct_data = stream_util.load_ct_file(input_filename)
    ct_data = stream_util.parse_CT(ct_data)

    path = Path(input_filename)
    filename = path.name
    if filename == 'bxds1':
        channel = 1
    elif filename == 'bxds2':
        channel = 2
    else:
        raise ValueError(f"Expected bxds1 or bxds2, got {filename}")

    file_size = path.stat().st_size
    if file_size % waveform_record_length != 0:
        raise ValueError(f"File size {file_size} is not divisible by {waveform_record_length}")
    ntraces = file_size // waveform_record_length

    offsets = np.arange(ntraces) * waveform_record_length

    n_ct_records = len(ct_data)
    if n_ct_records != ntraces:
        print(f"[WARNING] CT file has {n_ct_records} records but data file has {ntraces} traces")
        n_records = min(n_ct_records, ntraces)
        ct_data = ct_data.iloc[:n_records]
        offsets = offsets[:n_records]

    df_dict = {
        'tim': ct_data['tim'].values,
        'COMP_TIME': ct_data['COMP_TIME'].values,
        'rseq': ct_data['seq'].values,
        f'ch{channel}_offset': offsets,
    }

    df = pd.DataFrame(df_dict)
    df = df.set_index('rseq')

    with pd.option_context('future.no_silent_downcasting', True):
        df = df.fillna(-2**31)

    max_offset_allowed = file_size - waveform_record_length
    max_offset_df = df[f'ch{channel}_offset'].max()
    if max_offset_df > max_offset_allowed:
        print(f"[WARNING] Header has offset of {max_offset_df}, which is too large for a file of {file_size} bytes. Violating rows will be dropped.")
        df = df[df[f'ch{channel}_offset'] <= max_offset_allowed]

    return df


def get_header_information(f):
    """Return the dict shape that records_create wants: comp_time / radar_time / offset."""
    df = create_header_df(f)

    ch_cols = [col for col in df.columns if col.startswith('ch') and col.endswith('_offset')]

    offsets_array = df[ch_cols].to_numpy()
    # Final shape: Nb x Nx x Nc (boards x records x channels)
    offsets_array = np.expand_dims(offsets_array, axis=0).astype(np.int64)

    return {
        'comp_time': df['COMP_TIME'].values,
        'radar_time': df['tim'].values,
        'offset': offsets_array,
    }
