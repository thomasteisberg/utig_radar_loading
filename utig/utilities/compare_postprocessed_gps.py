"""
Compare post-processed GPS data against raw GPS streams from the UTIG file index.

This script loads a post-processed GPS file (e.g., IPUTG1B/EPUTG1B format) and
compares its timestamps and positions against raw GPS streams found in the UTIG
data archive. This is useful for checking whether a post-processed GPS file
actually corresponds to the prj/set/trn indicated by its filename, or if it
better matches a different transect.

Usage:
    uv run python compare_postprocessed_gps.py <postprocessed_gps_path> <trn_prefix>

    postprocessed_gps_path: Path to a post-processed GPS file (IPUTG1B/EPUTG1B/SPUTG1B)
    trn_prefix: Prefix to match transect names against (e.g., 'ATAL01' matches ATAL01a, ATAL01b, ...)

Example:
    uv run python compare_postprocessed_gps.py \\
        /resfs/.../IPUTG1B_2009006_DMC_JKB0a_ATAL01a_position.txt ATAL01
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from opr_ingest.utig import file_index, postprocessed_gps, stream_util

# ============================================================================
# Configuration
# ============================================================================

# Base path for the UTIG data archive and cache for file indexing
base_path = "/kucresis/scratch/data/UTIG"
cache_file = "outputs/file_index.csv"


def extract_trn_prefix(postprocessed_gps_path):
    """Extract the transect prefix from a post-processed GPS filename.

    E.g., 'IPUTG1B_2009006_DMC_JKB0a_ATAL01a_position.txt' -> 'ATAL01'
    (strips the trailing letter from the transect name)
    """
    stem = Path(postprocessed_gps_path).stem  # e.g., IPUTG1B_2009006_DMC_JKB0a_ATAL01a_position
    parts = stem.split('_')
    # parts: [type, doy, prj, set, trn, 'position']
    trn = parts[-2]  # e.g., ATAL01a
    # Strip trailing lowercase letter to get prefix
    if trn[-1].islower():
        return trn[:-1]
    return trn


def run_comparison(postprocessed_gps_path, trn_prefix, df_artifacts):
    """Run a single comparison of a post-processed GPS file against candidate streams."""

    print(f"\n{'#' * 100}")
    print(f"# {Path(postprocessed_gps_path).name}  (filter: trn starts with '{trn_prefix}')")
    print(f"{'#' * 100}")

    # ========================================================================
    # 1. Load post-processed GPS data
    # ========================================================================

    print(f"\nLoading post-processed GPS file: {postprocessed_gps_path}")
    df_postproc = postprocessed_gps.load_and_parse_postprocessed_gps_file(postprocessed_gps_path)
    print(f"  Records: {len(df_postproc)}")
    print(f"  GPS_TIME range: {df_postproc['GPS_TIME'].min():.2f} to {df_postproc['GPS_TIME'].max():.2f}")
    t_start_postproc = pd.Timestamp.utcfromtimestamp(df_postproc['GPS_TIME'].min())
    t_end_postproc = pd.Timestamp.utcfromtimestamp(df_postproc['GPS_TIME'].max())
    print(f"  Time range (UTC): {t_start_postproc} to {t_end_postproc}")
    print(f"  LAT range: {df_postproc['LAT'].min():.6f} to {df_postproc['LAT'].max():.6f}")
    print(f"  LON range: {df_postproc['LON'].min():.6f} to {df_postproc['LON'].max():.6f}")

    # ========================================================================
    # 2. Filter artifacts to candidate GPS streams
    # ========================================================================

    df_candidates = df_artifacts[df_artifacts['trn'].str.startswith(trn_prefix)]

    # Further filter to GPS stream types (with or without position data)
    gps_stream_types = ['GPSnc1', 'GPStp2', 'GPSap1', 'GPSap3', 'GPSkc1']
    df_candidates = df_candidates[df_candidates['stream'].isin(gps_stream_types)]

    # Only look at xds.gz files (the parseable text streams)
    df_candidates = df_candidates[df_candidates['file_name'] == 'xds.gz']

    # Filter to only candidates whose prj/set/trn has a corresponding RAD file
    radar_trns = df_artifacts[
        df_artifacts['stream'].str.startswith('RAD')
    ][['prj', 'set', 'trn']].drop_duplicates()
    df_candidates = df_candidates.merge(
        radar_trns, on=['prj', 'set', 'trn'], how='inner'
    )

    print(f"\nCandidate GPS streams with radar ({len(df_candidates)} files):")
    print(df_candidates[['prj', 'set', 'trn', 'stream', 'full_path']].to_string(index=False))

    # ========================================================================
    # 3. Load each candidate GPS stream and compare timestamps
    # ========================================================================

    results = []

    for idx, row in df_candidates.iterrows():
        label = f"{row['prj']}/{row['set']}/{row['trn']}/{row['stream']}"
        gps_path = row['full_path']
        print(f"\n--- Loading: {label} ---")
        print(f"    Path: {gps_path}")

        try:
            df_raw = stream_util.load_xds_stream_file(gps_path, parse=True)
        except Exception as e:
            print(f"    ERROR loading: {e}")
            results.append({
                'label': label, 'prj': row['prj'], 'set': row['set'],
                'trn': row['trn'], 'stream': row['stream'],
                'n_records': 0, 'error': str(e),
            })
            continue

        if 'GPS_TIME' not in df_raw.columns:
            print(f"    WARNING: No GPS_TIME column after parsing")
            results.append({
                'label': label, 'prj': row['prj'], 'set': row['set'],
                'trn': row['trn'], 'stream': row['stream'],
                'n_records': len(df_raw), 'error': 'No GPS_TIME',
            })
            continue

        has_position = 'LAT' in df_raw.columns and 'LON' in df_raw.columns

        # Filter out GPS_TIME outliers (corrupt records with epoch-era timestamps)
        n_before = len(df_raw)
        df_raw = df_raw[df_raw['GPS_TIME'] > 1e9]  # 1e9 ~ 2001-09-09
        n_filtered = n_before - len(df_raw)
        if n_filtered > 0:
            print(f"    Filtered {n_filtered} record(s) with GPS_TIME < 1e9 (epoch-era outliers)")
        if len(df_raw) == 0:
            print(f"    WARNING: No valid GPS_TIME records remain after filtering")
            results.append({
                'label': label, 'prj': row['prj'], 'set': row['set'],
                'trn': row['trn'], 'stream': row['stream'],
                'n_records': 0, 'error': 'All GPS_TIME values filtered',
            })
            continue

        # Check if a radar file exists for this prj/set/trn
        radar_match = df_artifacts[
            (df_artifacts['prj'] == row['prj']) &
            (df_artifacts['set'] == row['set']) &
            (df_artifacts['trn'] == row['trn']) &
            (df_artifacts['stream'].str.startswith('RAD'))
        ]
        has_radar = len(radar_match) > 0
        radar_streams = ', '.join(radar_match['stream'].unique()) if has_radar else ''
        print(f"    Radar files: {radar_streams if has_radar else 'NONE'}")

        t_start_raw = pd.Timestamp.utcfromtimestamp(df_raw['GPS_TIME'].min())
        t_end_raw = pd.Timestamp.utcfromtimestamp(df_raw['GPS_TIME'].max())

        print(f"    Records: {len(df_raw)}")
        print(f"    GPS_TIME range: {df_raw['GPS_TIME'].min():.2f} to {df_raw['GPS_TIME'].max():.2f}")
        print(f"    Time range (UTC): {t_start_raw} to {t_end_raw}")
        if has_position:
            print(f"    LAT range: {df_raw['LAT'].min():.6f} to {df_raw['LAT'].max():.6f}")
            print(f"    LON range: {df_raw['LON'].min():.6f} to {df_raw['LON'].max():.6f}")

        # Compute overlap between post-processed and raw GPS timestamps
        overlap_start = max(df_postproc['GPS_TIME'].min(), df_raw['GPS_TIME'].min())
        overlap_end = min(df_postproc['GPS_TIME'].max(), df_raw['GPS_TIME'].max())
        overlap_duration = max(0, overlap_end - overlap_start)

        postproc_duration = df_postproc['GPS_TIME'].max() - df_postproc['GPS_TIME'].min()
        raw_duration = df_raw['GPS_TIME'].max() - df_raw['GPS_TIME'].min()

        postproc_frac = overlap_duration / postproc_duration if postproc_duration > 0 else 0
        raw_frac = overlap_duration / raw_duration if raw_duration > 0 else 0

        print(f"    Overlap duration: {overlap_duration:.1f} s  "
              f"({postproc_frac*100:.1f}% of postproc, {raw_frac*100:.1f}% of raw)")

        result = {
            'label': label,
            'prj': row['prj'],
            'set': row['set'],
            'trn': row['trn'],
            'stream': row['stream'],
            'full_path': gps_path,
            'n_records': len(df_raw),
            'raw_start': df_raw['GPS_TIME'].min(),
            'raw_end': df_raw['GPS_TIME'].max(),
            'raw_start_utc': str(t_start_raw),
            'raw_end_utc': str(t_end_raw),
            'raw_duration_s': raw_duration,
            'overlap_duration_s': overlap_duration,
            'postproc_overlap_frac': postproc_frac,
            'raw_overlap_frac': raw_frac,
            'has_position': has_position,
            'has_radar': has_radar,
            'radar_streams': radar_streams,
            'error': None,
        }
        results.append(result)

    # ========================================================================
    # 4. Summary table
    # ========================================================================

    postproc_duration = df_postproc['GPS_TIME'].max() - df_postproc['GPS_TIME'].min()

    df_results = pd.DataFrame(results)
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"\nPost-processed file: {Path(postprocessed_gps_path).name}")
    print(f"  Time (UTC): {t_start_postproc} to {t_end_postproc}")
    print(f"  Duration: {postproc_duration:.1f} s ({postproc_duration/3600:.2f} h)\n")

    if len(df_results) == 0:
        print("  No candidates with radar data found.")
        print(f"\nFigure skipped (no candidates): {Path(postprocessed_gps_path).stem}")
        return

    summary_cols = ['label', 'n_records', 'has_radar', 'radar_streams',
                    'raw_start_utc', 'raw_end_utc',
                    'raw_duration_s', 'overlap_duration_s',
                    'postproc_overlap_frac', 'raw_overlap_frac', 'error']
    available_cols = [c for c in summary_cols if c in df_results.columns]
    print(df_results[available_cols].to_string(index=False))

    # ========================================================================
    # 5. Compare RADAR_TIME from radar CT files against GPS streams
    # ========================================================================

    print("\n" + "=" * 100)
    print("RADAR_TIME COMPARISON")
    print("=" * 100)

    # Collect radar CT info for transects that have radar
    radar_ct_results = []
    for _, r in df_results[df_results.get('has_radar', pd.Series(dtype=bool)) == True].iterrows():
        prj, set_, trn = r['prj'], r['set'], r['trn']

        # Find radar paths for this prj/set/trn
        radar_rows = df_artifacts[
            (df_artifacts['prj'] == prj) &
            (df_artifacts['set'] == set_) &
            (df_artifacts['trn'] == trn) &
            (df_artifacts['stream'].str.startswith('RAD'))
        ]
        for _, rad_row in radar_rows.drop_duplicates(subset=['prj', 'set', 'trn', 'stream']).iterrows():
            radar_dir = Path(rad_row['full_path']).parent
            try:
                ct_df = stream_util.load_ct_file(str(radar_dir), parse=True)
                radar_ct_results.append({
                    'label': f"{prj}/{set_}/{trn}/{rad_row['stream']}",
                    'n_records': len(ct_df),
                    'radar_time_min': ct_df['RADAR_TIME'].min(),
                    'radar_time_max': ct_df['RADAR_TIME'].max(),
                    'comp_time_min': ct_df['COMP_TIME'].min(),
                    'comp_time_max': ct_df['COMP_TIME'].max(),
                    'comp_time_start_utc': str(pd.Timestamp.utcfromtimestamp(ct_df['COMP_TIME'].min())),
                    'comp_time_end_utc': str(pd.Timestamp.utcfromtimestamp(ct_df['COMP_TIME'].max())),
                })
                print(f"\n  {prj}/{set_}/{trn}/{rad_row['stream']}:")
                print(f"    CT records: {len(ct_df)}")
                print(f"    RADAR_TIME range: {ct_df['RADAR_TIME'].min()} to {ct_df['RADAR_TIME'].max()}")
                print(f"    COMP_TIME (UTC): {pd.Timestamp.utcfromtimestamp(ct_df['COMP_TIME'].min())} to {pd.Timestamp.utcfromtimestamp(ct_df['COMP_TIME'].max())}")
            except Exception as e:
                print(f"\n  {prj}/{set_}/{trn}/{rad_row['stream']}: ERROR loading CT: {e}")

    # Also load RADAR_TIME from the GPS CT files for comparison
    print("\n  --- GPS stream RADAR_TIME ranges ---")
    gps_ct_results = []
    for _, r in df_results.dropna(subset=['raw_start']).iterrows():
        gps_dir = Path(r['full_path']).parent
        try:
            ct_df = stream_util.load_ct_file(str(gps_dir), parse=True)
            if 'RADAR_TIME' in ct_df.columns:
                gps_ct_results.append({
                    'label': r['label'],
                    'n_records': len(ct_df),
                    'radar_time_min': ct_df['RADAR_TIME'].min(),
                    'radar_time_max': ct_df['RADAR_TIME'].max(),
                })
                print(f"  {r['label']}: RADAR_TIME {ct_df['RADAR_TIME'].min()} to {ct_df['RADAR_TIME'].max()} ({len(ct_df)} records)")
        except Exception as e:
            print(f"  {r['label']}: ERROR loading CT: {e}")

    df_radar_ct = pd.DataFrame(radar_ct_results)
    df_gps_ct = pd.DataFrame(gps_ct_results)

    # ========================================================================
    # 6. Plot timeline comparison
    # ========================================================================

    # Dynamic figure height based on number of entries
    n_gps_bars = len(df_results.dropna(subset=['raw_start'])) + 1  # +1 for postproc
    n_ct_bars = len(gps_ct_results) + len(radar_ct_results)
    top_h = max(2, n_gps_bars * 0.5 + 1)
    mid_h = max(2, n_ct_bars * 0.5 + 1)
    bot_h = 4
    fig_h = top_h + mid_h + bot_h
    fig, axes = plt.subplots(3, 1, figsize=(14, fig_h),
                             gridspec_kw={'height_ratios': [top_h, mid_h, bot_h]})

    # --- Top panel: GPS_TIME timeline bars ---
    ax = axes[0]

    # Post-processed GPS bar
    ax.barh(0, postproc_duration, left=df_postproc['GPS_TIME'].min(),
            height=0.6, color='black', alpha=0.7, label='Post-processed')

    # Raw GPS bars — hatched if radar exists, solid if not
    valid_results = df_results.dropna(subset=['raw_start'])
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(valid_results), 1)))
    for i, (_, r) in enumerate(valid_results.iterrows()):
        hatch = '///' if r.get('has_radar', False) else None
        edgecolor = 'black' if r.get('has_radar', False) else None
        suffix = ' [RAD]' if r.get('has_radar', False) else ''
        ax.barh(i + 1, r['raw_duration_s'], left=r['raw_start'],
                height=0.6, color=colors[i], alpha=0.7,
                hatch=hatch, edgecolor=edgecolor,
                label=r['label'] + suffix)

    ax.set_yticks(range(len(valid_results) + 1))
    ylabels = []
    for _, r in valid_results.iterrows():
        marker = ' *' if r.get('has_radar', False) else ''
        ylabels.append(r['label'] + marker)
    ax.set_yticklabels(['Post-processed'] + ylabels)
    ax.set_xlabel('GPS_TIME (Unix seconds)')
    ax.set_title('GPS_TIME Overlap: Post-processed vs Raw GPS Streams  (* = has radar data)')
    ax.legend(loc='upper right', fontsize=8)

    # Set x-axis limits centered on the post-processed time range with padding,
    # so the post-processed bar is always visible (not compressed to a sliver)
    pp_min = df_postproc['GPS_TIME'].min()
    pp_max = df_postproc['GPS_TIME'].max()
    pp_span = pp_max - pp_min
    padding = max(pp_span * 2, 3600)  # At least 1 hour of padding on each side
    ax.set_xlim(pp_min - padding, pp_max + padding)
    ax.axvline(pp_min, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.axvline(pp_max, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    # --- Middle panel: RADAR_TIME timeline bars ---
    ax_rt = axes[1]

    all_ct_entries = []
    # GPS CT entries
    for i_gps, (_, r) in enumerate(df_gps_ct.iterrows()):
        all_ct_entries.append({
            'label': r['label'],
            'radar_time_min': r['radar_time_min'],
            'radar_time_max': r['radar_time_max'],
            'is_radar': False,
            'color_idx': i_gps,
        })
    # Radar CT entries
    for _, r in df_radar_ct.iterrows():
        all_ct_entries.append({
            'label': r['label'],
            'radar_time_min': r['radar_time_min'],
            'radar_time_max': r['radar_time_max'],
            'is_radar': True,
        })

    # Assign colors: GPS streams reuse the colors from the top panel, radar streams get red shades
    radar_colors = plt.cm.Reds(np.linspace(0.4, 0.8, max(len(df_radar_ct), 1)))
    i_radar = 0
    for j, entry in enumerate(all_ct_entries):
        duration = entry['radar_time_max'] - entry['radar_time_min']
        if entry['is_radar']:
            c = radar_colors[i_radar]
            i_radar += 1
            hatch = '///'
            edgecolor = 'black'
        else:
            c = colors[entry['color_idx']] if entry['color_idx'] < len(colors) else 'gray'
            hatch = None
            edgecolor = None
        ax_rt.barh(j, duration, left=entry['radar_time_min'],
                   height=0.6, color=c, alpha=0.7, hatch=hatch, edgecolor=edgecolor)

    ax_rt.set_yticks(range(len(all_ct_entries)))
    ax_rt.set_yticklabels([e['label'] for e in all_ct_entries])
    ax_rt.set_xlabel('RADAR_TIME (microseconds)')
    ax_rt.set_title('RADAR_TIME ranges: GPS streams vs Radar CT files')

    # --- Bottom panel: lat/lon comparison for streams with position data ---
    ax2 = axes[2]

    # Plot post-processed positions
    ax2.plot(df_postproc['LON'], df_postproc['LAT'], 'k-', linewidth=2,
             label='Post-processed', alpha=0.8)

    # Plot raw GPS positions for streams that have them
    for i, (_, r) in enumerate(valid_results.iterrows()):
        if not r.get('has_position', False):
            continue
        try:
            df_raw = stream_util.load_xds_stream_file(r['full_path'], parse=True)
            if 'LAT' in df_raw.columns and 'LON' in df_raw.columns:
                ax2.plot(df_raw['LON'], df_raw['LAT'], '--', color=colors[i],
                         linewidth=1.5, alpha=0.7, label=r['label'])
        except Exception:
            pass

    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('Position Comparison')
    ax2.legend(fontsize=8)
    ax2.set_aspect('equal')

    plt.tight_layout()
    output_dir = "outputs/gps_comparison"
    output_fig = f"{output_dir}/gps_comparison_{Path(postprocessed_gps_path).stem}.png"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(output_fig, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved to: {output_fig}")


if __name__ == '__main__':
    # Load file index once (shared across all comparisons)
    print("Loading file index...")
    df_files = file_index.load_file_index_df(base_path, cache_file, read_cache=True)
    df_artifacts = file_index.create_artifacts_df(df_files)

    if len(sys.argv) >= 3:
        # CLI mode: path and trn_prefix provided
        run_comparison(sys.argv[1], sys.argv[2], df_artifacts)
    elif len(sys.argv) == 2:
        # CLI mode: path provided, auto-extract trn_prefix
        gps_path = sys.argv[1]
        trn_prefix = extract_trn_prefix(gps_path)
        print(f"Auto-extracted trn_prefix: '{trn_prefix}'")
        run_comparison(gps_path, trn_prefix, df_artifacts)
    else:
        # Batch mode: run all 2008 season files
        gps_dir = "/resfs/GROUPS/CRESIS/dataproducts/metadata/2008_Antarctica_BaslerJKB/gps"
        files = [
            "IPUTG1B_2008364_MCM_JKB0a_BISL01a_position.txt",
            "IPUTG1B_2008364_MCM_JKB0a_BISL01b_position.txt",
            "IPUTG1B_2008364_MCM_JKB0a_RIS02a_position.txt",
            "IPUTG1B_2008364_MCM_JKB0a_WISL01a_position.txt",
            "IPUTG1B_2009006_DMC_JKB0a_ATAL01a_position.txt",
            "IPUTG1B_2009032_ASB_JKB0a_R08Wb_position.txt",
            "IPUTG1B_2009033_DMC_JKB0a_WLKX10a_position.txt",
        ]
        for fname in files:
            gps_path = f"{gps_dir}/{fname}"
            trn_prefix = extract_trn_prefix(gps_path)
            print(f"\nAuto-extracted trn_prefix: '{trn_prefix}' from {fname}")
            run_comparison(gps_path, trn_prefix, df_artifacts)
