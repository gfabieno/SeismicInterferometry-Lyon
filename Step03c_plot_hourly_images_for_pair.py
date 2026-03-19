"""Plot hourly images for a station pair and overlay the mean trace.

This module provides `plot_hourly_images_for_pair` previously embedded in
`STEP04_build_reference_window_selection.py`. The function reads per-segment
HDF5 xcorr results, stacks them into hourly images (TT, RR, ZZ), applies a
bandpass and overlays the mean trace (computed across hours) centered on the
image.
"""
from pathlib import Path
from typing import Optional
import torch
import matplotlib.pyplot as plt

from cross_correlation import butterworth_bandpass
from utils import rotate_NE_corr_to_RR_TT, compute_theta


def plot_hourly_images_for_pair(seg_files, coords, pairs, i, j, pidx,
                                nhours: int = None,
                                maxlag_s: Optional[float] = None,
                                fmin=2, fmax=5,
                                normalize: bool = True,
                                beam_xcorr_outdir: Optional[Path] = None):
    """Build hourly TT/RR/ZZ images for a pair and overlay the mean trace.

    Parameters mirror the previous implementation in STEP04. The overlay is
    computed as the arithmetic mean across the stacked hourly traces for each
    component and drawn at the vertical center of the image.
    """
    if nhours:
        seg_files = seg_files[:nhours]
    if len(seg_files) == 0:
        raise RuntimeError("No segment files found")

    TT_rows, RR_rows, ZZ_rows = [], [], []
    lags = None
    # compute azimuth for rotation: use compute_theta on the single pair (i,j)
    import numpy as _np
    theta = float(compute_theta(coords, _np.asarray([[i, j]], dtype=np.int32))[0])

    for f in seg_files:
        try:
            with h5py.File(f, "r") as h5:
                g = h5.get("xcorr", h5)
                lags = g["lags"][...].astype(np.float32)
                corr = g["corr"]
                ZZ = corr[pidx, 0, 0, :].astype(np.float32)
                NN = corr[pidx, 1, 1, :].astype(np.float32)
                NE = corr[pidx, 1, 2, :].astype(np.float32)
                EN = corr[pidx, 2, 1, :].astype(np.float32)
                EE = corr[pidx, 2, 2, :].astype(np.float32)
                RR, TT = rotate_NE_corr_to_RR_TT(NN, NE, EN, EE, np.asarray(theta))
                if maxlag_s is not None:
                    mask = (lags >= -maxlag_s) & (lags <= maxlag_s)
                    lags = lags[mask]
                    ZZ = ZZ[mask]
                    RR = RR[mask]
                    TT = TT[mask]
                TT_rows.append(TT)
                RR_rows.append(RR)
                ZZ_rows.append(ZZ)
        except Exception as e:
            print(f"Warning: skipping file {f} due to error: {e}")

    if lags is None:
        raise RuntimeError("Could not load lags from any segment file")

    TT_im = np.vstack(TT_rows)
    RR_im = np.vstack(RR_rows)
    ZZ_im = np.vstack(ZZ_rows)
    if normalize:
        TT_im = TT_im / np.sqrt(np.sum(TT_im**2, axis=-1, keepdims=True))
        RR_im = RR_im / np.sqrt(np.sum(RR_im**2, axis=-1, keepdims=True))
        ZZ_im = ZZ_im / np.sqrt(np.sum(ZZ_im**2, axis=-1, keepdims=True))

    dt = abs(float(lags[1] - lags[0]))
    TT_im = butterworth_bandpass(torch.from_numpy(TT_im), dt, fmin, fmax).cpu().numpy()
    RR_im = butterworth_bandpass(torch.from_numpy(RR_im), dt, fmin, fmax).cpu().numpy()
    ZZ_im = butterworth_bandpass(torch.from_numpy(ZZ_im), dt, fmin, fmax).cpu().numpy()

    # Create figure with a bottom axis (20% height) for the mean traces.
    # Use GridSpec to make top row (images) occupy 80% and bottom row 20%.
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[4, 1], hspace=0.05)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    # create one bottom axis per column, sharing x-axis with the corresponding top image
    axs_bot = [fig.add_subplot(gs[1, i], sharex=axs[i]) for i in range(3)]

    extent = (float(lags[0]), float(lags[-1]), float(len(seg_files) - 0.5), float(-0.5))
    axs[0].imshow(TT_im, aspect="auto", extent=extent, interpolation="nearest", cmap="gray")
    axs[0].set_title("TT")
    axs[0].set_xlabel("Lag (s)")
    axs[0].set_ylabel("Hour index")
    axs[0].invert_yaxis()
    axs[1].imshow(RR_im, aspect="auto", extent=extent, interpolation="nearest", cmap="gray")
    axs[1].set_title("RR")
    axs[1].set_xlabel("Lag (s)")
    axs[1].invert_yaxis()
    axs[2].imshow(ZZ_im, aspect="auto", extent=extent, interpolation="nearest", cmap="gray")
    axs[2].set_title("ZZ")
    axs[2].set_xlabel("Lag (s)")
    axs[2].invert_yaxis()

    # Compute mean traces (across hours) for each component and plot them in the bottom axis.
    mean_TT = np.nanmean(TT_im, axis=0)
    mean_RR = np.nanmean(RR_im, axis=0)
    mean_ZZ = np.nanmean(ZZ_im, axis=0)

    # Normalize by the maximum absolute value across all three means to display together
    all_abs_max = np.nanmax(np.abs(np.concatenate([mean_TT, mean_RR, mean_ZZ])))
    if all_abs_max == 0 or np.isnan(all_abs_max):
        all_abs_max = 1.0

    # Plot each mean trace on its own bottom axis
    axs_bot[0].plot(lags, mean_TT / all_abs_max, color="tab:blue", linewidth=1.2)
    axs_bot[0].set_title("Mean TT")
    axs_bot[0].grid(True, linestyle="--", alpha=0.3)

    axs_bot[1].plot(lags, mean_RR / all_abs_max, color="tab:orange", linewidth=1.2)
    axs_bot[1].set_title("Mean RR")
    axs_bot[1].grid(True, linestyle="--", alpha=0.3)

    axs_bot[2].plot(lags, mean_ZZ / all_abs_max, color="tab:green", linewidth=1.2)
    axs_bot[2].set_title("Mean ZZ")
    axs_bot[2].grid(True, linestyle="--", alpha=0.3)

    # Label only the bottom row x-axis and a common y-label
    for ax in axs:
        plt.setp(ax.get_xticklabels(), visible=False)
    axs_bot[0].set_xlabel("Lag (s)")
    axs_bot[0].set_ylabel("Normalized mean")

    plt.suptitle(f"Pair {pairs[pidx,0]}-{pairs[pidx,1]} | theta={theta:.2f} rad", fontsize=12)
    beam_xcorr_outdir = Path(beam_xcorr_outdir)
    beam_xcorr_outdir.mkdir(parents=True, exist_ok=True)

    out_name = f"hourly_pair_{pairs[pidx,0]}_{pairs[pidx,1]}.png"

    saved_path = beam_xcorr_outdir / out_name
    plt.savefig(saved_path, dpi=200)
    plt.close(fig)


    # --- Now compute and plot spectra: per-hour spectra (top) and mean spectrum (bottom)
    n = TT_im.shape[1]
    dt = abs(float(lags[1] - lags[0]))
    # frequency axis for rfft
    freqs = np.fft.rfftfreq(n, d=dt)

    # compute amplitude spectra per hour (rows -> hours)
    spec_TT = np.abs(np.fft.rfft(TT_im, axis=1))
    spec_RR = np.abs(np.fft.rfft(RR_im, axis=1))
    spec_ZZ = np.abs(np.fft.rfft(ZZ_im, axis=1))

    # mean spectra across hours
    mean_spec_TT = np.nanmean(spec_TT, axis=0)
    mean_spec_RR = np.nanmean(spec_RR, axis=0)
    mean_spec_ZZ = np.nanmean(spec_ZZ, axis=0)

    # create figure: 2x3 grid similar to image layout
    fig2 = plt.figure(figsize=(14, 6))
    gs2 = fig2.add_gridspec(2, 3, height_ratios=[4, 1], hspace=0.05)
    axs2_top = [fig2.add_subplot(gs2[0, i]) for i in range(3)]
    axs2_bot = [fig2.add_subplot(gs2[1, i], sharex=axs2_top[i]) for i in range(3)]

    extent_f = (float(freqs[0]), float(freqs[-1]), float(len(seg_files) - 0.5), float(-0.5))

    im0 = axs2_top[0].imshow(spec_TT, aspect='auto', extent=extent_f, interpolation='nearest', cmap='magma')
    axs2_top[0].set_title('Spectra TT (hours x freq)')
    axs2_top[0].invert_yaxis()
    axs2_top[1].imshow(spec_RR, aspect='auto', extent=extent_f, interpolation='nearest', cmap='magma')
    axs2_top[1].set_title('Spectra RR (hours x freq)')
    axs2_top[1].invert_yaxis()
    axs2_top[2].imshow(spec_ZZ, aspect='auto', extent=extent_f, interpolation='nearest', cmap='magma')
    axs2_top[2].set_title('Spectra ZZ (hours x freq)')
    axs2_top[2].invert_yaxis()

    # bottom: mean spectra lines
    axs2_bot[0].plot(freqs, mean_spec_TT, color='tab:blue')
    axs2_bot[0].set_title('Mean spectrum TT')
    axs2_bot[1].plot(freqs, mean_spec_RR, color='tab:orange')
    axs2_bot[1].set_title('Mean spectrum RR')
    axs2_bot[2].plot(freqs, mean_spec_ZZ, color='tab:green')
    axs2_bot[2].set_title('Mean spectrum ZZ')
    axs2_bot[0].set_xlim(right=2*fmax)
    axs2_bot[1].set_xlim(right=2*fmax)
    axs2_bot[2].set_xlim(right=2*fmax)

    for ax in axs2_top:
        plt.setp(ax.get_xticklabels(), visible=False)
    axs2_bot[0].set_xlabel('Frequency (Hz)')
    axs2_bot[0].set_ylabel('Amplitude')

    spec_name = f"hourly_pair_{pairs[pidx,0]}_{pairs[pidx,1]}_spectra.png"
    saved_spec = beam_xcorr_outdir / spec_name
    fig2.savefig(saved_spec, dpi=200)
    plt.close(fig2)

    return saved_path, saved_spec


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--beam-xcorr-dir", default="outputs/Step03_beamforming_xcorr",
                    type=Path,help="Root folder where Step03 wrote meta.h5 and segments/")
    ap.add_argument("--sta1", default="A002",
                    help="First station code (e.g., A001)")
    ap.add_argument("--sta2", default="A021",
                    help="Second station code (e.g., A002)")
    ap.add_argument("--nhours", default=None, type=int,
                    help="Limit to first N segments/hours")
    ap.add_argument("--maxlag", default=None, type=float,
                    help="Max lag (s) to display around zero")
    ap.add_argument("--out-png", default=None,
                    help="Optional explicit output PNG path")
    ap.add_argument("--show-doc", action="store_true",
                    help="Print docstring and exit")
    ap.add_argument("--normalize", action="store_true", default=True,
                    help="Normalize each trace to 0-1")
    args = ap.parse_args()

    if args.show_doc:
        print(__doc__)
        sys.exit(0)

    # Lazy imports
    import h5py
    import numpy as np
    from utils import load_meta, xcorr_segment_files

    beam_xcorr_dir = Path(args.beam_xcorr_dir)
    meta_path = beam_xcorr_dir / "meta.h5"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.h5 not found under {beam_xcorr_dir}")

    # load meta: stations, coords_xy_m, pairs_i_j
    with h5py.File(meta_path, "r") as h5:
        stations = [s.decode() if isinstance(s, (bytes, np.bytes_)) else str(s) for s in h5["meta"]["stations"][...]]
        coords = h5["meta"]["coords_xy_m"][...].astype(np.float32)
        pairs = h5["meta"]["pairs_i_j"][...].astype(np.int32)

    # build mappings
    sta_to_idx = {s: i for i, s in enumerate(stations)}
    pair_to_pidx = { (int(p[0]), int(p[1])): idx for idx, p in enumerate(pairs) }

    if args.sta1 not in sta_to_idx or args.sta2 not in sta_to_idx:
        raise KeyError(f"Stations not found in meta.h5: {args.sta1}, {args.sta2}")
    i = sta_to_idx[args.sta1]
    j = sta_to_idx[args.sta2]
    if i == j:
        raise ValueError("sta1 and sta2 must be different")
    if i > j:
        i, j = j, i

    pidx = pair_to_pidx.get((i, j), None)
    if pidx is None:
        raise KeyError(f"Pair ({args.sta1}, {args.sta2}) not found in meta pairs")

    # discover segment files
    seg_dir = beam_xcorr_dir / "segments"
    seg_files, times = xcorr_segment_files(seg_dir)

    # call plotting function
    beam_xcorr_outdir  = beam_xcorr_dir / "figures"/ "xcorr"

    saved = plot_hourly_images_for_pair(seg_files=seg_files, coords=coords, pairs=pairs, i=i, j=j, pidx=int(pidx), nhours=args.nhours, maxlag_s=args.maxlag, beam_xcorr_outdir=beam_xcorr_outdir)
    print(f"Wrote: {saved}")


