# Step04_plot_xcorr.py
#
# Plots for Step04 cross-correlations:
# 1) For one station pair: hourly TT, RR, ZZ as 2D images (hour vs lag) using imshow
# 2) Over whole monitoring period: average TT, RR, ZZ gathers with traces sorted by offset
# 3) Build / cache reference traces and per-pair windows and plot them (new)
#
# Usage examples:
#   python STEP04_build_reference_window_selection.py --xcorr-dir outputs/beam_xcorr --sta1 A002 --sta2 A021 --nhours 4
#   python STEP04_build_reference_window_selection.py --xcorr-dir outputs/beam_xcorr --build-reference --ref-method trim --ref-comps ZZ,RR,TT
#
# Notes:
# - Assumes Step03 wrote:
#     outputs/beam_xcorr/meta.h5 (with stations, coords_xy_m, pairs_i_j)
#     outputs/beam_xcorr/segments/bf_xc_YYYYMMDDTHHMMSSZ.h5 (with xcorr/corr, xcorr/lags, xcorr/pairs_i_j)
# - Assumes corr is (P,3,3,nlag) with components order [Z,N,E].

import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from cross_correlation import butterworth_bandpass
import torch
from typing import Optional, Tuple, Dict
from utils import (
    robust_ref_stack,
    extract_components_from_corr,
    percentile_clip,
    apply_window_to_refs,
    compute_offsets_m,
    compute_theta,
    rotate_NE_corr_to_RR_TT,
    xcorr_segment_files, load_meta,
)
from tqdm import tqdm


def azimuth_EN(coords_xy_m: np.ndarray, i: int, j: int) -> float:
    """Azimuth (radians) from station i to j, measured clockwise from North."""
    dE = float(coords_xy_m[j, 0] - coords_xy_m[i, 0])
    dN = float(coords_xy_m[j, 1] - coords_xy_m[i, 1])
    return np.arctan2(dE, dN)


def get_pair_indices(sta_to_idx, pair_to_pidx, sta1: str, sta2: str):
    i = sta_to_idx[sta1]
    j = sta_to_idx[sta2]
    if i == j:
        raise ValueError("sta1 and sta2 must be different")
    if i > j:
        i, j = j, i
    pidx = pair_to_pidx.get((i, j), None)
    if pidx is None:
        raise KeyError(f"Pair ({sta1}, {sta2}) not found in meta pairs")
    return i, j, pidx


# ---------------- New utilities: reference building, windows, plotting ----------------

def _as_date_str_list(xs) -> list[str]:
    out: list[str] = []
    for x in xs:
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode())
        else:
            out.append(str(x))
    return out



def _plot_reference_gather(refs: Dict[str, np.ndarray], lags: np.ndarray, offsets_m: np.ndarray, out_png: Path, *, clip_pct: float = 99.0, title: str = "Reference gather", windows: Optional[Tuple[np.ndarray, np.ndarray]] = None, n_offsets: Optional[int] = None):
    """Plot reference gather after resampling traces on a regular offset grid.

    refs: dict comp->(P,T) where P corresponds to offsets_m
    lags: (T,)
    offsets_m: (P) unsorted or sorted offsets in meters
    out_png: output path

    The function resamples each (P,T) component to a regular offset grid using linear
    interpolation across pairs (offset axis). Windows (t0,t1) are also interpolated
    to the same offset grid for plotting.
    """
    comps = list(refs.keys())
    n = len(comps)

    # Ensure arrays
    offsets_m = np.asarray(offsets_m, dtype=np.float32)
    lags = np.asarray(lags, dtype=np.float32)

    # sort by offsets just in case
    order = np.argsort(offsets_m)
    offsets_sorted = offsets_m[order]

    # Determine target regular offset grid
    if n_offsets is None:
        n_offsets = offsets_sorted.size
    else:
        n_offsets = int(n_offsets)
        if n_offsets < 1:
            n_offsets = offsets_sorted.size

    new_offsets = np.linspace(float(offsets_sorted[0]), float(offsets_sorted[-1]), n_offsets)

    extent = (float(new_offsets.min()), float(new_offsets.max()), float(lags[0]), float(lags[-1]))

    fig, axs = plt.subplots(1, n, figsize=(5 * n, 6), constrained_layout=True, sharey=True)
    if n == 1:
        axs = [axs]

    for ax, comp in zip(axs, comps):
        R = refs[comp]
        # reorder rows according to sorted offsets
        if R.shape[0] != offsets_m.size:
            raise ValueError(f"Reference component {comp} has incompatible first dim: {R.shape[0]} vs offsets {offsets_m.size}")
        R_sorted = R[order, :]

        # Interpolate each lag column across offsets to the new regular grid
        T = R_sorted.shape[1]
        R_reg = np.empty((n_offsets, T), dtype=np.float32)
        for ti in range(T):
            R_reg[:, ti] = np.interp(new_offsets, offsets_sorted, R_sorted[:, ti])

        img = R_reg.T
        vmin, vmax = percentile_clip(img, clip_pct)
        im = ax.imshow(img, aspect="auto", extent=extent, origin="lower", interpolation="nearest", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(comp)
        ax.set_xlabel("Offset (m)")
        ax.set_ylabel("Lag (s)")
        fig.colorbar(im, ax=ax, shrink=0.9)

        # If windows are provided (per-original-offset), interpolate them to new_offsets
        if windows is not None:
            t0_s, t1_s = windows
            t0_s = np.asarray(t0_s)
            t1_s = np.asarray(t1_s)
            if t0_s.size != offsets_m.size or t1_s.size != offsets_m.size:
                raise ValueError("Window arrays must have same length as offsets_m")
            t0_reg = np.interp(new_offsets, offsets_sorted, t0_s[order])
            t1_reg = np.interp(new_offsets, offsets_sorted, t1_s[order])
            ax.plot(new_offsets, t0_reg, "r-", lw=1.0, alpha=0.9)
            ax.plot(new_offsets, t1_reg, "r-", lw=1.0, alpha=0.9)
            # also plot the symmetric anti-causal windows (negative lags): [-t1, -t0]
            t0_reg_neg = -t1_reg
            t1_reg_neg = -t0_reg
            ax.plot(new_offsets, t0_reg_neg, "b--", lw=1.0, alpha=0.9)
            ax.plot(new_offsets, t1_reg_neg, "b--", lw=1.0, alpha=0.9)

    fig.suptitle(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def build_references(*, seg_files: list[Path], coords_xy_m: np.ndarray, pairs: np.ndarray, out_h5: Path, comps: list[str], ref_method: str = "mean", trim_pct: float = 0.1, max_files: int = 0, overwrite: bool = False, pair_chunk: Optional[int] = None):
    """Build and cache reference traces only (no window selection).

    This implementation reads each segment file once and extracts all requested components
    so we don't reopen HDF5 files per component.
    """
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    if out_h5.exists() and not overwrite:
        print(f"Reference file {out_h5} already exists; use --overwrite to replace")
        return
    if max_files and max_files > 0:
        seg_files = seg_files[: int(max_files)]
    if len(seg_files) == 0:
        raise RuntimeError("No segment files found to build reference")

    # Lag axis
    with h5py.File(seg_files[0], "r") as h5:
        g0 = h5.get("xcorr", h5)
        lags = g0["lags"][...].astype(np.float32)

    offsets_m = compute_offsets_m(coords_xy_m, pairs)
    theta = compute_theta(coords_xy_m, pairs)

    P = pairs.shape[0]
    D = len(seg_files)

    # effective pair chunk
    if pair_chunk is None:
        pair_chunk_eff = P
    else:
        pair_chunk_eff = int(pair_chunk)
        if pair_chunk_eff < 1:
            raise SystemExit("--pair-chunk must be a positive integer or omitted")
        if pair_chunk_eff > P:
            pair_chunk_eff = P

    comps_u = [c.strip().upper() for c in comps]

    refs: Dict[str, np.ndarray] = {c: np.zeros((P, lags.size), dtype=np.float32) for c in comps_u}
    seg_ids: list[str] = []

    # Print summary of operation
    print("Building reference traces with parameters:")
    print(f"  method = {ref_method}")
    print(f"  comps   = {','.join(comps_u)}")
    print(f"  n_pairs = {P}")
    print(f"  n_segs  = {D}")
    print(f"  pair_chunk = {pair_chunk_eff}")

    if ref_method.lower() == "mean":
        # Accumulate sums for all comps in one pass (open each file once)
        acc: Dict[str, np.ndarray] = {c: np.zeros((P, lags.size), dtype=np.float64) for c in comps_u}
        count = 0
        print("Building mean reference...")
        for fp in tqdm(seg_files, desc="Building mean reference", total=len(seg_files)):
            with h5py.File(fp, "r") as h5:
                g = h5.get("xcorr", h5)
                corr = g["corr"][...].astype(np.float32)
                comp_traces = extract_components_from_corr(corr, theta, comps_u)
                for comp, arr in comp_traces.items():
                    acc[comp] += arr.astype(np.float64)
                seg_ids.append(str(g.attrs.get("starttime", h5.attrs.get("starttime", fp.name))))
                count += 1
        if count < 1:
            raise RuntimeError("No segments to build reference")
        for comp in comps_u:
            refs[comp] = (acc[comp] / float(count)).astype(np.float32)

    else:
        # Median/alpha-trim path: process in pair-chunks but extract all comps per segment once
        print(f"Building reference in pair-chunks (chunk={pair_chunk_eff}) for comps: {','.join(comps_u)}")
        n_chunks = (P + pair_chunk_eff - 1) // pair_chunk_eff
        for chunk_i, p0 in enumerate(range(0, P, pair_chunk_eff), start=1):
            p1 = min(P, p0 + pair_chunk_eff)
            pc = p1 - p0
            print(f"  Chunk {chunk_i}/{n_chunks}: ref pairs [{p0}:{p1}] (pc={pc})")
            ncomps = len(comps_u)
            buf = np.empty((D, ncomps, pc, lags.size), dtype=np.float32)
            # iterate over segment files with progress bar
            for di, fp in enumerate(tqdm(seg_files, desc=f"    Chunk {chunk_i}/{n_chunks} files", total=D), start=0):
                with h5py.File(fp, "r") as h5:
                    g = h5.get("xcorr", h5)
                    corr = g["corr"][...].astype(np.float32)
                    comp_traces = extract_components_from_corr(corr, theta, comps_u)
                    for ci, comp in enumerate(comps_u):
                        buf[di, ci, :, :] = comp_traces[comp][p0:p1, :]
                    if di == 0:
                        seg_ids.append(str(g.attrs.get("starttime", h5.attrs.get("starttime", fp.name))))
            # aggregate per component for this chunk
            for ci, comp in enumerate(comps_u):
                ref_chunk = robust_ref_stack(buf[:, ci, :, :], method=ref_method, trim_pct=trim_pct)
                refs[comp][p0:p1, :] = ref_chunk
            print(f"  Finished chunk {chunk_i}/{n_chunks}")

    # write refs to HDF5
    with h5py.File(out_h5, "w") as h5:
        h5.attrs["ref_method"] = str(ref_method)
        h5.attrs["trim_pct"] = float(trim_pct)
        h5.create_dataset("lags", data=lags.astype(np.float32))
        h5.create_dataset("pairs_i_j", data=pairs.astype(np.int32))
        h5.create_dataset("offsets_m", data=offsets_m.astype(np.float32))
        h5.create_dataset("segments", data=np.array(_as_date_str_list(seg_ids), dtype="S"))
        gref = h5.create_group("ref")
        # store the canonical component names as a dataset for discovery by downstream steps
        comp_list = np.array([c.encode("utf-8") for c in comps_u], dtype="S")
        gref.create_dataset("components", data=comp_list)
        gref.attrs["n_components"] = int(len(comps_u))
        for comp, ref in refs.items():
            gref.create_dataset(comp, data=ref.astype(np.float32), compression="gzip", compression_opts=4)
        # also write a top-level attribute indicating this file was produced by STEP04
        h5.attrs["produced_by"] = "STEP04_build_reference_window_selection.py"
    return


def compute_windows(*, ref_h5: Path, window_mode: str = "velpeak", vref_mps: float = 1500.0, vel_search_half_width: float = 0.15, peak_comp: str = "RR", peak_half_width_s: float = 0.25, min_lag_s: float = 0.05, overwrite: bool = False):
    """Compute per-pair windows from an existing reference HDF5 file and add/update the window group.

    This lets the user build references once and then experiment with different window criteria.
    """
    if not ref_h5.exists():
        raise FileNotFoundError(f"Reference HDF5 not found: {ref_h5}")
    else:
        if "window" in h5py.File(ref_h5, "r") and not overwrite:
            print(f"Window group already exists in {ref_h5}; use --overwrite to replace")
            return

    with h5py.File(ref_h5, "r") as h5:
        lags = h5["lags"][...].astype(np.float32)
        offsets_m = h5["offsets_m"][...].astype(np.float32)
        pairs = h5["pairs_i_j"][...].astype(np.int32)
        refs = {k.decode(): h5["ref"][k][...].astype(np.float32)
                for k in h5["ref"]['components']}

    peak_comp = peak_comp.strip().upper()
    if peak_comp not in refs:
        peak_comp = list(refs.keys())[0]
    ref_for_peak = refs[peak_comp]

    P = pairs.shape[0]
    t0_s = np.zeros(P, dtype=np.float32)
    t1_s = np.zeros(P, dtype=np.float32)
    mode = window_mode.lower()

    if mode in ("velocity", "vel"):
        v = float(vref_mps)
        if v <= 0:
            raise ValueError("vref_mps must be > 0")
        t_arr = offsets_m / v
        # vel_search_half_width is now interpreted as full window width in seconds
        half_w = float(vel_search_half_width) / 2.0
        t0_s = (t_arr - half_w).astype(np.float32)
        t1_s = (t_arr + half_w).astype(np.float32)
        t0_s = np.maximum(t0_s, float(min_lag_s))

    elif mode in ("peak", "max"):
        half_w = float(peak_half_width_s)
        minlag = float(min_lag_s)
        pos_mask = lags >= minlag
        if not np.any(pos_mask):
            raise ValueError("No positive lags available for peak search")
        idx0 = np.where(pos_mask)[0][0]
        A = np.abs(ref_for_peak[:, idx0:])
        k = np.argmax(A, axis=1) + idx0
        tpk = lags[k]
        t0_s = (tpk - half_w).astype(np.float32)
        t1_s = (tpk + half_w).astype(np.float32)
        t0_s = np.maximum(t0_s, float(min_lag_s))

    elif mode in ("velpeak", "velocity_peak", "max_around_velocity"):
        v = float(vref_mps)
        if v <= 0:
            raise ValueError("vref_mps must be > 0")
        t_pred = offsets_m / v
        # half width used to define search window around predicted arrival (from vel_search_half_width, now in seconds as full width)
        half_vel_search = float(vel_search_half_width) / 2.0
        # half width used to define final window around detected peak (from peak_half_width_s)
        half_peak = float(peak_half_width_s)
        if half_peak <= 0:
            half_peak = 0.25
        for p in range(P):
            tmin = max(float(min_lag_s), float(t_pred[p] - half_vel_search))
            tmax = float(t_pred[p] + half_vel_search)
            mask = (lags >= tmin) & (lags <= tmax)
            if not np.any(mask):
                # fallback: use vel-based window centered on predicted arrival
                t0_s[p] = max(float(min_lag_s), float(t_pred[p] - half_vel_search))
                t1_s[p] = float(t_pred[p] + half_vel_search)
                continue
            # find lag of maximum absolute amplitude within the search window
            kk = np.argmax(np.abs(ref_for_peak[p, mask]))
            idxs = np.where(mask)[0]
            ipk = idxs[kk]
            tpk = float(lags[ipk])
            # define final window around the peak using half_peak
            t0_s[p] = max(float(min_lag_s), tpk - half_peak)
            t1_s[p] = tpk + half_peak

    else:
        raise ValueError("window_mode must be velocity|peak|velpeak")

    bad = t1_s <= t0_s
    if np.any(bad):
        t1_s[bad] = t0_s[bad] + float(max(0.1, peak_half_width_s))

    # write or update window group
    with h5py.File(ref_h5, "a") as h5:
        gwin = h5.create_group("window")
        gwin.create_dataset("t0_s", data=t0_s.astype(np.float32))
        gwin.create_dataset("t1_s", data=t1_s.astype(np.float32))
        gwin.attrs["window_mode"] = str(window_mode)
        gwin.attrs["vref_mps"] = float(vref_mps)


# ---------------- helper functions to load references and apply windows ----------------

def load_reference_and_windows(ref_h5: Path):
    """Load references HDF5 and return (lags, offsets_m, pairs, refs_dict, (t0_s,t1_s or (None,None)))."""
    if not ref_h5.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_h5}")
    with h5py.File(ref_h5, "r") as h5:
        lags = h5["lags"][...].astype(np.float32)
        offsets_m = h5["offsets_m"][...].astype(np.float32)
        pairs = h5["pairs_i_j"][...].astype(np.int32)
        refs = {k.decode(): h5["ref"][k][...].astype(np.float32) for k in h5["ref"]['components']}
        if "window" in h5:
            t0_s = h5["window"]["t0_s"][...].astype(np.float32)
            t1_s = h5["window"]["t1_s"][...].astype(np.float32)
        else:
            t0_s = None
            t1_s = None
    return lags, offsets_m, pairs, refs, (t0_s, t1_s)



# ---------------- existing plotting functions (slightly hardened) ----------------

def plot_hourly_images_for_pair(seg_files, coords, pairs, i, j, pidx, out_png: Path, nhours: int = None, maxlag_s: Optional[float] = None, ref_traces: Optional[Dict[str, np.ndarray]] = None, ref_lags: Optional[np.ndarray] = None, ref_pidx: Optional[int] = None):
    if nhours:
        seg_files = seg_files[:nhours]
    if len(seg_files) == 0:
        raise RuntimeError("No segment files found")
    TT_rows, RR_rows, ZZ_rows = [], [], []
    lags = None
    theta = azimuth_EN(coords, i, j)
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
    dt = float(lags[1] - lags[0])
    TT_im = butterworth_bandpass(torch.from_numpy(TT_im), dt, 2, 5).cpu()
    RR_im = butterworth_bandpass(torch.from_numpy(RR_im), dt, 2, 5).cpu()
    ZZ_im = butterworth_bandpass(torch.from_numpy(ZZ_im), dt, 2, 5).cpu()
    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True, sharey=True)
    extent = (float(lags[0]), float(lags[-1]), float(len(seg_files) - 0.5), float(-0.5))
    axs[0].imshow(TT_im, aspect="auto", extent=extent, interpolation="nearest")
    axs[0].set_title("TT")
    axs[0].set_xlabel("Lag (s)")
    axs[0].set_ylabel("Hour index")
    axs[1].imshow(RR_im, aspect="auto", extent=extent, interpolation="nearest")
    axs[1].set_title("RR")
    axs[1].set_xlabel("Lag (s)")
    axs[2].imshow(ZZ_im, aspect="auto", extent=extent, interpolation="nearest")
    axs[2].set_title("ZZ")
    axs[2].set_xlabel("Lag (s)")
    fig.suptitle(f"Hourly xcorr images for pair ({i},{j})")
    # Optionally overlay reference wiggle (centered vertically) if provided
    if ref_traces is not None and ref_lags is not None and ref_pidx is not None:
        try:
            comp_map = ["TT", "RR", "ZZ"]
            # image vertical extent: top (len(seg_files)-0.5) to -0.5
            y_top = float(len(seg_files) - 0.5)
            y_bottom = float(-0.5)
            mid_y = 0.5 * (y_top + y_bottom)
            vert_range = abs(y_top - y_bottom)
            wiggle_amp = 0.3 * vert_range
            # mask ref lags to current maxlag if specified
            if maxlag_s is not None:
                ref_mask = (ref_lags >= -float(maxlag_s)) & (ref_lags <= float(maxlag_s))
            else:
                ref_mask = slice(None)
            for ax, comp in zip(axs, comp_map):
                if comp not in ref_traces:
                    continue
                R = ref_traces[comp]
                if ref_pidx < 0 or ref_pidx >= R.shape[0]:
                    continue
                trace = R[ref_pidx, :].astype(np.float32)
                lags_ref = ref_lags[ref_mask]
                tr_use = trace[ref_mask]
                if tr_use.size == 0:
                    continue
                tr_norm = tr_use / (np.max(np.abs(tr_use)) + 1e-12)
                yvals = mid_y + tr_norm * wiggle_amp
                ax.plot(lags_ref, yvals, color="r", linewidth=1.0, alpha=0.9)
        except Exception as e:
            print(f"Warning: could not overlay reference wiggle: {e}")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ---------------- CLI / main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xcorr-dir", default="outputs/beam_xcorr", type=Path)
    ap.add_argument("--sta1", type=str, default="A002", help="station name as in meta.h5 (e.g., 1F.CC01)")
    ap.add_argument("--sta2", type=str, default="A021", help="station name as in meta.h5 (e.g., 1F.CC10)")
    ap.add_argument("--nhours", type=int, default=None, help="limit number of segment files for the hourly-pair plot")
    ap.add_argument("--maxlag", type=float, default=3, help="optional +/- maxlag seconds for plots")
    ap.add_argument("--max-files", type=int, default=0, help="limit hours for averaging and reference building (0 = all)")
    ap.add_argument("--build-reference", default=True,
                    action="store_true", help="build references.h5 from segment files")
    ap.add_argument("--build-windows", default=True,
                    action="store_true", help="compute per-pair windows from cached references")
    ap.add_argument("--ref-method", type=str, default="mean", choices=["mean", "median", "trim"], help="reference aggregation")
    ap.add_argument("--trim-pct", type=float, default=0.1, help="alpha-trim fraction per side (only for --ref-method trim)")
    ap.add_argument("--ref-comps", type=str, default="ZZ,RR,TT", help="comma-separated components to build/plot")
    ap.add_argument("--clip-pct", type=float, default=99.0, help="percentile for grayscale clipping (on |amplitude|)")
    ap.add_argument("--window-mode", type=str, default="velocity", choices=["velocity", "peak", "velpeak"], help="window selection method")
    ap.add_argument("--vref", type=float, default=400.0, help="reference velocity (m/s) for velocity-based picking")
    ap.add_argument("--vel-half-width", type=float, default=1, help="full window width in seconds around t=offset/vref for velocity windows (previously fractional)")
    ap.add_argument("--peak-comp", type=str, default="RR", help="component used to pick the peak (must be in refs)")
    ap.add_argument("--peak-half-width", type=float, default=0.25, help="half width (s) around peak for peak-based windows")
    ap.add_argument("--min-lag", type=float, default=0.05, help="minimum lag (s) for peak search")
    ap.add_argument("--overwrite", default=True,
                    action="store_true", help="overwrite cached references.h5")
    ap.add_argument("--pair-chunk", type=int, default=None, help="number of pairs per chunk for reference building (default: all)")
    ap.add_argument("--verbose", default=True,
                    action="store_true", help="print verbose progress and parameters")
    args = ap.parse_args()
    # Print a concise summary of parsed arguments when verbose
    if args.verbose:
        print("STEP04_build_reference_window_selection.py: starting with arguments:")
        print(f"  xcorr-dir   = {args.xcorr_dir}")
        print(f"  sta1,sta2   = {args.sta1},{args.sta2}")
        print(f"  build-ref   = {args.build_reference}  method={args.ref_method} comps={args.ref_comps} pair-chunk={args.pair_chunk}")
        print(f"  build-wins  = {args.build_windows}  window-mode={args.window_mode} vref={args.vref}")
        print(f"  maxlag      = {args.maxlag}  max-files={args.max_files}")
        print(f"  overwrite   = {args.overwrite}")

    meta_path = args.xcorr_dir / "meta.h5"
    if not meta_path.exists():
        alt = args.xcorr_dir.parent / "beam_xcorr" / "meta.h5"
        if alt.exists():
            meta_path = alt
        else:
            alt2 = Path("outputs/beam_xcorr") / "meta.h5"
            if alt2.exists():
                meta_path = alt2
            else:
                raise FileNotFoundError(f"meta.h5 not found in {args.xcorr_dir} or typical beam_xcorr locations")
    seg_dir = args.xcorr_dir / "segments"
    if not seg_dir.exists():
        alt_seg = args.xcorr_dir.parent / "beam_xcorr" / "segments"
        if alt_seg.exists():
            seg_dir = alt_seg
        else:
            alt_seg2 = Path("outputs/beam_xcorr") / "segments"
            if alt_seg2.exists():
                seg_dir = alt_seg2
            else:
                raise FileNotFoundError(f"segments/ not found in {args.xcorr_dir} or typical beam_xcorr locations")
    outdir = args.xcorr_dir / "xcorr_figures"
    outdir.mkdir(parents=True, exist_ok=True)
    stations, coords, pairs, sta_to_idx, pair_to_pidx, lags = load_meta(meta_path)
    # list HDF5 files and keep only those matching expected stamp
    seg_files, times =  xcorr_segment_files(seg_dir)

    # reference HDF5 path and components to use
    # incorporate the reference stacking method into the filename so different choices are saved separately
    ref_method_safe = str(args.ref_method).lower()
    if ref_method_safe == "trim":
        trim_str = str(float(args.trim_pct)).replace('.', 'p')
        ref_fname = f"references_{ref_method_safe}_pct{trim_str}.h5"
    else:
        ref_fname = f"references_{ref_method_safe}.h5"
    ref_h5 = args.xcorr_dir / ref_fname
    comps = [c.strip().upper() for c in str(args.ref_comps).split(",") if c.strip()]


    if args.verbose:
        print(f"Found {len(stations)} stations, {pairs.shape[0]} pairs, {len(seg_files)} segment files in {seg_dir}")

    # References + windows (ref_h5 and comps already defined above)
    refs, lags = None, None
    if args.build_reference:
        print(f"Step: build references -> {ref_h5}")
        build_references(seg_files=seg_files, coords_xy_m=coords, pairs=pairs, out_h5=ref_h5, comps=comps, ref_method=args.ref_method, trim_pct=float(args.trim_pct), max_files=int(args.max_files), overwrite=bool(args.overwrite), pair_chunk=args.pair_chunk)
        print(f"Built references and wrote: {ref_h5}")

    if args.build_windows:
        # compute windows from cached references (must exist)
        print(f"Step: compute windows (mode={args.window_mode}) and write into {ref_h5}")
        compute_windows(ref_h5=ref_h5, window_mode=args.window_mode, vref_mps=float(args.vref), vel_search_half_width=float(args.vel_half_width), peak_comp=args.peak_comp, peak_half_width_s=float(args.peak_half_width), min_lag_s=float(args.min_lag), overwrite=bool(args.overwrite))
        print(f"Windows computed and written into: {ref_h5}")

    if ref_h5.exists():
        lags, offsets_m, _pairs2, refs, (t0_s, t1_s) = load_reference_and_windows(ref_h5)
        order = np.argsort(offsets_m)
        offsets_m = offsets_m[order]
        refs_ord = {k: v[order, :] for k, v in refs.items() if k in comps}

        # If window data exist, order them; otherwise plot without windows
        if t0_s is None or t1_s is None:
            windows = None
            print("No window group found in references.h5 — plotting references without windows")
        else:
            t0_ord = t0_s[order]
            t1_ord = t1_s[order]
            windows = (t0_ord, t1_ord)

        out_png = outdir / f"reference_gather_{args.ref_method}.png"
        title = f"Reference gather, method={args.ref_method}, nhours={args.max_files}"
        _plot_reference_gather(refs_ord, lags, offsets_m, out_png, clip_pct=float(args.clip_pct), title=title, windows=windows)
        print(f"Wrote: {out_png}")

        # If windows exist, also plot windowed references (zeroed outside windows)
        if windows is not None:
            refs_win = apply_window_to_refs(refs, lags, t0_s, t1_s)
            refs_win_ord = {k: v[order, :] for k, v in refs_win.items() if k in comps}
            out_png2 = outdir / f"reference_gather_windowed_{args.ref_method}_{args.window_mode}.png"
            title2 = f"Reference gather windowed (method={args.ref_method}, mode={args.window_mode})"
            _plot_reference_gather(refs_win_ord, lags, offsets_m, out_png2, clip_pct=float(args.clip_pct), title=title2, windows=windows)
            print(f"Wrote: {out_png2}")


    if args.sta1 is not None and args.sta2 is not None:
        print(f"Step: hourly images requested for pair {args.sta1} - {args.sta2} (showing up to nhours={args.nhours})")
        i, j, pidx = get_pair_indices(sta_to_idx, pair_to_pidx, args.sta1, args.sta2)
        out_png = outdir / f"hourly_pair_{args.sta1.replace('.','_')}_{args.sta2.replace('.','_')}_{args.ref_method}.png"

        print(f"Plotting hourly images for pair {args.sta1} - {args.sta2}")
        try:
            if ref_h5.exists():
                # refs and lags were loaded above when ref_h5.exists()
                plot_hourly_images_for_pair(
                    seg_files=seg_files,
                    coords=coords,
                    pairs=pairs,
                    i=i,
                    j=j,
                    pidx=pidx,
                    out_png=out_png,
                    nhours=args.nhours,
                    maxlag_s=args.maxlag,
                    ref_traces=refs,
                    ref_lags=lags,
                    ref_pidx=pidx,
                )
            else:
                plot_hourly_images_for_pair(seg_files=seg_files, coords=coords, pairs=pairs, i=i, j=j, pidx=pidx, out_png=out_png, nhours=args.nhours, maxlag_s=args.maxlag)
            print(f"Wrote: {out_png}")
        except Exception as e:
            print(f"Warning: failed to create hourly image for pair {args.sta1}-{args.sta2}: {e}")


if __name__ == "__main__":
    main()
