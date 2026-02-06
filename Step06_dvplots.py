#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import h5py
from matplotlib import pyplot as plt
from typing import Optional

def _find_pair_index(stations: list[str], pairs: np.ndarray, sta_a: str, sta_b: str) -> Optional[int]:
    # Normalize station names
    sta_a = sta_a.strip()
    sta_b = sta_b.strip()
    try:
        i = stations.index(sta_a)
        j = stations.index(sta_b)
    except ValueError:
        return None
    # Pairs are stored as indices (i,j) with i<j
    ii, jj = (i, j) if i < j else (j, i)
    # Find matching row
    hits = np.where((pairs[:, 0] == ii) & (pairs[:, 1] == jj))[0]
    return int(hits[0]) if hits.size else None

def main() -> int:
    ap = argparse.ArgumentParser(description="Plot dv/v time series for a station pair from Step05 output.")
    ap.add_argument("--in-h5", default="outputs/STEP05_dvv_all_pairs/dv_stretching.h5",
                    type=Path, help="Path to Step05 HDF5 output")
    ap.add_argument("--station-a", default="A002",
                    type=str,  help="First station name (e.g., A001)")
    ap.add_argument("--station-b", default="A021",
                    type=str, help="Second station name (e.g., A002)")
    ap.add_argument("--series", type=str, default="all",
                    choices=["auto", "causal", "acausal", "sum", "all"],
                    help="Which dv/v series to plot. 'auto' selects any available (prefers sum). 'all' overlays all.")
    ap.add_argument("--min_cc", default=0.8, type=float,
                    help="Minimum cross-correlation value to consider")
    ap.add_argument("--out-png", default="outputs/Step06_dvv_plots",
                    type=Path, help="Output PNG path")
    ap.add_argument("--show", default=True,
                    action="store_true", help="Show the plot window")

    # Heatmap (time x dv/v for all pairs)
    ap.add_argument("--heatmap", action="store_true", default=True,
                    help="Also produce a 2D histogram heatmap (x=time, y=dv/v) aggregating all pairs per day.")
    ap.add_argument("--heatmap-bins", type=int, default=25,
                    help="Number of bins in dv/v for the heatmap (vertical resolution).")

    # Pair-wise histogram: x = station pair index (or label), y = dv/v histogram over time
    ap.add_argument("--pair-heatmap", action="store_true", default=True,
                    help="Produce a 2D histogram image where each column is the dv/v distribution over time for one station-pair.")
    ap.add_argument("--pair-heatmap-bins", type=int, default=100,
                    help="Number of dv/v bins for the pair-wise histogram (vertical resolution).")
    ap.add_argument("--pair-heatmap-log", action="store_true", default=False,
                    help="Use log color scale for pair-wise histogram image (adds 1 to counts before log).")

    # Global 1D histogram (all times × all pairs)
    ap.add_argument("--global-hist", action="store_true", default=True,
                    help="Also produce a global 1-D histogram of dv/v aggregating all times and all pairs.")
    ap.add_argument("--global-hist-bins", type=int, default=200,
                    help="Number of bins for the global 1-D histogram.")
    ap.add_argument("--global-min-cc", type=float, default=0.9,
                    help="Minimum cross-correlation threshold to apply for the global histogram. If omitted, uses --min_cc.")
    args = ap.parse_args()

    # Load HDF5
    with h5py.File(args.in_h5, "r") as h5:
        stations = [s.decode() for s in h5["stations"][...]]
        pairs = h5["pairs_i_j"][...].astype(np.int32)
        starttimes = [st.decode() for st in h5["starttimes"][...]]
        # Detect available dv/v datasets
        dv_names = [k for k in h5.keys() if k.startswith("dvv_")]
        cc_names = [k for k in h5.keys() if k.startswith("cc_")]

    # Pick series to plot (list or single)
    available = sorted([name.split("dvv_")[1] for name in dv_names])
    if args.series == "auto":
        # Prefer sum -> causal -> acausal
        for pref in ("sum", "causal", "acausal"):
            if pref in available:
                series = [pref]
                break
        else:
            series = [available[0]]
    elif args.series == "all":
        series = available
    else:
        if args.series not in available:
            raise SystemExit(f"Requested series '{args.series}' not available. Found: {available}")
        series = [args.series]

    # Choose a single effective series for the auxiliary/aggregate plots.
    # If user requested multiple (--series 'all'), prefer 'sum' when available else pick first available.
    if len(series) == 1:
        chosen_series = series[0]
    else:
        chosen_series = "sum" if "sum" in available else available[0]
    heat_series = pair_series = global_series = chosen_series

    # Find pair index
    pidx = _find_pair_index(stations, pairs, args.station_a, args.station_b)
    if pidx is None:
        raise SystemExit(f"Pair ({args.station_a},{args.station_b}) not found in pairs_i_j.")

    # Build time axis (parse ISO strings)
    # Use numpy datetime64 for plotting
    times = np.array(starttimes, dtype="datetime64[ns]")

    # Extract dv/v for selected pair across days
    dv_series = {}
    cc_series = {}
    with h5py.File(args.in_h5, "r") as h5:
        for s in series:
            dv_series[s] = h5[f"dvv_{s}"][..., pidx].astype(np.float32)  # shape (D,)
            cc_series[s] = h5[f"cc_{s}"][..., pidx].astype(np.float32)    # shape (D,)

    # If heatmap requested, read the full per-day/per-pair dvv matrix for the chosen series
    heatmap_data = None
    if args.heatmap:
        with h5py.File(args.in_h5, "r") as h5:
            ds_name = f"dvv_{heat_series}"
            if ds_name not in h5:
                raise SystemExit(f"Heatmap requested but dataset '{ds_name}' not found in {args.in_h5}. Available dvv datasets: {[k for k in h5.keys() if k.startswith('dvv_')]}")
            dv_all = h5[ds_name][...]  # (D,P)
            cc_ds = h5.get(f"cc_{heat_series}")
            cc_all = cc_ds[...] if cc_ds is not None else None
        # Build per-day histograms (x axis = day index, y bins = dv/v)
        D, P = dv_all.shape
        # mask using min_cc if cc_all available
        valid_mask = np.isfinite(dv_all)
        # if cc_all is not None:
        #     valid_mask &= (cc_all >= args.min_cc)
        # compute symmetric dv range from percentiles
        dv_vals_all = dv_all[valid_mask]
        if dv_vals_all.size == 0:
            vmin, vmax = -0.05, 0.05
        else:
            lo = np.percentile(dv_vals_all, 1.0)
            hi = np.percentile(dv_vals_all, 99.0)
            m = max(abs(lo), abs(hi))
            vmin, vmax = -m, m
        nbins = int(args.heatmap_bins)
        bin_edges = np.linspace(vmin, vmax, nbins + 1)
        hist = np.zeros((nbins, D), dtype=float)
        for di in range(D):
            row = dv_all[di, :]
            mask_row = np.isfinite(row)
            if cc_all is not None:
                mask_row &= (cc_all[di, :] >= args.min_cc)
            vals = row[mask_row]
            if vals.size:
                counts, _ = np.histogram(vals, bins=bin_edges)
                hist[:, di] = counts
        heatmap_data = dict(hist=hist, edges=bin_edges, vmin=vmin, vmax=vmax)

    # Pair-wise histogram across time (one histogram per pair)
    if args.pair_heatmap:
        sel_series = pair_series   # don't overwrite 'series' list used later
        with h5py.File(args.in_h5, "r") as h5:
            ds_name = f"dvv_{sel_series}"
            if ds_name not in h5:
                raise SystemExit(f"Pair-heatmap requested but dataset '{ds_name}' not found in {args.in_h5}. Available dvv datasets: {[k for k in h5.keys() if k.startswith('dvv_')]}")
            dv_all_pairs = h5[ds_name][...]  # (D,P)
            cc_ds = h5.get(f"cc_{sel_series}")
            cc_all_pairs = cc_ds[...] if cc_ds is not None else None
        D, P = dv_all_pairs.shape
        # Compute overall dv/v range (symmetric) from percentiles using valid values
        mask_valid = np.isfinite(dv_all_pairs)
        if cc_all_pairs is not None:
            mask_valid &= (cc_all_pairs >= args.min_cc)
        dv_vals = dv_all_pairs[mask_valid]
        if dv_vals.size == 0:
            vmin, vmax = -0.05, 0.05
        else:
            lo = np.percentile(dv_vals, 1.0)
            hi = np.percentile(dv_vals, 99.0)
            m = max(abs(lo), abs(hi))
            vmin, vmax = -m, m
        nbins = int(args.pair_heatmap_bins)
        bin_edges = np.linspace(vmin, vmax, nbins + 1)
        pair_hist = np.zeros((nbins, P), dtype=float)
        for p in range(P):
            col = dv_all_pairs[:, p]
            mask = np.isfinite(col)
            if cc_all_pairs is not None:
                mask &= (cc_all_pairs[:, p] >= args.min_cc)
            vals = col[mask]
            if vals.size:
                counts, _ = np.histogram(vals, bins=bin_edges)
                pair_hist[:, p] = counts

        # Plot pair-wise histogram: x = pair index (with sparse labels), y = dv/v
        fig_p, ax_p = plt.subplots(figsize=(12, 4))
        img = pair_hist  # (nbins, P)
        # extent: x from 0..P, y from vmin..vmax (in percent)
        im = ax_p.imshow(img, origin="lower", aspect="auto", cmap="magma",
                         extent=[0, P, vmin*100.0, vmax*100.0])
        # x tick labels: show a subset with station-pair names
        pair_labels = [f"{stations[i]}-{stations[j]}" for (i, j) in pairs]
        step = max(1, P // 20)
        xt_pos = np.arange(0.5, P, step)
        xt_lbls = [pair_labels[k] for k in range(0, P, step)]
        ax_p.set_xticks(xt_pos)
        ax_p.set_xticklabels(xt_lbls, rotation=90, fontsize=7)
        ax_p.set_xlabel("Station pair")
        ax_p.set_ylabel("dv/v (%)")
        ax_p.set_title(f"dv/v distribution over time per station-pair (series={sel_series})")
        cbar = fig_p.colorbar(im, ax=ax_p, label="counts")
        if args.pair_heatmap_log:
            # convert to log by taking log1p of counts -> recreate image for display (avoid modifying data)
            im.set_norm(plt.matplotlib.colors.LogNorm(vmin=max(1, img.min()), vmax=img.max()+1))
        out_png_p = args.out_png / f"dvv_pair_hist_{args.in_h5.stem}_{sel_series}.png"
        out_png_p.parent.mkdir(parents=True, exist_ok=True)
        fig_p.tight_layout()
        fig_p.savefig(out_png_p, dpi=200)
        print(f"Wrote pair-wise histogram {out_png_p}")
        if args.show:
            plt.show()
        else:
            plt.close(fig_p)

    # Global 1-D histogram across all times and pairs
    if args.global_hist:
        with h5py.File(args.in_h5, "r") as h5:
            ds_name = f"dvv_{global_series}"
            if ds_name not in h5:
                raise SystemExit(f"Global histogram requested but dataset '{ds_name}' not found in {args.in_h5}. Available dvv datasets: {[k for k in h5.keys() if k.startswith('dvv_')]}")
            dv_all = h5[ds_name][...]  # (D,P)
            cc_ds = h5.get(f"cc_{global_series}")
            cc_all = cc_ds[...] if cc_ds is not None else None
        # Mask invalid / low-cc values
        mask = np.isfinite(dv_all)
        # choose threshold: explicit global threshold if provided, else fallback to --min_cc
        min_cc_for_global = args.global_min_cc if args.global_min_cc is not None else args.min_cc
        if cc_all is not None and min_cc_for_global is not None:
            mask &= (cc_all >= float(min_cc_for_global))
        vals = dv_all[mask]
        if vals.size == 0:
            vals = np.array([0.0])
        # compute histogram
        nbins = int(args.global_hist_bins)
        # choose symmetric range from 1st/99th percentiles
        lo = np.percentile(vals, 1.0)
        hi = np.percentile(vals, 99.0)
        m = max(abs(lo), abs(hi), 1e-6)
        bin_edges = np.linspace(-m, m, nbins + 1)
        counts, edges = np.histogram(vals, bins=bin_edges)
        centers = 0.5 * (edges[:-1] + edges[1:])

        fig_g, ax_g = plt.subplots(figsize=(6, 3.5))
        ax_g.bar(centers * 100.0, counts, width=(edges[1]-edges[0]) * 100.0, align="center", color="C0", edgecolor="k", linewidth=0.2)
        ax_g.set_xlabel("dv/v (%)")
        ax_g.set_ylabel("counts")
        ax_g.set_title(f"Global dv/v histogram (all times × all pairs, series={global_series})")
        fig_g.tight_layout()
        out_png_g = args.out_png / f"dvv_global_hist_{args.in_h5.stem}_{global_series}.png"
        out_png_g.parent.mkdir(parents=True, exist_ok=True)
        fig_g.savefig(out_png_g, dpi=200)
        print(f"Wrote global histogram {out_png_g}")
        if args.show:
            plt.show()
        else:
            plt.close(fig_g)

    # Apply min_cc mask to pair plot series (unchanged behavior)
    for s in series:
        cc = cc_series[s]
        dv = dv_series[s]
        mask = cc < args.min_cc
        dv[mask] = np.nan
        dv_series[s] = dv
        kept = np.sum(~mask) / len(mask) * 100.0
        print(f"Series '{s}': kept {kept:.1f}% of dv/v values after cc >= {args.min_cc}")

    # Prepare plot
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for s in series:
        y = dv_series[s]
        ax.plot(times, y*100, marker="o", ms=3, lw=1.0, label=f"dvv {s}")

    #add x labels in hours after start of experiment under date
    #set xticks
    times_ticks = times[::max(1, len(times)//10)]
    hours_ticks = (times_ticks - times[0]) / np.timedelta64(1, 'h')
    ax.set_xticks(times_ticks)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels([f"{h:.1f}" for h in hours_ticks])
    ax2.set_xlabel("Hours after start")

    ax.grid(True, ls="--", alpha=0.4)
    ax.set_xlabel("Date")
    ax.set_ylabel("dv/v (%)")
    ax.set_title(f"dv/v for pair {args.station_a}–{args.station_b}")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    # Output file
    out_png = args.out_png / f"dvv_{args.in_h5.stem}_STA{args.station_a}-{args.station_b}_{args.series}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    print(f"Wrote {out_png}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    # Save heatmap if requested
    if args.heatmap and heatmap_data is not None:
        hist = heatmap_data["hist"]
        edges = heatmap_data["edges"]
        vmin = heatmap_data["vmin"] * 100.0
        vmax = heatmap_data["vmax"] * 100.0
        fig_h, ax_h = plt.subplots(figsize=(10, 4))
        # imshow expects (ny, nx) -> hist already (nbins, D)
        im = ax_h.imshow(hist, origin="lower", aspect="auto", cmap="plasma",
                         extent=[0, len(times), vmin, vmax])
        # x ticks: use a subset of dates
        step = max(1, len(times) // 10)
        xt_pos = np.arange(0.5, len(times), step)
        xt_labels = [str(t) for t in times[::step]]
        ax_h.set_xticks(xt_pos)
        ax_h.set_xticklabels(xt_labels, rotation=45, ha="right")
        ax_h.set_xlabel("Date")
        ax_h.set_ylabel("dv/v (%)")
        ax_h.set_title(f"dv/v distribution across all pairs (series={heat_series})")
        cbar = fig_h.colorbar(im, ax=ax_h, label="counts")
        out_png_h = args.out_png / f"dvv_heatmap_{args.in_h5.stem}_{heat_series}.png"
        out_png_h.parent.mkdir(parents=True, exist_ok=True)
        fig_h.tight_layout()
        fig_h.savefig(out_png_h, dpi=200)
        print(f"Wrote heatmap {out_png_h}")
        if args.show:
            plt.show()
        else:
            plt.close(fig_h)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
