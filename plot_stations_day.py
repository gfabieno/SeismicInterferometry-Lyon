# python
#!/usr/bin/env python3
"""
Plot stacked wiggles for two stations for one UTC day.

Usage example:
    python plot_two_stations_wiggles.py \
      --data-root data --network 1F --location 00 \
      --station A001 --station2 A002 --day 2018-09-15 \
      --channels DPZ,DPN,DPE --win-len 3600 --outdir outputs/figs
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from obspy import UTCDateTime
from utils import read_block_for_station


def load_station_day(data_root: str, network: str, location: str, station: str,
                     channels: str, starttime: str, endtime: str) -> tuple[list[str], np.ndarray, float, UTCDateTime]:
    """
    Read a full UTC day [day, day+1) for one station and return:
      comps: list of component letters (['Z','N','E'])
      data: ndarray (3, L) float32 in order comps
      fs: sampling rate (float)
      t0: UTCDateTime start of day
    """
    t0 = UTCDateTime(starttime)
    t1 = UTCDateTime(endtime)
    st = read_block_for_station(
        data_root=str(data_root),
        network=network,
        location=location,
        station=station,
        channels=channels,
        t0=t0,
        t1=t1,
    )
    if len(st) == 0:
        raise RuntimeError(f"No data read for station {station} on {starttime}")

    comp_order = [c[-1] for c in channels.split(",")]  # e.g. "DPZ" -> "Z"
    # Determine sampling rate from first available trace
    tr0 = None
    for ch in comp_order:
        tr = st.select(channel=f"*{ch}")
        if len(tr) > 0:
            tr0 = tr[0]
            break
    if tr0 is None:
        raise RuntimeError("No traces with requested components found")
    fs = tr0.stats.sampling_rate
    L = tr0.stats.npts

    # Build array (3, L)
    data = np.zeros((len(comp_order), L), dtype=np.float32)
    for i, ch in enumerate(comp_order):
        tr = st.select(channel=f"*{ch}")
        if len(tr) == 0:
            continue
        arr = tr[0].data.astype(np.float32)
        if arr.size >= L:
            data[i, :] = arr[:L]
        else:
            data[i, :arr.size] = arr
            # remainder stays zero (padding)
    return comp_order, data, fs, t0


def plot_wiggles_one_station(station: str, comp_order: list[str], data: np.ndarray,
                             fs: float, win_len: float, outdir: Path, day: str,
                             scale: float | None = None, cmap_positive: str = "black",
                             show: bool = False) -> None:
    """
    Create one figure with 3 stacked axes for Z,N,E. Each axis has wiggles stacked
    vertically, one row per window (earliest at top).
    """
    C, L = data.shape
    win_samp = int(round(win_len * fs))
    if win_samp < 1:
        raise ValueError("win_len produces zero samples")
    n_windows = int(np.ceil(L / win_samp))

    times_win = np.linspace(0.0, win_len, win_samp, endpoint=False) if win_samp > 0 else np.array([])

    # amplitude scaling: if not provided, set scale so wiggles occupy ~0.8 of vertical spacing
    if scale is None:
        # robust amplitude estimate per component
        max_abs = np.max(np.abs(data)) if data.size else 1.0
        scale = 0.9 * (1.0 / max_abs) if max_abs > 0 else 1.0

    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    comp_names = ["Z", "N", "E"]
    for ci in range(3):
        ax = axs[ci]
        comp_label = comp_order[ci] if ci < len(comp_order) else comp_names[ci]
        ax.set_ylabel(comp_label)
        # separation between wiggles (in plotted units)
        sep = 1.0  # base separation
        # compute offset array so earliest window appears at top
        offsets = np.arange(n_windows) * -sep
        for wi in range(n_windows):
            s0 = wi * win_samp
            s1 = min((wi + 1) * win_samp, L)
            seg = data[ci, s0:s1].astype(np.float32)
            # pad to win_samp
            if seg.size < win_samp:
                seg = np.pad(seg, (0, win_samp - seg.size), mode="constant")
            x = times_win
            y = offsets[wi] + seg * scale
            ax.plot(x, y, color="k", linewidth=0.6)
            # optionally fill positive lobes for readability
            ax.fill_between(x, offsets[wi], y, where=(y > offsets[wi]),
                            color=cmap_positive, linewidth=0, alpha=0.6)
        # aesthetics
        ax.set_ylim(offsets[-1] - sep * 0.5, offsets[0] + sep * 0.5)
        ax.set_yticks([])

    axs[-1].set_xlabel(f"Seconds since {day} UTC (window length {win_len:.1f} s)")
    fig.suptitle(f"Stacked wiggles for station {station} — day {day}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"wiggles_{station}_{day}.png"
    fig.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", type=Path)
    ap.add_argument("--network", default="1F", type=str)
    ap.add_argument("--location", default="00", type=str)
    ap.add_argument("--station", default="A001", type=str)
    ap.add_argument("--starttime", default="2018-09-15T00:00:00",
                    type=str, help="Start time (UTCDateTime format)")
    ap.add_argument("--endtime", default="2018-09-16T00:00:00",
                    type=str, help="End time (UTCDateTime format)")
    ap.add_argument("--channels", default="DPZ,DPN,DPE", type=str)
    ap.add_argument("--win-len", default=3600.0, type=float, help="Window length in seconds")
    ap.add_argument("--outdir", default=Path("outputs/figures"), type=Path)
    ap.add_argument("--show", action="store_true", help="If set, display the figure interactively")
    args = ap.parse_args()

    # first station
    comp1, data1, fs1, t0_1 = load_station_day(
        data_root=str(args.data_root),
        network=args.network,
        location=args.location,
        station=args.station,
        channels=args.channels,
        starttime=args.starttime,
        endtime=args.endtime,
    )
    print(f"Loaded data for station {args.station} from {args.starttime} to {args.endtime} UTC")

    # Create output directory for figures
    figs_dir = args.outdir
    figs_dir.mkdir(parents=True, exist_ok=True)

    plot_wiggles_one_station(args.station, comp1, data1, fs1, args.win_len, figs_dir, args.starttime, show=bool(args.show))


if __name__ == "__main__":
    main()