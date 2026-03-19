#!/usr/bin/env python3
"""
Daily averaging + plotting from hourly beamforming .h5 files.

Assumptions
- Each hourly file produced by the new Step03 contains datasets under group "beamforming":
    /beamforming/pwr_Z, /beamforming/pwr_R, /beamforming/pwr_T, /beamforming/freq
  (but the script still accepts older files where these datasets live at the root).
- Filenames from the new Step03 use the prefix "bf_xc_YYYYMMDDTHHMMSSZ.h5" (old prefix "beam_" is still supported).
- A meta file (meta.h5) exists with:
    /meta/slowness_s_per_m  (S,2) on a square grid

Outputs
- dayavg_YYYYMMDD.h5 (optional, --write-h5)
- dayavg_YYYYMMDD.png figures (2x2: Z, R, T, Total)
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import re
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from beamforming import plot_beamforming

# filepath-specific changes: accept both old "beam_" and new "bf_xc_" prefixes
STAMP_RE = re.compile(r"(?:beam|bf_xc)_(\d{8})T(\d{6})Z\.h5$")


def parse_day_from_filename(p: Path) -> dt.date | None:
    m = STAMP_RE.search(p.name)
    if not m:
        return None
    ymd = m.group(1)
    return dt.date(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]))


def list_segment_files(segments_dir: Path) -> list[Path]:
    # accept any .h5 and filter via the stamp regex (supports both beam_ and bf_xc_)
    files = sorted(segments_dir.glob("*.h5"))
    return [p for p in files if parse_day_from_filename(p) is not None]


def load_slowness_from_meta(meta_h5: Path) -> np.ndarray:
    with h5py.File(meta_h5, "r") as h5:
        slow = h5["meta/slowness_s_per_m"][...].astype(np.float32)  # (S,2)
    return slow


def infer_grid_shape_from_slowness(slow: np.ndarray) -> tuple[int, float, float, float, float]:
    """
    Returns: (nslow, sx_min, sx_max, sy_min, sy_max)
    Assumes square grid.
    """
    S = slow.shape[0]
    nslow = int(round(np.sqrt(S)))
    if nslow * nslow != S:
        raise ValueError(f"Slowness grid is not square: S={S}, sqrt(S)~{np.sqrt(S)}")
    sx = slow[:, 0]
    sy = slow[:, 1]
    return nslow, float(sx.min()), float(sx.max()), float(sy.min()), float(sy.max())


def average_files_for_day(files: list[Path]) -> dict[str, np.ndarray]:
    """
    Returns dict with mean pwr_Z/R/T (K,S), freq (K,), nfiles.
    Uses simple mean across available files (no weighting).

    This version reads datasets either from the "beamforming" group (new Step03)
    or directly from the root (older layout).
    """
    sumZ = sumR = sumT = None
    freq_ref = None
    n = 0

    for fp in files:
        try:
            with h5py.File(fp, "r") as h5:
                # prefer new layout with "beamforming" group
                if "beamforming" in h5:
                    bf = h5["beamforming"]
                else:
                    bf = h5
                pZ = bf["pwr_Z"][...].astype(np.float32)
                pR = bf["pwr_R"][...].astype(np.float32)
                pT = bf["pwr_T"][...].astype(np.float32)
                freq = bf["freq"][...].astype(np.float32)

            if freq_ref is None:
                freq_ref = freq
            else:
                if freq.shape != freq_ref.shape or np.max(np.abs(freq - freq_ref)) > 1e-6:
                    raise ValueError(f"Frequency mismatch in {fp.name}")

            if sumZ is None:
                sumZ = pZ.copy()
                sumR = pR.copy()
                sumT = pT.copy()
            else:
                if pZ.shape != sumZ.shape:
                    raise ValueError(f"Shape mismatch in {fp.name}: {pZ.shape} vs {sumZ.shape}")
                sumZ += pZ
                sumR += pR
                sumT += pT
        except Exception as e:
            print(f"[WARN] Skipping file {fp.name} due to error: {e}")
            continue

        n += 1

    if n == 0:
        raise ValueError("No files to average.")

    return {
        "pwr_Z": sumZ / n,
        "pwr_R": sumR / n,
        "pwr_T": sumT / n,
        "freq": freq_ref,
        "nfiles": np.array(n, dtype=np.int32),
    }


def save_daily_plot(out_png: Path,
                    day: dt.date,
                    meanZ: np.ndarray, meanR: np.ndarray, meanT: np.ndarray,
                    nslow: int,
                    extent,
                    nfiles: int):
    """
    mean*: (K,S) -> plot frequency-averaged maps.
    """
    Zmap = meanZ.mean(axis=0).reshape(nslow, nslow)
    Rmap = meanR.mean(axis=0).reshape(nslow, nslow)
    Tmap = meanT.mean(axis=0).reshape(nslow, nslow)
    Total = (meanZ + meanR + meanT).mean(axis=0).reshape(nslow, nslow)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
    plot_beamforming(extent, Zmap,  title="Beamforming Z", ax=axs[0, 0])
    plot_beamforming(extent, Rmap,  title="Beamforming R", ax=axs[0, 1])
    plot_beamforming(extent, Tmap,  title="Beamforming T", ax=axs[1, 0])
    plot_beamforming(extent, Total, title="Beamforming Total Power", ax=axs[1, 1])

    fig.suptitle(
        f"Daily average beamforming (UTC {day.strftime('%Y-%m-%d')}), nseg={nfiles}",
        fontsize=16)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)



def write_daily_h5(out_h5: Path, day: dt.date, daily: dict[str, np.ndarray]):
    with h5py.File(out_h5, "w") as h5:
        h5.attrs["day_utc"] = day.strftime("%Y-%m-%d")
        h5.create_dataset("pwr_Z", data=daily["pwr_Z"], compression="gzip")
        h5.create_dataset("pwr_R", data=daily["pwr_R"], compression="gzip")
        h5.create_dataset("pwr_T", data=daily["pwr_T"], compression="gzip")
        h5.create_dataset("freq",  data=daily["freq"],  compression="gzip")
        h5.create_dataset("nfiles", data=daily["nfiles"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir",
                    default="outputs/Step03_beamforming_xcorr", type=Path)
    ap.add_argument("--start-day", default=None, type=str,
                    help="Optional start day YYYY-MM-DD (UTC)")
    ap.add_argument("--end-day", default=None, type=str,
                    help="Optional end day YYYY-MM-DD (UTC), inclusive")
    ap.add_argument("--write-h5", action="store_true",
                    help="Also write daily averaged .h5 files")
    args = ap.parse_args()

    segments_dir = args.outdir / "segments"
    print(f"Segments read from {segments_dir}")
    figures_dir = args.outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output figures will be saved to {figures_dir}")
    meta_path = args.outdir / "meta.h5"

    slow = load_slowness_from_meta(meta_path)
    nslow, sx_min, sx_max, sy_min, sy_max = infer_grid_shape_from_slowness(slow)
    extent = [sx_min, sx_max, sy_min, sy_max]

    files = list_segment_files(segments_dir)
    if not files:
        raise SystemExit(f"No bf_xc_/beam_ *.h5 files found in {segments_dir}")

    # group by day
    by_day: dict[dt.date, list[Path]] = {}
    for fp in files:
        day = parse_day_from_filename(fp)
        if day is None:
            continue
        by_day.setdefault(day, []).append(fp)

    days = sorted(by_day.keys())

    if args.start_day:
        d0 = dt.date.fromisoformat(args.start_day)
        days = [d for d in days if d >= d0]
    if args.end_day:
        d1 = dt.date.fromisoformat(args.end_day)
        days = [d for d in days if d <= d1]

    if not days:
        raise SystemExit("No days to process after filtering.")

    for day in days:
        day_files = sorted(by_day[day])
        daily = average_files_for_day(day_files)

        out_png = figures_dir / f"beam_dayavg_{day.strftime('%Y%m%d')}.png"
        save_daily_plot(
            out_png=out_png,
            day=day,
            meanZ=daily["pwr_Z"],
            meanR=daily["pwr_R"],
            meanT=daily["pwr_T"],
            nslow=nslow,
            extent=extent,
            nfiles=int(daily["nfiles"][()]),
        )

        if args.write_h5:
            out_h5 = figures_dir / f"beam_dayavg_{day.strftime('%Y%m%d')}.h5"
            write_daily_h5(out_h5, day, daily)

        print(f"[OK] {day}  nseg={int(daily['nfiles'][()])}  -> {out_png.name}")


if __name__ == "__main__":
    main()
