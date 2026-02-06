# Step04_cross_correlation.py
import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from obspy import UTCDateTime, read_inventory

from utils import (
    stations_from_folders,
    inv_station_xy,
    start_segment_reader_thread,
)
from cross_correlation import cross_correlation, cc_lags, pairs_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", type=Path)
    ap.add_argument("--network", default="1F", type=str)
    ap.add_argument("--location", default="00", type=str)
    # IMPORTANT: order here defines x components order in the numpy block
    ap.add_argument("--channels", default="DPZ,DPN,DPE", type=str,
                    help="Channel order in the block (default: DPZ,DPN,DPE -> [Z,N,E])")

    ap.add_argument("--outdir", default="outputs/xcorr", type=Path)
    ap.add_argument("--starttime", default="2018-09-15T00:00:00", type=str)
    ap.add_argument("--endtime", default="2018-10-03T00:00:00", type=str)

    ap.add_argument("--seg-len", default=3600.0, type=float)
    ap.add_argument("--win-len", default=60.0, type=float, help="Window length (s) used in xcorr stacking")
    ap.add_argument("--fs", default=250.0, type=float)

    ap.add_argument("--fmin", default=1.0, type=float)
    ap.add_argument("--fmax", default=20.0, type=float)

    ap.add_argument("--onebit", default=False, type=bool)
    ap.add_argument("--whiten", default=True, type=bool)
    ap.add_argument("--eps", default=1e-2, type=float)
    ap.add_argument("--p-chunk", default=32, type=int)

    ap.add_argument("--maxlag", default=3, type=float,
                    help="If set, store only +/- maxlag seconds around lag0")
    ap.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_segments_dir = args.outdir / "segments"
    out_segments_dir.mkdir(parents=True, exist_ok=True)



    # ---- inventory / stations / coords ----
    inv = read_inventory(str(args.data_root / "metadata" / "*"), format="STATIONXML")
    stations = stations_from_folders(args.data_root, args.network)
    if len(stations) == 0:
        raise RuntimeError(f"No stations found under {args.data_root}/waveforms/{args.network}")

    coords = inv_station_xy(inv, outfile=args.outdir / "station_map_local_xy.png")
    device = torch.device(args.device)
    coords_t = torch.from_numpy(coords).to(device=device, dtype=torch.float32)

    # ---- time ----
    t0 = UTCDateTime(args.starttime)
    t1 = UTCDateTime(args.endtime)
    if t1 <= t0:
        raise ValueError("endtime must be greater than starttime")

    seg_len = float(args.seg_len)
    fs = float(args.fs)
    Lseg = int(round(seg_len * fs))
    wp = int(round(float(args.win_len) * fs))

    # ---- pair list (on device) ----
    N = len(stations)
    pairs = pairs_all(N, device=device)
    P = int(N*(N-1)//2)
    print(f"Stations: {N}, pairs (i<j): {P}")
    print(f"Processing: {t0} -> {t1} (hours={(t1-t0)/3600:.1f}), seg_len={seg_len}s, wp={wp} samples")

    # ---- lags for storage ----
    # cross_correlation returns length wp with lag0 centered at wp//2
    lags = cc_lags(wp=wp, fs=fs, device="cpu").numpy().astype(np.float32)

    # optional maxlag slice indices (robust even/odd)
    if args.maxlag is not None:
        half = int(round(float(args.maxlag) * fs))
        center = wp // 2
        i0 = max(0, center - half)
        i1 = min(wp, center + half + 1)  # symmetric odd-length around center
        lags = lags[i0:i1]
    else:
        i0, i1 = 0, wp

    # ---- meta (written once) ----
    meta_path = args.outdir / "meta.h5"
    if (not meta_path.exists()) or args.overwrite:
        with h5py.File(meta_path, "w") as h5m:
            meta = h5m.create_group("meta")
            meta.create_dataset("stations", data=np.array(stations, dtype="S"))
            meta.create_dataset("coords_xy_m", data=coords.astype(np.float32))
            meta.create_dataset("pairs_i_j", data=pairs.detach().cpu().numpy().astype(np.int32))
            meta.create_dataset("lags", data=lags)
            meta.attrs["network"] = args.network
            meta.attrs["location"] = args.location
            meta.attrs["channels_order"] = args.channels
            meta.attrs["fs_hz"] = fs
            meta.attrs["seg_len_sec"] = seg_len
            meta.attrs["win_len_sec"] = float(args.win_len)
            meta.attrs["fmin_hz"] = float(args.fmin)
            meta.attrs["fmax_hz"] = float(args.fmax)
            meta.attrs["onebit"] = bool(args.onebit)
            meta.attrs["whiten"] = bool(args.whiten)
            meta.attrs["eps"] = float(args.eps)

    # ---- per-hour output path ----
    def out_path_fn(seg_start: UTCDateTime) -> Path:
        stamp = seg_start.strftime("%Y%m%dT%H%M%SZ")
        return out_segments_dir / f"xcorr_{stamp}.h5"

    # ---- reader thread from utils (reads mseed -> (N,C,Lseg) float32 block) ----
    q, stop_event, th = start_segment_reader_thread(
        t0=t0,
        t1=t1,
        seg_len_s=seg_len,
        Lseg=Lseg,
        stations=stations,
        data_root=str(args.data_root),
        network=args.network,
        location=args.location,
        channels=args.channels,   # order defines block component order
        prefetch=2,
        overwrite=args.overwrite,
        out_path_fn=out_path_fn,
    )

    # ---- consume / compute / write ----
    n_done = 0
    t_global0 = time.perf_counter()
    nseg = int(np.ceil((t1 - t0) / seg_len))
    try:
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item

            segi, seg_start, seg_end, block_np, read_time = item
            if block_np is None:
                continue

            out_path = out_path_fn(seg_start)

            # to torch
            x = torch.from_numpy(block_np).to(device=device, dtype=torch.float32, non_blocking=True)

            header = f"Xcorr segment {segi + 1}/{nseg}. read={read_time:.1f}s"
            out = cross_correlation(
                x=x,
                coords=coords_t,
                fs=fs,
                wp=wp,
                detrend=True,
                flim=[args.fmin, args.fmax],
                p_chunk=int(args.p_chunk),
                pairs=pairs,
                header=header,
                onebit=bool(args.onebit),
                whiten=bool(args.whiten),
                eps=float(args.eps),
            )

            corr = out["corr"]  # (P,C,C,wp), lag0 centered at wp//2 (per our utils convention)

            # slice to +/- maxlag if requested
            corr = corr[..., i0:i1].contiguous()

            # write
            corr_cpu = corr.detach().to("cpu").numpy().astype(np.float32)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(out_path, "w") as h5:
                g = h5.create_group("xcorr")
                g.create_dataset("corr", data=corr_cpu, compression="gzip", compression_opts=4)
                g.create_dataset("lags", data=lags)
                g.create_dataset("pairs_i_j", data=pairs.detach().cpu().numpy().astype(np.int32))

                g.attrs["starttime"] = str(seg_start)
                g.attrs["endtime"] = str(seg_end)
                g.attrs["fs_hz"] = fs
                g.attrs["wp_samples"] = int(wp)
                g.attrs["fmin_hz"] = float(args.fmin)
                g.attrs["fmax_hz"] = float(args.fmax)
                g.attrs["onebit"] = bool(args.onebit)
                g.attrs["whiten"] = bool(args.whiten)
                g.attrs["read_time_s"] = float(read_time)
                g.attrs["channels_order"] = args.channels
                if args.maxlag is not None:
                    g.attrs["maxlag_s"] = float(args.maxlag)

    finally:
        stop_event.set()
        th.join(timeout=2.0)

    dt_s = time.perf_counter() - t_global0
    print(f"Done. Segments written: {n_done}. Elapsed: {dt_s/60:.1f} minutes.")


if __name__ == "__main__":
    main()
