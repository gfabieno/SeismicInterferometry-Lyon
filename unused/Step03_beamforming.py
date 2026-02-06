import h5py
import numpy as np
import torch
import argparse
from obspy import read_inventory, UTCDateTime
from utils import stations_from_folders, read_block_for_station, start_segment_reader_thread
from beamforming import plane_wave_beamforming, plot_beamforming
from utils import inv_station_xy
from pathlib import Path
import matplotlib.pyplot as plt
import threading
import queue
import time


def make_slowness_grid(sx_min, sx_max, sy_min, sy_max, ds, unit="s/km"):
    sx = np.arange(sx_min, sx_max + ds, ds)
    sy = np.arange(sy_min, sy_max + ds, ds)
    SX, SY = np.meshgrid(sx, sy, indexing="ij")
    grid = np.stack([SX.ravel(), SY.ravel()], axis=1).astype(np.float32)
    if unit == "s/km":
        grid /= 1000.0  # -> s/m
    return grid


def main_process(
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu",
):

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", type=Path,
                    help="Root folder, e.g. waveforms/")
    ap.add_argument("--network", default="1F", type=str,
                    help="Network folder under data-root (default: 1F)")
    ap.add_argument("--location", default="00", type=str,
                    help="Location code (default: 00)")
    ap.add_argument("--channels", default="DPZ,DPN,DPE", type=str,
                    help="Channels to read (default: DPZ,DPN,DPE)")
    ap.add_argument("--outdir", default="outputs/beamforming", type=Path)
    ap.add_argument("--starttime", default="2018-09-15T00:00:00",
                    type=str, help="Start time (UTCDateTime format)")
    ap.add_argument("--endtime", default="2018-10-03T00:00:00",
                    type=str, help="End time (UTCDateTime format)")
    ap.add_argument("--out_h5", default="processing/beamforming_results.h5",
                    type=str, help="Output HDF5 file for beamforming results")
    ap.add_argument("--seg_len", default=3600, type=float,
                    help="Segment length over which beamforming is averaged"
                         "in seconds (default: 3600s = 1 hour)")
    ap.add_argument("--win_len", default=10, type=float,
                    help="Window length for FFT beamforming in seconds (default: 10s)")
    ap.add_argument("--fmin", default=2.0, type=float,
                    help="Minimum frequency for bandpass (Hz)")
    ap.add_argument("--fmax", default=5.0, type=float,
                    help="Maximum frequency for bandpass (Hz)")
    ap.add_argument("--fs", default=250.0, type=float,
                    help="Sampling rate in Hz")
    ap.add_argument("--slow_max", default=5e-3, type=float,
                    help="Maximum slowness (s/m)")
    ap.add_argument("--ds", default=6e-5, type=float,
                    help="Slowness grid step (s/m)")
    ap.add_argument("--s_chunk", default=32, type=int,
                    help="Slowness chunk size for beamforming (default: 512)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output files")

    args = ap.parse_args()

    print("Starting beamforming process with arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    out_segments_dir = args.outdir / "segments"
    out_segments_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output segments will be saved to {out_segments_dir}")
    out_figures_dir = args.outdir / "figures"
    out_figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output figures will be saved to {out_figures_dir}")

    # -----------------------------
    # Inventory, stations, coords
    # -----------------------------
    inv = read_inventory(str(args.data_root /"metadata" / "*"), format="STATIONXML")
    stations = stations_from_folders(args.data_root, args.network)
    coords = inv_station_xy(inv,
                            outfile=out_figures_dir / "station_map_local_xy.png")
    coords_t = torch.from_numpy(coords).to(device=device, dtype=torch.float32)
    print(f"Found {len(stations)} stations for network {args.network}")


    # -----------------------------
    # Time handling (contiguous)
    # -----------------------------
    t0 = UTCDateTime(args.starttime)
    t1 = UTCDateTime(args.endtime)
    if t1 <= t0:
        raise ValueError("endtime must be greater than starttime")
    print(f"Processing from {t0} to {t1}, total {(t1 - t0)/3600:.1f} hours")


    seg_len = float(args.seg_len)
    wp = int(round(args.win_len * args.fs))
    Lseg = int(round(seg_len * args.fs))

    # -----------------------------
    # Slowness grid
    # -----------------------------
    slowness_grid = make_slowness_grid(
        sx_min=-args.slow_max, sx_max=args.slow_max,
        sy_min=-args.slow_max, sy_max=args.slow_max,
        ds=args.ds,
        unit="s/m",
    )
    slow_t = torch.from_numpy(slowness_grid).to(device=device)
    nslow = int(np.sqrt(slowness_grid.shape[0]))
    print(f"Slowness grid: {slowness_grid.shape[0]} points,")

    required_memory = (len(stations) * 3 * Lseg * 4 * 2 +  # input block
                       args.s_chunk * Lseg * 4 * 3 +  # FFT chunk
                       slowness_grid.shape[0] * 4 * 3 # output arrays
                       ) / (1024**3)
    print(f"Estimated memory requirement: {required_memory:.2f} GB")
    print(f"Reduce s_chunk if out-of-memory errors occur.")


    # -----------------------------
    # Metadata (written once)
    # -----------------------------
    meta_path = args.outdir / "meta.h5"
    with h5py.File(meta_path, "w") as h5m:
        meta = h5m.create_group("meta")
        meta.create_dataset("stations", data=np.array(stations, dtype="S"))
        meta.create_dataset("coords_xy_m", data=coords)
        meta.create_dataset("slowness_s_per_m", data=slowness_grid)
        meta.attrs["network"] = args.network
        meta.attrs["fs_hz"] = args.fs
        meta.attrs["seg_len_sec"] = args.seg_len
        meta.attrs["win_len_sec"] = args.win_len
        meta.attrs["fmin_hz"] = args.fmin
        meta.attrs["fmax_hz"] = args.fmax
        meta.attrs["ds_s_per_m"] = args.ds
        meta.attrs["slow_max"] = args.slow_max

    # -----------------------------
    # Main loop: contiguous segments
    # -----------------------------
    seg_start = t0
    segi = 0
    nseg = int(np.ceil((t1 - t0) / seg_len))

    q, stop_event, t_reader = start_segment_reader_thread(
        t0=t0,
        t1=t1,
        seg_len_s=seg_len,
        Lseg=Lseg,
        stations=stations,
        data_root=str(args.data_root),
        network=args.network,
        location=args.location,
        channels=args.channels,
        prefetch=2,
        overwrite=args.overwrite,
        out_path_fn=lambda seg_start: out_segments_dir / f"beam_{seg_start.strftime('%Y%m%dT%H%M%SZ')}.h5",
    )
    print("Started data reader thread.")


    try:
        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            segi, seg_start, seg_end, block, read_time = item
            stamp = seg_start.strftime("%Y%m%dT%H%M%SZ")
            out_path = out_segments_dir / f"beam_{stamp}.h5"

            # Skip segments already on disk (producer sent block=None)
            if block is None:
                print("Skipping existing segment, already processed:", stamp)
                continue

            header = f"Beamforming segment {segi + 1}/{nseg}. read={read_time:.1f}s"
            x_t = torch.from_numpy(block).to(device=device, non_blocking=True)
            res = plane_wave_beamforming(
                x=x_t,
                coords=coords_t,
                fs=args.fs,
                slow=slow_t,
                wp=wp,
                detrend=True,
                flim=[args.fmin, args.fmax],
                s_chunk=args.s_chunk,
                header=header,
            )

            # Immediately move outputs to CPU, then free GPU tensors
            pwr_Z = res["pwr_Z"].detach().cpu().numpy()
            pwr_R = res["pwr_R"].detach().cpu().numpy()
            pwr_T = res["pwr_T"].detach().cpu().numpy()
            freq = res["freq"].detach().cpu().numpy()

            # --- Write one file per segment ---
            with h5py.File(out_path, "w") as h5:
                h5.attrs["starttime"] = str(seg_start)
                h5.attrs["endtime"] = str(seg_end)
                h5.attrs["network"] = args.network
                h5.attrs["fs_hz"] = args.fs
                h5.attrs["seg_len_sec"] = args.seg_len
                h5.attrs["win_len_sec"] = args.win_len
                h5.attrs["fmin_hz"] = args.fmin
                h5.attrs["fmax_hz"] = args.fmax

                h5.create_dataset("pwr_Z", data=pwr_Z, compression="gzip")
                h5.create_dataset("pwr_R", data=pwr_R, compression="gzip")
                h5.create_dataset("pwr_T", data=pwr_T, compression="gzip")
                h5.create_dataset("freq", data=freq, compression="gzip")

            # --- Plot (CPU) ---
            slim = [-args.slow_max, args.slow_max, -args.slow_max, args.slow_max]
            pwr_total = (pwr_Z + pwr_R + pwr_T).mean(axis=0)  # (S,)
            score = pwr_total.reshape(nslow, nslow)

            fig, axs = plt.subplots(2, 2, figsize=(10, 10), sharex=True)
            plot_beamforming(slim, pwr_Z.mean(axis=0).reshape(nslow, nslow),
                             title="Beamforming Z", ax=axs[0, 0])
            plot_beamforming(slim, pwr_R.mean(axis=0).reshape(nslow, nslow),
                             title="Beamforming R", ax=axs[0, 1])
            plot_beamforming(slim, pwr_T.mean(axis=0).reshape(nslow, nslow),
                             title="Beamforming T", ax=axs[1, 0])
            plot_beamforming(slim, score,
                             title="Beamforming Total Power", ax=axs[1, 1])

            plt.suptitle("Beamforming of segment starting at " + stamp, fontsize=16)
            plt.tight_layout()
            fig_path = out_figures_dir / f"beam_{stamp}.png"
            plt.savefig(fig_path, dpi=300)
            plt.close(fig)

    finally:
        stop_event.set()
        t_reader.join(timeout=2.0)

if __name__ == "__main__":
    main_process()