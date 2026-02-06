# python
import h5py
import numpy as np
import torch
import argparse
from obspy import read_inventory, UTCDateTime
from utils import stations_from_folders, read_block_for_station, start_segment_reader_thread, window_qc_mask_station
from beamforming import plane_wave_beamforming, plot_beamforming, make_slowness_grid
from utils import inv_station_xy
from cross_correlation import cross_correlation, cc_lags, pairs_all, butterworth_bandpass
from pathlib import Path
import matplotlib.pyplot as plt


def main_process():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data", type=Path,
                    help="Root folder, e.g. waveforms/")
    ap.add_argument("--network", default="1F", type=str,
                    help="Network folder under data-root (default: 1F)")
    ap.add_argument("--location", default="00", type=str,
                    help="Location code (default: 00)")
    ap.add_argument("--channels", default="DPZ,DPN,DPE", type=str,
                    help="Channels to read (default: DPZ,DPN,DPE)")
    ap.add_argument("--outdir", default="outputs", type=Path)
    ap.add_argument("--starttime", default="2018-09-15T00:00:00",
                    type=str, help="Start time (UTCDateTime format)")
    ap.add_argument("--endtime", default="2018-10-03T00:00:00",
                    type=str, help="End time (UTCDateTime format)")

    # beamforming params
    ap.add_argument("--seg_len", default=3600, type=float,
                    help="Segment length over which beamforming is averaged in seconds")
    ap.add_argument("--win_len", default=20, type=float,
                    help="Window length for FFT beamforming in seconds")
    ap.add_argument("--fmin", default=2.0, type=float,
                    help="Minimum frequency for bandpass (Hz)")
    ap.add_argument("--fmax", default=5.0, type=float,
                    help="Maximum frequency for bandpass (Hz)")
    ap.add_argument("--fs", default=250.0, type=float,
                    help="Sampling rate in Hz")
    ap.add_argument("--slow_max", default=5e-3, type=float,
                    help="Maximum slowness (s/m)")
    ap.add_argument("--ds", default=8e-5, type=float,
                    help="Slowness grid step (s/m)")
    ap.add_argument("--s_chunk", default=32, type=int,
                    help="Slowness chunk size for beamforming")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing output files")

    # cross-correlation params
    ap.add_argument("--cc-win-len", default=60.0, type=float,
                    help="Window length (s) used in xcorr stacking")
    ap.add_argument("--p-chunk", default=32, type=int)
    ap.add_argument("--fmin_cc", default=1.0, type=float)
    ap.add_argument("--fmax_cc", default=20.0, type=float)
    ap.add_argument("--onebit", default=False, type=bool)
    ap.add_argument("--whiten", default=True, type=bool)
    ap.add_argument("--eps", default=1e-2, type=float)
    ap.add_argument("--maxlag", default=5.0, type=float,
                    help="If set, store only +/- maxlag seconds around lag0")
    ap.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")

    #QC parameters
    ap.add_argument("--qc-zero-frac-max", default=0.01, type=float)
    ap.add_argument("--qc-rms-z-max", default=10.0, type=float)
    ap.add_argument("--qc-spike-ratio-max", default=50.0, type=float)
    ap.add_argument("--qc-use-spectral", default=False, type=bool)
    ap.add_argument("--qc-band-energy-min", default=0.05, type=float)
    ap.add_argument("--qc-line-ratio-max", default=20.0, type=float)
    args = ap.parse_args()

    device = torch.device(args.device)

    print("Starting beamforming+xcor process with arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    outdir = args.outdir / "beam_xcorr"
    outdir.mkdir(parents=True, exist_ok=True)
    out_segments_dir = outdir / "segments"
    out_segments_dir.mkdir(parents=True, exist_ok=True)
    out_figures_dir = outdir / "figures" / "beamforming"
    out_figures_dir.mkdir(parents=True, exist_ok=True)

    # inventory / stations / coords
    inv = read_inventory(str(args.data_root / "metadata" / "*"), format="STATIONXML")
    stations = stations_from_folders(args.data_root, args.network)
    if len(stations) == 0:
        raise RuntimeError(f"No stations found under {args.data_root}/waveforms/{args.network}")
    coords = inv_station_xy(inv, outfile=outdir / "figures" / "station_map_local_xy.png")
    coords_t = torch.from_numpy(coords).to(device=device, dtype=torch.float32)
    N = len(stations)
    pairs = pairs_all(N, device=device)

    # time handling
    t0 = UTCDateTime(args.starttime)
    t1 = UTCDateTime(args.endtime)
    if t1 <= t0:
        raise ValueError("endtime must be greater than starttime")
    seg_len = float(args.seg_len)
    fs = float(args.fs)
    Lseg = int(round(seg_len * fs))

    # beamforming and xcorr window sizes in samples
    wp_beam = int(round(args.win_len * fs))
    wp_cc = int(round(args.cc_win_len * fs))

    # lags for storage (cpu)
    lags = cc_lags(wp=wp_cc, fs=fs, device="cpu").numpy().astype(np.float32)
    if args.maxlag is not None:
        half = int(round(float(args.maxlag) * fs))
        center = wp_cc // 2
        i0 = max(0, center - half)
        i1 = min(wp_cc, center + half + 1)
        lags = lags[i0:i1]
    else:
        i0, i1 = 0, wp_cc

    # slowness grid
    slow_t, _, _ = make_slowness_grid(args.slow_max, args.slow_max,
                                             args.ds, device=device)
    nslow = int(np.sqrt(slow_t.shape[0]))

    # meta (written once) - include both beamforming and xcorr meta
    meta_path = outdir / "meta.h5"
    if (not meta_path.exists()) or args.overwrite:
        with h5py.File(meta_path, "w") as h5m:
            meta = h5m.create_group("meta")
            meta.create_dataset("stations", data=np.array(stations, dtype="S"))
            meta.create_dataset("coords_xy_m", data=coords.astype(np.float32))
            meta.create_dataset("slowness_s_per_m", data=slow_t.cpu())
            meta.create_dataset("pairs_i_j", data=pairs.detach().cpu().numpy().astype(np.int32))
            meta.create_dataset("lags", data=lags)
            meta.attrs["network"] = args.network
            meta.attrs["fs_hz"] = fs
            meta.attrs["seg_len_sec"] = args.seg_len
            meta.attrs["win_len_beam_sec"] = args.win_len
            meta.attrs["win_len_xcorr_sec"] = args.cc_win_len
            meta.attrs["fmin_beam_hz"] = args.fmin
            meta.attrs["fmax_beam_hz"] = args.fmax
            meta.attrs["fmin_xcorr_hz"] = args.fmin_cc
            meta.attrs["fmax_xcorr_hz"] = args.fmax_cc
            meta.attrs["ds_s_per_m"] = args.ds
            meta.attrs["slow_max"] = args.slow_max
            #qc params
            meta.attrs["qc_zero_frac_max"] = args.qc_zero_frac_max
            meta.attrs["qc_rms_z_max"] = args.qc_rms_z_max
            meta.attrs["qc_spike_ratio_max"] = args.qc_spike_ratio_max
            meta.attrs["qc_use_spectral"] = args.qc_use_spectral
            meta.attrs["qc_band_energy_min"] = args.qc_band_energy_min
            meta.attrs["qc_line_ratio_max"] = args.qc_line_ratio_max

    # QC function
    qc_fun = lambda xw: window_qc_mask_station(
        xw, fs,
        flim=[args.fmin, args.fmax],
        detrend=True,
        zero_frac_max=args.qc_zero_frac_max,
        rms_z_max=args.qc_rms_z_max,
        spike_ratio_max=args.qc_spike_ratio_max,
        use_spectral=args.qc_use_spectral,
        band_energy_min=args.qc_band_energy_min,
        line_ratio_max=args.qc_line_ratio_max,
        eps=1e-8,
        verbose=True,
    )

    # start reader thread
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
        out_path_fn=lambda seg_start: out_segments_dir / f"bf_xc_{seg_start.strftime('%Y%m%dT%H%M%SZ')}.h5",
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
            out_path = out_segments_dir / f"bf_xc_{stamp}.h5"

            # producer may indicate skip (block is None)
            if block is None:
                print("Skipping existing segment, already processed:", stamp)
                continue

            print(f"Processing segment {segi + 1}/{nseg}. read={read_time:.1f}s")

            # to torch on device
            x_t = torch.from_numpy(block).to(device=device, dtype=torch.float32, non_blocking=True)

            # Beamforming
            res = plane_wave_beamforming(
                x=x_t,
                coords=coords_t,
                fs=fs,
                slow=slow_t,
                wp=wp_beam,
                detrend=True,
                flim=[args.fmin, args.fmax],
                s_chunk=args.s_chunk,
                header="Beamforming: ",
                qc_fun=qc_fun
            )
            pwr_Z = res["pwr_Z"].detach().cpu().numpy()
            pwr_R = res["pwr_R"].detach().cpu().numpy()
            pwr_T = res["pwr_T"].detach().cpu().numpy()
            freq = res["freq"].detach().cpu().numpy()

            # Cross-correlation
            out = cross_correlation(
                x=x_t,
                coords=coords_t,
                fs=fs,
                wp=wp_cc,
                detrend=True,
                flim=[args.fmin_cc, args.fmax_cc],
                p_chunk=int(args.p_chunk),
                pairs=pairs,
                header="Xcorr: ",
                onebit=bool(args.onebit),
                whiten=bool(args.whiten),
                eps=float(args.eps),
                qc_fun=qc_fun
            )
            corr = out["corr"]  # (P,C,C,wp_cc)
            corr = corr[..., i0:i1].contiguous()
            corr_cpu = corr.detach().to("cpu").numpy().astype(np.float32)

            # write both results into single HDF5 per segment
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(out_path, "w") as h5:
                # common attrs
                h5.attrs["starttime"] = str(seg_start)
                h5.attrs["endtime"] = str(seg_end)
                h5.attrs["network"] = args.network
                h5.attrs["fs_hz"] = fs
                h5.attrs["seg_len_sec"] = args.seg_len
                h5.attrs["read_time_s"] = float(read_time)

                # beamforming group
                g_bf = h5.create_group("beamforming")
                g_bf.create_dataset("pwr_Z", data=pwr_Z, compression="gzip")
                g_bf.create_dataset("pwr_R", data=pwr_R, compression="gzip")
                g_bf.create_dataset("pwr_T", data=pwr_T, compression="gzip")
                g_bf.create_dataset("freq", data=freq, compression="gzip")
                g_bf.attrs["fmin_hz"] = args.fmin
                g_bf.attrs["fmax_hz"] = args.fmax
                g_bf.attrs["win_len_sec"] = args.win_len

                # xcorr group
                g_xc = h5.create_group("xcorr")
                g_xc.create_dataset("corr", data=corr_cpu, compression="gzip", compression_opts=4)
                g_xc.create_dataset("lags", data=lags)
                g_xc.create_dataset("pairs_i_j", data=pairs.detach().cpu().numpy().astype(np.int32))
                g_xc.attrs["fmin_hz"] = args.fmin_cc
                g_xc.attrs["fmax_hz"] = args.fmax_cc
                g_xc.attrs["wp_samples"] = int(wp_cc)
                if args.maxlag is not None:
                    g_xc.attrs["maxlag_s"] = float(args.maxlag)

            # optional plotting (beamforming)
            slim = [-args.slow_max, args.slow_max, -args.slow_max, args.slow_max]
            pwr_total = (pwr_Z + pwr_R + pwr_T).mean(axis=0)
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
            plt.suptitle("Segment " + stamp, fontsize=16)
            plt.tight_layout()
            fig_path = out_figures_dir / f"bf_xc_{stamp}.png"
            plt.savefig(fig_path, dpi=300)
            plt.close(fig)

    finally:
        stop_event.set()
        t_reader.join(timeout=2.0)


if __name__ == "__main__":
    main_process()

