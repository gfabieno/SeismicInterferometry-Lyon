#!/usr/bin/env python3
"""Step05: dv/v for all station pairs.

This script computes dv/v time series for *all* station pairs by stretching
cross-correlations relative to a reference correlation.

It is designed to reuse your existing pipeline outputs from
`Step03_beamforming_xcorr.py`, which writes per-segment files named:

  outputs/.../segments/bf_xc_YYYYMMDDTHHMMSSZ.h5

Expected structure inside each file:
  /xcorr/corr      float32, shape (P, C, C, T)
  /xcorr/lags      float32, shape (T,) (optional; can also come from meta.h5)
  /xcorr/pairs_i_j int32,   shape (P, 2)

The script:
  1) Loads all segments within a date range.
  2) Builds a reference correlation per pair (mean or median across days).
  3) Selects a component (ZZ/NN/EE or any Cij).
  4) Estimates dv/v per day and pair using `dv_stretching.stretching_dvv_torch_vec`.
  5) Writes outputs to CSV and (optionally) HDF5.

Notes / conventions
-------------------
- dv/v is returned as a fractional change (e.g. 0.001 = +0.1%).
- The stretching search uses epsilons, where dv/v = -epsilon.
- Lag windowing for stretching can be specified in seconds with `--tmin-s/--tmax-s`.

Example
-------
python Step05_dvv_all_pairs.py \
  --in-dir outputs/beam_xcorr \
  --start-day 2018-09-15 --end-day 2018-09-20 \
  --component ZZ \
  --tmin-s 1.0 --tmax-s 3.0 \
  --eps-min -0.01 --eps-max 0.01 --eps-n 201 \
  --out-csv outputs/dvv/dvv_all_pairs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from dv_stretching import stretching_dvv_torch_vec
from utils import compute_theta, load_meta, xcorr_segment_files, extract_components_from_corr
from obspy import UTCDateTime
from cross_correlation import butterworth_bandpass

def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--in-dir", default=Path(
        "outputs/Step03_beamforming_xcorr"),
                    type=Path,
                    help="Directory containing segments/ and (optionally) meta.h5",
                    )
    ap.add_argument("--ref-h5", default=Path(
        "outputs/Step03_beamforming_xcorr/references_mean.h5"),
                    type=Path, help="Path to an existing reference HDF5 file created by STEP04",
                    )
    ap.add_argument("--out-h5", default=Path("outputs/STEP05_dvv_all_pairs/dv_stretching_all.h5"),
                    type=Path,  help="Optional HDF5 output")
    ap.add_argument( "--component",  default="TT",
        type=str,
        help="Component pair to stretch (e.g. ZZ, NN, EE, NE, or '0,0' indices)",
    )
    ap.add_argument("--starttime", default=None,#"2018-09-15T00:00:00",
                    type=str, help="Start time (UTCDateTime format)")
    ap.add_argument("--endtime", default=None,#"2018-09-17T00:00:00",
                    type=str, help="End time (UTCDateTime format)")

    ap.add_argument("--eps-min", type=float, default=-0.05)
    ap.add_argument("--eps-max", type=float, default=0.05)
    ap.add_argument("--eps-n", type=int, default=601)
    ap.add_argument( "--window-side",type=str, default="all",
        choices=["causal", "acausal", "sum", "all"],
        help="Which window(s) to use from reference: "
             "'causal', 'acausal', 'sum' (concatenate acausal+causal), "
             "or 'all' to compute all three estimates",)
    ap.add_argument("--apply-window", default=True,
                    action="store_true",
                    help="If set, apply per-pair windows from reference HDF5")
    ap.add_argument("--fmin", type=float, default=2)
    ap.add_argument("--fmax", type=float, default=5)
    ap.add_argument("--device",type=str,
                    #default="cuda" if torch.cuda.is_available() else "cpu")
                    default="cuda:1" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pair-chunk", type=int, default=32,
        help="Number of pairs to process at once (limits memory). If omitted, uses all pairs.",
    )
    ap.add_argument("--overwrite", action="store_true", default=False,
        help="If set, overwrite existing checkpointed chunks in --out-h5. Otherwise skip completed chunks.",
    )

    args = ap.parse_args()

    print("\n=== Step05 dv/v all pairs ===")
    print("Parameters:")
    print(f"  in_dir      : {args.in_dir}")
    print(f"  start_day   : {args.starttime}")
    print(f"  end_day     : {args.endtime}")
    print(f"  component   : {args.component}")
    print(f"  eps range   : [{args.eps_min}, {args.eps_max}] (n={args.eps_n})")
    print(f"  window side : {args.window_side}")
    print(f"  fmin/fmax   : {args.fmin} / {args.fmax} Hz")
    print(f"  apply win   : {args.apply_window}")
    print("  stretch win : per-pair windows read from --ref-h5 (window/t0_s, window/t1_s)")
    print(f"  device      : {args.device}")
    print(f"  out_h5      : {args.out_h5 if args.out_h5 is not None else None}")
    print(f"  pair_chunk : {args.pair_chunk}")
    print(f"  overwrite   : {args.overwrite}")
    print("============================\n")

    segments_dir = args.in_dir / "segments"
    meta_path = args.in_dir / "meta.h5"
    t0 = UTCDateTime(args.starttime) if args.starttime is not None else None
    t1 = UTCDateTime(args.endtime) if args.endtime is not None else None

    # Load coords for optional ZNE->ZTR rotation
    stations, coords, pairs, sta_to_idx, pair_to_pidx, lags = load_meta(meta_path)
    dt = abs(float(lags[1] - lags[0]))
    T = int(lags.size)
    # make sure first T//2 are negative lags, then zero, then positive
    if not (lags[T//2] == 0.0 and np.all(lags[:T//2] < 0.0) and np.all(lags[T//2+1:] > 0.0)):
        raise ValueError("Lags must be centered at zero with negative lags first")
    theta = compute_theta(coords, pairs)
    theta = torch.from_numpy(theta).to(args.device)

    # Discover available segment files first (use attrs for dates)
    files, segtimes = xcorr_segment_files(segments_dir, tmin=t0, tmax=t1)
    starttimes = [st for st, et in segtimes]

    if not files:
        raise SystemExit(
            f"No segment files with starttime/endtime attrs found in {segments_dir}. "
            "Expected files written by Step03_beamforming_xcorr.py."
        )

    print(f"Available times: {min(starttimes)} .. {max(starttimes)}")
    print(f"Selected files: {len(files)}")
    print(f"First file: {files[0].name}")
    print(f"Last file : {files[-1].name}\n")

    # First pass: determine P and pairs from first file
    P = int(pairs.shape[0])
    print(f"Detected number of pairs P={P}")

    # Resolve pair chunk size (None -> all pairs)
    pair_chunk = P if (args.pair_chunk is None) else int(args.pair_chunk)
    if pair_chunk < 1:
        raise SystemExit("--pair-chunk must be a positive integer or omitted")
    if pair_chunk > P:
        pair_chunk = P
    print(f"Effective pair_chunk={pair_chunk}")

    dev = torch.device(args.device)

    ref_h5_path = Path(args.ref_h5)
    print(f"Loading reference from: {ref_h5_path}")
    with h5py.File(ref_h5_path, 'r') as h5:
        comp_up = args.component.strip().upper()
        ref_np =  h5['ref'][comp_up][...].astype(np.float32)
        if args.apply_window:
            wmin = torch.from_numpy(h5['window']['t0_s'][...] / dt).to(dtype=torch.int32, device=dev)
            wmax = torch.from_numpy(h5['window']['t1_s'][...] / dt).to(dtype=torch.int32, device=dev)
        else:
            wmin, wmax = None, None

    P_ref = int(ref_np.shape[0])
    if P_ref != P:
        raise ValueError("Reference has different number of pairs than current data")

    # Convert to torch tensor on device
    ref = torch.from_numpy(ref_np).to(device=dev, dtype=torch.float32)
    print(f"Loaded reference with shape {ref_np.shape}")

    # Prepare days list and now compute dv/v in pair-chunks (checkpointed)
    epsilons = torch.linspace(float(args.eps_min), float(args.eps_max), int(args.eps_n), device=dev)
    print(f"Stretching epsilons: E={epsilons.numel()} from {float(epsilons[0])} to {float(epsilons[-1])}\n")

    D = len(files)
    dv_types = ['causal', 'acausal', 'sum'] if args.window_side == 'all' else [args.window_side]
    comp_u = [args.component.strip().upper()]

    # Ensure out dir exists and open checkpoint HDF5
    args.out_h5.parent.mkdir(parents=True, exist_ok=True)
    print(f"Using checkpoint HDF5: {args.out_h5} (overwrite={args.overwrite})")
    h5_out = h5py.File(args.out_h5, "a")
    # create metadata/attrs on first creation
    if "pairs_i_j" not in h5_out or "stations" not in h5_out or "coords" not in h5_out:
        h5_out.attrs["component"] = args.component
        h5_out.attrs["ref_h5"] = str(ref_h5_path)
        h5_out.attrs["eps_min"] = args.eps_min
        h5_out.attrs["eps_max"] = args.eps_max
        h5_out.attrs["eps_n"] = args.eps_n
        h5_out.attrs["window_side"] = args.window_side
        h5_out.attrs["apply_window"] = int(args.apply_window)
        if "starttimes" not in h5_out:
            h5_out.create_dataset("starttimes", data=np.array(starttimes, dtype="S"))
        if "pairs_i_j" not in h5_out:
            h5_out.create_dataset("pairs_i_j", data=pairs)
        if "stations" not in h5_out:
            h5_out.create_dataset("stations", data=np.array(stations, dtype="S"))
        if "coords" not in h5_out:
            h5_out.create_dataset("coords", data=coords)
        h5_out.flush()

    # Ensure dvv/cc datasets exist with shape (D,P) and NaN fill
    for dv_type in dv_types:
        ds_dvv = f"dvv_{dv_type}"
        ds_cc = f"cc_{dv_type}"
        if ds_dvv not in h5_out:
            h5_out.create_dataset(ds_dvv, shape=(D, P), dtype="f4", fillvalue=np.nan)
        if ds_cc not in h5_out:
            h5_out.create_dataset(ds_cc, shape=(D, P), dtype="f4", fillvalue=np.nan)

    print(f"Computing dv/v in pair-chunks (chunk={pair_chunk}) ...")
    n_chunks = (P + pair_chunk - 1) // pair_chunk
    for p0 in tqdm(range(0, P, pair_chunk), total=n_chunks, desc="Streching pairs", unit=" chunks", dynamic_ncols=True):
        p1 = min(P, p0 + pair_chunk)

        # Process each day/file for this chunk; checkpoint per-file (row) so reruns can resume per-day
        for di, fp in enumerate(files):
            # If not overwrite, check whether this day's values for this chunk are already computed
            skip_day = False
            if not args.overwrite:
                skip_day = True
                for dv_type in dv_types:
                    ds = h5_out[f"dvv_{dv_type}"][di, p0:p1]
                    if not np.isfinite(ds[...]).all():
                        skip_day = False
                        break
            if skip_day:
                # already done for this day and this chunk
                continue

            # read segment file only when needed
            with h5py.File(fp, "r") as h5:
                corr_np = h5["xcorr/corr"][p0:p1, ...]  # (pc,C,C,T)
                corr = torch.from_numpy(corr_np).to(device=dev, dtype=torch.float32)
            comp_traces = extract_components_from_corr(corr, theta[p0:p1], comp_u)

            for comp in comp_traces:
                tracesc = comp_traces[comp]
                ref_segmentc = ref[p0:p1, :]
                if args.fmin is not None or args.fmax is not None:
                    tracesc = butterworth_bandpass(
                        tracesc,
                        dt=dt,
                        fmin=args.fmin,
                        fmax=args.fmax,
                        order=4,
                    )
                    ref_segmentc = butterworth_bandpass(
                        ref_segmentc,
                        dt=dt,
                        fmin=args.fmin,
                        fmax=args.fmax,
                        order=4,
                    )
                for side in dv_types:
                    if side == 'causal':
                        traces = tracesc[:, T//2:]  # (pc, T_causal)
                        ref_segment = ref_segmentc[:, T//2:]  # (pc, T_causal)
                    elif side == 'acausal':
                        traces = torch.flip(tracesc[:, :T//2+1], dims=[1])
                        ref_segment = torch.flip(ref_segmentc[:, :T//2+1], dims=[1])
                    elif side == 'sum':
                        traces_causal = tracesc[:, T//2:2*int(T//2)+1]
                        traces_acausal = torch.flip(tracesc[:, :T//2+1], dims=[1])
                        traces = traces_causal + traces_acausal
                        ref_causal = ref_segmentc[:, T//2:2*int(T//2)+1]
                        ref_acausal = torch.flip(ref_segmentc[:, :T//2+1], dims=[1])
                        ref_segment = ref_causal + ref_acausal
                    else:
                        raise ValueError(f"Unknown window_side: {side}")

                    # perform stretching dv/v estimation
                    with torch.no_grad():
                        dvv_pc, cc_pc = stretching_dvv_torch_vec(
                            ref=ref_segment,
                            cur=traces,
                            epsilons=epsilons,
                            tmin=wmin[p0:p1] if wmin is not None else 0,
                            tmax=wmax[p0:p1] if wmax is not None else None,
                        )  # (pc,), (pc,)

                    # write into checkpoint file per-day (row) for this chunk
                    h5_out[f"dvv_{side}"][di, p0:p1] = dvv_pc.detach().cpu().numpy().astype(np.float32)
                    h5_out[f"cc_{side}"][di, p0:p1] = cc_pc.detach().cpu().numpy().astype(np.float32)

                    # free GPU memory between day computations
                    if dev.type == 'cuda':
                        torch.cuda.empty_cache()

            # flush after finishing this day's work for the current chunk so checkpoint is durable
            h5_out.flush()

    # close checkpoint file
    h5_out.close()

    print(f"Checkpointed results in {args.out_h5}")


if __name__ == "__main__":
    raise SystemExit(main())
