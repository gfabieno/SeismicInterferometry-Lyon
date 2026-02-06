from __future__ import annotations

import datetime as dt
import os

from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
from matplotlib import pyplot as plt
from obspy import read, UTCDateTime, Stream
from obspy.core.inventory import Inventory
from pygeodesy import LocalCartesian, EcefKarney
import queue, threading, time
from cross_correlation import butterworth_bandpass
import re
import h5py
import torch

def inv_get_coords(inv: Inventory, net: str, sta: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Get the latitude, longitude, and elevation of a station from an inventory.
    """
    try:
        inv_sel = inv.select(network=net, station=sta)
        if not inv_sel.networks or not inv_sel.networks[0].stations:
            return (None, None, None)
        s = inv_sel.networks[0].stations[0]
        return (s.latitude, s.longitude, s.elevation)
    except Exception:
        return (None, None, None)


def discover_mseed_files(root: Path, network: str = "*",
                         station: str = "*", year: str = "*") -> List[Path]:
    """
    Fast discovery for:
      root/waveforms/network/station/year/*.mseed
    """
    net_dir = root / "waveforms"

    if not net_dir.exists():
        return []
    return sorted(net_dir.glob(f"{network}/{station}/{year}/*.mseed"))


def stations_from_folders(root: Path, network: str = "1F") -> List[str]:
    """
    Returns stations found as folder names under root/waveforms/<network>/.
    Example: ['A001', 'A002', ...]
    """
    net_dir = root / "waveforms" / network
    if not net_dir.exists():
        return []
    return sorted([p.name for p in net_dir.iterdir() if p.is_dir()])


def mseed_path(out_root: str, network:str, location: str, station: str,  channels: str, day: dt.date) -> str:
    chans = channels.replace(",", "-")
    year_dir = os.path.join(out_root, "waveforms", network, station, f"{day.year:04d}")
    return os.path.join(year_dir, f"{network}.{station}.{location}.{chans}.{day.strftime('%Y%m%d')}.mseed")


def inv_station_xy(
    inventory,
    outfile: Path = None,
    annotate: bool = False,
    ref_latlon=None
) -> np.ndarray:
    """
    Plot station locations in a local Cartesian ENU frame (meters)
    using PyGeodesy only (NO PROJ, NO Cartopy).
    """

    names = []
    lats = []
    lons = []

    for net in inventory:
        for sta in net.stations:
            if sta.latitude is None or sta.longitude is None:
                continue
            names.append(f"{net.code}.{sta.code}")
            lats.append(float(sta.latitude))
            lons.append(float(sta.longitude))

    if not lats:
        raise ValueError("No station coordinates found")

    lats = np.asarray(lats)
    lons = np.asarray(lons)

    if ref_latlon is None:
        lat0 = lats.mean()
        lon0 = lons.mean()
    else:
        lat0, lon0 = ref_latlon

    lc = LocalCartesian(lat0, lon0, 0.0, ecef=EcefKarney())

    xs = np.zeros(len(lats))
    ys = np.zeros(len(lats))

    for i, (la, lo) in enumerate(zip(lats, lons)):
        r = lc.forward(la, lo, 0.0)
        xs[i] = r.x if hasattr(r, "x") else r[0]
        ys[i] = r.y if hasattr(r, "y") else r[1]

    if outfile is not None:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(xs, ys, s=25)

        if annotate:
            for name, x, y in zip(names, xs, ys):
                ax.text(x, y, name, fontsize=7)

        ax.set_aspect("equal")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title("Station locations (local Cartesian, PyGeodesy)")
        ax.grid(True)

        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(outfile, dpi=200)
        plt.close(fig)

    return np.column_stack([xs, ys])


def iter_days(t0: UTCDateTime, t1: UTCDateTime):
    """
    Yield datetime.date objects from t0.date() to t1.date(), inclusive.
    """
    d0 = dt.date(t0.year, t0.month, t0.day)
    d1 = dt.date(t1.year, t1.month, t1.day)
    d = d0
    one = dt.timedelta(days=1)
    while d <= d1:
        yield d
        d += one


def read_block_for_station(
    data_root: str,
    network: str,
    location: str,
    station: str,
    channels:str,
    t0: UTCDateTime,
    t1: UTCDateTime,
) -> Stream:
    """
    Read a time block [t0, t1] for one station using deterministic daily paths.
    """
    st = Stream()

    for day in iter_days(t0, t1):
        f = mseed_path(data_root, network, location, station, channels, day)
        try:
            st += read(f, starttime=t0, endtime=t1, format="MSEED")
        except Exception:
            # file missing or unreadable → just skip
            print(f"Warning: could not read file {f} for station {station}")
            pass

    if len(st) == 0:
        return st

    st.merge(method=1, fill_value="interpolate")
    st.trim(t0, t1, pad=True, fill_value=0.0)
    return st


def start_segment_reader_thread(
    t0: UTCDateTime,
    t1: UTCDateTime,
    seg_len_s: float,
    Lseg: int,
    stations: list[str],
    data_root: str,
    network: str,
    location: str,
    channels: str,  # e.g. "DPZ,DPN,DPE"
    prefetch: int,
    overwrite: bool,
    out_path_fn,  # out_path_fn(seg_start) -> Path
):
    """
    Producer thread that reads 3C blocks and pushes them to a queue.

    Queue items:
      (segi, seg_start, seg_end, block, read_time)
      - block shape: (Nsta, 3, Lseg)
      - block is None if segment is skipped (already on disk)
      - None is sentinel (done)
      - Exception object may be pushed on error
    """
    q = queue.Queue(maxsize=prefetch)
    stop_event = threading.Event()
    comp_order = channels.split(",")
    # throw an error for now if it is not Z, N, E as other functions expect that
    if len(comp_order) == 3 and (
            "Z" not in comp_order[0] or
            "N" not in comp_order[1] or
            "E" not in comp_order[2]):
        raise ValueError("component order should be Z, N, E")

    def _run():
        try:
            seg_start = t0
            segi = 0

            while (seg_start + seg_len_s <= t1 + 1e-6) and (not stop_event.is_set()):
                segi += 1
                seg_end = seg_start + seg_len_s

                out_path = out_path_fn(seg_start)
                if out_path is None:
                    paths = []
                elif isinstance(out_path, (list, tuple, set)):
                    paths = [Path(p) for p in out_path]
                else:
                    paths = [Path(out_path)]

                if paths and all(p.exists() for p in paths) and (not overwrite):
                    q.put((segi, seg_start, seg_end, None, 0.0), block=True)
                    seg_start += seg_len_s
                    continue

                block = np.zeros((len(stations), len(comp_order), Lseg), dtype=np.float32)
                t_read0 = time.perf_counter()

                for j, sta in enumerate(stations):
                    if stop_event.is_set():
                        break

                    st = read_block_for_station(
                        data_root=data_root,
                        network=network,
                        location=location,
                        station=sta,
                        channels=channels,
                        t0=seg_start,
                        t1=seg_end,
                    )
                    if len(st) == 0:
                        continue

                    # Fill [Z,N,E] (or whatever comp_order specifies)
                    for k, ch in enumerate(comp_order):
                        tr = st.select(channel=f"*{ch}")
                        if len(tr) == 0:
                            continue
                        data = tr[0].data.astype(np.float32)

                        if len(data) != Lseg:
                            data = np.pad(data[:Lseg], (0, max(0, Lseg - len(data))), mode="constant")

                        block[j, k, :] = data

                read_time = time.perf_counter() - t_read0
                q.put((segi, seg_start, seg_end, block, read_time), block=True)
                seg_start += seg_len_s

        except Exception as e:
            q.put(e, block=True)
        finally:
            q.put(None, block=True)

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return q, stop_event, th


def robust_ref_stack(X: np.ndarray, method: str = "mean", trim_pct: float = 0.1) -> np.ndarray:
    """Aggregate reference across segments (utility for STEP04).

    X shape: (S, P, T)
    method: 'mean'|'median'|'trim'
    trim_pct: fraction removed at both ends for 'trim'
    Returns: (P,T) float32
    """
    if X.ndim != 3:
        raise ValueError(f"Expected X (S,P,T), got {X.shape}")
    method_l = method.lower()
    if method_l == "mean":
        return X.mean(axis=0).astype(np.float32)
    if method_l == "median":
        return np.median(X, axis=0).astype(np.float32)
    if method_l in ("trim", "alpha", "alpha-trim"):
        r = float(trim_pct)
        S = X.shape[0]
        if S < 3:
            return X.mean(axis=0).astype(np.float32)
        lo = int(np.floor(r * S))
        hi = int(np.ceil((1.0 - r) * S))
        if hi <= lo + 1:
            return X.mean(axis=0).astype(np.float32)
        Xs = np.sort(X, axis=0)
        return Xs[lo:hi].mean(axis=0).astype(np.float32)
    raise ValueError("Unknown reference method")

# allow torch or numpy arrays
def extract_components_from_corr(corr: np.ndarray | torch.Tensor,
                                 theta: np.ndarray | torch.Tensor,
                                 comps: list[str]) -> Dict[str, np.ndarray | torch.Tensor]:
    """Extract requested components from corr (P,3,3,T) in one pass.

    Returns dict mapping comp->(P,T). Supported comps: 'ZZ','RR','TT'.
    Theta is (P,) azimuths in radians.
    """
    comps_u = [c.strip().upper() for c in comps]
    out: Dict[str, np.ndarray] = {}

    # Base components
    ZZ = corr[:, 0, 0, :]
    if "ZZ" in comps_u:
        out["ZZ"] = ZZ

    # If RR/TT requested, compute from N/E submatrix and theta
    need_rt = any(c in ("RR", "TT") for c in comps_u)
    if need_rt:
        NN = corr[:, 1, 1, :]
        NE = corr[:, 1, 2, :]
        EN = corr[:, 2, 1, :]
        EE = corr[:, 2, 2, :]
        if isinstance(corr, torch.Tensor):
            c = torch.cos(theta)
            s = torch.sin(theta)
        else:
            c = np.cos(theta)
            s = np.sin(theta)
        cs = (c * s)[:, None]
        c2 = (c * c)[:, None]
        s2 = (s * s)[:, None]
        RR = c2 * NN + s2 * EE + cs * (NE + EN)
        TT = s2 * NN + c2 * EE - cs * (NE + EN)
        if "RR" in comps_u:
            out["RR"] = RR
        if "TT" in comps_u:
            out["TT"] = TT

    return out


def percentile_clip(img: np.ndarray, p: float) -> Tuple[float, float]:
    """Compute symmetric percentile clip bounds for an image (on absolute values).

    Returns (vmin, vmax) suitable for imshow with origin='lower'.
    """
    p = float(p)
    if p <= 0:
        vmax = float(np.max(np.abs(img))) if img.size else 1.0
    else:
        vmax = float(np.percentile(np.abs(img), p))
    if vmax <= 0:
        vmax = 1.0
    return -vmax, vmax


def apply_window_to_refs(refs: Dict[str, np.ndarray], lags_s: np.ndarray, t0_s: Optional[np.ndarray], t1_s: Optional[np.ndarray]) -> Dict[str, np.ndarray]:
    """Return a copy of refs where samples outside [t0,t1] per pair are zeroed.

    Now applies the window symmetrically: keeps samples in the causal window
    [t0,t1] and in the anti-causal (negative-lag) window [-t1,-t0].

    If t0_s or t1_s is None, return original refs (copy).
    refs: dict comp->(P,T)
    lags_s: (T,)
    t0_s,t1_s: (P,) or None
    """
    out: Dict[str, np.ndarray] = {}
    if t0_s is None or t1_s is None:
        for comp, R in refs.items():
            out[comp] = R.copy()
        return out

    P = t0_s.size
    T = lags_s.size
    # Ensure lags_s is numpy array
    lags = np.asarray(lags_s)
    for comp, R in refs.items():
        R2 = R.copy()
        if R2.shape[0] != P:
            raise ValueError(f"Reference component {comp} has incompatible first dim: {R2.shape[0]} vs P={P}")
        for p in range(P):
            t0 = float(t0_s[p])
            t1 = float(t1_s[p])
            # causal mask: lags in [t0, t1]
            causal_mask = (lags >= t0) & (lags <= t1)
            # anti-causal mask: lags in [-t1, -t0]
            anti_mask = (lags >= -t1) & (lags <= -t0)
            keep_mask = causal_mask | anti_mask
            R2[p, ~keep_mask] = 0.0
        out[comp] = R2
    return out


def compute_offsets_m(coords_xy_m: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Compute pairwise inter-station distances (meters) for pairs array.

    coords_xy_m: (N,2) array of station local coordinates [E, N] in meters.
    pairs: (P,2) array of integer station indices (i,j) with i < j.

    Returns: offsets_m: (P,) float32 distances in meters.
    """
    i_idx = pairs[:, 0].astype(np.int32)
    j_idx = pairs[:, 1].astype(np.int32)
    dE = coords_xy_m[j_idx, 0] - coords_xy_m[i_idx, 0]
    dN = coords_xy_m[j_idx, 1] - coords_xy_m[i_idx, 1]
    return np.sqrt(dE * dE + dN * dN).astype(np.float32)


def compute_theta(coords_xy_m: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Compute azimuth (radians) from station i to j measured clockwise from North.

    Returns array of length P with azimuths theta = arctan2(dE, dN).
    Note: this convention matches other scripts in the repo expecting theta such that
    R =  cos(theta)*N + sin(theta)*E.
    """
    i_idx = pairs[:, 0].astype(np.int32)
    j_idx = pairs[:, 1].astype(np.int32)
    dE = coords_xy_m[j_idx, 0] - coords_xy_m[i_idx, 0]
    dN = coords_xy_m[j_idx, 1] - coords_xy_m[i_idx, 1]
    return np.arctan2(dE, dN).astype(np.float32)


def rotate_NE_corr_to_RR_TT(NN: np.ndarray, NE: np.ndarray, EN: np.ndarray, EE: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate N/E correlation components to R/T for given azimuth(s) theta.

    Inputs can be arrays over lag with leading dimension P for pairs.
    NN,NE,EN,EE shapes: (P,T) or (T,) for single-pair; theta: (P,) or scalar.
    Returns (RR, TT) with same shape as NN.
    """
    # Ensure theta is array-shaped for broadcasting
    th = np.asarray(theta)
    c = np.cos(th).astype(np.float32)
    s = np.sin(th).astype(np.float32)
    # c2, s2 shapes (P,1) if th is (P,)
    c2 = (c * c)[..., None]
    s2 = (s * s)[..., None]
    cs = (c * s)[..., None]
    RR = c2 * NN + s2 * EE + cs * (NE + EN)
    TT = s2 * NN + c2 * EE - cs * (NE + EN)
    return RR.astype(np.float32), TT.astype(np.float32)


def component_to_pairtrace(corr: np.ndarray, theta: np.ndarray, comp: str) -> np.ndarray:
    """Extract (P,T) traces for a requested component from corr (P,3,3,T).

    Supports 'ZZ','RR','TT'. This mirrors logic previously present in STEP04.
    """
    compu = comp.strip().upper()
    if compu == "ZZ":
        return corr[:, 0, 0, :].astype(np.float32)
    NN = corr[:, 1, 1, :].astype(np.float32)
    NE = corr[:, 1, 2, :].astype(np.float32)
    EN = corr[:, 2, 1, :].astype(np.float32)
    EE = corr[:, 2, 2, :].astype(np.float32)
    RR, TT = rotate_NE_corr_to_RR_TT(NN, NE, EN, EE, theta)
    if compu == "RR":
        return RR
    if compu == "TT":
        return TT
    raise ValueError(f"Unsupported comp: {comp}")


def segment_times_from_attrs(seg_path: Path) -> Tuple[UTCDateTime, UTCDateTime]:
    """Read starttime/endtime attributes from an HDF5 segment file and return two datetimes.

    Raises KeyError if attrs missing.
    """
    with h5py.File(seg_path, "r") as h5:
        t0 = UTCDateTime(h5.attrs["starttime"])
        t1 = UTCDateTime(h5.attrs["endtime"])
    return t0, t1


def day_from_attrs(seg_path: Path) -> dt.date:
    """Return the date corresponding to the segment starttime attribute."""
    t0, _ = segment_times_from_attrs(seg_path)
    return t0.date()


def xcorr_segment_files(seg_dir: Path,
                        tmin: UTCDateTime=None,
                        tmax: UTCDateTime=None) ->  Tuple[List[Path], List[Tuple[UTCDateTime, UTCDateTime]]]:
    """List all cross-correlation segment files in a directory.

    This is a thin helper used across STEP scripts to discover segment files.
    """
    # accept xcorr_*, bf_xc_* and legacy beam_* segment files
    STAMP_RE = re.compile(r"(?:xcorr|bf_xc|beam)_(\d{8})T(\d{6})Z\.h5$")
    files = sorted([p for p in seg_dir.glob("*.h5") if p.is_file()])
    files = [p for p in files if STAMP_RE.search(p.name)]
    times = [segment_times_from_attrs(p) for p in files]
    if tmin is not None:
        files = [p for i, p in enumerate(files) if times[i][0] >= tmin]
        times = [t for i, t in enumerate(times) if t[0] >= tmin]
    if tmax is not None:
        files = [p for i, p in enumerate(files) if times[i][0] < tmax]
        times = [t for i, t in enumerate(times) if t[0] < tmax]
    return files, times


@torch.no_grad()
def window_qc_mask_station(
    xw: torch.Tensor,
    fs: float,
    flim=(None, None),
    detrend: bool = True,
    zero_frac_max: float = 0.01,
    rms_z_max: float = 6.0,
    spike_ratio_max: float = 50.0,
    use_spectral: bool = False,
    band_energy_min: float = 0.05,
    line_ratio_max: float = 20.0,
    eps: float = 1e-8,
    verbose: bool = False,
):
    """Quality-control mask on windows xw with same semantics as STEP03.window_qc_mask_station.

    xw: (B,N,C,wp) torch tensor.
    Returns mask (B,N) boolean (True means keep).
    """
    if xw.ndim != 4:
        raise ValueError(f"xw must be (B,N,C,wp), got {tuple(xw.shape)}")

    B, N, C, wp = xw.shape
    dev = xw.device
    finite_ok = torch.isfinite(xw).all(dim=(2, 3))
    zero_frac = (xw == 0).float().mean(dim=(2, 3))
    zero_ok = zero_frac <= zero_frac_max

    var = xw.var(dim=-1)
    var_ok = (var > 0).all(dim=2)

    mask = finite_ok & zero_ok & var_ok
    if detrend:
        x = xw - xw.mean(dim=-1, keepdim=True)
    else:
        x = xw

    nfft = wp
    fmin, fmax = flim
    if fmin or fmax:
        xb = butterworth_bandpass(x, dt=1 / fs, fmin=fmin, fmax=fmax, order=4)
    else:
        xb = x

    rms = torch.sqrt((xb ** 2).mean(dim=-1) + eps)
    med = rms.median(dim=0).values
    mad = (rms - med).abs().median(dim=0).values + eps
    z = (rms - med).abs() / (1.4826 * mad)
    rms_ok = (z <= rms_z_max).all(dim=-1)

    peak = xb.abs().amax(dim=-1)
    spike_ratio = peak / (rms + eps)
    spike_ok = (spike_ratio <= spike_ratio_max).all(dim=-1)

    mask &= rms_ok & spike_ok

    if use_spectral:
        X = torch.fft.rfft(x, n=nfft, dim=-1)
        Xb = torch.fft.rfft(xb, n=nfft, dim=-1)
        Pxx = (X.abs() ** 2).sum(dim=2) + eps
        Pxx_b = (Xb.abs() ** 2).sum(dim=2) + eps
        E_tot = Pxx.sum(dim=-1)
        E_band = Pxx_b.sum(dim=-1)
        band_frac = (E_band / E_tot)
        band_ok = band_frac >= band_energy_min
        Ab = Xb.abs().sum(dim=2) + eps
        freqs = torch.fft.rfftfreq(nfft, d=1.0 / fs).to(dev)
        fb_mask = (freqs > fmin) & (freqs < fmax)
        Ab2 = Ab[..., fb_mask]
        line_ratio = Ab2.amax(dim=-1) / (Ab2.median(dim=-1).values + eps)
        line_ok = line_ratio <= line_ratio_max
        mask &= band_ok & line_ok

    if verbose:
        n_total = B * N
        n_bad = n_total - mask.sum().item()
        print(f"QC Total: {n_bad}/{n_total} windows/stations failed")

    return mask


def load_meta(meta_path: Path):
    with h5py.File(meta_path, "r") as h5:
        stations = [s.decode() for s in h5["meta/stations"][...]]
        coords = h5["meta/coords_xy_m"][...].astype(np.float32)  # (N,2) [E,N]
        pairs = h5["meta/pairs_i_j"][...].astype(np.int32)       # (P,2)
        lags = h5["meta/lags"][...].astype(np.float32)          # (T,)
    sta_to_idx = {s: i for i, s in enumerate(stations)}
    pair_to_pidx = {(int(i), int(j)): p for p, (i, j) in enumerate(pairs)}
    return stations, coords, pairs, sta_to_idx, pair_to_pidx, lags
