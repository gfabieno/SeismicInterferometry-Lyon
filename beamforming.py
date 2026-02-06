import math
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** int(math.ceil(math.log2(n)))

def slowness_to_azimuth(slow_s_per_m: torch.Tensor) -> torch.Tensor:
    """
    slow_s_per_m: (S,2) with (sx, sy) = (East, North) in s/m
    returns azimuth radians, clockwise from North, direction of propagation.
    """
    sx = slow_s_per_m[:, 0]
    sy = slow_s_per_m[:, 1]
    return torch.atan2(sx, sy)  # (S,)

def make_taper(wp: int, taper_pct: float, device: torch.device) -> torch.Tensor:
    """
    Simple Tukey-like taper: cosine ramp on both ends, flat in the middle.
    taper_pct is fraction of window length tapered on EACH side (0..0.5).
    Example: taper_pct=0.05 -> 5% ramp on left + 5% on right.
    """
    taper_pct = float(max(0.0, min(0.5, taper_pct)))
    if taper_pct == 0.0:
        return torch.ones(wp, device=device, dtype=torch.float32)

    m = int(round(taper_pct * wp))
    if m < 2:
        return torch.ones(wp, device=device, dtype=torch.float32)

    w = torch.ones(wp, device=device, dtype=torch.float32)
    # cosine ramp from 0 to 1
    n = torch.arange(m, device=device, dtype=torch.float32)
    ramp = 0.5 * (1.0 - torch.cos(math.pi * n / (m - 1)))

    w[:m] = ramp
    w[-m:] = torch.flip(ramp, dims=[0])
    return w


@torch.no_grad()
def plane_wave_beamforming(
    x: torch.Tensor,            # (N, C, L) float32, components [Z, N, E]
    coords: torch.Tensor,       # (N, 2) float32, (E, N) in meters
    fs: float,  # sampling rate in Hz
    slow: torch.Tensor,         # (S, 2)  (sx, sy) with sx=E, sy=N in s/m
    wp: int,                    # window length in samples
    detrend: bool = True,       # remove mean per window per station/component
    flim: List[float] = [None, None],  # [fmin, fmax] in Hz
    s_chunk: int = 512,         # slowness chunk size
    header: str = "Beamforming", # optional header for progress bar
    qc_fun: Optional[callable] = None
) -> Dict[str, torch.Tensor]:
    """
    Plane-wave beamforming in frequency domain with rotation from N/E to R/T
    for 3C.

    Args:
        x (torch.Tensor): Input time series, shape (N, C, L), dtype float32.
                          N = number of stations,
                          C = components (1 or 3; order [Z, N, E]),
                          L = time samples.
        coords (torch.Tensor): Station coordinates in meters, shape (N, 2).
                               Columns are (E, N) = (East, North).
        fs (float): Sampling rate in Hz.
        slow (torch.Tensor): Slowness grid, shape (S, 2), dtype float32.
                             Each row is (sx, sy) in s/m with sx = East, sy = North.
        wp (int): Processing window length in samples.
                  Must be a power of two and <= L.
        detrend (bool, optional): If True, remove the mean per window per station/component.
                                  Default is True.
        flim (list of float, optional): Frequency limits [fmin, fmax] in Hz.
                                        Defaults to [None, None] (i.e., use all frequencies).
        s_chunk (int, optional): Number of slowness samples processed per chunk
                                 to limit memory use. Default is 512.
        header (str, optional): Optional header for progress bar.
                                Default is ""Beamforming"".
        qc_fun (callable, optional): Optional quality control function called per window.

    Returns:
        dict: A dictionary with keys:
            "pwr_Z" (torch.Tensor): Power proxy for Z component, shape (K, S)
                                    where K = number of selected frequency bins
                                    and S = number of slowness grid points.
            "pwr_R" (torch.Tensor or None): Power proxy for radial component (R)
            "pwr_T" (torch.Tensor or None): Power proxy for transverse component (T)
            "freq" (torch.Tensor): Frequencies (Hz) of size (K,)
    """
    dev = x.device

    N, C, L = x.shape
    if C not in [1, 3]:
        raise ValueError("x must have 1 or 3 components.")
    if coords.shape != (N, 2):
        raise ValueError(f"coords must be (N,2), got {coords.shape}")
    if slow.ndim != 2 or slow.shape[1] != 2:
        raise ValueError("slow be (S,2)")

    # Overlap 50% with a taper of 25% so each sample is tapered only once
    overlap = 0.5
    step = int(round(wp * (1.0 - overlap)))

    # Build sliding windows as a view: (N,C,B,wp)
    if L < wp:
        raise ValueError("Time-batch shorter than one Wp window.")

    xw = x.unfold(dimension=-1, size=wp, step=step)  # (N,C,B,wp)
    # Reorder to (B,N,C,wp)
    xw = xw.permute(2, 0, 1, 3).contiguous()
    B = xw.shape[0]
    if B < 1:
        raise ValueError("No windows produced (check wp/overlap/length).")
    if qc_fun:
        w = qc_fun(xw).to(torch.float32)  # (B,N) float32 weights
        nkeep = w.sum().item()
        if nkeep < 1:
            raise ValueError("No windows passed QC.")
        print(f"{header}: QC kept {nkeep} out of {B * N} total ({nkeep / (B * N) * 100:.1f}%)")
    else:
        w = None
        nkeep = B * N

    # Detrend (remove mean per window, per station/component)
    if detrend:
        xw = xw - xw.mean(dim=-1, keepdim=True)

    # Taper per window
    taper = make_taper(wp, taper_pct=0.25, device=dev).view(1, 1, 1, wp)
    xw = xw * taper

    # Frequencies and mask
    nfft = _next_pow2(wp)
    freqs = torch.fft.rfftfreq(nfft, d=1.0 / fs).to(dev)
    fmin, fmax = flim
    if fmin is None:
        fmin = 0.0
    if fmax is None:
        fmax = fs / 2.0
    fmask = (freqs >= fmin) & (freqs <= fmax)
    omega = (2.0 * math.pi * freqs[fmask]).to(torch.float32)  # (K,)
    K = int(omega.numel())

    # FFT for all windows/stations/components: (B,N,C,K)
    X = torch.fft.rfft(xw, n=nfft, dim=-1)[..., fmask]
    if w is not None:
        X = X * w.view(B, N, 1, 1)
    XZ = X[:, :, 0, :]  # (B,N,K)
    if C==3:
        XN = X[:, :, 1, :]
        XE = X[:, :, 2, :]

    if C==3:
        # Slowness -> azimuth (propagation direction), clockwise from North
        az = slowness_to_azimuth(slow)  # (S,)
        ca = torch.cos(az).to(torch.float32)
        sa = torch.sin(az).to(torch.float32)

    # Delays dt = sx*x + sy*y, coords are (E,N)
    center = torch.mean(coords, dim=0, keepdim=True)
    delays = slow @ (coords-center).T  # (S,N)

    if delays.abs().max() > 0.1225 * wp * (1.0 / fs):
        print("Warning: max delay exceeds window length, "
              "wrap-around may occur. Increase wp.")

    S = slow.shape[0]
    outZ = torch.empty((K, S), device=dev, dtype=torch.float32)
    if C==3:
        outR = torch.empty((K, S), device=dev, dtype=torch.float32)
        outT = torch.empty((K, S), device=dev, dtype=torch.float32)
    else:
        outR = None
        outT = None

    #clear cache if using GPU
    if 'cuda' in str(dev):
        torch.cuda.empty_cache()

    # add progress bar
    for s0 in tqdm(range(0, S, s_chunk),
                   desc=header, unit=" chunks", dynamic_ncols=True):
        s1 = min(S, s0 + s_chunk)
        Sc = s1 - s0

        XZ_c = XZ.unsqueeze(1).expand(-1, Sc, -1, -1)
        # rotate in frequency domain: XR/XT are (B,Sc,N,K)
        if C==3:
            ca_c = ca[s0:s1].view(1, Sc, 1, 1)  # (Sc,1,1)
            sa_c = sa[s0:s1].view(1, Sc, 1, 1)
            XR = ca_c * XN.unsqueeze(1) + sa_c * XE.unsqueeze(1)
            XT = -sa_c * XN.unsqueeze(1) + ca_c * XE.unsqueeze(1)

        # Phase ramps: (Sc,N,K)
        dt = delays[s0:s1]  # (Sc,N)
        phase = torch.exp(-1j * dt.unsqueeze(-1) * omega.view(1, 1, K))
        # Stack across stations -> (B,Sc,K)
        BZ = torch.sum(XZ_c * phase.unsqueeze(0), dim=2)  # (B,Sc,K)
        if C==3:
            BR = torch.sum(XR * phase.unsqueeze(0), dim=2)
            BT = torch.sum(XT * phase.unsqueeze(0), dim=2)

        # Power proxy: mean over windows -> (Sc,K)
        pZ = torch.sum(torch.abs(BZ) ** 2, dim=0)
        if C==3:
            pR = torch.sum(torch.abs(BR) ** 2, dim=0)  # (Sc, K)
            pT = torch.sum(torch.abs(BT) ** 2, dim=0)

        outZ[:, s0:s1] = pZ.T / nkeep
        if C==3:
            outR[:, s0:s1] = pR.T / nkeep
            outT[:, s0:s1] = pT.T / nkeep

    return {
        "pwr_Z": outZ,
        "pwr_R": outR,
        "pwr_T": outT,
        "freq": freqs[fmask]
    }


def plot_beamforming(
    slim: tuple[float, float, float, float],
    score: np.ndarray,
    v_levels: np.ndarray | None = None,
    angle_deg_step: float = 30.0,
    angle_deg_list: list[float] | None = None,
    show_colorbar: bool = True,
    title: str = "Beamforming output (sx, sy) with polar overlay",
    cmap: str = "viridis",
    ax = None,
    units: str = "s/m"
):
    """
    Plot beamforming power on a Cartesian slowness grid using imshow, and overlay:
      - isovelocity curves: |s| = 1/v  (circles in (sx,sy))
      - angle lines: azimuth lines (clockwise from North), drawn as rays from origin

    Conventions:
      sx = East  (x-axis)
      sy = North (y-axis)
      azimuth θ measured clockwise from North:
        sx = |s| * sin(θ), sy = |s| * cos(θ)

    Parameters
    ----------
    slim : tuple of float
        (sx_min, sx_max, sy_min, sy_max) extents for imshow (in m/s)
    score : 2D array (ny,nx)
        Beam power.
    v_levels : array-like, optional
        Velocities (m/s) at which to draw isovelocity circles.
        If None, chooses a few based on slowness extent.
    angle_deg_step : float
        Step in degrees for azimuth rays if angle_deg_list is None.
    angle_deg_list : list of float, optional
        Explicit azimuth angles (deg) for rays. Overrides angle_deg_step.
    show_colorbar : bool
        Add colorbar.
    title, cmap : str
        Plot cosmetics.
    """
    score = np.asarray(score)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    im = ax.imshow(
        score,
        origin="lower",
        extent=slim,
        aspect="equal",
        cmap=cmap,
    )

    ax.set_xlabel(f"sx ({units}) [East]")
    ax.set_ylabel(f"sy ({units}) [North]")
    ax.set_title(title)

    sx_min, sx_max, sy_min, sy_max = slim
    # Determine maximum slowness radius visible (for overlay scaling)
    smax_visible = min(
        max(abs(sx_min), abs(sx_max)),
        max(abs(sy_min), abs(sy_max))
    )

    # --- Isovelocity circles: |s| = 1/v ---
    if v_levels is None:
        # pick a few circles that fit in the plot
        # choose velocities corresponding to fractions of smax_visible
        # v = 1/|s|
        # ensure finite and reasonable
        s_levels = np.array([0.25, 0.5, 0.75, 1.0], dtype=float) * smax_visible
        s_levels = s_levels[s_levels > 0]
        v_levels = 1.0 / s_levels
        # sort increasing velocity (larger v = smaller circle)
        v_levels = np.sort(v_levels)

    v_levels = np.asarray(v_levels, dtype=float)
    theta = np.linspace(0, 2*np.pi, 720)

    for v in v_levels:
        if not np.isfinite(v) or v <= 0:
            continue
        r = 1.0 / v  # slowness radius
        if r <= 0 or r > smax_visible * 1.05:
            continue
        x = r * np.sin(theta)  # sx
        y = r * np.cos(theta)  # sy
        ax.plot(x, y, "--", linewidth=1.0, color="white", alpha=0.5)
        # label near North (top of circle)
        ax.text(0.0, r, f"{v:.0f} {units}", ha="left", va="bottom", fontsize=9,
                color="white")

    # --- Azimuth rays ---
    if angle_deg_list is None:
        angle_deg_list = list(np.arange(0, 360, angle_deg_step, dtype=float))
    else:
        angle_deg_list = [float(a) for a in angle_deg_list]

    # Ray length (in slowness units)
    ray_len = smax_visible * 0.98

    for ang in angle_deg_list:
        th = np.deg2rad(ang)
        # line from origin: sx = r*sin(th), sy = r*cos(th)
        x = np.array([0.0, ray_len * np.sin(th)])
        y = np.array([0.0, ray_len * np.cos(th)])
        ax.plot(x, y, "--", linewidth=1.0, color="white", alpha=0.5)

        # label at end
        ax.text(
            x[1]*0.87, y[1]*0.87,
            f"{ang:.0f}°",
            ha="center", va="center",
            fontsize=9, color="white"
        )

    # origin marker
    ax.plot([0], [0], marker="+", markersize=10)

    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(
            position="right",
            size="5%",  # width of colorbar
            pad=0.02  # gap between plot and colorbar
        )
        cb = fig.colorbar(im, cax=cax)
        #cb.set_label("beam power (arb.)")


def make_slowness_grid(sx_lim: float, sy_lim: float, ds: float, device="cpu") -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Returns:
      slow: (S,2) torch tensor
      SX, SY: 2D numpy grids for plotting
    """
    sx = np.arange(-sx_lim, sx_lim + ds, ds, dtype=np.float32)
    sy = np.arange(-sy_lim, sy_lim + ds, ds, dtype=np.float32)
    SX, SY = np.meshgrid(sx, sy, indexing="xy")  # (ny,nx)
    slow = torch.from_numpy(np.stack([SX.ravel(), SY.ravel()], axis=1)).to(device=device)
    return slow, SX, SY
