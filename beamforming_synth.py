# ----------------------------
# Synthetic beamforming: Ricker plane waves (Rayleigh + Love)
# ----------------------------

import math
from typing import Tuple, Dict
import numpy as np
import torch
from beamforming import plane_wave_beamforming, plot_beamforming, \
    make_slowness_grid
from beamforming import _next_pow2
import matplotlib.pyplot as plt

def ricker(t: torch.Tensor, f0: float, t0: float) -> torch.Tensor:
    """
    Ricker wavelet centered at t0 with central frequency f0.
    """
    x = math.pi * f0 * (t - t0)
    return (1.0 - 2.0 * x**2) * torch.exp(-x**2)


def az_to_slow(s: float, az_rad: float) -> torch.Tensor:
    """
    az measured clockwise from North.
    slow=(sx,sy) with sx=E=s*sin(az), sy=N=s*cos(az)
    """
    return torch.tensor([s * math.sin(az_rad), s * math.cos(az_rad)], dtype=torch.float32)


@torch.no_grad()
def shift_via_phase(sig: torch.Tensor, dt_sec: torch.Tensor, fs: float) -> torch.Tensor:
    """
    Fractional shift using FFT phase ramp.
    sig: (L,)
    dt_sec: (N,) seconds
    returns: (N,L)
    """
    device = sig.device
    L = sig.numel()
    nfft = _next_pow2(L)
    freqs = torch.fft.rfftfreq(nfft, d=1.0/fs).to(device)
    omega = 2 * math.pi * freqs
    S = torch.fft.rfft(sig, n=nfft)  # (K,)
    phase = torch.exp(1j * dt_sec.unsqueeze(-1) * omega.view(1, -1))  # (N,K)
    y = torch.fft.irfft(S.view(1, -1) * phase, n=nfft)[..., :L]
    return y.real


def synth_3c_rayleigh_love(
    coords_xy_m: torch.Tensor,   # (N,2) meters (E,N)
    fs: float,
    L: int,
    v_rayleigh: float,
    v_love: float,
    az_rayleigh_deg: float,
    az_love_deg: float,
    f0_rayleigh: float,
    f0_love: float,
    amp_rayleigh: float = 1.0,
    amp_love: float = 0.8,
    rayleigh_ellipticity: Tuple[float, float] = (1.0, 0.6),  # (Z_amp, R_amp)
    noise_std: float = 0.3,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Builds synthetic 3C data as DPE,DPN,DPZ then returns xZNE (N,3,L) [Z,N,E].

    Rayleigh: Z + Radial (elliptical) along propagation direction
    Love: Transverse only (horizontal, perpendicular to propagation)

    Returns:
      xZNE: (N,3,L) float32 [Z,N,E]
      truth: dict with true slowness vectors for rayleigh & love (sx,sy)
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    coords = coords_xy_m.to(device=device, dtype=torch.float32)
    N = coords.shape[0]
    t = torch.arange(L, device=device, dtype=torch.float32) / fs

    # Slowness vectors
    azR = math.radians(az_rayleigh_deg)
    azL = math.radians(az_love_deg)
    sR = 1.0 / v_rayleigh
    sL = 1.0 / v_love
    slow_R = az_to_slow(sR, azR).to(device)
    slow_L = az_to_slow(sL, azL).to(device)

    # Base Ricker pulses with different center freqs
    t0 = (L / fs) * 0.4
    wR = ricker(t, f0_rayleigh, t0)
    wL = ricker(t, f0_love, t0 + 2.0 / f0_love)  # shift love a bit so they overlap less

    wR = wR / (wR.std() + 1e-12)
    wL = wL / (wL.std() + 1e-12)

    # Delays per station: dt = sx*x + sy*y
    dtR = (coords @ slow_R.view(2, 1)).squeeze(-1)  # (N,)
    dtL = (coords @ slow_L.view(2, 1)).squeeze(-1)

    # Shift signals per station
    yR = shift_via_phase(wR, dtR, fs)  # (N,L)
    yL = shift_via_phase(wL, dtL, fs)

    # Polarization projection
    # Rayleigh: Z + Radial along azR
    caR, saR = math.cos(azR), math.sin(azR)
    Zamp, Ramp = rayleigh_ellipticity

    Z_R = (amp_rayleigh * Zamp) * yR
    N_R = (amp_rayleigh * Ramp * caR) * yR
    E_R = (amp_rayleigh * Ramp * saR) * yR

    # Love: Transverse only (perp to propagation). Unit transverse is +90deg from radial.
    # T = -N*sin(az) + E*cos(az) so if we want pure T motion with amplitude A:
    # set N = -A*sin(az), E = A*cos(az)
    caL, saL = math.cos(azL), math.sin(azL)

    Z_L = torch.zeros_like(yL)
    N_L = (amp_love * (-saL)) * yL
    E_L = (amp_love * (caL)) * yL

    # Sum components
    Z = Z_R + Z_L
    Nn = N_R + N_L
    Ee = E_R + E_L

    # Add incoherent station noise
    Z = Z + noise_std * torch.randn((N, L), generator=g, device=device)
    Nn = Nn + noise_std * torch.randn((N, L), generator=g, device=device)
    Ee = Ee + noise_std * torch.randn((N, L), generator=g, device=device)

    # Return xZNE (N,3,L)
    xZNE = torch.stack([Z, Nn, Ee], dim=1).to(torch.float32)

    truth = {"slow_R": slow_R.detach().cpu(), "slow_L": slow_L.detach().cpu()}
    return xZNE, truth

# ----------------------------
# End-to-end synthetic test
# ----------------------------

def run_synthetic_demo(device: str = "cuda:1" if torch.cuda.is_available() else "cpu"):
    torch.manual_seed(0)

    # Array geometry: 10 sensors in ~500 m aperture
    N = 10
    coords = torch.randn(N, 2) * 250.0  # meters (E,N)
    coords = coords - coords.mean(dim=0, keepdim=True)

    # Signal parameters
    fs = 250.0
    dur = 120.0  # seconds
    L = int(dur * fs)

    # Rayleigh + Love (different azimuth & center frequency)
    xZNE, truth = synth_3c_rayleigh_love(
        coords_xy_m=coords,
        fs=fs,
        L=L,
        v_rayleigh=3200.0,
        v_love=2800.0,
        az_rayleigh_deg=40.0,
        az_love_deg=140.0,
        f0_rayleigh=2.5,
        f0_love=5.0,
        amp_rayleigh=1.0,
        amp_love=0.8,
        rayleigh_ellipticity=(1.0, 0.7),
        noise_std=0.35,
        seed=1,
        device=device,
    )
    X, Y = np.meshgrid(np.arange(-125, 125), np.arange(-125, 125))
    coords_mesh = np.stack([X.ravel(), Y.ravel()], axis=1)
    coords_mesh = torch.from_numpy(coords_mesh).float().to(device)
    Wavefield, truth = synth_3c_rayleigh_love(
        coords_xy_m=coords_mesh,
        fs=fs,
        L=int(1*fs),
        v_rayleigh=3200.0,
        v_love=2800.0,
        az_rayleigh_deg=40,
        az_love_deg=140.0,
        f0_rayleigh=2.5,
        f0_love=5.0,
        amp_rayleigh=1.0,
        amp_love=0,
        rayleigh_ellipticity=(1.0, 0.7),
        noise_std=0,
        seed=1,
        device=device,
    )
    plt.figure()
    plt.imshow(torch.argmax(Wavefield[:, 0, :], dim=-1).reshape(*X.shape, -1).cpu()/fs,
               extent=[-125, 125, -125, 125], origin="lower")
    plt.xlabel("East (m)")
    plt.ylabel("North (m)")
    plt.title("Traveltime field (Z component)")
    plt.colorbar()
    plt.show()

    # Slowness grid (s/m): pick bounds around 1/v ~ 3e-4 s/m
    slow_grid, SX, SY = make_slowness_grid(
        sx_lim=6e-4, sy_lim=6e-4, ds=1.5e-5, device=device
    )

    # Beamform settings
    Wp_sec = 10.0
    wp = int(Wp_sec * fs)

    # Run beamforming
    res = plane_wave_beamforming(
        xZNE.to(device),
        coords.to(device),
        fs,
        slow_grid,
        wp,
        detrend=True,
        flim=[1, 8],
    )

    # Combine components and average over frequency to make a 2D score map
    pwr = (res["pwr_Z"] + res["pwr_R"] + res["pwr_T"]).mean(dim=0)  # (S,)
    score = pwr.detach().cpu().numpy().reshape(SX.shape)  # (ny,nx)

    # Plot results
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    slim = [SX.min(), SX.max(), SY.min(), SY.max()]
    plot_beamforming(slim, res["pwr_Z"].mean(dim=0).cpu().reshape(SX.shape),
                     title="Beamforming Z", ax=ax[0])
    plot_beamforming(slim, res["pwr_R"].mean(dim=0).cpu().reshape(SX.shape),  #truth=truth,
                     title="Beamforming R", ax=ax[1])
    plot_beamforming(slim, res["pwr_T"].mean(dim=0).cpu().reshape(SX.shape),  #truth=truth,
                     title="Beamforming T", ax=ax[2])
    plot_beamforming(slim, score,
                     title="Beamforming Total Power", ax=ax[3])
    for a in ax:
        a.plot(truth["slow_R"][0].item(), truth["slow_R"][1].item(), "r*",
                   markersize=12, label="True Rayleigh")
        a.plot(truth["slow_L"][0].item(), truth["slow_L"][1].item(), "g*",
                markersize=12, label="True Love")
        a.legend()
    plt.tight_layout()
    plt.show()



    # Optional: report best grid point
    best_idx = int(np.argmax(score))
    best_sy, best_sx = np.unravel_index(best_idx, score.shape)  # note imshow grid indexing
    sx_best = SX[best_sy, best_sx]
    sy_best = SY[best_sy, best_sx]
    az_best = (math.degrees(math.atan2(sx_best, sy_best)) + 360.0) % 360.0
    v_app = 1.0 / math.hypot(sx_best, sy_best)
    print(f"Best peak: sx={sx_best:.3e} s/m, sy={sy_best:.3e} s/m, az≈{az_best:.1f}°, v_app≈{v_app:.0f} m/s")


if __name__ == "__main__":
    run_synthetic_demo()