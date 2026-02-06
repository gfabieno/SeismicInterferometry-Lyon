import math
import torch

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 2 ** int(math.ceil(math.log2(n)))


def make_taper(wp: int, taper_pct: float, device: torch.device) -> torch.Tensor:
    taper_pct = float(max(0.0, min(0.5, taper_pct)))
    if taper_pct == 0.0:
        return torch.ones(wp, device=device, dtype=torch.float32)

    m = int(round(taper_pct * wp))
    if m < 2:
        return torch.ones(wp, device=device, dtype=torch.float32)

    w = torch.ones(wp, device=device, dtype=torch.float32)
    n = torch.arange(m, device=device, dtype=torch.float32)
    ramp = 0.5 * (1.0 - torch.cos(math.pi * n / (m - 1)))
    w[:m] = ramp
    w[-m:] = torch.flip(ramp, dims=[0])
    return w


def pairs_all(N: int, device: torch.device) -> torch.Tensor:
    ii, jj = torch.triu_indices(N, N, offset=1, device=device)
    return torch.stack([ii, jj], dim=1).to(torch.long)  # (P,2)

def cc_lags(wp: int, fs: float=1, device=None) -> torch.Tensor:
    """
    Return lag times (in seconds) for centered cross-correlation output.

    Convention:
      - length = wp
      - lag = 0 at index wp//2
      - negative lags on the left, positive on the right
      - matches SciPy correlate sign convention

    Returns
    -------
    lags : (wp,) torch.Tensor
        Lag times in seconds.
    """
    device = device or "cpu"
    center = wp // 2
    lags = torch.arange(wp, device=device) - center
    return lags / fs

import torch

def butterworth_bandpass_weight(
    freqs: torch.Tensor,
    fmin: float,
    fmax: float,
    order: int = 4,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Butterworth-like smooth bandpass magnitude weight in frequency domain.
    freqs: rfftfreq axis (>=0), shape (K,)
    returns weights in [0,1], shape (K,)
    """
    if fmin is None:
        fmin = 0.0
    if fmax is None:
        fmax = float(freqs.max().item())

    f = torch.clamp(freqs, min=eps)

    n2 = 2 * int(order)

    # High-pass at fmin
    if fmin <= 0.0:
        hp = torch.ones_like(f, dtype=torch.float32)
    else:
        hp = 1.0 / torch.sqrt(1.0 + (fmin / f) ** n2)

    # Low-pass at fmax
    if fmax >= float(freqs.max().item()):
        lp = torch.ones_like(f, dtype=torch.float32)
    else:
        lp = 1.0 / torch.sqrt(1.0 + (f / fmax) ** n2)

    w = (hp * lp).to(torch.float32)
    return w

def butterworth_bandpass(x: torch.Tensor,
    dt: float,
    fmin: float,
    fmax: float,
    order: int = 4,
    eps: float = 1e-12,
                         ) -> torch.Tensor:
    nfft = x.shape[-1]
    X = torch.fft.rfft(x, n=nfft, dim=-1)
    nfft = x.shape[-1]
    dev = x.device
    freqs = torch.fft.rfftfreq(nfft, d=dt).to(dev)
    w = butterworth_bandpass_weight(freqs, fmin, fmax, order=order, eps=eps)
    X = X * w
    return torch.fft.irfft(X, n=nfft, dim=-1)




@torch.no_grad()
def cross_correlation(
    x: torch.Tensor,                  # (N,C,L) float32, comps [Z,N,E]
    coords: torch.Tensor,             # (N,2) float32 (unused, kept for API symmetry)
    fs: float,
    wp: int,
    detrend: bool = True,
    flim: list = [None, None],        # [fmin, fmax] Hz
    p_chunk: int = None,
    pairs=None,                       # None or (P,2) long
    header: str = "Cross-correlation",
    onebit: bool = False,
    whiten: bool = False,
    eps: float = 1e-6,
    qc_fun = None,
):
    """
    Window-stacked FFT cross-correlation.

    Returns:
      dict with:
        corr  : (P, C, C, wp) float32 (circular corr, lag0 at index 0)
        pairs : (P,2) int64
        freq  : (K,) float32 (selected rFFT freqs)
    """
    dev = x.device
    if x.ndim != 3:
        raise ValueError(f"x must be (N,C,L), got {x.shape}")
    N, C, L = x.shape
    if coords is not None and coords.shape != (N, 2):
        raise ValueError(f"coords must be (N,2), got {coords.shape}")
    if wp > L:
        raise ValueError("wp must be <= L")
    if p_chunk is None:
        p_chunk = N - 1


    # Match beamforming convention: 50% overlap, 25% taper
    overlap = 0.5
    step = int(round(wp * (1.0 - overlap)))
    if step < 1:
        raise ValueError("Invalid wp/overlap -> step < 1")

    # (N,C,B,wp) then (B,N,C,wp)
    xw = x.unfold(dimension=-1, size=wp, step=step).permute(2, 0, 1, 3).contiguous()
    B = xw.shape[0]
    if B < 1:
        raise ValueError("No windows produced (check wp/length).")

    if qc_fun is not None:
        print("Applying QC function to windows...")
        w = qc_fun(xw).to(torch.float32)  # (B,N) float32 weights
        nkeep = w.sum().item()
        if nkeep < 1:
            raise ValueError("No windows passed QC.")
        print(f"{header}: QC kept {nkeep} out of {B * N} total ({nkeep / (B * N) * 100:.1f}%)")
    else:
        w = None


    if detrend:
        xw = xw - xw.mean(dim=-1, keepdim=True)

    taper = make_taper(wp, taper_pct=0.25, device=dev).view(1, 1, 1, wp)
    xw = xw * taper

    if onebit:
        xw = torch.sign(xw)
        xw = torch.where(xw == 0, torch.zeros((), device=dev), xw)

    nfft = _next_pow2(wp)
    freqs = torch.fft.rfftfreq(nfft, d=1.0 / fs).to(dev)

    fmin, fmax = flim
    if fmin is None:
        fmin = 0.0
    if fmax is None:
        fmax = fs / 2.0
    wband = butterworth_bandpass_weight(freqs, fmin, fmax, order=4).to(dev)
    wband = wband.view(1, 1, 1, -1)  # broadcast over (B,N,C,Kfull)


    # X: (B,N,C,K)
    X = torch.fft.rfft(xw, n=nfft, dim=-1)
    K = X.shape[-1]
    X = X * wband
    if whiten:
        # amplitude normalize within band (per window, station, component)
        amp = torch.abs(X)
        # make eps relative to max amplitude
        amp_max = torch.amax(amp, dim=-1, keepdim=True)
        amp = amp_max * float(eps) + amp
        X = X / amp
        X = X * wband

    # pairs
    if pairs is None:
        pairs_t = pairs_all(N, dev)
    else:
        pairs_t = torch.as_tensor(pairs, device=dev).to(torch.long)
        if pairs_t.ndim != 2 or pairs_t.shape[1] != 2:
            raise ValueError("pairs must be (P,2)")
    P = pairs_t.shape[0]

    # output: (P, C, C, wp)
    corr = torch.empty((P, C, C, wp), device=dev, dtype=torch.float32)

    #clear cache if using GPU
    if 'cuda' in str(dev):
        torch.cuda.empty_cache()

    # progress bar like beamforming
    it = range(0, P, p_chunk)
    if tqdm is not None:
        it = tqdm(it, desc=header, unit=" chunks", dynamic_ncols=True)

    for p0 in it:
        p1 = min(P, p0 + p_chunk)
        pc = p1 - p0
        ij = pairs_t[p0:p1]          # (pc,2)
        i = ij[:, 0]
        j = ij[:, 1]

        # Xi, Xj: (B,pc,C,K)
        Xi = X[:, i, :, :]
        Xj = X[:, j, :, :]

        # Cross-spectrum averaged over windows:
        # A: (B, pc, C, 1, K)
        A = Xi.unsqueeze(-2)
        # Bc: (B, pc, 1, C, K)  (conjugated)
        Bc = torch.conj(Xj).unsqueeze(-3)

        # prod: (B, pc, C, C, K)
        prod = A * Bc

        # S: (pc, C, C, K)
        if w is not None:
            wi = w[:, i].view(B, pc, 1, 1, 1)
            wj = w[:, j].view(B, pc, 1, 1, 1)
            wij = wi * wj
            S = (prod * wij).sum(dim=0) / (wij.sum(dim=0) + eps)
        else:
            S = prod.mean(dim=0)

        # IFFT to time: (pc,C,C,nfft)
        cc = torch.fft.irfft(S, n=nfft, dim=-1)

        # shift lag0 at center
        cc = torch.roll(cc, shifts=nfft // 2, dims=-1)

        #keep only wp samples around lag0
        cc = cc[..., nfft//2 - wp//2 : nfft//2 + (wp - wp//2)]
        # keep first wp samples (circular corr); lag0 at index 0
        corr[p0:p1] = cc.to(torch.float32)

    return {
        "corr": corr,               # (P,C,C,wp)
        "pairs": pairs_t,           # (P,2)
        "freq": freqs.to(torch.float32),
        "wp": torch.tensor(wp, device=dev),
        "fs": torch.tensor(float(fs), device=dev),
    }
