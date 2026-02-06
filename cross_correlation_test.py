# test_cross_correlation.py
# Run with: pytest -q
import torch
from cross_correlation import cross_correlation, cc_lags
import scipy.signal
import numpy as np

def _make_delayed_spike_pair(wp: int, delay: int, device="cpu"):
    """
    Build x: (N=2, C=3, L=wp) with a single spike on Z component.
    Station 1 is delayed by `delay` samples relative to station 0.
    """
    N, C, L = 2, 3, wp
    x = torch.zeros((N, C, L), dtype=torch.float32, device=device)

    # Put a spike safely away from edges
    t0 = L // 4
    t1 = t0 - delay
    assert 0 <= t1 < L

    x[0, 0, t0] = 1.0  # station 0, Z
    x[1, 0, t1] = 1.0  # station 1, Z delayed
    return x


def _peak_offset_from_center(y_1d: torch.Tensor) -> int:
    """
    Return signed offset of argmax from center index (wp//2).
    """
    wp = y_1d.numel()
    k = int(torch.argmax(torch.abs(y_1d)).item())
    return int(cc_lags(wp=wp)[k].item())


def test_xcorr_3d_shape_and_component_isolation_even():
    wp = 1000   # even
    delay = 37
    fs = 250.0

    x = _make_delayed_spike_pair(wp=wp, delay=delay)
    coords = torch.zeros((2, 2), dtype=torch.float32)
    pairs = torch.tensor([[0, 1]], dtype=torch.long)

    out = cross_correlation(
        x=x,
        coords=coords,
        fs=fs,
        wp=wp,
        detrend=False,
        flim=[None, None],
        p_chunk=64,
        pairs=pairs,
        header="test",
        onebit=False,
        whiten=False,
    )

    corr = out["corr"]  # (P, C, C, wp)
    assert corr.shape == (1, 3, 3, wp)

    # Peak should be on ZZ only; other component pairs should be ~0
    zz = corr[0, 0, 0, :]
    zn = corr[0, 0, 1, :]
    ze = corr[0, 0, 2, :]
    nz = corr[0, 1, 0, :]

    # ZZ peak offset should be +/- delay (sign depends on correlation convention)
    off = _peak_offset_from_center(zz)
    assert off == delay

    # Non-ZZ pairs should have no significant peak
    assert torch.max(torch.abs(zn)).item() < 1e-6
    assert torch.max(torch.abs(ze)).item() < 1e-6
    assert torch.max(torch.abs(nz)).item() < 1e-6


def test_xcorr_lag0_centering_odd():
    wp = 1001   # odd
    delay = 21
    fs = 250.0

    x = _make_delayed_spike_pair(wp=wp, delay=delay)
    coords = torch.zeros((2, 2), dtype=torch.float32)
    pairs = torch.tensor([[0, 1]], dtype=torch.long)

    out = cross_correlation(
        x=x,
        coords=coords,
        fs=fs,
        wp=wp,
        detrend=False,
        flim=[None, None],
        p_chunk=64,
        pairs=pairs,
        header="test",
        onebit=False,
        whiten=False,
    )

    corr = out["corr"]
    assert corr.shape == (1, 3, 3, wp)

    zz = corr[0, 0, 0, :]
    center = wp // 2

    # Basic centering check: lag=0 corresponds to index center.
    # For two different (delayed) spikes, the maximum should be at center +/- delay.
    off = int(torch.argmax(torch.abs(zz)).item()) - center
    assert abs(off) == delay


def test_xcorr_even_and_odd_have_lag0_at_center_index():
    """
    Stronger centering test: if both stations have a spike at the SAME time,
    the correlation maximum must be exactly at lag=0 -> index wp//2.
    """
    fs = 250.0
    for wp in (1000, 1001):
        x = torch.zeros((2, 3, wp), dtype=torch.float32)
        t0 = wp // 3
        x[0, 0, t0] = 1.0
        x[1, 0, t0] = 1.0  # no delay
        coords = torch.zeros((2, 2), dtype=torch.float32)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)

        out = cross_correlation(
            x=x,
            coords=coords,
            fs=fs,
            wp=wp,
            detrend=False,
            flim=[None, None],
            p_chunk=64,
            pairs=pairs,
            header="test",
            onebit=False,
            whiten=False,
        )

        zz = out["corr"][0, 0, 0, :]
        kmax = int(torch.argmax(torch.abs(zz)).item())
        assert kmax == (wp // 2)

def test_cc_lags():
    # test out cc_lags against scipy's correlate lag calculation
    for wp in (1000, 1001):
        lags = cc_lags(wp=wp).cpu().numpy()
        # scipy lags
        scipy_lags = scipy.signal.correlation_lags(wp, wp, mode='same')
        assert lags.shape == (wp,)
        assert scipy_lags.shape == (wp,)
        assert all(abs(lags - scipy_lags) < 1e-6)

def test_cc_scipy():
    # test cross_correlation output against scipy correlate for simple signals
    wp = 1001
    fs = 250.0
    x = _make_delayed_spike_pair(wp=wp, delay=21)
    coords = torch.zeros((2, 2), dtype=torch.float32)
    pairs = torch.tensor([[0, 1]], dtype=torch.long)

    out = cross_correlation(
        x=x,
        coords=coords,
        fs=fs,
        wp=wp,
        detrend=False,
        flim=[None, None],
        p_chunk=64,
        pairs=pairs,
        header="test",
        onebit=False,
        whiten=False,
    )

    zz = out["corr"][0, 0, 0, :]
    zz_scipy = scipy.signal.correlate(x[0,0,:].cpu().numpy(), x[1,0,:].cpu().numpy(), mode='same')
    import matplotlib.pyplot as plt
    plt.plot(zz.cpu().numpy(), label='cross_correlation')
    plt.plot(zz_scipy, label='scipy correlate', linestyle='dashed')
    plt.legend()
    plt.show()
    assert zz.shape == zz_scipy.shape
    assert np.max(zz.cpu().numpy() - zz_scipy) < 1e-6




if __name__ == "__main__":
    test_xcorr_3d_shape_and_component_isolation_even()
    test_xcorr_lag0_centering_odd()
    test_xcorr_even_and_odd_have_lag0_at_center_index()
    test_cc_lags()
    test_cc_scipy()
    print("All tests passed.")