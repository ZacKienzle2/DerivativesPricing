"""Dupire local-volatility bootstrapper from a market IV surface.

Dupire 1994:
    `sigma_loc^2(K, T) = (C_T + (r - q) K C_K + q C) / (0.5 K^2 C_KK)`
where `C(K, T)` is the European call price as a function of strike and
maturity. Numerical derivatives are computed by finite differences on the
input IV surface (mapped through Black-Scholes) with central differences
in `(K, T)`. Output is clamped to a positive interval and returned as a
ready-to-use `LocalVolProcess`.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from ..pricers._analytic_kernels import bs_price_jit
from ..processes.local_vol import LocalVolProcess


def _bs_call_grid(
    s0: float,
    strikes: npt.NDArray,
    maturities: npt.NDArray,
    iv_grid: npt.NDArray,
    r: float,
    q: float,
) -> npt.NDArray:
    """Builds a `(len(strikes), len(maturities))` Black-Scholes call grid."""
    out = np.empty_like(iv_grid)
    for i in range(strikes.size):
        for j in range(maturities.size):
            out[i, j] = bs_price_jit(
                s0, float(strikes[i]), float(maturities[j]),
                r, q, float(iv_grid[i, j]), True,
            )
    return out


def dupire_local_vol_grid(
    s0: float,
    strikes: npt.NDArray,
    maturities: npt.NDArray,
    iv_grid: npt.NDArray,
    r: float = 0.0,
    q: float = 0.0,
    floor: float = 1e-3,
    ceil: float = 5.0,
) -> npt.NDArray:
    """Inverts an IV surface to a local-volatility grid via Dupire.

    Args:
        s0: Spot.
        strikes: Strike axis, sorted ascending, shape `(N_K,)`.
        maturities: Maturity axis, sorted ascending and strictly positive,
            shape `(N_T,)`.
        iv_grid: Implied vols, shape `(N_K, N_T)`.
        r: Risk-free rate.
        q: Continuous dividend yield.
        floor, ceil: Local-vol clamps applied per gridpoint.

    Returns:
        Local-vol surface shape `(N_K, N_T)`. Boundary cells reuse the
        nearest interior derivative since central differences are not
        available there.
    """
    ks = np.asarray(strikes, dtype=np.float64)
    ts = np.asarray(maturities, dtype=np.float64)
    iv = np.asarray(iv_grid, dtype=np.float64)
    if iv.shape != (ks.size, ts.size):
        raise ValueError("iv_grid shape must equal (len(strikes), len(maturities)).")
    if not np.all(np.diff(ks) > 0) or not np.all(np.diff(ts) > 0):
        raise ValueError("strikes and maturities must be strictly increasing.")
    if not np.all(ts > 0):
        raise ValueError("maturities must be strictly positive.")

    c = _bs_call_grid(s0, ks, ts, iv, r, q)
    n_k = ks.size
    n_t = ts.size
    sigma_loc_sq = np.empty((n_k, n_t))

    for i in range(n_k):
        for j in range(n_t):
            ip = min(i + 1, n_k - 1)
            im = max(i - 1, 0)
            jp = min(j + 1, n_t - 1)
            jm = max(j - 1, 0)
            dk_p = ks[ip] - ks[i] if ip > i else 1.0
            dk_m = ks[i] - ks[im] if im < i else 1.0
            dt_p = ts[jp] - ts[j] if jp > j else 1.0
            dt_m = ts[j] - ts[jm] if jm < j else 1.0

            if jp == j:
                c_t = (c[i, j] - c[i, jm]) / dt_m
            elif jm == j:
                c_t = (c[i, jp] - c[i, j]) / dt_p
            else:
                c_t = (c[i, jp] - c[i, jm]) / (dt_p + dt_m)

            if ip == i:
                c_k = (c[i, j] - c[im, j]) / dk_m
            elif im == i:
                c_k = (c[ip, j] - c[i, j]) / dk_p
            else:
                c_k = (c[ip, j] - c[im, j]) / (dk_p + dk_m)

            if ip == i or im == i:
                c_kk = max(1e-12, abs(c[ip, j] - 2.0 * c[i, j] + c[im, j]) /
                           max((dk_p * dk_m), 1e-12))
            else:
                c_kk = max(1e-12,
                           (c[ip, j] - 2.0 * c[i, j] + c[im, j]) /
                           (dk_p * dk_m))
            num = c_t + (r - q) * ks[i] * c_k + q * c[i, j]
            den = 0.5 * ks[i] * ks[i] * c_kk
            val = num / max(den, 1e-12)
            sigma_loc_sq[i, j] = max(min(val, ceil * ceil), floor * floor)
    return np.sqrt(sigma_loc_sq)


def build_local_vol_process(
    s0: float,
    strikes: npt.NDArray,
    maturities: npt.NDArray,
    iv_grid: npt.NDArray,
    r: float = 0.0,
    q: float = 0.0,
    floor: float = 1e-3,
    ceil: float = 5.0,
    smoothing: Optional[float] = None,
) -> LocalVolProcess:
    """Builds a `LocalVolProcess` from a market IV surface.

    Args:
        s0: Spot.
        strikes: Strike axis.
        maturities: Maturity axis.
        iv_grid: IV surface.
        r, q: Rate and dividend yield.
        floor, ceil: Local-vol clamps.
        smoothing: Optional std-dev for a separable Gaussian smoother
            applied to the inverted grid; passes `None` to skip.

    Returns:
        Configured `LocalVolProcess` ready for Monte Carlo.
    """
    grid = dupire_local_vol_grid(
        s0, strikes, maturities, iv_grid, r, q, floor, ceil
    )
    if smoothing is not None and smoothing > 0.0:
        try:
            from scipy.ndimage import gaussian_filter

            grid = gaussian_filter(grid, sigma=smoothing, mode="nearest")
        except ImportError:
            pass
    return LocalVolProcess(s0, r, q, strikes, maturities, grid)
