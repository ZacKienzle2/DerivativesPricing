"""Rough Bergomi (rBergomi) stochastic-volatility process.

Bayer-Friz-Gatheral 2016. Uses an exact discretisation: precomputes the
Cholesky factor of the fractional Brownian motion covariance matrix once
and applies it per-path inside the JIT loop. Variance is
`v_t = xi0 * exp(eta * W^H_t - 0.5 * eta^2 * t^{2H})` for a Volterra fBM
with Hurst index `H` typically in `(0, 0.5)` (rough regime).

Memory: `O(N^2)` from the lower-triangular factor; suitable for grids up
to a few hundred steps. Fall back to the hybrid scheme for larger N.
"""

import math
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from .base import BaseProcess


def _fbm_cholesky(num_steps: int, dt: float, hurst: float) -> npt.NDArray[np.float64]:
    """Cholesky factor of the fractional Brownian motion covariance.

    Args:
        num_steps: Number of timesteps.
        dt: Step size.
        hurst: Hurst exponent in `(0, 1)`.

    Returns:
        Lower-triangular `(num_steps, num_steps)` factor `L` such that
        `cov(W^H_i, W^H_j) = (L L^T)[i, j]` for `i, j = 1..N`.
    """
    times = np.arange(1, num_steps + 1, dtype=np.float64) * dt
    h2 = 2.0 * hurst
    cov = np.empty((num_steps, num_steps))
    for i in range(num_steps):
        for j in range(num_steps):
            ti = times[i]
            tj = times[j]
            cov[i, j] = 0.5 * (ti ** h2 + tj ** h2 - abs(ti - tj) ** h2)
    cov += 1e-12 * np.eye(num_steps)
    return np.linalg.cholesky(cov)


@njit(fastmath=True, parallel=True, cache=True, boundscheck=False)
def _rbergomi_jit(
    s0: float,
    r: float,
    q: float,
    xi0: float,
    eta: float,
    rho: float,
    hurst: float,
    t: float,
    num_steps: int,
    chol: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """rBergomi spot path simulator.

    Args:
        s0: Initial spot.
        r, q: Rate and dividend yield.
        xi0: Forward variance (assumed flat across `t`).
        eta: Vol-of-vol scale.
        rho: Correlation between variance driver and spot Brownian.
        hurst: Hurst exponent.
        t: Total horizon.
        num_steps: Discretisation steps.
        chol: Pre-computed fBM Cholesky factor `(num_steps, num_steps)`.
        z: Driver normals shape `(num_sims, num_steps, 2)` for variance and
            spot-orthogonal axes.

    Returns:
        Spot paths shape `(num_sims, num_steps + 1)`.
    """
    num_sims = z.shape[0]
    dt = t / num_steps
    sqrt_dt = math.sqrt(dt)
    rho_perp = math.sqrt(1.0 - rho * rho)
    h2 = 2.0 * hurst
    times = np.empty(num_steps)
    for k in range(num_steps):
        times[k] = (k + 1) * dt

    paths = np.empty((num_sims, num_steps + 1))

    for i in prange(num_sims):
        paths[i, 0] = s0
        z_v = z[i, :, 0]
        z_s = z[i, :, 1]
        # Build correlated fBM samples via Cholesky.
        w_h = np.zeros(num_steps)
        for k in range(num_steps):
            acc = 0.0
            for j in range(k + 1):
                acc += chol[k, j] * z_v[j]
            w_h[k] = acc

        log_s = math.log(s0)
        prev_t = 0.0
        for step in range(num_steps):
            tau = times[step]
            log_v = math.log(xi0) + eta * w_h[step] - 0.5 * eta * eta * tau ** h2
            v = math.exp(log_v)
            sigma_step = math.sqrt(v)
            spot_innov = (
                rho * z_v[step] + rho_perp * z_s[step]
            )
            log_s += (r - q - 0.5 * v) * dt + sigma_step * sqrt_dt * spot_innov
            paths[i, step + 1] = math.exp(log_s)
            prev_t = tau
    return paths


class RBergomiProcess(BaseProcess):
    """Rough Bergomi dynamics with exact Cholesky discretisation.

    `dS_t / S_t = (r - q) dt + sqrt(v_t) (rho dW^H_t + sqrt(1-rho^2) dW_t^perp)`
    `v_t = xi0 * exp(eta * W^H_t - 0.5 * eta^2 * t^{2H})`
    """

    def __init__(
        self,
        s0: float,
        r: float,
        q: float,
        xi0: float,
        eta: float,
        rho: float,
        hurst: float,
    ):
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("rho must be in [-1, 1].")
        if not (0.0 < hurst < 1.0):
            raise ValueError("hurst must be in (0, 1).")
        if xi0 <= 0.0 or eta < 0.0:
            raise ValueError("xi0 must be positive, eta non-negative.")
        self._s0 = float(s0)
        self._r = float(r)
        self._q = float(q)
        self._xi0 = float(xi0)
        self._eta = float(eta)
        self._rho = float(rho)
        self._hurst = float(hurst)
        self._chol_cache: Tuple[int, float, npt.NDArray[np.float64]] = (0, 0.0, np.empty((0, 0)))

    @property
    def noise_dim(self) -> int:
        return 2

    @property
    def s0(self) -> float:
        return self._s0

    @property
    def r(self) -> float:
        return self._r

    def _ensure_chol(self, num_steps: int, dt: float) -> npt.NDArray[np.float64]:
        n_cached, dt_cached, chol = self._chol_cache
        if n_cached == num_steps and abs(dt_cached - dt) < 1e-15:
            return chol
        chol = _fbm_cholesky(num_steps, dt, self._hurst)
        self._chol_cache = (num_steps, dt, chol)
        return chol

    def simulate_paths(
        self,
        num_sims: int,
        num_steps: int,
        t: float,
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Simulates rBergomi spot paths."""
        if z.ndim != 3 or z.shape[2] != 2:
            raise ValueError("RBergomi requires z of shape (num_sims, num_steps, 2).")
        dt = t / num_steps
        chol = self._ensure_chol(num_steps, dt)
        return _rbergomi_jit(
            self._s0, self._r, self._q,
            self._xi0, self._eta, self._rho, self._hurst,
            t, num_steps, chol, z,
        )

    @property
    def signature(self) -> Tuple:
        return (
            type(self).__name__,
            self._s0, self._r, self._q,
            self._xi0, self._eta, self._rho, self._hurst,
        )
