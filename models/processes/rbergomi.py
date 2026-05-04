"""Rough Bergomi (rBergomi) stochastic-volatility process.

Bayer-Friz-Gatheral 2016 dynamics with the Bennedsen-Lunde-Pakkanen 2017
(BLP) hybrid scheme at `kappa = 1`: the nearest interval is integrated
exactly via a bivariate Gaussian whose covariance is computed in closed
form, while the far tail is built by a discrete convolution of i.i.d.
Brownian increments against optimally moment-matched weights. The
convolution dispatches through `numpy.fft.rfft`, dropping path generation
from `O(N^2)` Cholesky to `O(N log N)` per sample, and freeing the
`O(N^2)` lower-triangular factor that constrained the original
implementation.

Variance is `v_t = xi0 * exp(eta * W^H_t - 0.5 * eta^2 * t^{2H})` for a
Volterra fractional Brownian motion `W^H` with Hurst index `H` in the
rough regime `(0, 0.5)`. For `H == 0.5` the kernel degenerates to a
classical Brownian motion and we fall back to `cumsum` of the increments.
"""

import math

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from .base import BaseProcess
from .registry import autoregister

_HURST_BM_TOL = 1e-9


def _hybrid_far_kernel(
    num_steps: int, dt: float, hurst: float
) -> npt.NDArray[np.float64]:
    """Optimal moment-matched far-interval weights for the BLP hybrid.

    Returns the kernel `K` such that `W^H_arr[i] = Y2[i] + sum_{lag=1..i}
    K[lag] * dW[i - lag]`, with `K[0] = 0`. Element `K[j]` corresponds to
    the constant approximation of `(b_{j+1} * dt)^{H - 1/2}` over an
    interval of width `dt` placed `j` steps back from the current time.

    Args:
        num_steps: Number of timesteps `N`.
        dt: Step width.
        hurst: Hurst exponent in `(0, 0.5) cup (0.5, 1)`.

    Returns:
        1D array of length `num_steps`.
    """
    kernel = np.zeros(num_steps, dtype=np.float64)
    if num_steps < 2:
        return kernel
    h2 = 2.0 * hurst
    alpha = hurst - 0.5
    j = np.arange(2, num_steps + 1, dtype=np.float64)
    inc = (j ** h2 - (j - 1.0) ** h2) / h2
    b = inc ** (1.0 / (h2 - 1.0))
    kernel[1:] = (b * dt) ** alpha
    return kernel


def _convolve_far(
    kernel: npt.NDArray[np.float64], dW: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Per-path causal linear convolution via real FFT.

    Args:
        kernel: 1D kernel of length `N`.
        dW: Brownian increments shape `(num_sims, N)`.

    Returns:
        Truncated convolution shape `(num_sims, N)` such that
        `C[s, i] = sum_{lag=0..i} kernel[lag] * dW[s, i - lag]`.
    """
    n = dW.shape[1]
    fft_len = 1
    while fft_len < 2 * n:
        fft_len <<= 1
    f_k = np.fft.rfft(kernel, n=fft_len)
    f_u = np.fft.rfft(dW, n=fft_len, axis=1)
    return np.fft.irfft(f_k * f_u, n=fft_len, axis=1)[:, :n]


def _exact_near(
    z_v: npt.NDArray[np.float64],
    y_perp: npt.NDArray[np.float64],
    dt: float,
    hurst: float,
) -> npt.NDArray[np.float64]:
    """Exact bivariate sample of the near-interval Volterra integral.

    Builds `Y2[s, i]` correlated with `dW_v[s, i] = sqrt(dt) * z_v[s, i]`
    so that `(dW_v, Y2)` matches the joint covariance of the increment and
    its kernel-weighted Volterra integral over `[(i) dt, (i + 1) dt]`.

    Args:
        z_v: Standard normals shape `(num_sims, num_steps)` driving variance.
        y_perp: Independent standard normals shape `(num_sims, num_steps)`.
        dt: Step width.
        hurst: Hurst exponent.

    Returns:
        Near-interval samples shape `(num_sims, num_steps)`.
    """
    cov = dt ** (hurst + 0.5) / (hurst + 0.5)
    var_y2 = dt ** (2.0 * hurst) / (2.0 * hurst)
    a = cov / math.sqrt(dt)
    b = math.sqrt(max(var_y2 - a * a, 0.0))
    return a * z_v + b * y_perp


@njit(fastmath=True, parallel=True, cache=True, boundscheck=False)
def _rbergomi_hybrid_jit(
    s0: float,
    r: float,
    q: float,
    xi0: float,
    eta: float,
    rho: float,
    hurst: float,
    t: float,
    num_steps: int,
    z_v: npt.NDArray[np.float64],
    z_spot: npt.NDArray[np.float64],
    w_h: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """rBergomi spot evolution given a precomputed Volterra fBM trajectory.

    Args:
        s0: Initial spot.
        r: Risk-free rate.
        q: Continuous dividend yield.
        xi0: Forward variance (assumed flat across `t`).
        eta: Vol-of-vol scale.
        rho: Correlation between variance driver and spot Brownian.
        hurst: Hurst exponent.
        t: Total horizon.
        num_steps: Discretisation steps `N`.
        z_v: Standard normals `(num_sims, N)` driving `dW_v`.
        z_spot: Standard normals `(num_sims, N)` for the spot orthogonal axis.
        w_h: Volterra fBM samples at times `dt..N dt`, shape `(num_sims, N)`.

    Returns:
        Spot paths shape `(num_sims, N + 1)`.
    """
    num_sims = z_v.shape[0]
    dt = t / num_steps
    rho_perp = math.sqrt(1.0 - rho * rho)
    h2 = 2.0 * hurst
    log_xi0 = math.log(xi0)
    eta_sq_half = 0.5 * eta * eta

    paths = np.empty((num_sims, num_steps + 1))
    for i in prange(num_sims):
        log_s = math.log(s0)
        paths[i, 0] = s0
        for k in range(num_steps):
            if k == 0:
                v = xi0
            else:
                tau_prev = k * dt
                log_v = (
                    log_xi0
                    + eta * w_h[i, k - 1]
                    - eta_sq_half * tau_prev ** h2
                )
                v = math.exp(log_v)
            spot_innov = rho * z_v[i, k] + rho_perp * z_spot[i, k]
            log_s += (r - q - 0.5 * v) * dt + math.sqrt(v * dt) * spot_innov
            paths[i, k + 1] = math.exp(log_s)
    return paths


@autoregister("RBergomi", noise_dim=3)
class RBergomiProcess(BaseProcess):
    """Rough Bergomi dynamics with the BLP hybrid scheme.

    `dS_t / S_t = (r - q) dt + sqrt(v_t) (rho dW_v + sqrt(1 - rho^2) dW^perp)`
    `v_t = xi0 * exp(eta * W^H_t - 0.5 * eta^2 * t^{2H})`

    The driver normals are stacked as `z[..., 0] = z_v` (variance), `z[...,
    1] = y_perp` (orthogonal noise for the bivariate near integral) and
    `z[..., 2] = z_spot` (orthogonal noise for the spot Brownian).
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
        self._kernel_cache: tuple[
            int, float, npt.NDArray[np.float64]
        ] = (0, 0.0, np.empty(0))

    @property
    def noise_dim(self) -> int:
        return 3

    @property
    def s0(self) -> float:
        return self._s0

    @property
    def r(self) -> float:
        return self._r

    def _kernel_for(self, num_steps: int, dt: float) -> npt.NDArray[np.float64]:
        """Returns a cached far-interval weight kernel."""
        n_cached, dt_cached, kernel = self._kernel_cache
        if n_cached == num_steps and abs(dt_cached - dt) < 1e-15:
            return kernel
        kernel = _hybrid_far_kernel(num_steps, dt, self._hurst)
        self._kernel_cache = (num_steps, dt, kernel)
        return kernel

    def _volterra_fbm(
        self,
        z_v: npt.NDArray[np.float64],
        y_perp: npt.NDArray[np.float64],
        dt: float,
    ) -> npt.NDArray[np.float64]:
        """Builds the Riemann-Liouville Volterra fBM trajectory.

        Returned process is scaled by `sqrt(2H)` so that
        `var(W^H_t) = t^{2H}`, matching the rBergomi convention used by
        the variance martingale correction `-0.5 eta^2 t^{2H}`. For
        `H == 0.5` the kernel collapses to the indicator on `[0, t]` and
        the result reduces to standard Brownian motion.
        """
        if abs(self._hurst - 0.5) < _HURST_BM_TOL:
            return math.sqrt(dt) * np.cumsum(z_v, axis=1)
        scale = math.sqrt(2.0 * self._hurst)
        near = _exact_near(z_v, y_perp, dt, self._hurst)
        kernel = self._kernel_for(z_v.shape[1], dt)
        d_w = math.sqrt(dt) * z_v
        far = _convolve_far(kernel, d_w)
        return scale * (near + far)

    def simulate_paths(
        self,
        num_sims: int,
        num_steps: int,
        t: float,
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Simulates rBergomi spot paths."""
        if z.ndim != 3 or z.shape[2] != 3:
            raise ValueError(
                "RBergomi requires z of shape (num_sims, num_steps, 3)."
            )
        dt = t / num_steps
        z_v = np.ascontiguousarray(z[:, :, 0])
        y_perp = np.ascontiguousarray(z[:, :, 1])
        z_spot = np.ascontiguousarray(z[:, :, 2])
        w_h = self._volterra_fbm(z_v, y_perp, dt)
        return _rbergomi_hybrid_jit(
            self._s0, self._r, self._q,
            self._xi0, self._eta, self._rho, self._hurst,
            t, num_steps, z_v, z_spot, w_h,
        )

    @property
    def signature(self) -> tuple:
        return (
            type(self).__name__,
            self._s0, self._r, self._q,
            self._xi0, self._eta, self._rho, self._hurst,
        )
