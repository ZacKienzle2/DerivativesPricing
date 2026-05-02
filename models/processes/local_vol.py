"""Local-volatility (Dupire) process with bilinear interpolation."""

import math
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from .base import BaseProcess


@njit(cache=True, fastmath=True, inline="always", boundscheck=False)
def _bilinear_lookup(
    s: float,
    t: float,
    grid_s: npt.NDArray[np.float64],
    grid_t: npt.NDArray[np.float64],
    grid_v: npt.NDArray[np.float64],
) -> float:
    """Bilinear interpolation of `grid_v[i, j]` at `(s, t)`.

    Out-of-range queries clamp to the boundary value (flat extrapolation).
    """
    n_s = grid_s.size
    n_t = grid_t.size
    if s <= grid_s[0]:
        i0 = 0
        i1 = 0
        ws = 0.0
    elif s >= grid_s[n_s - 1]:
        i0 = n_s - 1
        i1 = n_s - 1
        ws = 0.0
    else:
        i0 = 0
        for k in range(n_s - 1):
            if grid_s[k + 1] > s:
                i0 = k
                break
        i1 = i0 + 1
        ws = (s - grid_s[i0]) / (grid_s[i1] - grid_s[i0])

    if t <= grid_t[0]:
        j0 = 0
        j1 = 0
        wt = 0.0
    elif t >= grid_t[n_t - 1]:
        j0 = n_t - 1
        j1 = n_t - 1
        wt = 0.0
    else:
        j0 = 0
        for k in range(n_t - 1):
            if grid_t[k + 1] > t:
                j0 = k
                break
        j1 = j0 + 1
        wt = (t - grid_t[j0]) / (grid_t[j1] - grid_t[j0])

    v00 = grid_v[i0, j0]
    v01 = grid_v[i0, j1]
    v10 = grid_v[i1, j0]
    v11 = grid_v[i1, j1]
    return (
        v00 * (1.0 - ws) * (1.0 - wt)
        + v10 * ws * (1.0 - wt)
        + v01 * (1.0 - ws) * wt
        + v11 * ws * wt
    )


@njit(fastmath=True, parallel=True, cache=True, boundscheck=False)
def _local_vol_jit(
    s0: float,
    r: float,
    q: float,
    t: float,
    num_steps: int,
    grid_s: npt.NDArray[np.float64],
    grid_t: npt.NDArray[np.float64],
    grid_v: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Log-Euler simulation with state-dependent volatility from a 2D grid."""
    num_sims = z.shape[0]
    dt = t / num_steps
    sqrt_dt = math.sqrt(dt)
    rq = r - q
    paths = np.empty((num_sims, num_steps + 1))
    for i in prange(num_sims):
        paths[i, 0] = s0
        log_s = math.log(s0)
        for j in range(num_steps):
            s = math.exp(log_s)
            tau = j * dt
            sigma = _bilinear_lookup(s, tau, grid_s, grid_t, grid_v)
            log_s += (rq - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z[i, j]
            paths[i, j + 1] = math.exp(log_s)
    return paths


class LocalVolProcess(BaseProcess):
    """Dupire local-volatility dynamics.

    `dS_t/S_t = (r - q) dt + sigma_loc(S_t, t) dW_t`
    where `sigma_loc` is supplied as a 2D grid `(grid_s, grid_t, grid_v)`
    interpolated bilinearly. Build the grid externally from a Dupire-
    inverted IV surface or any other source of `sigma_loc(S, t)`.
    """

    def __init__(
        self,
        s0: float,
        r: float,
        q: float,
        grid_s: npt.NDArray[np.float64],
        grid_t: npt.NDArray[np.float64],
        grid_v: npt.NDArray[np.float64],
    ):
        grid_s = np.asarray(grid_s, dtype=np.float64)
        grid_t = np.asarray(grid_t, dtype=np.float64)
        grid_v = np.asarray(grid_v, dtype=np.float64)
        if grid_v.shape != (grid_s.size, grid_t.size):
            raise ValueError(
                "grid_v shape must equal (len(grid_s), len(grid_t))."
            )
        if not np.all(np.diff(grid_s) > 0) or not np.all(np.diff(grid_t) > 0):
            raise ValueError("grid_s and grid_t must be strictly increasing.")
        self._s0 = float(s0)
        self._r = float(r)
        self._q = float(q)
        self._grid_s = grid_s
        self._grid_t = grid_t
        self._grid_v = grid_v

    @classmethod
    def constant_vol(
        cls, s0: float, r: float, q: float, sigma: float
    ) -> "LocalVolProcess":
        """Builds a constant-vol degenerate local-vol process for testing."""
        gs = np.array([0.0, 1e6])
        gt = np.array([0.0, 1e3])
        gv = np.full((2, 2), sigma)
        return cls(s0, r, q, gs, gt, gv)

    @property
    def noise_dim(self) -> int:
        return 1

    @property
    def s0(self) -> float:
        return self._s0

    @property
    def r(self) -> float:
        return self._r

    def simulate_paths(
        self,
        num_sims: int,
        num_steps: int,
        t: float,
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Simulates spot paths under the local-vol surface."""
        return _local_vol_jit(
            self._s0, self._r, self._q,
            t, num_steps,
            self._grid_s, self._grid_t, self._grid_v,
            z,
        )

    @property
    def signature(self) -> Tuple:
        return (
            type(self).__name__, self._s0, self._r, self._q,
            self._grid_s.tobytes(), self._grid_t.tobytes(), self._grid_v.tobytes(),
        )
