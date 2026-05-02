"""Geometric Brownian Motion path generation kernels.

Provides JIT-compiled, thread-parallel path generators and a vectorised
terminal-price routine. Loop invariants are hoisted out of the inner step
loop and arithmetic is performed in log-space to avoid repeated `exp` round
trips.
"""

import numpy as np
import numpy.typing as npt
from numba import njit, prange


@njit(fastmath=True, parallel=True, cache=True, boundscheck=False)
def generate_paths_jit(
    s0: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    num_steps: int,
    z_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Simulates Geometric Brownian Motion asset paths.

    Uses the exact log-space discretisation
    `S_{t+dt} = S_t * exp((r - q - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z)`,
    accumulating in log-space within each path so the per-step cost is one
    add and one exp. Outer simulation dimension is parallelised via `prange`.

    Args:
        s0: Initial asset price at t=0.
        t: Total time to maturity (annualised years).
        r: Annualised risk-free rate.
        q: Annualised continuous dividend yield.
        sigma: Annualised volatility.
        num_steps: Number of discrete time steps.
        z_matrix: Pre-generated standard normals, shape (num_sims, num_steps).

    Returns:
        2D array of paths, shape (num_sims, num_steps + 1), including s0.
    """
    num_sims = z_matrix.shape[0]
    dt = t / num_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    log_s0 = np.log(s0)
    paths = np.empty((num_sims, num_steps + 1))
    for i in prange(num_sims):
        paths[i, 0] = s0
        log_s = log_s0
        for j in range(num_steps):
            log_s += drift + vol * z_matrix[i, j]
            paths[i, j + 1] = np.exp(log_s)
    return paths


def generate_final_prices_jit(
    s0: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    num_steps: int,
    z_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Simulates terminal asset prices for Geometric Brownian Motion.

    Uses the closed-form sum of log-increments
    `S_T = S_0 * exp(num_steps*drift + sigma*sqrt(dt) * sum_j Z_{ij})`.
    Single BLAS-vectorised reduction; no Python loop, no JIT compile cost.

    Args:
        s0: Initial asset price at t=0.
        t: Total time to maturity (annualised years).
        r: Annualised risk-free rate.
        q: Annualised continuous dividend yield.
        sigma: Annualised volatility.
        num_steps: Number of discrete time steps.
        z_matrix: Pre-generated standard normals, shape (num_sims, num_steps).

    Returns:
        1D array of terminal prices at time `t`, shape (num_sims,).
    """
    dt = t / num_steps
    drift_total = (r - q - 0.5 * sigma * sigma) * t
    vol = sigma * np.sqrt(dt)
    return s0 * np.exp(drift_total + vol * z_matrix.sum(axis=1))
