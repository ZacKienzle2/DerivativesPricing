# models/simulation.py

import numpy as np
import numpy.typing as npt
from numba import prange


@njit(nopython=True, fastmath=True, parallel=True)
def generate_paths_jit(
    s0: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    num_steps: int,
    z_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Generates Geometric Brownian Motion asset price paths.

    Simulates `num_sims` asset price paths over `num_steps` using the
    exact log-space discretization of the GBM SDE:
    S_{t+dt} = S_t * exp((r - q - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z)
    where Z is a standard normal random variable.

    Args:
        s0 (float): Initial asset price at t=0.
        t (float): Total time to maturity, in annualized years.
        r (float): Annualized risk-free interest rate.
        q (float): Annualized continuous dividend yield.
        sigma (float): Annualized asset volatility.
        num_steps (int): Number of discrete time steps for the simulation.
        z_matrix (npt.NDArray[np.float64]): Pre-generated standard normal
            random variables, shape (num_sims, num_steps).

    Returns:
        npt.NDArray[np.float64]: A 2D array of simulated asset paths,
            shape (num_sims, num_steps + 1), including s0.
    """
    num_sims = z_matrix.shape[0]
    dt = t / num_steps
    paths = np.zeros((num_sims, num_steps + 1))
    paths[:, 0] = s0
    for i in prange(num_sims):
        s_t = s0
        for j in range(num_steps):
            drift = (r - q - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * z_matrix[i, j]
            s_t *= np.exp(drift + diffusion)
            paths[i, j + 1] = s_t
    return paths


@njit(nopython=True, fastmath=True, parallel=True)
def generate_final_prices_jit(
    s0: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    num_steps: int,
    z_matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Simulates the terminal asset prices for a Geometric Brownian Motion.

    Calculates the asset price at time `t` for `num_sims` paths by
    discretizing the GBM SDE over `num_steps` intervals. This function
    only returns the final price for each path, not the full path history.

    Args:
        s0 (float): Initial asset price at t=0.
        t (float): Total time to maturity, in annualized years.
        r (float): Annualized risk-free interest rate.
        q (float): Annualized continuous dividend yield.
        sigma (float): Annualized asset volatility.
        num_steps (int): Number of discrete time steps for the simulation.
        z_matrix (npt.NDArray[np.float64]): Pre-generated standard normal
            random variables, shape (num_sims, num_steps).

    Returns:
        npt.NDArray[np.float64]: A 1D array of simulated terminal asset prices
            at time `t`, shape (num_sims,).
    """
    num_sims = z_matrix.shape[0]
    dt = t / num_steps
    final_prices = np.zeros(num_sims)
    for i in prange(num_sims):
        s_t = s0
        for j in range(num_steps):
            drift = (r - q - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * z_matrix[i, j]
            s_t *= np.exp(drift + diffusion)
        final_prices[i] = s_t
    return final_prices
