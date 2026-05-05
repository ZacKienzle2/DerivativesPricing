"""JIT-compiled scalar analytic kernels (Black-Scholes, Kemna-Vorst).

Pure-math `numba` kernels using `math.erf` directly. No scipy round-trip,
no array allocation. Callable from inside other JIT kernels (e.g. control
variate routines) without object-mode fallback.
"""

import math

import numpy as np
import numpy.typing as npt
from numba import njit, prange

_INV_SQRT_2 = 0.7071067811865476


@njit(cache=True, fastmath=True, inline="always")
def norm_cdf(x: float) -> float:
    """Scalar standard-normal CDF via `math.erf`."""
    return 0.5 * (1.0 + math.erf(x * _INV_SQRT_2))


@njit(cache=True, fastmath=True)
def bs_price_jit(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
) -> float:
    """Black-Scholes-Merton price for a European vanilla option.

    Returns intrinsic value when `t <= 0` or `sigma <= 0`. All other inputs
    are assumed pre-validated.
    """
    if t <= 0.0 or sigma <= 0.0:
        if is_call:
            return s - k if s > k else 0.0
        return k - s if k > s else 0.0
    sigma_sqrt_t = sigma * math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)
    if is_call:
        return s * disc_q * norm_cdf(d1) - k * disc_r * norm_cdf(d2)
    return k * disc_r * norm_cdf(-d2) - s * disc_q * norm_cdf(-d1)


@njit(cache=True, fastmath=True)
def discrete_geom_asian_price_jit(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    m: int,
    is_call: bool,
) -> float:
    """Closed-form price for a discretely-sampled geometric Asian.

    Sample times `t_i = i*T/m` for `i=1..m`. Collapses to standard BS with
    effective volatility `sigma_G^2 = sigma^2 * (m+1)(2m+1)/(6 m^2)` and
    cost-of-carry `b_G = (r-q-sigma^2/2)*(m+1)/(2m) + 0.5*sigma_G^2`.

    Args:
        s, k, t, r, q, sigma: Standard option parameters.
        m: Number of discrete sample dates.
        is_call: True for call.

    Returns:
        Risk-neutral expected discounted payoff.
    """
    if t <= 0.0 or sigma <= 0.0 or m <= 0:
        if is_call:
            return s - k if s > k else 0.0
        return k - s if k > s else 0.0
    fm = float(m)
    sigma_g2 = sigma * sigma * (fm + 1.0) * (2.0 * fm + 1.0) / (6.0 * fm * fm)
    b_g = (r - q - 0.5 * sigma * sigma) * (fm + 1.0) / (2.0 * fm) + 0.5 * sigma_g2
    sigma_g_sqrt_t = math.sqrt(sigma_g2 * t)
    d1 = (math.log(s / k) + (b_g + 0.5 * sigma_g2) * t) / sigma_g_sqrt_t
    d2 = d1 - sigma_g_sqrt_t
    forward = s * math.exp((b_g - r) * t)
    disc = math.exp(-r * t)
    if is_call:
        return forward * norm_cdf(d1) - k * disc * norm_cdf(d2)
    return k * disc * norm_cdf(-d2) - forward * norm_cdf(-d1)


@njit(cache=True, fastmath=True)
def bs_full_greeks_jit(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
) -> tuple[float, float, float, float, float, float]:
    """Computes BS price plus the standard 5-Greek vector in one pass.

    Returns:
        Tuple `(price, delta, gamma, vega, theta, rho)`.
        Vega is per unit of `sigma` (not per percent).
        Theta is per year (not per day).
        Rho is per unit of `r`.
    """
    if t <= 0.0 or sigma <= 0.0:
        intrinsic = (s - k) if is_call else (k - s)
        return (
            intrinsic if intrinsic > 0.0 else 0.0,
            1.0 if (is_call and s > k) else (-1.0 if (not is_call and s < k) else 0.0),
            0.0, 0.0, 0.0, 0.0,
        )
    sqrt_t = math.sqrt(t)
    sigma_sqrt_t = sigma * sqrt_t
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    disc_q = math.exp(-q * t)
    disc_r = math.exp(-r * t)
    n_d1 = norm_cdf(d1)
    n_d2 = norm_cdf(d2)
    pdf_d1 = math.exp(-0.5 * d1 * d1) * 0.39894228040143270  # 1/sqrt(2*pi)

    gamma = disc_q * pdf_d1 / (s * sigma_sqrt_t)
    vega = s * disc_q * pdf_d1 * sqrt_t

    if is_call:
        price = s * disc_q * n_d1 - k * disc_r * n_d2
        delta = disc_q * n_d1
        theta = (
            -s * disc_q * pdf_d1 * sigma / (2.0 * sqrt_t)
            - r * k * disc_r * n_d2
            + q * s * disc_q * n_d1
        )
        rho = k * t * disc_r * n_d2
    else:
        n_md1 = 1.0 - n_d1
        n_md2 = 1.0 - n_d2
        price = k * disc_r * n_md2 - s * disc_q * n_md1
        delta = -disc_q * n_md1
        theta = (
            -s * disc_q * pdf_d1 * sigma / (2.0 * sqrt_t)
            + r * k * disc_r * n_md2
            - q * s * disc_q * n_md1
        )
        rho = -k * t * disc_r * n_md2

    return price, delta, gamma, vega, theta, rho


@njit(cache=True, fastmath=True)
def kemna_vorst_price_jit(
    s: float,
    k: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
) -> float:
    """Kemna-Vorst closed-form price for a continuously-averaged geometric
    Asian option.

    Reduces to BS with adjusted volatility `sigma/sqrt(3)` and adjusted
    cost-of-carry `(r - q - 0.5*sigma^2)/2`.
    """
    if t <= 0.0 or sigma <= 0.0:
        if is_call:
            return s - k if s > k else 0.0
        return k - s if k > s else 0.0
    sigma_a = sigma / math.sqrt(3.0)
    mu_a = (r - q - 0.5 * sigma * sigma) / 2.0
    b_a = mu_a + r - q
    sigma_a_sqrt_t = sigma_a * math.sqrt(t)
    d1 = (math.log(s / k) + (b_a + 0.5 * sigma_a * sigma_a) * t) / sigma_a_sqrt_t
    d2 = d1 - sigma_a_sqrt_t
    forward = s * math.exp((b_a - r) * t)
    disc = math.exp(-r * t)
    if is_call:
        return forward * norm_cdf(d1) - k * disc * norm_cdf(d2)
    return k * disc * norm_cdf(-d2) - forward * norm_cdf(-d1)


@njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def bs_greek_surface_jit(
    s_arr: npt.NDArray[np.float64],
    t_arr: npt.NDArray[np.float64],
    k: float,
    r: float,
    q: float,
    sigma: float,
    is_call: bool,
    out: npt.NDArray[np.float64],
) -> None:
    """Fills `out[6, ns, nt]` with BS price + 5 Greeks across an `(S, T)` grid.

    Single parallel sweep over the spot axis. Inner time loop is sequential
    so adjacent `(s_i, t_j)` evaluations stay warm in cache. `out` must be
    pre-allocated as `(6, s_arr.size, t_arr.size)` so callers can swap layouts
    without touching this kernel.

    Output channel order matches `bs_full_greeks_jit`:
    `(price, delta, gamma, vega, theta, rho)`.

    Args:
        s_arr: Spot grid.
        t_arr: Maturity grid.
        k, r, q, sigma: Fixed contract / market parameters.
        is_call: True for call.
        out: Pre-allocated output tensor.
    """
    ns = s_arr.shape[0]
    nt = t_arr.shape[0]
    for i in prange(ns):
        s = s_arr[i]
        for j in range(nt):
            price, delta, gamma, vega, theta, rho = bs_full_greeks_jit(
                s, k, t_arr[j], r, q, sigma, is_call,
            )
            out[0, i, j] = price
            out[1, i, j] = delta
            out[2, i, j] = gamma
            out[3, i, j] = vega
            out[4, i, j] = theta
            out[5, i, j] = rho


@njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def bs_portfolio_aggregate_jit(
    s_arr: npt.NDArray[np.float64],
    k_arr: npt.NDArray[np.float64],
    t_arr: npt.NDArray[np.float64],
    r_arr: npt.NDArray[np.float64],
    q_arr: npt.NDArray[np.float64],
    sigma_arr: npt.NDArray[np.float64],
    is_call_arr: npt.NDArray[np.bool_],
    qty_arr: npt.NDArray[np.float64],
) -> tuple[float, float, float, float, float, float]:
    """Quantity-weighted Greek aggregation over a vanilla portfolio.

    Parallel reduction across positions. Each thread accumulates locally and
    the runtime fuses the partials at the `prange` join.

    Returns:
        Tuple `(price, delta, gamma, vega, theta, rho)`.
    """
    n = s_arr.shape[0]
    p_acc = 0.0
    d_acc = 0.0
    g_acc = 0.0
    v_acc = 0.0
    th_acc = 0.0
    rh_acc = 0.0
    for i in prange(n):
        price, delta, gamma, vega, theta, rho = bs_full_greeks_jit(
            s_arr[i], k_arr[i], t_arr[i], r_arr[i], q_arr[i], sigma_arr[i],
            is_call_arr[i],
        )
        q = qty_arr[i]
        p_acc += q * price
        d_acc += q * delta
        g_acc += q * gamma
        v_acc += q * vega
        th_acc += q * theta
        rh_acc += q * rho
    return p_acc, d_acc, g_acc, v_acc, th_acc, rh_acc
