"""Fang-Oosterlee 2008 COS Fourier pricer.

European call/put prices for any model whose chf is exposed in `_chf`. The
JIT inner loop sums `N` cosine basis terms, each computed scalar-by-scalar
to keep the working set in registers. Cumulant-based truncation (`L`
standard deviations either side of the mean) bounds the integration domain.

Memory: O(1) per call. Wall-clock: sub-millisecond at `N = 128`.
"""

import math
from typing import Tuple

from numba import njit

from ._chf import bates_chf_re_im, heston_chf_re_im


@njit(cache=True, fastmath=True, inline="always")
def _cumulants_heston(
    t: float, r: float, q: float, kappa: float, theta: float, v0: float
) -> Tuple[float, float]:
    """Approximate first two cumulants of `log(S_T / S_0)` under Heston."""
    e_kt = math.exp(-kappa * t)
    int_var = v0 * (1.0 - e_kt) / kappa + theta * (t - (1.0 - e_kt) / kappa)
    c1 = (r - q) * t - 0.5 * int_var
    c2 = int_var
    return c1, c2


@njit(cache=True, fastmath=True, inline="always")
def _cumulants_bates(
    t: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    v0: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
) -> Tuple[float, float]:
    """First two cumulants for Bates: Heston + jump contributions."""
    c1_h, c2_h = _cumulants_heston(t, r, q, kappa, theta, v0)
    m = math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    c1 = c1_h + lam * t * mu_j - lam * m * t
    c2 = c2_h + lam * t * (mu_j * mu_j + sigma_j * sigma_j)
    return c1, c2


@njit(cache=True, fastmath=True, inline="always")
def _payoff_coeff(
    kk: int,
    omega_k: float,
    a: float,
    b: float,
    c_lo: float,
    c_hi: float,
    is_call: bool,
    k_strike: float,
) -> float:
    """Computes the K-scaled payoff cosine coefficient `V_k`."""
    bma = b - a
    e_lo = math.exp(c_lo)
    e_hi = math.exp(c_hi)
    if kk == 0:
        chi_k = e_hi - e_lo
        psi_k = c_hi - c_lo
    else:
        ang_hi = omega_k * (c_hi - a)
        ang_lo = omega_k * (c_lo - a)
        denom = 1.0 + omega_k * omega_k
        chi_k = (
            math.cos(ang_hi) * e_hi
            - math.cos(ang_lo) * e_lo
            + omega_k * math.sin(ang_hi) * e_hi
            - omega_k * math.sin(ang_lo) * e_lo
        ) / denom
        psi_k = (math.sin(ang_hi) - math.sin(ang_lo)) / omega_k
    scale = 2.0 * k_strike / bma
    if is_call:
        return scale * (chi_k - psi_k)
    return scale * (-chi_k + psi_k)


@njit(cache=True, fastmath=True)
def cos_heston_price_jit(
    s0: float,
    k_strike: float,
    t: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    eta: float,
    rho: float,
    v0: float,
    is_call: bool,
    n_terms: int,
    l_truncate: float,
) -> float:
    """Heston European price via the COS expansion."""
    c1, c2 = _cumulants_heston(t, r, q, kappa, theta, v0)
    width = l_truncate * math.sqrt(abs(c2))
    a = c1 - width
    b = c1 + width
    bma = b - a
    x0 = math.log(s0 / k_strike)

    if is_call:
        c_lo, c_hi = 0.0, b
    else:
        c_lo, c_hi = a, 0.0

    total = 0.0
    for kk in range(n_terms):
        omega_k = kk * math.pi / bma
        if kk == 0:
            chf_re, chf_im = 1.0, 0.0
        else:
            chf_re, chf_im = heston_chf_re_im(
                omega_k, t, r, q, kappa, theta, eta, rho, v0
            )
        v_k = _payoff_coeff(kk, omega_k, a, b, c_lo, c_hi, is_call, k_strike)
        rot = omega_k * (x0 - a)
        re = chf_re * math.cos(rot) - chf_im * math.sin(rot)
        if kk == 0:
            re *= 0.5
        total += re * v_k
    return math.exp(-r * t) * total


@njit(cache=True, fastmath=True)
def cos_bates_price_jit(
    s0: float,
    k_strike: float,
    t: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    eta: float,
    rho: float,
    v0: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    is_call: bool,
    n_terms: int,
    l_truncate: float,
) -> float:
    """Bates European price via the COS expansion."""
    c1, c2 = _cumulants_bates(t, r, q, kappa, theta, v0, lam, mu_j, sigma_j)
    width = l_truncate * math.sqrt(abs(c2))
    a = c1 - width
    b = c1 + width
    bma = b - a
    x0 = math.log(s0 / k_strike)

    if is_call:
        c_lo, c_hi = 0.0, b
    else:
        c_lo, c_hi = a, 0.0

    total = 0.0
    for kk in range(n_terms):
        omega_k = kk * math.pi / bma
        if kk == 0:
            chf_re, chf_im = 1.0, 0.0
        else:
            chf_re, chf_im = bates_chf_re_im(
                omega_k, t, r, q, kappa, theta, eta, rho, v0,
                lam, mu_j, sigma_j,
            )
        v_k = _payoff_coeff(kk, omega_k, a, b, c_lo, c_hi, is_call, k_strike)
        rot = omega_k * (x0 - a)
        re = chf_re * math.cos(rot) - chf_im * math.sin(rot)
        if kk == 0:
            re *= 0.5
        total += re * v_k
    return math.exp(-r * t) * total
