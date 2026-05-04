"""Risk-neutral characteristic functions for Fourier pricers.

Each kernel returns the chf of `x_T = log(S_T / S_0)` evaluated at a real
frequency `u`, split into real and imaginary parts so downstream loops can
stay in pure-real arithmetic. Internal complex math runs through numba's
`cmath` support; the Albrecher form of the Heston chf is used to avoid
branch-cut issues for long maturities.
"""

import cmath
import math

from numba import njit


@njit(cache=True, fastmath=True, inline="always")
def heston_chf_re_im(
    u: float,
    t: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    eta: float,
    rho: float,
    v0: float,
) -> tuple[float, float]:
    """Heston risk-neutral chf of `log(S_T / S_0)` (Albrecher / Schoutens form).

    Args:
        u: Real frequency.
        t: Time horizon.
        r: Risk-free rate.
        q: Dividend yield.
        kappa, theta, eta, rho, v0: Heston parameters.

    Returns:
        `(real, imag)` components of `phi(u; T)`.
    """
    iu = 1j * u
    eta2 = eta * eta
    xi = kappa - rho * eta * iu
    d = cmath.sqrt(xi * xi + eta2 * (iu + u * u))
    g2 = (xi - d) / (xi + d)
    one = 1.0 + 0j
    expdT = cmath.exp(-d * t)
    log_term = cmath.log((one - g2 * expdT) / (one - g2))
    c_term = (r - q) * iu * t + (kappa * theta / eta2) * (
        (xi - d) * t - 2.0 * log_term
    )
    d_term = (xi - d) / eta2 * (one - expdT) / (one - g2 * expdT)
    val = cmath.exp(c_term + d_term * v0)
    return val.real, val.imag


@njit(cache=True, fastmath=True, inline="always")
def bates_chf_re_im(
    u: float,
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
) -> tuple[float, float]:
    """Bates jump-diffusion chf (Heston + Merton jumps).

    Adds a compound Poisson factor `exp(lam*T*(exp(i*u*mu_J - 0.5*sigma_J^2*u^2) - 1))`
    to the Heston chf and compensates the diffusion drift by `-lam*m`, where
    `m = exp(mu_J + 0.5*sigma_J^2) - 1` is the mean jump size.
    """
    iu = 1j * u
    eta2 = eta * eta
    m = math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    xi = kappa - rho * eta * iu
    d = cmath.sqrt(xi * xi + eta2 * (iu + u * u))
    g2 = (xi - d) / (xi + d)
    one = 1.0 + 0j
    expdT = cmath.exp(-d * t)
    log_term = cmath.log((one - g2 * expdT) / (one - g2))
    c_term = (r - q - lam * m) * iu * t + (kappa * theta / eta2) * (
        (xi - d) * t - 2.0 * log_term
    )
    d_term = (xi - d) / eta2 * (one - expdT) / (one - g2 * expdT)
    jump_term = lam * t * (
        cmath.exp(iu * mu_j - 0.5 * sigma_j * sigma_j * u * u) - 1.0
    )
    val = cmath.exp(c_term + d_term * v0 + jump_term)
    return val.real, val.imag
