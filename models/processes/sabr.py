"""SABR stochastic-volatility process: Hagan analytic IV + path simulator."""

import math
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from .base import BaseProcess
from .registry import autoregister


@njit(cache=True, fastmath=True)
def hagan_lognormal_vol_jit(
    f: float,
    k: float,
    t: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Hagan 2002 lognormal SABR implied volatility.

    Args:
        f: Forward price of the underlying.
        k: Strike.
        t: Time to expiry.
        alpha: Initial volatility (`sigma_0`).
        beta: CEV exponent in `[0, 1]`.
        rho: Correlation between forward and vol.
        nu: Volatility of volatility.

    Returns:
        Black-implied volatility.
    """
    if f <= 0.0 or k <= 0.0 or t <= 0.0 or alpha <= 0.0:
        return 0.0
    one_minus_beta = 1.0 - beta
    fk = f * k
    fk_pow = fk ** (one_minus_beta * 0.5)
    log_fk = math.log(f / k)

    if abs(f - k) < 1e-12:
        a_term = (
            ((one_minus_beta * one_minus_beta) / 24.0)
            * (alpha * alpha)
            / (fk_pow * fk_pow)
        )
        b_term = 0.25 * rho * beta * nu * alpha / fk_pow
        c_term = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
        return alpha / fk_pow * (1.0 + (a_term + b_term + c_term) * t)

    z = (nu / alpha) * fk_pow * log_fk
    denom = math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho
    x_z = math.log(denom / (1.0 - rho)) if denom > 0.0 and (1.0 - rho) > 0.0 else 0.0
    if abs(x_z) < 1e-14:
        x_ratio = 1.0
    else:
        x_ratio = z / x_z

    log_fk_sq = log_fk * log_fk
    a_term = (
        ((one_minus_beta * one_minus_beta) / 24.0) * (alpha * alpha) / (fk_pow * fk_pow)
    )
    b_term = 0.25 * rho * beta * nu * alpha / fk_pow
    c_term = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    numerator = alpha * x_ratio
    denominator = (
        fk_pow
        * (
            1.0
            + (one_minus_beta * one_minus_beta / 24.0) * log_fk_sq
            + (one_minus_beta ** 4 / 1920.0) * log_fk_sq * log_fk_sq
        )
    )
    return numerator / denominator * (1.0 + (a_term + b_term + c_term) * t)


@njit(cache=True, fastmath=True, inline="always")
def _black_price(
    f: float, k: float, t: float, discount: float, sigma: float, is_call: bool
) -> float:
    """Black 1976 forward-pricing formula."""
    sigma_sqrt_t = sigma * math.sqrt(t)
    d1 = (math.log(f / k) + 0.5 * sigma * sigma * t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    n1 = 0.5 * (1.0 + math.erf(d1 * 0.7071067811865476))
    n2 = 0.5 * (1.0 + math.erf(d2 * 0.7071067811865476))
    if is_call:
        return discount * (f * n1 - k * n2)
    return discount * (k * (1.0 - n2) - f * (1.0 - n1))


@njit(cache=True, fastmath=True)
def sabr_price_jit(
    f: float,
    k: float,
    t: float,
    discount: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    is_call: bool,
) -> float:
    """SABR European price: Hagan IV fed into the Black 1976 forward formula."""
    iv = hagan_lognormal_vol_jit(f, k, t, alpha, beta, rho, nu)
    if iv <= 0.0:
        intrinsic = (f - k) if is_call else (k - f)
        return discount * (intrinsic if intrinsic > 0.0 else 0.0)
    return _black_price(f, k, t, discount, iv, is_call)


@njit(fastmath=True, parallel=True, cache=True, boundscheck=False)
def _sabr_euler_jit(
    f0: float,
    alpha0: float,
    beta: float,
    rho: float,
    nu: float,
    t: float,
    num_steps: int,
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Log-Euler simulation of SABR forward and stochastic vol.

    Use only for Monte Carlo path-dependent payoffs; for European pricing the
    Hagan analytic formula is faster and more accurate.
    """
    num_sims = z.shape[0]
    dt = t / num_steps
    sqrt_dt = math.sqrt(dt)
    sqrt_one_minus = math.sqrt(1.0 - rho * rho)
    paths = np.empty((num_sims, num_steps + 1))

    for i in prange(num_sims):
        paths[i, 0] = f0
        f = f0
        a = alpha0
        for j in range(num_steps):
            zf = z[i, j, 0]
            zv = z[i, j, 1]
            dw_f = sqrt_dt * (rho * zv + sqrt_one_minus * zf)
            dw_v = sqrt_dt * zv
            if f > 0.0:
                f = f + a * (f ** beta) * dw_f
                if f < 0.0:
                    f = 0.0
            a = a * math.exp(-0.5 * nu * nu * dt + nu * dw_v)
            paths[i, j + 1] = f
    return paths


@autoregister("SABR", noise_dim=2)
class SABRProcess(BaseProcess):
    """SABR forward-rate dynamics with Hagan analytic IV available."""

    def __init__(
        self,
        f0: float,
        alpha: float,
        beta: float,
        rho: float,
        nu: float,
        r: float = 0.0,
    ):
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("rho must be in [-1, 1].")
        if not (0.0 <= beta <= 1.0):
            raise ValueError("beta must be in [0, 1].")
        if alpha <= 0.0 or nu < 0.0:
            raise ValueError("alpha must be positive, nu non-negative.")
        self._f0 = float(f0)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._rho = float(rho)
        self._nu = float(nu)
        self._r = float(r)

    @property
    def noise_dim(self) -> int:
        return 2

    @property
    def s0(self) -> float:
        return self._f0

    @property
    def r(self) -> float:
        return self._r

    def implied_vol(self, k: float, t: float) -> float:
        """Returns the Hagan lognormal implied vol at strike `k`, expiry `t`."""
        return float(
            hagan_lognormal_vol_jit(
                self._f0, k, t, self._alpha, self._beta, self._rho, self._nu
            )
        )

    def european_price(
        self, k: float, t: float, is_call: bool, discount: float = 1.0
    ) -> float:
        """Returns the Hagan-fed Black price for a European vanilla."""
        return float(
            sabr_price_jit(
                self._f0, k, t, discount,
                self._alpha, self._beta, self._rho, self._nu, is_call,
            )
        )

    def simulate_paths(
        self,
        num_sims: int,
        num_steps: int,
        t: float,
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Simulates SABR forward paths via log-Euler."""
        if z.ndim != 3 or z.shape[2] != 2:
            raise ValueError("SABR requires z of shape (num_sims, num_steps, 2).")
        return _sabr_euler_jit(
            self._f0, self._alpha, self._beta, self._rho, self._nu,
            t, num_steps, z,
        )

    @property
    def signature(self) -> Tuple:
        return (
            type(self).__name__,
            self._f0, self._alpha, self._beta, self._rho, self._nu, self._r,
        )

    @property
    def supports_analytic_european(self) -> bool:
        return True

    def analytic_european_price(
        self, k: float, t: float, is_call: bool
    ) -> float:
        """Hagan-fed Black 1976 European price under SABR."""
        return self.european_price(k, t, is_call, discount=math.exp(-self._r * t))
