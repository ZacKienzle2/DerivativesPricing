"""Pathwise + likelihood-ratio Greeks for European vanilla Monte Carlo.

A single JIT prange kernel produces price together with delta, vega and
gamma in one sweep over the simulated terminal driver. Delta and vega use
the pathwise estimator (which exists because the call/put payoff is
piecewise smooth), while gamma uses the likelihood-ratio estimator
(payoff has a kink so pathwise gamma is undefined at the strike).

The estimator follows Glasserman (2003) chapter 7. Multi-step GBM collapses
to a single terminal `Z ~ N(0, 1)` because the standard normal sum is
distribution-equivalent to one draw with the same total variance.
"""

import math
from typing import Dict, Optional

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from models.options import AsianOption, VanillaOption


@njit(cache=True, parallel=True, fastmath=True, boundscheck=False)
def _asian_pathwise_greeks_jit(
    s0: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    z: npt.NDArray[np.float64],
    is_call: bool,
) -> npt.NDArray[np.float64]:
    """Per-path pathwise / LRM Greeks for arithmetic Asian under GBM.

    Pathwise estimators are valid for delta and vega because the average
    is linear in `S_0` and smooth in `sigma`. Gamma uses the LRM score on
    the first-step normal `Z_1`, the only driver that depends on `S_0`.

    Args:
        s0, k, r, q, sigma, t: Standard option parameters.
        z: Per-step normals, shape `(N, num_steps)`.
        is_call: True for call.

    Returns:
        Length-4 vector `[price, delta, gamma, vega]` (already discounted).
    """
    n = z.shape[0]
    num_steps = z.shape[1]
    dt = t / num_steps
    sqrt_dt = math.sqrt(dt)
    drift = (r - q - 0.5 * sigma * sigma) * dt
    inv_steps = 1.0 / num_steps
    df = math.exp(-r * t)

    inv_s0_sig_sq_dt = 1.0 / (s0 * s0 * sigma * sigma * dt)
    inv_s0_sq_sig_sqrt_dt = 1.0 / (s0 * s0 * sigma * sqrt_dt)

    price_sum = 0.0
    delta_sum = 0.0
    vega_sum = 0.0
    gamma_sum = 0.0

    for i in prange(n):
        log_s = math.log(s0)
        avg = 0.0
        vega_acc = 0.0
        w_t = 0.0
        for j in range(num_steps):
            z_ij = z[i, j]
            w_t += sqrt_dt * z_ij
            log_s += drift + sigma * sqrt_dt * z_ij
            s = math.exp(log_s)
            avg += s
            vega_acc += s * (-sigma * (j + 1) * dt + w_t)
        avg *= inv_steps
        vega_acc *= inv_steps

        if is_call:
            in_money = 1.0 if avg > k else 0.0
            payoff = avg - k if in_money > 0.0 else 0.0
            sign = 1.0
        else:
            in_money = 1.0 if avg < k else 0.0
            payoff = k - avg if in_money > 0.0 else 0.0
            sign = -1.0

        z_first = z[i, 0]
        price_sum += payoff
        delta_sum += sign * in_money * avg / s0
        vega_sum += sign * in_money * vega_acc
        gamma_sum += payoff * (
            z_first * z_first * inv_s0_sig_sq_dt
            - inv_s0_sig_sq_dt
            - z_first * inv_s0_sq_sig_sqrt_dt
        )

    inv_n = 1.0 / n
    out = np.empty(4)
    out[0] = df * price_sum * inv_n
    out[1] = df * delta_sum * inv_n
    out[2] = df * gamma_sum * inv_n
    out[3] = df * vega_sum * inv_n
    return out


@njit(cache=True, parallel=True, fastmath=True, boundscheck=False)
def _vanilla_pathwise_greeks_jit(
    s0: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    z_terminal: npt.NDArray[np.float64],
    is_call: bool,
) -> npt.NDArray[np.float64]:
    """Per-path pathwise/LRM Greek estimators for European vanilla.

    Args:
        s0, k, r, q, sigma, t: Standard option parameters.
        z_terminal: Per-path standardised terminal normal driver, shape `(N,)`.
        is_call: True for call.

    Returns:
        Discounted Greek estimator means as a length-4 vector
        `[price, delta, gamma, vega]`. The caller divides each by `N`.
    """
    n = z_terminal.size
    sqrt_t = math.sqrt(t)
    drift = (r - q - 0.5 * sigma * sigma) * t
    df = math.exp(-r * t)

    inv_s0_sig_sq_t = 1.0 / (s0 * s0 * sigma * sigma * t)
    inv_s0_sq_sig_sqrt_t = 1.0 / (s0 * s0 * sigma * sqrt_t)

    price_sum = 0.0
    delta_sum = 0.0
    vega_sum = 0.0
    gamma_sum = 0.0

    for i in prange(n):
        z = z_terminal[i]
        st = s0 * math.exp(drift + sigma * sqrt_t * z)
        if is_call:
            in_money = 1.0 if st > k else 0.0
            payoff = st - k if in_money > 0.0 else 0.0
            sign = 1.0
        else:
            in_money = 1.0 if st < k else 0.0
            payoff = k - st if in_money > 0.0 else 0.0
            sign = -1.0

        price_sum += payoff
        delta_sum += sign * in_money * st / s0
        vega_sum += sign * in_money * st * (-sigma * t + sqrt_t * z)
        gamma_sum += payoff * (
            z * z * inv_s0_sig_sq_t
            - inv_s0_sig_sq_t
            - z * inv_s0_sq_sig_sqrt_t
        )

    inv_n = 1.0 / n
    out = np.empty(4)
    out[0] = df * price_sum * inv_n
    out[1] = df * delta_sum * inv_n
    out[2] = df * gamma_sum * inv_n
    out[3] = df * vega_sum * inv_n
    return out


class VanillaMCGreeks:
    """Single-pass MC Greek calculator for vanilla European options.

    Args:
        option: A `VanillaOption`.
        num_sims: Number of paths.
        seed: Optional seed for the PCG64 driver.
    """

    def __init__(
        self,
        option: VanillaOption,
        num_sims: int,
        seed: Optional[int] = None,
    ):
        if not isinstance(option, VanillaOption):
            raise TypeError("VanillaMCGreeks requires a VanillaOption.")
        self.option = option
        self.num_sims = num_sims
        self._rng = np.random.default_rng(seed)

    def compute(self) -> Dict[str, float]:
        """Returns `{price, delta, gamma, vega}`."""
        opt = self.option
        z = self._rng.standard_normal(self.num_sims)
        out = _vanilla_pathwise_greeks_jit(
            opt.S, opt.K, opt.r, opt.q, opt.sigma, opt.T,
            z, opt.option_type == "call",
        )
        return {
            "price": float(out[0]),
            "delta": float(out[1]),
            "gamma": float(out[2]),
            "vega": float(out[3]),
        }


class AsianMCGreeks:
    """Pathwise / LRM MC Greeks for arithmetic-average Asian options.

    Args:
        option: An `AsianOption` with `avg_type='arithmetic'`.
        num_sims: Number of paths.
        num_steps: Sampling steps per path.
        seed: Optional PCG64 seed.
    """

    def __init__(
        self,
        option: AsianOption,
        num_sims: int,
        num_steps: int,
        seed: Optional[int] = None,
    ):
        if not isinstance(option, AsianOption):
            raise TypeError("AsianMCGreeks requires an AsianOption.")
        if option.avg_type != "arithmetic":
            raise ValueError(
                "AsianMCGreeks supports arithmetic-average Asians only."
            )
        self.option = option
        self.num_sims = num_sims
        self.num_steps = num_steps
        self._rng = np.random.default_rng(seed)

    def compute(self) -> Dict[str, float]:
        """Returns `{price, delta, gamma, vega}`."""
        opt = self.option
        z = self._rng.standard_normal((self.num_sims, self.num_steps))
        out = _asian_pathwise_greeks_jit(
            opt.S, opt.K, opt.r, opt.q, opt.sigma, opt.T,
            z, opt.option_type == "call",
        )
        return {
            "price": float(out[0]),
            "delta": float(out[1]),
            "gamma": float(out[2]),
            "vega": float(out[3]),
        }
