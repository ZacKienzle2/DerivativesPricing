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

from models.options import VanillaOption


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
