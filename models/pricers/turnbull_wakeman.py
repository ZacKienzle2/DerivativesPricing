"""Turnbull-Wakeman 1991 moment-matched arithmetic Asian pricer."""

from typing import Tuple

import numpy as np
from scipy.stats import norm

from models.options import AsianOption
from .base_pricer import BasePricer


class TurnbullWakemanPricer(BasePricer):
    """
    Prices arithmetic Asian options using the Turnbull-Wakeman (1991)
    moment-matching approximation.
    """

    def __init__(self, option: AsianOption):
        if not (isinstance(option, AsianOption) and option.avg_type == "arithmetic"):
            raise TypeError("This pricer is for arithmetic Asian options.")
        super().__init__(option)

    def _calculate_moments(self) -> Tuple[float, float]:
        """Calculates the first two moments of the asset price's arithmetic average."""
        s, t, r, q, sigma = (
            self.option.S,
            self.option.T,
            self.option.r,
            self.option.q,
            self.option.sigma,
        )
        b = r - q

        if np.isclose(b, 0.0):
            m1 = s
        else:
            m1 = s * (np.exp(b * t) - 1) / (b * t)

        b_plus_sigma_sq = b + sigma**2
        two_b_plus_sigma_sq = 2 * b + sigma**2

        if any(
            np.isclose(val, 0.0) for val in [b_plus_sigma_sq, two_b_plus_sigma_sq, b, t]
        ):
            return m1, np.nan

        term1 = (2 * s**2 * np.exp(two_b_plus_sigma_sq * t)) / (
            b_plus_sigma_sq * two_b_plus_sigma_sq * (t**2)
        )
        term2 = (2 * s**2 / (b * t**2)) * (
            1 / two_b_plus_sigma_sq - (np.exp(b * t) / b_plus_sigma_sq)
        )

        m2 = term1 + term2
        return m1, m2

    def price(self) -> Tuple[float, float]:
        """Calculates the approximate option price."""
        k, t, r = self.option.K, self.option.T, self.option.r
        m1, m2 = self._calculate_moments()

        if np.isnan(m2) or m1**2 <= 0 or m2 <= 0:
            return np.nan, 0.0

        try:
            variance_approx = np.log(m2 / m1**2)
        except (ValueError, ZeroDivisionError):
            return np.nan, 0.0

        if variance_approx < 0:
            return np.nan, 0.0

        sigma_approx = np.sqrt(variance_approx / t)

        if np.isclose(sigma_approx, 0.0):
            payoff = m1 - k if self.option.option_type == "call" else k - m1
            return np.exp(-r * t) * max(0, payoff), 0.0

        d1 = (np.log(m1 / k) + 0.5 * variance_approx) / (sigma_approx * np.sqrt(t))
        d2 = d1 - sigma_approx * np.sqrt(t)

        if self.option.option_type == "call":
            price = np.exp(-r * t) * (m1 * norm.cdf(d1) - k * norm.cdf(d2))
        else:
            price = np.exp(-r * t) * (k * norm.cdf(-d2) - m1 * norm.cdf(-d1))

        return float(price), 0.0
