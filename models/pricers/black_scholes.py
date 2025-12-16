# models/pricers/black_scholes.py
# Implementation of the Black-Scholes-Merton model for vanilla options.

import numpy as np
import numpy.typing as npt
from scipy.stats import norm
from typing import Tuple

from .base_pricer import BasePricer
from models.options import VanillaOption


class BlackScholesPricer(BasePricer):
    """
    Prices European vanilla options using the analytical Black-Scholes formula.
    """

    def __init__(self, option: VanillaOption):
        if not isinstance(option, VanillaOption):
            raise TypeError("BlackScholesPricer is for VanillaOption only.")
        super().__init__(option)
        self.d1, self.d2 = self._calculate_d1_d2()

    def _calculate_d1_d2(
        self,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calculates the d1 and d2 terms of the Black-Scholes formula."""
        s, k, t, r, q, sigma = (
            self.option.S,
            self.option.K,
            self.option.T,
            self.option.r,
            self.option.q,
            self.option.sigma,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = np.log(s / k) + (r - q + 0.5 * sigma**2) * t
            denominator = sigma * np.sqrt(t)
            d1 = np.where(
                denominator > 1e-9, numerator / denominator, np.inf * np.sign(s - k)
            )

        d2 = d1 - denominator
        return np.nan_to_num(d1), np.nan_to_num(d2)

    def price(self) -> Tuple[float, float]:
        """Calculates the theoretical price of the option."""
        s, k, t, r, q = (
            self.option.S,
            self.option.K,
            self.option.T,
            self.option.r,
            self.option.q,
        )

        if self.option.option_type == "call":
            price_val = s * np.exp(-q * t) * norm.cdf(self.d1) - k * np.exp(
                -r * t
            ) * norm.cdf(self.d2)
        else:
            price_val = k * np.exp(-r * t) * norm.cdf(-self.d2) - s * np.exp(
                -q * t
            ) * norm.cdf(-self.d1)
        return float(price_val), 0.0
