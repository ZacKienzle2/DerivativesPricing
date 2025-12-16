# models/pricers/kemna_vorst.py
from typing import Tuple
import numpy as np
from scipy.stats import norm

from models.options import AsianOption
from .base_pricer import BasePricer


class KemnaVorstPricer(BasePricer):
    """
    Prices European Asian options on the GEOMETRIC average using the
    Kemna-Vorst (1990) exact closed-form solution.
    """

    def __init__(self, option: AsianOption):
        if option.avg_type != "geometric":
            raise TypeError("KemnaVorstPricer is only for GEOMETRIC Asian options.")
        super().__init__(option)

    def price(self) -> Tuple[float, float]:
        """Calculates the price using the Kemna-Vorst formula."""
        s, k, t, r, q, sigma = (
            self.option.S,
            self.option.K,
            self.option.T,
            self.option.r,
            self.option.q,
            self.option.sigma,
        )
        is_call = self.option.option_type == "call"

        sigma_a = sigma / np.sqrt(3)
        mu_a = (r - q - 0.5 * sigma**2) / 2
        b_a = mu_a + r - q

        d1 = (np.log(s / k) + (b_a + 0.5 * sigma_a**2) * t) / (sigma_a * np.sqrt(t))
        d2 = d1 - sigma_a * np.sqrt(t)

        if is_call:
            price = s * np.exp((b_a - r) * t) * norm.cdf(d1) - k * np.exp(
                -r * t
            ) * norm.cdf(d2)
        else:
            price = k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(
                (b_a - r) * t
            ) * norm.cdf(-d1)

        return float(price), 0.0
