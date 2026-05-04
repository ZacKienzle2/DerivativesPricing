"""Reiner-Rubinstein 1991 closed-form barrier option pricer."""


import numpy as np
from scipy.special import ndtr as norm_cdf

from ..options import BarrierOption, VanillaOption
from .base_pricer import BasePricer
from .black_scholes import BlackScholesPricer


class ReinerRubinsteinPricer(BasePricer):
    """
    Prices European barrier options using the Reiner-Rubinstein (1991)
    closed-form analytical solution under the Black-Scholes model with
    continuous monitoring.
    """

    def __init__(self, option: BarrierOption):
        if not isinstance(option, BarrierOption):
            raise TypeError("This pricer is for BarrierOption contracts only.")
        super().__init__(option)

        self._precompute_terms()

    def _precompute_terms(self) -> None:
        """Calculates and stores common intermediate terms for the formulas."""
        opt = self.option
        s, k, h = opt.S, opt.K, opt.barrier_level
        t, r, q, sigma = opt.T, opt.r, opt.q, opt.sigma

        vanilla_equivalent = VanillaOption(s, k, t, r, sigma, opt.option_type, q)
        self.vanilla_pricer = BlackScholesPricer(vanilla_equivalent)

        sigma_sqrt_t = max(sigma * np.sqrt(t), 1e-12)
        self.d1 = (
            np.log(s / k) + (r - q + 0.5 * sigma * sigma) * t
        ) / sigma_sqrt_t
        self.d2 = self.d1 - sigma_sqrt_t

        self.mu = (r - q) - (sigma**2) / 2
        self.lambda_ = 1 + self.mu / (sigma**2)

        self.x1 = np.log(s / h) / (sigma * np.sqrt(t)) + self.lambda_ * sigma * np.sqrt(
            t
        )
        self.x2 = np.log(s / h) / (sigma * np.sqrt(t)) - self.lambda_ * sigma * np.sqrt(
            t
        )
        self.y1 = np.log(h**2 / (s * k)) / (
            sigma * np.sqrt(t)
        ) + self.lambda_ * sigma * np.sqrt(t)
        self.y2 = np.log(h**2 / (s * k)) / (
            sigma * np.sqrt(t)
        ) - self.lambda_ * sigma * np.sqrt(t)
        self.z = np.log(h / s) / (sigma * np.sqrt(t)) + self.lambda_ * sigma * np.sqrt(
            t
        )

    def price(self) -> tuple[float, float]:
        """
        Calculates the barrier option price by dispatching to the correct formula.
        """
        s, k, h = self.option.S, self.option.K, self.option.barrier_level
        t, r, q, sigma = self.option.T, self.option.r, self.option.q, self.option.sigma

        # Set +1 for a call, -1 for a put
        phi = 1.0 if self.option.option_type == "call" else -1.0

        # Set +1 for down barriers, -1 for up barriers
        eta = 1.0 if "down" in self.option.barrier_type else -1.0

        d1 = self.d1
        d2 = self.d2

        # Common building block terms used across all barrier types
        term1 = s * np.exp(-q * t) * norm_cdf(phi * d1) - k * np.exp(-r * t) * norm_cdf(
            phi * d2
        )

        term2 = s * np.exp(-q * t) * norm_cdf(phi * self.x1) - k * np.exp(
            -r * t
        ) * norm_cdf(phi * self.x1 - phi * sigma * np.sqrt(t))

        term3 = s * np.exp(-q * t) * (h / s) ** (2 * self.lambda_) * norm_cdf(
            eta * self.y1
        ) - k * np.exp(-r * t) * (h / s) ** (2 * self.lambda_ - 2) * norm_cdf(
            eta * self.y1 - eta * sigma * np.sqrt(t)
        )

        term4 = s * np.exp(-q * t) * (h / s) ** (2 * self.lambda_) * norm_cdf(
            eta * self.z
        ) - k * np.exp(-r * t) * (h / s) ** (2 * self.lambda_ - 2) * norm_cdf(
            eta * self.z - eta * sigma * np.sqrt(t)
        )

        price = 0.0

        if self.option.barrier_type == "down-and-in":
            if k > h:
                price = term3
            else:
                price = term2 - term4

        elif self.option.barrier_type == "up-and-in":
            if k < h:
                price = term3
            else:
                price = term2 - term4

        elif self.option.barrier_type == "down-and-out":
            vanilla_price, _ = self.vanilla_pricer.price()
            if k > h:
                price = vanilla_price - term3
            else:
                price = term1 - (term2 - term4)

        elif self.option.barrier_type == "up-and-out":
            vanilla_price, _ = self.vanilla_pricer.price()
            if k < h:
                price = vanilla_price - term3
            else:
                price = term1 - (term2 - term4)

        return float(price), 0.0
