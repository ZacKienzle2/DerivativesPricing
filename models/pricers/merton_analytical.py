"""Merton 1976 analytical jump-diffusion pricer."""

import math
from typing import Any

import numpy as np
from scipy.special import ndtr as norm_cdf

from ..options import BaseOption, VanillaOption
from .base_pricer import BasePricer


class MertonAnalyticalPricer(BasePricer):
    def __init__(
        self,
        option: BaseOption,
        lam: float,
        mu_j: float,
        sigma_j: float,
        max_iter: int = 100,
        stop_cond: float = 1e-15,
    ):
        if not isinstance(option, VanillaOption):
            raise TypeError("MertonAnalyticalPricer is for VanillaOption only.")
        super().__init__(option)
        self.lam = lam
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.max_iter = max_iter
        self.stop_cond = stop_cond

    def get_params(self) -> dict[str, Any]:
        return {
            "lam": self.lam,
            "mu_j": self.mu_j,
            "sigma_j": self.sigma_j,
            "max_iter": self.max_iter,
            "stop_cond": self.stop_cond,
        }

    def _bs_price(
        self, s: float, k: float, t: float, r: float, q: float, sigma: float
    ) -> float:
        d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)

        if self.option.option_type == "call":
            return s * np.exp(-q * t) * norm_cdf(d1) - k * np.exp(-r * t) * norm_cdf(d2)
        else:
            return k * np.exp(-r * t) * norm_cdf(-d2) - s * np.exp(-q * t) * norm_cdf(
                -d1
            )

    def price(self) -> tuple[float, float]:
        opt = self.option
        s, k, t, r, q, sigma = opt.S, opt.K, opt.T, opt.r, opt.q, opt.sigma
        is_call = opt.option_type == "call"

        v_price = 0.0
        jump_comp = np.exp(self.mu_j + 0.5 * self.sigma_j**2)
        lam_prime = self.lam * jump_comp

        if t <= 0:
            return float(np.maximum(0, s - k if is_call else k - s)), 0.0

        for i in range(self.max_iter):
            r_i = (
                r
                - self.lam * (jump_comp - 1)
                + (i * (self.mu_j + 0.5 * self.sigma_j**2)) / t
            )
            sigma_i = np.sqrt(sigma**2 + (i * self.sigma_j**2) / t)

            try:
                poisson_prob = (
                    np.exp(-lam_prime * t) * (lam_prime * t) ** i
                ) / math.factorial(i)
            except (OverflowError, ValueError):
                break

            term_price = self._bs_price(s, k, t, r_i, q, sigma_i)
            sum_term = poisson_prob * term_price
            v_price += sum_term

            if sum_term < self.stop_cond:
                break

        return float(v_price), 0.0
