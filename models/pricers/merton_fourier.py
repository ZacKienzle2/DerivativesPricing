"""Merton jump-diffusion Fourier pricer via FFT inversion."""

from typing import Any

import numpy as np
from scipy.interpolate import interp1d

from ..options import BaseOption, VanillaOption
from .base_pricer import BasePricer


class MertonFourierPricer(BasePricer):
    def __init__(
        self,
        option: BaseOption,
        lam: float,
        mu_j: float,
        sigma_j: float,
        ngrid: int = 2**10,
        xwidth: int = 12,
        alpha: float = -1.0,
    ):
        if not isinstance(option, VanillaOption):
            raise TypeError("MertonFourierPricer is for VanillaOption only.")
        super().__init__(option)
        self.lam = lam
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.ngrid = ngrid
        self.xwidth = xwidth
        self.alpha = alpha if option.option_type == "call" else -alpha

    def get_params(self) -> dict[str, Any]:
        return {
            "lam": self.lam,
            "mu_j": self.mu_j,
            "sigma_j": self.sigma_j,
            "ngrid": self.ngrid,
            "xwidth": self.xwidth,
            "alpha": self.alpha,
        }

    def _calculate_payoff_transform(self, x, xi, k, l_bound, u_bound):
        xi_2 = self.alpha + 1j * xi

        with np.errstate(divide="ignore", invalid="ignore"):
            g_transform = (
                np.exp(u_bound * (1 + xi_2)) - np.exp(l_bound * (1 + xi_2))
            ) / (1 + xi_2) - (
                np.exp(k + u_bound * xi_2) - np.exp(k + l_bound * xi_2)
            ) / xi_2

        if self.alpha == 0:
            g_transform[int(self.ngrid / 2)] = (
                np.exp(u_bound) - np.exp(l_bound) - np.exp(k) * (u_bound - l_bound)
            )
        elif self.alpha == -1:
            g_transform[int(self.ngrid / 2)] = (
                u_bound - l_bound + np.exp(k - u_bound) - np.exp(k - l_bound)
            )

        return g_transform

    def _calculate_characteristic_function(
        self, xi, t, r, q, sigma, lam, mu_j, sigma_j
    ):
        mu_rn = r - q - 0.5 * sigma**2 - lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)

        xi_adj = xi + 1j * self.alpha

        psi = (
            1j * mu_rn * xi_adj
            - 0.5 * (sigma * xi_adj) ** 2
            + lam * (np.exp(1j * mu_j * xi_adj - 0.5 * (sigma_j * xi_adj) ** 2) - 1)
        )

        return np.exp(psi * t)

    def price(self) -> tuple[float, float]:
        opt = self.option
        s, k, t, r, q, sigma = opt.S, opt.K, opt.T, opt.r, opt.q, opt.sigma

        n_half = int(self.ngrid / 2)
        b = self.xwidth / 2
        dx = self.xwidth / self.ngrid
        x = dx * np.linspace(-n_half, n_half - 1, self.ngrid)

        dxi = 2 * np.pi / self.xwidth
        xi = dxi * np.linspace(-n_half, n_half - 1, self.ngrid)

        s_prices = s * np.exp(x)
        u_bound = np.log(s * np.exp(b) / s)
        l_bound_market = np.log(s * np.exp(-b) / s)
        k_log = np.log(k / s)

        if opt.option_type == "call":
            l_bound = max(l_bound_market, k_log)
            u_bound = u_bound
        else:
            l_bound = u_bound
            u_bound = min(k_log, l_bound_market)

        g_transform = self._calculate_payoff_transform(x, xi, k_log, l_bound, u_bound)

        char_func = self._calculate_characteristic_function(
            xi, t, r, q, sigma, self.lam, self.mu_j, self.sigma_j
        )

        integrand = g_transform * np.conj(char_func)

        price_fft = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(integrand)))
        price_vec = np.exp(-r * t) * np.real(price_fft) / self.xwidth

        price = float(interp1d(s_prices, price_vec, kind="slinear")(s))

        return price, 0.0
