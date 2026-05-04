"""Geometric Brownian Motion process."""


import numpy as np
import numpy.typing as npt

from ..options import BaseOption
from ..simulation import generate_final_prices_jit, generate_paths_jit
from .base import BaseProcess
from .registry import autoregister


@autoregister("GBM", noise_dim=1)
class GBMProcess(BaseProcess):
    """Standard log-normal GBM dynamics.

    `dS_t = (r - q) S_t dt + sigma S_t dW_t`. Wraps the existing JIT
    `generate_paths_jit` kernel.
    """

    def __init__(self, s0: float, r: float, q: float, sigma: float):
        self._s0 = float(s0)
        self._r = float(r)
        self._q = float(q)
        self._sigma = float(sigma)

    @classmethod
    def from_option(cls, option: BaseOption) -> "GBMProcess":
        """Constructs a GBM process from a single-asset option's parameters."""
        return cls(option.S, option.r, option.q, option.sigma)

    @property
    def noise_dim(self) -> int:
        return 1

    @property
    def s0(self) -> float:
        return self._s0

    @property
    def r(self) -> float:
        return self._r

    @property
    def q(self) -> float:
        return self._q

    @property
    def sigma(self) -> float:
        return self._sigma

    def simulate_paths(
        self,
        num_sims: int,
        num_steps: int,
        t: float,
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Simulates `num_sims` GBM paths over `[0, t]`."""
        return generate_paths_jit(
            self._s0, t, self._r, self._q, self._sigma, num_steps, z
        )

    def simulate_terminal(
        self,
        num_sims: int,
        num_steps: int,
        t: float,
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Returns terminal prices via the closed-form log increment sum."""
        return generate_final_prices_jit(
            self._s0, t, self._r, self._q, self._sigma, num_steps, z
        )

    @property
    def signature(self) -> tuple:
        return (type(self).__name__, self._s0, self._r, self._q, self._sigma)

    @property
    def supports_analytic_european(self) -> bool:
        return True

    def analytic_european_price(
        self, k: float, t: float, is_call: bool
    ) -> float:
        """Black-Scholes European price under GBM."""
        from ..pricers._analytic_kernels import bs_price_jit

        return float(
            bs_price_jit(self._s0, k, t, self._r, self._q, self._sigma, is_call)
        )
