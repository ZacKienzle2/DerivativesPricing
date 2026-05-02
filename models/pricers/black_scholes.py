"""Black-Scholes-Merton analytic pricer."""

from typing import Tuple

from models.options import VanillaOption

from ._analytic_kernels import bs_price_jit
from .base_pricer import BasePricer


class BlackScholesPricer(BasePricer):
    """Prices European vanilla options via the BSM closed-form.

    Backed by a JIT-compiled scalar kernel so the call cost is dominated by
    a few `math.exp` and `math.erf` invocations rather than a scipy
    round-trip.
    """

    def __init__(self, option: VanillaOption):
        if not isinstance(option, VanillaOption):
            raise TypeError("BlackScholesPricer is for VanillaOption only.")
        super().__init__(option)

    def price(self) -> Tuple[float, float]:
        """Returns `(price, 0.0)` from the analytic formula."""
        opt = self.option
        return (
            bs_price_jit(
                opt.S,
                opt.K,
                opt.T,
                opt.r,
                opt.q,
                opt.sigma,
                opt.option_type == "call",
            ),
            0.0,
        )
