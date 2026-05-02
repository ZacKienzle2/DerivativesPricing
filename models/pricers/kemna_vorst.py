"""Kemna-Vorst closed-form geometric Asian pricer."""

from typing import Tuple

from models.options import AsianOption

from ._analytic_kernels import kemna_vorst_price_jit
from .base_pricer import BasePricer


class KemnaVorstPricer(BasePricer):
    """Prices continuously-averaged geometric Asian options.

    Wraps the JIT kernel `kemna_vorst_price_jit` so the same numerical core
    can be reused as a control variate inside Monte Carlo pricing of
    arithmetic-average Asians.
    """

    def __init__(self, option: AsianOption):
        if option.avg_type != "geometric":
            raise TypeError("KemnaVorstPricer is only for GEOMETRIC Asian options.")
        super().__init__(option)

    def price(self) -> Tuple[float, float]:
        """Returns `(price, 0.0)` from the Kemna-Vorst formula."""
        opt = self.option
        return (
            kemna_vorst_price_jit(
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
