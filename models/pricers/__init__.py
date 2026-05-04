"""Pricer surface and registry for the derivatives library.

JIT kernels are warmed up eagerly on package import unless the
`DERIVATIVES_PRICING_NO_WARMUP` environment variable is set, in which
case callers can opt in by invoking `prewarm()` explicitly.
"""

import os

from ._warmup import _warmup as _warmup
from .base_pricer import BasePricer
from .black_scholes import BlackScholesPricer
from .haug_haug_margrabe import HaugHaugMargrabePricer
from .kemna_vorst import KemnaVorstPricer
from .lattice_pricer import LatticePricer
from .levy_pricer import LevyPricer
from .longstaff_schwartz import LongstaffSchwartzPricer
from .monte_carlo import MonteCarloPricer
from .registry import PricerRegistry, PricingService
from .reiner_rubinstein import ReinerRubinsteinPricer
from .turnbull_wakeman import TurnbullWakemanPricer


def prewarm() -> None:
    """Forces a JIT compile pass for every cached numba kernel."""
    _warmup()


if os.environ.get("DERIVATIVES_PRICING_NO_WARMUP") != "1":
    prewarm()


__all__ = [
    "BasePricer",
    "BlackScholesPricer",
    "HaugHaugMargrabePricer",
    "KemnaVorstPricer",
    "LatticePricer",
    "LevyPricer",
    "LongstaffSchwartzPricer",
    "MonteCarloPricer",
    "PricerRegistry",
    "PricingService",
    "ReinerRubinsteinPricer",
    "TurnbullWakemanPricer",
    "prewarm",
]
