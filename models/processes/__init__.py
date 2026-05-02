"""Stochastic-process abstractions for path-based pricers.

Each process implements `BaseProcess.simulate_paths(num_sims, num_steps, z)`
returning an asset-price path tensor. The Monte Carlo pricer accepts any
concrete process and delegates dynamics to it, decoupling option payoff
logic from underlying-asset modelling.
"""

from .base import BaseProcess
from .bates import BatesProcess
from .gbm import GBMProcess
from .heston import HestonProcess
from .local_vol import LocalVolProcess
from .sabr import SABRProcess

__all__ = [
    "BaseProcess",
    "BatesProcess",
    "GBMProcess",
    "HestonProcess",
    "LocalVolProcess",
    "SABRProcess",
]
