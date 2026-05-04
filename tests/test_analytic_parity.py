"""Analytic vs numeric parity smoke."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def vanilla_call():
    from models.options import VanillaOption

    return VanillaOption(
        s=100.0, k=100.0, t=1.0, r=0.05, sigma=0.2, option_type="call"
    )


def test_bs_matches_crr(vanilla_call):
    from models.pricers import BlackScholesPricer
    from models.pricers.binomial_tree import BinomialTreePricer

    bs = BlackScholesPricer(vanilla_call).price()[0]
    crr = BinomialTreePricer(vanilla_call, 800).price()[0]
    assert abs(bs - crr) < 0.05


def test_bs_full_greeks_signs(vanilla_call):
    from models.pricers import BlackScholesPricer

    g = BlackScholesPricer(vanilla_call).greeks()
    assert g["price"] > 0
    assert 0 < g["delta"] < 1
    assert g["gamma"] > 0
    assert g["vega"] > 0
    assert g["theta"] < 0
    assert g["rho"] > 0


def test_cos_heston_pricer_runs():
    from models.pricers.cos_pricer import cos_heston_price_jit

    price = cos_heston_price_jit(
        100.0, 100.0, 1.0, 0.05, 0.0,
        2.0, 0.04, 0.3, -0.5, 0.04,
        True, 256, 12.0,
    )
    assert 5.0 < price < 20.0
