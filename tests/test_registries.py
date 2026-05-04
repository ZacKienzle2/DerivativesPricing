"""Registry coverage checks."""

from __future__ import annotations


def test_pricer_registry_method_coverage():
    from models.options import (
        AmericanOption,
        AsianOption,
        BarrierOption,
        BasketOption,
        VanillaOption,
    )
    from models.pricers import PricerRegistry

    coverage = {
        VanillaOption: {"default", "black_scholes", "monte_carlo", "binomial"},
        AmericanOption: {"default", "binomial", "lsm"},
        AsianOption: {"default", "monte_carlo", "kemna_vorst"},
        BarrierOption: {"default", "reiner_rubinstein"},
        BasketOption: {"default", "monte_carlo"},
    }
    for option_cls, required in coverage.items():
        methods = set(PricerRegistry.methods_for(option_cls))
        missing = required - methods
        assert not missing, f"{option_cls.__name__} missing methods {missing}"


def test_process_registry_full_set():
    from models.processes import list_processes

    expected = {"GBM", "Heston", "Bates", "SABR", "LocalVol", "RBergomi"}
    assert expected.issubset(set(list_processes()))


def test_greek_calculator_registry_dispatches():
    from models.options import VanillaOption
    from models.pricers import BlackScholesPricer
    from utils.greeks import GreekEngine

    opt = VanillaOption(
        s=100.0, k=100.0, t=1.0, r=0.05, sigma=0.2, option_type="call"
    )
    pricer = BlackScholesPricer(opt)
    calc = GreekEngine().get_calculator(pricer)
    assert calc.__class__.__name__ == "BlackScholesGreekCalculator"
