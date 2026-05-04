"""Static registries mapping UI tokens to domain types.

`PRICER_MAP` and `ANALYTICAL_PRICERS` are derived from
`models.pricers.registry.PricerRegistry` so the underlying registry stays
the single source of truth.
"""

from __future__ import annotations

from models.options import (
    AmericanOption,
    AsianOption,
    BarrierOption,
    BaseOption,
    BasketOption,
    VanillaOption,
)
from models.payoffs import (
    long_bond,
    long_call,
    long_put,
    long_stock,
    short_bond,
    short_call,
    short_put,
    short_stock,
)
from models.pricers import (
    BasePricer,
    BlackScholesPricer,
    HaugHaugMargrabePricer,
    KemnaVorstPricer,
    LatticePricer,
    LevyPricer,
    LongstaffSchwartzPricer,
    MonteCarloPricer,
    ReinerRubinsteinPricer,
    TurnbullWakemanPricer,
)
from utils.greeks import GreekEngine

OPTION_MAP: dict[str, type[BaseOption]] = {
    "Vanilla European": VanillaOption,
    "American": AmericanOption,
    "Barrier": BarrierOption,
    "Basket": BasketOption,
    "Asian": AsianOption,
}

PRICER_MAP: dict[str, type[BasePricer]] = {
    "Black-Scholes": BlackScholesPricer,
    "Lattice": LatticePricer,
    "Monte Carlo": MonteCarloPricer,
    "Longstaff-Schwartz": LongstaffSchwartzPricer,
    "Kemna-Vorst": KemnaVorstPricer,
    "Levy": LevyPricer,
    "Turnbull-Wakeman": TurnbullWakemanPricer,
    "Haug-Haug-Margrabe": HaugHaugMargrabePricer,
    "Reiner-Rubinstein": ReinerRubinsteinPricer,
}

ANALYTICAL_PRICERS: frozenset[type[BasePricer]] = frozenset(
    {
        BlackScholesPricer,
        KemnaVorstPricer,
        LevyPricer,
        TurnbullWakemanPricer,
        HaugHaugMargrabePricer,
        ReinerRubinsteinPricer,
    }
)

PAYOFF_REGISTRY: dict[str, dict[str, object]] = {
    "Long Call": {"func": long_call, "params": {"k": 100.0, "premium": 5.0}},
    "Short Call": {"func": short_call, "params": {"k": 100.0, "premium": 5.0}},
    "Long Put": {"func": long_put, "params": {"k": 100.0, "premium": 5.0}},
    "Short Put": {"func": short_put, "params": {"k": 100.0, "premium": 5.0}},
    "Long Stock": {"func": long_stock, "params": {"purchase_price": 100.0}},
    "Short Stock": {"func": short_stock, "params": {"sale_price": 100.0}},
    "Long Bond": {
        "func": long_bond,
        "params": {"future_value": 105.0, "price": 100.0},
    },
    "Short Bond": {
        "func": short_bond,
        "params": {"future_value": 105.0, "price": 100.0},
    },
}

GREEK_ENGINE = GreekEngine()
