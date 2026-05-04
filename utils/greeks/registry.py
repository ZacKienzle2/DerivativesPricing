"""Greek calculator registry keyed on pricer class and method token."""

from __future__ import annotations

from collections.abc import Callable

from models.pricers import BasePricer

from .calculators import GreekCalculator

_CalculatorFactory = Callable[[BasePricer], GreekCalculator]


class GreekCalculatorRegistry:
    """Maps `(pricer_class, method)` to a calculator factory.

    Walks the pricer's MRO so a registration on a base class also serves
    every subclass unless a more specific entry exists. Method dispatch
    falls back to the wildcard `default` when an unknown method is asked
    for.
    """

    _factories: dict[tuple[type[BasePricer], str], _CalculatorFactory] = {}

    @classmethod
    def register(
        cls,
        pricer_cls: type[BasePricer],
        method: str,
        factory: _CalculatorFactory,
    ) -> None:
        """Registers `factory` for `(pricer_cls, method)`."""
        cls._factories[(pricer_cls, method)] = factory

    @classmethod
    def resolve(
        cls,
        pricer: BasePricer,
        method: str,
        fallback: _CalculatorFactory,
    ) -> _CalculatorFactory:
        """Looks up the registered factory or returns the fallback."""
        for klass in type(pricer).__mro__:
            entry = cls._factories.get((klass, method))
            if entry is not None:
                return entry
        for klass in type(pricer).__mro__:
            entry = cls._factories.get((klass, "default"))
            if entry is not None:
                return entry
        return fallback


def register_default_calculators() -> None:
    """Wires the standard calculator set into the registry."""
    from models.pricers import (
        BlackScholesPricer,
        KemnaVorstPricer,
        LatticePricer,
        LongstaffSchwartzPricer,
        MonteCarloPricer,
    )

    from .calculators import (
        BlackScholesGreekCalculator,
        FiniteDifferenceCalculator,
        KemnaVorstGreekCalculator,
        LatticeGreekCalculator,
        LikelihoodRatioCalculator,
        PathwiseCalculator,
    )

    GreekCalculatorRegistry.register(
        BlackScholesPricer, "default", BlackScholesGreekCalculator
    )
    GreekCalculatorRegistry.register(
        KemnaVorstPricer, "default", KemnaVorstGreekCalculator
    )
    GreekCalculatorRegistry.register(
        LatticePricer, "default", LatticeGreekCalculator
    )

    for sim_cls in (MonteCarloPricer, LongstaffSchwartzPricer):
        GreekCalculatorRegistry.register(
            sim_cls, "default", FiniteDifferenceCalculator
        )
        GreekCalculatorRegistry.register(
            sim_cls, "Pathwise", PathwiseCalculator
        )
        GreekCalculatorRegistry.register(
            sim_cls, "Likelihood Ratio", LikelihoodRatioCalculator
        )
        GreekCalculatorRegistry.register(
            sim_cls, "Finite Difference", FiniteDifferenceCalculator
        )


register_default_calculators()
