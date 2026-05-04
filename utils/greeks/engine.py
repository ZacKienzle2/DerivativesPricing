"""Greek engine that resolves calculators through the registry."""

from __future__ import annotations

from models.pricers import BasePricer

from .calculators import FiniteDifferenceCalculator, GreekCalculator
from .registry import GreekCalculatorRegistry


class GreekEngine:
    """Selects the appropriate `GreekCalculator` for a pricer instance.

    Strategy lookup goes through `GreekCalculatorRegistry`, which keys on
    `(pricer_class, method)` and walks the pricer's MRO so registrations
    on base classes also serve subclasses. Falls back to a finite
    difference calculator when the registry has no entry.
    """

    def get_calculator(
        self, pricer: BasePricer, method: str = "default"
    ) -> GreekCalculator:
        """Returns a calculator for `pricer` under the chosen method."""
        factory = GreekCalculatorRegistry.resolve(
            pricer, method, FiniteDifferenceCalculator
        )
        return factory(pricer)
