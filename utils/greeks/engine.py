"""Greek calculator factory dispatching on pricer type."""

from __future__ import annotations

from models.pricers import (
    BasePricer,
    BlackScholesPricer,
    MonteCarloPricer,
    LongstaffSchwartzPricer,
    LatticePricer,
    KemnaVorstPricer,
)
from .calculators import (
    BlackScholesGreekCalculator,
    FiniteDifferenceCalculator,
    GreekCalculator,
    KemnaVorstGreekCalculator,
    LatticeGreekCalculator,
    LikelihoodRatioCalculator,
    PathwiseCalculator,
)


class GreekEngine:
    """
    A factory for creating the appropriate Greek calculator based on the pricer
    and the desired calculation method (the "strategy").
    """

    def get_calculator(
        self, pricer: BasePricer, method: str = "default"
    ) -> GreekCalculator:
        """
        Selects and returns the correct Greek calculator instance.

        Args:
            pricer: The pricer instance for which to calculate Greeks.
            method: The user-selected method (e.g., "Pathwise", "Finite Difference").

        Returns:
            An instance of a GreekCalculator subclass.
        """
        # Handle pricers with a single, non-negotiable analytical method
        if isinstance(pricer, BlackScholesPricer):
            return BlackScholesGreekCalculator(pricer)
        if isinstance(pricer, KemnaVorstPricer):
            return KemnaVorstGreekCalculator(pricer)

        # Handle pricers with user-selectable methods
        if isinstance(pricer, (MonteCarloPricer, LongstaffSchwartzPricer)):
            if method == "Pathwise":
                return PathwiseCalculator(pricer)
            if method == "Likelihood Ratio":
                return LikelihoodRatioCalculator(pricer)
            # The default method for simulation-based pricers is Finite Difference
            return FiniteDifferenceCalculator(pricer)

        # Handle the Lattice pricer
        if isinstance(pricer, LatticePricer):
            return LatticeGreekCalculator(pricer)

        # As a general fallback for any other numerical pricer, use Finite Difference
        return FiniteDifferenceCalculator(pricer)
