"""Common-Random-Number (CRN) finite-difference Greeks.

Wraps any callable pricer in a central-difference bumping engine. Same
seed across all bumps so MC noise cancels between paired runs and the
Greek estimates are dominated by the deterministic bump response. Works
for analytic, FD or MC pricers as long as they accept a fresh option /
process tuple per call.
"""

import copy
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GreekShifts:
    """Per-Greek bump sizes used by the CRN engine."""

    delta: float = 0.5
    gamma: float = 0.5
    vega: float = 0.005
    theta: float = 1.0 / 365.0
    rho: float = 0.0001


def _bumped(option: Any, attr: str, shift: float) -> Any:
    """Returns a copy of `option` with `attr` shifted by `shift`."""
    cloned = copy.copy(option)
    object.__setattr__(cloned, attr, getattr(cloned, attr) + shift)
    return cloned


class CRNSensitivities:
    """Common-random-number Greek estimator.

    Args:
        pricer_factory: Callable that takes an option and returns a pricer
            instance whose `.price()` returns `(price, std_err)`. The
            factory is responsible for wiring any process / seed plumbing.
        shifts: `GreekShifts` carrying per-Greek bump sizes.
        greeks: Iterable of Greek names to compute. Defaults to all five.
    """

    def __init__(
        self,
        pricer_factory: Callable[[Any], Any],
        shifts: GreekShifts = GreekShifts(),
        greeks: Iterable[str] | None = None,
    ):
        self.pricer_factory = pricer_factory
        self.shifts = shifts
        self.greeks = tuple(greeks) if greeks else ("delta", "gamma", "vega", "theta", "rho")

    def _price(self, option: Any) -> float:
        return float(self.pricer_factory(option).price()[0])

    def compute(self, option: Any) -> dict[str, float]:
        """Returns `{price, delta, gamma, vega, theta, rho}` for the option."""
        out: dict[str, float] = {}
        base_price = self._price(option)
        out["price"] = base_price

        if "delta" in self.greeks or "gamma" in self.greeks:
            h = self.shifts.delta
            up = self._price(_bumped(option, "S", h))
            dn = self._price(_bumped(option, "S", -h))
            if "delta" in self.greeks:
                out["delta"] = (up - dn) / (2.0 * h)
            if "gamma" in self.greeks:
                out["gamma"] = (up - 2.0 * base_price + dn) / (h * h)

        if "vega" in self.greeks:
            h = self.shifts.vega
            up = self._price(_bumped(option, "sigma", h))
            dn = self._price(_bumped(option, "sigma", -h))
            out["vega"] = (up - dn) / (2.0 * h)

        if "theta" in self.greeks:
            h = self.shifts.theta
            base_t = option.T
            if base_t > h:
                up = self._price(_bumped(option, "T", h))
                dn = self._price(_bumped(option, "T", -h))
                out["theta"] = (dn - up) / (2.0 * h)
            else:
                up = self._price(_bumped(option, "T", h))
                out["theta"] = (base_price - up) / h

        if "rho" in self.greeks:
            h = self.shifts.rho
            up = self._price(_bumped(option, "r", h))
            dn = self._price(_bumped(option, "r", -h))
            out["rho"] = (up - dn) / (2.0 * h)

        return out
