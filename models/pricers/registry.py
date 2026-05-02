"""Pricer registry and memoising pricing service.

`PricerRegistry` maps `(option_class, method_name) -> pricer_class`. MRO
walks let a registration on a base class also serve subclasses unless a
more-specific entry exists.

`PricingService` is a thin façade over the registry that LRU-caches results
keyed on `(option_class, method, frozen_option_state, frozen_kwargs)`.
Repeat calls with identical inputs return in `O(1)` without invoking the
underlying pricer at all.
"""

from collections import OrderedDict
from typing import Any, Dict, Hashable, Tuple, Type

import numpy as np

from ..options import (
    AmericanOption,
    AsianOption,
    BarrierOption,
    BaseOption,
    BasketOption,
    VanillaOption,
)
from .base_pricer import BasePricer
from .binomial_tree import BinomialTreePricer
from .black_scholes import BlackScholesPricer
from .crank_nicholson import CrankNicolsonPricer
from .explicit_fd import ExplicitFDPricer
from .implicit_fd import ImplicitFDPricer
from .kemna_vorst import KemnaVorstPricer
from .longstaff_schwartz import LongstaffSchwartzPricer
from .monte_carlo import MonteCarloPricer


class PricerRegistry:
    """Maps `(option_class, method_name)` to a pricer class."""

    _factories: Dict[Tuple[Type[BaseOption], str], Type[BasePricer]] = {}

    @classmethod
    def register(
        cls,
        option_cls: Type[BaseOption],
        method: str,
        pricer_cls: Type[BasePricer],
    ) -> None:
        """Registers `pricer_cls` as the handler for `(option_cls, method)`."""
        cls._factories[(option_cls, method)] = pricer_cls

    @classmethod
    def get(
        cls, option_cls: Type[BaseOption], method: str
    ) -> Type[BasePricer]:
        """Looks up the pricer for `option_cls` walking the MRO.

        Raises:
            KeyError: when no registration matches.
        """
        for klass in option_cls.__mro__:
            entry = cls._factories.get((klass, method))
            if entry is not None:
                return entry
        raise KeyError(f"No pricer registered for ({option_cls.__name__!r}, {method!r}).")

    @classmethod
    def methods_for(cls, option_cls: Type[BaseOption]) -> Tuple[str, ...]:
        """Returns the method tokens registered for `option_cls`."""
        out = []
        for klass in option_cls.__mro__:
            for (cls_k, method), _ in cls._factories.items():
                if cls_k is klass and method not in out:
                    out.append(method)
        return tuple(out)


def _hashable(value: Any) -> Hashable:
    """Maps mutable container values into a hashable tuple form."""
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


def _option_signature(opt: BaseOption) -> Tuple[Hashable, ...]:
    """Builds a hashable tuple of every slot value across the MRO."""
    seen: set = set()
    parts: list = []
    for klass in type(opt).__mro__:
        slots = getattr(klass, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in seen:
                continue
            seen.add(name)
            try:
                parts.append((name, _hashable(getattr(opt, name))))
            except AttributeError:
                continue
    return tuple(parts)


class PricingService:
    """Memoising pricing façade.

    Args:
        cache_size: Maximum entries kept in the LRU cache (default 256).
    """

    def __init__(self, cache_size: int = 256):
        self._cache: "OrderedDict[Tuple, Tuple[float, float]]" = OrderedDict()
        self._max = cache_size

    def _key(
        self, option: BaseOption, method: str, kwargs: Dict[str, Any]
    ) -> Tuple:
        opt_sig = _option_signature(option)
        kw_sig = tuple(sorted((k, _hashable(v)) for k, v in kwargs.items()))
        return (type(option).__name__, method, opt_sig, kw_sig)

    def price(
        self,
        option: BaseOption,
        method: str = "default",
        **kwargs: Any,
    ) -> Tuple[float, float]:
        """Prices `option` via the `method`-registered pricer.

        Hits the LRU cache on repeat calls with identical option state and
        kwargs.
        """
        key = self._key(option, method, kwargs)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        pricer_cls = PricerRegistry.get(type(option), method)
        pricer = pricer_cls(option, **kwargs)
        result = pricer.price()
        self._cache[key] = result
        if len(self._cache) > self._max:
            self._cache.popitem(last=False)
        return result

    def clear_cache(self) -> None:
        """Drops every memoised result."""
        self._cache.clear()


def _register_defaults() -> None:
    """Wires the default pricer set."""
    from .haug_haug_margrabe import HaugHaugMargrabePricer
    from .lattice_pricer import LatticePricer
    from .levy_pricer import LevyPricer
    from .reiner_rubinstein import ReinerRubinsteinPricer
    from .turnbull_wakeman import TurnbullWakemanPricer

    PricerRegistry.register(VanillaOption, "default", BlackScholesPricer)
    PricerRegistry.register(VanillaOption, "black_scholes", BlackScholesPricer)
    PricerRegistry.register(VanillaOption, "monte_carlo", MonteCarloPricer)
    PricerRegistry.register(VanillaOption, "binomial", BinomialTreePricer)
    PricerRegistry.register(VanillaOption, "lattice", LatticePricer)
    PricerRegistry.register(VanillaOption, "implicit_fd", ImplicitFDPricer)
    PricerRegistry.register(VanillaOption, "crank_nicolson", CrankNicolsonPricer)
    PricerRegistry.register(VanillaOption, "explicit_fd", ExplicitFDPricer)

    PricerRegistry.register(AmericanOption, "default", CrankNicolsonPricer)
    PricerRegistry.register(AmericanOption, "binomial", BinomialTreePricer)
    PricerRegistry.register(AmericanOption, "lattice", LatticePricer)
    PricerRegistry.register(AmericanOption, "lsm", LongstaffSchwartzPricer)
    PricerRegistry.register(
        AmericanOption, "longstaff_schwartz", LongstaffSchwartzPricer
    )
    PricerRegistry.register(AmericanOption, "implicit_fd", ImplicitFDPricer)
    PricerRegistry.register(AmericanOption, "crank_nicolson", CrankNicolsonPricer)
    PricerRegistry.register(AmericanOption, "explicit_fd", ExplicitFDPricer)

    PricerRegistry.register(AsianOption, "default", MonteCarloPricer)
    PricerRegistry.register(AsianOption, "monte_carlo", MonteCarloPricer)
    PricerRegistry.register(AsianOption, "binomial", BinomialTreePricer)
    PricerRegistry.register(AsianOption, "lattice", LatticePricer)
    PricerRegistry.register(AsianOption, "kemna_vorst", KemnaVorstPricer)
    PricerRegistry.register(AsianOption, "levy", LevyPricer)
    PricerRegistry.register(AsianOption, "turnbull_wakeman", TurnbullWakemanPricer)
    PricerRegistry.register(AsianOption, "haug_haug_margrabe", HaugHaugMargrabePricer)

    PricerRegistry.register(BarrierOption, "default", ReinerRubinsteinPricer)
    PricerRegistry.register(BarrierOption, "reiner_rubinstein", ReinerRubinsteinPricer)
    PricerRegistry.register(BarrierOption, "monte_carlo", MonteCarloPricer)
    PricerRegistry.register(BarrierOption, "lattice", LatticePricer)

    PricerRegistry.register(BasketOption, "default", MonteCarloPricer)
    PricerRegistry.register(BasketOption, "monte_carlo", MonteCarloPricer)


_register_defaults()
