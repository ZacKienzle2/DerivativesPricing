"""Typed pricing request DTO and parser.

Replaces the legacy `dict[str, Any]` request bundle that flowed through the
service layer. Validates the UI-shaped dict once at the boundary, resolves
every registry lookup ahead of time, and exposes an immutable value object
that downstream helpers consume without re-validating.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from models.options import BaseOption
from models.pricers import BasePricer

from ._validation import ValidationError
from .registry import ANALYTICAL_PRICERS, OPTION_MAP, PRICER_MAP

_RESERVED_MODEL_KEYS: frozenset[str] = frozenset({"greek_method"})


@dataclass(frozen=True, slots=True)
class PricingRequest:
    """Validated, pre-resolved bundle of inputs for a single pricing job.

    All registry lookups, kwargs filtering and key checks happen once at
    construction. Service helpers operate on this immutable object instead
    of an untyped dict, eliminating per-cell validation, copy and lookup
    cost on the surface and Greek-strip code paths.

    Attributes:
        option_type: Registered option-type label, e.g. `"Vanilla European"`.
        pricer_type: Registered pricer-type label, e.g. `"Black-Scholes"`.
        contract_params: Read-only mapping of option constructor kwargs.
        model_params: Read-only mapping of pricer kwargs (greek_method
            removed; that lives in `greek_method`).
        greek_method: Calculator strategy key, defaults to `"default"`.
        option_cls: Resolved `BaseOption` subclass.
        pricer_cls: Resolved `BasePricer` subclass.
        is_analytic_pricer: Whether `pricer_cls` belongs to the analytic
            set that ignores `model_params` at construction.
    """

    option_type: str
    pricer_type: str
    contract_params: Mapping[str, Any]
    model_params: Mapping[str, Any]
    greek_method: str
    option_cls: type[BaseOption]
    pricer_cls: type[BasePricer]
    is_analytic_pricer: bool

    @classmethod
    def from_dict(
        cls,
        inputs: Mapping[str, Any],
        option_flavour: str | None = None,
    ) -> "PricingRequest":
        """Validates a UI dict and returns an immutable `PricingRequest`."""
        option_type = inputs.get("option_type")
        if option_type not in OPTION_MAP:
            raise ValidationError(f"Unknown option_type {option_type!r}")
        pricer_type = inputs.get("pricer_type")
        if pricer_type not in PRICER_MAP:
            raise ValidationError(f"Unknown pricer_type {pricer_type!r}")
        if "contract_params" not in inputs:
            raise ValidationError("inputs missing contract_params")

        contract: dict[str, Any] = dict(inputs["contract_params"])
        if option_flavour is not None:
            contract["option_type"] = option_flavour

        raw_model: Mapping[str, Any] = inputs.get("model_params") or {}
        greek_method = str(raw_model.get("greek_method", "default"))
        model_params = {
            key: value
            for key, value in raw_model.items()
            if key not in _RESERVED_MODEL_KEYS
        }

        pricer_cls = PRICER_MAP[pricer_type]
        return cls(
            option_type=option_type,
            pricer_type=pricer_type,
            contract_params=MappingProxyType(contract),
            model_params=MappingProxyType(model_params),
            greek_method=greek_method,
            option_cls=OPTION_MAP[option_type],
            pricer_cls=pricer_cls,
            is_analytic_pricer=pricer_cls in ANALYTICAL_PRICERS,
        )

    def with_overrides(
        self,
        *,
        option_flavour: str | None = None,
        **contract_overrides: Any,
    ) -> "PricingRequest":
        """Returns a new request with patched contract attributes.

        Cheap on the hot path: a single dict copy plus a `MappingProxyType`
        wrap. The pre-resolved class references are reused unchanged.
        """
        contract: dict[str, Any] = dict(self.contract_params)
        if option_flavour is not None:
            contract["option_type"] = option_flavour
        if contract_overrides:
            contract.update(contract_overrides)
        return PricingRequest(
            option_type=self.option_type,
            pricer_type=self.pricer_type,
            contract_params=MappingProxyType(contract),
            model_params=self.model_params,
            greek_method=self.greek_method,
            option_cls=self.option_cls,
            pricer_cls=self.pricer_cls,
            is_analytic_pricer=self.is_analytic_pricer,
        )

    def build(self) -> tuple[BaseOption, BasePricer]:
        """Constructs the option and pricer instances."""
        option = self.option_cls(**self.contract_params)
        if self.is_analytic_pricer:
            return option, self.pricer_cls(option)
        return option, self.pricer_cls(option, **self.model_params)
