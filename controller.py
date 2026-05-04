"""Backward compatibility shim for the legacy controller module.

All implementations now live under the `services` package. This module
re-exports the public surface so existing callers keep working. Importing
this module emits a `DeprecationWarning` so consumers know to migrate
their imports to `services`.
"""

import warnings

warnings.warn(
    "controller is deprecated, import from services instead",
    DeprecationWarning,
    stacklevel=2,
)

from services import (
    ANALYTICAL_PRICERS,
    GREEK_ENGINE,
    OPTION_MAP,
    PAYOFF_REGISTRY,
    PRICER_MAP,
    aggregate_portfolio_greeks,
    compute_greek_surface,
    fit_heston_to_quotes,
    fit_svi_slice,
    generate_synthetic_quotes,
    get_greek_data,
    get_logger,
    get_option_and_pricer,
    get_point_pricing_context,
    get_surface_data,
    simulate_process_paths,
)

__all__ = [
    "ANALYTICAL_PRICERS",
    "GREEK_ENGINE",
    "OPTION_MAP",
    "PAYOFF_REGISTRY",
    "PRICER_MAP",
    "aggregate_portfolio_greeks",
    "compute_greek_surface",
    "fit_heston_to_quotes",
    "fit_svi_slice",
    "generate_synthetic_quotes",
    "get_greek_data",
    "get_logger",
    "get_option_and_pricer",
    "get_point_pricing_context",
    "get_surface_data",
    "simulate_process_paths",
]
