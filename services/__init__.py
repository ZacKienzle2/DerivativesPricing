"""Service layer between the Streamlit dashboard and the pricing core."""

from .calibration import (
    fit_heston_to_quotes,
    fit_svi_slice,
    generate_synthetic_quotes,
)
from .greeks import (
    aggregate_portfolio_greeks,
    compute_greek_surface,
    get_greek_data,
)
from .lab import simulate_process_paths
from .logging import get_logger
from .presets import BATES_PRESETS, HESTON_PRESETS, PROCESS_LAB_PRESETS
from .pricing import (
    get_option_and_pricer,
    get_point_pricing_context,
    get_surface_data,
)
from .registry import (
    ANALYTICAL_PRICERS,
    GREEK_ENGINE,
    OPTION_MAP,
    PAYOFF_REGISTRY,
    PRICER_MAP,
)

__all__ = [
    "ANALYTICAL_PRICERS",
    "BATES_PRESETS",
    "GREEK_ENGINE",
    "HESTON_PRESETS",
    "OPTION_MAP",
    "PAYOFF_REGISTRY",
    "PRICER_MAP",
    "PROCESS_LAB_PRESETS",
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
