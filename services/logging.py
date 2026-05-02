"""Project logging configuration."""

import logging
import os

_LEVEL = os.environ.get("DERIVATIVES_PRICING_LOG_LEVEL", "INFO").upper()
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def _configure() -> None:
    root = logging.getLogger("derivatives_pricing")
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(_LEVEL)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Returns a project logger under the `derivatives_pricing` namespace."""
    _configure()
    return logging.getLogger(f"derivatives_pricing.{name}")
