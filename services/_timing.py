"""Lightweight timing decorator that emits structured log records."""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from .logging import get_logger

F = TypeVar("F", bound=Callable[..., Any])

_log = get_logger("timing")


def timed(name: str | None = None) -> Callable[[F], F]:
    """Decorator that times the wrapped callable and logs the duration.

    Args:
        name: Optional override for the metric name; defaults to the
            qualified function name.

    Returns:
        A wrapper that runs the function, records the elapsed time and
        emits a debug level log line with the metric name and duration.
    """

    def decorator(func: F) -> F:
        metric = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                _log.debug("%s %.3f ms", metric, elapsed_ms)

        return wrapper  # type: ignore[return-value]

    return decorator
