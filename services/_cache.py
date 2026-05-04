"""Cache decorator that adapts to the runtime environment.

Picks `streamlit.cache_data` when a Streamlit script context is active,
falling back to `functools.lru_cache` for plain Python callers. The lru
fallback fingerprints ndarrays through `.tobytes()` and dicts/lists
recursively so calls with NumPy arrays or nested mappings still hit the
cache instead of bypassing it.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])


def _streamlit_runtime_available() -> bool:
    try:
        from streamlit.runtime import exists

        return bool(exists())
    except Exception:
        return False


def _fingerprint(value: Any) -> Any:
    """Returns a hashable surrogate that uniquely identifies `value`."""
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            value.shape,
            value.dtype.str,
            value.tobytes(),
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__name__,) + tuple(_fingerprint(v) for v in value)
    if isinstance(value, dict):
        return ("dict",) + tuple(
            (k, _fingerprint(v)) for k, v in sorted(value.items())
        )
    if isinstance(value, set):
        return ("set",) + tuple(_fingerprint(v) for v in sorted(value))
    try:
        hash(value)
        return value
    except TypeError:
        return ("repr", repr(value))


def cached(maxsize: int = 256, show_spinner: bool = False) -> Callable[[F], F]:
    """Returns a memoising decorator that adapts to the runtime."""

    def decorator(func: F) -> F:
        state: dict[str, Any] = {"impl": None, "kind": None, "store": None}

        def _lru_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            return (
                tuple(_fingerprint(a) for a in args),
                tuple((k, _fingerprint(v)) for k, v in sorted(kwargs.items())),
            )

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if state["impl"] is None:
                if _streamlit_runtime_available():
                    import streamlit as st

                    state["impl"] = st.cache_data(show_spinner=show_spinner)(
                        func
                    )
                    state["kind"] = "streamlit"
                else:
                    store: dict[Any, Any] = {}
                    state["store"] = store
                    order: list[Any] = []

                    def lru_call(*a: Any, **kw: Any) -> Any:
                        key = _lru_key(a, kw)
                        if key in store:
                            return store[key]
                        result = func(*a, **kw)
                        store[key] = result
                        order.append(key)
                        if len(order) > maxsize:
                            old = order.pop(0)
                            store.pop(old, None)
                        return result

                    state["impl"] = lru_call
                    state["kind"] = "lru"
            return state["impl"](*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
