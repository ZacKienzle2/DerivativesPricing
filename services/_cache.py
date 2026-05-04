"""Cache decorator that adapts to the runtime environment.

Picks `streamlit.cache_data` when a Streamlit script context is active,
falling back to `functools.lru_cache` for hashable arguments and a plain
pass through for unhashable ones (dicts, ndarrays). Lets the service
layer remain agnostic about the host runtime so the same helpers can be
used from Streamlit, plain Python scripts, tests or notebooks.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _streamlit_runtime_available() -> bool:
    try:
        from streamlit.runtime import exists

        return bool(exists())
    except Exception:
        return False


def cached(maxsize: int = 256, show_spinner: bool = False) -> Callable[[F], F]:
    """Returns a memoising decorator that adapts to the runtime."""

    def decorator(func: F) -> F:
        state: dict[str, Any] = {"impl": None, "kind": None}

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
                    state["impl"] = functools.lru_cache(maxsize=maxsize)(func)
                    state["kind"] = "lru"
            if state["kind"] == "lru":
                try:
                    hash((args, tuple(sorted(kwargs.items()))))
                except TypeError:
                    return func(*args, **kwargs)
            return state["impl"](*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
