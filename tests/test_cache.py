"""Cache adapter behaviour."""

from __future__ import annotations

import numpy as np

from services._cache import _fingerprint, cached


def test_fingerprint_ndarray_content():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    c = np.array([1.0, 2.0, 4.0])
    assert _fingerprint(a) == _fingerprint(b)
    assert _fingerprint(a) != _fingerprint(c)


def test_fingerprint_dict_order_invariant():
    d1 = {"a": 1, "b": 2}
    d2 = {"b": 2, "a": 1}
    assert _fingerprint(d1) == _fingerprint(d2)


def test_cached_with_ndarray_args_hits_once():
    calls = {"n": 0}

    @cached()
    def add(arr: np.ndarray, x: float) -> float:
        calls["n"] += 1
        return float(arr.sum()) + x

    a = np.array([1.0, 2.0, 3.0])
    add(a, 4.0)
    add(a, 4.0)
    add(np.array([1.0, 2.0, 3.0]), 4.0)
    assert calls["n"] == 1
