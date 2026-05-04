"""Lightweight validation helpers for service boundary inputs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


class ValidationError(ValueError):
    """Raised when a public service input fails its precondition."""


def positive_float(value: Any, name: str) -> float:
    """Returns `value` as a strictly positive float."""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(out) or out <= 0.0:
        raise ValidationError(f"{name} must be a positive finite number, got {out}")
    return out


def nonneg_float(value: Any, name: str) -> float:
    """Returns `value` as a non negative finite float."""
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(out) or out < 0.0:
        raise ValidationError(
            f"{name} must be a non negative finite number, got {out}"
        )
    return out


def in_range(value: Any, name: str, low: float, high: float) -> float:
    """Returns `value` clamped to `[low, high]` after checking finiteness."""
    out = nonneg_float(value, name) if low >= 0 else float(value)
    if not (low <= out <= high):
        raise ValidationError(f"{name} must lie in [{low}, {high}], got {out}")
    return out


def positive_int(value: Any, name: str) -> int:
    """Returns `value` as a strictly positive int."""
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} must be int-like, got {value!r}") from exc
    if out <= 0:
        raise ValidationError(f"{name} must be > 0, got {out}")
    return out


def nonempty(value: Sequence[Any], name: str) -> Sequence[Any]:
    """Asserts `value` is a non empty sequence."""
    if value is None or len(value) == 0:
        raise ValidationError(f"{name} must be a non empty sequence")
    return value


def ndarray_1d(value: Any, name: str) -> np.ndarray:
    """Coerces `value` to a 1D float64 ndarray."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be 1D, got shape {arr.shape}")
    if arr.size == 0:
        raise ValidationError(f"{name} must be non empty")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} must contain only finite values")
    return arr
