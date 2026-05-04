"""Service boundary validation guards."""

from __future__ import annotations

import numpy as np
import pytest

from services._validation import (
    ValidationError,
    in_range,
    ndarray_1d,
    nonempty,
    nonneg_float,
    positive_float,
    positive_int,
)


def test_positive_float_rejects_zero():
    with pytest.raises(ValidationError):
        positive_float(0.0, "spot")


def test_positive_float_rejects_nan():
    with pytest.raises(ValidationError):
        positive_float(float("nan"), "spot")


def test_nonneg_float_accepts_zero():
    assert nonneg_float(0.0, "q") == 0.0


def test_in_range_clamps_outside():
    with pytest.raises(ValidationError):
        in_range(2.0, "rho", -1.0, 1.0)


def test_positive_int_rejects_negative():
    with pytest.raises(ValidationError):
        positive_int(-3, "n")


def test_nonempty_rejects_empty():
    with pytest.raises(ValidationError):
        nonempty([], "strikes")


def test_ndarray_1d_rejects_2d():
    with pytest.raises(ValidationError):
        ndarray_1d(np.zeros((2, 2)), "strikes")


def test_ndarray_1d_rejects_inf():
    with pytest.raises(ValidationError):
        ndarray_1d(np.array([1.0, float("inf"), 3.0]), "strikes")


def test_pricing_validation_rejects_unknown_pricer():
    from services.pricing import get_option_and_pricer

    bad = {
        "option_type": "Vanilla European",
        "pricer_type": "Bogus",
        "contract_params": {"s": 100, "k": 100, "t": 1.0, "r": 0.05, "sigma": 0.2, "q": 0.0},
    }
    with pytest.raises(ValidationError):
        get_option_and_pricer(bad)
