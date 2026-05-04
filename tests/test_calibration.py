"""Calibration round-trip smoke."""

from __future__ import annotations

import numpy as np


def test_heston_recovers_truth_from_synthetic():
    from models.calibration import HestonCalibrator
    from models.processes import HestonProcess
    from utils.synthetic_quotes import synthetic_call_quotes

    truth = HestonProcess(
        s0=100, v0=0.04, r=0.05, q=0.0,
        kappa=2.0, theta=0.04, eta=0.3, rho=-0.5,
    )
    strikes = np.linspace(80.0, 120.0, 7)
    mats = np.array([0.5, 1.0, 2.0])
    quotes = synthetic_call_quotes(truth, strikes, mats, spread_bps=0.0, seed=7)

    calib = HestonCalibrator(s0=100.0, r=0.05)
    flat_k, flat_t = np.meshgrid(strikes, mats, indexing="ij")
    res = calib.calibrate(
        flat_k.ravel(), quotes.prices.ravel(),
        maturities=flat_t.ravel(),
        x0=np.array([1.5, 0.05, 0.4, -0.3, 0.05]),
    )
    fitted = res.params
    for key, expected in (
        ("kappa", 2.0), ("theta", 0.04), ("eta", 0.3),
        ("rho", -0.5), ("v0", 0.04),
    ):
        assert abs(fitted[key] - expected) < 5e-3, (
            f"{key} drift {fitted[key]} vs {expected}"
        )


def test_svi_fit_is_close_to_market():
    from models.calibration import SVICalibrator

    strikes = np.array([60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0])
    ivs = np.array([0.30, 0.26, 0.23, 0.21, 0.20, 0.21, 0.23])
    res = SVICalibrator().calibrate(strikes, ivs, f0=100.0, t=1.0)
    _, model = SVICalibrator.evaluate(res.params, strikes, 100.0, 1.0)
    assert np.sqrt(((model - ivs) ** 2).mean()) < 0.005
