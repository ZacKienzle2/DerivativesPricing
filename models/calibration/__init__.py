"""Calibration package: market-data fitters for stochastic-vol models.

Each calibrator implements `BaseCalibrator.calibrate(market)` and returns a
ready-to-use process / parameter tuple. SABR fits Hagan IV; Heston fits
prices through the COS Fourier pricer.
"""

from .base import BaseCalibrator, CalibrationResult
from .dupire import build_local_vol_process, dupire_local_vol_grid
from .heston import HestonCalibrator
from .sabr import SABRCalibrator
from .svi import SVICalibrator

__all__ = [
    "BaseCalibrator",
    "CalibrationResult",
    "HestonCalibrator",
    "SABRCalibrator",
    "SVICalibrator",
    "build_local_vol_process",
    "dupire_local_vol_grid",
]
