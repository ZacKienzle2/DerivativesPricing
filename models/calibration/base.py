"""Calibration base classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy.typing as npt


@dataclass(frozen=True)
class CalibrationResult:
    """Carrier for fitted parameters and diagnostics.

    Attributes:
        params: Mapping from parameter name to fitted value.
        residual_norm: L2 norm of fit residuals.
        n_iter: Iteration count.
        converged: Whether the optimiser reported convergence.
        cost: Optimiser final cost (typically 0.5 * residual_norm^2).
    """

    params: Dict[str, float]
    residual_norm: float
    n_iter: int
    converged: bool
    cost: float


class BaseCalibrator(ABC):
    """Abstract calibration interface."""

    @abstractmethod
    def calibrate(
        self,
        strikes: npt.NDArray,
        market: npt.NDArray,
        **kwargs: Any,
    ) -> CalibrationResult:
        """Fits model parameters to a market slice.

        Args:
            strikes: Strike grid.
            market: Per-strike market data (IV or price; concrete fitter
                documents which).
            **kwargs: Extra fitter-specific arguments (e.g. `t`, `r`).

        Returns:
            `CalibrationResult` carrying fitted parameters and diagnostics.
        """
