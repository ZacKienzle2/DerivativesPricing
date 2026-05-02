"""SABR calibrator fitting Hagan implied vols to a single-maturity slice."""

from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from ..processes.sabr import SABRProcess, hagan_lognormal_vol_jit
from .base import BaseCalibrator, CalibrationResult


class SABRCalibrator(BaseCalibrator):
    """Fits `(alpha, rho, nu)` of SABR to a market IV slice.

    `beta` is held fixed (default `0.5`). Hagan 2002 lognormal expansion is
    used as the model IV. Optimisation runs through
    `scipy.optimize.least_squares` with reflective bounds.
    """

    def __init__(self, beta: float = 0.5):
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1].")
        self.beta = beta

    def calibrate(
        self,
        strikes: npt.NDArray,
        market: npt.NDArray,
        f0: Optional[float] = None,
        t: Optional[float] = None,
        weights: Optional[npt.NDArray] = None,
        x0: Optional[npt.NDArray] = None,
        **kwargs,
    ) -> CalibrationResult:
        """Calibrates `(alpha, rho, nu)` to market IVs.

        Args:
            strikes: Strike grid, shape `(M,)`.
            market: Market implied vols at each strike, shape `(M,)`.
            f0: Forward price.
            t: Maturity in years.
            weights: Optional weights, shape `(M,)`. Defaults to uniform.
            x0: Optional initial guess `[alpha, rho, nu]`.

        Returns:
            CalibrationResult with `{alpha, rho, nu}`.
        """
        if f0 is None or t is None:
            raise ValueError("f0 and t are required.")
        ks = np.asarray(strikes, dtype=np.float64)
        ivs = np.asarray(market, dtype=np.float64)
        if ks.shape != ivs.shape:
            raise ValueError("strikes and market must have the same shape.")
        w = (
            np.ones_like(ivs)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )

        beta = self.beta

        def residuals(theta: npt.NDArray) -> npt.NDArray:
            alpha, rho, nu = theta
            model = np.empty_like(ivs)
            for idx in range(ks.size):
                model[idx] = hagan_lognormal_vol_jit(
                    f0, ks[idx], t, alpha, beta, rho, nu
                )
            return w * (model - ivs)

        guess = (
            np.array([float(ivs.mean()) * f0 ** (1.0 - beta), 0.0, 0.5])
            if x0 is None
            else np.asarray(x0, dtype=np.float64)
        )
        result = least_squares(
            residuals,
            guess,
            bounds=([1e-6, -0.999, 1e-6], [10.0, 0.999, 5.0]),
            method="trf",
            xtol=1e-10,
            ftol=1e-10,
        )
        alpha, rho, nu = result.x
        return CalibrationResult(
            params={"alpha": float(alpha), "rho": float(rho), "nu": float(nu),
                    "beta": float(beta)},
            residual_norm=float(np.linalg.norm(result.fun)),
            n_iter=int(result.nfev),
            converged=bool(result.success),
            cost=float(result.cost),
        )

    def to_process(self, params: dict, f0: float, r: float = 0.0) -> SABRProcess:
        """Builds a `SABRProcess` from a fitted parameter dict."""
        return SABRProcess(
            f0=f0,
            alpha=params["alpha"],
            beta=params["beta"],
            rho=params["rho"],
            nu=params["nu"],
            r=r,
        )
