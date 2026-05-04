"""Stochastic Volatility Inspired (SVI) parametrisation and calibrator.

Gatheral 2004 raw SVI:
    `w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))`
where `w` is total implied variance, `k` is log-moneyness `log(K/F)`. Five
parameters `(a, b, rho, m, sigma)` per maturity slice. Calibration runs
through scipy least-squares with the standard no-butterfly constraints
`b * (1 + |rho|) <= 4` and `a + b * sigma * sqrt(1 - rho^2) >= 0`.
"""

import math

import numpy as np
import numpy.typing as npt
from numba import njit
from scipy.optimize import least_squares

from .base import BaseCalibrator, CalibrationResult


@njit(cache=True, fastmath=True, inline="always")
def svi_total_variance(
    k: float, a: float, b: float, rho: float, m: float, sigma: float
) -> float:
    """Raw-SVI total variance at log-moneyness `k`."""
    diff = k - m
    return a + b * (rho * diff + math.sqrt(diff * diff + sigma * sigma))


@njit(cache=True, fastmath=True)
def svi_iv_jit(
    k: float,
    t: float,
    a: float,
    b: float,
    rho: float,
    m: float,
    sigma: float,
) -> float:
    """Black-implied vol at log-moneyness `k` and maturity `t` from raw SVI."""
    if t <= 0.0:
        return 0.0
    w = svi_total_variance(k, a, b, rho, m, sigma)
    if w <= 0.0:
        return 0.0
    return math.sqrt(w / t)


def svi_butterfly_ok(
    a: float, b: float, rho: float, m: float, sigma: float
) -> bool:
    """Checks the standard SVI no-butterfly conditions.

    Conditions enforced:
        * `b * (1 + |rho|) <= 4 / T_min` (slope bound; we use 4 as a proxy)
        * `a + b * sigma * sqrt(1 - rho^2) >= 0`  (positive minimum variance)
    """
    if b < 0.0 or sigma <= 0.0:
        return False
    if not (-1.0 < rho < 1.0):
        return False
    if a + b * sigma * math.sqrt(1.0 - rho * rho) < 0.0:
        return False
    if b * (1.0 + abs(rho)) > 4.0:
        return False
    return True


class SVICalibrator(BaseCalibrator):
    """Per-maturity raw-SVI fitter against a market total-variance slice.

    Uses `scipy.optimize.least_squares` with reflective bounds. Initial
    guess heuristic seeds `(a, b, rho, m, sigma)` from sample moments of
    the input slice and runs a single trust-region solve.
    """

    def __init__(self) -> None:
        pass

    def calibrate(
        self,
        strikes: npt.NDArray,
        market: npt.NDArray,
        f0: float | None = None,
        t: float | None = None,
        weights: npt.NDArray | None = None,
        x0: npt.NDArray | None = None,
        **kwargs,
    ) -> CalibrationResult:
        """Calibrates raw SVI to a market IV slice.

        Args:
            strikes: Strike grid, shape `(M,)`.
            market: Market implied vols at each strike, shape `(M,)`.
            f0: Forward price.
            t: Maturity in years.
            weights: Optional weights, shape `(M,)`. Defaults to uniform.
            x0: Optional initial guess `[a, b, rho, m, sigma]`.

        Returns:
            CalibrationResult with `{a, b, rho, m, sigma}`.
        """
        if f0 is None or t is None:
            raise ValueError("f0 and t are required.")
        ks = np.log(np.asarray(strikes, dtype=np.float64) / f0)
        ivs = np.asarray(market, dtype=np.float64)
        total_var = ivs * ivs * t
        if weights is None:
            w = np.ones_like(ivs)
        else:
            w = np.asarray(weights, dtype=np.float64)

        if x0 is None:
            atm_var = float(np.interp(0.0, ks, total_var))
            slope = float(np.std(total_var))
            x0 = np.array([
                max(atm_var * 0.1, 1e-4),
                max(slope, 1e-4),
                -0.3,
                float(ks[np.argmin(total_var)]),
                0.1,
            ])

        def residuals(theta: npt.NDArray) -> npt.NDArray:
            a, b, rho, m, sig = theta
            model = np.empty_like(total_var)
            for i in range(ks.size):
                model[i] = svi_total_variance(float(ks[i]), a, b, rho, m, sig)
            return w * (model - total_var)

        result = least_squares(
            residuals,
            x0,
            bounds=(
                [-1.0, 1e-6, -0.999, -2.0, 1e-4],
                [10.0, 10.0, 0.999, 2.0, 5.0],
            ),
            method="trf",
            xtol=1e-10,
            ftol=1e-10,
        )
        a, b, rho, m, sig = result.x
        return CalibrationResult(
            params={
                "a": float(a),
                "b": float(b),
                "rho": float(rho),
                "m": float(m),
                "sigma": float(sig),
            },
            residual_norm=float(np.linalg.norm(result.fun)),
            n_iter=int(result.nfev),
            converged=bool(result.success and svi_butterfly_ok(a, b, rho, m, sig)),
            cost=float(result.cost),
        )

    @staticmethod
    def evaluate(
        params: dict,
        strikes: npt.NDArray,
        f0: float,
        t: float,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Returns `(strikes, ivs)` evaluated from a fitted SVI slice."""
        ks = np.log(np.asarray(strikes, dtype=np.float64) / f0)
        ivs = np.empty_like(ks)
        a = params["a"]
        b = params["b"]
        rho = params["rho"]
        m = params["m"]
        sig = params["sigma"]
        for i in range(ks.size):
            ivs[i] = svi_iv_jit(float(ks[i]), t, a, b, rho, m, sig)
        return np.asarray(strikes, dtype=np.float64), ivs
