"""SABR calibrator fitting Hagan implied vols to a single-maturity slice."""


import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from ..pricers._analytic_kernels import bs_full_greeks_jit
from ..processes.sabr import SABRProcess, hagan_lognormal_vol_jit
from .base import BaseCalibrator, CalibrationResult


def _vega_weights(
    f0: float, strikes: npt.NDArray, t: float, ivs: npt.NDArray
) -> npt.NDArray:
    """BS vega per strike at the supplied IVs (no discount factor)."""
    out = np.empty_like(ivs)
    for i in range(strikes.size):
        _, _, _, vega, _, _ = bs_full_greeks_jit(
            f0, float(strikes[i]), t, 0.0, 0.0, float(ivs[i]), True
        )
        out[i] = vega
    return out


def _spread_weights(bids: npt.NDArray, asks: npt.NDArray) -> npt.NDArray:
    """Inverse-spread weights, clamped to a tiny floor for tight quotes."""
    spread = np.maximum(np.asarray(asks) - np.asarray(bids), 1e-6)
    return 1.0 / spread


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
        f0: float | None = None,
        t: float | None = None,
        weights: npt.NDArray | str | None = None,
        bids: npt.NDArray | None = None,
        asks: npt.NDArray | None = None,
        x0: npt.NDArray | None = None,
        **kwargs,
    ) -> CalibrationResult:
        """Calibrates `(alpha, rho, nu)` to market IVs.

        Args:
            strikes: Strike grid, shape `(M,)`.
            market: Market implied vols at each strike, shape `(M,)`.
            f0: Forward price.
            t: Maturity in years.
            weights: Per-strike weights, an explicit array or the string
                `"vega"` to weight by Black vega at the market IV, the string
                `"spread"` to weight by inverse `ask - bid` (requires `bids`
                and `asks`), `"vega_spread"` to combine the two, or `None`
                for uniform.
            bids, asks: Bid/ask quote arrays, required for the `"spread"`
                or `"vega_spread"` modes.
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

        if isinstance(weights, str):
            if weights in ("vega", "vega_spread"):
                w = _vega_weights(f0, ks, t, ivs)
            else:
                w = np.ones_like(ivs)
            if weights in ("spread", "vega_spread"):
                if bids is None or asks is None:
                    raise ValueError(
                        f"weights={weights!r} requires bids and asks arrays."
                    )
                w = w * _spread_weights(np.asarray(bids), np.asarray(asks))
            w_sum = w.sum()
            if w_sum > 0.0:
                w = w * (w.size / w_sum)
        elif weights is None:
            w = np.ones_like(ivs)
        else:
            w = np.asarray(weights, dtype=np.float64)

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
