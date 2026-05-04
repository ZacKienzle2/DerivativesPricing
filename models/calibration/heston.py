"""Heston calibrator fitting COS-priced calls to market quotes."""


import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from ..pricers._analytic_kernels import bs_full_greeks_jit
from ..pricers.cos_pricer import cos_heston_price_jit
from ..processes.heston import HestonProcess
from .base import BaseCalibrator, CalibrationResult


def _spread_weights(bids: npt.NDArray, asks: npt.NDArray) -> npt.NDArray:
    """Inverse bid-ask spread weights, clamped to a tiny floor."""
    spread = np.maximum(np.asarray(asks) - np.asarray(bids), 1e-6)
    return 1.0 / spread


def _vega_weights_from_iv(
    s0: float,
    strikes: npt.NDArray,
    ts: npt.NDArray,
    market_iv: npt.NDArray,
    r: float,
    q: float,
) -> npt.NDArray:
    """BS vega per quote, evaluated at the supplied market IVs."""
    out = np.empty_like(market_iv)
    for i in range(strikes.size):
        _, _, _, vega, _, _ = bs_full_greeks_jit(
            s0, float(strikes[i]), float(ts[i]), r, q, float(market_iv[i]), True
        )
        out[i] = vega
    return out


class HestonCalibrator(BaseCalibrator):
    """Fits `(kappa, theta, eta, rho, v0)` Heston parameters to market prices.

    Loss is per-quote relative price error. Pricing is delegated to the COS
    Fourier pricer for sub-millisecond evaluations, making the inner
    optimisation loop cheap enough that scipy's trust-region reflective
    least-squares converges in tens of iterations on a 50-quote slice.
    """

    def __init__(
        self,
        s0: float,
        r: float = 0.0,
        q: float = 0.0,
        n_terms: int = 256,
        l_truncate: float = 12.0,
    ):
        self.s0 = float(s0)
        self.r = float(r)
        self.q = float(q)
        self.n_terms = int(n_terms)
        self.l_truncate = float(l_truncate)

    def calibrate(
        self,
        strikes: npt.NDArray,
        market: npt.NDArray,
        maturities: npt.NDArray | None = None,
        is_call: npt.NDArray | None = None,
        weights: npt.NDArray | str | None = None,
        bids: npt.NDArray | None = None,
        asks: npt.NDArray | None = None,
        market_iv: npt.NDArray | None = None,
        x0: npt.NDArray | None = None,
        **kwargs,
    ) -> CalibrationResult:
        """Calibrates Heston parameters to a multi-maturity market slice.

        Args:
            strikes: Strike grid, shape `(M,)`.
            market: Market prices, shape `(M,)`.
            maturities: Per-quote maturities, shape `(M,)`.
            is_call: Per-quote call/put flag, shape `(M,)` (default all-calls).
            weights: Optional weights, shape `(M,)`.
            x0: Optional initial guess `[kappa, theta, eta, rho, v0]`.

        Returns:
            CalibrationResult with `{kappa, theta, eta, rho, v0}`.
        """
        ks = np.asarray(strikes, dtype=np.float64)
        prices = np.asarray(market, dtype=np.float64)
        if maturities is None:
            raise ValueError("maturities array is required.")
        ts = np.asarray(maturities, dtype=np.float64)
        if is_call is None:
            calls = np.ones(ks.size, dtype=np.bool_)
        else:
            calls = np.asarray(is_call, dtype=np.bool_)
        if not (ks.shape == prices.shape == ts.shape == calls.shape):
            raise ValueError("strikes, prices, maturities, is_call shape mismatch.")

        if isinstance(weights, str):
            if weights in ("vega", "vega_spread"):
                if market_iv is None:
                    raise ValueError(
                        "weights='vega' requires market_iv array."
                    )
                w = _vega_weights_from_iv(
                    self.s0, ks, ts, np.asarray(market_iv, dtype=np.float64),
                    self.r, self.q,
                )
            else:
                w = np.ones_like(prices)
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
            w = np.ones_like(prices)
        else:
            w = np.asarray(weights, dtype=np.float64)

        s0, r, q = self.s0, self.r, self.q
        n_terms = self.n_terms
        l_trunc = self.l_truncate

        def residuals(theta: npt.NDArray) -> npt.NDArray:
            kappa, theta_lr, eta, rho, v0 = theta
            model = np.empty_like(prices)
            for idx in range(ks.size):
                model[idx] = cos_heston_price_jit(
                    s0, ks[idx], ts[idx], r, q,
                    kappa, theta_lr, eta, rho, v0,
                    bool(calls[idx]), n_terms, l_trunc,
                )
            scale = np.where(np.abs(prices) > 1e-8, np.abs(prices), 1.0)
            return w * (model - prices) / scale

        guess = (
            np.array([2.0, 0.04, 0.4, -0.5, 0.04])
            if x0 is None
            else np.asarray(x0, dtype=np.float64)
        )
        result = least_squares(
            residuals,
            guess,
            bounds=(
                [1e-3, 1e-6, 1e-6, -0.999, 1e-6],
                [20.0, 2.0, 5.0, 0.999, 2.0],
            ),
            method="trf",
            xtol=1e-9,
            ftol=1e-9,
        )
        kappa, theta_lr, eta, rho, v0 = result.x
        return CalibrationResult(
            params={
                "kappa": float(kappa),
                "theta": float(theta_lr),
                "eta": float(eta),
                "rho": float(rho),
                "v0": float(v0),
            },
            residual_norm=float(np.linalg.norm(result.fun)),
            n_iter=int(result.nfev),
            converged=bool(result.success),
            cost=float(result.cost),
        )

    def to_process(self, params: dict) -> HestonProcess:
        """Builds a `HestonProcess` from a fitted parameter dict."""
        return HestonProcess(
            s0=self.s0,
            v0=params["v0"],
            r=self.r,
            q=self.q,
            kappa=params["kappa"],
            theta=params["theta"],
            eta=params["eta"],
            rho=params["rho"],
        )
