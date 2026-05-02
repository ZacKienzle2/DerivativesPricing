"""Heston stochastic-volatility process via Andersen's QE scheme."""

import math
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from .base import BaseProcess

_PSI_C = 1.5


@njit(fastmath=True, parallel=True, cache=True, boundscheck=False)
def _heston_qe_jit(
    s0: float,
    v0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    eta: float,
    rho: float,
    t: float,
    num_steps: int,
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Andersen 2008 quadratic-exponential Heston path simulator.

    Pre-computes per-step constants once; per-path inner loop performs the
    QE branching plus the conditional log-spot update derived from the
    exact Heston martingale relationship.

    Args:
        s0: Initial spot.
        v0: Initial variance.
        r: Risk-free rate.
        q: Continuous dividend yield.
        kappa: Mean-reversion speed.
        theta: Long-run variance.
        eta: Volatility of variance.
        rho: Spot/variance correlation.
        t: Total horizon.
        num_steps: Discretisation steps.
        z: Driver normals, shape `(num_sims, num_steps, 2)`. The two slots
            are the variance driver and the spot orthogonal driver.

    Returns:
        Spot-price paths, shape `(num_sims, num_steps + 1)`.
    """
    num_sims = z.shape[0]
    dt = t / num_steps

    exp_kdt = math.exp(-kappa * dt)
    one_minus = 1.0 - exp_kdt
    eta_sq = eta * eta
    coef_s2_a = eta_sq / kappa * exp_kdt * one_minus if kappa > 0.0 else eta_sq * dt
    coef_s2_b = (
        theta * eta_sq / (2.0 * kappa) * one_minus * one_minus
        if kappa > 0.0
        else 0.0
    )
    log_s0 = math.log(s0)
    drift = (r - q) * dt
    k0 = -rho * kappa * theta * dt / eta if eta != 0.0 else 0.0
    k1 = 0.5 * dt * (kappa * rho / eta - 0.5) - rho / eta if eta != 0.0 else 0.0
    k2 = 0.5 * dt * (kappa * rho / eta - 0.5) + rho / eta if eta != 0.0 else 0.0
    k3 = 0.5 * dt * (1.0 - rho * rho)
    k4 = k3

    paths = np.empty((num_sims, num_steps + 1))

    for i in prange(num_sims):
        paths[i, 0] = s0
        log_s = log_s0
        v = v0
        for j in range(num_steps):
            zv = z[i, j, 0]
            zs = z[i, j, 1]

            m = theta + (v - theta) * exp_kdt
            s2 = v * coef_s2_a + coef_s2_b
            psi = s2 / (m * m) if m > 0.0 else 1e16

            if psi <= _PSI_C:
                inv = 2.0 / psi
                b2 = inv - 1.0 + math.sqrt(inv * (inv - 1.0))
                b = math.sqrt(b2) if b2 > 0.0 else 0.0
                a = m / (1.0 + b2)
                v_next = a * (b + zv) * (b + zv)
            else:
                p = (psi - 1.0) / (psi + 1.0)
                beta = (1.0 - p) / m if m > 0.0 else 0.0
                u = 0.5 * (1.0 + math.erf(zv * 0.7071067811865476))
                if u <= p:
                    v_next = 0.0
                else:
                    v_next = math.log((1.0 - p) / (1.0 - u)) / beta if beta > 0.0 else 0.0

            log_s += (
                drift
                + k0
                + k1 * v
                + k2 * v_next
                + math.sqrt(k3 * v + k4 * v_next) * zs
            )
            paths[i, j + 1] = math.exp(log_s)
            v = v_next
    return paths


class HestonProcess(BaseProcess):
    """Heston stochastic-volatility dynamics under the QE scheme.

    `dS_t / S_t = (r - q) dt + sqrt(v_t) dW_t^S`
    `dv_t = kappa (theta - v_t) dt + eta sqrt(v_t) dW_t^v`
    `d<W^S, W^v>_t = rho dt`
    """

    def __init__(
        self,
        s0: float,
        v0: float,
        r: float,
        q: float,
        kappa: float,
        theta: float,
        eta: float,
        rho: float,
    ):
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("rho must be in [-1, 1].")
        if v0 < 0.0 or theta < 0.0 or kappa < 0.0 or eta < 0.0:
            raise ValueError("v0, theta, kappa, eta must be non-negative.")
        self._s0 = float(s0)
        self._v0 = float(v0)
        self._r = float(r)
        self._q = float(q)
        self._kappa = float(kappa)
        self._theta = float(theta)
        self._eta = float(eta)
        self._rho = float(rho)

    @property
    def noise_dim(self) -> int:
        return 2

    @property
    def s0(self) -> float:
        return self._s0

    @property
    def r(self) -> float:
        return self._r

    def simulate_paths(
        self,
        num_sims: int,
        num_steps: int,
        t: float,
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Simulates Heston spot paths via Andersen QE."""
        if z.ndim != 3 or z.shape[2] != 2:
            raise ValueError("Heston requires z of shape (num_sims, num_steps, 2).")
        return _heston_qe_jit(
            self._s0,
            self._v0,
            self._r,
            self._q,
            self._kappa,
            self._theta,
            self._eta,
            self._rho,
            t,
            num_steps,
            z,
        )

    @property
    def signature(self) -> Tuple:
        return (
            type(self).__name__,
            self._s0, self._v0, self._r, self._q,
            self._kappa, self._theta, self._eta, self._rho,
        )
