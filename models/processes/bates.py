"""Bates jump-diffusion process: Heston QE plus compound Poisson jumps."""

import math
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from .base import BaseProcess
from .heston import _PSI_C
from .registry import autoregister


@njit(fastmath=True, parallel=True, cache=True, boundscheck=False)
def _bates_qe_jit(
    s0: float,
    v0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    eta: float,
    rho: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    t: float,
    num_steps: int,
    z: npt.NDArray[np.float64],
    poisson: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Bates simulator: Andersen QE for variance plus log-normal Merton jumps.

    Jump counts are passed in as `poisson[i, j] ~ Poisson(lam * dt)` so the
    inner JIT loop can stay deterministic. Each step's compound jump is
    aggregated as `N*mu_j + sqrt(N)*sigma_j*Z_j` because the sum of `N`
    i.i.d. normals shares that distribution.

    Args:
        s0: Initial spot.
        v0: Initial variance.
        r: Risk-free rate.
        q: Dividend yield.
        kappa, theta, eta, rho: Heston diffusion parameters.
        lam, mu_j, sigma_j: Jump intensity and log-normal jump moments.
        t: Total horizon.
        num_steps: Discretisation steps.
        z: Driver normals shape `(num_sims, num_steps, 3)` for variance,
            spot-orthogonal and jump-magnitude axes.
        poisson: Pre-sampled jump counts shape `(num_sims, num_steps)`.

    Returns:
        Spot paths shape `(num_sims, num_steps + 1)`.
    """
    num_sims = z.shape[0]
    dt = t / num_steps
    exp_kdt = math.exp(-kappa * dt)
    one_minus = 1.0 - exp_kdt
    eta_sq = eta * eta
    coef_s2_a = (
        eta_sq / kappa * exp_kdt * one_minus if kappa > 0.0 else eta_sq * dt
    )
    coef_s2_b = (
        theta * eta_sq / (2.0 * kappa) * one_minus * one_minus
        if kappa > 0.0
        else 0.0
    )
    log_s0 = math.log(s0)
    m_jump = math.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0
    drift = (r - q - lam * m_jump) * dt
    k0 = -rho * kappa * theta * dt / eta if eta != 0.0 else 0.0
    k1 = (
        0.5 * dt * (kappa * rho / eta - 0.5) - rho / eta if eta != 0.0 else 0.0
    )
    k2 = (
        0.5 * dt * (kappa * rho / eta - 0.5) + rho / eta if eta != 0.0 else 0.0
    )
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
            zj = z[i, j, 2]

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
                    v_next = (
                        math.log((1.0 - p) / (1.0 - u)) / beta
                        if beta > 0.0
                        else 0.0
                    )

            n_jumps = poisson[i, j]
            if n_jumps > 0:
                jump_sum = n_jumps * mu_j + math.sqrt(float(n_jumps)) * sigma_j * zj
            else:
                jump_sum = 0.0

            log_s += (
                drift
                + k0
                + k1 * v
                + k2 * v_next
                + math.sqrt(k3 * v + k4 * v_next) * zs
                + jump_sum
            )
            paths[i, j + 1] = math.exp(log_s)
            v = v_next
    return paths


@autoregister("Bates", noise_dim=3)
class BatesProcess(BaseProcess):
    """Bates 1996 jump-diffusion: Heston variance plus log-normal Merton jumps.

    `dS_t/S_t = (r - q - lam*m) dt + sqrt(v_t) dW^S + (Y_t - 1) dN_t`
    `dv_t = kappa*(theta - v_t) dt + eta*sqrt(v_t) dW^v`
    where `N_t ~ Poisson(lam * t)` and `log Y_t ~ N(mu_J, sigma_J^2)`,
    with `m = exp(mu_J + 0.5*sigma_J^2) - 1` the martingale compensator.
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
        lam: float,
        mu_j: float,
        sigma_j: float,
        seed: Optional[int] = None,
    ):
        if not (-1.0 <= rho <= 1.0):
            raise ValueError("rho must be in [-1, 1].")
        if min(v0, theta, kappa, eta, lam, sigma_j) < 0.0:
            raise ValueError("v0, theta, kappa, eta, lam, sigma_j must be non-negative.")
        self._s0 = float(s0)
        self._v0 = float(v0)
        self._r = float(r)
        self._q = float(q)
        self._kappa = float(kappa)
        self._theta = float(theta)
        self._eta = float(eta)
        self._rho = float(rho)
        self._lam = float(lam)
        self._mu_j = float(mu_j)
        self._sigma_j = float(sigma_j)
        self._rng = np.random.default_rng(seed)

    @property
    def noise_dim(self) -> int:
        return 3

    @property
    def s0(self) -> float:
        return self._s0

    @property
    def r(self) -> float:
        return self._r

    @property
    def jump_params(self) -> Tuple[float, float, float]:
        """Returns `(lam, mu_j, sigma_j)`."""
        return self._lam, self._mu_j, self._sigma_j

    def simulate_paths(
        self,
        num_sims: int,
        num_steps: int,
        t: float,
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Simulates Bates spot paths."""
        if z.ndim != 3 or z.shape[2] != 3:
            raise ValueError("Bates requires z of shape (num_sims, num_steps, 3).")
        dt = t / num_steps
        lam_dt = self._lam * dt
        poisson = self._rng.poisson(lam_dt, size=(num_sims, num_steps)).astype(
            np.int64
        )
        return _bates_qe_jit(
            self._s0, self._v0, self._r, self._q,
            self._kappa, self._theta, self._eta, self._rho,
            self._lam, self._mu_j, self._sigma_j,
            t, num_steps, z, poisson,
        )

    @property
    def signature(self) -> Tuple:
        return (
            type(self).__name__,
            self._s0, self._v0, self._r, self._q,
            self._kappa, self._theta, self._eta, self._rho,
            self._lam, self._mu_j, self._sigma_j,
        )

    @property
    def supports_analytic_european(self) -> bool:
        return True

    def analytic_european_price(
        self, k: float, t: float, is_call: bool
    ) -> float:
        """COS Fourier-pricer European price under Bates."""
        from ..pricers.cos_pricer import cos_bates_price_jit

        return float(
            cos_bates_price_jit(
                self._s0, k, t, self._r, self._q,
                self._kappa, self._theta, self._eta, self._rho, self._v0,
                self._lam, self._mu_j, self._sigma_j,
                is_call, 256, 12.0,
            )
        )
