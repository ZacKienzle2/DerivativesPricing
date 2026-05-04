"""Multilevel Monte Carlo (MLMC) for European vanilla under GBM.

Giles 2008. The estimator telescopes a sequence of resolutions:
    `E[P_L] = E[P_0] + sum_{l=1..L} E[P_l - P_{l-1}]`
with the level-`l` correction simulated using shared Brownian increments
between coarse and fine grids. Variance per level decays geometrically so
the optimal sample budget is concentrated on the cheapest (coarsest)
level. The implementation here keeps the design simple: fixed level
budgets supplied by the caller, GBM-only kernel, JIT prange for the
coarse-fine path pair.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from ..options import VanillaOption
from .base_pricer import BasePricer


@dataclass(frozen=True, slots=True)
class MLMCLevelStat:
    """Per-level diagnostics carried back from `mlmc_european_jit`."""

    samples: int
    mean: float
    variance: float


@njit(cache=True, parallel=True, fastmath=True, boundscheck=False)
def _mlmc_level_jit(
    s0: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    fine_steps: int,
    coarse_steps: int,
    is_call: bool,
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Per-path level-`l` correction `P_l - P_{l-1}` for level `l > 0`.

    Args:
        s0, k, r, q, sigma, t: GBM and option parameters.
        fine_steps: Number of fine-grid timesteps for level `l`.
        coarse_steps: Number of coarse-grid timesteps for level `l-1`
            (typically `fine_steps // 2`).
        is_call: True for call.
        z: Driver normals on the fine grid, shape `(num_sims, fine_steps)`.

    Returns:
        Per-path corrections, shape `(num_sims,)`. Caller multiplies by the
        risk-free discount factor.
    """
    num_sims = z.shape[0]
    fine_dt = t / fine_steps
    coarse_dt = t / coarse_steps
    drift_f = (r - q - 0.5 * sigma * sigma) * fine_dt
    drift_c = (r - q - 0.5 * sigma * sigma) * coarse_dt
    sqrt_fine = math.sqrt(fine_dt)
    factor = fine_steps // coarse_steps

    out = np.empty(num_sims)
    for i in prange(num_sims):
        log_fine = math.log(s0)
        log_coarse = math.log(s0)
        for c in range(coarse_steps):
            inc_sum = 0.0
            for sub in range(factor):
                z_ij = z[i, c * factor + sub]
                log_fine += drift_f + sigma * sqrt_fine * z_ij
                inc_sum += z_ij
            log_coarse += drift_c + sigma * sqrt_fine * inc_sum
        s_fine = math.exp(log_fine)
        s_coarse = math.exp(log_coarse)
        if is_call:
            p_fine = s_fine - k if s_fine > k else 0.0
            p_coarse = s_coarse - k if s_coarse > k else 0.0
        else:
            p_fine = k - s_fine if s_fine < k else 0.0
            p_coarse = k - s_coarse if s_coarse < k else 0.0
        out[i] = p_fine - p_coarse
    return out


@njit(cache=True, parallel=True, fastmath=True, boundscheck=False)
def _mlmc_base_jit(
    s0: float,
    k: float,
    r: float,
    q: float,
    sigma: float,
    t: float,
    num_steps: int,
    is_call: bool,
    z: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Level-0 raw payoff under GBM, shape `(num_sims,)`."""
    num_sims = z.shape[0]
    dt = t / num_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    sqrt_dt = math.sqrt(dt)
    out = np.empty(num_sims)
    for i in prange(num_sims):
        log_s = math.log(s0)
        for j in range(num_steps):
            log_s += drift + sigma * sqrt_dt * z[i, j]
        s_t = math.exp(log_s)
        if is_call:
            out[i] = s_t - k if s_t > k else 0.0
        else:
            out[i] = k - s_t if s_t < k else 0.0
    return out


class MLMCVanillaPricer(BasePricer):
    """Multilevel Monte Carlo pricer for European vanillas under GBM.

    Args:
        option: A `VanillaOption`.
        levels: Number of refinement levels above the coarsest grid. Total
            grids: `levels + 1`. Refinement factor is fixed at 2.
        base_steps: Steps on the coarsest grid (level 0).
        samples_per_level: Iterable of length `levels + 1` giving the
            sample budget for each level. If omitted, a geometric decay
            heuristic is used.
        seed: Optional PCG64 seed.
    """

    def __init__(
        self,
        option: VanillaOption,
        levels: int = 4,
        base_steps: int = 8,
        samples_per_level: tuple[int, ...] | None = None,
        seed: int | None = None,
    ):
        if not isinstance(option, VanillaOption):
            raise TypeError("MLMCVanillaPricer is for VanillaOption only.")
        super().__init__(option)
        self.levels = levels
        self.base_steps = base_steps
        if samples_per_level is None:
            samples_per_level = tuple(
                max(64, int(8192 / (4 ** lvl))) for lvl in range(levels + 1)
            )
        if len(samples_per_level) != levels + 1:
            raise ValueError("samples_per_level must have length levels + 1.")
        self.samples_per_level = tuple(int(n) for n in samples_per_level)
        self._rng = np.random.default_rng(seed)
        self.level_stats: tuple[MLMCLevelStat, ...] = ()

    def get_params(self) -> dict:
        """Returns the configuration of this pricer."""
        return {
            "levels": self.levels,
            "base_steps": self.base_steps,
            "samples_per_level": self.samples_per_level,
        }

    def price(self) -> tuple[float, float]:
        """Returns `(price, standard_error)` from the MLMC telescoping sum."""
        opt = self.option
        is_call = opt.option_type == "call"
        df = math.exp(-opt.r * opt.T)

        total = 0.0
        var_total = 0.0
        stats = []

        # Level 0
        n_l = self.samples_per_level[0]
        steps_l = self.base_steps
        z = self._rng.standard_normal((n_l, steps_l))
        payoffs = _mlmc_base_jit(
            opt.S, opt.K, opt.r, opt.q, opt.sigma, opt.T,
            steps_l, is_call, z,
        )
        mean_l = float(payoffs.mean())
        var_l = float(payoffs.var(ddof=1)) if n_l > 1 else 0.0
        total += mean_l
        var_total += var_l / n_l
        stats.append(MLMCLevelStat(n_l, mean_l, var_l))

        # Levels 1..L
        for lvl in range(1, self.levels + 1):
            n_l = self.samples_per_level[lvl]
            fine = self.base_steps * (2 ** lvl)
            coarse = fine // 2
            z = self._rng.standard_normal((n_l, fine))
            corrections = _mlmc_level_jit(
                opt.S, opt.K, opt.r, opt.q, opt.sigma, opt.T,
                fine, coarse, is_call, z,
            )
            mean_l = float(corrections.mean())
            var_l = float(corrections.var(ddof=1)) if n_l > 1 else 0.0
            total += mean_l
            var_total += var_l / n_l
            stats.append(MLMCLevelStat(n_l, mean_l, var_l))

        self.level_stats = tuple(stats)
        return df * total, df * math.sqrt(var_total)
