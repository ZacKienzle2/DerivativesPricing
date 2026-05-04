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

    The default driver is Giles' adaptive allocation: a short pilot run
    estimates per-level variances `V_l`, then the remaining sample budget
    is split as `N_l ~ sqrt(V_l / C_l)` to minimise estimator variance for
    fixed compute. Cost `C_l` is proxied by `base_steps * 2^l`. A static
    `samples_per_level` tuple disables the pilot and runs the legacy
    fixed-budget loop verbatim.

    Args:
        option: A `VanillaOption`.
        levels: Number of refinement levels above the coarsest grid. Total
            grids: `levels + 1`. Refinement factor is fixed at 2.
        base_steps: Steps on the coarsest grid (level 0).
        samples_per_level: Iterable of length `levels + 1` overriding the
            adaptive allocator with a fixed budget.
        total_budget: Total sample budget for the adaptive allocator.
        pilot_per_level: Pilot samples drawn per level when adaptive.
        seed: Optional PCG64 seed.
    """

    def __init__(
        self,
        option: VanillaOption,
        levels: int = 4,
        base_steps: int = 8,
        samples_per_level: tuple[int, ...] | None = None,
        total_budget: int | None = None,
        pilot_per_level: int = 128,
        seed: int | None = None,
    ):
        if not isinstance(option, VanillaOption):
            raise TypeError("MLMCVanillaPricer is for VanillaOption only.")
        super().__init__(option)
        self.levels = levels
        self.base_steps = base_steps
        if samples_per_level is not None:
            if len(samples_per_level) != levels + 1:
                raise ValueError("samples_per_level must have length levels + 1.")
            self.samples_per_level: tuple[int, ...] | None = tuple(
                int(n) for n in samples_per_level
            )
        else:
            self.samples_per_level = None
        self.total_budget = (
            int(total_budget)
            if total_budget is not None
            else 8192 * (levels + 1)
        )
        self.pilot_per_level = max(2, int(pilot_per_level))
        self._rng = np.random.default_rng(seed)
        self.level_stats: tuple[MLMCLevelStat, ...] = ()

    def get_params(self) -> dict:
        """Returns the configuration of this pricer."""
        return {
            "levels": self.levels,
            "base_steps": self.base_steps,
            "samples_per_level": self.samples_per_level,
            "total_budget": self.total_budget,
            "pilot_per_level": self.pilot_per_level,
        }

    def _level_steps(self, lvl: int) -> tuple[int, int]:
        """Returns `(fine, coarse)` step counts for level `lvl`."""
        if lvl == 0:
            return self.base_steps, 0
        fine = self.base_steps * (2 ** lvl)
        return fine, fine // 2

    def _level_cost(self, lvl: int) -> float:
        """Proxy compute cost: timesteps walked per sample at level `lvl`."""
        fine, coarse = self._level_steps(lvl)
        return float(fine + coarse)

    def _draw_level(
        self, lvl: int, n: int, is_call: bool
    ) -> npt.NDArray[np.float64]:
        """Returns level-`lvl` per-sample cash flows for `n` paths."""
        opt = self.option
        if lvl == 0:
            z = self._rng.standard_normal((n, self.base_steps))
            return _mlmc_base_jit(
                opt.S, opt.K, opt.r, opt.q, opt.sigma, opt.T,
                self.base_steps, is_call, z,
            )
        fine, coarse = self._level_steps(lvl)
        z = self._rng.standard_normal((n, fine))
        return _mlmc_level_jit(
            opt.S, opt.K, opt.r, opt.q, opt.sigma, opt.T,
            fine, coarse, is_call, z,
        )

    def _allocate(
        self, variances: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.int64]:
        """Splits `total_budget` across levels via `sqrt(V_l / C_l)` rule."""
        costs = np.array(
            [self._level_cost(lvl) for lvl in range(self.levels + 1)],
            dtype=np.float64,
        )
        weights = np.sqrt(np.maximum(variances, 0.0) / costs)
        s = weights.sum()
        if s <= 0.0:
            n_each = max(self.pilot_per_level, self.total_budget // (self.levels + 1))
            return np.full(self.levels + 1, n_each, dtype=np.int64)
        share = weights / s
        budget = np.maximum(
            np.ceil(self.total_budget * share).astype(np.int64),
            self.pilot_per_level,
        )
        return budget

    def price(self) -> tuple[float, float]:
        """Returns `(price, standard_error)` from the MLMC telescoping sum."""
        opt = self.option
        is_call = opt.option_type == "call"
        df = math.exp(-opt.r * opt.T)

        if self.samples_per_level is not None:
            return self._price_fixed(is_call, df)
        return self._price_adaptive(is_call, df)

    def _price_fixed(self, is_call: bool, df: float) -> tuple[float, float]:
        """Runs the legacy fixed-budget MLMC loop."""
        assert self.samples_per_level is not None
        total = 0.0
        var_total = 0.0
        stats: list[MLMCLevelStat] = []
        for lvl in range(self.levels + 1):
            n_l = self.samples_per_level[lvl]
            samples = self._draw_level(lvl, n_l, is_call)
            mean_l = float(samples.mean())
            var_l = float(samples.var(ddof=1)) if n_l > 1 else 0.0
            total += mean_l
            var_total += var_l / max(n_l, 1)
            stats.append(MLMCLevelStat(n_l, mean_l, var_l))
        self.level_stats = tuple(stats)
        return df * total, df * math.sqrt(var_total)

    def _price_adaptive(self, is_call: bool, df: float) -> tuple[float, float]:
        """Runs a pilot + variance-optimal allocation across levels."""
        n_levels = self.levels + 1
        pilot_payoffs: list[npt.NDArray[np.float64]] = []
        pilot_var = np.zeros(n_levels, dtype=np.float64)
        for lvl in range(n_levels):
            samples = self._draw_level(lvl, self.pilot_per_level, is_call)
            pilot_payoffs.append(samples)
            pilot_var[lvl] = float(samples.var(ddof=1))

        budget = self._allocate(pilot_var)

        total = 0.0
        var_total = 0.0
        stats: list[MLMCLevelStat] = []
        for lvl in range(n_levels):
            n_target = int(budget[lvl])
            n_extra = max(0, n_target - self.pilot_per_level)
            if n_extra > 0:
                extra = self._draw_level(lvl, n_extra, is_call)
                samples = np.concatenate([pilot_payoffs[lvl], extra])
            else:
                samples = pilot_payoffs[lvl]
            n_total = samples.shape[0]
            mean_l = float(samples.mean())
            var_l = float(samples.var(ddof=1)) if n_total > 1 else 0.0
            total += mean_l
            var_total += var_l / max(n_total, 1)
            stats.append(MLMCLevelStat(n_total, mean_l, var_l))
        self.level_stats = tuple(stats)
        return df * total, df * math.sqrt(var_total)
