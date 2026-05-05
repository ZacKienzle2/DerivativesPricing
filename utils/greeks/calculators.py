"""Greek calculators across analytic, lattice and Monte Carlo pricers."""

from __future__ import annotations

import concurrent.futures
import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from scipy.stats import norm

from models.options import (
    AmericanOption,
    AsianOption,
    BarrierOption,
    BasketOption,
    VanillaOption,
)
from models.pricers.base_pricer import BasePricer
from models.pricers.black_scholes import BlackScholesPricer
from models.pricers.kemna_vorst import KemnaVorstPricer
from models.pricers.lattice_pricer import LatticePricer
from models.pricers.monte_carlo import MonteCarloPricer, _calculate_barrier_payoffs_jit


@njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def _calculate_pathwise_estimators_jit(
    z_matrix: npt.NDArray[np.float64],
    s0: float,
    k: float,
    r: float,
    t: float,
    sigma: float,
    q: float,
    is_call: bool,
) -> tuple[float, float, float, float, float]:
    """Pathwise estimators for delta, vega, rho and dV/dT under GBM.

    Single parallel sweep over simulations; per-step constants are hoisted
    out of the inner time loop and accumulators reduce in `prange`.

    Args:
        z_matrix: I.I.D. driver normals shape `(num_sims, num_steps)`.
        s0, k, r, t, sigma, q: GBM and contract parameters.
        is_call: True for call.

    Returns:
        Tuple `(delta, gamma, vega, rho, dv_dt)`. `gamma` is NaN
        (non-estimable pathwise for vanillas).
    """
    num_sims, num_steps = z_matrix.shape
    dt = t / num_steps
    sqrt_dt = math.sqrt(dt)
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol_step = sigma * sqrt_dt
    drift_t = r - q - 0.5 * sigma * sigma
    inv_2t = 0.5 / t if t > 1e-12 else 0.0
    inv_s0 = 1.0 / s0

    delta_acc = 0.0
    vega_acc = 0.0
    rho_acc = 0.0
    theta_acc = 0.0

    for i in prange(num_sims):
        log_s = 0.0
        z_sum = 0.0
        for j in range(num_steps):
            z = z_matrix[i, j]
            log_s += drift + vol_step * z
            z_sum += z
        s_t = s0 * math.exp(log_s)

        if is_call:
            payoff = s_t - k if s_t > k else 0.0
            indicator = 1.0 if s_t > k else 0.0
        else:
            payoff = k - s_t if s_t < k else 0.0
            indicator = -1.0 if s_t < k else 0.0

        w_t = z_sum * sqrt_dt
        d_st_d_sigma = s_t * (w_t - sigma * t)
        d_st_d_r = s_t * t
        d_st_d_t = s_t * (drift_t + sigma * w_t * inv_2t)

        delta_acc += indicator * s_t * inv_s0
        vega_acc += indicator * d_st_d_sigma
        rho_acc += indicator * d_st_d_r - t * payoff
        theta_acc += -r * payoff + indicator * d_st_d_t

    inv_n = 1.0 / num_sims
    return (
        delta_acc * inv_n,
        np.nan,
        vega_acc * inv_n,
        rho_acc * inv_n,
        theta_acc * inv_n,
    )


@njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def _calculate_lrm_greeks_jit(
    discounted_payoffs: npt.NDArray[np.float64],
    z_terminal: npt.NDArray[np.float64],
    s0: float,
    r: float,
    t: float,
    sigma: float,
) -> tuple[float, float, float, float]:
    """Likelihood-ratio Greeks under GBM in a single parallel reduction.

    Args:
        discounted_payoffs: Shape `(num_sims,)`.
        z_terminal: Per-path standardised terminal Brownian shape `(num_sims,)`.
        s0, r, t, sigma: GBM and contract parameters.

    Returns:
        `(delta, vega, rho, theta)`.
    """
    n = discounted_payoffs.shape[0]
    sqrt_t = math.sqrt(t)
    sigma_sqrt_t = sigma * sqrt_t
    inv_score_d = 1.0 / (sigma_sqrt_t * s0)
    inv_sigma = 1.0 / sigma
    inv_2t = 0.5 / t
    drift_corr = (r - 0.5 * sigma * sigma) / sigma_sqrt_t
    sqrt_t_over_sigma = sqrt_t / sigma

    delta_acc = 0.0
    vega_acc = 0.0
    rho_acc = 0.0
    theta_acc = 0.0
    for i in prange(n):
        z = z_terminal[i]
        z2 = z * z
        z2m1 = z2 - 1.0
        payoff = discounted_payoffs[i]
        delta_acc += payoff * z * inv_score_d
        vega_acc += payoff * (z2m1 * inv_sigma - z * sqrt_t)
        rho_acc += payoff * z * sqrt_t_over_sigma
        theta_acc += payoff * (drift_corr * z + z2m1 * inv_2t)

    inv_n = 1.0 / n
    return delta_acc * inv_n, vega_acc * inv_n, rho_acc * inv_n, -theta_acc * inv_n


@njit(parallel=True, fastmath=True, cache=True, boundscheck=False)
def _lattice_greeks_jit(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    num_steps: int,
    option_type_code: int,
    exercise_type_code: int,
) -> tuple[float, float, float, float]:
    """CRR backward induction with ping-pong buffers and incremental spot grid.

    Captures the three-node value triplet at step 2 to compute delta, gamma
    and theta in one pass. Inner reduction is `prange` over node index;
    spot levels at the current step are derived from a precomputed terminal
    grid by a scalar multiplier `s_scale` updated per timestep, eliminating
    `np.arange`/`pow` calls from the hot loop.

    Args:
        S, K, T, r, q, sigma: Standard option parameters.
        num_steps: Number of CRR steps.
        option_type_code: 1 for call, 0 for put.
        exercise_type_code: 1 for American, 0 for European.

    Returns:
        Tuple `(price, delta, gamma, theta)`.
    """
    if num_steps < 2:
        return np.nan, np.nan, np.nan, np.nan

    dt = T / num_steps
    sqrt_dt = math.sqrt(dt)
    discount = math.exp(-r * dt)
    u = math.exp(sigma * sqrt_dt)
    d = 1.0 / u
    pu = (math.exp((r - q) * dt) - d) / (u - d)
    pd = 1.0 - pu
    ratio = d / u
    is_call = option_type_code == 1
    is_amer = exercise_type_code == 1

    n_nodes = num_steps + 1
    st_base = np.empty(n_nodes)
    st = S * u ** num_steps
    for j in range(n_nodes):
        st_base[j] = st
        st *= ratio

    vals_curr = np.empty(n_nodes)
    vals_next = np.empty(n_nodes)
    for j in range(n_nodes):
        s = st_base[j]
        v = s - K if is_call else K - s
        vals_curr[j] = v if v > 0.0 else 0.0

    s_scale = 1.0
    v_uu = 0.0
    v_ud = 0.0
    v_dd = 0.0

    for i in range(num_steps - 1, -1, -1):
        s_scale *= d
        if i == 1:
            v_uu = vals_curr[0]
            v_ud = vals_curr[1]
            v_dd = vals_curr[2]
        n_i = i + 1
        for j in prange(n_i):
            cont = (pu * vals_curr[j] + pd * vals_curr[j + 1]) * discount
            if is_amer:
                s = st_base[j] * s_scale
                ex = s - K if is_call else K - s
                if ex > cont:
                    cont = ex
            vals_next[j] = cont
        vals_curr, vals_next = vals_next, vals_curr

    price = vals_curr[0]

    s_uu = S * u * u
    s_dd = S * d * d
    spread = s_uu - s_dd
    delta = (v_uu - v_dd) / spread if spread != 0.0 else np.nan

    up_step = s_uu - S
    dn_step = S - s_dd
    if up_step > 0.0 and dn_step > 0.0 and spread > 0.0:
        gamma = ((v_uu - v_ud) / up_step - (v_ud - v_dd) / dn_step) / (0.5 * spread)
    else:
        gamma = np.nan

    theta = (v_ud - price) / (2.0 * dt)

    return price, delta, gamma, theta


class IncompatibleCalculatorError(TypeError):
    pass


class GreekCalculator(ABC):
    def __init__(self, pricer: BasePricer):
        self.pricer = pricer
        self.option = pricer.option

    @abstractmethod
    def calculate(self) -> dict[str, Any]:
        pass


class BlackScholesGreekCalculator(GreekCalculator):
    def __init__(self, pricer: BasePricer):
        if not isinstance(pricer, BlackScholesPricer):
            raise IncompatibleCalculatorError(
                "This calculator is only for the BlackScholesPricer."
            )
        super().__init__(pricer)
        opt = self.option
        sigma_sqrt_t = max(opt.sigma * np.sqrt(opt.T), 1e-12)
        self.d1 = (
            np.log(opt.S / opt.K)
            + (opt.r - opt.q + 0.5 * opt.sigma * opt.sigma) * opt.T
        ) / sigma_sqrt_t
        self.d2 = self.d1 - sigma_sqrt_t

    def calculate(self) -> dict[str, Any]:
        opt = self.option
        s, k, t, r, q, sigma = opt.S, opt.K, opt.T, opt.r, opt.q, opt.sigma
        if opt.option_type == "call":
            delta = np.exp(-q * t) * norm.cdf(self.d1)
            theta_term2 = -r * k * np.exp(-r * t) * norm.cdf(self.d2)
            theta_term3 = q * s * np.exp(-q * t) * norm.cdf(self.d1)
            rho = k * t * np.exp(-r * t) * norm.cdf(self.d2)
        else:
            delta = np.exp(-q * t) * (norm.cdf(self.d1) - 1.0)
            theta_term2 = r * k * np.exp(-r * t) * norm.cdf(-self.d2)
            theta_term3 = -q * s * np.exp(-q * t) * norm.cdf(-self.d1)
            rho = -k * t * np.exp(-r * t) * norm.cdf(-self.d2)
        gamma = (np.exp(-q * t) * norm.pdf(self.d1)) / (s * sigma * np.sqrt(t))
        vega = s * np.exp(-q * t) * norm.pdf(self.d1) * np.sqrt(t)
        theta_term1 = -(s * np.exp(-q * t) * norm.pdf(self.d1) * sigma) / (
            2 * np.sqrt(t)
        )
        theta = theta_term1 + theta_term2 + theta_term3
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega * 0.01,
            "theta": theta / 365.0,
            "rho": rho * 0.01,
        }


class FiniteDifferenceCalculator(GreekCalculator):
    def _get_bumped_price(
        self, bump_attr: str, bump_val: float, z_matrix=None
    ) -> float:
        params: dict[str, Any] = {
            "s": self.option.S,
            "k": self.option.K,
            "t": self.option.T,
            "r": self.option.r,
            "sigma": self.option.sigma,
            "option_type": self.option.option_type,
            "q": self.option.q,
        }
        if isinstance(self.option, BarrierOption):
            params["barrier_level"] = self.option.barrier_level
            params["barrier_type"] = self.option.barrier_type

        if bump_attr == "t" and (params["t"] + bump_val) <= 0:
            params["t"] = 1e-9
        else:
            params[bump_attr] += bump_val

        option_copy = self.option.__class__(**params)
        pricer_params: dict[str, Any] = {}
        if hasattr(self.pricer, "get_params"):
            pricer_params = self.pricer.get_params()

        pricer_copy = self.pricer.__class__(option_copy, **pricer_params)
        if z_matrix is not None and hasattr(pricer_copy, "z_matrix"):
            pricer_copy.z_matrix = z_matrix
        price_result = pricer_copy.price()
        price = price_result[0] if isinstance(price_result, tuple) else price_result
        return float(price)

    def _calculate_fallback(self) -> dict[str, Any]:
        z_matrix = None
        if isinstance(self.pricer, MonteCarloPricer) and self.pricer.use_crn:
            if self.pricer.z_matrix is None:
                self.pricer.z_matrix = self.pricer._generate_z_matrix()
            z_matrix = self.pricer.z_matrix

        cbrt_epsilon = np.finfo(float).eps ** (1 / 3)
        s_bump = max(self.option.S * cbrt_epsilon, 1e-8)
        sig_bump = max(self.option.sigma * cbrt_epsilon, 1e-8)
        r_bump = max(self.option.r * cbrt_epsilon, 1e-8)
        t_bump = 1 / 365.0

        price_res = self.pricer.price()
        price_base = (
            float(price_res[0]) if isinstance(price_res, tuple) else float(price_res)
        )

        price_s_up = self._get_bumped_price("s", s_bump, z_matrix)
        price_s_down = self._get_bumped_price("s", -s_bump, z_matrix)
        delta = (price_s_up - price_s_down) / (2 * s_bump)
        gamma = (price_s_up - 2 * price_base + price_s_down) / (s_bump**2)

        price_sig_up = self._get_bumped_price("sigma", sig_bump, z_matrix)
        price_sig_down = self._get_bumped_price("sigma", -sig_bump, z_matrix)
        vega = (price_sig_up - price_sig_down) / (2 * sig_bump)

        price_r_up = self._get_bumped_price("r", r_bump, z_matrix)
        price_r_down = self._get_bumped_price("r", -r_bump, z_matrix)
        rho = (price_r_up - price_r_down) / (2 * r_bump)

        price_t_fwd = self._get_bumped_price("t", -t_bump, z_matrix)
        theta = (price_t_fwd - price_base) / t_bump

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega * 0.01,
            "theta": theta / 365.0,
            "rho": rho * 0.01,
        }

    def _calculate_mc_optimized(self) -> dict[str, Any]:
        opt = self.option
        pricer = self.pricer
        if pricer.z_matrix is None:
            pricer.z_matrix = pricer._generate_z_matrix()
        base_paths = pricer._generate_paths()
        is_call = opt.option_type == "call"
        payoffs_base: npt.NDArray[np.float64]

        if isinstance(opt, VanillaOption):
            final_prices = base_paths[:, -1]
            payoffs_base = (
                np.maximum(final_prices - opt.K, 0.0)
                if is_call
                else np.maximum(opt.K - final_prices, 0.0)
            )
        elif isinstance(opt, BarrierOption):
            b_map = {
                "up-and-in": 0,
                "up-and-out": 1,
                "down-and-in": 2,
                "down-and-out": 3,
            }
            payoffs_base = _calculate_barrier_payoffs_jit(
                base_paths, opt.K, opt.barrier_level, b_map[opt.barrier_type], is_call
            )

        discounted_payoffs = payoffs_base * np.exp(-opt.r * opt.T)
        price_base = float(np.mean(discounted_payoffs))

        cbrt_epsilon = np.finfo(float).eps ** (1 / 3)
        s_bump = max(opt.S * cbrt_epsilon, 1e-8)
        sig_bump = max(opt.sigma * cbrt_epsilon, 1e-8)
        t_bump = 1 / 365.0

        s_up, s_down = opt.S + s_bump, opt.S - s_bump
        paths_up = base_paths * (s_up / opt.S)
        paths_down = base_paths * (s_down / opt.S)
        payoffs_up: npt.NDArray[np.float64]
        payoffs_down: npt.NDArray[np.float64]

        if isinstance(opt, VanillaOption):
            payoffs_up = (
                np.maximum(paths_up[:, -1] - opt.K, 0.0)
                if is_call
                else np.maximum(opt.K - paths_up[:, -1], 0.0)
            )
            payoffs_down = (
                np.maximum(paths_down[:, -1] - opt.K, 0.0)
                if is_call
                else np.maximum(opt.K - paths_down[:, -1], 0.0)
            )
        elif isinstance(opt, BarrierOption):
            b_map = {
                "up-and-in": 0,
                "up-and-out": 1,
                "down-and-in": 2,
                "down-and-out": 3,
            }
            payoffs_up = _calculate_barrier_payoffs_jit(
                paths_up, opt.K, opt.barrier_level, b_map[opt.barrier_type], is_call
            )
            payoffs_down = _calculate_barrier_payoffs_jit(
                paths_down, opt.K, opt.barrier_level, b_map[opt.barrier_type], is_call
            )

        price_s_up = float(np.mean(payoffs_up * np.exp(-opt.r * opt.T)))
        price_s_down = float(np.mean(payoffs_down * np.exp(-opt.r * opt.T)))
        delta = (price_s_up - price_s_down) / (2 * s_bump)
        gamma = (price_s_up - 2 * price_base + price_s_down) / (s_bump**2)

        rho_paths = -opt.T * discounted_payoffs
        rho = np.mean(rho_paths)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks = {
                "sig_up": executor.submit(
                    self._get_bumped_price, "sigma", sig_bump, pricer.z_matrix
                ),
                "sig_down": executor.submit(
                    self._get_bumped_price, "sigma", -sig_bump, pricer.z_matrix
                ),
                "t_fwd": executor.submit(
                    self._get_bumped_price, "t", -t_bump, pricer.z_matrix
                ),
            }
            results = {key: future.result() for key, future in tasks.items()}

        vega = (results["sig_up"] - results["sig_down"]) / (2 * sig_bump)
        theta = (results["t_fwd"] - price_base) / t_bump

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega * 0.01,
            "theta": theta / 365.0,
            "rho": rho * 0.01,
        }

    def calculate(self) -> dict[str, Any]:
        if isinstance(self.pricer, MonteCarloPricer) and isinstance(
            self.option, (VanillaOption, BarrierOption)
        ):
            return self._calculate_mc_optimized()
        if isinstance(self.option, BasketOption):
            return {"info": "Finite Difference Greeks are not supported."}
        return self._calculate_fallback()


class PathwiseCalculator(GreekCalculator):
    def calculate(self) -> dict[str, Any]:
        if not isinstance(self.pricer, MonteCarloPricer):
            raise IncompatibleCalculatorError(
                "Pathwise method requires MonteCarloPricer."
            )
        if isinstance(
            self.option, (BasketOption, AsianOption, BarrierOption, AmericanOption)
        ):
            return {"info": "Pathwise method not supported for this option type."}
        if self.pricer.z_matrix is None:
            self.pricer.z_matrix = self.pricer._generate_z_matrix()
        opt = self.option
        delta, gamma, vega, rho_raw, dv_dt_raw = _calculate_pathwise_estimators_jit(
            self.pricer.z_matrix,
            opt.S,
            opt.K,
            opt.r,
            opt.T,
            opt.sigma,
            opt.q,
            (opt.option_type == "call"),
        )
        discount = np.exp(-opt.r * opt.T)
        theta = -dv_dt_raw * discount
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega * 0.01,
            "theta": theta / 365.0,
            "rho": rho_raw * discount * 0.01,
        }


class LikelihoodRatioCalculator(GreekCalculator):
    def calculate(self) -> dict[str, Any]:
        if not isinstance(self.pricer, MonteCarloPricer):
            raise IncompatibleCalculatorError("LRM requires MonteCarloPricer.")
        if isinstance(
            self.option, (BasketOption, AsianOption, BarrierOption, AmericanOption)
        ):
            return {"info": "LRM not supported for this option type."}
        if (
            not hasattr(self.pricer, "discounted_payoffs")
            or self.pricer.discounted_payoffs is None
        ):
            self.pricer.price()
        try:
            z_steps = self.pricer.z_matrix
            payoffs = self.pricer.discounted_payoffs
            if z_steps is None or payoffs is None:
                raise IncompatibleCalculatorError("Pricer failed to store results.")
            z_terminal = np.sum(z_steps, axis=1) / np.sqrt(self.pricer.num_steps)
            delta, vega, rho, theta = _calculate_lrm_greeks_jit(
                payoffs,
                z_terminal,
                self.option.S,
                self.option.r,
                self.option.T,
                self.option.sigma,
            )
            return {
                "delta": delta,
                "vega": vega * 0.01,
                "theta": theta / 365.0,
                "rho": rho * 0.01,
            }
        except (AttributeError, ZeroDivisionError) as e:
            raise IncompatibleCalculatorError(f"LRM failed. Details: {e}")


class LatticeGreekCalculator(GreekCalculator):
    def calculate(self) -> dict[str, Any]:
        if not isinstance(self.pricer, LatticePricer):
            raise IncompatibleCalculatorError(
                "This calculator is only for the LatticePricer."
            )

        is_fallback_case = self.pricer.model != "CRR" or isinstance(
            self.option, (BarrierOption, AsianOption)
        )

        if is_fallback_case:
            fd_calc = FiniteDifferenceCalculator(self.pricer)
            return fd_calc._calculate_fallback()

        opt = self.option
        option_type_code = 1 if opt.option_type == "call" else 0
        exercise_type_code = 1 if isinstance(opt, AmericanOption) else 0

        _, delta, gamma, theta = _lattice_greeks_jit(
            opt.S,
            opt.K,
            opt.T,
            opt.r,
            opt.q,
            opt.sigma,
            self.pricer.num_steps,
            option_type_code,
            exercise_type_code,
        )

        def get_bumped_price(attr, val):
            params = {
                "s": opt.S,
                "k": opt.K,
                "t": opt.T,
                "r": opt.r,
                "sigma": opt.sigma,
                "option_type": opt.option_type,
                "q": opt.q,
            }
            if isinstance(opt, AmericanOption):
                pass
            params[attr] += val
            bumped_opt = opt.__class__(**params)
            pricer_params = self.pricer.get_params()
            bumped_pricer = self.pricer.__class__(bumped_opt, **pricer_params)
            price, _ = bumped_pricer.price()
            return price

        cbrt_epsilon = np.finfo(float).eps ** (1 / 3)
        sig_bump = max(self.option.sigma * cbrt_epsilon, 1e-8)
        r_bump = max(self.option.r * cbrt_epsilon, 1e-8)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks = {
                "sig_up": executor.submit(get_bumped_price, "sigma", sig_bump),
                "sig_down": executor.submit(get_bumped_price, "sigma", -sig_bump),
                "r_up": executor.submit(get_bumped_price, "r", r_bump),
                "r_down": executor.submit(get_bumped_price, "r", -r_bump),
            }
            prices = {key: future.result() for key, future in tasks.items()}

        vega = (prices["sig_up"] - prices["sig_down"]) / (2 * sig_bump)
        rho = (prices["r_up"] - prices["r_down"]) / (2 * r_bump)

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega * 0.01,
            "theta": theta / 365.0,
            "rho": rho * 0.01,
        }


class KemnaVorstGreekCalculator(GreekCalculator):
    def __init__(self, pricer: BasePricer):
        if not isinstance(pricer, KemnaVorstPricer):
            raise IncompatibleCalculatorError(
                "KemnaVorstGreekCalculator is only for KemnaVorstPricer."
            )
        super().__init__(pricer)
        self._precompute()

    def _precompute(self) -> None:
        opt = self.option
        s, k, t, r, q, sigma = opt.S, opt.K, opt.T, opt.r, opt.q, opt.sigma
        self.sigma_a = sigma / np.sqrt(3)
        mu_a = (r - q - 0.5 * sigma**2) / 2
        self.b_a = mu_a + r - q
        self.d1 = (np.log(s / k) + (self.b_a + 0.5 * self.sigma_a**2) * t) / (
            self.sigma_a * np.sqrt(t)
        )
        self.d2 = self.d1 - self.sigma_a * np.sqrt(t)
        self.N_d1 = norm.cdf(self.d1)
        self.N_prime_d1 = norm.pdf(self.d1)
        self.N_d2 = norm.cdf(self.d2)

    def calculate(self) -> dict[str, Any]:
        opt = self.option
        s, k, t, r = opt.S, opt.K, opt.T, opt.r
        delta_call = np.exp((self.b_a - r) * t) * self.N_d1
        gamma = (
            np.exp((self.b_a - r) * t)
            * self.N_prime_d1
            / (s * self.sigma_a * np.sqrt(t))
        )
        vega_term1 = (np.sqrt(t) / np.sqrt(3)) * self.N_prime_d1
        vega_term2 = 0.5 * opt.sigma * t * self.N_d1
        vega = s * np.exp((self.b_a - r) * t) * (vega_term1 - vega_term2)
        theta_term1_call = s * (self.b_a - r) * np.exp((self.b_a - r) * t) * self.N_d1
        theta_term2_call = r * k * np.exp(-r * t) * self.N_d2
        theta_term3 = (
            s
            * np.exp((self.b_a - r) * t)
            * self.N_prime_d1
            * self.sigma_a
            / (2 * np.sqrt(t))
        )
        theta_call = -(theta_term1_call + theta_term2_call + theta_term3)
        rho_term1_call = 1.5 * s * t * np.exp((self.b_a - r) * t) * self.N_d1
        rho_term2_call = s * t * np.exp((self.b_a - r) * t) * self.N_d1
        rho_term3_call = k * t * np.exp(-r * t) * self.N_d2
        rho_call = rho_term1_call - rho_term2_call + rho_term3_call
        if opt.option_type == "call":
            return {
                "delta": delta_call,
                "gamma": gamma,
                "vega": vega * 0.01,
                "theta": theta_call / 365.0,
                "rho": rho_call * 0.01,
            }
        else:
            delta_put = delta_call - np.exp((self.b_a - r) * t)
            theta_put = (
                theta_call
                + k * r * np.exp(-r * t)
                - s * (self.b_a - r) * np.exp((self.b_a - r) * t)
            )
            rho_put = (
                rho_call
                - k * t * np.exp(-r * t)
                - 1.5 * s * t * np.exp((self.b_a - r) * t)
                + s * t * np.exp((self.b_a - r) * t)
            )
            return {
                "delta": delta_put,
                "gamma": gamma,
                "vega": vega * 0.01,
                "theta": theta_put / 365.0,
                "rho": rho_put * 0.01,
            }
