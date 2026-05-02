from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import concurrent.futures
import numpy as np
import numpy.typing as npt
from numba import jit
from scipy.stats import norm

from models.options import (
    VanillaOption,
    BarrierOption,
    BasketOption,
    AsianOption,
    AmericanOption,
)
from models.pricers.black_scholes import BlackScholesPricer
from models.pricers.base_pricer import BasePricer
from models.pricers.monte_carlo import MonteCarloPricer, _calculate_barrier_payoffs_jit
from models.pricers.lattice_pricer import LatticePricer
from models.pricers.kemna_vorst import KemnaVorstPricer


@jit(nopython=True, fastmath=True)
def _calculate_pathwise_estimators_jit(
    z_matrix: npt.NDArray[np.float64],
    s0: float,
    k: float,
    r: float,
    t: float,
    sigma: float,
    q: float,
    is_call: bool,
) -> Tuple[npt.NDArray[np.float64], float, float, float, float, float]:
    num_sims, num_steps = z_matrix.shape
    dt = t / num_steps
    st_base = np.empty(num_sims, dtype=np.float64)
    for i in range(num_sims):
        s_t = s0
        for j in range(num_steps):
            drift = (r - q - 0.5 * sigma**2) * dt
            diffusion = sigma * np.sqrt(dt) * z_matrix[i, j]
            s_t *= np.exp(drift + diffusion)
        st_base[i] = s_t
    payoff_base = (
        np.maximum(st_base - k, 0.0) if is_call else np.maximum(k - st_base, 0.0)
    )
    d_payoff_d_s = (
        (st_base > k).astype(np.float64)
        if is_call
        else (st_base < k).astype(np.float64) * -1.0
    )
    delta = np.mean(d_payoff_d_s * st_base / s0)
    z_terminal_sum = np.sum(z_matrix, axis=1)
    w_t = z_terminal_sum * np.sqrt(dt)
    d_st_d_sigma = st_base * (w_t - sigma * t)
    vega = np.mean(d_payoff_d_s * d_st_d_sigma)
    d_st_d_r = st_base * t
    rho_paths = d_payoff_d_s * d_st_d_r - t * payoff_base
    rho = np.mean(rho_paths)
    drift_term_theta = r - q - 0.5 * sigma**2
    diffusion_term_theta = sigma * w_t / (2 * t) if t > 1e-9 else 0.0 * w_t
    d_st_d_t = st_base * (drift_term_theta + diffusion_term_theta)
    dv_dt_paths = -r * payoff_base + (d_payoff_d_s * d_st_d_t)
    dv_dt = np.mean(dv_dt_paths)
    gamma = np.nan
    return payoff_base, delta, gamma, vega, rho, dv_dt


@jit(nopython=True, fastmath=True)
def _calculate_lrm_greeks_jit(
    discounted_payoffs: npt.NDArray[np.float64],
    z_terminal: npt.NDArray[np.float64],
    s0: float,
    r: float,
    t: float,
    sigma: float,
) -> Tuple[float, float, float, float]:
    sqrt_t = np.sqrt(t)
    sigma_sqrt_t = sigma * sqrt_t
    w_s0 = z_terminal / (sigma_sqrt_t * s0)
    w_sigma = (z_terminal**2 - 1) / sigma - z_terminal * sqrt_t
    w_r = z_terminal * sqrt_t / sigma
    w_t = (r - 0.5 * sigma**2) * z_terminal / (sigma_sqrt_t) + (z_terminal**2 - 1) / (
        2 * t
    )
    delta = np.mean(discounted_payoffs * w_s0)
    vega = np.mean(discounted_payoffs * w_sigma)
    rho = np.mean(discounted_payoffs * w_r)
    theta = -np.mean(discounted_payoffs * w_t)
    return delta, vega, rho, theta


@jit(nopython=True, fastmath=True)
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
) -> Tuple[float, float, float, float]:
    if num_steps < 2:
        return np.nan, np.nan, np.nan, np.nan

    dt = T / num_steps
    discount = np.exp(-r * dt)
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    pu = (np.exp((r - q) * dt) - d) / (u - d)
    pd = 1.0 - pu

    st = S * (u ** np.arange(num_steps, -1, -1)) * (d ** np.arange(0, num_steps + 1))
    option_values = (
        np.maximum(0.0, st - K) if option_type_code == 1 else np.maximum(0.0, K - st)
    )

    v_nodes_at_t2 = np.empty(0, dtype=np.float64)

    for i in range(num_steps - 1, -1, -1):
        # Store the node values at step i=2 (time t-2dt) to calculate Gamma and Theta
        if i == 1:
            v_nodes_at_t2 = option_values.copy()

        st = S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i + 1))
        option_values_slice = option_values[: i + 2]
        continuation_value = (
            pu * option_values_slice[:-1] + pd * option_values_slice[1:]
        ) * discount
        option_values = option_values[: i + 1]
        option_values[:] = continuation_value

        if exercise_type_code == 1:  # American Exercise
            exercise_value = st - K if option_type_code == 1 else K - st
            option_values = np.maximum(option_values, exercise_value)

    price = option_values[0]

    # Greeks from nodes. v_nodes_at_t2 contains [V_uu, V_ud, V_dd]
    v_up_up = v_nodes_at_t2[0]
    v_up_down = v_nodes_at_t2[1]
    v_down_down = v_nodes_at_t2[2]

    # More stable delta calculation from step 2 nodes
    delta = (v_up_up - v_down_down) / (S * u**2 - S * d**2)

    # Standard Gamma calculation from step 2 nodes
    gamma_num = ((v_up_up - v_up_down) / (S * u**2 - S)) - (
        (v_up_down - v_down_down) / (S - S * d**2)
    )
    gamma_den = 0.5 * (S * u**2 - S * d**2)
    gamma = gamma_num / gamma_den if gamma_den != 0 else np.nan

    # Theta calculation using the central node from step 2
    theta = (v_up_down - price) / (2 * dt)

    return price, delta, gamma, theta


class IncompatibleCalculatorError(TypeError):
    pass


class GreekCalculator(ABC):
    def __init__(self, pricer: BasePricer):
        self.pricer = pricer
        self.option = pricer.option

    @abstractmethod
    def calculate(self) -> Dict[str, Any]:
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

    def calculate(self) -> Dict[str, Any]:
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
        params: Dict[str, Any] = {
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
        pricer_params: Dict[str, Any] = {}
        if hasattr(self.pricer, "get_params"):
            pricer_params = self.pricer.get_params()

        pricer_copy = self.pricer.__class__(option_copy, **pricer_params)
        if z_matrix is not None and hasattr(pricer_copy, "z_matrix"):
            pricer_copy.z_matrix = z_matrix
        price_result = pricer_copy.price()
        price = price_result[0] if isinstance(price_result, tuple) else price_result
        return float(price)

    def _calculate_fallback(self) -> Dict[str, Any]:
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

    def _calculate_mc_optimized(self) -> Dict[str, Any]:
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

    def calculate(self) -> Dict[str, Any]:
        if isinstance(self.pricer, MonteCarloPricer) and isinstance(
            self.option, (VanillaOption, BarrierOption)
        ):
            return self._calculate_mc_optimized()
        if isinstance(self.option, BasketOption):
            return {"info": "Finite Difference Greeks are not supported."}
        return self._calculate_fallback()


class PathwiseCalculator(GreekCalculator):
    def calculate(self) -> Dict[str, Any]:
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
        _, delta, gamma, vega, rho_raw, dv_dt_raw = _calculate_pathwise_estimators_jit(
            z_matrix=self.pricer.z_matrix,
            s0=opt.S,
            k=opt.K,
            r=opt.r,
            t=opt.T,
            sigma=opt.sigma,
            q=opt.q,
            is_call=(opt.option_type == "call"),
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
    def calculate(self) -> Dict[str, Any]:
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
    def calculate(self) -> Dict[str, Any]:
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

        price, delta, gamma, theta = _lattice_greeks_jit(
            S=opt.S,
            K=opt.K,
            T=opt.T,
            r=opt.r,
            q=opt.q,
            sigma=opt.sigma,
            num_steps=self.pricer.num_steps,
            option_type_code=option_type_code,
            exercise_type_code=exercise_type_code,
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

    def calculate(self) -> Dict[str, Any]:
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
