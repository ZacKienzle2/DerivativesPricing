from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt
from scipy.stats import gmean, qmc, norm
from numba import jit, prange

from ..options import (
    AsianOption,
    BarrierOption,
    BaseOption,
    BasketOption,
    VanillaOption,
)
from .base_pricer import BasePricer
from ..simulation import generate_paths_jit, generate_final_prices_jit


@jit(nopython=True, fastmath=True, parallel=True)
def _calculate_barrier_payoffs_jit(
    paths: npt.NDArray[np.float64],
    strike: float,
    barrier: float,
    barrier_type_code: int,
    is_call: bool,
) -> npt.NDArray[np.float64]:
    """Calculates the payoffs for a barrier option using JIT compilation.

    Args:
        paths (npt.NDArray[np.float64]): Simulated asset price paths.
        strike (float): Strike price of the option.
        barrier (float): Barrier level for the option.
        barrier_type_code (int): Barrier type code (0: up-and-in, 1: up-and-out, 2: down-and-in, 3: down-and-out).
        is_call (bool): True for call option, False for put option.

    Returns:
        npt.NDArray[np.float64]: Array of payoffs for each simulation.
    """
    final_prices = paths[:, -1]
    num_sims = paths.shape[0]
    num_steps = paths.shape[1]
    payoffs = np.zeros(num_sims)

    for i in prange(num_sims):
        crossed = False
        for j in range(1, num_steps):  # Start from 1, skip initial price
            if barrier_type_code == 0 or barrier_type_code == 1:  # Up barriers
                if paths[i, j] >= barrier:
                    crossed = True
                    break
            else:  # Down barriers (codes 2, 3)
                if paths[i, j] <= barrier:
                    crossed = True
                    break

        # Determine if option is active based on barrier type
        if barrier_type_code == 0:  # up-and-in
            valid = crossed
        elif barrier_type_code == 1:  # up-and-out
            valid = not crossed
        elif barrier_type_code == 2:  # down-and-in
            valid = crossed
        else:  # barrier_type_code == 3, down-and-out
            valid = not crossed

        if valid:
            if is_call:
                payoffs[i] = max(final_prices[i] - strike, 0.0)
            else:
                payoffs[i] = max(strike - final_prices[i], 0.0)
    return payoffs


@jit(nopython=True, fastmath=True)
def _calculate_basket_payoffs_jit(
    paths: npt.NDArray[np.float64],
    strike: float,
    weights: npt.NDArray[np.float64],
    is_call: bool,
) -> npt.NDArray[np.float64]:
    """Calculates the payoffs for a basket option using JIT compilation.

    Args:
        paths (npt.NDArray[np.float64]): Simulated asset price paths.
        strike (float): Strike price of the option.
        weights (npt.NDArray[np.float64]): Weights for the basket components.
        is_call (bool): True for call option, False for put option.

    Returns:
        npt.NDArray[np.float64]: Array of payoffs for each simulation.
    """
    final_prices = paths[:, :, -1].copy()
    basket_values = np.zeros(final_prices.shape[1])
    for i in range(final_prices.shape[1]):
        basket_values[i] = np.dot(weights, final_prices[:, i])

    return (
        np.maximum(basket_values - strike, 0.0)
        if is_call
        else np.maximum(strike - basket_values, 0.0)
    )


class MonteCarloPricer(BasePricer):
    def __init__(
        self,
        option: BaseOption,
        num_sims: int,
        num_steps: int,
        variance_reduction: Optional[List[str]] = None,
        use_crn: bool = True,
        **kwargs: Any,
    ):
        super().__init__(option)
        self.num_sims = num_sims
        self.num_steps = num_steps
        self.variance_reduction = variance_reduction or []
        self.use_crn = use_crn
        self.z_matrix: Optional[npt.NDArray[np.float64]] = None
        self.convergence_data: Optional[npt.NDArray[np.float64]] = None
        self.discounted_payoffs: Optional[npt.NDArray[np.float64]] = None

    def get_params(self) -> Dict[str, Any]:
        """Retrieves the parameters for the Monte Carlo simulation.

        Returns:
            Dict[str, Any]: A dictionary containing the simulation parameters.
        """
        return {
            "num_sims": self.num_sims,
            "num_steps": self.num_steps,
            "variance_reduction": self.variance_reduction,
            "use_crn": self.use_crn,
        }

    def _generate_z_matrix(self) -> npt.NDArray[np.float64]:
        """Generates a matrix of standard normal random variables.

        Returns:
            npt.NDArray[np.float64]: A matrix of shape (num_sims, num_steps) containing
            standard normal random variables.
        """
        use_antithetic = "antithetic" in self.variance_reduction

        # Adjust simulation count for antithetic variates
        if use_antithetic and self.num_sims % 2 != 0:
            self.num_sims += 1

        num_base_sims = self.num_sims // 2 if use_antithetic else self.num_sims

        # Generate uniform samples (either pseudo-random or quasi-random)
        if "sobol" in self.variance_reduction:
            sampler = qmc.Sobol(d=self.num_steps, scramble=True)
            # Find the smallest power of 2 greater than or equal to num_base_sims
            m = int(np.ceil(np.log2(num_base_sims)))
            samples = sampler.random_base2(m=m)[:num_base_sims]
        else:
            samples = np.random.rand(num_base_sims, self.num_steps)

        # Apply antithetic transform if enabled
        if use_antithetic:
            samples = np.concatenate([samples, 1 - samples], axis=0)

        # Convert uniform samples to standard normal using the inverse CDF
        z_matrix = norm.ppf(samples)

        # Handle potential infinite values from ppf(0) or ppf(1)
        z_matrix[np.isinf(z_matrix)] = 0.0

        return z_matrix

    def _generate_paths(self) -> npt.NDArray[np.float64]:
        if self.z_matrix is None:
            self.z_matrix = self._generate_z_matrix()
        if isinstance(self.option, BasketOption):
            return self._generate_correlated_paths()
        return generate_paths_jit(
            s0=self.option.S,
            t=self.option.T,
            r=self.option.r,
            q=self.option.q,
            sigma=self.option.sigma,
            num_steps=self.num_steps,
            z_matrix=self.z_matrix,
        )

    def _generate_correlated_paths(self) -> npt.NDArray[np.float64]:
        """Generates correlated paths for the basket option.

        Raises:
            TypeError: If the option is not a BasketOption.

        Returns:
            npt.NDArray[np.float64]: Generated paths of shape (num_assets, num_sims, num_steps + 1).
        """
        opt = self.option
        if not isinstance(opt, BasketOption):
            raise TypeError("For BasketOption only.")

        # Generate correlated random numbers for the basket
        # Note: This part does not use the unified z_matrix from _generate_z_matrix
        # to keep the basket logic separate. QMC/AV could be extended here if needed.
        z = np.random.standard_normal((opt.num_assets, self.num_sims, self.num_steps))
        cholesky = np.linalg.cholesky(opt.corr_matrix)
        correlated_z = np.einsum("ij,jkl->ikl", cholesky, z)

        paths = np.zeros((opt.num_assets, self.num_sims, self.num_steps + 1))
        paths[:, :, 0] = np.array(opt.initial_prices)[:, np.newaxis]
        dt = opt.T / self.num_steps
        vol = np.array(opt.volatilities)[:, np.newaxis]
        for i in range(self.num_steps):
            drift = (opt.r - 0.5 * vol**2) * dt
            diffusion = vol * np.sqrt(dt) * correlated_z[:, :, i]
            paths[:, :, i + 1] = paths[:, :, i] * np.exp(drift + diffusion)
        return paths

    def price(self) -> Tuple[float, float]:
        if self.z_matrix is None and not isinstance(self.option, BasketOption):
            self.z_matrix = self._generate_z_matrix()

        opt, is_call = self.option, self.option.option_type == "call"
        payoffs: npt.NDArray[np.float64]

        if isinstance(opt, VanillaOption):
            final_prices = generate_final_prices_jit(
                s0=opt.S,
                t=opt.T,
                r=opt.r,
                q=opt.q,
                sigma=opt.sigma,
                num_steps=self.num_steps,
                z_matrix=self.z_matrix,
            )
            payoffs = (
                np.maximum(final_prices - opt.K, 0.0)
                if is_call
                else np.maximum(opt.K - final_prices, 0.0)
            )

        elif isinstance(opt, BasketOption):
            paths = self._generate_paths()
            payoffs = _calculate_basket_payoffs_jit(
                paths, opt.K, np.array(opt.weights), is_call
            )

        elif isinstance(opt, BarrierOption):
            paths = self._generate_paths()
            b_map = {
                "up-and-in": 0,
                "up-and-out": 1,
                "down-and-in": 2,
                "down-and-out": 3,
            }
            payoffs = _calculate_barrier_payoffs_jit(
                paths, opt.K, opt.barrier_level, b_map[opt.barrier_type], is_call
            )

        elif isinstance(opt, AsianOption):
            paths = self._generate_paths()
            if opt.avg_type == "arithmetic":
                avg_prices = np.mean(paths[:, 1:], axis=1)
            else:  # Geometric
                avg_prices = gmean(paths[:, 1:], axis=1)
            payoffs = (
                np.maximum(avg_prices - opt.K, 0.0)
                if is_call
                else np.maximum(opt.K - avg_prices, 0.0)
            )

        else:
            raise TypeError(f"MC pricer not implemented for {type(opt).__name__}")

        self.discounted_payoffs = payoffs * np.exp(-opt.r * opt.T)
        assert self.discounted_payoffs is not None
        price = np.mean(self.discounted_payoffs)
        std_err = np.std(self.discounted_payoffs) / np.sqrt(self.num_sims)

        cumulative_avg = np.cumsum(self.discounted_payoffs) / (
            np.arange(self.num_sims) + 1
        )
        self.convergence_data = cumulative_avg.astype(np.float64)

        return float(price), float(std_err)
