from typing import Any, Dict, Optional, Tuple
import numpy as np
import numpy.typing as npt
from numba import jit
from ..options import BaseOption
from .base_pricer import BasePricer
from ..simulation import generate_paths_jit


@jit(nopython=True, fastmath=True)
def _backward_induction_jit(
    paths: npt.NDArray[np.float64],
    k: float,
    r: float,
    dt: float,
    is_call: bool,
    poly_degree: int,
) -> npt.NDArray[np.float64]:
    """Performs backward induction for option pricing.

    Args:
        paths (npt.NDArray[np.float64]): Simulated asset price paths.
        k (float): Strike price of the option.
        r (float): Risk-free interest rate.
        dt (float): Time increment.
        is_call (bool): True if the option is a call option, False if it's a put option.
        poly_degree (int): Degree of the polynomial for regression.

    Returns:
        npt.NDArray[np.float64]: Cash flows at each time step.
    """
    num_sims, num_steps = paths.shape[0], paths.shape[1] - 1
    cash_flows = (
        np.maximum(paths[:, -1] - k, 0.0)
        if is_call
        else np.maximum(k - paths[:, -1], 0.0)
    )
    for t in range(num_steps - 1, 0, -1):
        cash_flows *= np.exp(-r * dt)
        exercise_value = np.maximum(
            paths[:, t] - k if is_call else k - paths[:, t], 0.0
        )
        in_money_mask = exercise_value > 0
        if np.sum(in_money_mask) > poly_degree:
            x, y = paths[in_money_mask, t], cash_flows[in_money_mask]
            vandermonde = np.vander(x, N=poly_degree + 1)
            coeffs = np.linalg.solve(vandermonde.T @ vandermonde, vandermonde.T @ y)
            continuation = np.zeros_like(x)
            for i in range(len(coeffs)):
                continuation += coeffs[i] * (x ** (poly_degree - i))
            exercise_idx = np.where(exercise_value[in_money_mask] > continuation)[0]
            update_idx = np.where(in_money_mask)[0][exercise_idx]
            cash_flows[update_idx] = exercise_value[update_idx]
    return cash_flows


class LongstaffSchwartzPricer(BasePricer):
    def __init__(
        self,
        option: BaseOption,
        num_sims: int,
        num_steps: int,
        poly_degree: int = 3,
        use_crn: bool = True,
        z_matrix: Optional[npt.NDArray[np.float64]] = None,
    ):
        super().__init__(option)
        self.num_sims, self.num_steps = num_sims, num_steps
        self.poly_degree, self.use_crn = poly_degree, use_crn
        self.z_matrix: Optional[npt.NDArray[np.float64]] = z_matrix
        self.convergence_data: Optional[npt.NDArray[np.float64]] = None

    def get_params(self) -> Dict[str, Any]:
        """Returns the parameters used for the Longstaff-Schwartz pricer.

        Returns:
            Dict[str, Any]: A dictionary containing the pricer parameters.
        """
        return {
            "num_sims": self.num_sims,
            "num_steps": self.num_steps,
            "poly_degree": self.poly_degree,
            "use_crn": self.use_crn,
            "z_matrix": self.z_matrix,
        }

    def _generate_z_matrix(self) -> npt.NDArray[np.float64]:
        """Generates the Z matrix for the Longstaff-Schwartz pricer.

        Returns:
            npt.NDArray[np.float64]: The generated Z matrix.
        """
        return np.random.standard_normal((self.num_sims, self.num_steps))

    def price(self) -> Tuple[float, float]:
        """Prices the option using the Longstaff-Schwartz method.

        Returns:
            Tuple[float, float]: The option price and standard error.
        """
        if self.z_matrix is None:
            self.z_matrix = self._generate_z_matrix()
        opt = self.option
        dt = opt.T / self.num_steps
        paths = generate_paths_jit(
            opt.S, opt.T, opt.r, opt.q, opt.sigma, self.num_steps, self.z_matrix
        )
        cash_flows_t1 = _backward_induction_jit(
            paths=paths,
            k=opt.K,
            r=opt.r,
            dt=dt,
            is_call=(opt.option_type == "call"),
            poly_degree=self.poly_degree,
        )
        discounted_cfs = cash_flows_t1 * np.exp(-opt.r * dt)
        price = np.mean(discounted_cfs)
        std_err = np.std(discounted_cfs) / np.sqrt(self.num_sims)
        self.convergence_data = np.cumsum(discounted_cfs) / (
            np.arange(self.num_sims) + 1
        )
        return float(price), float(std_err)
