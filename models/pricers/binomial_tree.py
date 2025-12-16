# models/pricers/binomial_tree.py
from typing import Dict, Any, Tuple, List
import numpy as np

from models.options import BaseOption, AsianOption, AmericanOption
from .base_pricer import BasePricer


class BinomialTreePricer(BasePricer):
    """
    Prices options using a Cox-Ross-Rubinstein (CRR) binomial tree.
    This pricer can handle European, American, and Asian style options by
    dispatching to the appropriate path-dependent or path-independent solver.
    """

    def __init__(self, option: BaseOption, num_steps: int):
        super().__init__(option)
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("Number of steps must be a positive integer.")
        self.num_steps = num_steps

    def get_params(self) -> Dict[str, Any]:
        return {"num_steps": self.num_steps}

    def _calculate_path_independent_price(self) -> Tuple[float, float]:
        """Prices path-independent options (e.g., European, American)."""
        s, k, t, r, sigma, q = (
            self.option.S,
            self.option.K,
            self.option.T,
            self.option.r,
            self.option.sigma,
            self.option.q,
        )
        is_call = self.option.option_type == "call"
        is_american = isinstance(self.option, AmericanOption)

        dt = t / self.num_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        prob = (np.exp((r - q) * dt) - d) / (u - d)
        exp_r_dt = np.exp(-r * dt)

        j = np.arange(self.num_steps + 1)
        st = s * (u ** (self.num_steps - j)) * (d**j)
        option_values = np.maximum(0, st - k) if is_call else np.maximum(0, k - st)

        for i in range(self.num_steps - 1, -1, -1):
            option_values = (
                prob * option_values[:-1] + (1 - prob) * option_values[1:]
            ) * exp_r_dt
            if is_american:
                j = np.arange(i + 1)
                st = s * (u ** (i - j)) * (d**j)
                exercise_values = st - k if is_call else k - st
                option_values = np.maximum(option_values, exercise_values)

        return float(option_values[0]), 0.0

    def _calculate_path_dependent_price(self) -> Tuple[float, float]:
        """Prices path-dependent options using Hull-White."""
        if not isinstance(self.option, AsianOption):
            raise TypeError("This method is for Asian options only.")

        s, k, t, r, sigma, q = (
            self.option.S,
            self.option.K,
            self.option.T,
            self.option.r,
            self.option.sigma,
            self.option.q,
        )
        is_call = self.option.option_type == "call"
        dt = t / self.num_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        prob = (np.exp((r - q) * dt) - d) / (u - d)
        exp_r_dt = np.exp(-r * dt)

        sums_at_nodes: List[List[List[float]]] = [[[s]]]

        for i in range(1, self.num_steps + 1):
            prev_sums_level = sums_at_nodes[i - 1]
            next_sums_level = []
            for j in range(i + 1):
                price = s * (u**j) * (d ** (i - j))
                node_sums = []
                if j > 0:
                    node_sums.extend([val + price for val in prev_sums_level[j - 1]])
                if j < i:
                    node_sums.extend([val + price for val in prev_sums_level[j]])
                unique_sums = sorted(list(set(node_sums)))
                next_sums_level.append(unique_sums)
            sums_at_nodes.append(next_sums_level)

        payoffs_at_nodes = []
        for j in range(self.num_steps + 1):
            averages = np.array(sums_at_nodes[self.num_steps][j]) / (self.num_steps + 1)
            payoff = (
                np.maximum(0, averages - k) if is_call else np.maximum(0, k - averages)
            )
            payoffs_at_nodes.append(payoff)

        for i in range(self.num_steps - 1, -1, -1):
            next_payoffs_level = []
            for j in range(i + 1):
                current_sums = sums_at_nodes[i][j]
                up_sums_next, down_sums_next = (
                    sums_at_nodes[i + 1][j + 1],
                    sums_at_nodes[i + 1][j],
                )
                up_payoffs, down_payoffs = payoffs_at_nodes[j + 1], payoffs_at_nodes[j]
                price = s * (u**j) * (d ** (i - j))

                up_interp = np.interp(
                    np.array(current_sums) + price * u, up_sums_next, up_payoffs
                )
                down_interp = np.interp(
                    np.array(current_sums) + price * d, down_sums_next, down_payoffs
                )

                expected_values = (
                    prob * up_interp + (1 - prob) * down_interp
                ) * exp_r_dt
                next_payoffs_level.append(expected_values)
            payoffs_at_nodes = next_payoffs_level

        return float(payoffs_at_nodes[0][0]), 0.0

    def price(self) -> Tuple[float, float]:
        """Dispatches to the correct pricing algorithm based on option type."""
        if isinstance(self.option, AsianOption):
            return self._calculate_path_dependent_price()
        return self._calculate_path_independent_price()
