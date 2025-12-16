from typing import Any, Dict, Tuple

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from .base_pricer import BasePricer
from ..options import BaseOption, AmericanOption


class CrankNicolsonPricer(BasePricer):
    def __init__(self, option: BaseOption, num_steps: int, num_points: int):
        super().__init__(option)
        self.n_steps = num_steps
        self.n_points = num_points
        self.is_american = isinstance(option, AmericanOption)

    def get_params(self) -> Dict[str, Any]:
        return {"num_steps": self.n_steps, "num_points": self.n_points}

    def price(self) -> Tuple[float, float]:
        opt = self.option
        k, t, r, q, sigma = opt.K, opt.T, opt.r, opt.q, opt.sigma
        is_call = opt.option_type == "call"

        s_max = 2.0 * k
        s_min = 0.0

        dt = t / self.n_steps
        ds = (s_max - s_min) / self.n_points

        vs = np.linspace(s_min, s_max, self.n_points + 1)
        vi = np.arange(0, self.n_points + 1)
        time_vec = np.arange(0, self.n_steps + 1)

        a = 0.25 * dt * (sigma**2 * (vi**2) - (r - q) * vi)
        b = -0.5 * dt * (sigma**2 * (vi**2) + r)
        c = 0.25 * dt * (sigma**2 * (vi**2) + (r - q) * vi)

        m1_diag = (1 - b)[1:-1]
        m1_upper = -c[1:-2]
        m1_lower = -a[2:-1]
        m1 = diags([m1_lower, m1_diag, m1_upper], [-1, 0, 1], format="csc")

        m2_diag = (1 + b)[1:-1]
        m2_upper = c[1:-2]
        m2_lower = a[2:-1]
        m2 = diags([m2_lower, m2_diag, m2_upper], [-1, 0, 1], format="csc")

        vals = np.zeros((self.n_points + 1, self.n_steps + 1))

        if is_call:
            vals[:, -1] = np.maximum(vs - k, 0)
            vals[0, :] = 0.0
            vals[-1, :] = s_max - k * np.exp(-r * dt * (self.n_steps - time_vec))
        else:
            vals[:, -1] = np.maximum(k - vs, 0)
            vals[0, :] = k * np.exp(-r * dt * (self.n_steps - time_vec))
            vals[-1, :] = 0.0

        solver = splu(m1)

        for j in range(self.n_steps - 1, -1, -1):
            b_vec = m2.dot(vals[1:-1, j + 1])
            vals[1:-1, j] = solver.solve(b_vec)

            if self.is_american:
                if is_call:
                    exercise_val = np.maximum(vs[1:-1] - k, 0)
                else:
                    exercise_val = np.maximum(k - vs[1:-1], 0)

                vals[1:-1, j] = np.maximum(vals[1:-1, j], exercise_val)

        price = np.interp(opt.S, vs, vals[:, 0])

        return float(price), 0.0
