"""Generic CRR or Boyle trinomial lattice pricer."""

from typing import Any, Dict, Tuple

import numpy as np
from numba import jit

from models.options import AmericanOption, BarrierOption, BaseOption, VanillaOption
from .base_pricer import BasePricer


@jit(nopython=True, fastmath=True)
def _lattice_pricer_jit(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    num_steps: int,
    model_code: int,
    option_type_code: int,
    exercise_type_code: int,
    barrier_type_code: int,
    barrier_level: float,
) -> float:
    dt = T / num_steps
    discount = np.exp(-r * dt)

    u, d, pu, pd, pm = 0.0, 0.0, 0.0, 0.0, 0.0
    is_trinomial = False

    if model_code == 0:
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        if abs(u - d) < 1e-12:
            return np.nan
        else:
            pu = (np.exp((r - q) * dt) - d) / (u - d)
        pd = 1.0 - pu
        if not (0.0 < pu < 1.0):
            return np.nan
    elif model_code == 1:
        is_trinomial = True
        dx = sigma * np.sqrt(3 * dt)
        nu = r - q - 0.5 * sigma**2
        pu = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 + (nu * dt) / dx)
        pd = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 - (nu * dt) / dx)
        pm = 1.0 - pu - pd
        u = np.exp(dx)
        d = np.exp(-dx)
        if not (0.0 <= pu <= 1.0 and 0.0 <= pd <= 1.0 and 0.0 <= pm <= 1.0):
            return np.nan

    if is_trinomial:
        num_nodes = 2 * num_steps + 1
        st = S * (u ** np.arange(num_steps, -num_steps - 1, -1))
    else:
        num_nodes = num_steps + 1
        st = (
            S * (u ** np.arange(num_steps, -1, -1)) * (d ** np.arange(0, num_steps + 1))
        )

    option_values = np.zeros(num_nodes)
    if option_type_code == 1:
        option_values[:] = np.maximum(0.0, st - K)
    else:
        option_values[:] = np.maximum(0.0, K - st)

    if barrier_type_code == 2:
        option_values[st >= barrier_level] = 0.0
    elif barrier_type_code == 4:
        option_values[st <= barrier_level] = 0.0

    for i in range(num_steps - 1, -1, -1):
        if is_trinomial:
            st = S * (u ** np.arange(i, -i - 1, -1))
            current_size = 2 * i + 1
            option_values_next_step = option_values[: current_size + 2].copy()
            st_next_step = S * (u ** np.arange(i + 1, -(i + 1) - 1, -1))

            if barrier_type_code == 2:
                option_values_next_step[st_next_step >= barrier_level] = 0.0
            elif barrier_type_code == 4:
                option_values_next_step[st_next_step <= barrier_level] = 0.0

            continuation_value = (
                pu * option_values_next_step[:-2]
                + pm * option_values_next_step[1:-1]
                + pd * option_values_next_step[2:]
            ) * discount
            option_values = option_values[:current_size]
            option_values[:] = continuation_value

        else:
            st = S * (u ** np.arange(i, -1, -1)) * (d ** np.arange(0, i + 1))
            option_values_next_step = option_values[: i + 2].copy()
            st_next_step = (
                S * (u ** np.arange(i + 1, -1, -1)) * (d ** np.arange(0, i + 1 + 1))
            )

            if barrier_type_code == 2:
                option_values_next_step[st_next_step >= barrier_level] = 0.0
            elif barrier_type_code == 4:
                option_values_next_step[st_next_step <= barrier_level] = 0.0

            continuation_value = (
                pu * option_values_next_step[:-1] + pd * option_values_next_step[1:]
            ) * discount
            option_values = option_values[: i + 1]
            option_values[:] = continuation_value

        if exercise_type_code == 1:
            if option_type_code == 1:
                exercise_value = np.maximum(0.0, st - K)
            else:
                exercise_value = np.maximum(0.0, K - st)
            option_values = np.maximum(option_values, exercise_value)

            if barrier_type_code == 2:
                option_values[st >= barrier_level] = 0.0
            elif barrier_type_code == 4:
                option_values[st <= barrier_level] = 0.0

    if np.isnan(option_values[0]):
        return 0.0
    return option_values[0]


class LatticePricer(BasePricer):
    _MODEL_MAP = {"CRR": 0, "Boyle": 1}
    _BARRIER_MAP = {
        "up-and-in": 1,
        "up-and-out": 2,
        "down-and-in": 3,
        "down-and-out": 4,
    }

    def __init__(self, option: BaseOption, num_steps: int, model: str = "CRR"):
        super().__init__(option)
        if not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("Number of steps must be a positive integer.")
        if model not in self._MODEL_MAP:
            raise ValueError(
                f"Unsupported lattice model: {model}. Choose from {list(self._MODEL_MAP.keys())}"
            )
        self.num_steps = num_steps
        self.model = model

    def get_params(self) -> Dict[str, Any]:
        return {"num_steps": self.num_steps, "model": self.model}

    def price(self) -> Tuple[float, float]:
        opt = self.option
        option_type_code = 1 if opt.option_type == "call" else 0
        exercise_type_code = 1 if isinstance(opt, AmericanOption) else 0

        barrier_type_code = 0
        barrier_level = 0.0
        is_in_option = False
        steps_to_use = self.num_steps

        if isinstance(opt, BarrierOption):
            barrier_level = opt.barrier_level
            barrier_type_code = self._BARRIER_MAP.get(opt.barrier_type, 0)
            is_in_option = "in" in opt.barrier_type

            if (opt.barrier_type == "down-and-out" and opt.S <= barrier_level) or (
                opt.barrier_type == "up-and-out" and opt.S >= barrier_level
            ):
                return 0.0, 0.0
            if (opt.barrier_type == "down-and-in" and opt.S <= barrier_level) or (
                opt.barrier_type == "up-and-in" and opt.S >= barrier_level
            ):
                vanilla_opt_class = (
                    AmericanOption if isinstance(opt, AmericanOption) else VanillaOption
                )
                vanilla_opt = vanilla_opt_class(
                    s=opt.S,
                    k=opt.K,
                    t=opt.T,
                    r=opt.r,
                    sigma=opt.sigma,
                    option_type=opt.option_type,
                    q=opt.q,
                )
                vanilla_price = _lattice_pricer_jit(
                    S=vanilla_opt.S,
                    K=vanilla_opt.K,
                    T=vanilla_opt.T,
                    r=vanilla_opt.r,
                    q=vanilla_opt.q,
                    sigma=vanilla_opt.sigma,
                    num_steps=self.num_steps,
                    model_code=self._MODEL_MAP[self.model],
                    option_type_code=option_type_code,
                    exercise_type_code=exercise_type_code,
                    barrier_type_code=0,
                    barrier_level=0.0,
                )
                return max(
                    0.0, vanilla_price if not np.isnan(vanilla_price) else 0.0
                ), 0.0

            # Boyle-Lau Adjustment only for CRR
            if self.model == "CRR" and opt.S > 0 and barrier_level > 0:
                max_time_steps = max(
                    1000, self.num_steps * 5
                )  # Heuristic cap like QuantLib
                divisor = 0.0
                if abs(opt.S - barrier_level) > 1e-9:  # Avoid log(1)
                    if opt.S > barrier_level:
                        divisor = np.log(opt.S / barrier_level) ** 2
                    else:
                        divisor = np.log(barrier_level / opt.S) ** 2

                if divisor > 1e-12:  # Check divisor is not too small
                    optimum_steps_calc = self.num_steps  # Start assuming original steps
                    for i in range(1, self.num_steps):
                        optimum = int(((i**2) * (opt.sigma**2) * opt.T) / divisor)
                        if self.num_steps < optimum:
                            optimum_steps_calc = optimum
                            break
                    steps_to_use = min(optimum_steps_calc, max_time_steps)

        jit_barrier_code = 0
        if barrier_type_code == 1:
            jit_barrier_code = 2
        elif barrier_type_code == 3:
            jit_barrier_code = 4
        elif barrier_type_code in [2, 4]:
            jit_barrier_code = barrier_type_code

        out_price = _lattice_pricer_jit(
            S=opt.S,
            K=opt.K,
            T=opt.T,
            r=opt.r,
            q=opt.q,
            sigma=opt.sigma,
            num_steps=steps_to_use,
            model_code=self._MODEL_MAP[self.model],
            option_type_code=option_type_code,
            exercise_type_code=exercise_type_code,
            barrier_type_code=jit_barrier_code,
            barrier_level=barrier_level,
        )

        if np.isnan(out_price):
            final_price = 0.0
        elif is_in_option:
            vanilla_price = _lattice_pricer_jit(
                S=opt.S,
                K=opt.K,
                T=opt.T,
                r=opt.r,
                q=opt.q,
                sigma=opt.sigma,
                num_steps=steps_to_use,
                model_code=self._MODEL_MAP[self.model],
                option_type_code=option_type_code,
                exercise_type_code=exercise_type_code,
                barrier_type_code=0,
                barrier_level=0.0,
            )
            if np.isnan(vanilla_price):
                final_price = 0.0
            else:
                final_price = max(0.0, vanilla_price - out_price)
        else:
            final_price = max(0.0, out_price)

        return float(final_price), 0.0
