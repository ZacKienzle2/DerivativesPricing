"""Base process interface."""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as npt


class BaseProcess(ABC):
    """Abstract underlying-asset dynamics.

    Concrete processes implement `simulate_paths` and report their per-step
    Z-matrix dimensionality through `noise_dim`. The Monte Carlo pricer
    treats any concrete process as a black-box path generator.
    """

    @property
    @abstractmethod
    def noise_dim(self) -> int:
        """Number of independent normals required per timestep per path."""

    @abstractmethod
    def simulate_paths(
        self,
        num_sims: int,
        num_steps: int,
        t: float,
        z: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Generates asset-price paths.

        Args:
            num_sims: Number of paths.
            num_steps: Number of discrete time-steps.
            t: Total simulation horizon.
            z: Driver normals, shape `(num_sims, num_steps, noise_dim)` for
                multi-noise models or `(num_sims, num_steps)` when
                `noise_dim == 1`.

        Returns:
            Path tensor of shape `(num_sims, num_steps + 1)`.
        """

    @property
    @abstractmethod
    def s0(self) -> float:
        """Initial asset price."""

    @property
    @abstractmethod
    def r(self) -> float:
        """Risk-free rate (used for discounting)."""

    @property
    def signature(self) -> Tuple:
        """Hashable identifier for caching purposes."""
        return (type(self).__name__, self.s0, self.r)
