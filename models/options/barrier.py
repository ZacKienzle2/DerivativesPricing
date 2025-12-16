# models/options/barrier.py
# Defines a barrier option contract.

from .base_option import BaseOption


class BarrierOption(BaseOption):
    """
    Represents a barrier option, which is path-dependent.

    This contract becomes activated or extinguished if the underlying
    asset price crosses a predetermined barrier level.
    """

    _valid_barrier_types = {"down-and-in", "up-and-in", "down-and-out", "up-and-out"}

    def __init__(
        self,
        s: float,
        k: float,
        t: float,
        r: float,
        sigma: float,
        option_type: str,
        barrier_level: float,
        barrier_type: str,
        q: float = 0.0,
    ):
        super().__init__(s, k, t, r, sigma, option_type, q)

        # Validation for barrier-specific parameters.
        if barrier_level < 0:
            raise ValueError("Barrier level must be a non-negative number.")

        barrier_type = barrier_type.lower()
        if barrier_type not in self._valid_barrier_types:
            raise ValueError(
                f"Invalid barrier type. Choose from: {self._valid_barrier_types}"
            )

        self.barrier_level = barrier_level
        self.barrier_type = barrier_type
