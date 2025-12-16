# models/options/american.py
# Defines an American-style option contract.

from .base_option import BaseOption


class AmericanOption(BaseOption):
    """
    Represents an American-style option.

    This class distinguishes the contract as having the right of early
    exercise, which must be handled by an appropriate pricing model.
    """

    def __init__(
        self,
        s: float,
        k: float,
        t: float,
        r: float,
        sigma: float,
        option_type: str,
        q: float = 0.0,
    ):
        # Initialise via the parent class.
        super().__init__(s, k, t, r, sigma, option_type, q)
