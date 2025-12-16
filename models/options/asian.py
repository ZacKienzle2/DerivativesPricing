# models/options/asian.py
from typing import Literal
from .base_option import BaseOption


class AsianOption(BaseOption):
    """
    Represents an Asian option, whose payoff depends on the average
    price of the underlying asset over a specified period.
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
        avg_type: Literal["arithmetic", "geometric"] = "arithmetic",
    ):
        """
        Initializes the Asian option contract.

        Args:
            avg_type (str): The type of averaging
            ('arithmetic' or 'geometric').
        """
        super().__init__(s, k, t, r, sigma, option_type, q)

        if avg_type not in ["arithmetic", "geometric"]:
            raise ValueError("avg_type must be 'arithmetic' or 'geometric'.")
        self.avg_type = avg_type
