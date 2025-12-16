# models/options/base_option.py
# Abstract Base Class for all option contracts.

from abc import ABC


class BaseOption(ABC):
    """
    An abstract base class for option contracts.

    This class acts as a data container, initializing and validating
    the core parameters shared by all option types. It does not
    contain any pricing logic itself.
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
        """
        Initializes and validates the core option parameters.

        Args:
            s (float): Current price of the underlying asset.
            k (float): Strike price of the option.
            t (float): Time to expiration in years.
            r (float): Risk-free interest rate (annualized).
            sigma (float): Volatility of the underlying (annualized).
            option_type (str): The type of option, 'call' or 'put'.
            q (float): The dividend yield of the underlying (annualized).
        """
        # Universal input validation for all option types
        params = [s, k, t, r, sigma, q]
        if not all(i >= 0 for i in params):
            raise ValueError("Core numeric parameters must be non-negative numbers.")

        if t <= 0:  # Avoid division-by-zero or log(0) issues in pricers
            t = 1e-9  # Use a small epsilon for zero time to expiry

        option_type = option_type.lower()
        if option_type not in ["call", "put"]:
            raise ValueError("Option type must be 'call' or 'put'.")

        self.S = s
        self.K = k
        self.T = t
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.q = q
