# models/options/vanilla.py
# Defines a standard European-style option contract.

from .base_option import BaseOption


class VanillaOption(BaseOption):
    """
    Represents a standard vanilla European option.

    This option type has no additional parameters beyond the base contract
    and can only be exercised at expiration.
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
