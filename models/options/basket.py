"""Basket option contract on a portfolio of assets."""


import numpy as np
import numpy.typing as npt

from .base_option import BaseOption


class BasketOption(BaseOption):
    """
    Represents a basket option on a weighted average of multiple assets.
    This implementation uses NumPy for efficient numerical operations.
    """

    __slots__ = (
        "num_assets",
        "initial_prices",
        "volatilities",
        "weights",
        "corr_matrix",
    )

    def __init__(
        self,
        initial_prices: list[float] | npt.NDArray[np.float64],
        k: float,
        t: float,
        r: float,
        volatilities: list[float] | npt.NDArray[np.float64],
        option_type: str,
        weights: list[float] | npt.NDArray[np.float64],
        corr_matrix: npt.NDArray[np.float64],
    ):
        prices_arr = np.asarray(initial_prices, dtype=float)
        vol_arr = np.asarray(volatilities, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)
        num_assets = prices_arr.size

        # Validation for basket-specific parameters
        if not (vol_arr.size == num_assets and weights_arr.size == num_assets):
            raise ValueError(
                "Lists for prices, volatilities, and weights must have the same length."
            )
        if corr_matrix.shape != (num_assets, num_assets):
            raise ValueError(
                "Correlation matrix must be square with dimensions equal to "
                "the number of assets."
            )
        if not np.isclose(weights_arr.sum(), 1.0):
            raise ValueError("Weights must sum to 1.0.")

        # Calculate initial basket price using a dot product.
        basket_s = np.dot(prices_arr, weights_arr)

        # Pass placeholder for sigma, as pricers
        # will use the correlation matrix.
        super().__init__(s=basket_s, k=k, t=t, r=r, sigma=0.0, option_type=option_type)

        # Store the multi-asset parameters as NumPy arrays.
        self.num_assets = num_assets  # Needed by pricers.
        self.initial_prices = prices_arr
        self.volatilities = vol_arr
        self.weights = weights_arr
        self.corr_matrix = np.asarray(corr_matrix, dtype=float)
