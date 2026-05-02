"""Payoff functions for financial instruments at expiration."""

import numpy as np
import numpy.typing as npt


def long_call(
    s: npt.NDArray[np.float64], k: float, premium: float
) -> npt.NDArray[np.float64]:
    """Calculates the profit/loss (P/L) from a long call option position.

    The P/L is the payoff max(S - K, 0) minus the premium paid.

    Args:
        s (npt.NDArray[np.float64]):
            Array of underlying asset prices at expiration.
        k (float): The strike price of the option.
        premium (float):
            The price (premium) paid to acquire the option.

    Returns:
        npt.NDArray[np.float64]:
            The profit/loss for each corresponding underlying price.
    """
    return np.maximum(s - k, 0) - premium


def short_call(
    s: npt.NDArray[np.float64], k: float, premium: float
) -> npt.NDArray[np.float64]:
    """Calculates the profit/loss (P/L) from a short call option position.

    The P/L is the premium received minus the payoff obligation max(S - K, 0).

    Args:
        s (npt.NDArray[np.float64]):
            Array of underlying asset prices at expiration.
        k (float): The strike price of the option.
        premium (float):
            The price (premium) received for selling the option.

    Returns:
        npt.NDArray[np.float64]:
            The profit/loss for each corresponding underlying price.
    """
    return premium - np.maximum(s - k, 0)


def long_put(
    s: npt.NDArray[np.float64], k: float, premium: float
) -> npt.NDArray[np.float64]:
    """Calculates the profit/loss (P/L) from a long put option position.

    The P/L is the payoff max(K - S, 0) minus the premium paid.

    Args:
        s (npt.NDArray[np.float64]):
            Array of underlying asset prices at expiration.
        k (float): The strike price of the option.
        premium (float):
            The price (premium) paid to acquire the option.

    Returns:
        npt.NDArray[np.float64]:
            The profit/loss for each corresponding underlying price.
    """
    return np.maximum(k - s, 0) - premium


def short_put(
    s: npt.NDArray[np.float64], k: float, premium: float
) -> npt.NDArray[np.float64]:
    """Calculates the profit/loss (P/L) from a short put option position.

    The P/L is the premium received minus the payoff obligation max(K - S, 0).

    Args:
        s (npt.NDArray[np.float64]):
            Array of underlying asset prices at expiration.
        k (float): The strike price of the option.
        premium (float):
            The price (premium) received for selling the option.

    Returns:
        npt.NDArray[np.float64]:
            The profit/loss for each corresponding underlying price.
    """
    return premium - np.maximum(k - s, 0)


def long_stock(
    s: npt.NDArray[np.float64], purchase_price: float
) -> npt.NDArray[np.float64]:
    """Calculates the profit/loss (P/L) from a long stock position.

    The P/L is the current price (S) minus the initial purchase price.

    Args:
        s (npt.NDArray[np.float64]):
            Array of underlying asset prices at expiration.
        purchase_price (float):
            The price at which the stock was purchased.

    Returns:
        npt.NDArray[np.float64]:
            The profit/loss for each corresponding underlying price.
    """
    return s - purchase_price


def short_stock(
    s: npt.NDArray[np.float64], sale_price: float
) -> npt.NDArray[np.float64]:
    """Calculates the profit/loss (P/L) from a short stock position.

    The P/L is the initial sale price minus the current price (S).

    Args:
        s (npt.NDArray[np.float64]):
            Array of underlying asset prices at expiration.
        sale_price (float):
            The price at which the stock was sold short.

    Returns:
        npt.NDArray[np.float64]:
            The profit/loss for each corresponding underlying price.
    """
    return sale_price - s


def long_bond(
    s_range: npt.NDArray[np.float64], future_value: float, price: float
) -> npt.NDArray[np.float64]:
    """Calculates the profit/loss (P/L) from a long bond position.

    Calculates the P/L for a zero-coupon bond held to maturity.
    Note: The payoff is constant and independent of the underlying `s_range`.

    Args:
        s_range (npt.NDArray[np.float64]):
            Array of underlying asset prices, used only to determine
            the shape of the output array.
        future_value (float):
            The face value (par) of the bond received at maturity.
        price (float): The price paid to acquire the bond.

    Returns:
        npt.NDArray[np.float64]: A constant array of the bond's profit/loss.
    """
    profit = future_value - price
    return np.full_like(s_range, profit)


def short_bond(
    s_range: npt.NDArray[np.float64], future_value: float, price: float
) -> npt.NDArray[np.float64]:
    """Calculates the profit/loss (P/L) from a short bond position.

    Calculates the P/L for shorting a zero-coupon bond held to maturity.
    Note: The payoff is constant and independent of the underlying `s_range`.

    Args:
        s_range (npt.NDArray[np.float64]):
            Array of underlying asset prices, used only to determine
            the shape of the output array.
        future_value (float):
            The face value (par) of the bond to be paid at maturity.
        price (float): The price received for short-selling the bond.

    Returns:
        npt.NDArray[np.float64]: A constant array of the bond's profit/loss.
    """
    profit = price - future_value
    return np.full_like(s_range, profit)
