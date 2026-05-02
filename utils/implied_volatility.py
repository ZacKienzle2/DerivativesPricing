"""Black-Scholes implied vol fitter and Yahoo option chain loader."""

import numpy as np
import pandas as pd
import yfinance as yf  # type: ignore
from scipy.stats import norm  # type: ignore
from typing import Tuple, Dict, Any
from numpy.typing import NDArray
from numpy import float64, bool_, floating


def _n_cdf(x: NDArray[floating[Any]]) -> NDArray[floating[Any]]:
    """Calculates the standard normal cumulative distribution function (CDF)."""
    return norm.cdf(x)


def _n_pdf(x: NDArray[floating[Any]]) -> NDArray[floating[Any]]:
    """Calculates the standard normal probability density function (PDF)."""
    return norm.pdf(x)


def calculate_implied_volatility(
    option_type: NDArray[float64],
    market_price: NDArray[float64],
    spot_price: float,
    strike_price: NDArray[float64],
    time_to_expiry: NDArray[float64],
    risk_free_rate: NDArray[float64],
    dividend_yield: NDArray[float64],
    max_iterations: int = 20,
    tolerance: float = 1e-9,
) -> Tuple[NDArray[float64], NDArray[bool_]]:
    """Calculates Black-Scholes implied volatility using Newton-Raphson.

    This function calculate the implied volatility for an entire array of
    options at once. It first converts all puts to their call equivalents
    using put-call parity, then filters out any contracts that violate
    no-arbitrage bounds before solving.

    Args:
        option_type: An array indicating the option type. 1.0 for calls,
            -1.0 for puts.
        market_price: The observed market price of each option.
        spot_price: The current price of the underlying asset.
        strike_price: The strike price of each option.
        time_to_expiry: The time to expiry for each option, in years.
        risk_free_rate: The risk-free interest rate.
        dividend_yield: The continuous dividend yield of the underlying.
        max_iterations: The maximum number of iterations for the solver.
        tolerance: The convergence tolerance for the price error.

    Returns:
        A tuple containing:
            - An array of the calculated implied volatilities (sigma).
            - A boolean array indicating which calculations converged.
    """
    params = (
        market_price,
        strike_price,
        time_to_expiry,
        risk_free_rate,
        dividend_yield,
        option_type,
    )
    market_price, strike_price, time_to_expiry, r, q, cp = map(np.asarray, params)

    if market_price.size == 0:
        return np.array([]), np.array([])

    # Convert all puts to their call equivalents using put-call parity.
    # This simplifies the solver to only handle call option pricing.
    is_put = cp == -1
    call_equiv_price = np.copy(market_price)
    call_equiv_price[is_put] = (
        market_price[is_put]
        + spot_price * np.exp(-q[is_put] * time_to_expiry[is_put])
        - strike_price[is_put] * np.exp(-r[is_put] * time_to_expiry[is_put])
    )

    # Filter out contracts that violate no-arbitrage bounds.
    # A call's price must be above its intrinsic value but below the spot.
    intrinsic_val = np.maximum(
        0,
        spot_price * np.exp(-q * time_to_expiry)
        - strike_price * np.exp(-r * time_to_expiry),
    )
    is_valid = (call_equiv_price >= intrinsic_val) & (
        call_equiv_price < spot_price * np.exp(-q * time_to_expiry)
    )

    # Initialise result arrays.
    sigma = np.full_like(market_price, np.nan)
    converged = np.zeros_like(market_price, dtype=bool)

    if not np.any(is_valid):
        return sigma, converged

    # Prepare calculation arrays with only the valid contracts.
    p_calc = call_equiv_price[is_valid]
    k_calc = strike_price[is_valid]
    t_calc = time_to_expiry[is_valid]
    r_calc = r[is_valid]
    q_calc = q[is_valid]

    # Use a standard formula for an initial volatility guess.
    sigma_guess = np.sqrt(2 * np.pi / t_calc) * (p_calc / spot_price)
    sigma_guess = np.nan_to_num(sigma_guess, nan=0.2, posinf=0.2, neginf=0.2)
    sigma_iter = sigma_guess

    # Initialise variables used inside the loop.
    price_bs = np.zeros_like(p_calc)

    # Newton-Raphson iteration loop.
    for _ in range(max_iterations):
        # Black-Scholes formula components.
        d1 = (
            np.log(spot_price / k_calc)
            + (r_calc - q_calc + 0.5 * sigma_iter**2) * t_calc
        ) / (sigma_iter * np.sqrt(t_calc) + 1e-9)
        d2 = d1 - sigma_iter * np.sqrt(t_calc)

        price_bs = spot_price * np.exp(-q_calc * t_calc) * _n_cdf(d1) - k_calc * np.exp(
            -r_calc * t_calc
        ) * _n_cdf(d2)
        vega = spot_price * np.exp(-q_calc * t_calc) * _n_pdf(d1) * np.sqrt(t_calc)

        # Calculate the error and check for convergence.
        price_error = price_bs - p_calc
        if not np.any(np.abs(price_error) > tolerance):
            break

        # Newton-Raphson update step.
        sigma_iter -= price_error / (vega + 1e-9)

    # Final check on convergence and update the results array.
    final_error = price_bs - p_calc
    final_converged = np.abs(final_error) <= tolerance
    sigma_iter[~final_converged] = np.nan

    sigma[is_valid] = sigma_iter
    converged[is_valid] = final_converged

    return sigma, converged


def fetch_and_prepare_data(
    ticker: str,
    risk_free_rate: float,
    dividend_yield: float,
    num_expiries: int = 12,
    strike_range_pct: float = 0.20,
) -> Dict[str, Any]:
    """Fetches, filters, and prepares option data from Yahoo Finance.

    This function retrieves all available option chains for a given number
    of expiry dates, then applies a series of professional-grade filters to
    isolate the most liquid and reliable contracts for analysis.

    Args:
        ticker: The stock ticker symbol, e.g., 'SPY'.
        risk_free_rate: The constant risk-free interest rate to use.
        dividend_yield: The constant continuous dividend yield.
        num_expiries: The number of expiry dates to fetch from the chain.
        strike_range_pct: The percentage range around the spot price for
            filtering strikes (e.g., 0.20 for +/- 20%).

    Returns:
        A dictionary containing NumPy arrays of the prepared option data,
        ready for the volatility solver. Returns an empty dictionary if
        no valid data can be fetched or filtered.
    """
    yf_ticker = yf.Ticker(ticker)
    spot_price = yf_ticker.history(period="1d")["Close"].iloc[-1]
    expiries = yf_ticker.options[:num_expiries]

    # Sequentially fetch data for each expiry date.
    all_options_df = pd.DataFrame()
    for expiry_str in expiries:
        chain = yf_ticker.option_chain(expiry_str)
        chain_df = pd.concat([chain.calls, chain.puts], ignore_index=True)
        all_options_df = pd.concat([all_options_df, chain_df], ignore_index=True)

    if all_options_df.empty:
        return {}

    # Calculate metrics needed for filtering.
    all_options_df["mid_price"] = (all_options_df["bid"] + all_options_df["ask"]) / 2
    all_options_df["spread_pct"] = np.where(
        all_options_df["mid_price"] > 0,
        (all_options_df["ask"] - all_options_df["bid"]) / all_options_df["mid_price"],
        np.inf,
    )

    # Filter for liquid contracts within the desired strike range.
    min_strike = spot_price * (1 - strike_range_pct)
    max_strike = spot_price * (1 + strike_range_pct)
    is_liquid = (
        (all_options_df["volume"].fillna(0) > 5)
        & (all_options_df["openInterest"].fillna(0) > 10)
        & (all_options_df["bid"] > 0.01)
        & (all_options_df["ask"] > 0.01)
        & (all_options_df["spread_pct"] < 0.50)
        & (all_options_df["strike"] >= min_strike)
        & (all_options_df["strike"] <= max_strike)
    )
    liquid_df = all_options_df[is_liquid].copy()

    if liquid_df.empty:
        return {}

    # Calculate time to expiry in years.
    today = pd.to_datetime("today").normalize()
    date_str = liquid_df["contractSymbol"].str[len(ticker) : len(ticker) + 6]
    liquid_df["expirationDate"] = pd.to_datetime(date_str, format="%y%m%d")
    liquid_df["time_to_expiry"] = (liquid_df["expirationDate"] - today).dt.days / 365.0
    liquid_df = liquid_df[liquid_df["time_to_expiry"] > 0.001]

    # Return data in the dictionary format expected by the solver.
    return {
        "option_type": np.where(
            liquid_df["contractSymbol"].str.contains("C"), 1.0, -1.0
        ),
        "market_price": liquid_df["mid_price"].to_numpy(dtype=float64),
        "spot_price": spot_price,
        "strike_price": liquid_df["strike"].to_numpy(dtype=float64),
        "time_to_expiry": liquid_df["time_to_expiry"].to_numpy(dtype=float64),
        "risk_free_rate": np.full(len(liquid_df), risk_free_rate, dtype=float64),
        "dividend_yield": np.full(len(liquid_df), dividend_yield, dtype=float64),
    }
