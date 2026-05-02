"""Shared finite-difference Black-Scholes solver primitives.

The whole backward time loop is JIT-compiled and uses a custom Thomas
tridiagonal solver whose LU decomposition is computed once before the loop
(LHS is time-invariant). Working memory is `O(N)`: a single rolling vector
of node values plus three N-length factor caches.
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import njit

from ..options import AmericanOption, BaseOption


@njit(fastmath=True, cache=True, inline="always")
def _thomas_factor(
    sub: npt.NDArray[np.float64],
    diag: npt.NDArray[np.float64],
    sup: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """LU decomposes a tridiagonal matrix.

    Args:
        sub: Sub-diagonal, length n. `sub[0]` ignored.
        diag: Main diagonal, length n.
        sup: Super-diagonal, length n. `sup[n-1]` ignored.

    Returns:
        Tuple `(c_prime, inv_diag)` factor caches reusable in `_thomas_solve`.
    """
    n = diag.size
    c_prime = np.empty(n)
    inv_diag = np.empty(n)
    inv_diag[0] = 1.0 / diag[0]
    c_prime[0] = sup[0] * inv_diag[0]
    for i in range(1, n):
        denom = diag[i] - sub[i] * c_prime[i - 1]
        inv_diag[i] = 1.0 / denom
        if i < n - 1:
            c_prime[i] = sup[i] * inv_diag[i]
    return c_prime, inv_diag


@njit(fastmath=True, cache=True, inline="always")
def _thomas_solve(
    sub: npt.NDArray[np.float64],
    c_prime: npt.NDArray[np.float64],
    inv_diag: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
) -> None:
    """Solves a pre-factored tridiagonal system in place.

    Args:
        sub: Sub-diagonal of original matrix.
        c_prime: Cached super-diagonal factors from `_thomas_factor`.
        inv_diag: Cached pivot inverses from `_thomas_factor`.
        rhs: Right-hand-side vector (read-only).
        out: Output buffer, same length as `rhs`.
    """
    n = rhs.size
    out[0] = rhs[0] * inv_diag[0]
    for i in range(1, n):
        out[i] = (rhs[i] - sub[i] * out[i - 1]) * inv_diag[i]
    for i in range(n - 2, -1, -1):
        out[i] -= c_prime[i] * out[i + 1]


@njit(fastmath=True, cache=True)
def _solve_explicit_jit(
    vs: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    lower_bc: npt.NDArray[np.float64],
    upper_bc: npt.NDArray[np.float64],
    exercise: npt.NDArray[np.float64],
    is_american: bool,
    is_call: bool,
    k: float,
) -> npt.NDArray[np.float64]:
    """Explicit FTCS backward sweep with rolling buffer."""
    n = vs.size
    n_steps = lower_bc.size - 1

    if is_call:
        vals_next = np.maximum(vs - k, 0.0)
    else:
        vals_next = np.maximum(k - vs, 0.0)
    vals_curr = np.empty(n)

    for j in range(n_steps - 1, -1, -1):
        vals_curr[0] = lower_bc[j]
        vals_curr[n - 1] = upper_bc[j]
        for i in range(1, n - 1):
            vals_curr[i] = (
                a[i] * vals_next[i - 1] + b[i] * vals_next[i] + c[i] * vals_next[i + 1]
            )
        if is_american:
            for i in range(1, n - 1):
                if exercise[i - 1] > vals_curr[i]:
                    vals_curr[i] = exercise[i - 1]
        for i in range(n):
            vals_next[i] = vals_curr[i]
    return vals_next


@njit(fastmath=True, cache=True)
def _solve_implicit_jit(
    vs: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    lower_bc: npt.NDArray[np.float64],
    upper_bc: npt.NDArray[np.float64],
    exercise: npt.NDArray[np.float64],
    is_american: bool,
    is_call: bool,
    k: float,
) -> npt.NDArray[np.float64]:
    """Implicit BTCS backward sweep with cached Thomas factors."""
    n = vs.size
    n_steps = lower_bc.size - 1
    n_int = n - 2

    sub = np.empty(n_int)
    diag = np.empty(n_int)
    sup = np.empty(n_int)
    for i in range(n_int):
        diag[i] = b[i + 1]
    for i in range(1, n_int):
        sub[i] = a[i + 1]
    for i in range(n_int - 1):
        sup[i] = c[i + 1]

    c_prime, inv_diag = _thomas_factor(sub, diag, sup)

    if is_call:
        vals_next = np.maximum(vs - k, 0.0)
    else:
        vals_next = np.maximum(k - vs, 0.0)
    vals_curr = np.empty(n)
    rhs = np.empty(n_int)
    interior = np.empty(n_int)

    for j in range(n_steps - 1, -1, -1):
        for i in range(n_int):
            rhs[i] = vals_next[i + 1]
        rhs[0] -= a[1] * lower_bc[j]
        rhs[n_int - 1] -= c[n - 2] * upper_bc[j]
        _thomas_solve(sub, c_prime, inv_diag, rhs, interior)

        vals_curr[0] = lower_bc[j]
        vals_curr[n - 1] = upper_bc[j]
        for i in range(n_int):
            vals_curr[i + 1] = interior[i]
        if is_american:
            for i in range(1, n - 1):
                if exercise[i - 1] > vals_curr[i]:
                    vals_curr[i] = exercise[i - 1]
        for i in range(n):
            vals_next[i] = vals_curr[i]
    return vals_next


@njit(fastmath=True, cache=True)
def _solve_cn_jit(
    vs: npt.NDArray[np.float64],
    a_l: npt.NDArray[np.float64],
    b_l: npt.NDArray[np.float64],
    c_l: npt.NDArray[np.float64],
    a_r: npt.NDArray[np.float64],
    b_r: npt.NDArray[np.float64],
    c_r: npt.NDArray[np.float64],
    lower_bc: npt.NDArray[np.float64],
    upper_bc: npt.NDArray[np.float64],
    exercise: npt.NDArray[np.float64],
    is_american: bool,
    is_call: bool,
    k: float,
) -> npt.NDArray[np.float64]:
    """Crank-Nicolson backward sweep with cached Thomas factors."""
    n = vs.size
    n_steps = lower_bc.size - 1
    n_int = n - 2

    sub_l = np.empty(n_int)
    diag_l = np.empty(n_int)
    sup_l = np.empty(n_int)
    for i in range(n_int):
        diag_l[i] = b_l[i + 1]
    for i in range(1, n_int):
        sub_l[i] = a_l[i + 1]
    for i in range(n_int - 1):
        sup_l[i] = c_l[i + 1]

    c_prime, inv_diag = _thomas_factor(sub_l, diag_l, sup_l)

    if is_call:
        vals_next = np.maximum(vs - k, 0.0)
    else:
        vals_next = np.maximum(k - vs, 0.0)
    vals_curr = np.empty(n)
    rhs = np.empty(n_int)
    interior = np.empty(n_int)

    for j in range(n_steps - 1, -1, -1):
        for i in range(n_int):
            v_left = vals_next[i] if i > 0 else vals_next[0]
            v_mid = vals_next[i + 1]
            v_right = vals_next[i + 2]
            sub_term = a_r[i + 1] * v_left if i > 0 else 0.0
            sup_term = c_r[i + 1] * v_right if i < n_int - 1 else 0.0
            rhs[i] = sub_term + b_r[i + 1] * v_mid + sup_term
        rhs[0] += a_r[1] * vals_next[0] - a_l[1] * lower_bc[j]
        rhs[n_int - 1] += c_r[n - 2] * vals_next[n - 1] - c_l[n - 2] * upper_bc[j]
        _thomas_solve(sub_l, c_prime, inv_diag, rhs, interior)

        vals_curr[0] = lower_bc[j]
        vals_curr[n - 1] = upper_bc[j]
        for i in range(n_int):
            vals_curr[i + 1] = interior[i]
        if is_american:
            for i in range(1, n - 1):
                if exercise[i - 1] > vals_curr[i]:
                    vals_curr[i] = exercise[i - 1]
        for i in range(n):
            vals_next[i] = vals_curr[i]
    return vals_next


def _grid(opt: BaseOption, n_steps: int, n_points: int) -> Tuple[
    float, npt.NDArray[np.float64], npt.NDArray[np.float64], float
]:
    """Builds the asset-price grid and timestep."""
    s_max = max(
        2.0 * opt.K,
        opt.K * np.exp(opt.r * opt.T + 4.0 * opt.sigma * np.sqrt(opt.T)),
    )
    dt = opt.T / n_steps
    vs = np.linspace(0.0, s_max, n_points + 1)
    vi = np.arange(0, n_points + 1, dtype=np.float64)
    return s_max, vs, vi, dt


def _boundary_values(
    s_max: float,
    k: float,
    r: float,
    dt: float,
    n_steps: int,
    is_call: bool,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Pre-computes boundary value vectors indexed by timestep."""
    tau = dt * (n_steps - np.arange(n_steps + 1, dtype=np.float64))
    if is_call:
        return np.zeros(n_steps + 1), s_max - k * np.exp(-r * tau)
    return k * np.exp(-r * tau), np.zeros(n_steps + 1)


def solve_fd(
    option: BaseOption,
    n_steps: int,
    n_points: int,
    scheme: str,
) -> float:
    """Runs a backward-time FD solve and interpolates the price at S0.

    Args:
        option: Option contract.
        n_steps: Number of timesteps.
        n_points: Number of asset-price intervals.
        scheme: One of `"explicit"`, `"implicit"`, `"crank_nicolson"`.

    Returns:
        Theoretical price at `option.S` from linear interpolation.
    """
    is_call = option.option_type == "call"
    is_american = isinstance(option, AmericanOption)
    k, r, q, sigma = option.K, option.r, option.q, option.sigma

    s_max, vs, vi, dt = _grid(option, n_steps, n_points)
    lower_bc, upper_bc = _boundary_values(s_max, k, r, dt, n_steps, is_call)

    if is_american:
        if is_call:
            exercise = np.maximum(vs[1:-1] - k, 0.0)
        else:
            exercise = np.maximum(k - vs[1:-1], 0.0)
    else:
        exercise = np.empty(0)

    sig2 = sigma * sigma
    vi2 = vi * vi

    if scheme == "explicit":
        a = 0.5 * dt * (sig2 * vi2 - (r - q) * vi)
        b = 1.0 - dt * (sig2 * vi2 + r)
        c = 0.5 * dt * (sig2 * vi2 + (r - q) * vi)
        result = _solve_explicit_jit(
            vs, a, b, c, lower_bc, upper_bc, exercise, is_american, is_call, k
        )

    elif scheme == "implicit":
        a = 0.5 * dt * ((r - q) * vi - sig2 * vi2)
        b = 1.0 + dt * (sig2 * vi2 + r)
        c = 0.5 * dt * (-(r - q) * vi - sig2 * vi2)
        result = _solve_implicit_jit(
            vs, a, b, c, lower_bc, upper_bc, exercise, is_american, is_call, k
        )

    elif scheme == "crank_nicolson":
        a_h = 0.25 * dt * (sig2 * vi2 - (r - q) * vi)
        b_h = -0.5 * dt * (sig2 * vi2 + r)
        c_h = 0.25 * dt * (sig2 * vi2 + (r - q) * vi)
        a_l = -a_h
        b_l = 1.0 - b_h
        c_l = -c_h
        a_r = a_h
        b_r = 1.0 + b_h
        c_r = c_h
        result = _solve_cn_jit(
            vs,
            a_l, b_l, c_l,
            a_r, b_r, c_r,
            lower_bc, upper_bc,
            exercise, is_american, is_call, k,
        )

    else:
        raise ValueError(f"Unknown scheme: {scheme!r}")

    return float(np.interp(option.S, vs, result))
