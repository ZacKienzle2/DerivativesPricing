"""Shared finite-difference Black-Scholes solver primitives.

Backward time loops are JIT-compiled with cached Thomas tridiagonal LU
factors and PSOR for American LCP solves. Crank-Nicolson supports
Rannacher smoothing (first two steps run fully implicit) to suppress
payoff-kink oscillation and restore second-order accuracy.

Working memory is `O(N)`: a rolling vector of node values plus a few
N-length factor caches.
"""

from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import njit

from ..options import AmericanOption, BaseOption

_AM_PROJECT = 0
_AM_PSOR = 1

_PSOR_OMEGA = 1.2
_PSOR_TOL = 1e-9
_PSOR_MAX_ITER = 200


@njit(fastmath=True, cache=True, inline="always", boundscheck=False)
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


@njit(fastmath=True, cache=True, inline="always", boundscheck=False)
def _thomas_solve(
    sub: npt.NDArray[np.float64],
    c_prime: npt.NDArray[np.float64],
    inv_diag: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
) -> None:
    """Solves a pre-factored tridiagonal system in place."""
    n = rhs.size
    out[0] = rhs[0] * inv_diag[0]
    for i in range(1, n):
        out[i] = (rhs[i] - sub[i] * out[i - 1]) * inv_diag[i]
    for i in range(n - 2, -1, -1):
        out[i] -= c_prime[i] * out[i + 1]


@njit(fastmath=True, cache=True, boundscheck=False)
def _psor_solve(
    sub: npt.NDArray[np.float64],
    diag: npt.NDArray[np.float64],
    sup: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    omega: float,
    tol: float,
    max_iter: int,
) -> None:
    """Projected SOR solve for the LCP `Av >= rhs, v >= phi`.

    Iterates Gauss-Seidel sweeps with relaxation `omega` and projects each
    component onto `[phi[i], inf)`. Uses `v` as both warm-start input and
    output buffer.
    """
    n = v.size
    for _ in range(max_iter):
        max_change = 0.0
        for i in range(n):
            sl = sub[i] * v[i - 1] if i > 0 else 0.0
            su = sup[i] * v[i + 1] if i < n - 1 else 0.0
            v_unproj = (rhs[i] - sl - su) / diag[i]
            v_new = v[i] + omega * (v_unproj - v[i])
            if v_new < phi[i]:
                v_new = phi[i]
            delta = v_new - v[i]
            if delta < 0.0:
                delta = -delta
            if delta > max_change:
                max_change = delta
            v[i] = v_new
        if max_change < tol:
            return


@njit(fastmath=True, cache=True, boundscheck=False)
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


@njit(fastmath=True, cache=True, boundscheck=False)
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
    american_method: int,
) -> npt.NDArray[np.float64]:
    """Implicit BTCS backward sweep with cached Thomas factors and PSOR."""
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

    use_psor = is_american and (american_method == _AM_PSOR)

    if use_psor:
        c_prime = np.empty(n_int)
        inv_diag = np.empty(n_int)
    else:
        c_prime, inv_diag = _thomas_factor(sub, diag, sup)

    if is_call:
        vals_next = np.maximum(vs - k, 0.0)
    else:
        vals_next = np.maximum(k - vs, 0.0)
    vals_curr = np.empty(n)
    rhs = np.empty(n_int)
    interior = np.empty(n_int)

    if use_psor:
        for i in range(n_int):
            interior[i] = vals_next[i + 1]

    for j in range(n_steps - 1, -1, -1):
        for i in range(n_int):
            rhs[i] = vals_next[i + 1]
        rhs[0] -= a[1] * lower_bc[j]
        rhs[n_int - 1] -= c[n - 2] * upper_bc[j]

        if use_psor:
            _psor_solve(
                sub, diag, sup, rhs, exercise, interior,
                _PSOR_OMEGA, _PSOR_TOL, _PSOR_MAX_ITER,
            )
        else:
            _thomas_solve(sub, c_prime, inv_diag, rhs, interior)

        vals_curr[0] = lower_bc[j]
        vals_curr[n - 1] = upper_bc[j]
        for i in range(n_int):
            vals_curr[i + 1] = interior[i]
        if is_american and not use_psor:
            for i in range(1, n - 1):
                if exercise[i - 1] > vals_curr[i]:
                    vals_curr[i] = exercise[i - 1]
        for i in range(n):
            vals_next[i] = vals_curr[i]
    return vals_next


@njit(fastmath=True, cache=True, boundscheck=False)
def _solve_cn_jit(
    vs: npt.NDArray[np.float64],
    a_l: npt.NDArray[np.float64],
    b_l: npt.NDArray[np.float64],
    c_l: npt.NDArray[np.float64],
    a_r: npt.NDArray[np.float64],
    b_r: npt.NDArray[np.float64],
    c_r: npt.NDArray[np.float64],
    a_imp: npt.NDArray[np.float64],
    b_imp: npt.NDArray[np.float64],
    c_imp: npt.NDArray[np.float64],
    lower_bc: npt.NDArray[np.float64],
    upper_bc: npt.NDArray[np.float64],
    exercise: npt.NDArray[np.float64],
    is_american: bool,
    is_call: bool,
    k: float,
    american_method: int,
    rannacher_steps: int,
) -> npt.NDArray[np.float64]:
    """CN backward sweep with optional Rannacher smoothing and PSOR.

    Args:
        a_l, b_l, c_l: CN LHS triplet.
        a_r, b_r, c_r: CN RHS triplet.
        a_imp, b_imp, c_imp: Fully-implicit triplet for Rannacher steps.
        lower_bc, upper_bc: Per-timestep boundary value vectors.
        exercise: Intrinsic-value vector at interior nodes (empty if European).
        is_american, is_call, k: Option parameters.
        american_method: 0 for max-projection, 1 for PSOR.
        rannacher_steps: Number of leading implicit steps (typical: 2).
    """
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

    sub_i = np.empty(n_int)
    diag_i = np.empty(n_int)
    sup_i = np.empty(n_int)
    for i in range(n_int):
        diag_i[i] = b_imp[i + 1]
    for i in range(1, n_int):
        sub_i[i] = a_imp[i + 1]
    for i in range(n_int - 1):
        sup_i[i] = c_imp[i + 1]

    use_psor = is_american and (american_method == _AM_PSOR)

    if use_psor:
        cp_l = np.empty(n_int)
        id_l = np.empty(n_int)
        cp_i = np.empty(n_int)
        id_i = np.empty(n_int)
    else:
        cp_l, id_l = _thomas_factor(sub_l, diag_l, sup_l)
        cp_i, id_i = _thomas_factor(sub_i, diag_i, sup_i)

    if is_call:
        vals_next = np.maximum(vs - k, 0.0)
    else:
        vals_next = np.maximum(k - vs, 0.0)
    vals_curr = np.empty(n)
    rhs = np.empty(n_int)
    interior = np.empty(n_int)

    if use_psor:
        for i in range(n_int):
            interior[i] = vals_next[i + 1]

    for j in range(n_steps - 1, -1, -1):
        steps_remaining = n_steps - j
        use_implicit = steps_remaining <= rannacher_steps

        if use_implicit:
            for i in range(n_int):
                rhs[i] = vals_next[i + 1]
            rhs[0] -= a_imp[1] * lower_bc[j]
            rhs[n_int - 1] -= c_imp[n - 2] * upper_bc[j]
            if use_psor:
                _psor_solve(
                    sub_i, diag_i, sup_i, rhs, exercise, interior,
                    _PSOR_OMEGA, _PSOR_TOL, _PSOR_MAX_ITER,
                )
            else:
                _thomas_solve(sub_i, cp_i, id_i, rhs, interior)
        else:
            for i in range(n_int):
                v_mid = vals_next[i + 1]
                sub_term = a_r[i + 1] * vals_next[i] if i > 0 else 0.0
                sup_term = (
                    c_r[i + 1] * vals_next[i + 2] if i < n_int - 1 else 0.0
                )
                rhs[i] = sub_term + b_r[i + 1] * v_mid + sup_term
            rhs[0] += a_r[1] * vals_next[0] - a_l[1] * lower_bc[j]
            rhs[n_int - 1] += (
                c_r[n - 2] * vals_next[n - 1] - c_l[n - 2] * upper_bc[j]
            )
            if use_psor:
                _psor_solve(
                    sub_l, diag_l, sup_l, rhs, exercise, interior,
                    _PSOR_OMEGA, _PSOR_TOL, _PSOR_MAX_ITER,
                )
            else:
                _thomas_solve(sub_l, cp_l, id_l, rhs, interior)

        vals_curr[0] = lower_bc[j]
        vals_curr[n - 1] = upper_bc[j]
        for i in range(n_int):
            vals_curr[i + 1] = interior[i]
        if is_american and not use_psor:
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
    american_method: str = "project",
    rannacher_steps: int = 2,
) -> float:
    """Runs a backward-time FD solve and interpolates the price at S0.

    Args:
        option: Option contract.
        n_steps: Number of timesteps.
        n_points: Number of asset-price intervals.
        scheme: One of `"explicit"`, `"implicit"`, `"crank_nicolson"`.
        american_method: `"project"` (operator-splitting max-projection) or
            `"psor"` (LCP-correct projected SOR).
        rannacher_steps: Leading implicit steps inside CN to damp payoff-kink
            oscillation. Ignored for non-CN schemes.

    Returns:
        Theoretical price at `option.S` from linear interpolation.
    """
    is_call = option.option_type == "call"
    is_american = isinstance(option, AmericanOption)
    k, r, q, sigma = option.K, option.r, option.q, option.sigma
    method_code = _AM_PSOR if american_method == "psor" else _AM_PROJECT

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
            vs, a, b, c, lower_bc, upper_bc,
            exercise, is_american, is_call, k, method_code,
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
        a_imp = 0.5 * dt * ((r - q) * vi - sig2 * vi2)
        b_imp = 1.0 + dt * (sig2 * vi2 + r)
        c_imp = 0.5 * dt * (-(r - q) * vi - sig2 * vi2)
        result = _solve_cn_jit(
            vs,
            a_l, b_l, c_l,
            a_r, b_r, c_r,
            a_imp, b_imp, c_imp,
            lower_bc, upper_bc,
            exercise, is_american, is_call, k,
            method_code, max(0, rannacher_steps),
        )

    else:
        raise ValueError(f"Unknown scheme: {scheme!r}")

    return float(np.interp(option.S, vs, result))
