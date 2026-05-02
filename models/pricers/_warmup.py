"""Eager JIT compilation of every numba kernel.

Called once on package import. Touches every cached kernel with the smallest
viable input so that first-user-call latency is amortised here. Pickled
caches (`cache=True`) make subsequent imports cheap.

If `DERIVATIVES_PRICING_NO_WARMUP=1` is in the environment the warmup is
skipped; useful in tight CI contexts where pricers will not be invoked.
"""

import os

import numpy as np


def _warmup() -> None:
    """Forces compilation of every JIT kernel."""
    if os.environ.get("DERIVATIVES_PRICING_NO_WARMUP") == "1":
        return

    from ..processes.bates import _bates_qe_jit
    from ..processes.heston import _heston_qe_jit
    from ..processes.local_vol import _bilinear_lookup, _local_vol_jit
    from ..processes.rbergomi import _rbergomi_jit
    from ..processes.sabr import _sabr_euler_jit, hagan_lognormal_vol_jit, sabr_price_jit
    from ..simulation import generate_paths_jit
    try:
        from utils.adjoint_greeks import (
            _asian_pathwise_greeks_jit,
            _vanilla_pathwise_greeks_jit,
        )
    except ImportError:
        _asian_pathwise_greeks_jit = None
        _vanilla_pathwise_greeks_jit = None
    from ._analytic_kernels import (
        bs_full_greeks_jit,
        bs_price_jit,
        discrete_geom_asian_price_jit,
        kemna_vorst_price_jit,
    )
    from .cos_pricer import cos_bates_price_jit, cos_heston_price_jit
    from .crr_batched import crr_batched_jit
    from ._fd_common import _solve_cn_jit, _solve_explicit_jit, _solve_implicit_jit
    from .binomial_tree import _asian_hull_white_jit, _european_american_jit
    from .lattice_pricer import _lattice_pricer_jit
    from .longstaff_schwartz import _backward_induction_jit
    from .monte_carlo import (
        _asian_dual_average_jit,
        _brownian_bridge_jit,
        _build_bb_tree,
        _calculate_barrier_payoffs_jit,
        _generate_correlated_paths_jit,
    )

    z = np.zeros((2, 2), dtype=np.float64)
    paths_2d = generate_paths_jit(100.0, 1.0, 0.05, 0.0, 0.2, 2, z)

    bs_price_jit(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, True)
    kemna_vorst_price_jit(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, True)
    discrete_geom_asian_price_jit(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 2, True)

    _backward_induction_jit(paths_2d, 100.0, 0.05, 0.5, True, 2)

    _calculate_barrier_payoffs_jit(paths_2d, 100.0, 110.0, 0, True)
    _asian_dual_average_jit(paths_2d)

    chol = np.eye(2, dtype=np.float64)
    z3 = np.zeros((2, 2, 2), dtype=np.float64)
    _generate_correlated_paths_jit(
        np.array([100.0, 100.0]),
        np.array([0.2, 0.2]),
        0.05,
        1.0,
        2,
        chol,
        z3,
    )

    _european_american_jit(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 2, True, False)
    _asian_hull_white_jit(100.0, 100.0, 1.0, 0.05, 0.2, 0.0, 2, True, 2)

    vs = np.linspace(0.0, 200.0, 5)
    a = np.full(5, -0.1)
    b = np.full(5, 1.0)
    c = np.full(5, -0.1)
    bcs = np.zeros(3)
    _solve_explicit_jit(
        vs, a, b, c, bcs, bcs, np.empty(0), False, True, 100.0
    )
    _solve_implicit_jit(
        vs, a, b, c, bcs, bcs, np.empty(0), False, True, 100.0, 0
    )
    _solve_cn_jit(
        vs,
        a, b, c,
        a, b, c,
        a, b, c,
        bcs, bcs,
        np.empty(0), False, True, 100.0,
        0, 0,
    )

    _lattice_pricer_jit(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 2, 0, 1, 0, 0, 0.0)

    left, right, bridge = _build_bb_tree(2)
    _brownian_bridge_jit(z, 0.5, left, right, bridge)

    bs_full_greeks_jit(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, True)

    z3_heston = np.zeros((2, 2, 2), dtype=np.float64)
    _heston_qe_jit(
        100.0, 0.04, 0.05, 0.0, 2.0, 0.04, 0.3, -0.5, 1.0, 2, z3_heston
    )

    hagan_lognormal_vol_jit(100.0, 100.0, 1.0, 0.2, 0.5, -0.3, 0.4)
    sabr_price_jit(100.0, 100.0, 1.0, 0.95, 0.2, 0.5, -0.3, 0.4, True)
    _sabr_euler_jit(100.0, 0.2, 0.5, -0.3, 0.4, 1.0, 2, z3_heston)

    cos_heston_price_jit(
        100.0, 100.0, 1.0, 0.05, 0.0,
        2.0, 0.04, 0.3, -0.5, 0.04,
        True, 16, 10.0,
    )
    cos_bates_price_jit(
        100.0, 100.0, 1.0, 0.05, 0.0,
        2.0, 0.04, 0.3, -0.5, 0.04,
        0.1, -0.05, 0.1,
        True, 16, 10.0,
    )

    z_bates = np.zeros((2, 2, 3), dtype=np.float64)
    poisson = np.zeros((2, 2), dtype=np.int64)
    _bates_qe_jit(
        100.0, 0.04, 0.05, 0.0, 2.0, 0.04, 0.3, -0.5,
        0.1, -0.05, 0.1, 1.0, 2, z_bates, poisson,
    )

    grid_s = np.array([50.0, 100.0, 150.0])
    grid_t = np.array([0.0, 0.5, 1.0])
    grid_v = np.full((3, 3), 0.2)
    _bilinear_lookup(100.0, 0.5, grid_s, grid_t, grid_v)
    _local_vol_jit(100.0, 0.05, 0.0, 1.0, 2, grid_s, grid_t, grid_v, z)

    crr_batched_jit(
        100.0, np.array([90.0, 100.0, 110.0]), 1.0, 0.05, 0.0, 0.2, 4, True, False,
    )

    chol_rb = np.eye(2, dtype=np.float64) * 0.5
    z_rb = np.zeros((2, 2, 2), dtype=np.float64)
    _rbergomi_jit(
        100.0, 0.05, 0.0, 0.04, 1.0, -0.5, 0.1, 1.0, 2, chol_rb, z_rb,
    )

    if _vanilla_pathwise_greeks_jit is not None:
        zg = np.zeros(2, dtype=np.float64)
        _vanilla_pathwise_greeks_jit(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, zg, True
        )
    if _asian_pathwise_greeks_jit is not None:
        zga = np.zeros((2, 2), dtype=np.float64)
        _asian_pathwise_greeks_jit(
            100.0, 100.0, 0.05, 0.0, 0.2, 1.0, zga, True
        )
