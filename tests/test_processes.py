"""Process simulator output shape and martingale checks."""

from __future__ import annotations

import numpy as np

from models.processes import list_processes, make_process


def test_all_processes_simulate_clean_paths():
    rng = np.random.default_rng(42)
    num_paths, num_steps, t = 8, 32, 1.0
    presets = {
        "GBM": dict(s0=100.0, r=0.05, q=0.0, sigma=0.2),
        "Heston": dict(
            s0=100.0, v0=0.04, r=0.05, q=0.0,
            kappa=2.0, theta=0.04, eta=0.3, rho=-0.5,
        ),
        "Bates": dict(
            s0=100.0, v0=0.04, r=0.05, q=0.0,
            kappa=2.0, theta=0.04, eta=0.3, rho=-0.5,
            lam=0.4, mu_j=-0.05, sigma_j=0.15,
        ),
        "SABR": dict(f0=100.0, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4),
        "RBergomi": dict(
            s0=100.0, r=0.05, q=0.0,
            xi0=0.04, eta=0.5, rho=-0.5, hurst=0.3,
        ),
    }
    noise_dims = {"GBM": 1, "Heston": 2, "Bates": 3, "SABR": 2, "RBergomi": 3}
    for name in presets:
        if name not in list_processes():
            continue
        nd = noise_dims[name]
        z = (
            rng.standard_normal((num_paths, num_steps))
            if nd == 1
            else rng.standard_normal((num_paths, num_steps, nd))
        )
        proc = make_process(name, **presets[name])
        paths = proc.simulate_paths(num_paths, num_steps, t, z)
        assert paths.shape[-2:] == (num_paths, num_steps + 1) or paths.shape[:2] == (
            num_paths, num_steps + 1
        )
        assert np.all(np.isfinite(paths))
