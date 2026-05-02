"""Process Lab simulation helper."""

from typing import Any, Dict

import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def simulate_process_paths(
    process_name: str,
    params: Dict[str, Any],
    num_paths: int = 64,
    num_steps: int = 252,
    t: float = 1.0,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Simulates a small batch of paths from any registered process.

    Args:
        process_name: One of `"GBM"`, `"Heston"`, `"Bates"`, `"SABR"`,
            `"LocalVol"`, `"RBergomi"`.
        params: Per-process keyword arguments.
        num_paths: Number of paths to simulate.
        num_steps: Discretisation steps.
        t: Total horizon.
        seed: PCG64 seed.

    Returns:
        Dict with `times`, `paths`, `terminal`.
    """
    from models.processes import (
        BatesProcess,
        GBMProcess,
        HestonProcess,
        LocalVolProcess,
        RBergomiProcess,
        SABRProcess,
    )

    rng = np.random.default_rng(seed)
    factories = {
        "GBM": (GBMProcess, 1),
        "Heston": (HestonProcess, 2),
        "Bates": (BatesProcess, 3),
        "SABR": (SABRProcess, 2),
        "LocalVol": (LocalVolProcess, 1),
        "RBergomi": (RBergomiProcess, 2),
    }
    if process_name not in factories:
        raise ValueError(f"Unknown process: {process_name!r}")
    cls, noise_dim = factories[process_name]
    process = cls(**params)
    if noise_dim == 1:
        z = rng.standard_normal((num_paths, num_steps))
    else:
        z = rng.standard_normal((num_paths, num_steps, noise_dim))
    paths = process.simulate_paths(num_paths, num_steps, t, z)
    times = np.linspace(0.0, t, num_steps + 1)
    terminal = paths[:, -1] if paths.ndim == 2 else paths[0, :, -1]
    return {"times": times, "paths": paths, "terminal": terminal}
