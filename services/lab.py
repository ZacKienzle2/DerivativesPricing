"""Process Lab simulation helper."""

from __future__ import annotations

from typing import Any

import numpy as np

from models.processes import get_spec, make_process

from ._cache import cached
from ._timing import timed


@cached()
@timed("services.lab.simulate")
def simulate_process_paths(
    process_name: str,
    params: dict[str, Any],
    num_paths: int = 64,
    num_steps: int = 252,
    t: float = 1.0,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Simulates a batch of paths from any registered process.

    Resolves `process_name` against `models.processes.registry`, so any
    new process registered through `register_process` is automatically
    available here without code changes.
    """
    spec = get_spec(process_name)
    rng = np.random.default_rng(seed)
    if spec.noise_dim == 1:
        z = rng.standard_normal((num_paths, num_steps))
    else:
        z = rng.standard_normal((num_paths, num_steps, spec.noise_dim))
    process = make_process(process_name, **params)
    paths = process.simulate_paths(num_paths, num_steps, t, z)
    times = np.linspace(0.0, t, num_steps + 1)
    terminal = paths[:, -1] if paths.ndim == 2 else paths[0, :, -1]
    return {"times": times, "paths": paths, "terminal": terminal}
