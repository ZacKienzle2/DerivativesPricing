"""Default parameter presets surfaced through the dashboard."""

from collections.abc import Mapping
from typing import Any

PROCESS_LAB_PRESETS: Mapping[str, dict[str, Any]] = {
    "GBM": {"s0": 100.0, "r": 0.05, "q": 0.0, "sigma": 0.2},
    "Heston": {
        "s0": 100.0, "v0": 0.04, "r": 0.05, "q": 0.0,
        "kappa": 2.0, "theta": 0.04, "eta": 0.3, "rho": -0.5,
    },
    "Bates": {
        "s0": 100.0, "v0": 0.04, "r": 0.05, "q": 0.0,
        "kappa": 2.0, "theta": 0.04, "eta": 0.3, "rho": -0.5,
        "lam": 0.4, "mu_j": -0.05, "sigma_j": 0.15,
    },
    "RBergomi": {
        "s0": 100.0, "r": 0.05, "q": 0.0,
        "xi0": 0.04, "eta": 1.5, "rho": -0.7, "hurst": 0.1,
    },
}

HESTON_PRESETS: Mapping[str, dict[str, float]] = {
    "Equity index (skew)": {
        "kappa": 2.0, "theta": 0.04, "eta": 0.3, "rho": -0.7, "v0": 0.04,
    },
    "FX (mild smile)": {
        "kappa": 1.5, "theta": 0.02, "eta": 0.4, "rho": -0.1, "v0": 0.025,
    },
    "Crypto (high vol-of-vol)": {
        "kappa": 3.0, "theta": 0.40, "eta": 1.2, "rho": -0.4, "v0": 0.45,
    },
    "Calm market": {
        "kappa": 1.0, "theta": 0.02, "eta": 0.15, "rho": -0.3, "v0": 0.02,
    },
    "Stress (high skew)": {
        "kappa": 4.0, "theta": 0.10, "eta": 0.8, "rho": -0.85, "v0": 0.12,
    },
}

BATES_PRESETS: Mapping[str, dict[str, float]] = {
    "Light jumps": {
        "kappa": 2.0, "theta": 0.04, "eta": 0.3, "rho": -0.5, "v0": 0.04,
        "lam": 0.3, "mu_j": -0.04, "sigma_j": 0.10,
    },
    "Crash-prone": {
        "kappa": 2.5, "theta": 0.05, "eta": 0.4, "rho": -0.6, "v0": 0.05,
        "lam": 0.8, "mu_j": -0.10, "sigma_j": 0.20,
    },
}
