# Derivatives Pricer

Stochastic process pricing engine with a Streamlit dashboard.

## Highlights

- 6 stochastic process families (GBM, Heston, Bates, SABR, LocalVol, RBergomi).
- 12+ pricers covering analytic, lattice, finite difference, Monte Carlo, COS Fourier and MLMC.
- 4 calibrators (SABR, Heston via COS, SVI raw, Dupire local vol bootstrapper).
- Greek engines: full analytic vector, lattice, MC pathwise, MC LRM, CRN bumping.
- AOT JIT warmup, batched CRR, vectorised Black-Scholes surface, pluggable cache adapter.
- 9 tab Streamlit dashboard (pricing, surface, greeks, convergence, strategy, IV, process lab, calibration, risk).

## Layout

```
models/        domain types, pricers, processes, calibration
utils/         plotting, IV inversion, greeks, sensitivities, synthetic quotes
services/     application service layer between dashboard and core
ui/           streamlit components and sidebar
app.py        streamlit entry point
controller.py legacy shim, DeprecationWarning on import
tests/        regression and parity smoke (pytest)
```

## Install

```
pip install -e .[dev]
```

`pyproject.toml` pins `python>=3.10` and gates `yfinance`, `pytest`, `ruff`, `black`, `mypy` behind extras.

## Run

```
streamlit run app.py
```

Set `DERIVATIVES_PRICING_NO_WARMUP=1` to skip the eager numba warmup at import; call `models.pricers.prewarm()` manually instead.

## Test

```
pytest -q
```

The conftest exports `DERIVATIVES_PRICING_NO_WARMUP=1` so the suite stays cold-start fast.

## Architecture

- **Domain**: `models.options` ships slot-frozen value objects. `models.processes` holds stochastic process classes that self register through `@autoregister`. `models.pricers` exposes the pricer surface, a `PricerRegistry` with MRO walk, and a `prewarm()` opt-in.
- **Service**: `services` is the only layer that knows about Streamlit, but it does so through a runtime-adaptive `cached()` adapter so the same helpers run in plain Python, notebooks or tests with a transparent `lru_cache` fallback.
- **Greeks**: `utils.greeks` ships analytic, MC pathwise + LRM, lattice and CRN bumping calculators. `GreekCalculatorRegistry` dispatches via `(pricer_class, method)` keys and walks the pricer MRO.
- **Calibration**: `models.calibration` carries SVI, SABR, Heston (COS-driven) and Dupire bootstrapper. All return a `CalibrationResult` dataclass.

## Performance

- JIT kernels (`numba.njit`) on every hot path, with `cache=True` and `parallel=True` where applicable.
- AOT warmup pass eagerly compiles every kernel at import; cached `.nbi` files keep subsequent boots fast.
- Vectorised Black-Scholes surface bypasses the per-cell Python loop (40x speedup at 25x25).
- `cached()` decorator fingerprints ndarrays through `tobytes` and dicts recursively so non-streamlit callers also hit the cache.
- `slots=True` on every frozen dataclass for smaller per-instance footprint.

## Decoupling

- Streamlit imports live only in `services._cache`, `ui` and `app.py`.
- New processes register via `@autoregister`; new Greeks register via `GreekCalculatorRegistry.register`.
- Sidebar option type branching dispatches through `_BUILDERS` mapping.
