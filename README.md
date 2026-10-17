# Driftline
## High-Performance GBM Portfolio Simulator

Driftline is a high-performance Monte Carlo simulation engine for multi-asset portfolios under correlated Geometric Brownian Motion. It generates over one million price paths using fully vectorized matrix operations, with optional periodic rebalancing.

---

## Overview

This simulator models the joint evolution of an arbitrary number of assets under GBM, including cross-asset correlations via Cholesky decomposition. Portfolio values are computed at each timestep and passed to a vectorized risk engine that computes Value at Risk, Conditional VaR, and barrier breach probability. An optional rebalancing engine periodically resets holdings to target weights without modifying the path generation layer.

The design separates concerns cleanly into three independent stages:

```
_worker()                        → price paths  (n_paths, n_assets, n_steps+1)
compute_portfolio_values()       → portfolio PV (n_paths, n_steps+1)
compute_risk()                   → RiskMetrics
```

---

## Mathematical Foundations

### Geometric Brownian Motion

Driftline models asset price dynamics with **geometric Brownian motion (GBM)**, a continuous-time stochastic process widely used in mathematical finance. Each asset $i$ evolves according to the stochastic differential equation:

$$
dS_i(t) = \mu_i S_i(t)\, dt + \sigma_i S_i(t)\, dW_i(t)
$$

where $S_i(t)$ is the asset price, $\mu_i$ is the annualized drift, $\sigma_i$ is the annualized volatility, and $W_i(t)$ is a standard Wiener process. The exact discrete solution over a timestep $\Delta t = T / N$ is:

$$
S_i(t + \Delta t) = S_i(t) \cdot \exp\!\left[
\left(\mu_i - \tfrac{1}{2}\sigma_i^2\right)\Delta t
+ \sigma_i \sqrt{\Delta t}\, Z_i
\right]
$$

where $Z_i \sim \mathcal{N}(0, 1)$. The correction term $-\frac{1}{2}\sigma_i^2$ follows from Itô's lemma and ensures that the expected asset price satisfies:

$$
\mathbb{E}[S_i(t)] = S_{0,i} e^{\mu_i t}
$$

Without this correction, the simulated asset prices would have an incorrect expected value.

### Correlated Shocks via Cholesky Decomposition

Driftline models cross-asset dependence through a symmetric positive-definite correlation matrix $\rho \in \mathbb{R}^{n \times n}$ with unit diagonal. Its Cholesky factorization yields a lower triangular matrix $L$ such that:

$$
\rho = LL^\top
$$

Independent standard normal samples $Z \sim \mathcal{N}(0, I)$ produce correlated samples through the transformation:

$$
Z_{\text{corr}} = ZL^\top
$$

The resulting covariance matrix is:

$$
\operatorname{Cov}(Z_{\text{corr}}) = LL^\top = \rho
$$

In the implementation, $Z$ has shape:

$$
(N_{\text{paths}}, N_{\text{steps}}, N_{\text{assets}})
$$

The asset axis is last, so the correlation transform can use a batched matrix multiplication without an explicit copy of the full random tensor.

### Portfolio Value

Let $w_i$ denote the fractional weight of asset $i$, with $\sum_i w_i = 1$, and let $V_0$ denote the initial portfolio value. The initial share count for asset $i$ is:

$$
\text{shares}_i = \frac{w_i V_0}{S_{0,i}}
$$

For a buy-and-hold portfolio, the value at time $t$ is:

$$
V(t) = \sum_{i=1}^{n} \text{shares}_i S_i(t)
= V_0 \sum_{i=1}^{n} \frac{w_i S_i(t)}{S_{0,i}}
$$

Its analytical expected value provides a reference for the simulated mean path:

$$
\mathbb{E}[V(t)] = V_0 \sum_{i=1}^{n} w_i e^{\mu_i t}
$$

This expectation applies to the buy-and-hold portfolio, not to a portfolio that undergoes periodic rebalancing.

### Portfolio Rebalancing

At each rebalancing date $t_r$, Driftline resets the portfolio to its target weights. The updated share holdings are:

$$
\text{shares}_i(t_r^+) =
\frac{w_i V(t_r)}{S_i(t_r)}
$$

This operation preserves portfolio value because no external cash enters or leaves the portfolio. It also restores the target allocation after asset price movements cause the weights to drift. The `compute_portfolio_values()` function processes rebalancing events in a loop and vectorizes each event across all simulated paths. The `_worker()` function remains unchanged.

### Value at Risk and Conditional Value at Risk

Let $L = V_0 - V_T$ denote terminal portfolio loss, where positive values represent losses. For confidence level $\alpha$, Value at Risk (VaR) is:

$$
\operatorname{VaR}_{\alpha} = F_L^{-1}(\alpha)
$$

Conditional Value at Risk (CVaR), also known as Expected Shortfall, is:

$$
\operatorname{CVaR}_{\alpha}
= \mathbb{E}\!\left[
L \mid L \geq \operatorname{VaR}_{\alpha}
\right]
$$

VaR measures the loss threshold at confidence level $\alpha$, while CVaR measures the average loss in the tail beyond that threshold. Driftline estimates both metrics from simulated terminal losses with `np.percentile` and a Boolean mask, without explicit Python loops.

---

## Architecture

### Data Layout

```
asset_paths      : float32  (n_paths, n_assets, n_steps+1)  C-order, row-major
portfolio_values : float32  (n_paths, n_steps+1)
```

Assets occupy axis 1. The time axis is last. `np.cumprod(axis=2)` traverses each row sequentially in memory, which is cache-optimal under C-order layout.

`float32` halves storage requirements relative to `float64` and can reduce memory traffic. For a 3-asset simulation at 1M paths × 252 steps, `asset_paths` occupies approximately 3.04 GB in `float32` versus 6.07 GB in `float64`. Set `params.dtype = np.float64` when higher numerical precision is required.

### Concurrency Model

`ThreadPool` (OS threads, not subprocesses) is used. NumPy releases the Python GIL during C-extension calls, so the compute-heavy operations (`rng.standard_normal`, `np.exp`, `Z @ L.T`, `np.cumprod`) all execute in parallel across the threads. Each thread operates on a contiguous slice of paths with its own RNG state. The results are then combined with a single `np.concatenate` call.

Path slices are distributed evenly across worker threads in the ThreadPool, with any remainder assigned to the first workers.

### Rebalancing Architecture

The implementation places rebalancing entirely within `compute_portfolio_values()` rather than `_worker()`. Asset prices evolve according to GBM independent of portfolio-level operations. It operates on prices and does not generate price paths.

The segment-based approach:

```
boundaries = [0, rebal_steps, 2*rebal_steps, ..., n_steps]

for each segment [a, b]:
    seg_pv = (shares * prices[:, :, a:b+1]).sum(axis=1)  ← vectorized over all paths
    update shares at b                                    ← vectorized over all paths
```

The loop iterates over rebalancing events (4 for quarterly), not over paths (1,000,000) or steps (252).

---

## Installation

```bash
pip install numpy matplotlib
```

Python 3.10+ is required for the `dict[str, ...]` type annotation syntax. No other dependencies.

---

## Usage

### Running the Default Benchmark

```python
from gbm_simulator import benchmark, REBAL_FREQ

# Default: 1M paths, 252 steps, quarterly rebalancing, 3 runs
benchmark()

# Disable rebalancing
benchmark(rebal_steps=None)

# Annual rebalancing, 5 runs, no plot
benchmark(rebal_steps=REBAL_FREQ["annual"], runs=5, plot=False)

# Custom interval: every 20 steps
benchmark(rebal_steps=20)
```

### `benchmark()` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_paths` | `int` | `1_000_000` | Total Monte Carlo paths |
| `n_steps` | `int` | `252` | Discrete timesteps (252 = daily, one trading year) |
| `T` | `float` | `1.0` | Simulation horizon in years |
| `runs` | `int` | `3` | Timed repetitions; reports best / worst / mean / stddev |
| `rebal_steps` | `Optional[int]` | `REBAL_FREQ["quarterly"]` | Rebalancing interval in steps; `None` disables |
| `plot` | `bool` | `True` | Save `simulation_results.png` to the working directory |

### Using the Simulator Directly

```python
import numpy as np
from gbm_simulator import GBMParams, GBMSimulator, compute_risk, REBAL_FREQ

params = GBMParams(
    S0      = np.array([100.0, 95.0, 120.0]),
    mu      = np.array([0.09,  0.03,  0.05]),
    sigma   = np.array([0.22,  0.06,  0.18]),
    weights = np.array([0.50,  0.30,  0.20]),
    rho     = np.array([
        [1.00, 0.25, 0.10],
        [0.25, 1.00, 0.05],
        [0.10, 0.05, 1.00],
    ]),
    T           = 1.0,
    n_steps     = 252,
    n_paths     = 500_000,
    rebal_steps = REBAL_FREQ["monthly"],
    names       = ["Equity", "Bond", "Commodity"],
)

result = GBMSimulator(params).run()

print(result.asset_paths.shape)       # (500_000, 3, 253)
print(result.portfolio_values.shape)  # (500_000, 253)

risk = compute_risk(result.portfolio_values, params)
print(f"VaR 95%:  ${risk.var_95:.2f}")
print(f"CVaR 99%: ${risk.cvar_99:.2f}")
print(f"Barrier breach: {risk.barrier_breach * 100:.3f}%")
```

---

## `GBMParams` Reference

| Field | Type | Description |
|---|---|---|
| `S0` | `array (n,)` | Initial asset prices |
| `mu` | `array (n,)` | Annualized drift rates |
| `sigma` | `array (n,)` | Annualized volatilities |
| `weights` | `array (n,)` | Fractional allocation weights; must sum to 1.0 |
| `rho` | `array (n, n)` | Correlation matrix; symmetric, PSD, unit diagonal |
| `T` | `float` | Simulation horizon in years |
| `n_steps` | `int` | Number of discrete time steps |
| `n_paths` | `int` | Total Monte Carlo paths |
| `rebal_steps` | `Optional[int]` | Rebalancing interval in steps; `None` = no rebalancing |
| `names` | `list[str]` | Asset labels for output; auto-generated if omitted |
| `dtype` | `type` | Storage dtype for path arrays; default `np.float32` |
| `L` | `array (n, n)` | Cholesky factor; computed automatically, read-only |

### Rebalancing Frequency Constants

```python
REBAL_FREQ = {
    "daily":      1,    # every step
    "weekly":     5,
    "monthly":   21,
    "quarterly": 63,
    "semiannual": 126,
    "annual":    252,   # once per year (assumes n_steps=252)
    "never":    None,   # equivalent to rebal_steps=None
}
```

All values assume `n_steps=252`. For non-standard step counts, compute the equivalent interval manually and pass an integer directly.

---

## Output

### Console Report

```
════════════════════════════════════════════════════════════════════════
  Multi-Asset GBM Portfolio Simulator
════════════════════════════════════════════════════════════════════════
  Assets       :            3  (Equity, Bond, Commodity)
  Paths        :    1,000,000
  Rebalancing  :    quarterly
  ...

  Run  1  |  Time:   1840.3 ms  |  Mem:   312.4 MB  |  Throughput:   543,421 paths/s

  Portfolio Risk Analytics  (rebalancing: quarterly)
  ────────────────────────────────────────────────────
    VaR  95%  : $     43.26  (13.73% of portfolio)
    VaR  99%  : $     64.51  (20.48% of portfolio)
    CVaR 95%  : $     56.34  (17.89% of portfolio)
    CVaR 99%  : $     74.38  (23.61% of portfolio)
    Barrier    :      0.51%  of paths breached 75% barrier
```

### Visualization Plot (`simulation_results.png`)

The figure is automatically generated during `benchmark()` using Matplotlib (Agg backend) and saved as:

<p align="center">
  <img src="./assets/simulation_results.png" alt="GBM Portfolio Simulation" width="900"/>
</p>

- **Panel 1** — 100 randomly sampled portfolio trajectories with analytical mean and risk barrier  
- **Panel 2** — Normalized asset price dynamics (S / S₀)  
- **Panel 3** — Terminal portfolio distribution with VaR and CVaR

---

## Performance Notes

Observed throughput on an 8-core machine (3-asset portfolio, 252 steps, float32):

| Paths | Rebalancing | Approx. time |
|---|---|---|
| 100,000 | none | ~80 ms |
| 100,000 | quarterly | ~90 ms |
| 1,000,000 | none | ~800 ms |
| 1,000,000 | quarterly | ~890 ms |

The rebalancing overhead (~11%) is dominated by the float32 → float64 upcast in `compute_portfolio_values`, not by the segment iteration itself. The segment loop runs 4 times for quarterly rebalancing regardless of path count.

Under heavy workloads, the simulation's performance is limited by memory bandwidth rather than CPU processing speed. If peak RAM usage is a concern, the large price-path matrices can be processed in sequential batches to reduce memory consumption.

---

## Author

**Robert Tran**  
Applied Mathematics & Finance \
San Diego State University

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.