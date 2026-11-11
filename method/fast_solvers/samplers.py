"""Paper high-order samplers: theta-RK-2 (Alg. 1) and theta-Trapezoidal (Alg. 2).

Both are two-stage Poisson tau-leaping schemes for the reverse CTMC on
X = [S]^d with single-site jumps.  Following the paper (and the authors'
released implementation) the two stages are combined per jump increment nu:
the intensity of jump nu at the intermediate state is paired with the
intensity of the same jump nu at the step's starting state, and jumps that
would leave [0, S) carry no intensity.  Updates with more than one jump along a
site are rejected.  The time grid is the paper's arithmetic sequence and each
step costs two intensity evaluations, so a budget of B evaluations gives B / 2
steps.
"""

from __future__ import annotations

import numpy as np


def poisson_leap(states: np.ndarray, rates: np.ndarray, dt: float, rng) -> np.ndarray:
    """Independent Poisson channels per (site, destination), multi-jump rejection."""
    S = rates.shape[-1]
    total = rates.sum(axis=-1)
    counts = rng.poisson(np.maximum(total * dt, 0.0))
    jump = counts == 1
    out = states.copy()
    if np.any(jump):
        r = rates[jump]
        cdf = np.cumsum(r, axis=1)
        u = rng.random(r.shape[0]) * cdf[:, -1]
        out[jump] = np.minimum((cdf <= u[:, None]).sum(axis=1), S - 1)
    return out


def align_by_jump(rates: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Re-index intensities evaluated at `source` by jump increment from `target`.

    out[..., y] = rates[..., source + (y - target)], the intensity of the same
    jump nu = y - target taken from `source`; zero where that jump leaves [0, S).
    """
    S = rates.shape[-1]
    idx = (source - target)[..., None] + np.arange(S)
    valid = (idx >= 0) & (idx < S)
    gathered = np.take_along_axis(rates, np.clip(idx, 0, S - 1), axis=-1)
    return np.where(valid, gathered, 0.0)


def uniform_grid(horizon: float, steps: int) -> np.ndarray:
    return np.linspace(0.0, horizon, steps + 1)


def theta_rk2(problem, theta: float = 0.5, grid=None) -> np.ndarray:
    """Algorithm 1: theta-Runge-Kutta-2."""
    if not 0.0 < theta <= 1.0:
        raise ValueError("theta must lie in (0, 1]")
    steps = problem.nfe_budget // 2
    grid = uniform_grid(problem.horizon, steps) if grid is None else grid
    rng = problem.rng
    x = np.array(problem.initial_states, dtype=np.int64)
    w0, w1 = 1.0 - 1.0 / (2.0 * theta), 1.0 / (2.0 * theta)
    for s0, s1 in zip(grid[:-1], grid[1:]):
        delta = float(s1 - s0)
        mu0 = problem.rates(x, s0)
        x_mid = poisson_leap(x, mu0, theta * delta, rng)
        mu_mid = problem.rates(x_mid, s0 + theta * delta)
        combined = w0 * mu0 + w1 * align_by_jump(mu_mid, x_mid, x)
        combined = np.where(mu0 > 0.0, np.maximum(combined, 0.0), 0.0)
        x = poisson_leap(x, combined, delta, rng)
    return x


def theta_trapezoidal(problem, theta: float = 0.5, grid=None) -> np.ndarray:
    """Algorithm 2: theta-Trapezoidal."""
    if not 0.0 < theta < 1.0:
        raise ValueError("theta must lie in (0, 1)")
    steps = problem.nfe_budget // 2
    grid = uniform_grid(problem.horizon, steps) if grid is None else grid
    rng = problem.rng
    x = np.array(problem.initial_states, dtype=np.int64)
    alpha1 = 1.0 / (2.0 * theta * (1.0 - theta))
    alpha2 = ((1.0 - theta) ** 2 + theta**2) / (2.0 * theta * (1.0 - theta))
    for s0, s1 in zip(grid[:-1], grid[1:]):
        delta = float(s1 - s0)
        mu0 = problem.rates(x, s0)
        x_mid = poisson_leap(x, mu0, theta * delta, rng)
        mu_mid = problem.rates(x_mid, s0 + theta * delta)
        extrapolated = alpha1 * mu_mid - alpha2 * align_by_jump(mu0, x, x_mid)
        x = poisson_leap(x_mid, np.maximum(extrapolated, 0.0), (1.0 - theta) * delta, rng)
    return x


SOLVERS = {"theta_rk2": theta_rk2, "theta_trapezoidal": theta_trapezoidal}
