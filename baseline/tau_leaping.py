"""First-order Poisson tau-leaping baseline (paper Eq. 3.1, Alg. 3).

Registered through SOLVERS so the evaluators can run it.  Uses the uniform
time grid of the paper with one intensity evaluation per step.
"""

from __future__ import annotations

import numpy as np


def poisson_leap(states: np.ndarray, rates: np.ndarray, dt: float, rng) -> np.ndarray:
    """One tau-leap with independent Poisson channels per (site, destination).

    Sites whose channels fire more than once in total are left unchanged, the
    paper's rejection convention.  Sampling the per-site total from a Poisson
    and, when it equals one, the channel proportionally to its rate has the
    same law as drawing every channel independently.
    """
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


def tau_leaping(problem) -> np.ndarray:
    steps = problem.nfe_budget
    grid = np.linspace(0.0, problem.horizon, steps + 1)
    x = np.array(problem.initial_states, dtype=np.int64)
    for s0, s1 in zip(grid[:-1], grid[1:]):
        x = poisson_leap(x, problem.rates(x, s0), s1 - s0, problem.rng)
    return x


SOLVERS = {"tau_leaping": tau_leaping}
