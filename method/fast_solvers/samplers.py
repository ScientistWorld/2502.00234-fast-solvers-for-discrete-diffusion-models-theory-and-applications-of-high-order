"""Paper high-order samplers for finite-state discrete diffusion reverse CTMCs.

This module implements the computational steps from Algorithms 1 and 2 of
"Fast Solvers for Discrete Diffusion Models": theta-RK-2 and theta-Trapezoidal.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

RateFn = Callable[[np.ndarray, float], np.ndarray]


def _apply_poisson_channels(
    states: np.ndarray,
    rates: np.ndarray,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Apply one tau-leap channel update with rejection of multi-jump states.

    The paper follows common tau-leaping practice and rejects updates with
    multiple jumps along one dimension.  In this one-dimensional categorical
    toy model, that means accepting only samples with exactly one destination
    channel firing during the sub-step.
    """
    if dt <= 0:
        return states.copy(), 0.0
    counts = rng.poisson(np.clip(rates * dt, 0.0, None))
    jump_totals = counts.sum(axis=1)
    accepted = jump_totals == 1
    next_states = states.copy()
    if np.any(accepted):
        next_states[accepted] = np.argmax(counts[accepted], axis=1)
    rejection_rate = float(np.mean(jump_totals > 1))
    return next_states, rejection_rate


def theta_rk2(
    initial_states: np.ndarray,
    time_grid: np.ndarray,
    rate_fn: RateFn,
    rng: np.random.Generator,
    theta: float = 0.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Algorithm 1: theta-Runge-Kutta-2 method."""
    if not (0.0 < theta <= 1.0):
        raise ValueError("theta must be in (0, 1]")
    states = np.asarray(initial_states, dtype=np.int64).copy()
    rejections = []
    positive_fracs = []
    for s0, s1 in zip(time_grid[:-1], time_grid[1:]):
        delta = float(s1 - s0)
        rates0 = rate_fn(states, float(s0))
        intermediate, reject1 = _apply_poisson_channels(states, rates0, theta * delta, rng)
        rho = float((1.0 - theta) * s0 + theta * s1)
        rates_mid = rate_fn(intermediate, rho)
        extrapolated = (1.0 - 1.0 / (2.0 * theta)) * rates0 + (1.0 / (2.0 * theta)) * rates_mid
        extrapolated = np.where(rates0 > 0.0, np.maximum(extrapolated, 0.0), 0.0)
        positive_fracs.append(float(np.mean(extrapolated[rates0 > 0.0] > 0.0)))
        states, reject2 = _apply_poisson_channels(states, extrapolated, delta, rng)
        rejections.extend([reject1, reject2])
    return states, {
        "rejection_rate": float(np.mean(rejections) if rejections else 0.0),
        "positive_intensity_fraction": float(np.mean(positive_fracs) if positive_fracs else 1.0),
    }


def theta_trapezoidal(
    initial_states: np.ndarray,
    time_grid: np.ndarray,
    rate_fn: RateFn,
    rng: np.random.Generator,
    theta: float = 0.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Algorithm 2: theta-Trapezoidal method."""
    if not (0.0 < theta < 1.0):
        raise ValueError("theta must be in (0, 1) for trapezoidal coefficients")
    states = np.asarray(initial_states, dtype=np.int64).copy()
    alpha1 = 1.0 / (2.0 * theta * (1.0 - theta))
    alpha2 = ((1.0 - theta) ** 2 + theta**2) / (2.0 * theta * (1.0 - theta))
    rejections = []
    positive_fracs = []
    for s0, s1 in zip(time_grid[:-1], time_grid[1:]):
        delta = float(s1 - s0)
        rates0 = rate_fn(states, float(s0))
        intermediate, reject1 = _apply_poisson_channels(states, rates0, theta * delta, rng)
        rho = float((1.0 - theta) * s0 + theta * s1)
        rates_mid = rate_fn(intermediate, rho)
        extrapolated = alpha1 * rates_mid - alpha2 * rates0
        active = (rates_mid > 0.0) | (rates0 > 0.0)
        positive_fracs.append(float(np.mean(extrapolated[active] > 0.0)))
        extrapolated = np.maximum(extrapolated, 0.0)
        states, reject2 = _apply_poisson_channels(intermediate, extrapolated, (1.0 - theta) * delta, rng)
        rejections.extend([reject1, reject2])
    return states, {
        "rejection_rate": float(np.mean(rejections) if rejections else 0.0),
        "positive_intensity_fraction": float(np.mean(positive_fracs) if positive_fracs else 1.0),
    }
