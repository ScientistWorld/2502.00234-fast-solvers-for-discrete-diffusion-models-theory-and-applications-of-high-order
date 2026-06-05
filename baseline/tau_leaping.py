"""Standalone tau-leaping baseline for finite-state reverse CTMCs."""

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
    """Apply one tau-leap update, rejecting multi-jump proposals."""
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


def tau_leaping(
    initial_states: np.ndarray,
    time_grid: np.ndarray,
    rate_fn: RateFn,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    """First-order Poisson tau-leaping update from the paper's Eq. (3.1)."""
    states = np.asarray(initial_states, dtype=np.int64).copy()
    rejections = []
    for s0, s1 in zip(time_grid[:-1], time_grid[1:]):
        rates = rate_fn(states, float(s0))
        states, reject = _apply_poisson_channels(states, rates, float(s1 - s0), rng)
        rejections.append(reject)
    return states, {"rejection_rate": float(np.mean(rejections) if rejections else 0.0)}
