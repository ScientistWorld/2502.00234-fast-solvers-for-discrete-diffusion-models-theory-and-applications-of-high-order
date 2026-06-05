"""Analytic 15-state discrete diffusion model from Ren et al. (2025).

The forward CTMC has rate matrix Q = E / d - I.  Its marginal distribution is

    p_t = ((1 - exp(-t)) / d) * 1 + exp(-t) * p_0.

The reverse-process jump rate from current state x to proposed state y at
reverse time s is Q(x, y) * p_{T-s}(y) / p_{T-s}(x), for x != y.
"""

from __future__ import annotations

import numpy as np


DEFAULT_TARGET = np.array(
    [
        0.018037,
        0.044851,
        0.071665,
        0.098479,
        0.125293,
        0.032348,
        0.059162,
        0.085976,
        0.112790,
        0.139604,
        0.024864,
        0.051678,
        0.078492,
        0.105306,
        0.051455,
    ],
    dtype=np.float64,
)
DEFAULT_TARGET = DEFAULT_TARGET / DEFAULT_TARGET.sum()


def marginal_distribution(t: float, target: np.ndarray = DEFAULT_TARGET) -> np.ndarray:
    """Forward-process marginal p_t for Q = E / d - I."""
    target = np.asarray(target, dtype=np.float64)
    d = target.shape[0]
    decay = np.exp(-float(t))
    return ((1.0 - decay) / d) + decay * target


def reverse_rates(
    states: np.ndarray,
    reverse_time: float,
    horizon: float,
    target: np.ndarray = DEFAULT_TARGET,
) -> np.ndarray:
    """Return rates for all destination states for each current state.

    Output shape is (len(states), d).  The diagonal entries are zero because
    Poisson jump channels only enumerate x -> y, y != x.
    """
    target = np.asarray(target, dtype=np.float64)
    states = np.asarray(states, dtype=np.int64)
    d = target.shape[0]
    forward_t = max(float(horizon) - float(reverse_time), 0.0)
    p_t = marginal_distribution(forward_t, target)
    rates = (p_t[None, :] / p_t[states, None]) / d
    rates[np.arange(states.shape[0]), states] = 0.0
    return rates


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Compute KL(p || q) with a small floor for empirical zero bins."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q = np.clip(q, eps, None)
    q = q / q.sum()
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))

