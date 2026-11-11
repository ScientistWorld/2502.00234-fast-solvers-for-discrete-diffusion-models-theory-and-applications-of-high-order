"""Hidden test instances of the multi-site analytic discrete diffusion problem.

State space X = [S]^d (d sites, S states per site).  Each site is noised
independently by the uniform-rate CTMC Q = E / S - I of the paper's toy model,
run for horizon T = 12.  The data distribution p0 is a mixture of K product
distributions ("modes") with Dirichlet(1) mixture weights; each mode's
per-site factor is drawn from a sparse Dirichlet, so the modes are separated,
unequally weighted clusters of sequences.  Forward marginals stay mixtures of
products, p_t = sum_k w_k prod_i (c_t + e_t pi_{k,i}), with e_t = exp(-t) and
c_t = (1 - e_t) / S, so the exact reverse-process jump intensities are
available in closed form.

This module holds the generator, the exact intensity oracle, the exact
low-order marginals and likelihood profile used for scoring, and an exact
sampler used only on the evaluation side.  It is evaluation code: solvers
never receive these objects, only the intensities that the harness returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUM_STATES = 15
NUM_SITES = 48
HORIZON = 12.0
NUM_COMPONENTS = 8
FACTOR_CONCENTRATION = 0.1
MIXTURE_CONCENTRATION = 1.0
PROBABILITY_FLOOR = 1e-12
WEIGHT_FLOOR = 1e-3


@dataclass(frozen=True)
class Instance:
    weights: np.ndarray  # (K,)
    factors: np.ndarray  # (K, d, S)
    num_states: int
    num_sites: int
    horizon: float


def make_instance(
    seed: int,
    num_sites: int = NUM_SITES,
    num_states: int = NUM_STATES,
    num_components: int = NUM_COMPONENTS,
    horizon: float = HORIZON,
    factor_concentration: float = FACTOR_CONCENTRATION,
) -> Instance:
    rng = np.random.default_rng(seed)
    K, d, S = num_components, num_sites, num_states
    weights = rng.dirichlet(np.full(K, MIXTURE_CONCENTRATION)) + WEIGHT_FLOOR
    weights = weights / weights.sum()
    factors = rng.dirichlet(np.full(S, factor_concentration), size=(K, d)) + PROBABILITY_FLOOR
    factors = factors / factors.sum(axis=-1, keepdims=True)
    return Instance(weights, factors, S, d, float(horizon))


class Oracle:
    """Exact reverse-CTMC jump intensities for one instance."""

    def __init__(self, inst: Instance):
        self.inst = inst
        self._log_w = np.log(inst.weights)
        self._factors_t = np.ascontiguousarray(np.moveaxis(inst.factors, 1, 0))  # (d, K, S)

    def rates(self, states: np.ndarray, s) -> np.ndarray:
        """Intensities of single-site jumps x -> x^{i->y} at reverse time s.

        states: (m, d) integer array; s: scalar or (m,) reverse times in [0, T].
        Returns (m, d, S) with rates[j, i, y] = p_t(x^{i->y}) / p_t(x) / S for
        y != x_i (forward time t = T - s) and 0 for y == x_i.
        """
        inst = self.inst
        S, d = inst.num_states, inst.num_sites
        x = np.asarray(states, dtype=np.int64)
        m = x.shape[0]
        if m == 0:
            return np.zeros((0, d, S))
        t = inst.horizon - np.asarray(s, dtype=np.float64)
        decay = np.exp(-t)
        floor = (1.0 - decay) / S
        per_row = decay.ndim > 0
        # f_{k,i}(x_i) for every row: (m, K, d)
        pix = np.moveaxis(inst.factors[:, np.arange(d)[None, :], x], 0, 1)
        if per_row:
            fx = floor[:, None, None] + decay[:, None, None] * pix
        else:
            fx = floor + decay * pix
        logits = self._log_w + np.log(fx).sum(axis=-1)  # (m, K)
        logits -= logits.max(axis=1, keepdims=True)
        post = np.exp(logits)
        post /= post.sum(axis=1, keepdims=True)
        g = post[:, :, None] / fx  # (m, K, d)
        total = g.sum(axis=1)  # (m, d)
        mixed = np.matmul(np.moveaxis(g, 2, 0), self._factors_t)  # (d, m, S)
        ratio = np.ascontiguousarray(np.moveaxis(mixed, 0, 1))  # (m, d, S)
        if per_row:
            ratio *= decay[:, None, None]
            ratio += (floor[:, None] * total)[:, :, None]
        else:
            ratio *= decay
            ratio += (floor * total)[:, :, None]
        ratio /= S
        np.put_along_axis(ratio, x[:, :, None], 0.0, axis=-1)
        return ratio


def site_pairs(num_sites: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(num_sites) for j in range(i + 1, num_sites)]


def exact_marginals(inst: Instance) -> tuple[np.ndarray, np.ndarray]:
    """Exact site marginals (d, S) and site-pair marginals (P, S, S) of p0."""
    w, f = inst.weights, inst.factors
    site = np.einsum("k,kia->ia", w, f)
    pairs = site_pairs(inst.num_sites)
    ii = np.array([p[0] for p in pairs], dtype=np.int64)
    jj = np.array([p[1] for p in pairs], dtype=np.int64)
    pair = np.einsum("k,kpa,kpb->pab", w, f[:, ii, :], f[:, jj, :])
    return site, pair


def _component_logits(inst: Instance, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.int64)
    d = inst.num_sites
    logf = np.log(inst.factors[:, np.arange(d)[None, :], x])  # (K, m, d)
    return np.log(inst.weights)[:, None] + logf.sum(axis=-1)  # (K, m)


def log_prob(inst: Instance, x: np.ndarray) -> np.ndarray:
    """Exact log p0(x) for an (m, d) array of states."""
    logits = _component_logits(inst, x)
    top = logits.max(axis=0)
    return top + np.log(np.exp(logits - top).sum(axis=0))


def profile_cells(inst: Instance, x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Cell index (most likely mode, log p0 bin) of every sample."""
    logits = _component_logits(inst, x)
    top = logits.max(axis=0)
    lp = top + np.log(np.exp(logits - top).sum(axis=0))
    mode = logits.argmax(axis=0)
    return mode * (edges.size + 1) + np.searchsorted(edges, lp, side="right")


def likelihood_profile(
    inst: Instance, rng: np.random.Generator, num_reference: int, num_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reference distribution of (most likely mode, log p0 bin) under p0.

    Bins are equal-mass quantiles of log p0(x) over a large exact sample.
    Returns interior bin edges and the reference probability of every cell.
    """
    chunks = [
        sample_exact(inst, min(20_000, num_reference - k), rng)
        for k in range(0, num_reference, 20_000)
    ]
    lps = np.concatenate([log_prob(inst, c) for c in chunks])
    edges = np.unique(np.quantile(lps, np.arange(1, num_bins) / num_bins))
    cells = np.concatenate([profile_cells(inst, c, edges) for c in chunks])
    size = inst.weights.shape[0] * (edges.size + 1)
    return edges, np.bincount(cells, minlength=size) / cells.size


def sample_exact(inst: Instance, n: int, rng: np.random.Generator) -> np.ndarray:
    """Exact samples from p0 (evaluation-side sanity checks only)."""
    S, d = inst.num_states, inst.num_sites
    comp = rng.choice(inst.weights.shape[0], size=n, p=inst.weights)
    cdf_all = np.cumsum(inst.factors, axis=-1)  # (K, d, S)
    out = np.empty((n, d), dtype=np.int64)
    for start in range(0, n, 20_000):
        c = comp[start : start + 20_000]
        cdf = cdf_all[c]  # (b, d, S)
        u = rng.random((c.size, d, 1)) * cdf[..., -1:]
        out[start : start + c.size] = np.minimum((cdf <= u).sum(axis=-1), S - 1)
    return out
