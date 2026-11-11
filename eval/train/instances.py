"""Public train instances of the multi-site analytic discrete diffusion problem.

State space X = [S]^d (d sites, S states per site).  Each site is noised
independently by the uniform-rate CTMC Q = E / S - I of the paper's toy model,
run for horizon T = 12.  The data distribution p0 is a mixture of K Markov
chains; every component visits the sites in its own random order, and its
initial distribution and transition rows are drawn uniformly from the simplex.
The forward marginals p_t stay mixtures of hidden Markov models, so the exact
reverse-process jump intensities are available in closed form through
forward-backward recursions.

This module holds the generator, the exact intensity oracle, the exact
low-order marginals used for scoring, and an exact sampler used only by
evaluation-side sanity checks.  It is evaluation code: solvers never receive
these objects, only the intensities that the harness returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUM_STATES = 15
NUM_SITES = 10
HORIZON = 12.0
NUM_COMPONENTS = 3
SIMPLEX_CONCENTRATION = 1.0
MIXTURE_CONCENTRATION = 2.0


@dataclass(frozen=True)
class Instance:
    weights: np.ndarray  # (K,)
    orders: np.ndarray  # (K, d): site visited at chain position j
    initial: np.ndarray  # (K, S)
    transitions: np.ndarray  # (K, d - 1, S, S)
    num_states: int
    num_sites: int
    horizon: float


def make_instance(
    seed: int,
    num_sites: int = NUM_SITES,
    num_states: int = NUM_STATES,
    num_components: int = NUM_COMPONENTS,
    horizon: float = HORIZON,
) -> Instance:
    rng = np.random.default_rng(seed)
    K, d, S = num_components, num_sites, num_states
    weights = rng.dirichlet(np.full(K, MIXTURE_CONCENTRATION))
    orders = np.stack([rng.permutation(d) for _ in range(K)]).astype(np.int64)
    initial = rng.dirichlet(np.full(S, SIMPLEX_CONCENTRATION), size=K)
    transitions = rng.dirichlet(
        np.full(S, SIMPLEX_CONCENTRATION), size=(K, max(d - 1, 0), S)
    )
    return Instance(weights, orders, initial, transitions, S, d, float(horizon))


class Oracle:
    """Exact reverse-CTMC jump intensities for one instance."""

    def __init__(self, inst: Instance):
        self.inst = inst
        self._states = np.arange(inst.num_states)

    def rates(self, states: np.ndarray, s) -> np.ndarray:
        """Intensities of single-site jumps x -> x^{i->y} at reverse time s.

        states: (m, d) integer array; s: scalar or (m,) reverse times in [0, T].
        Returns (m, d, S) with rates[j, i, y] = p_t(x^{i->y}) / p_t(x) / S for
        y != x_i (forward time t = T - s) and 0 for y == x_i.
        """
        inst = self.inst
        S, d = inst.num_states, inst.num_sites
        states = np.asarray(states, dtype=np.int64)
        m = states.shape[0]
        if m == 0:
            return np.zeros((0, d, S))
        t = inst.horizon - np.asarray(s, dtype=np.float64)
        decay = np.exp(-t)
        floor = (1.0 - decay) / S
        if decay.ndim:
            decay = decay[:, None]
            floor = floor[:, None]
        onehot = states[:, :, None] == self._states  # (m, d, S)
        prob = np.zeros(m)
        neighbours = np.zeros((m, d, S))
        for k in range(inst.weights.shape[0]):
            order = inst.orders[k]
            A = inst.transitions[k]
            # Emission K_t(z_j | a) in chain order: (d, m, S).
            emit = floor + decay * np.moveaxis(onehot[:, order, :], 1, 0)
            phi = np.empty((d, m, S))
            beta = np.empty((d, m, S))
            phi[0] = inst.initial[k]
            for j in range(d - 1):
                phi[j + 1] = S * ((phi[j] * emit[j]) @ A[j])
            beta[d - 1] = 1.0
            for j in range(d - 2, -1, -1):
                beta[j] = S * ((emit[j + 1] * beta[j + 1]) @ A[j].T)
            pb = phi * beta  # every slice carries the same S^(d-1) scale
            total = (pb[d - 1] * emit[d - 1]).sum(axis=-1)
            nb = floor * pb.sum(axis=-1)[..., None] + decay * pb  # (d, m, S)
            neighbours[:, order, :] += inst.weights[k] * np.moveaxis(nb, 0, 1)
            prob += inst.weights[k] * total
        rates = neighbours / (S * prob[:, None, None])
        rates[onehot] = 0.0
        return rates


def site_pairs(num_sites: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(num_sites) for j in range(i + 1, num_sites)]


def exact_marginals(inst: Instance) -> tuple[np.ndarray, np.ndarray]:
    """Exact site marginals (d, S) and site-pair marginals (P, S, S) of p0."""
    S, d = inst.num_states, inst.num_sites
    pairs = site_pairs(d)
    pair_index = {pair: idx for idx, pair in enumerate(pairs)}
    site = np.zeros((d, S))
    pair = np.zeros((len(pairs), S, S))
    for k in range(inst.weights.shape[0]):
        w = inst.weights[k]
        order = inst.orders[k]
        A = inst.transitions[k]
        chain = np.empty((d, S))
        chain[0] = inst.initial[k]
        for j in range(d - 1):
            chain[j + 1] = chain[j] @ A[j]
        for j in range(d):
            site[order[j]] += w * chain[j]
        for j in range(d):
            M = np.eye(S)
            for l in range(j + 1, d):
                M = M @ A[l - 1]
                joint = chain[j][:, None] * M
                a, b = int(order[j]), int(order[l])
                if a < b:
                    pair[pair_index[(a, b)]] += w * joint
                else:
                    pair[pair_index[(b, a)]] += w * joint.T
    return site, pair


def sample_exact(inst: Instance, n: int, rng: np.random.Generator) -> np.ndarray:
    """Exact samples from p0 (evaluation-side sanity checks only)."""
    S, d = inst.num_states, inst.num_sites
    comp = rng.choice(inst.weights.shape[0], size=n, p=inst.weights)
    out = np.empty((n, d), dtype=np.int64)

    def draw(probs: np.ndarray) -> np.ndarray:
        cdf = np.cumsum(probs, axis=-1)
        u = rng.random(probs.shape[0]) * cdf[:, -1]
        return np.minimum((cdf <= u[:, None]).sum(axis=1), S - 1)

    for k in range(inst.weights.shape[0]):
        idx = np.nonzero(comp == k)[0]
        if idx.size == 0:
            continue
        order = inst.orders[k]
        a = draw(np.broadcast_to(inst.initial[k], (idx.size, S)))
        out[idx, order[0]] = a
        for j in range(d - 1):
            a = draw(inst.transitions[k][j][a])
            out[idx, order[j + 1]] = a
    return out
