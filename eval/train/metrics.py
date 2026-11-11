"""Distribution-quality metrics for multi-site samples (train slice)."""

from __future__ import annotations

import math

import numpy as np

# Kullback-Leibler values are in nats.  With add-half smoothing the largest
# attainable value is below log(2n + k) < 13 for every configuration used
# here, so FAIL_KL is strictly worse than any real output.
FAIL_KL = 20.0
KL_MIN = 1e-12
ORDER_BOUNDS = (-6.0, 6.0)


def _kl_rows(p: np.ndarray, counts: np.ndarray, n: int) -> np.ndarray:
    """KL(p || q_hat) along the last axis, q_hat = add-half smoothed counts."""
    k = p.shape[-1]
    q = (counts + 0.5) / (n + 0.5 * k)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * (np.log(p) - np.log(q)), 0.0)
    return terms.sum(axis=-1)


def _clip(value: float) -> float:
    return float(np.clip(value, KL_MIN, FAIL_KL))


def sample_kls(
    samples: np.ndarray,
    site_marginals: np.ndarray,
    pair_marginals: np.ndarray,
    pairs: list[tuple[int, int]],
    profile_cells: np.ndarray,
    profile_probs: np.ndarray,
) -> dict[str, float]:
    """KL from the target to the samples on three evaluator-fixed projections.

    site_kl:     mean over sites of KL on the 15-state site marginal.
    pair_kl:     mean over all site pairs of KL on the 225-cell pair marginal.
    profile_kl:  KL on the joint (most likely mode, log p0(x) decile) profile,
                 which sees mode proportions and the typicality of the full
                 joint configuration.
    """
    n, d = samples.shape
    S = site_marginals.shape[1]
    site_counts = np.stack(
        [np.bincount(samples[:, i], minlength=S) for i in range(d)]
    ).astype(np.float64)
    pair_counts = np.stack(
        [
            np.bincount(samples[:, i] * S + samples[:, j], minlength=S * S)
            for i, j in pairs
        ]
    ).astype(np.float64)
    profile_counts = np.bincount(profile_cells, minlength=profile_probs.size).astype(
        np.float64
    )
    site_kl = _kl_rows(site_marginals, site_counts, n)
    pair_kl = _kl_rows(pair_marginals.reshape(len(pairs), S * S), pair_counts, n)
    profile_kl = _kl_rows(profile_probs, profile_counts, n)
    return {
        "site_kl": _clip(np.mean(site_kl)),
        "pair_kl": _clip(np.mean(pair_kl)),
        "profile_kl": _clip(profile_kl),
    }


def pair_kl_floor(num_states: int, num_samples: int) -> float:
    """First-order expected pair KL of an exact sampler: (k - 1) / (2n)."""
    return (num_states**2 - 1) / (2.0 * num_samples)


def metric_names(budgets: list[int]) -> list[str]:
    names = [f"log10_pair_kl_nfe{b}" for b in budgets]
    return names + ["log10_profile_kl_mean", "log10_site_kl_mean", "convergence_order"]


def failure_metrics(budgets: list[int]) -> dict[str, float]:
    worst = math.log10(FAIL_KL)
    metrics = {f"log10_pair_kl_nfe{b}": worst for b in budgets}
    metrics["log10_profile_kl_mean"] = worst
    metrics["log10_site_kl_mean"] = worst
    metrics["convergence_order"] = ORDER_BOUNDS[0]
    return metrics


def aggregate(
    runs: dict[tuple[int, int], dict[str, float]],
    num_instances: int,
    budgets: list[int],
    floor: float,
) -> dict[str, float]:
    """Combine per-(instance, budget) KLs into the scored metric vector.

    Per budget, KLs are averaged over instances in log10 space (geometric
    mean).  The convergence order is the negative least-squares log-log slope
    of the instance-geometric-mean pair KL against the NFE budget, after
    removing the first-order finite-sample floor (clipped at a tenth of it).
    """
    metrics: dict[str, float] = {}
    log_excess, site_logs, profile_logs = [], [], []
    for b in budgets:
        rows = [runs[(i, b)] for i in range(num_instances)]
        pair = np.array([r["pair_kl"] for r in rows])
        metrics[f"log10_pair_kl_nfe{b}"] = float(np.mean(np.log10(pair)))
        site_logs.extend(np.log10([r["site_kl"] for r in rows]).tolist())
        profile_logs.extend(np.log10([r["profile_kl"] for r in rows]).tolist())
        excess = np.maximum(pair - floor, 0.1 * floor)
        log_excess.append(float(np.mean(np.log(excess))))
    metrics["log10_profile_kl_mean"] = float(np.mean(profile_logs))
    metrics["log10_site_kl_mean"] = float(np.mean(site_logs))
    slope = np.polyfit(np.log(np.asarray(budgets, dtype=np.float64)), log_excess, 1)[0]
    lo, hi = ORDER_BOUNDS
    metrics["convergence_order"] = float(np.clip(-slope, lo, hi))
    return metrics
