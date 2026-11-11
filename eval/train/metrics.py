"""Distribution-quality metrics for multi-site samples (train slice)."""

from __future__ import annotations

import math

import numpy as np

# Kullback-Leibler values are reported in nats.  With add-half smoothing the
# largest attainable value is log(2n + k) < 13 for every configuration used
# here, so FAIL_KL is strictly worse than any real output.
FAIL_KL = 20.0
ORDER_BOUNDS = (-6.0, 6.0)


def _kl_rows(p: np.ndarray, counts: np.ndarray, n: int) -> np.ndarray:
    """KL(p || q_hat) along the last axis, q_hat add-half smoothed counts."""
    k = p.shape[-1]
    q = (counts + 0.5) / (n + 0.5 * k)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, p * (np.log(p) - np.log(q)), 0.0)
    return terms.sum(axis=-1)


def sample_kls(
    samples: np.ndarray,
    site_marginals: np.ndarray,
    pair_marginals: np.ndarray,
    pairs: list[tuple[int, int]],
) -> dict[str, float]:
    """Mean site-marginal KL and mean site-pair-marginal KL of a sample set."""
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
    site_kl = _kl_rows(site_marginals, site_counts, n)
    pair_kl = _kl_rows(pair_marginals.reshape(len(pairs), S * S), pair_counts, n)
    return {
        "site_kl": float(np.clip(np.mean(site_kl), 1e-12, FAIL_KL)),
        "pair_kl": float(np.clip(np.mean(pair_kl), 1e-12, FAIL_KL)),
    }


def pair_kl_floor(num_states: int, num_samples: int) -> float:
    """First-order expected pair KL of an exact sampler: (k - 1) / (2n)."""
    return (num_states**2 - 1) / (2.0 * num_samples)


def metric_names(budgets: list[int]) -> list[str]:
    names = [f"log10_pair_kl_nfe{b}" for b in budgets]
    return names + ["log10_site_kl_mean", "convergence_order"]


def failure_metrics(budgets: list[int]) -> dict[str, float]:
    worst = math.log10(FAIL_KL)
    metrics = {f"log10_pair_kl_nfe{b}": worst for b in budgets}
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

    Per budget, KL is averaged over instances in log10 space (a geometric
    mean).  The convergence order is the negative log-log slope of the
    instance-geometric-mean pair KL against the NFE budget, after removing the
    first-order finite-sample floor (clipped at a tenth of that floor).
    """
    metrics: dict[str, float] = {}
    log_excess = []
    site_logs = []
    for b in budgets:
        pair = np.array([runs[(i, b)]["pair_kl"] for i in range(num_instances)])
        site = np.array([runs[(i, b)]["site_kl"] for i in range(num_instances)])
        metrics[f"log10_pair_kl_nfe{b}"] = float(np.mean(np.log10(pair)))
        site_logs.extend(np.log10(site).tolist())
        excess = np.maximum(pair - floor, 0.1 * floor)
        log_excess.append(float(np.mean(np.log(excess))))
    metrics["log10_site_kl_mean"] = float(np.mean(site_logs))
    slope = np.polyfit(np.log(np.asarray(budgets, dtype=np.float64)), log_excess, 1)[0]
    lo, hi = ORDER_BOUNDS
    metrics["convergence_order"] = float(np.clip(-slope, lo, hi))
    return metrics
