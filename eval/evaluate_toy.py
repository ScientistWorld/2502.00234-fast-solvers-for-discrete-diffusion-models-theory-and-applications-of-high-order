#!/usr/bin/env python3
"""Evaluate saved toy-model solver outputs.

The evaluator intentionally reads artifacts from outputs/ and avoids solution
package imports.  Future methods can write the same JSON format and be scored
by the same script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.toy_discrete_diffusion.model import DEFAULT_TARGET, kl_divergence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/toy"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores.json"))
    return parser.parse_args()


def _load(path: Path) -> dict:
    if not path.exists():
        return {"methods": {}}
    return json.loads(path.read_text())


def _method_metrics(runs: list[dict]) -> dict[str, float]:
    by_steps = {int(run["steps"]): run for run in runs}
    kls = {}
    distributions = {}
    rejections = []
    positives = []
    for steps, run in by_steps.items():
        dist = np.asarray(run["distribution"], dtype=np.float64)
        total = float(dist.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("distribution must have positive finite mass")
        dist = dist / total
        distributions[steps] = dist
        kls[steps] = kl_divergence(DEFAULT_TARGET, dist)
        diagnostics = run.get("diagnostics", {})
        if "rejection_rate" in diagnostics:
            rejections.append(float(diagnostics["rejection_rate"]))
        if "positive_intensity_fraction" in diagnostics:
            positives.append(float(diagnostics["positive_intensity_fraction"]))

    common_steps = sorted(kls)
    if len(common_steps) >= 2:
        x = np.log(np.asarray(common_steps, dtype=np.float64))
        y = np.log(np.asarray([kls[s] for s in common_steps], dtype=np.float64))
        slope = float(-np.polyfit(x, y, deg=1)[0])
    else:
        slope = 0.0

    selected = max(common_steps)
    selected_run = by_steps[selected]
    selected_dist = distributions[selected]
    abs_error = np.abs(DEFAULT_TARGET - selected_dist)
    even_state_indices = np.asarray([0, 2, 4, 6, 8, 10, 12, 14], dtype=np.int64)
    target_subset = DEFAULT_TARGET[even_state_indices]
    dist_subset = selected_dist[even_state_indices]
    target_subset_mass = float(target_subset.sum())
    dist_subset_mass = float(dist_subset.sum())
    subset_kl = kl_divergence(
        target_subset / target_subset_mass,
        dist_subset / dist_subset_mass,
    )
    metrics = {
        "kl_at_max_steps": float(kls[selected]),
        "convergence_order": slope,
        "mean_kl": float(np.mean([kls[s] for s in common_steps])),
        "mean_total_variation": float(
            np.mean(
                [
                    0.5 * np.sum(np.abs(DEFAULT_TARGET - distributions[s]))
                    for s in common_steps
                ]
            )
        ),
        "total_variation_at_max_steps": float(0.5 * np.sum(abs_error)),
        "worst_state_abs_error_at_max_steps": float(np.max(abs_error)),
        "slice_state_subset_kl_at_max_steps": float(subset_kl),
        "slice_state_subset_mass_abs_error_at_max_steps": float(
            abs(target_subset_mass - dist_subset_mass)
        ),
        "max_steps": float(selected),
        "max_nfe": float(selected_run.get("nfe", selected)),
        "mean_rejection_rate": float(np.mean(rejections) if rejections else 0.0),
        "positive_intensity_fraction": float(np.mean(positives) if positives else 0.0),
    }
    return metrics


def main() -> None:
    args = parse_args()
    method_data = _load(args.output_dir / "method_outputs.json")
    baseline_data = _load(args.output_dir / "baseline_outputs.json")

    results = {}
    for source in (baseline_data, method_data):
        for method_name, runs in source.get("methods", {}).items():
            if runs:
                results[method_name] = _method_metrics(runs)

    scores = {
        "experiments": {
            "toy_15_state_convergence": {
                "results": results,
            }
        }
    }
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
