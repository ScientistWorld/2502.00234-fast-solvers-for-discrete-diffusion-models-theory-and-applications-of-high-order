#!/usr/bin/env python3
"""Visible train-slice scoring for saved toy solver outputs.

This evaluator reads method artifacts from outputs/ and does not import
method-specific code.  The train slice uses even-ranked step budgets from each
method's saved convergence grid, leaving odd-ranked budgets for eval/test/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.toy_discrete_diffusion.model import DEFAULT_TARGET, kl_divergence


EXPERIMENT = "toy_15_state_convergence"
SLICE = "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/toy"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_train.json"))
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _collect_methods(output_dir: Path) -> dict[str, list[dict]]:
    methods: dict[str, list[dict]] = {}
    for path in sorted(output_dir.glob("*_outputs.json")):
        payload = _load_json(path)
        for method_name, runs in payload.get("methods", {}).items():
            if not runs:
                continue
            methods.setdefault(method_name, []).extend(runs)
    return methods


def _unique_runs_by_step(runs: list[dict]) -> list[dict]:
    by_step: dict[int, dict] = {}
    for run in runs:
        by_step[int(run["steps"])] = run
    return [by_step[step] for step in sorted(by_step)]


def _slice_runs(runs: list[dict]) -> list[dict]:
    ordered = _unique_runs_by_step(runs)
    selected = [run for index, run in enumerate(ordered) if index % 2 == 0]
    if len(selected) < 2:
        raise ValueError(
            "train slice requires at least two even-ranked step budgets; "
            f"found {len(selected)} from {len(ordered)} unique runs"
        )
    return selected


def _dist(run: dict) -> np.ndarray:
    dist = np.asarray(run["distribution"], dtype=np.float64)
    if dist.ndim != 1 or dist.shape != DEFAULT_TARGET.shape:
        raise ValueError("distribution has unexpected shape")
    total = float(dist.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("distribution must have positive finite mass")
    return dist / total


def _method_metrics(runs: list[dict]) -> dict[str, float]:
    selected_runs = _slice_runs(runs)
    kls: dict[int, float] = {}
    distributions: dict[int, np.ndarray] = {}
    rejections = []
    positives = []

    for run in selected_runs:
        steps = int(run["steps"])
        dist = _dist(run)
        distributions[steps] = dist
        kls[steps] = kl_divergence(DEFAULT_TARGET, dist)
        diagnostics = run.get("diagnostics", {})
        if "rejection_rate" in diagnostics:
            rejections.append(float(diagnostics["rejection_rate"]))
        if "positive_intensity_fraction" in diagnostics:
            positives.append(float(diagnostics["positive_intensity_fraction"]))

    common_steps = sorted(kls)
    x = np.log(np.asarray(common_steps, dtype=np.float64))
    y = np.log(np.asarray([max(kls[s], 1e-12) for s in common_steps], dtype=np.float64))
    slope = float(-np.polyfit(x, y, deg=1)[0])

    max_steps = max(common_steps)
    selected_run = next(run for run in selected_runs if int(run["steps"]) == max_steps)
    max_dist = distributions[max_steps]
    abs_error = np.abs(DEFAULT_TARGET - max_dist)

    metrics = {
        "convergence_order": slope,
        "kl_at_max_steps": float(kls[max_steps]),
        "mean_kl": float(np.mean([kls[s] for s in common_steps])),
        "total_variation_at_max_steps": float(0.5 * np.sum(abs_error)),
        "worst_state_abs_error_at_max_steps": float(np.max(abs_error)),
        "max_steps": float(max_steps),
        "max_nfe": float(selected_run.get("nfe", max_steps)),
        "mean_rejection_rate": float(np.mean(rejections) if rejections else 0.0),
    }
    if positives:
        metrics["positive_intensity_fraction"] = float(np.mean(positives))
    return metrics


def main() -> None:
    args = parse_args()
    methods = _collect_methods(args.output_dir)
    if not methods:
        raise FileNotFoundError(f"no method artifacts found in {args.output_dir}")

    results = {name: _method_metrics(runs) for name, runs in sorted(methods.items())}
    scores = {
        "_meta": {
            "slice": SLICE,
            "artifact_source": str(args.output_dir),
            "split": "even-ranked saved step budgets",
        },
        "experiments": {
            EXPERIMENT: {
                "_slice": SLICE,
                "_split": "even-ranked saved step budgets",
                "results": results,
            }
        },
    }
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
