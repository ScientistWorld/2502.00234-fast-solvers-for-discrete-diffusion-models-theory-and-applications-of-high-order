#!/usr/bin/env python3
"""Held-out test-slice scoring for saved toy solver outputs.

This evaluator reads the same method artifact interface as eval/train/ but
scores odd-ranked saved step budgets and the complementary held-out state
subset.  When scoring/scores_train.json exists, the test slice scores exactly
the method names present there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.toy_discrete_diffusion.model import DEFAULT_TARGET, kl_divergence


EXPERIMENT = "toy_15_state_convergence"
SLICE = "test"
STATE_INDICES = np.asarray([1, 3, 5, 7, 9, 11, 13], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/toy"))
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_test.json"))
    parser.add_argument("--train-scores", type=Path, default=Path("scoring/scores_train.json"))
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


def _train_method_names(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    payload = _load_json(path)
    experiments = payload.get("experiments", {})
    exp = experiments.get(EXPERIMENT, {})
    results = exp.get("results", {})
    if not isinstance(results, dict) or not results:
        return None
    return sorted(name for name in results if not name.startswith("_"))


def _unique_runs_by_step(runs: list[dict]) -> list[dict]:
    by_step: dict[int, dict] = {}
    for run in runs:
        by_step[int(run["steps"])] = run
    return [by_step[step] for step in sorted(by_step)]


def _slice_runs(runs: list[dict]) -> list[dict]:
    ordered = _unique_runs_by_step(runs)
    selected = [run for index, run in enumerate(ordered) if index % 2 == 1]
    if len(selected) < 2:
        raise ValueError(
            "test slice requires at least two odd-ranked step budgets; "
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


def _subset_kl(target: np.ndarray, dist: np.ndarray) -> float:
    target_subset = target[STATE_INDICES]
    dist_subset = dist[STATE_INDICES]
    target_mass = float(target_subset.sum())
    dist_mass = float(dist_subset.sum())
    if target_mass <= 0.0 or dist_mass <= 0.0:
        raise ValueError("state subset must have positive target and predicted mass")
    return kl_divergence(target_subset / target_mass, dist_subset / dist_mass)


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
    subset_target_mass = float(DEFAULT_TARGET[STATE_INDICES].sum())
    subset_dist_mass = float(max_dist[STATE_INDICES].sum())

    metrics = {
        "convergence_order": slope,
        "kl_at_max_steps": float(kls[max_steps]),
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
        "slice_state_subset_kl_at_max_steps": _subset_kl(DEFAULT_TARGET, max_dist),
        "slice_state_subset_mass_abs_error_at_max_steps": abs(
            subset_target_mass - subset_dist_mass
        ),
        "max_steps": float(max_steps),
        "max_nfe": float(selected_run.get("nfe", max_steps)),
        "mean_rejection_rate": float(np.mean(rejections) if rejections else 0.0),
        "positive_intensity_fraction": float(np.mean(positives) if positives else 0.0),
    }
    return metrics


def main() -> None:
    args = parse_args()
    methods = _collect_methods(args.output_dir)
    if not methods:
        raise FileNotFoundError(f"no method artifacts found in {args.output_dir}")

    train_names = _train_method_names(args.train_scores)
    if train_names is None:
        method_names = sorted(methods)
    else:
        missing = [name for name in train_names if name not in methods]
        if missing:
            raise FileNotFoundError(
                "test slice could not find artifacts for train-scored methods: "
                + ", ".join(missing)
            )
        method_names = train_names

    results = {name: _method_metrics(methods[name]) for name in method_names}
    scores = {
        "_meta": {
            "slice": SLICE,
            "artifact_source": str(args.output_dir),
            "train_scores": str(args.train_scores),
            "split": "odd-ranked saved step budgets",
            "state_indices": STATE_INDICES.tolist(),
        },
        "experiments": {
            EXPERIMENT: {
                "_slice": SLICE,
                "_split": "odd-ranked saved step budgets",
                "_state_indices": STATE_INDICES.tolist(),
                "results": results,
            }
        },
    }
    args.scores.parent.mkdir(parents=True, exist_ok=True)
    args.scores.write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
