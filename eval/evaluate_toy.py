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
    rejections = []
    positives = []
    for steps, run in by_steps.items():
        dist = np.asarray(run["distribution"], dtype=np.float64)
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
    metrics = {
        "kl_at_max_steps": float(kls[selected]),
        "convergence_order": slope,
        "max_steps": float(selected),
        "mean_rejection_rate": float(np.mean(rejections) if rejections else 0.0),
    }
    if positives:
        metrics["positive_intensity_fraction"] = float(np.mean(positives))
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
