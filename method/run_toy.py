#!/usr/bin/env python3
"""Run the paper's high-order solvers on the analytic 15-state toy model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.toy_discrete_diffusion.model import DEFAULT_TARGET, reverse_rates
from method.fast_solvers.samplers import theta_rk2, theta_trapezoidal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/toy"))
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument("--steps", type=str, default="16,32,64,128")
    parser.add_argument("--horizon", type=float, default=12.0)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    d = DEFAULT_TARGET.shape[0]
    step_counts = [int(x) for x in args.steps.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    summary: dict[str, object] = {
        "experiment": "toy_15_state_convergence",
        "samples": args.samples,
        "horizon": args.horizon,
        "theta": args.theta,
        "target": DEFAULT_TARGET.tolist(),
        "methods": {},
    }

    for method_name, sampler in {
        "theta_rk2": theta_rk2,
        "theta_trapezoidal": theta_trapezoidal,
    }.items():
        method_runs = []
        for steps in step_counts:
            initial = rng.integers(0, d, size=args.samples, endpoint=False)
            grid = np.linspace(0.0, args.horizon, steps + 1)

            def rate_fn(states: np.ndarray, s: float) -> np.ndarray:
                return reverse_rates(states, s, args.horizon, DEFAULT_TARGET)

            final_states, diagnostics = sampler(initial, grid, rate_fn, rng, theta=args.theta)
            counts = np.bincount(final_states, minlength=d).astype(int)
            dist = counts / counts.sum()
            method_runs.append(
                {
                    "steps": steps,
                    "nfe": 2 * steps,
                    "counts": counts.tolist(),
                    "distribution": dist.tolist(),
                    "diagnostics": diagnostics,
                }
            )
        summary["methods"][method_name] = method_runs

    output_path = args.output_dir / "method_outputs.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

