#!/usr/bin/env python3
"""Train-slice evaluation: run every registered solver on the public instances.

Solvers are discovered on disk (any module under method/ or baseline/ that
defines a top-level SOLVERS dict), run in a separate process against the
exact intensity oracle under evaluator-fixed NFE budgets, and scored from the
returned samples.  A solver that fails any run (exception, time limit, budget
overrun, invalid samples) receives the worst value of every scored metric.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.train import harness  # noqa: E402
from eval.train.instances import (  # noqa: E402
    HORIZON,
    NUM_SITES,
    NUM_STATES,
    Oracle,
    exact_marginals,
    make_instance,
    site_pairs,
)
from eval.train.metrics import (  # noqa: E402
    aggregate,
    failure_metrics,
    pair_kl_floor,
    sample_kls,
)

EXPERIMENT = "multisite_reverse_ctmc"
SLICE = "train"
INSTANCE_SEEDS = [20250201, 20250202, 20250203, 20250204]
NFE_BUDGETS = [32, 64, 128, 256]
NUM_SAMPLES = 50_000
RUN_TIMEOUT_SECONDS = 1800.0
SEED_BASE = 7_340_000


def _instances():
    out = []
    for seed in INSTANCE_SEEDS:
        inst = make_instance(seed, NUM_SITES, NUM_STATES, horizon=HORIZON)
        site, pair = exact_marginals(inst)
        out.append((inst, Oracle(inst), site, pair))
    return out


def run_seeds(instance: int, budget: int, seed_offset: int) -> tuple[int, int]:
    """Initial-state and solver seeds, shared by every method (paired runs)."""
    base = SEED_BASE + 1000 * seed_offset + 37 * instance + budget
    return base, base + 500_000


def discover(extra_files: list[str]) -> list[tuple[str, str | None]]:
    """(file, solver name) pairs; name None marks a file that failed to list."""
    entries = []
    for file in list(harness.solver_files()) + list(extra_files):
        try:
            names = harness.list_solvers(file)
        except Exception as exc:  # noqa: BLE001
            print(f"[{SLICE}] cannot list solvers in {file}: {exc}", file=sys.stderr)
            entries.append((file, None))
            continue
        if not names:
            entries.append((file, None))
        entries.extend((file, name) for name in names)
    return entries


def _keys(entries):
    counts: dict[str, int] = {}
    for file, name in entries:
        key = name if name is not None else file
        counts[key] = counts.get(key, 0) + 1
    keys = []
    for file, name in entries:
        key = name if name is not None else file
        keys.append(key if counts[key] == 1 else f"{key}@{file}")
    return keys


def evaluate_solver(file, name, instances, seed_offset=0, log=True):
    runs = {}
    diag_runs = []
    floor = pair_kl_floor(NUM_STATES, NUM_SAMPLES)
    if name is None:
        return failure_metrics(NFE_BUDGETS), {"status": "failed", "reason": "no solvers listed"}
    for i, (inst, oracle, site, pair) in enumerate(instances):
        for budget in NFE_BUDGETS:
            init_seed, solver_seed = run_seeds(i, budget, seed_offset)
            x0 = np.random.default_rng(init_seed).integers(
                0, inst.num_states, size=(NUM_SAMPLES, inst.num_sites)
            )
            try:
                samples, info = harness.run_solver(
                    file, name, oracle, x0, budget, solver_seed, RUN_TIMEOUT_SECONDS
                )
            except Exception as exc:  # noqa: BLE001
                reason = f"instance {i}, budget {budget}: {exc}"
                if log:
                    print(f"[{SLICE}] {name}: FAILED {reason}", file=sys.stderr)
                return failure_metrics(NFE_BUDGETS), {
                    "status": "failed",
                    "reason": reason[-2000:],
                    "runs": diag_runs,
                }
            kls = sample_kls(samples, site, pair, site_pairs(inst.num_sites))
            runs[(i, budget)] = kls
            diag_runs.append({"instance": i, "nfe_budget": budget, **kls, **info})
            if log:
                print(
                    f"[{SLICE}] {name} inst={i} nfe={budget}: pair_kl={kls['pair_kl']:.5g} "
                    f"site_kl={kls['site_kl']:.5g} used={info['nfe_used']} "
                    f"{info['seconds']}s",
                    file=sys.stderr,
                    flush=True,
                )
    metrics = aggregate(runs, len(instances), NFE_BUDGETS, floor)
    return metrics, {"status": "ok", "runs": diag_runs}


def evaluate_all(extra_files=(), only=None, seed_offset=0):
    instances = _instances()
    entries = discover(list(extra_files))
    keys = _keys(entries)
    results, diagnostics = {}, {}
    for key, (file, name) in zip(keys, entries):
        if only and key not in only:
            continue
        start = time.monotonic()
        metrics, diag = evaluate_solver(file, name, instances, seed_offset)
        diag.update({"file": file, "seconds": round(time.monotonic() - start, 1)})
        results[key] = metrics
        diagnostics[key] = diag
    return results, diagnostics


def scores_document(results, diagnostics, seed_offset=0):
    return {
        "_meta": {
            "slice": SLICE,
            "instances": len(INSTANCE_SEEDS),
            "num_sites": NUM_SITES,
            "num_states": NUM_STATES,
            "horizon": HORIZON,
            "nfe_budgets": NFE_BUDGETS,
            "num_samples": NUM_SAMPLES,
            "seed_offset": seed_offset,
        },
        "experiments": {EXPERIMENT: {"results": results}},
        "_diagnostics": diagnostics,
    }


def write_atomic(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path("scoring/scores_train.json"))
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these method keys")
    parser.add_argument("--solver-file", action="append", default=[], help="extra solver module")
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    if args.scores.exists():
        args.scores.unlink()
    results, diagnostics = evaluate_all(args.solver_file, args.only, args.seed_offset)
    doc = scores_document(results, diagnostics, args.seed_offset)
    write_atomic(args.scores, doc)
    print(json.dumps(doc["experiments"], indent=2))


if __name__ == "__main__":
    main()
