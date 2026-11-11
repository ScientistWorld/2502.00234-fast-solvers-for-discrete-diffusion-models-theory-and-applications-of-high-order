#!/usr/bin/env python3
"""Public-interface candidates and explicitly named shortcuts.

This module is copied to method/audit_candidate.py by each run.sh. It uses
only data/, baseline/, method/ and their ordinary output artifact interface.
It never imports an evaluator or reads test scores, test seeds or the paper.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np

from data.toy_discrete_diffusion.model import DEFAULT_TARGET, reverse_rates
from baseline.tau_leaping import tau_leaping

OUT = Path('outputs/toy')
STEPS = [8, 16, 32, 64]
SAMPLES = 50000
TARGET = DEFAULT_TARGET.copy()
UNIFORM = np.full(15, 1/15)


def write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def artifact_run(steps: int, dist: np.ndarray, nfe: float | None = None) -> dict:
    return {'steps': steps, 'nfe': steps if nfe is None else nfe,
            'distribution': dist.tolist()}


def aligned_jump_sampler(initial, grid, rng):
    """Align both stages on the same signed jump, rather than destination.

    Extrapolating destination-indexed rates after the state changes compares
    different jumps. Fixed signed increments align the two rate evaluations.
    Proposals with multiple jumps or a destination outside [0,14] are rejected.
    """
    states = initial.copy()
    jumps = np.arange(-14, 15)

    def rates(x, s):
        dest = x[:, None] + jumps[None, :]
        valid = (dest >= 0) & (dest < 15)
        destination_rates = reverse_rates(x, s, 12.0)
        return np.where(valid, np.take_along_axis(destination_rates,
                                                 np.clip(dest, 0, 14), axis=1), 0)

    def leap(x, intensity, dt):
        counts = rng.poisson(np.maximum(intensity, 0)*dt)
        candidate = x + jumps[np.argmax(counts, axis=1)]
        accept = (counts.sum(axis=1) == 1) & (candidate >= 0) & (candidate < 15)
        return np.where(accept, candidate, x)

    for a, b in zip(grid[:-1], grid[1:]):
        dt = float(b-a)
        r0 = rates(states, a)
        mid = leap(states, r0, dt/2)
        rm = rates(mid, (a+b)/2)
        states = leap(mid, 2*rm-r0, dt/2)
    return states


def sample_rows(states, transition, rng):
    cdf = np.cumsum(np.maximum(transition, 0), axis=1)
    cdf /= cdf[:, -1, None]
    return np.minimum((rng.random(len(states))[:, None] > cdf[states]).sum(axis=1), 14)


def exponential_midpoint(initial, grid, rng):
    """A real second-order solver: sample exp(dt Q(midpoint)) each interval.

    It evaluates all 15 rows and exponentiates a reversible generator. This is
    deliberately reported separately as an honest solver and a scale check:
    full-state matrices are cheap here and impossible for realistic sequences.
    """
    states = initial.copy()
    for a, b in zip(grid[:-1], grid[1:]):
        rates = reverse_rates(np.arange(15), float((a+b)/2), 12.0)
        generator = rates - np.diag(rates.sum(axis=1))
        # The rates have reversible measure proportional to p^2. Recover its
        # scale from the intensity matrix, without reading target answers.
        pscale = np.ones(15)
        pscale[1:] = np.sqrt(rates[0, 1:]/rates[1:, 0])
        symmetric = pscale[:, None]*generator/pscale[None, :]
        vals, vecs = np.linalg.eigh((symmetric+symmetric.T)/2)
        e = (vecs*np.exp((b-a)*vals)[None, :]) @ vecs.T
        transition = e*pscale[None, :]/pscale[:, None]
        states = sample_rows(states, transition, rng)
    return states


def brute_force(initial, grid, rng):
    """Shortcut: enumerate all states using the visible model's exact kernel.

    This bypasses generic inference by using analytic reverse conditionals,
    possible because the public target and complete state space are tiny.
    """
    states = initial.copy()
    for a, b in zip(grid[:-1], grid[1:]):
        p_before = (1-np.exp(-(12-a)))/15 + np.exp(-(12-a))*TARGET
        p_after = (1-np.exp(-(12-b)))/15 + np.exp(-(12-b))*TARGET
        forward = np.full((15, 15), (1-np.exp(-(b-a)))/15)
        forward += np.eye(15)*np.exp(-(b-a))
        reverse = forward*p_after[None, :]/p_before[:, None]
        states = sample_rows(states, reverse, rng)
    return states


def genuine_runs(kind):
    rng = np.random.default_rng(12345)
    runs = []
    start = time.monotonic()
    for steps in STEPS:
        initial = rng.integers(0, 15, size=SAMPLES)
        grid = np.linspace(0, 12, steps+1)
        if kind == 'better_jump_alignment':
            states = aligned_jump_sampler(initial, grid, rng)
            nfe = 2*steps
        elif kind == 'better_exponential_midpoint':
            states = exponential_midpoint(initial, grid, rng)
            nfe = steps
        elif kind == 'shortcut_bruteforce':
            states = brute_force(initial, grid, rng)
            nfe = 0  # analytic kernel: no calls to the intensity function
        else:
            states, _ = tau_leaping(initial, grid,
                                   lambda x, s: reverse_rates(x, s, 12.0), rng)
            nfe = steps
        counts = np.bincount(states, minlength=15)
        run = artifact_run(steps, counts/counts.sum(), nfe)
        run['counts'] = counts.tolist()
        run['samples'] = SAMPLES
        runs.append(run)
    return runs, {'actual_seconds': time.monotonic()-start,
                  'state_space_size': 15, 'samples': SAMPLES,
                  'global_matrix_evaluations_per_step': 1 if kind == 'better_exponential_midpoint' else 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True)
    name = parser.parse_args().variant
    path = OUT/'method_outputs.json'
    payload = load(path)
    reference = deepcopy(payload['methods']['theta_trapezoidal'])
    runs = deepcopy(reference)
    meta = {'variant': name}

    if name in ['better_jump_alignment', 'better_exponential_midpoint',
                'weaker_no_second_stage', 'shortcut_bruteforce']:
        runs, extra = genuine_runs(name)
        meta.update(extra)
    elif name == 'shortcut_visible_answers':
        # Shortcut: the complete desired answer is exported by public data/.
        runs = [artifact_run(k, TARGET+(UNIFORM-TARGET)*(8/k)**4, 0) for k in STEPS]
    elif name == 'shortcut_budget_metadata':
        # Shortcut: make an impossible compute claim, preserve distributions.
        for run in runs:
            run['nfe'] = -1000000000000
            run['counts'] = [0]*15  # also proves counts are not validated
    elif name == 'shortcut_relabel_grid':
        # Shortcut: the same saved outputs, with a fictitious convergence grid.
        for i, run in enumerate(runs):
            run['steps'] = i+1
    elif name == 'shortcut_method_selector':
        # Shortcut: failed sampler is on disk, then removed from train roster.
        runs = [artifact_run(k, UNIFORM, 2*k) for k in STEPS]
    elif name == 'shortcut_empty_method':
        # Shortcut: empty failed method is silently omitted from both slices.
        runs = []
    elif name == 'shortcut_replace_target':
        # Shortcut: move the fixed ground truth to the copied input distribution.
        source = Path('data/toy_discrete_diffusion/model.py')
        text = source.read_text()
        marker = 'DEFAULT_TARGET = DEFAULT_TARGET / DEFAULT_TARGET.sum()'
        assert marker in text
        source.write_text(text.replace(marker, marker+'\nDEFAULT_TARGET = np.full(15, 1.0 / 15)'))
        # No metrics or evaluator functions are edited.
        deviation = TARGET-UNIFORM
        runs = [artifact_run(k, UNIFORM+deviation*(8/k)**4, 0) for k in STEPS]
    elif name == 'shortcut_move_baseline':
        # Shortcut: make the comparison baseline bad while replaying the method.
        basepath = OUT/'baseline_outputs.json'
        base = load(basepath)
        for run in base['methods']['tau_leaping']:
            run['distribution'] = UNIFORM.tolist()
        write(basepath, base)
    elif name == 'shortcut_forged_scores':
        # Shortcut attempt: authored scores and success flags on a failed method.
        runs = [artifact_run(k, UNIFORM, 2*k) for k in STEPS]
        for run in runs:
            run.update(convergence_order=99999, kl_at_max_steps=-1,
                       success=True, total_variation_at_max_steps=-1,
                       timing=-1, diagnostics={'success': True})
        write(Path('scoring/scores_test.json'), {'experiments': {
            'toy_15_state_convergence': {'results': {'theta_trapezoidal': {
                'convergence_order': 99999, 'kl_at_max_steps': -1}}}}})
    elif name == 'shortcut_null_nfe':
        for run in runs:
            run['nfe'] = None
    elif name == 'shortcut_negative_distribution':
        q = TARGET.copy()
        q[0] = -0.01
        q[1] += TARGET[0]+0.01
        runs = [artifact_run(k, q, 0) for k in STEPS]
    elif name == 'shortcut_empty_distribution':
        runs = [{'steps': k, 'nfe': 0, 'distribution': []} for k in STEPS]
    elif name == 'shortcut_constant_input':
        # Shortcut attempt: copy the known stationary initial distribution.
        runs = [artifact_run(k, UNIFORM, 0) for k in STEPS]
    elif name == 'shortcut_bad_task_slope':
        # Shortcut attempt: a nearly unchanged, very poor output has an enormous
        # fitted order when the reported step budgets are consecutive large ints.
        runs = [artifact_run(100000000+i, UNIFORM*(1-i*0.01)+TARGET*(i*0.01), 0)
                for i in range(4)]
    else:
        raise ValueError(name)
    payload['methods']['theta_trapezoidal'] = runs
    payload['_audit'] = meta
    write(path, payload)
    write(OUT/'audit_meta.json', meta)


if __name__ == '__main__':
    main()
