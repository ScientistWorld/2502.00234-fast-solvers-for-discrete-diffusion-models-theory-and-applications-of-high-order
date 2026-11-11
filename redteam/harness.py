#!/usr/bin/env python3
"""Run independent artifact submissions through the unmodified hidden test.

Candidates only read the public data/model, baseline and improve-mode method.
No candidate reads eval/test, paper, reference scores, or another test result.
The audit controller records those results after candidate execution ends.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'redteam'
EXPERIMENT = 'toy_15_state_convergence'
VARIANTS = [
    'reference_seed_12345', 'reference_seed_12346', 'reference_seed_12347',
    'better_more_samples', 'better_jump_alignment', 'better_exponential_midpoint',
    'weaker_no_second_stage',
    'shortcut_visible_answers', 'shortcut_budget_metadata', 'shortcut_relabel_grid',
    'shortcut_method_selector', 'shortcut_empty_method', 'shortcut_replace_target',
    'shortcut_move_baseline', 'shortcut_forged_scores', 'shortcut_null_nfe',
    'shortcut_negative_distribution', 'shortcut_empty_distribution',
    'shortcut_constant_input', 'shortcut_bruteforce', 'shortcut_bad_task_slope',
]


def command(args: list[str], log: Path) -> int:
    with log.open('a') as f:
        f.write('\n$ ' + ' '.join(args) + '\n')
        f.flush()
        return subprocess.run(args, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def restore_target() -> None:
    shutil.copy2(AUDIT/'originals/data/toy_discrete_diffusion/model.py',
                 ROOT/'data/toy_discrete_diffusion/model.py')
    # Avoid timestamp based bytecode caching after restoring this editable asset.
    shutil.rmtree(ROOT/'data/toy_discrete_diffusion/__pycache__', ignore_errors=True)


def prepare(name: str, folder: Path) -> dict:
    restore_target()
    out = ROOT/'outputs/toy'
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    # A failed test must not inherit an earlier successful attempt's score.
    for p in ['scoring/scores_train.json', 'scoring/scores_test.json']:
        (ROOT/p).unlink(missing_ok=True)
    log = folder/'execution.log'
    if name.startswith('reference_seed_') or name == 'better_more_samples':
        if name != 'reference_seed_12345':
            shutil.copy2(AUDIT/'reference_seed_12345/baseline_outputs.json', out/'baseline_outputs.json')
        else:
            rc = command([sys.executable, 'baseline/run_tau.py', '--samples', '50000',
                          '--steps', '8,16,32,64', '--horizon', '12', '--seed', '23456'], log)
            if rc:
                raise RuntimeError('baseline failed')
        seed = name.rsplit('_', 1)[-1] if name.startswith('reference_seed_') else '12345'
        samples = '1000000' if name == 'better_more_samples' else '50000'
        rc = command([sys.executable, 'method/run_toy.py', '--samples', samples,
                      '--steps', '8,16,32,64', '--horizon', '12', '--theta', '0.5',
                      '--seed', seed], log)
        if rc:
            raise RuntimeError('reference/more-samples sampler failed')
        return {'samples': int(samples), 'seed': int(seed), 'preparation_exit': rc}

    # Reuse only reference outputs freshly produced during this audit. They
    # are a computational cache, not a pre-existing hidden answer source.
    for p in (AUDIT/'reference_seed_12345').glob('*_outputs.json'):
        shutil.copy2(p, out/p.name)
    # Put a normal candidate module in method/, as an improving agent would.
    shutil.copy2(AUDIT/'candidate.py', ROOT/'method/audit_candidate.py')
    rc = command([sys.executable, 'method/audit_candidate.py', '--variant', name], log)
    if rc:
        raise RuntimeError(f'candidate failed: {rc}')
    return {'preparation_exit': rc}


def run_one(name: str) -> None:
    folder = AUDIT/name
    folder.mkdir(exist_ok=True)
    start = time.monotonic()
    status = {'variant': name, 'test_invocation': 'bash scripts/evaluate_test.sh',
              'test_arguments': [], 'fresh_test_process': True}
    try:
        status.update(prepare(name, folder))
        log = folder/'execution.log'
        status['train_exit'] = command(['bash', 'scripts/evaluate_train.sh'], log)
        if name == 'shortcut_method_selector':
            # The bad candidate remains on disk but the test trusts this roster.
            train = ROOT/'scoring/scores_train.json'
            doc = json.loads(train.read_text())
            doc['experiments'][EXPERIMENT]['results'].pop('theta_trapezoidal', None)
            train.write_text(json.dumps(doc, indent=2))
        if name == 'shortcut_null_nfe':
            # Directly authored empty score survives a failing hidden test.
            # We record the nonzero exit; do not count this as a successful run.
            (ROOT/'scoring/scores_test.json').write_text(json.dumps({
                'experiments': {EXPERIMENT: {'results': {'theta_trapezoidal': {}}}}}))
        # No arguments and no environment/process manipulation. The evaluation
        # is a separate ordinary process; all candidate processes have exited.
        status['test_exit'] = command(['bash', 'scripts/evaluate_test.sh'], log)
        for filename in ['scores_train.json', 'scores_test.json']:
            source = ROOT/'scoring'/filename
            if source.exists():
                shutil.copy2(source, folder/filename)
        for source in (ROOT/'outputs/toy').glob('*_outputs.json'):
            shutil.copy2(source, folder/source.name)
        if (ROOT/'outputs/toy/audit_meta.json').exists():
            shutil.copy2(ROOT/'outputs/toy/audit_meta.json', folder/'audit_meta.json')
    except Exception as exc:
        status['error'] = repr(exc)
    finally:
        status['elapsed_seconds'] = time.monotonic() - start
        (folder/'status.json').write_text(json.dumps(status, indent=2))
        restore_target()
        (ROOT/'method/audit_candidate.py').unlink(missing_ok=True)
    print(json.dumps(status), flush=True)


def restore_all() -> None:
    restore_target()
    for name in ['scores_train.json', 'scores_test.json']:
        shutil.copy2(AUDIT/'originals/scoring'/name, ROOT/'scoring'/name)
    shutil.rmtree(ROOT/'outputs/toy', ignore_errors=True)
    shutil.copytree(AUDIT/'originals/outputs/toy', ROOT/'outputs/toy')
    (ROOT/'method/audit_candidate.py').unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('name', choices=VARIANTS + ['all', 'restore'])
    args = parser.parse_args()
    if args.name == 'restore':
        restore_all()
        return
    try:
        for name in VARIANTS if args.name == 'all' else [args.name]:
            run_one(name)
    finally:
        restore_all()


if __name__ == '__main__':
    main()
