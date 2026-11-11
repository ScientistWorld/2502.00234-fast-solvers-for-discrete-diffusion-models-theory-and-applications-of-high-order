#!/usr/bin/env python3
"""Audit-only rank calculation: directions, absolute coefficients, cohort SD.

The future ranking cohort is unknowable. We report the exact raw vectors and
an explicit, reproducible audit cohort rather than silently invent its scales.
Missing values are skipped, following the ranking rule in the audit request.
Nonzero evaluation exits are reported separately, never called valid successes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parent
EXP = 'toy_15_state_convergence'
METHOD = 'theta_trapezoidal'


def read(path):
    return json.loads(path.read_text())


def results(doc):
    return doc.get('experiments', {}).get(EXP, {}).get('results', {})


def vector(folder):
    path = folder/'scores_test.json'
    return results(read(path)).get(METHOD, {}) if path.exists() else {}


def main():
    spec = read(BASE/'reference_scoring.json')['experiments'][EXP]
    metrics = spec['metrics']
    ref = results(read(BASE/'original_reference_test.json'))[METHOD]
    folders = sorted(p.parent for p in BASE.glob('*/status.json'))
    vectors = {p.name: vector(p) for p in folders}
    statuses = {p.name: read(p/'status.json') for p in folders}
    # This fixed cohort includes every successfully executed attempt, including
    # absent method results. Failed jobs are excluded from estimated SD.
    cohort = [name for name in vectors if statuses[name].get('test_exit') == 0]
    def scales_for(names):
        scales = {}
        for metric in metrics:
            values = [float(ref[metric])] if metric in ref else []
            values.extend(float(vectors[name][metric]) for name in names
                          if isinstance(vectors[name].get(metric), (int, float))
                          and math.isfinite(vectors[name][metric]))
            sd = statistics.pstdev(values) if values else 0
            scales[metric] = sd if sd > 0 else None
        return scales

    scale = scales_for(cohort)

    def rank(v, scales):
        contribution = {}
        for metric, cfg in metrics.items():
            if metric not in v or metric not in ref or not scales.get(metric):
                continue
            if not isinstance(v[metric], (float, int)) or not math.isfinite(v[metric]):
                continue
            sign = 1 if cfg['higher_is_better'] else -1
            contribution[metric] = (spec.get('weight', 1)*abs(cfg.get('coefficient', 1))*
                                    sign*(v[metric]-ref[metric])/scales[metric])
        return sum(contribution.values()), contribution

    # A separate quality-only cohort avoids shortcut extremes disguising the
    # sensitivity check. It uses exactly the same prescribed rank formula.
    quality = [name for name in cohort if name.startswith(('reference_', 'better_', 'weaker_'))]
    quality_scale = scales_for(quality)

    summary = {'reference_method': METHOD, 'reference_metrics': ref,
               'rank_formula': 'sum(weight * abs(coefficient) * direction * (attempt-reference) / population_sd)',
               'cohort': ['original_reference']+cohort, 'population_sd': scale,
               'quality_cohort': ['original_reference']+quality,
               'quality_population_sd': quality_scale,
               'missing_reference_test_metrics': sorted(set(metrics)-set(ref)),
               'reference_metrics_at_mathematical_optimum': [], 'attempts': {}}
    for name, v in vectors.items():
        score, contributions = rank(v, scale)
        # Every ranked shortcut must be included in its own normalization
        # cohort as well; extrapolating a fixed quality-only SD would exaggerate
        # unbounded metadata values and violate the requested ranking rule.
        sensitivity_cohort = quality if name in quality else quality + [name]
        sensitivity_scale = scales_for(sensitivity_cohort)
        qscore, qcontrib = rank(v, sensitivity_scale)
        entry = {'test_exit': statuses[name].get('test_exit'),
                 'train_exit': statuses[name].get('train_exit'),
                 'score_delta': score, 'quality_cohort_score_delta': qscore,
                 'metrics': v, 'contributions': contributions,
                 'quality_cohort_contributions': qcontrib,
                 'quality_sensitivity_cohort': ['original_reference']+sensitivity_cohort,
                 'quality_sensitivity_population_sd': sensitivity_scale,
                 'scored_methods': sorted(results(read(BASE/name/'scores_test.json')))
                    if (BASE/name/'scores_test.json').exists() else [],
                 'absent_scored_metrics': sorted(set(metrics)-set(v))}
        if entry['test_exit'] != 0:
            entry['qualification'] = 'failed test; numeric skipped-value result is not a valid success'
        elif not v:
            entry['qualification'] = 'successful evaluator but absent candidate metrics; skipped losses tie reference'
        elif name.startswith('shortcut_') and abs(score) < 1e-10:
            entry['qualification'] = 'shortcut tie'
        summary['attempts'][name] = entry
        (BASE/name/'rank.json').write_text(json.dumps(entry, indent=2))
    seed_names = [n for n in quality if n.startswith('reference_seed_')]
    seed_spread = {}
    for metric in ref:
        values = [vectors[n][metric] for n in seed_names if metric in vectors[n]]
        if values:
            seed_spread[metric] = {'min': min(values), 'max': max(values),
                                   'range': max(values)-min(values)}
    seed_scores = [summary['attempts'][n]['score_delta'] for n in seed_names]
    qscores = [summary['attempts'][n]['quality_cohort_score_delta'] for n in seed_names]
    summary['seed_spread'] = seed_spread
    summary['seed_score_spread'] = {'min': min(seed_scores), 'max': max(seed_scores),
                                  'range': max(seed_scores)-min(seed_scores)}
    summary['quality_seed_score_spread'] = {'min': min(qscores), 'max': max(qscores),
                                          'range': max(qscores)-min(qscores)}
    genuine_midpoints = [n for n in quality if n.startswith('better_exponential_midpoint')
                         and not n.endswith('more_samples')]
    midpoint_scores = [summary['attempts'][n]['quality_cohort_score_delta'] for n in genuine_midpoints]
    summary['midpoint_quality_seed_score_spread'] = {
        'min': min(midpoint_scores), 'max': max(midpoint_scores),
        'range': max(midpoint_scores)-min(midpoint_scores)}
    # Same-budget comparison from observed sampler artifacts, without using a
    # second, non-ranking evaluator or calling the method again.
    paper_check = {}
    for n in seed_names:
        p = BASE/n
        methods = read(p/'method_outputs.json')['methods']
        base = read(p/'baseline_outputs.json')['methods']['tau_leaping']
        target = read(p/'method_outputs.json')['target']
        def kl(run):
            q = run['distribution']
            return sum(t*math.log(t/max(x, 1e-12)) for t, x in zip(target, q))
        paper_check[n] = {
            'tau_nfe64_kl': kl(next(r for r in base if r['nfe'] == 64)),
            'trapezoidal_nfe64_kl': kl(next(r for r in methods[METHOD] if r['nfe'] == 64)),
            'rk2_nfe64_kl': kl(next(r for r in methods['theta_rk2'] if r['nfe'] == 64)),
            'test_methods': results(read(p/'scores_test.json'))}
    summary['reproduction_check'] = paper_check
    (BASE/'summary.json').write_text(json.dumps(summary, indent=2))
    lines = ['| Attempt | test exit | rank delta (audit cohort) | rank delta (quality cohort) | order | KL@max | max NFE |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for n, entry in summary['attempts'].items():
        v = entry['metrics']
        def fmt(key):
            return f'{v[key]:.8g}' if key in v else 'missing'
        lines.append(f"| {n} | {entry['test_exit']} | {entry['score_delta']:.6f} | "
                     f"{entry['quality_cohort_score_delta']:.6f} | {fmt('convergence_order')} | "
                     f"{fmt('kl_at_max_steps')} | {fmt('max_nfe')} |")
    (BASE/'results.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines), flush=True)
    print('Seed rank spread:', summary['seed_score_spread'], flush=True)
    print('Quality-only seed rank spread:', summary['quality_seed_score_spread'], flush=True)


if __name__ == '__main__':
    main()
