# Evaluation

The evaluators run every registered solver themselves; nothing a solver writes
to disk is read.  For each instance and each NFE budget in `{16, 32, 64, 128}`
(fewer evaluations than the 48 sites), the solver is run once on 10,000
samples with initial states and random seed shared by all methods, so methods
are compared on paired inputs.  The evaluator answers every intensity request
exactly, counts evaluations, and scores the returned samples against the exact
data distribution, which only the evaluator holds.

The train slice uses four public instances; the hidden test uses four
held-out instances drawn from the same family, with the same budgets, sample
size and metrics.

## Quantities

For a sample set `q_hat` (with add-half smoothing of empirical counts) and the
exact data distribution `p0`, KL divergences `KL(p0 || q_hat)` are computed on
three evaluator-fixed projections:

- **site marginals**: the 15-state distribution of each site, averaged over
  sites;
- **site-pair marginals**: the 225-cell joint of each of the 1128 site pairs,
  averaged over pairs;
- **mode and likelihood profile**: the joint distribution of (most likely mode
  of the sample, decile of its exact data log-likelihood), 80 cells.  It sees
  how mass is split across modes and whether whole 48-site configurations are
  typical of the data.

## Scored metrics (experiment `multisite_reverse_ctmc`)

| metric | direction | meaning |
|---|---|---|
| `log10_pair_kl_nfe16/32/64/128` | lower | log10 pair-marginal KL at each budget (geometric mean over instances) |
| `log10_profile_kl_mean` | lower | log10 profile KL, averaged over instances and budgets |
| `log10_site_kl_mean` | lower | log10 site-marginal KL, averaged over instances and budgets |
| `convergence_order` | higher | negative log-log slope of pair KL against NFE after removing the finite-sample floor |

A failed method receives `log10(20) = 1.301` for every KL metric and `-6` for
the convergence order.  The finite-sample floor at 10,000 samples is about
`0.011` for pair KL and `0.004` for profile KL.

Scores: `scripts/evaluate_train.sh` writes `scoring/scores_train.json`
(`scripts/evaluate.sh` writes the same train evaluation to
`scoring/scores.json`); the hidden `scripts/evaluate_test.sh` takes no
arguments and writes `scoring/scores_test.json`.
