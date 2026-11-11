# Hardening report (iteration 11)

## Summary

The gym was rebuilt. The earlier test scored JSON artifacts the method wrote itself: 15-state distributions labelled with step counts and NFE. The target was public, so the method could write the answer down, misreport its compute, enumerate the 15 states, or omit itself. In the redteam's audit, eight shortcuts paid or tied.

The hidden test now **runs every registered solver itself**. It does this in a separate process, on held-out data distributions that exist only inside `eval/test/` at test time. The evaluator answers each intensity request exactly, **counts NFE itself** and refuses any request over the budget. It **scores only the returned samples**. Any failure gets the worst value of every metric.

The problem is the paper's analytic toy model, scaled up:

| | Paper toy | This gym |
|---|---|---|
| States per site | 15 | 15 |
| Sites | 1 | 48 |
| Forward noise per site | `Q = E/15 - I` | `Q = E/15 - I` |
| Horizon | `T = 12` | `T = 12` |
| Time grid | arithmetic | arithmetic |
| θ | 1/2 | 1/2 |
| Scores | exact | exact |
| Sample size | 10⁶ | 10,000 |
| Comparison regime | step counts | matched NFE, with NFE below the number of sites |

The matched-NFE, NFE-below-sequence-length regime is the one of the paper's Sec. 6.2–6.4. In it, enumeration, exact simulation and site-by-site Markov chain sampling are not viable.

The paper's θ-Trapezoidal sampler is the reference, and it is now ranked against honest alternatives:
- it beats τ-leaping and θ-RK-2 at every matched budget;
- better samplers of the same family beat it by about 400 times the reference's seed spread;
- every shortcut tried scores below it.

## Did better results score higher?

The rank is ΔR = Σ |coef| · direction · (v − v_ref) / sd, over the metrics of `scoring/reference.json`.

- "Honest population" means the reference seeds, the baselines and the variants.
- "All" adds the shortcuts.
- Source: `hardening/audit/ranks.md` and `ranks.json`, from job `463cd21e-60a`. All numbers are on the hidden test.

| Attempt | What it is | ΔR (honest) | ΔR (all) |
|---|---|---:|---:|
| `better_trap_cubic_grid` | paper's θ-Trapezoidal, forward times t_n = T(1−n/N)³ | **+9.23** | +6.29 |
| `better_trap_quadratic_grid` | same, t_n = T(1−n/N)² | **+8.19** | +5.42 |
| reference, seed offsets 1 and 2 | θ-Trapezoidal, other initial states and RNG seeds | +0.018, −0.003 | +0.010, +0.001 |
| `weaker_trap_destination_indexed` | the earlier gym's θ-Trapezoidal (see below) | −0.043 | −0.029 |
| `tau_leaping` (baseline; seed offset 1) | first order | **−0.95** (−0.94) | −0.58 |
| `theta_rk2` | paper's θ-RK-2 | **−3.63** | −2.23 |

**Seed noise.** Across the three reference seeds, ΔR spans **0.020**. The per-metric spread is 0.002–0.007 in log10 KL and 0.010 in order. Both better variants beat the reference by more than 400 times that noise. τ-leaping is about 45 times below it.

**Before** (REDTEAM, old gym):
- **The improvement did not resolve at the default sample size.** The genuine improvement (exponential midpoint, with matrix exponentials of the full 15-state generator) gained +0.19 in the audit population. In the "genuine" population its three seeds gave −0.67, +0.89 and −1.06, inside the reference's seed spread of 1.44. It only resolved at 20 times the samples, with +3.65.
- **The ablation was penalised.** Removing the second stage gave −1.42 / −12.47.

**Trade-off with the old gym's audit improvement.** That exponential-midpoint solver needs the full S^d generator. At 15^48 states it no longer exists, by design.

**Train and test agree** for honest methods; the train slice runs about 0.05 log10 harder.

| Method | Slice | log10 pair KL at 16 / 128 NFE | log10 profile KL | Order |
|---|---|---|---:|---:|
| θ-Trapezoidal | train | 0.110 / −1.087 | 0.647 | 1.397 |
| θ-Trapezoidal | test | 0.061 / −1.145 | 0.621 | 1.419 |
| cubic-grid variant | train | −0.482 / −1.910 | −0.778 | 2.653 |
| cubic-grid variant | test | −0.516 / −1.903 | −0.815 | 2.596 |

## Shortcuts: fix and score before → after

"Before" is the redteam's ΔR in its audit population. "After" is ΔR on the new hidden test, honest population and then all attempts. A failed run receives log10(20) = 1.301 on every KL metric and −6 on the order.

| Old shortcut | Fix | Before | After |
|---|---|---:|---:|
| Visible answers (public target written out) | Held-out targets are generated only inside `eval/test/` at test time. The audit sampled a public train instance exactly as the "answer" (`shortcut_public_train_target`). | **+0.360** | **−8.48 / −5.01** |
| Budget metadata (NFE −10¹², zero counts) | NFE is counted by the evaluator, `ceil(rows/n)` per request; over-budget requests are refused and fail the run. The audit spent 2× the budget (`shortcut_overspend`). | **+0.939** | **−18.69** (failed) |
| Relabel grid (step labels 1–4) | Budgets 16/32/64/128 are fixed by the evaluator. The method reports nothing, and the order is computed from evaluator budgets. | +0.000006 | structurally impossible |
| Method selector / empty method (omitted → skipped) | Every `SOLVERS` registry under `method/` and `baseline/` is discovered and scored. Empty or unimportable registries and failing solvers get worst values. | tie (0) | worst values; see below |
| Replace target (edited `data/` asset) | `eval/test/` is self-contained (numpy and stdlib only). Nothing in `data/`, `method/` or `baseline/` reaches ground truth. | **+0.361** | structurally impossible |
| Move baseline (overwrite baseline artifact) | The baseline is run from `baseline/` code by the evaluator. No metric depends on another method's output. | tie (0) | structurally impossible |
| Brute force (enumerate the 15-state kernel from the public target) | 15⁴⁸ states, hidden target. Enumerating via queries would need ~4×10⁵³ query rows (15⁴⁸ ≈ 3×10⁵⁶ states, at most 673 revealed per row) against a budget of 128 × 10⁴. | **+0.237** | not applicable |
| Exact simulation, budget-capped (`alt_gillespie_capped`) | Same scale-up as brute force. | n/a | −6.00 / −3.60 |
| Single-site Gibbs on data-end conditionals (`alt_gibbs_t0`) | Same scale-up. | n/a | −6.30 / −3.77 |
| Parallel (one-shot) Gibbs (`alt_parallel_gibbs_t0`) | Same scale-up. | n/a | −6.45 / −3.85 |
| Forged scores / stale score file | Scores come only from returned samples. The score file is deleted at start and written atomically. The discovery demo pre-wrote a fabricated `scores_test.json`, and it was replaced. | −7.53 (rejected) | replaced |
| Null NFE / empty distribution (evaluator crashed, stale file survived) | The harness catches every per-method failure and assigns worst values. Score files never survive from earlier runs. The audit returned no samples (`shortcut_empty_output`). | invalid (exit 1) | **−18.69** (failed) |
| Negative distribution | Outputs are states, not distributions, and are validated for shape and range. The audit returned states outside the range (`shortcut_out_of_range`). | accepted (−7.84) | **−18.69** (failed) |
| Constant input (return the uniform start, 0 NFE) | — | −7.53 | −6.00 / −3.60 |
| Huge-slope labels | Same fix as relabel grid. | −3.00 | structurally impossible |
| *New:* intensities outside [0, T] | Requests outside [0, T] are refused and fail the run (`shortcut_negative_time`). | — | −18.69 (failed) |
| *New:* in-process introspection for instance data | Solvers run in a separate process, launched with `python -I`. The only channel is JSON headers plus raw arrays; nothing from the solver is unpickled. The audit search through `gc` found nothing (`shortcut_introspect`). | — | −6.00 / −3.60 |

The "after" column of the last row is the score of its fallback: returning the initial states.

**Discovery demo** (`hardening/discovery_demo.sh`, results in `hardening/audit/discovery_demo/`). It runs `bash scripts/evaluate_test.sh` with no arguments on a scratch copy of the gym. That copy's `method/` holds four failing registries:
- one solver that crashes;
- one that returns no samples;
- an empty `SOLVERS = {}`;
- a module whose import fails.

All four appear in the regenerated `scores_test.json`, keyed by name or file, with the worst value of every metric. The fabricated pre-existing scores did not survive.

## Shortcuts left open, and why

- **Identifying the target from intensity queries.** Ratios returned by `rates()`, for example along paths of single-site changes at the data end, determine mode weights and factors for anyone who knows the instance family. Such a solver could fit the mixture and sample it directly.
  - Why open: the public train generator documents the family, and an exact-score oracle cannot hide it. Closing it would need a learned, inexact score, which the paper's toy deliberately excludes.
  - Mitigations: held-out instances; eight modes of different sharpness; 48 sites; a budget of at most 128 calls. It is listed as out of scope in `scoring/DIRECTION.md`, so the grader must catch it. I did not run this attack.
- **Reading `eval/test/` at run time.** During the hidden test, the held-out generator is on disk in the same container, and a solver process could import it. That would be a blatant cheat, and per the instructions I did not build defences against the environment.
- **Partial saturation at 128 NFE for the best variant.** The cubic-grid variant's pair KL at 128 NFE (−1.90 log10) and site KL are near the 10,000-sample floor (−1.95 for pairs). Saturation is limited to those metrics at the largest budget; the profile metric (floor −2.4) and the lower budgets still separate methods.
- **`convergence_order` on near-floor methods.** It is fitted after subtracting the analytic floor, but a method close to the floor at the top budgets reads a lower order. Its coefficient is 0.5 against 4 for the KL metrics, so its net effect on genuine improvements is small, as the cubic-grid result shows.

## Reference score before → after

**Before:** the old test slice, 50k samples, odd-ranked budgets 16 and 64, one public target.

| Metric | Value |
|---|---:|
| KL at max steps | 0.000514 |
| Fitted order | 2.36 |
| Mean KL | 0.00704 |
| TV at max steps | 0.00955 |
| max NFE | 128 |
| Matched-NFE metrics | never written by the hidden test |

**After:** θ-Trapezoidal on the hidden test (`scoring/scores_test.json`).

| Metric | Value |
|---|---:|
| `log10_pair_kl_nfe16` | 0.061 |
| `log10_pair_kl_nfe32` | −0.303 |
| `log10_pair_kl_nfe64` | −0.726 |
| `log10_pair_kl_nfe128` | −1.145 |
| `log10_profile_kl_mean` | 0.621 |
| `log10_site_kl_mean` | −1.112 |
| `convergence_order` | 1.419 |

Baselines on the hidden test, in the same metric order:

| Method | pair 16 | pair 32 | pair 64 | pair 128 | profile | site | order |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tau_leaping` | 0.109 | −0.191 | −0.587 | −0.988 | 0.664 | −0.943 | 1.275 |
| `theta_rk2` | 0.239 | 0.126 | −0.141 | −0.499 | 0.778 | −0.572 | 0.838 |

The hidden test runs all three registered methods in about 21 minutes on one CPU.

## Reproduction of the paper's claim

- **Multi-site gym, hidden test, matched NFE.** θ-Trapezoidal has lower pair, profile and site KL than τ-leaping and θ-RK-2 at every budget. Its order is 1.42, against 1.28 for τ-leaping and 0.84 for θ-RK-2: the paper's ordering, Trapezoidal ahead with RK-2 entering its asymptotic regime late.
- **One-site paper toy, unscored** (`hardening/paper_toy.py`, `hardening/audit/paper_toy.json`). Target uniform on the simplex, T = 12, θ = 1/2, 2×10⁵ samples (paper: 10⁶), steps 16–256 (paper: 16–1024). Slopes against steps:

| Method | Slope 16–256 | Slope, last four points | Paper |
|---|---:|---:|---:|
| θ-Trapezoidal | 1.77 | 1.84 | 1.937 |
| θ-RK-2 | 1.38 | 1.49 | 1.543 (last four: 1.819) |
| τ-leaping | 1.43 | — | not reported |

  KL at 256 steps: τ-leaping 0.0038, θ-Trapezoidal 0.00047, θ-RK-2 0.0045. Trapezoidal is lowest at every step count.

- **Restored to the paper's version.**
  - *Jump indexing.* The two stages are now combined per jump increment ν, as in Algorithms 1–2 and the authors' released sampler (`radd/sampling.py`, offset gather). The earlier gym combined them per destination state. On the hidden test, the old destination-indexed version scores −0.043 relative to the restored one; that is about twice the seed spread, so the gap is small but the restored version is not worse.
  - *Paper toy settings.* The toy uses the paper's Dirichlet(1) target, which the earlier gym had replaced with a handcrafted target, and a step range closer to the paper's.

## Headroom check

**Bounds:**
- **The reference sits at no bound.** Its pair KL at 128 NFE is 0.072 against the 0.011 floor, and its profile KL averages about 4.2 nats against a 0.004 floor.
- **One simple variant nears the bound on part of the metrics only.** A better time grid (cubic, +9.23) brings pair and site KL at 128 NFE to the floor. Its 16–64 NFE pair KL and its profile KL (−0.81 log10, about 0.15 nats) remain far from the bound, so headroom remains.

**What was added for headroom:**
- the 48-site scale;
- the mode-aware profile metric;
- budgets below the site count.

The design was not accepted on the first try:
- **Round 1** (`hardening/round1_dense_targets/`): 10 sites with chain-mixture targets drawn uniformly from the simplex. There, simple single-site Gibbs on the data-end conditionals ranked first (+9.45), above the reference.
- **Calibrations** (`hardening/calibration/`): with sparse chain targets and then sparse product targets, one-shot parallel Gibbs still beat the uniform-grid reference at low NFE.
- **Final family:** modes ordered so that the heavier modes are sharper. From noise, MCMC lands in the broad, light modes. The reverse process allocates mass by weight, and the variants rank as above.
- **Paper's real experiments:** not added. RADD text, MaskGIT images and LLaDA on GSM8K need GPU models and external evaluators, beyond this iteration's budget.

**Brute force fails.**
- **Enumeration does not apply.** The space has 15⁴⁸ states.
- **Exact simulation fails within the budget.** `alt_gillespie_capped` scores −6.00.
- **MCMC on the exact data conditionals fails.** `alt_gibbs_t0` scores −6.30 and `alt_parallel_gibbs_t0` −6.45.

**Baselines are sensible.** τ-leaping is clearly below the reference (−0.95) and θ-RK-2 further (−3.63), and the better variants show the scale leaves room above it.

## Weights

There is one experiment, with weight 1.0. The metric coefficients are:

| Metric | Coefficient |
|---|---:|
| Pair KL at each budget | 0.75 each (3.0 total) |
| Profile KL | 1.0 |
| Site KL | 0.5 |
| Order | 0.5 |

**Why.** The paper's headline across all of its experiments is sample quality at matched NFE, which pair KL at each budget measures. The profile carries the joint, mode-proportion part of the KL that pairs miss. Site KL is the paper's one-site quantity. Order is the toy experiment's summary statistic; it is primary for continuity, but weighted low because it is a derived slope.

## Things not fixed

- **θ-RK-2 is now a comparison baseline in `reference.json`.** The earlier iterations' results entry gave θ-RK-2 `convergence_order: 2.0`. The paper does not report that value: its Fig. 2 slope for RK-2 is 1.543, and the gym's multi-site pre-asymptotic order (0.84) is on neither scale. The instructions say off-scale paper numbers go under `_` keys and that the setup must not be changed to match them. So that entry is kept unchanged under `_paper_reported`, next to the paper's slopes, and θ-RK-2, the paper's warm-up scheme, is scored as a comparison baseline. θ-Trapezoidal keeps the paper's second-order claim (2.0) against a measured 1.42. `python validate.py` passes in default, `--post-design`, `--post-reproducer` and `--compare` modes.
- **The soft "eval looks detached" warning remains.** The evaluators deliberately run no-training solvers, so validate's soft heuristic flags them.
- **Remaining departures from the paper's toy.**
  - Sample size: 10,000 per (instance, budget) rather than 10⁶, because of a 1-CPU budget.
  - Confidence intervals: four held-out instances with paired seeds stand in for bootstrap confidence intervals.
  - The multi-site target family is my construction; the paper's toy has one site.

## Files and reruns

| Path | Contents |
|---|---|
| `eval/train/` | Public slice. |
| `eval/test/` | Hidden slice: same code, held-out seeds, no imports between the two. |
| `hardening/sync_test_eval.py` | Regenerates `eval/test/` from `eval/train/`. |
| `hardening/candidates.py` | Every audit attempt above. |
| `hardening/run_audit.py` | Runs the attempts through either evaluator; resumable. |
| `hardening/analyze.py` | The rank calculation. |
| `hardening/oracle_check.py` | Brute-force check of the exact intensities, marginals, profile and sampler on a tiny instance. Maximum relative error is 2e-15 on both slices. |
| `hardening/discovery_demo.sh` | The no-argument hidden-test demo. |
| `hardening/paper_toy.py` | The one-site Fig. 2 run. |

`scripts/run.sh` reruns all of it and reproduces the reference end to end; no training is involved.

The jobs, all CPU:

| Job | What it was | Outcome |
|---|---|---|
| `f9c42a28-65f` | Calibration | Cancelled after the timing and failure-path checks. |
| `ef4d43fc-71c` | Round 1 | Its paper toy was cancelled. |
| `398646ec-b8e` | Sparse-chain calibration | Completed. |
| `5d1e0b08-1d2` | Product-family calibration | Completed. |
| `463cd21e-60a` | Final run | Completed. |
