# Progress

## What Works

- Paper analysis completed for the core algorithms and toy experiment.
- Implemented tau-leaping, theta-RK-2, and theta-Trapezoidal Poisson jump solvers.
- Implemented the paper's 15-state analytic CTMC with exact reverse rates.
- Implemented evaluation that reads saved sampler artifacts and writes `scoring/scores.json`.
- The scored core experiment compares tau-leaping, theta-RK-2, and
  theta-Trapezoidal on saved artifacts.
- The tau-leaping baseline is now implemented in `baseline/` rather than
  importing from `method/`, so future method edits do not alter the baseline.

## Results

The CPU reproduction completed successfully on the 15-state analytic benchmark.

| Method | KL at common NFE 64 | KL at 64 steps | Fitted order | Max steps | Max NFE |
|---|---:|---:|---:|---:|---:|
| tau-leaping | 0.007809 | 0.007809 | 1.306 | 64 | 64 |
| theta-RK-2 | 0.017753 | 0.006443 | 1.330 | 64 | 128 |
| theta-Trapezoidal | 0.003075 | 0.000514 | 2.188 | 64 | 128 |

Theta-Trapezoidal has lower KL than tau-leaping at the matched 64-NFE budget
and is more than an order of magnitude lower in KL at the same maximum step
count.  It exhibits the expected approximately second-order convergence.
Theta-RK-2 is included as a faithful implementation, but it does not beat
tau-leaping at matched NFE in this reduced-sample run.

## Remaining

- Optional future work: reproduce the large text, image, and GSM8K experiments
  using RADD, MaskGIT, and LLaDA assets.
- Optional future work: add more toy-model settings, such as theta sweeps or
  higher sample counts, if more CPU time is available.
- The packaged environment is complete for `core_claim`.

## Issues

- Exact KL values from Figure 2 are not tabulated in the markdown, so
  `reference.json` records the high-order solvers' stated/theoretical
  second-order behavior as the numeric paper target. The tau-leaping baseline
  remains in scores for comparison, but its theoretical first-order rate is not
  treated as an exact paper-reported measurement.
- The scheduler gave the CPU job a 1-hour timeout and one CPU. The sample count
  is scaled from the paper's 1,000,000 to 50,000 for the next run while keeping
  the same CTMC, solvers, uniform grids, and KL evaluation procedure.
- The scaled run completed in about 60 seconds and produced stable enough
  results to support the core claim.
- A continuation audit found theta-RK-2 was generated but not scored. This is
  fixed; the expanded CPU validation rerun completed and wrote all three
  methods to `scoring/scores.json`.
- A later audit found the baseline imported tau-leaping from `method/`; this
  is fixed. CPU validation job `13ceb5df-0d1` completed successfully, refreshed
  `scoring/scores.json` with `max_nfe`, and passed `python validate.py --compare`.
- The main evaluator now reports the same distribution-quality, state-slice,
  and compute-constraint metrics declared in `reference.json`.
  Final verification also removed method-name leakage from method-agnostic
  public docs and changed the main evaluator to score any method key present
  in the saved artifact format.
- A final audit found the headline metrics compared methods at equal maximum
  step count but not equal NFE.  The evaluator and target schema now include
  `kl_at_common_nfe`, `total_variation_at_common_nfe`, and `common_nfe`; the
  current saved artifacts support the core claim for theta-Trapezoidal at
  common NFE 64.
- Final verification removed method-specific, self-reported diagnostics from
  the scored metric schema.  The remaining constraint metrics are derived from
  the artifact distributions and machine-independent compute budgets.

## Deviations from Paper

- The current reproduction focuses on the paper's own analytic 15-state
  experiment instead of the large text/image/LLM experiments. This is a scale
  and resource choice, not an algorithm replacement: the implemented samplers
  are the paper's Poisson-update algorithms.
- The toy experiment sample count is reduced from 1,000,000 to 50,000 to fit
  the available one-CPU, one-hour job limit. The sampling procedure itself is
  unchanged.
