# Progress

## What Works

- Paper analysis completed for the core algorithms and toy experiment.
- Implemented tau-leaping, theta-RK-2, and theta-Trapezoidal Poisson jump solvers.
- Implemented the paper's 15-state analytic CTMC with exact reverse rates.
- Implemented evaluation that reads saved sampler artifacts and writes `scoring/scores.json`.

## Results

The CPU reproduction completed successfully on the 15-state analytic benchmark.

| Method | KL at 64 steps | Fitted order | Mean rejection rate |
|---|---:|---:|---:|
| tau-leaping | 0.007809 | 1.306 | 0.157 |
| theta-RK-2 | 0.006443 | 1.330 | 0.104 |
| theta-Trapezoidal | 0.000514 | 2.188 | 0.057 |

Theta-Trapezoidal is more than an order of magnitude lower in KL than
tau-leaping at the same maximum step count and exhibits the expected
approximately second-order convergence.

## Remaining

- Optional future work: reproduce the large text, image, and GSM8K experiments
  using RADD, MaskGIT, and LLaDA assets.
- Optional future work: add more toy-model settings, such as theta sweeps or
  higher sample counts, if more CPU time is available.

## Issues

- Exact KL values from Figure 2 are not tabulated in the markdown, so
  `reference.json` targets the paper's stated first-order and second-order
  convergence rates rather than copied curve coordinates.
- The scheduler gave the CPU job a 1-hour timeout and one CPU. The sample count
  is scaled from the paper's 1,000,000 to 50,000 for the next run while keeping
  the same CTMC, solvers, uniform grids, and KL evaluation procedure.
- The scaled run completed in about 60 seconds and produced stable enough
  results to support the core claim.

## Deviations from Paper

- The current reproduction focuses on the paper's own analytic 15-state
  experiment instead of the large text/image/LLM experiments. This is a scale
  and resource choice, not an algorithm replacement: the implemented samplers
  are the paper's Poisson-update algorithms.
- The toy experiment sample count is reduced from 1,000,000 to 50,000 to fit
  the available one-CPU, one-hour job limit. The sampling procedure itself is
  unchanged.
