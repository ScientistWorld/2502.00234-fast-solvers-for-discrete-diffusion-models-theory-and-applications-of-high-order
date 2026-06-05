# Progress

## What Works

- Paper analysis completed for the core algorithms and toy experiment.
- Implemented tau-leaping, theta-RK-2, and theta-Trapezoidal Poisson jump solvers.
- Implemented the paper's 15-state analytic CTMC with exact reverse rates.
- Implemented evaluation that reads saved sampler artifacts and writes `scoring/scores.json`.

## Results

No full reproduction scores yet. A local tiny smoke run could not execute because
the login-node Python environment does not currently have `numpy`; the
  reproducible container setup installs it through `environment/setup.sh`.
- The first submitted job failed before running the samplers because the
  container did not have `/home/user` on Python's import path for script-path
  execution. This has been fixed with package markers and `PYTHONPATH`.

## Remaining

- Run the submitted CPU job to generate 50,000 samples per method/step count.
- Inspect `scoring/scores.json` and compare against the reference convergence
  expectations.
- If the measured slopes are too noisy, increase sample count or adjust the
  evaluated step grid while keeping the same paper procedure.

## Issues

- Exact KL values from Figure 2 are not tabulated in the markdown, so
  `reference.json` targets the paper's stated first-order and second-order
  convergence rates rather than copied curve coordinates.
- The scheduler gave the CPU job a 1-hour timeout and one CPU. The sample count
  is scaled from the paper's 1,000,000 to 50,000 for the next run while keeping
  the same CTMC, solvers, uniform grids, and KL evaluation procedure.

## Deviations from Paper

- The current reproduction focuses on the paper's own analytic 15-state
  experiment instead of the large text/image/LLM experiments. This is a scale
  and resource choice, not an algorithm replacement: the implemented samplers
  are the paper's Poisson-update algorithms.
- The toy experiment sample count is reduced from 1,000,000 to 50,000 to fit
  the available one-CPU, one-hour job limit. The sampling procedure itself is
  unchanged.
