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

## Remaining

- Run the submitted CPU job to generate 1,000,000 samples per method/step count.
- Inspect `scoring/scores.json` and compare against the reference convergence
  expectations.
- If the measured slopes are too noisy, increase sample count or adjust the
  evaluated step grid while keeping the same paper procedure.

## Issues

- Exact KL values from Figure 2 are not tabulated in the markdown, so
  `reference.json` targets the paper's stated first-order and second-order
  convergence rates rather than copied curve coordinates.

## Deviations from Paper

- The current reproduction focuses on the paper's own analytic 15-state
  experiment instead of the large text/image/LLM experiments. This is a scale
  and resource choice, not an algorithm replacement: the implemented samplers
  are the paper's Poisson-update algorithms.
