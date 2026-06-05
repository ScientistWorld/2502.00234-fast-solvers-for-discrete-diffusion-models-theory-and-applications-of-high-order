# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-05 03:21] - method_runs
- Implemented tau-leaping, theta-RK-2, and theta-Trapezoidal Poisson jump solvers for the finite-state reverse CTMC.
- Ran the end-to-end analytic 15-state experiment and wrote sampler artifacts under `outputs/toy/`.

### [2026-06-05 03:22] - core_claim
- Evaluated saved artifacts with method-independent code in `eval/`.
- Reproduced the paper's central high-order claim on the analytic toy benchmark: theta-Trapezoidal fitted convergence order is 2.188 versus 1.306 for tau-leaping, with lower KL at the same 64-step budget.
