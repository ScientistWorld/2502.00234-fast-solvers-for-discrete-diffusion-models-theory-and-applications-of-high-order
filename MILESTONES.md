# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-05 03:21] - method_runs
- Implemented tau-leaping, theta-RK-2, and theta-Trapezoidal Poisson jump solvers for the finite-state reverse CTMC.
- Ran the end-to-end analytic 15-state experiment and wrote sampler artifacts under `outputs/toy/`.

### [2026-06-05 03:22] - core_claim
- Evaluated saved artifacts with method-independent code in `eval/`.
- Reproduced the paper's central high-order claim on the analytic toy benchmark: theta-Trapezoidal fitted convergence order is 2.188 versus 1.306 for tau-leaping, with lower KL at the same 64-step budget.

### [2026-06-05 03:30] - method_runs
- Continuation audit reset the milestone to `method_runs`.
- Found the implementation is real, but the evaluator omitted theta-RK-2 despite generating its artifacts; updated scoring to include it and submitted a CPU validation rerun.

### [2026-06-05 03:27] - core_claim
- CPU validation rerun completed the analytic 15-state experiment and wrote tau-leaping, theta-RK-2, and theta-Trapezoidal scores.
- Fixed `reference.json` schema for theta-RK-2 and restored `core_claim`; theta-Trapezoidal remains second-order and lower-KL than both compared samplers under the same maximum step count.

### [2026-06-05 03:34] - core_claim
- Separated the tau-leaping baseline from `method/` and added `max_nfe` as an explicit compute constraint.
- Submitting another CPU validation job to refresh `scoring/scores.json` with the new metric.

### [2026-06-05 03:37] - core_claim
- CPU validation job `13ceb5df-0d1` completed successfully after baseline separation and `max_nfe` scoring changes.
- `scoring/scores.json` now reports the NFE constraint; `python validate.py --compare` passes, and theta-Trapezoidal remains second-order with much lower KL than tau-leaping.

### [2026-06-05 03:58] - core_claim
- Audited the main evaluator and found it lagged the train/test evaluators by omitting distribution-quality and compute-constraint metrics declared in `reference.json`.
- Updated `eval/evaluate_toy.py` and `scripts/run.sh`; submitting a CPU test job to regenerate artifacts and refresh all score files inside the packaged container.

### [2026-06-05 04:02] - core_claim
- Final verification kept the milestone at `core_claim`.
- Removed method-name leakage from method-agnostic docs and made the main evaluator score any method key in the artifact format.
- Regenerated `scoring/scores.json` from saved artifacts and confirmed `python validate.py --compare` passes.

### [2026-06-05 04:08] - core_claim
- Added explicit matched-NFE scoring after an audit found the headline comparison only matched step count.
- Existing saved artifacts support the core theta-Trapezoidal claim at common NFE 64: KL 0.003075 versus 0.007809 for tau-leaping.
- Submitting a CPU validation job to run the updated evaluator inside the packaged container.
