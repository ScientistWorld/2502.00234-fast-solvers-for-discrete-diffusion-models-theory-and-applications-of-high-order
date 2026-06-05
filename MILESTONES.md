# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-05 00:00] - none
- Read the paper and identified the core high-order discrete diffusion solvers.
- Implemented the analytic 15-state toy-model gym, tau-leaping baseline, theta-RK-2, theta-Trapezoidal, artifact-based evaluation, and scoring contract.
- Local smoke execution is blocked by missing login-node `numpy`; dependency is declared in `environment/setup.sh` for the submitted container job.

### [2026-06-05 00:00] - none
- First compute job failed before sampler execution due to missing `/home/user`
  on Python import path.
- Added package markers, set `PYTHONPATH` in scripts, and scaled the next CPU
  run to 50,000 samples for the observed 1-hour CPU timeout.

### [2026-06-05 00:00] - core_claim
- The end-to-end CPU reproduction completed successfully.
- On the paper's analytic 15-state toy experiment, theta-Trapezoidal reached
  KL 0.000514 at 64 steps versus tau-leaping KL 0.007809.
- The fitted convergence order was 2.188 for theta-Trapezoidal versus 1.306
  for tau-leaping, supporting the paper's core second-order solver claim.
- Validation passed with `python validate.py --compare`.

### [2026-06-05 03:20] - core_claim
- Corrected `reference.json` so theoretical tau-leaping order is not treated as
  an exact paper-reported primary-metric value.
- Kept tau-leaping in reproduced scores as the comparison baseline and scored
  theta-Trapezoidal as the proposed method for the core claim.
- Re-ran `python validate.py --compare`; all checks passed.

## Stop Justification
- Completed at milestone `core_claim`.
