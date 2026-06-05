# Reproduction Milestones

**Current: none**

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

## Stop Justification
- Not stopped. A CPU job is being submitted to run the full end-to-end reproduction.
