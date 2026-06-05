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

## Stop Justification
- Completed at milestone `core_claim`. The environment implements and scores
  the paper's core high-order solver on the paper's analytic toy benchmark.
  Larger text, image, and LLM experiments remain future extensions because they
  require large pretrained checkpoints and benchmark assets beyond the compact
  CPU reproduction packaged here.
