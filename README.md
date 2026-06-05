# Overview

This reproduction implements the core high-order inference algorithms from
"Fast Solvers for Discrete Diffusion Models" and evaluates them on the paper's
analytic 15-state toy model.  The experiment tests whether `theta`-Trapezoidal
and `theta`-RK-2 achieve faster KL convergence than first-order tau-leaping
under the same discrete diffusion reverse-process setting.

---

# Problem Definition

The task is to sample from a discrete diffusion model's reverse continuous-time
Markov chain on a finite categorical state space.  A solver receives an initial
sample from the forward process stationary distribution, a reverse-time grid,
and transition intensities for all allowed categorical jumps.  It must produce
samples whose empirical distribution is close to the target data distribution.

The core gym uses the paper's 15-state analytic model.  The forward process has
rate matrix `Q = E / 15 - I`, target distribution `p0`, horizon `T = 12`, and
closed-form forward marginals.  Because the exact score is available, the
experiment isolates numerical error from model-estimation error.

---

# Method

The paper introduces high-order Poisson tau-leaping solvers for discrete
diffusion inference.  The baseline tau-leaping method samples independent
Poisson counts for each possible jump using the current reverse CTMC intensity.

The proposed `theta`-RK-2 solver first takes a fractional tau-leap to a
theta-section point, evaluates the intensity there, then performs a full update
from the original state using the Runge-Kutta weighted intensity
`(1 - 1 / (2 theta)) * mu_start + (1 / (2 theta)) * mu_mid`, clamped positive.

The proposed `theta`-Trapezoidal solver uses the same intermediate tau-leap, but
the second stage starts from the intermediate state and runs for
`(1 - theta) * Delta` with extrapolated intensity
`alpha1 * mu_mid - alpha2 * mu_start`, where
`alpha1 = 1 / (2 theta (1 - theta))` and
`alpha2 = ((1 - theta)^2 + theta^2) / (2 theta (1 - theta))`.

Both high-order implementations reject updates with multiple jumps along the
single categorical dimension, matching the paper's well-posedness convention.

---

# Evaluation

Evaluation reads JSON artifacts written by samplers under `outputs/toy/`.
For each method and step count it computes `KL(p0 || q_hat_T)` from the saved
empirical distribution.  It then fits the log-log slope of KL versus step count
to estimate the convergence order.

Primary metric:

- `convergence_order`: higher is better; the paper predicts first order for
  tau-leaping and second order for the high-order schemes.

Constraint metrics:

- `kl_at_max_steps`: lower is better at the fixed maximum step count.
- `max_steps`: lower is better; comparisons must use the same step budget.
- `max_nfe`: lower is better; this records score/intensity evaluations so
  two-stage solvers remain accountable for their larger per-step cost.
- `mean_rejection_rate`: lower is better; multiple jumps in one categorical
  dimension are rejected as in the paper.
- `positive_intensity_fraction`: higher is better; extrapolated intensities
  should remain mostly positive before clamping.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 9m | **GPU**: 0.0h
- **Jobs**: 2 total (1 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 3 total (1 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim` | **Status**: done
- **Working time**: 4m | **GPU**: 0.0h
- **Jobs**: 4 total (2 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: core_claim**
