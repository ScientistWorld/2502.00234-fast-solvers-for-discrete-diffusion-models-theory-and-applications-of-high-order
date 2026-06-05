# Evaluation

Evaluation reads JSON artifacts written by samplers under `outputs/toy/`.
For each method and step count it computes `KL(p0 || q_hat_T)` from the saved
empirical distribution.  It then fits the log-log slope of KL versus step count
to estimate the convergence order.

Primary metric:

- `convergence_order`: higher is better; it estimates how rapidly empirical
  distribution error decreases as the time grid is refined.

Constraint metrics:

- `kl_at_max_steps`: lower is better at the fixed maximum step count.
- `max_steps`: lower is better; comparisons must use the same step budget.
- `max_nfe`: lower is better; this records transition-intensity evaluations so
  comparisons remain fair when a sampler uses more than one evaluation per
  saved step.
