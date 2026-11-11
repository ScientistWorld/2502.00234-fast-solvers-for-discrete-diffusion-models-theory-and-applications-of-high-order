# Constraints

- The problem is fixed by the evaluators: 48 sites with 15 states, the
  uniform-rate forward noise per site, horizon `T = 12`, uniform initial
  states, held-out data distributions, 10,000 samples, and NFE budgets
  16/32/64/128.  Methods cannot change any of these.
- The only access to an instance is `problem.rates(states, s)`.  Each request
  of `m` rows costs `ceil(m / num_samples)` evaluations; the evaluator refuses
  requests beyond `nfe_budget`, times outside `[0, T]`, invalid states, or more
  than `4 * num_samples` rows, and the run then fails.
- Solvers must return `(num_samples, num_sites)` integer states in
  `[0, num_states)` within 30 minutes per run.  A failed run on any instance or
  budget gives the method the worst value of every scored metric.
- All scored numbers are computed by the evaluator from the returned samples.
  Files written by a method are not read; score files are deleted and
  regenerated atomically on every evaluation.
- Every registry under `method/` and `baseline/` (top-level `SOLVERS` dict) is
  scored; methods cannot opt out by editing other files.
