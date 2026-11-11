# Targets

Optimize `multisite_reverse_ctmc` (the only experiment, weight 1.0).

Benefit metrics (all lower is better except the order):

- `log10_pair_kl_nfe16`, `log10_pair_kl_nfe32`, `log10_pair_kl_nfe64`,
  `log10_pair_kl_nfe128` — pair-marginal KL at each evaluator-fixed NFE budget
  (coefficient 0.75 each).
- `log10_profile_kl_mean` — KL on the (mode, log-likelihood decile) profile
  (coefficient 1.0).
- `log10_site_kl_mean` — single-site KL (coefficient 0.5).
- `convergence_order` — decay rate of pair KL with NFE, primary metric
  (coefficient 0.5, higher is better).

Constraint (enforced, not traded off): every sample of every budget must be
produced within `nfe_budget` intensity evaluations counted by the evaluator,
within the time limit, as valid states.  Any violation fails the method and
assigns the worst value of every metric above.
