# Targets

Optimize `toy_15_state_convergence`.

The primary target is high `convergence_order` from saved empirical
distributions across the saved convergence grid.  A valid improvement should
reduce `kl_at_max_steps` without increasing the fixed computational budget
(`max_steps` and `max_nfe`) or depending on rejection/positivity constraints
being ignored.
