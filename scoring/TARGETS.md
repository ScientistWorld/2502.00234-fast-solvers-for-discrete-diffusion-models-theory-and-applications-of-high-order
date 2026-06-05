# Targets

Optimize `toy_15_state_convergence`.

The primary target is high `convergence_order` from saved empirical
distributions across the saved convergence grid.  A valid improvement should
reduce `kl_at_common_nfe`, the KL measured at the largest NFE budget shared by
all scored methods, without increasing the fixed computational budget
(`max_steps`, `max_nfe`, and `common_nfe`).
