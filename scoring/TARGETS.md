# Targets

Optimize `toy_15_state_convergence`.

The primary target is high `convergence_order` from saved empirical
distributions for tau-leaping, theta-RK2, and theta-Trapezoidal artifacts.  A
valid improvement should reduce `kl_at_max_steps` without increasing the fixed
computational budget (`max_steps`) or depending on lower rejection/positivity
constraints being ignored.
