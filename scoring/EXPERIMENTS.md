# Experiments

## toy_15_state_convergence

This experiment checks whether a sampler approaches the known target
distribution efficiently when numerical error is the main source of error.  It
compares empirical KL across multiple uniform step counts for tau-leaping,
theta-RK2, and theta-Trapezoidal outputs and summarizes the rate at which error
decreases, while also reporting fixed-budget and validity constraints.
The `max_nfe` metric records the score/intensity evaluation budget, which is
the paper's main compute-normalization axis for larger models.

The packaged reproduction uses 50,000 samples per method and step count.  The
result supports the target: theta-Trapezoidal has much lower KL than the
first-order baseline and theta-RK2 at the same maximum step count, with an
empirical convergence order near two.
