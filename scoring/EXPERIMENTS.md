# Experiments

## toy_15_state_convergence

This experiment checks whether a sampler approaches the known target
distribution efficiently when numerical error is the main source of error.  It
compares empirical KL across multiple uniform step counts and summarizes the
rate at which error decreases, while also reporting fixed-budget and validity
constraints.

The packaged reproduction uses 50,000 samples per method and step count.  The
result supports the target: the best high-order solver has much lower KL than
the first-order baseline at the same maximum step count and an empirical
convergence order near two.
