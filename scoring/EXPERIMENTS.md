# Experiments

## toy_15_state_convergence

This experiment checks whether a sampler approaches the known target
distribution efficiently when numerical error is the main source of error.  It
compares empirical KL across multiple uniform step counts and summarizes the
rate at which error decreases, while also reporting fixed-budget and validity
constraints.
