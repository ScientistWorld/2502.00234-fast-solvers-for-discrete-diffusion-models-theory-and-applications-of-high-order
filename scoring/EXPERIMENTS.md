# Experiments

## toy_15_state_convergence

This experiment checks whether a sampler approaches the known target
distribution efficiently when numerical error is the main source of error.  It
compares empirical KL across multiple uniform step counts and summarizes the
rate at which error decreases, while also reporting fixed-budget and validity
constraints.  The `max_nfe` metric records the score/intensity evaluation
budget so improvements cannot be obtained by hiding extra evaluations.  The
`kl_at_common_nfe` and `total_variation_at_common_nfe` metrics compare methods
at the largest function-evaluation budget present for every scored method.

The packaged reproduction uses 50,000 samples per method and step count.  The
baseline scores leave clear room for improvement: the best included sampler
has lower matched-NFE KL than simpler included samplers, with an empirical
convergence order near two.
