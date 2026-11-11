# Experiments

## multisite_reverse_ctmc

This experiment asks how well a sampler can simulate the reverse process of a
discrete diffusion model when the number of score evaluations is limited and
smaller than the number of dimensions.  The score is exact, so the only source
of error is the simulation itself, as in the paper's analytic toy model; the
state space has 48 sites of 15 states, too large to enumerate, and the data
distributions have several separated modes of unequal weight and spread.

Each method is run by the evaluator on the same instances, initial states and
random seeds at four evaluation budgets (16, 32, 64 and 128 per sample), and
its samples are compared with the exact data distribution.  The experiment
establishes:

- sample quality at matched compute, across the budget range (pair-marginal
  KL at each budget);
- whether the whole configuration of each sample is right, including how mass
  is split between modes (mode-and-likelihood profile KL);
- per-site fidelity, the quantity of the paper's one-site toy (site KL);
- how fast quality improves as the budget grows (convergence order).

The compute budget is a hard constraint enforced by the evaluator: a method
that requests more evaluations than allowed, or returns invalid samples, fails
and receives the worst value of every metric.  Held-out instances are scored
by the hidden test with the same budgets, sample size and metrics as the
public train instances.
