# Constraints

Use the same 15-state analytic CTMC, target distribution, horizon `T = 12`, and
uniform time grids unless explicitly adding a separately named experiment.
Evaluation must read method artifacts rather than rerunning a sampler.
Report both the step count and NFE budget so faster apparent convergence is not
achieved by hiding extra score/intensity evaluations.  Matched-compute claims
must use `common_nfe`, the largest NFE value present for every scored method,
rather than comparing only the largest saved step count.

Future methods may change the solver but must report the same output JSON
format under `outputs/toy/` so `scripts/evaluate.sh` can score them.
