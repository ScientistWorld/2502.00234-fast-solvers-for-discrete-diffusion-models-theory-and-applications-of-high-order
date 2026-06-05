# Constraints

Use the same 15-state analytic CTMC, target distribution, horizon `T = 12`, and
uniform time grids unless explicitly adding a separately named experiment.
Evaluation must read method artifacts rather than rerunning a sampler.

Future methods may change the solver but must report the same output JSON
format under `outputs/toy/` so `scripts/evaluate.sh` can score them.
