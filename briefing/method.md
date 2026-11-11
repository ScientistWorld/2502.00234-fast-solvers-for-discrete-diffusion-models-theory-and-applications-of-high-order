# Method

The paper introduces high-order Poisson tau-leaping solvers for discrete
diffusion inference.  The baseline tau-leaping method (`baseline/tau_leaping.py`)
samples independent Poisson counts for each possible single-site jump using
the intensity at the start of each step, over an arithmetic time grid with one
evaluation per step.

`method/fast_solvers/samplers.py` implements both of the paper's two-stage
solvers (Algorithms 1 and 2) with `theta = 1/2`, an arithmetic grid on
`[0, T]` and two evaluations per step (so a budget of `B` evaluations gives
`B / 2` steps):

- **theta-RK-2**: tau-leap for `theta * Delta` to an intermediate state,
  evaluate the intensity there, then tau-leap a full step from the original
  state with the weighted intensity
  `1[mu_start > 0] * ((1 - 1/(2 theta)) mu_start + (1/(2 theta)) mu_mid)_+`.
- **theta-Trapezoidal** (the reference): the same first stage, then a second
  tau-leap from the intermediate state for `(1 - theta) * Delta` with the
  extrapolated intensity `(alpha1 mu_mid - alpha2 mu_start)_+`, where
  `alpha1 = 1 / (2 theta (1 - theta))` and
  `alpha2 = ((1 - theta)^2 + theta^2) / (2 theta (1 - theta))`.

As in the paper and the authors' released sampler, the two stages are combined
per jump increment `nu`: the intensity of jump `nu` at one state is paired with
the intensity of the same jump `nu` at the other, and jumps that would leave
`[0, S)` carry no intensity.  Updates with more than one jump along a site are
rejected.  Both solvers are registered in `SOLVERS` and run by the evaluators.
