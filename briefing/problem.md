# Problem Definition

The task is to sample from a discrete diffusion model's reverse continuous-time
Markov chain on a finite categorical state space.  A solver receives an initial
sample from the forward process stationary distribution, a reverse-time grid,
and transition intensities for all allowed categorical jumps.  It must produce
samples whose empirical distribution is close to the target data distribution.

The core gym uses the paper's 15-state analytic model.  The forward process has
rate matrix `Q = E / 15 - I`, target distribution `p0`, horizon `T = 12`, and
closed-form forward marginals.  Because the exact score is available, the
experiment isolates numerical error from model-estimation error.
