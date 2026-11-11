# Overview

This gym is built from "Fast Solvers for Discrete Diffusion Models: Theory and
Applications of High-Order Algorithms" (arXiv 2502.00234).  The task is
inference for discrete diffusion models: simulating the reverse CTMC on a
multi-site categorical space well within a small budget of intensity
evaluations.  The analytic setting extends the paper's exact-score toy model
(15 states, uniform-rate noise, `T = 12`) from one site to 48 sites, the
regime where the number of evaluations is smaller than the number of sites,
so that exact simulation and site-by-site Markov chain sampling are not
viable and samplers must update all sites in parallel.

The reference is the paper's sampler at the paper's settings, with the paper's
first-order baseline and second two-stage scheme as comparison methods.  See
`briefing/problem.md` for the solver interface and `briefing/evaluation.md`
for the measurement.
