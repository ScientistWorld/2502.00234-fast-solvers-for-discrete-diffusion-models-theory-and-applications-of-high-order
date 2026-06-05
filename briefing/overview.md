# Overview

This reproduction implements the core high-order inference algorithms from
"Fast Solvers for Discrete Diffusion Models" and evaluates them on the paper's
analytic 15-state toy model.  The experiment tests whether `theta`-Trapezoidal
and `theta`-RK-2 achieve faster KL convergence than first-order tau-leaping
under the same discrete diffusion reverse-process setting.
