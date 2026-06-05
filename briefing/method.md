# Method

The paper introduces high-order Poisson tau-leaping solvers for discrete
diffusion inference.  The baseline tau-leaping method samples independent
Poisson counts for each possible jump using the current reverse CTMC intensity.

The proposed `theta`-RK-2 solver first takes a fractional tau-leap to a
theta-section point, evaluates the intensity there, then performs a full update
from the original state using the Runge-Kutta weighted intensity
`(1 - 1 / (2 theta)) * mu_start + (1 / (2 theta)) * mu_mid`, clamped positive.

The proposed `theta`-Trapezoidal solver uses the same intermediate tau-leap, but
the second stage starts from the intermediate state and runs for
`(1 - theta) * Delta` with extrapolated intensity
`alpha1 * mu_mid - alpha2 * mu_start`, where
`alpha1 = 1 / (2 theta (1 - theta))` and
`alpha2 = ((1 - theta)^2 + theta^2) / (2 theta (1 - theta))`.

Both high-order implementations reject updates with multiple jumps along the
single categorical dimension, matching the paper's well-posedness convention.
