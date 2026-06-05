# Direction

Work within numerical inference for discrete diffusion reverse CTMCs.
Improvements should refine how a sampler uses transition intensities, time
grids, and finite-state update validity within the same problem family.

Do not replace the task with supervised fitting to the target distribution or
direct multinomial sampling from the known target; that bypasses the inference
algorithm being benchmarked.
