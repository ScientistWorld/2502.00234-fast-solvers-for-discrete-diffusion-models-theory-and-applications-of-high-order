# Direction

Work within numerical inference for discrete diffusion reverse CTMCs:
improvements should change how a sampler uses the reverse-process intensities
over time — the discretization of the time interval, how intensities from
several evaluations are combined, how jumps are drawn and how invalid updates
are handled — while respecting the evaluation budget.

Not in scope, and treated as shortcuts by the grader:

- fitting or identifying the data distribution from intensity queries (for
  example recovering mode weights and factors from ratios) and then sampling
  the fitted model directly;
- replacing the reverse process by Markov chain sampling on the data-end
  conditionals alone;
- reading evaluator files or process state to learn held-out instances.
