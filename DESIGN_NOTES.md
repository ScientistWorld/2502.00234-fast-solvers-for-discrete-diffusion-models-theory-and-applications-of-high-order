# Train/Test Evaluation Design Notes

## Checked

- Reviewed `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side. The train wrapper calls `eval/train/` and writes `scoring/scores_train.json`; the test wrapper calls `eval/test/` and writes `scoring/scores_test.json`.
- Checked that both split evaluators read saved method artifacts from `outputs/toy/*_outputs.json` and do not import from `method/`.
- Grepped for train/test cross-imports. The only `eval/train` and `eval/test` references are expected wrapper calls and docstrings.
- Checked `scoring/reference.json` for paper-agreement metrics and wall-clock metrics. None are active.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stubs are present.
- Ran `bash -n` on both split wrapper scripts.

## Changed

- Strengthened the anti-shortcut split beyond alternating step budgets by adding complementary state-subset scoring:
  - train scores even-ranked saved step budgets and states `[0, 2, 4, 6, 8, 10, 12, 14]`;
  - test scores odd-ranked saved step budgets and states `[1, 3, 5, 7, 9, 11, 13]`.
- Added `slice_state_subset_kl_at_max_steps` and `slice_state_subset_mass_abs_error_at_max_steps` to both evaluators. These make a method that only optimizes visible aggregate metrics more likely to show a train/test gap.
- Added `mean_total_variation` across the scored step grid to reduce the incentive to optimize only the largest step budget.
- Recorded the slice state indices in score metadata to make runner-side debugging easier.
- Completed all active metric metadata in `scoring/reference.json` with `higher_is_better`, `coefficient`, and `description`.

## Runtime Concerns For Next Iteration

- The split evaluators require at least two unique step budgets per slice, so a future method must save at least four distinct step counts. If a method emits fewer budgets, both scripts will fail intentionally rather than silently scoring a degenerate convergence estimate.
- The toy problem exposes the analytic target distribution in `data/`, so no metric-only split can fully prevent a method from hard-coding target-shaped outputs. The added state-subset and budget splits raise the cost of narrow metric gaming, but a stronger future design would need additional generated CTMC instances or perturbations if the method artifact contract is expanded.
- I did not run either evaluator or produce score files in this pass, per the source-edit-only instruction.
