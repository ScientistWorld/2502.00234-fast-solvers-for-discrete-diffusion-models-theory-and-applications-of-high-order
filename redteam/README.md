# Benchmark scoring audit

Each attempt has its own `run.sh`. It prepares normal `outputs/toy/*_outputs.json`
artifacts, runs the visible train scorer, then invokes the restored hidden test
as `bash scripts/evaluate_test.sh` with **no arguments** in a separate process.
Candidate processes have exited before the hidden test starts. Nothing edits
`eval/`, the hidden script, Python libraries, import resolution, or processes.
The harness restores public data and clears the previous scores between runs.
The full CPU batch is `bash redteam/run_all.sh`.

Candidate implementation is shared in `candidate.py`; each folder's
`variant.json` selects a concrete branch. Reference and more-samples variants
use the actual improve-mode `method/run_toy.py`. Reference artifacts are
regenerated in this batch, then cached for checks that only change artifacts.
No shortcut reads the paper, hidden evaluator, original scores or audit rank
specification. Shortcuts that replay the supplied method target improve mode;
public-answer, metadata, selector, and ground-truth weaknesses also apply when
starting from scratch.

`originals/` contains audit-only restoration backups. `original_reference_test.json`
and `reference_scoring.json` are read only by the audit analysis, never candidates.
`initial_status.txt` records pre-existing workspace changes that the audit does
not own. Original README.md and manifest.json are device mounts, and four old
trajectory files were already absent.

Each folder retains normal output artifacts, train/test scores if written,
`status.json` with subprocess exit codes, and `rank.json` after batch analysis.
Failed hidden tests are not counted as successful shortcuts, even if an empty
or stale score remains. Absence of a method in an otherwise successful hidden
test is scored by the ranking rule's missing-value skip, hence a zero delta.

Rank delta uses absolute coefficient magnitudes, `higher_is_better`, experiment
weight, and population standard deviations over the listed audit attempts.
The reference delta is zero. The future ranking population is unknown, so
`summary.json` names the normalization cohort and also reports a separate
quality-only cohort for genuine methods. For each shortcut's secondary
sensitivity score, that shortcut is added to the quality cohort before
computing standard deviations. These secondary scores consequently do not
share one normalization population. Raw metric vectors and per-metric
contributions make the results reproducible with other populations. Failed
test exits have null rank deltas, never a claimed zero-score success.

The two CPU jobs ran 24 attempts. `job_evidence/` retains their logs and job
metadata; `summary.json` and `results.md` contain the final recalculation.
Earlier score tables printed in the first job's log use an earlier cohort;
its secondary shortcut scores were subsequently corrected to include the
shortcut in the normalization population.
