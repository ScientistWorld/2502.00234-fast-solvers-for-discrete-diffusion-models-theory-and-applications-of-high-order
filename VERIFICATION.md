> Historical: this file describes the pre-hardening one-site artifact gym (iterations 1-10). The current gym is documented in `HARDENING.md`, `briefing/` and `scoring/`.

# Verification Report

## Initial Assessment

The workspace started this verification at `core_claim`.  That milestone was
justified: the paper's finite-state toy experiment is implemented with saved
sampler artifacts, the tau-leaping baseline is outside `method/`, evaluation
reads artifacts rather than rerunning the method, and the reproduced
theta-Trapezoidal result supports the central high-order claim on the analytic
15-state benchmark.

## Issues Found

- Critical: the scored schema included method-specific, self-reported
  diagnostics (`mean_rejection_rate` and `positive_intensity_fraction`).  These
  were weak gym metrics because future methods could fabricate or omit them,
  and they leaked details of one solver family into the evaluation contract.
- Critical: strict method-agnostic documentation contained solver-specific
  wording such as two-stage updates, rejected jumps, extrapolated intensities,
  and clamping.
- Minor: `scoring/reference.json` did not explicitly explain why the paper
  reference contains convergence-order targets but not tabulated toy KL values.

## Fixes Applied

- Removed the method-specific diagnostics from `scoring/reference.json`,
  `eval/evaluate_toy.py`, `eval/train/evaluate_toy_split.py`, and
  `eval/test/evaluate_toy_split.py`; refreshed the committed score files from
  existing artifacts by deleting those unscored fields.
- Reworded `briefing/evaluation.md`, `scoring/TARGETS.md`, and
  `scoring/DIRECTION.md` to describe distribution quality, fixed step/NFE
  budgets, and the allowed finite-state inference direction without leaking the
  paper's implementation details.
- Added a `notes` field to `scoring/reference.json` explaining that Figure 2
  reports fitted toy KL behavior rather than exact numeric KL tables.
- Removed generated `__pycache__` directories from the workspace tree.
- Confirmed `python validate.py --compare` passes after the scoring cleanup.

## Final Judgment

The workspace is ready to serve as a gym for the reproduced core claim.  Improve
mode has the paper method, baseline, scores, and documentation.  From-scratch
mode has method-agnostic problem and evaluation documents, a data/model
definition under `data/`, reusable artifact-based evaluators, and scoring based
on scientific output quality plus machine-independent compute constraints.  The
main limitation is scope: the gym covers the paper's analytic toy benchmark
rather than the large text, image, and GSM8K experiments.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: high
- Key limitation: analytic 15-state benchmark only
