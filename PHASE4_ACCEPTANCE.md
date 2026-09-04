# Phase 4 Lightweight Acceptance Note

Date: 2026-09-04

Phase 4 is retained as an exploratory single-axis multi-line geometry result. It
is not a natural no-label QA benchmark and is not used as the Phase 5 primary
result.

The acceptance script recomputed the stored sample-level results without rerunning
OCR or Geometry:

- 65 samples from 32 charts: 53 multi-line and 12 single-line.
- Train/test chart overlap: 0.
- Runtime rows reporting hidden Gold/bbox access: 0.
- Multi-line Auto-column: 37/53 coverage, MAE 0.2272, median AE 0.1708.
- Multi-line Auto-local: 38/53 coverage, MAE 0.3121, median AE 0.1600.
- Runtime outcomes across all samples: 45 success, 17 ambiguous series, and 3
  line-localization failures.

Thirty representative Auto-local overlays were selected: 20 successes and 10
rejections, covering single- and multi-line charts. Contact-sheet inspection found
that successful target-x markers and fitted points were generally placed on the
intended line, while the rejection examples visibly contained missing local line
pixels or series ambiguity. This is a sanity audit, not a claim of full manual
annotation accuracy.

Decision: preserve Phase 4 for the paper's multi-line feasibility/coverage
discussion. Use the natural no-label Phase 5 experiment for the main Raw VLM vs
Geometry comparison. The legacy exhaustive hash validator is not part of the
Phase 5 execution path.
