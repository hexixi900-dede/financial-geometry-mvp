# Phase 5 Plan: Natural No-label Geometry QA

## Material Passport

- Origin: experiment planning
- Date: 2026-09-04
- Status: approved for implementation
- Working branch: `phase5-natural-qa`

## Research question

On financial charts that naturally omit point/bar value labels, does
question-conditioned Geometry answer numerical questions more accurately than a
direct VLM?

## Scope

- Primary charts: simple vertical Bar and single-series Line with one left y-axis.
- Secondary diagnostic: single-axis multi-line only after the existing Phase 4
  prototype passes a short acceptance audit.
- Questions: direct read, absolute difference, and percentage growth/change.
- Deferred: grouped/stacked Bar, dual-axis, combination, Area, multi-panel, Agent,
  RL, LoRA, and new calibration models.

## Stage A: accept and freeze the Phase 4 prototype

Run a lightweight acceptance check without rerunning the experiment:

1. Recompute the stored Phase 4 coverage and MAE from the sample CSV.
2. Confirm train/test charts do not overlap and runtime rows report no Gold/bbox use.
3. Select 30 representative overlays (20 successes and 10 rejections) for review.
4. Keep Phase 4 as exploratory evidence. Do not add artifact hashes or a broad
   reproducibility framework.

## Stage B: natural no-label pilot

Use FinChart-Bench QA because the FinMME strict natural-Bar screen retained only
one usable chart. Build a first pilot with 50-100 charts and roughly 100-200
questions, without padding the sample if strict charts are scarce.

Runtime inputs are limited to image and question. Gold and the dataset reasoning
field are stored separately and joined only during evaluation.

Pipeline:

`question -> operation/x targets -> OCR axis/labels -> target bar or line point -> values -> Python calculation`

## Systems and metrics

Compare:

1. Raw Qwen2.5-VL-7B direct answer.
2. Geometry without calibration.
3. Geometry plus the frozen Phase 1 calibration only as an optional ablation.

Primary metric: tolerance accuracy with abstentions counted as incorrect.
Secondary metrics: coverage, conditional accuracy, MAE, median absolute error,
relative error, and axis-normalized error. Because FinChart-Bench has approximate
answers but no official tolerance, freeze the pilot tolerance before predictions:

`max(5% of |Gold|, 2% of the recovered y-axis span, 1e-6)`

## Decision gate

- If the pilot shows a useful Geometry gain, freeze the code and scale the same
  pipeline to 100-200 questions or more without adding features.
- If coverage is the main problem, fix only the largest rejection bucket.
- If target selection is correct but geometry is wrong, fix the corresponding Bar
  or Line recovery step.
- Calibration remains optional unless it improves held-out chart results.

## Implementation budget

- Reuse Phase 2 Bar and Phase 3 Line functions.
- Add at most three main scripts and about 800 lines of new Python.
- Use one sample CSV, one metrics CSV, one report, and at most 30 overlays.
- Keep only necessary checks: no chart split overlap and no Gold/bbox runtime input.
- No artifact-hash verifier, registry, new framework, or mirrored unit-test suite.

