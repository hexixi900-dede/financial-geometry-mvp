# Phase 5 Corrected Plan: FinMME Geometry Scale-up

## Why this plan replaces the attempted FinChart-Bench pilot

The project was built and validated on FinMME. FinChart-Bench was named only as
a fallback if FinMME could not supply enough natural no-label questions. The
attempted FinChart pilot selected questions before chart structure, so 124 of
132 rows failed the combined `no value labels + recoverable single y-axis` gate
and no row reached a successful end-to-end Geometry answer. That run is kept as
a diagnostic artifact and is not a Geometry performance result.

This plan supersedes `PHASE5_PLAN.md` for execution. The older plan and its zero-
coverage diagnostic stay in the repository as an audit trail.

## Evidence already frozen

| Evidence line | Data | Current result | Interpretation |
|---|---:|---|---|
| Phase 1 masked Geometry | 87 FinMME labels: 72 Bar, 15 Line | Raw test MAE 1.7603; calibrated 1.4216 | Useful narrow module evidence |
| Phase 2 natural Bar QA | 1 FinMME question | Geometry 0/1; VLM 0/1 | Inconclusive because the strict subset is too small |
| Phase 3 single-Line Auto-X | 17 labels on 8 charts | Auto-X local MAE 0.1747, 100% coverage | X semantic grounding is worth retaining |
| Phase 4 multi-Line prototype | 65 labels on 32 charts | 53 multi-line rows; 38 successful, 15 ambiguous | Exploratory; manual review still limited |

No Phase 1-4 result, split, masked image, or overlay will be modified.

## Data accounting

- FinMME is already local: 11,099 questions, 4,446 deduplicated chart images.
- It contains 1,852 Numerical/Calculation questions with official Gold and
  absolute tolerance (one tolerance is missing).
- The Phase 1 candidate scan deeply inspected 929 charts, found 107 viable
  label-linked single-axis charts, and constructed 87 masked-value samples.
- The existing cheap prefilter identified 1,178 additional eligible charts.
  Earlier partial jobs scanned 611 of them and merged those rows into the 929-
  chart file. Exactly 567 eligible charts remain unscanned.
- At the observed rate, 567 charts are expected to yield roughly 40 viable
  charts and 30-40 accepted samples. This is an estimate, not a quota.

## Research questions

### Primary: larger Geometry module validation

On unseen FinMME charts with reliable visible value labels, after masking only
the value text, how accurately can single-axis Bar/Line Geometry recover that
value? This extends the 87-sample module-isolation experiment without changing
its rules.

### Secondary: semantic X targeting

For newly accepted Line samples, how much performance is lost when target x is
obtained from x-axis OCR and question semantics rather than the hidden value-
label center? For Bar samples, Question-to-target evaluation is included only
when an x-axis label can select the bar without any hidden bbox/Gold input.

### Separate natural-QA statement

The strict FinMME natural no-label Bar screen produced only one accepted
question. This is reported as a dataset limitation, not padded with out-of-scope
charts. A direct VLM comparison is run only if a later structure-first rescreen
produces a nontrivial natural subset.

## Scope

Included:

- vertical Bar and single-Line charts;
- one recoverable linear y-axis;
- value-label masking that preserves bar tops, line pixels, axes, x labels,
  legend, and time/category text;
- Oracle-X Geometry as an upper bound and Auto-X where semantic anchors exist;
- the frozen Phase 1 calibrator as an optional external held-out ablation;
- raw, calibrated, relative, and axis-normalized errors.

Excluded:

- dual-axis Bar+Line combinations, including the Cross-Border Revenue example;
- stacked/grouped Bar, multi-panel, Area, Pie, Boxplot;
- Agent, RL, LoRA, or a new calibration model;
- forcing a target sample count by weakening structural rules.

## Execution stages

1. **Freeze and branch.** Work only on `phase5-natural-qa`; write all new
   experiment artifacts under `phase5_finmme_scale/`.
2. **Rebuild remaining manifests.** Subtract the 929 already scanned chart IDs
   from the FinMME eligible manifest, producing two disjoint shards totaling
   567 charts.
3. **CPU deep scan.** Reuse the existing EasyOCR/axis/data-label scanner. Run
   two 3-thread resumable CPU shards; no GPU initialization.
4. **Masked construction.** Merge only the new scan rows, then reuse the Phase 1
   constructor in the isolated Phase 5 output directory. Do not append to the
   original 87 samples.
5. **Primary evaluation.** Report new held-out Raw Geometry metrics separately,
   then a clearly labelled combined 87 + new descriptive summary.
6. **Semantic targeting.** Run Auto-X on eligible new Line samples and automatic
   x-label-to-bar matching on eligible new Bar samples. Abstain when anchors are
   unreliable.
7. **Calibration ablation.** Apply the already frozen Phase 1 calibrator to the
   new charts only. Compare it with zero correction and global/type constant
   bias; do not retrain on the new test charts.
8. **Audit and report.** Save sample CSV, rejection counts, chart-disjoint
   accounting, and 30 representative overlays. Produce
   `PHASE5_FINMME_SCALE_REPORT.md`.
9. **Natural VLM gate.** Only if structure-first FinMME filtering yields enough
   natural questions, run Qwen2.5-VL-7B with the safe 3-check GPU waiter.

## Medium-complexity implementation budget

- Reuse `partition_remaining_manifest.py`, `scan_candidates.py`,
  `run_geometry_samples.py`, Phase 3 grounding, and the frozen calibrator.
- Add one small summarizer/evaluator and one shell runner.
- Add a `--threads` option to the scanner so two CPU shards use at most six
  cores total.
- No hashes, artifact registry, orchestration framework, or duplicate test
  suite.
- Required safety checks are limited to disjoint chart IDs, no hidden target
  fields in Auto-X inputs, and no modification of frozen Phase 1-4 paths.

## Decision rule after Phase 5

- If new held-out Geometry error remains low, the paper can claim the core
  recovery mechanism scales within simple single-axis charts.
- If Auto-X degrades sharply, the next method work is semantic target grounding,
  not calibration.
- If Geometry remains good but natural QA is still too sparse, keep the module
  claim and obtain a better natural-chart QA dataset later.
- Only after this gate is passed should the project add dual-axis series-to-axis
  assignment and evaluate combination charts.
