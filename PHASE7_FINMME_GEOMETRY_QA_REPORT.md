# Phase 7: End-to-End FinMME Geometry QA

## Decision

The present Geometry pipeline does **not** improve final FinMME question-level accuracy over the cached Raw Qwen-VL baseline. The practical system variant—use Geometry when its full pipeline succeeds and otherwise fall back to Raw VLM—scores **55.49%** under the FinMME official-code rule, versus **55.59%** for Raw VLM alone (−0.10 percentage points). Hard-routing every automatically selected question to Geometry scores **54.98%** (−0.61 points).

This negative end-to-end result does not invalidate Geometry recovery itself. Earlier module-isolation tests remain unchanged: accepted Bars reached 97.4% within 2% of the y-axis span, and the strict Phase 3 single-Line subset recovered 17/17 targets with semantic X grounding. Phase 7 shows that the current automatic chart/target/series routing is not yet reliable enough to convert that measurement accuracy into dataset-level QA gains.

## Scope and protocol

- Dataset: the existing 11,099-question FinMME reproduction; no new data or image search.
- Baseline: cached full-dataset Qwen2.5-VL-7B predictions and metrics.
- Geometry scope: automatically selected natural no-value-label, simple single-y-axis Bar/Line questions; numerical and numeric-option single-choice only; direct read, absolute difference, and growth rate only.
- Excluded by construction: multi-axis, stacked, Area, Pie, Boxplot, aggregate/extremum questions, and non-numeric options.
- Question path: question → semantic target/operation → Bar or Line Geometry → Python calculation → FinMME Gold/tolerance scoring.
- Line path: semantic X grounding and local fitting from Phase 3, plus existing series resolution.
- Refusal policy: a failed Geometry stage is incorrect for the hard route. The fallback route uses cached Raw VLM only when Geometry refuses before producing an answer.
- No GPU or VLM inference was run in this phase; the cached Raw VLM baseline was reused.

The automatic prefilter selected 167 questions from 143 charts: 53 Bar and 114 Line; 59 numerical and 108 single-choice; 122 direct, 20 difference, and 25 growth-rate questions. No questions were hand-picked after looking at Gold.

## Final accuracy

Primary numbers use the FinMME official-code correctness field for the cached baseline and the same exact choice-letter / numerical-tolerance rule for Geometry.

| System on all 11,099 FinMME questions | Correct | Accuracy | Change vs Raw VLM |
|---|---:|---:|---:|
| Raw Qwen-VL | 6,170 | 55.59% | — |
| Geometry hard route on 167 selected questions | 6,102 | 54.98% | −0.61 pp |
| Geometry-on-success, Raw-VLM-on-refusal | 6,159 | **55.49%** | −0.10 pp |

For compatibility with the existing reproduction's paper-protocol score, the corresponding accuracies are 55.61%, 54.98%, and 55.49%. The conclusion is unchanged.

## Automatically routed subset

Geometry completed the full path on 49/167 questions (29.3% coverage) and answered 12 of those 49 correctly (24.5%). Counting refusals as errors gives 12/167 = 7.2%. Raw VLM answered 80/167 = 47.9% correctly. The fallback system answered 69/167 = 41.3% correctly.

| Slice | Questions | Geometry coverage | Raw VLM | Geometry, all selected | Geometry, completed only | Fallback |
|---|---:|---:|---:|---:|---:|---:|
| All | 167 | 29.3% | 47.9% | 7.2% | 24.5% | 41.3% |
| Bar | 53 | 9.4% | 64.2% | 3.8% | 40.0% | 64.2% |
| Line | 114 | 38.6% | 40.4% | 8.8% | 22.7% | 30.7% |
| Numerical | 59 | 33.9% | 16.9% | 3.4% | 10.0% | 15.3% |
| Single choice | 108 | 26.9% | 64.8% | 9.3% | 34.5% | 55.6% |
| Direct read | 122 | 29.5% | 61.5% | 9.8% | 33.3% | 52.5% |
| Difference | 20 | 45.0% | 15.0% | 0.0% | 0.0% | 15.0% |
| Growth rate | 25 | 16.0% | 8.0% | 0.0% | 0.0% | 8.0% |

The paired outcome on the 167 routed questions was: 85 both wrong, 70 Raw-only correct, 10 both correct, and 2 Geometry-only correct. Thus the current route loses substantially more correct Raw answers than it rescues.

## Failure audit

Of the 155 Geometry-incorrect questions:

| Failure stage | Count |
|---|---:|
| X grounding / series ambiguity / line fitting | 70 |
| Bar runtime structure check | 46 |
| Full Geometry answer produced but wrong | 37 |
| Semantic Bar target not found | 2 |

The most common concrete refusal statuses were `x_grounding_unrecoverable` (34 single-target plus 20 two-target cases), `bar_count_out_of_scope` (27), direct numeric annotation detected at runtime (15), and ambiguous series (11). Only 5 Bar and 44 Line questions reached a final answer.

A small Gold-free confidence-gate audit did not expose a deployable winning slice: Bar-only fallback was neutral, while Line, numerical, single-choice, and direct-read fallback were all below their Raw VLM baselines. Local-fit residuals also did not separate the two Geometry-only wins from the Geometry losses. No tuned gate is reported.

## Leakage and reproducibility checks

- All 167 input IDs, prediction IDs, and scored rows are unique and match one-to-one.
- Runtime input and prediction records contain none of `answer`, `reference`, `gold`, `tolerance`, `data_label_bbox`, or `center_x`.
- Gold and tolerance are stored separately and joined only by `evaluate` after predictions exist.
- The sample CSV records question, semantic target, detected X label, selected target geometry, y-axis ticks, recovered values, Python reasoning, prediction, Gold, tolerance, and all correctness flags.
- The scoring subcommand has no OCR/GPU import requirement, so cached predictions can be re-evaluated without rerunning Geometry.

## Interpretation and next decision

The paper idea remains plausible as a **measurement tool**, but the current fully automatic chain is not yet a competitive **QA system**. Scaling the same pipeline to more images would mostly scale routing errors and is therefore not the next useful experiment.

The shortest evidence-driven next step is to keep the same frozen 167-question set and use a VLM only for semantic target/series/operation parsing, while forcing all numeric evidence and arithmetic through Geometry/Python. That tests the intended division of labor directly without adding new chart types, agents, training, or more image curation. It should be treated as a new experiment; Phase 7 itself is complete and negative.

## Artifacts

- `src/finmme_geometry_qa.py` — prepare, resumable CPU Geometry, and independent evaluation commands.
- `results/phase7_finmme_geometry_qa_samples.csv` — 167 sample-level audit rows.
- `results/phase7_finmme_geometry_qa_metrics.csv` — routed slices plus full-dataset official and paper-protocol accuracy.
- `results/phase7_finmme_geometry_qa_summary.json` — runtime failure counts.

Phase 1–6 artifacts and reported results were not modified.
