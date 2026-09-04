# Phase 5 — FinMME Natural No-Label Line Geometry Report

## Executive conclusion

This phase closes the previously missing FinMME experiment: original questions on charts that naturally lack data-value labels, with `Question → target → x grounding → line geometry → Python answer` and no Gold-side target coordinate.

The run is useful as a **negative diagnostic**, but it is not a valid test of the claim that Geometry beats a Raw VLM. The automatic funnel retained 22 charts, yet manual overlay audit found that only four were genuine single-line charts with a reasonably interpretable point target. Three of the seven nominally clean charts were actually scatter, multi-line, or area charts. At least one remaining sample has an unmistakable question/Gold alignment problem: sample 1703 gives about 238–240 bps from both the visible curve and Raw VLM, while FinMME Gold is 30.

Therefore:

- The Phase 3 result remains valid: x-axis semantic grounding is useful on masked, simple single-line module-isolation samples.
- This phase does **not** show that the end-to-end natural FinMME Geometry route beats Raw VLM.
- It also does **not** refute Geometry. The evaluation subset is too small and structurally/semantically contaminated for that claim.
- The next experiment should be a 30–50 chart, manually audited natural point-reading benchmark. Scaling the current automatic filter would mainly scale screening errors.

## What was run

- Data source: existing local FinMME records and images only; no FinChart-Bench sample is used in this result.
- Runtime input: original image, original question-derived operation/temporal targets, and automatically parsed series phrase.
- Runtime exclusions: Gold, tolerance, Raw VLM answer, data-label bbox, manual target coordinate, and hidden value-label center.
- Geometry path: EasyOCR x/y labels → exact or bracketed temporal grounding → Phase 4 line components → Phase 3 local robust fitting → y-axis interpolation → Python operation.
- Raw VLM baseline: reused existing complete Qwen2.5-VL-7B FinMME predictions; no GPU rerun was required.
- Calibration: the frozen Phase 1 calibrator was applied to each recovered component before Python reasoning. It was not retrained on these charts.
- Outputs: 22 sample records and 41 target-level debug overlays.

## Dataset funnel

| Stage | Count | Meaning |
|---|---:|---|
| Deduplicated FinMME charts in the Phase 4 scan | 4,446 | Existing full-corpus audit |
| Natural no-value-label, one detected line, usable x anchors | 623 | Structural prefilter only |
| Accepted numerical questions after semantic/structure rules | 22 | One question per chart |
| Nominal primary-clean tier | 7 | No obvious annotations from the automatic scan |
| Annotated exploratory tier | 15 | Mostly stock charts with target-price/rating annotations |
| Geometry returned a final answer | 7 | 31.8% overall coverage |
| Manually confirmed clean quantitative E2E samples | 0 | Gold/target consistency was not strong enough for a clean claim |

Important rejection groups include 236 questions that were not simple one/two-point reads, 16 possible Bar or Bar/Line combinations, five interval/boundary targets, and four P/E reference-band charts. The low final count is a property of the strict scope and the FinMME question/chart mix; it should not be filled by relaxing the definition.

## Official-tolerance diagnostic metrics

The numbers below use the FinMME-provided numerical `reference` and `tolerance` in display units. Missing Geometry predictions count as incorrect in accuracy and are excluded only from conditional MAE.

| Scope | Method | Coverage | Accuracy | Conditional MAE | Median absolute error |
|---|---|---:|---:|---:|---:|
| All 22 | Raw VLM | 100.0% | 4.5% | 278.288 | 29.750 |
| All 22 | Geometry column | 31.8% | 0.0% | 39.254 | 7.500 |
| All 22 | Geometry local fit | 31.8% | 0.0% | 40.599 | 13.841 |
| All 22 | Geometry + frozen calibration | 31.8% | 0.0% | 39.640 | 7.472 |
| Nominal primary 7 | Raw VLM | 100.0% | 0.0% | 41.290 | 16.000 |
| Nominal primary 7 | Geometry local fit | 100.0% | 0.0% | 40.599 | 13.841 |
| Nominal primary 7 | Geometry + frozen calibration | 100.0% | 0.0% | 39.640 | 7.472 |
| Annotated 15 | Raw VLM | 100.0% | 6.7% | 388.887 | 55.000 |
| Annotated 15 | Geometry local fit | 0.0% | 0.0% | — | — |

These metrics are reported for audit completeness, not as paper evidence. The denominator contains structure false positives and suspected Gold misalignment. Calibration changes the seven covered predictions slightly but produces no correct answer; it has no demonstrated value in this phase.

## Failure attribution

### X grounding: dominant coverage blocker

All 15 annotated stock-chart questions failed conservatively with `x_grounding_unrecoverable` (30 target failures). The questions often request the first or last displayed month, while OCR recovers only interior labels. The current rule requires a direct anchor or two bracketing anchors and correctly refuses to extrapolate beyond them.

This is a concrete next module task: endpoint-aware anchor recovery or tightly bounded extrapolation, tested separately. It should not be enabled by silently guessing an endpoint.

### Structure screening: three false positives

- 8245 is a scatter plot; the line detector followed border-like pixels.
- 1286 contains three PMI lines; the full-corpus scan labeled it as one line and the runtime selected a non-target component.
- 1511 is a filled area chart; the recovered path follows the area boundary rather than a line-series point.

These charts are outside the frozen simple-Line scope and must not be counted as Geometry failures.

### Gold/question alignment: quantitative evaluation blocker

- 1703: Geometry = 237.68 bps, Raw VLM = 240 bps, visible curve ≈ 238–240 bps, Gold = 30 bps.
- 9007: the two visible points differ by roughly 1–2 percentage points; Geometry = 1.16, Raw VLM = 4.5, Gold = 15.
- 3864: direct x grounding is correct; Geometry reads 31.56 (column 33.79), Raw VLM reads 35, Gold is 40 with tolerance 1.
- 1797: the requested 3QFY24 coordinate lies on a steep unmarked segment; Geometry returns a difference of 5.79 while Gold is 1.

The per-sample evidence is preserved in `results/samples.csv`, `audit/geometry_predictions.jsonl`, the 41 overlays, and `results/manual_audit.csv`.

## Relationship to earlier phases

| Scope | Status before/after this run |
|---|---|
| Simple masked Bar | Phase 1: 72 samples, complete module-isolation evidence |
| Simple masked Line | Phase 1/3: 15 original labels; Phase 3 Auto-X 17 points on 8 charts, useful and retained |
| Calibration | Phase 1 test MAE improved from 1.7603 to 1.4216; later Line evidence does not justify enabling it by default |
| Natural no-label Bar | Phase 2: only one accepted question; inconclusive |
| Natural no-label Line | This phase: pipeline closed, but benchmark contamination prevents a comparative claim |
| Multi-line single-axis | Phase 4 exploratory: 65 points/32 charts; 45 automatic successes, 17 ambiguous, 3 localization failures; still needs a clean external validation set |
| Complex Bar | Not yet validated beyond the simple Phase 1 detector |
| Dual-axis Bar+Line | Not attempted and remains out of scope |

The separate completion scan of the 567 FinMME charts omitted by the earlier deep scan added only two masked-value samples (one Bar, one Line). On this two-chart external holdout, Raw MAE was 0.2812 and learned-calibrator MAE was 0.2731; the sample is too small for a calibration conclusion. It does show that the original 87-sample Phase 1 set was not missing a large reservoir of strict easy cases.

## Decision and shortest next plan

1. Freeze this run as a screening/benchmark-quality diagnostic. Do not optimize against its Gold.
2. Manually audit 30–50 natural no-label point-reading charts before running Geometry. Audit may decide inclusion and verify the answer, but runtime still receives no target coordinate or Gold.
3. Prefer exact interior dates/categories first. Add endpoint recovery as a separately measured ablation.
4. Rerun the unchanged Geometry and existing Raw VLM baseline on that set.
5. Only after a clean single-line result, extend to one controlled complexity at a time: multi-line single-axis first, then grouped Bar. Do not jump to dual-axis combinations.

This is the fastest path to a defensible paper result. The Geometry implementation itself is not unusually complicated; the present bottleneck is constructing an evaluation set where chart type, semantic target, and Gold truly describe the same visible quantity.

## Reproducibility and safety

- Command wrapper: `scripts/run_phase5_finmme_natural_line.sh` (`prepare`, `run`, `evaluate`).
- Main implementation: `src/phase5_finmme_natural_line.py`.
- X grounding fixes: full English month names and scalar-exact numeric/date matching in `src/phase3_line_core.py`.
- The run used CPU only (approximately four cores). No GPU process was started, stopped, paused, or modified.
- Phase 1–4 result files were not changed.
