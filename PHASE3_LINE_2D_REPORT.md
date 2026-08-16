# Phase 3: Line Chart X-axis Grounding + 2D Geometry Validation

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-08-16
- Verification Status: VERIFIED
- Version Label: phase3_line_2d_v1
- Experiment ID: FGMVP-P3-LINE-2D-20260816
- Overall Confidence: CAUTION — deterministic and fully audited, but only 17 labels from 8 strict charts

## Executive verdict

Within the deliberately narrow scope of simple, single-series, single-y-axis Line charts, x-axis semantic grounding is usable. The automatic path recovered all 17 targets (13 direct OCR anchors and 4 bracketed interpolations), with mean x error 0.44 px and maximum 2.75 px relative to the construction-time semantic target. All 17 target/series choices passed overlay review.

The old Phase 1 value-label-center path is not a valid empirical upper bound on this subset. Value labels are often placed beside a point, not centered on it: their center x was displaced by 36.62 px on average (median 23.5 px; maximum 93 px). Consequently, Auto-X + local fitting had **lower**, not higher, error than the nominal Oracle-X proxy: MAE 0.1747 versus 1.5624. The requested `Oracle → Auto` “loss” is therefore **−1.3877 MAE** (an 88.8% relative improvement), not a loss.

Local fitting has incremental value, but it is a robustness mechanism rather than a universal win. It improved 8 paired samples, worsened 8, and recovered the one point missed by the narrow-column read. It reduced median absolute error from 0.1308 to 0.0745, reduced full-sample MAE from 0.1910 to 0.1747, and raised coverage from 94.1% to 100%. The chart-cluster bootstrap interval for the paired column-minus-local MAE difference crosses zero, so the mean benefit is not conclusive at 8 charts.

Calibration should not be enabled by default from this phase. On only 4 eligible samples from 3 charts unseen to the Phase 1 calibrator training set, the global median-bias baseline was best (MAE 0.1219), the learned calibrator was second (0.1608), and mean-bias baselines were harmful. This is too small to select a permanent correction rule.

## Scope and preservation

- Phase 1 and Phase 2 outputs were not modified.
- All 12 Phase 1 frozen hashes passed before and after Phase 3.
- The Phase 2 validator passed before and after Phase 3.
- Phase 3 used CPU OCR and OpenCV only; `CUDA_VISIBLE_DEVICES` was empty.
- Two unrelated GPU compute processes remained active throughout. No other process was killed, paused, or modified.
- No VLM was needed, so a GPU waiter was not started.
- Excluded by design: dual-axis, Area, stacked, multi-line, Agent, RL, and LoRA.

## Candidate accounting

The existing scan contained 19 charts with at least one apparent Line-linked data label. A targeted CPU scan exhausted the remaining 51 Line-priority FinMME charts and found one additional viable chart:

| Additional-scan outcome | Charts |
|---|---:|
| Viable Line-linked label | 1 |
| No geometry-linked Line label | 30 |
| No recoverable linear left y-axis | 18 |
| Multiple y-axes/panels | 2 |

The resulting 20-chart structure audit retained 8 simple single-series charts and excluded 12:

| Structure decision | Charts |
|---|---:|
| Included: simple single-series, single-y-axis Line | 8 |
| Multiple Line series | 8 |
| Line plus reference series | 1 |
| Dual-axis/multiple series | 1 |
| Complex annotated multi-series | 1 |
| Bar misclassified as Line | 1 |

Across the 8 included charts, 17 labels were accepted and 3 were rejected because no reliable semantic x target could be constructed. The experiment did not relax chart complexity to increase sample count.

## Mask and no-leak contract

Each accepted sample preserves the original numeric label as pseudo-Gold and masks only that text. A first overlay audit found that generic polygon inpainting could touch the edge of a thick line in five cases. The Phase 3 mask was therefore corrected before final measurement: a 5×5 dilation of the detected series pixels is subtracted from the text mask, and every row records `mask_overlaps_protected_series_pixels=0`.

The automatic parser function has the fixed signature:

```text
auto_geometry_pipeline(reader, masked_bgr, question)
```

It does not accept pseudo-Gold, the masked value-label bbox, or the value-label center x. The hidden label metadata exists only under `construction_only` for masking, the historical Oracle-X proxy, and audit. Static AST validation and all sample contracts reported zero hidden-input violations.

## Implemented 2D path

1. Run OCR on the masked chart and localize the linear left y-axis.
2. Identify the x-axis band from the zero/baseline pixel and y-axis extent.
3. Save every usable x anchor as `(label, bbox, center_x, confidence, source, scalar)`.
4. If horizontal OCR does not recover at least three temporal anchors, rotate the x-axis band both ways and map OCR boxes back to original coordinates. This recovers vertical year/month labels.
5. Parse the semantic target from the question.
6. Use an exact normalized label/canonical match when available; otherwise use piecewise linear interpolation between two bracketing year/date/quarter/numeric anchors. Never extrapolate outside the anchors.
7. Detect the dominant single Line component and save its series ID, BGR color, bbox, coverage, and candidate diagnostics.
8. Compare a narrow-column read with a robust local fit over `target_x ± window`. The fit uses per-column median pixels, iteratively reweighted regression, and two-sided piecewise fits when both sides are available.
9. Convert the final subpixel y through the OCR-derived y-axis mapping.

## Main geometry results

All errors are against the masked label's pseudo-Gold. “Large error” means axis-normalized error greater than 2%.

| Method | Coverage | MAE | Median AE | Mean axis-normalized error | Large errors |
|---|---:|---:|---:|---:|---:|
| Oracle-X: old value-label-center local fit | 17/17 (100%) | 1.5624 | 1.2149 | 4.3296% | 9 |
| Auto-X: narrow column | 16/17 (94.1%) | 0.1910 | 0.1308 | 0.5135% | 0 |
| Auto-X + robust local fitting | 17/17 (100%) | **0.1747** | **0.0745** | **0.5004%** | **0** |

Chart-cluster bootstrap comparisons (3,000 resamples; descriptive because there are only 8 charts):

| Comparison | Observed left MAE − right MAE | 95% cluster-bootstrap interval | Interpretation |
|---|---:|---:|---|
| Auto column − Oracle-X | −1.2817 | [−1.8437, −0.3826] | Auto column is better on the paired subset |
| Auto local − Oracle-X | −1.3877 | [−1.9494, −0.4290] | Auto local is better; nominal Oracle is not an upper bound |
| Auto column − Auto local | +0.0334 | [−0.0187, +0.0948] | Local tends to help, but the interval crosses zero |

## X-axis grounding results

| Grounding type | Samples | Charts | Mean x error | Max x error | Local-fit MAE | Mean axis-normalized error |
|---|---:|---:|---:|---:|---:|---:|
| Direct anchor | 13 | 8 | 0.00 px | 0.00 px | 0.2130 | 0.6105% |
| Piecewise interpolation | 4 | 2 | 1.875 px | 2.75 px | 0.0503 | 0.1426% |
| Overall | 17 | 8 | 0.441 px | 2.75 px | 0.1747 | 0.5004% |

The direct-anchor x error is exactly zero by construction: the semantic target is a displayed OCR anchor, and runtime re-detects the same anchor on the masked chart. The meaningful checks are that the masked runtime independently recovers the anchor, the selected Line point matches the intended pseudo-Gold, and no hidden label center is passed. All three checks passed on 13/13 direct cases. Interpolation supplies the stronger 2D test because the target label is absent; those four cases were recovered within 2.75 px.

## Does local fitting help thick or ambiguous lines?

Partly:

- 8/16 paired samples improved and 8/16 worsened.
- One additional sample was recoverable only with a windowed fit, raising coverage to 100%.
- Median AE fell 43.0% (0.1308 → 0.0745).
- Full-sample MAE fell 8.5% (0.1910 → 0.1747).
- The largest gain was a right endpoint: absolute error improved by 0.3589.
- The largest deterioration was only 0.0606.
- Neither Auto method had an error above 2% of the y-axis span.

Thus local fitting is useful for coverage, thick strokes, and subpixel centering, but it should retain confidence/endpoint diagnostics. It is not justified as an unconditional replacement based on 8 charts.

## Failure attribution

At an audit threshold of 1% of y-axis span:

| Attribution | Samples |
|---|---:|
| Within expected error | 15 |
| Line fitting / local point placement | 2 |
| X grounding | 0 |
| Series matching | 0 |
| Y-axis mapping | 0 |

All 17 overlays were manually reviewed: target grounding was correct in 17/17 and the intended single series was selected in 17/17. The two errors above 1% were small local line-placement errors. Endpoint underfit remained visible in two other overlays but fell below the 1% threshold after the line-protected masking correction. This subset cannot establish multi-series matching quality because multi-line charts were intentionally excluded.

## Calibration and constant-bias controls

Calibration evaluation excludes every Phase 3 chart present in the Phase 1 calibrator's training split. That leaves only 4 labels from 3 charts, so the numbers are diagnostic only.

| Correction on Auto-X + local | MAE | Median AE | Mean axis-normalized error |
|---|---:|---:|---:|
| No correction | 0.2156 | 0.2190 | 0.4001% |
| Global mean bias (+0.6626) | 0.4470 | 0.4436 | 1.9140% |
| Line-type mean bias (+0.9785) | 0.7630 | 0.7595 | 3.0175% |
| Global median bias (+0.1128) | **0.1219** | **0.1061** | **0.2166%** |
| Line-type median bias (+0.3669) | 0.1513 | 0.1479 | 0.8812% |
| Learned Phase 1 calibrator | 0.1608 | 0.1658 | 0.2388% |

The learned calibrator is not the best control, and mean biases transfer poorly. The global median bias is best here, but 4 samples are insufficient for model selection. Phase 3 therefore does **not** retain any calibration rule by default; Raw Auto-X + local remains the primary result.

## Statistical integrity and fallacy scan

- No p-values or confirmatory significance claims are made.
- Chart is the resampling unit, so repeated labels from one chart are not treated as independent in uncertainty estimation.
- Train/test splits are chart-disjoint with zero overlap.
- Calibration is evaluated only on charts unseen to its Phase 1 training set.
- Multiple method comparisons are exploratory; no multiplicity-adjusted discovery claim is made.

Fallacy coverage: **11/11 checked**.

| Fallacy | Status | Phase 3 assessment |
|---|---|---|
| Simpson's paradox | NOTE | Aggregate and chart-cluster estimates are both retained; 8 charts are too few for a stable subgroup claim |
| Ecological fallacy | Not detected | Claims are limited to this chart/sample subset |
| Berkson's paradox | CAUTION | Strict eligibility selection limits generalization to easy single-series charts |
| Collider bias | Not applicable | No causal/control-variable model |
| Base-rate neglect | Not applicable | No diagnostic conditional-probability claim |
| Regression to the mean | Not applicable | No extreme-score pre/post design |
| Survivorship bias | CAUTION | Accepted/rejected counts are fully reported; results still condition on recoverable labelled charts |
| Look-elsewhere effect | NOTE | Three primary methods plus exploratory correction baselines are all reported, not selectively filtered |
| Garden of forking paths | CAUTION | This is an exploratory MVP; the line-protection correction and all audit decisions are documented |
| Correlation ≠ causation | Not detected | No causal language or intervention claim |
| Reverse causality | Not applicable | No directional causal relationship |

## Reproducibility and audit

The final CPU construction and analysis were rerun with unchanged code and inputs. SHA-256 comparison covered 121 artifacts: the sample JSONL, all Phase 3 CSV/JSON outputs, 17 masked images, 51 full overlays, and 40 published overlays. Result: `REPRODUCIBLE`, with exact hash match, zero missing, zero extra, and zero mismatched artifacts.

Validation also confirmed:

- 12/12 Phase 1 frozen hashes match.
- Phase 2 validator status is `ok`.
- 17 samples from 8 charts.
- 13 direct and 4 interpolated targets.
- 51 full overlays and 40 repository overlays.
- Zero chart split overlap.
- Zero hidden parser inputs.
- Stored MAEs exactly recompute from the sample-level CSV.

## Required questions answered

**Can x-axis labels stably replace the original value-label center x?**
Yes for this strict subset. Runtime semantic grounding recovered 17/17 targets, including 4 targets obtained only by interpolation. The original value-label center was far less aligned with the actual semantic x position.

**How much does Auto-X lose relative to Oracle-X?**
It does not lose here. Auto-X + local improves MAE by 1.3877 (88.8%) relative to the old label-center proxy. This reveals that the requested Oracle proxy is not a true point-x oracle.

**Does local fitting help under blur/thick strokes?**
It improves median accuracy and coverage, and slightly improves overall MAE, but paired mean uncertainty crosses zero. Keep it with confidence/endpoint diagnostics rather than assuming it always wins.

**Where do Line failures now come from?**
On this subset, the two errors above 1% axis span come from line fitting/local point placement. No x-grounding, series-selection, or y-axis-mapping failure survived audit. Series matching was not stress-tested because multi-line charts were excluded.

**Is expansion to multi-line single-axis worthwhile?**
Yes, as the next isolated experiment. Single-series x grounding is strong enough to justify adding `question target series → legend/color → series component` with explicit ambiguity abstention. The present result must not be used as evidence that multi-series matching already works.

## Artifact map

- `src/phase3_line_core.py` — x anchors, semantic grounding/interpolation, series detection, local fitting, overlays.
- `src/run_phase3_line_2d.py` — additional scan, construction, experiment, metrics, calibration controls.
- `src/validate_phase3.py` — frozen-result, no-leak, split, metric, and overlay checks.
- `src/verify_phase3_reproducibility.py` — deterministic 121-artifact hash comparison.
- `config/phase3_line_chart_decisions.csv` — all 20 structure decisions and reasons.
- `results/phase3_line_samples.csv` — sample-level question, anchors, target x, series, fit, ticks, values, Gold, and errors.
- `results/phase3_method_metrics.csv` — primary method metrics.
- `results/phase3_pairwise_metrics.csv` — chart-cluster bootstrap comparisons.
- `results/phase3_local_fitting_effects.csv` — per-sample local-fit wins/losses/coverage recovery.
- `results/phase3_calibration_bias_metrics.csv` — raw, global/type constants, and learned calibrator.
- `results/phase3_failure_attribution.csv` and `results/phase3_visual_audit.csv` — programmatic and overlay audits.
- `results/phase3_chart_screening.csv` and `results/phase3_chart_split.csv` — selection and leakage audit.
- `results/phase3_reproducibility.json` — exact deterministic rerun verdict.
- `phase3_debug_overlays/` — 51 full local artifacts retained on the server.
- `examples/phase3_debug_overlays/` — 40 curated repository overlays.
- `logs/phase3_execution.log` and `logs/phase3_gpu_safety.log` — execution and GPU-safety summaries.
