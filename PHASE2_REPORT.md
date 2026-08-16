# Phase 2 — Natural No-label Bar End-to-End Validation

## Executive verdict

- **Geometry recovery usable for this screened subset:** NO / NOT YET SHOWN. Raw Geometry accuracy was 0/1 (0.0%) versus Raw VLM 0/1 (0.0%), a paired delta of +0.0 percentage points. The chart-cluster bootstrap 95% interval is [+0.0, +0.0] points. Fewer than 10 accepted charts is treated as insufficient to claim usability or superiority, regardless of the point estimate.
- **Calibration adds extra value:** NO in this pilot. Calibrated Geometry accuracy was 0/1 (0.0%). Its paired delta versus Raw Geometry was +0.0 points, bootstrap 95% interval [+0.0, +0.0]. This calibration conclusion is separate from the Geometry conclusion.
- No GPU inference was run. The complete, already-existing Qwen2.5-VL-7B direct baseline was reused byte-for-byte.

## Scope and protocol

Phase 1's 87 masked-value samples and all frozen Phase 1 result files were left unchanged. Phase 2 uses naturally unlabeled FinMME charts only: simple vertical Bar charts, one fitted left y-axis, no detected second/additional y-axis, at most one coarse prefilter sloped segment, two to thirty consistent bars, readable x labels, and no non-axis numeric OCR token in the plot interior. Multiple marker series, stacked, grouped/ambiguous, combination, multi-panel, Line, Pie, Area, and other chart types are rejected.

The source pool had 11,099 FinMME questions and 4,446 chart hashes. Existing Phase 1 screening identified 537 charts with no geometry-linked data label; the stricter Phase 2 question/structure prefilter produced 99 evaluated question candidates. The final clean subset contains **1 question(s) on 1 distinct chart(s)** ({'numerical': 1}; operations {'difference': 1}). It was not padded to a requested count.

### Leakage boundary

The detector receives only image, question, options, question type, chart hash, and non-Gold structure prefilter fields. It does not receive Gold, tolerance, hidden value-label boxes, or manual target coordinates. `Question → semantic target` matches normalized question text to CPU-OCR x-axis labels; all bars are detected first and the matched label selects a bar by its automatically assigned x-position cell. Gold and official tolerance are joined by `sample_id` only after semantic targeting, bar selection, y-axis interpolation, calibration, and Python reasoning have completed. Phase 1/Phase 2 chart overlap is **0**.

## Results

| System | Correct / N | Accuracy | Numerical MAE | Numerical median AE |
|---|---:|---:|---:|---:|
| Raw VLM (Qwen2.5-VL-7B Direct) | 0 / 1 | 0.0% | 18999975.000 | 18999975.000 |
| Geometry without Calibration | 0 / 1 | 0.0% | 5.135 | 5.135 |
| Geometry + Phase 1 Calibration | 0 / 1 | 0.0% | 5.137 | 5.137 |

Official absolute Numerical tolerance is used per record. Single-choice rows, when retained, require the official option letter. For direct reads Geometry returns the interpolated value; for difference and growth-rate questions the target values come only from Geometry and deterministic Python performs the operation.

Exact paired McNemar p-values are 1.0000 for Raw Geometry versus Raw VLM and 1.0000 for Calibrated versus Raw Geometry. With one chart and no discordant correctness pair, both the exact test and the degenerate `[0, 0]` bootstrap interval are non-informative; no significance or superiority claim is possible.

## Target / Geometry / reasoning audit

- Accepted target matches require an exact/compact or high-confidence fuzzy OCR–question match (score at least 0.78); ambiguous direct targets are rejected.
- All accepted rows store the semantic target, every detected x label, selected bar ID/bbox, y ticks `(value, pixel_y)`, bar top `(x, y)`, per-target raw/calibrated values, final predictions, Gold, tolerance, correctness, and failure attribution.
- Failure attribution counts: {'geometry_or_semantic_decomposition': 1}. A wrong direct read with a high-confidence target is attributed to Geometry. For derived answers, component-level Gold is unavailable, so a wrong answer is conservatively labelled `geometry_or_semantic_decomposition` rather than pretending the stage is identifiable.
- 41 per-sample overlays were generated. Red denotes selected bars/tops, gray other detected bars, magenta axis ticks, and blue the inferred plot box. Overlays do not feed back into targeting. 40 representative overlays were visually reviewed after predictions were frozen.
- Accepted-sample manual review: `{"9346": {"all_target_bars_detected": true, "geometry_visually_consistent": true, "manual_failure_attribution": "possible_official_gold_chart_mismatch", "manual_no_direct_value_label": true, "manual_reasoning_correct": true, "manual_simple_single_axis_bar": true, "manual_target_correct": true, "notes": "The visible bar tops interpolate to approximately 11.7 (2022) and 31.6 (2024F), so the deterministic difference is approximately 19.9. The official Gold is 25 with tolerance 2, which is not visually supported by these bars. Official scoring remains unchanged."}}`. Official correctness is not changed by this audit. In particular, a suspected Gold/chart mismatch remains officially scored as wrong.

## Screening audit

Rejection counts: `{"bar_count_out_of_scope": 55, "clean_natural_no_label_bar_and_supported_target": 1, "direct_data_value_or_numeric_annotation_present": 15, "multiple_time_categories_assigned_to_one_bar": 1, "multiple_y_axes_or_panels": 7, "possible_marker_series_or_combination": 4, "semantic_target_not_found": 2, "x_axis_labels_unreadable_or_grouped_bars": 14}`.

The dominant rejects are structurally out of scope, unreadable/grouped x-label layouts, detected numeric annotations, or failure to find a unique semantic target. These rejects are part of the audit trail in `results/phase2_screening.csv`; they are not silently discarded.

## Statistical interpretation and risk scan

Confidence level: **CAUTION**. The study is a deliberately high-precision, selected pilot rather than a random FinMME sample. Chart-cluster bootstrap respects the chart as the sampling unit, and one accepted question per chart prevents repeated-chart pseudo-replication. The main risks are selection bias/Berkson-type conditioning on easy charts and garden-of-forking-paths risk from exploratory thresholds. No causal claim is made. Simpson reversal, ecological inference, base-rate claims, regression-to-mean claims, survivorship claims, look-elsewhere significance claims, collider adjustment, reverse causality, and correlation-as-causation are not applicable to the stated paired predictive comparison; all 11 protocol checks were considered.

## Reproducibility and Material Passport

- **Material identity:** FinMME released records and already-cached natural chart images; no source dataset was downloaded or modified. Qwen2.5-VL-7B Direct prediction SHA-256: `26c9a75df82631dd45a1b0c012debde0a9d8ae261d6bcf9ded3d3c25d9cc42ce`. Phase 1 calibrator SHA-256: `33b6e7300543dc86cd4ac015782e180ee2d1771821f73c70c70737dd415e41f9`.
- **Material mode:** source records/images and Raw VLM predictions are read-only inputs. Derived OCR caches, sample tables, overlays, metrics, and this report are new Phase 2 outputs.
- **Transformations:** CPU EasyOCR → linear left-axis fit → quantized-color bar components → x-label/bar association → question-label matching → pixel/value interpolation → optional frozen Phase 1 error correction → deterministic Python QA → official scoring.
- **Lineage:** `phase2_audit/accepted_samples.jsonl` SHA-256 `069fcec4ff781c9234bac08f89a2ef40f84c8557ddf041c2e949d407ca361fd9`; sample CSV SHA-256 `d79bbbc8dd03d4feaea2b0a8f1bc89d76fb17519a1c1e95728675120e895536c`. Each row carries `chart_id`, `sample_id`, and input image path.
- **Environment:** Python 3.10.20, OpenCV 5.0.0, NumPy 2.2.6, EasyOCR 1.7.2; OCR runs with `CUDA_VISIBLE_DEVICES` empty and six CPU threads.
- **Determinism:** downstream evaluation and bootstrap are deterministic at seed 20260816. OCR is environment-sensitive; its cache is append-only and retained. A full rerun should compare row/column structure and metrics, not wall time.

## Artifacts

- `results/phase2_sample_audit.csv` — sample-level complete audit
- `results/phase2_screening.csv` — accepted/rejected candidates and reasons
- `results/phase2_metrics.csv` — overall and stratified system metrics
- `results/phase2_paired_comparisons.csv` — paired bootstrap and exact McNemar results
- `results/phase2_failure_audit.csv` — target/geometry/reasoning attribution
- `results/phase2_manual_audit.csv` — post-prediction visual review and suspected annotation issues
- `results/phase2_chart_split.csv` — Phase 2 held-out chart list and Phase 1 overlap check
- `phase2_audit/accepted_samples.jsonl` — nested machine-readable audit
- `phase2_audit/chart_analysis_cache.jsonl` — recoverable CPU OCR/geometry cache
- `phase2_debug_overlays/` — visual audit overlays
