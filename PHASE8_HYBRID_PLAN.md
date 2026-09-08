# Phase 8 Plan: VLM Semantics + Geometry Evidence

## Correction to Phase 7

Phase 7 used regular expressions and OCR-string matching for semantic planning. It therefore tested `rules → Geometry → Python`, not the intended paper method. Its negative score is retained as an ablation and must not be used as the verdict on the hybrid idea.

## Research question

On FinMME questions whose answers require geometric reading of natural no-value-label, single-y-axis Bar/Line charts, does `VLM semantic planning → Geometry value extraction → Python calculation` outperform a Raw VLM answer?

The VLM may output only chart type, target x/category, target series, operation, and a short non-numeric reason. It may not output a y value or final answer. Gold, tolerance, answer options, hidden data-label boxes, and construction coordinates are absent from planner inputs. Geometry selection does not use options; Python receives them only after values have been measured, to format a single-choice answer.

## Fixed evaluation layers

1. **Frozen paired layer:** the 167 Phase 7 questions. This compares complete implementations on the same questions. Once Geometry implementation bugs are repaired, this is no longer a one-factor semantic-parser ablation; the before-repair run is retained separately.
2. **Broad automatic layer:** every numerical or numeric-option single-choice question associated with an automatically detected, natural no-value-label, single-y-axis Bar/Line candidate. No manual image selection and no Gold-based filtering.
3. **Full FinMME selective system:** use hybrid Geometry only when planner and Geometry both succeed; otherwise retain the cached Raw VLM prediction. Report accuracy over all 11,099 questions.
4. **Visible-value-label stratum:** retain Raw VLM. Geometry is not required to beat direct reading when the answer is printed on the chart.

## Method

1. Prepare the broad cohort and a full-dataset structural/label audit from existing FinMME records and the existing chart scan.
2. Run Qwen2.5-VL-7B with image, caption, and question only. Save strict JSON semantic plans. Do not provide options or Gold.
3. Validate the plan schema and reject unsupported operations rather than inventing targets.
4. For Bars, match each VLM target label to OCR-labelled bars, read bar top, and interpolate through y ticks.
5. For Lines, ground each VLM target label on the x axis, select the VLM-named series through the existing legend/color resolver, and use robust local fitting.
6. Execute direct, absolute-difference, and growth-rate operations in Python. Convert numerical results to choice letters only after Geometry, using the official options.
7. Join Gold/tolerance only during evaluation. Count planner/Geometry failures as wrong in the hard route and use Raw VLM only in the selective fallback route.

## Required comparisons

- Raw VLM.
- Phase 7 rule semantics + Geometry.
- Phase 8 VLM semantics + Geometry.
- Full selective hybrid.
- Bar/Line, numerical/single-choice, direct/difference/growth, label stratum, and failure-stage slices.

## Stop rule

Do not search for more images or add chart types during this phase. First finish the frozen paired layer and the broad automatic layer. Only after those scores exist may a new phase consider multi-axis, stacked charts, Agent, RL, or training.

## Recovered original method and execution corrections

The original Phase 2 user request explicitly states: “VLM 只解析 targets 和 operation；所需数值全部由 Geometry 获取；Python 执行计算。” The same request defines the selective policy: “有直接数字 → VLM；无直接数字 → Geometry”. These requirements predate Phase 7 and are the method contract here. No additional full manuscript was found in the accessible project history.

The complete VLM run finished on all 2,087 candidates. CPU implementation review then identified three concrete defects: string Bar IDs incorrectly cast to integers; structured targets converted back to sentences and reparsed; and full dates discarded by the historical x-anchor parser. A pixel audit also found a frame mistaken for a black data line because the old neutral channel excluded near-black pixels. Phase 8 fixes these in its own adapter and retains the earlier phases unchanged.

The historical scan did not inspect labels on 1,995 charts. Those charts are classified as **unknown**, rather than as no-label. The full audit contains 551 label-detected charts, 1,900 label-not-detected charts, and 1,995 unknown charts. These are automatic screening strata, not human-confirmed ground truth. The 2,087-question cohort is unchanged by this reporting correction.

No Calibrator is trained or selected in Phase 8. The earlier learned and constant-bias experiments remain separate. This phase tests the uncalibrated semantic/measurement/answer path first.
