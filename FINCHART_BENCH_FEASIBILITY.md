# FinChart-Bench Feasibility Pre-Audit (Phase 4, Part K)

Date: 2026-09-03. Scope: metadata + 47-image sample audit only. FinChart-Bench was NOT present on the server; HuggingFace is unreachable from the server, but GitHub (Tizzzzy/FinChart-Bench) is reachable, and only metadata plus a small deterministic image sample (every ~24th QA image, 48 names, 47 downloaded) were fetched into `phase4_audit/finchart_bench/`. No full dataset migration was performed.

## Dataset facts (from repo metadata)

- 1,200 financial chart images (2015-2024); 7,016 questions total: TF 2,384 / MC 2,350 / QA 2,285 rows.
- QA split: all 2,285 answers are numeric; each row carries a `reasoning` field with intermediate approximate values (e.g. "approximately $710M"), which is component-level evidence usable for partial-credit audit of a geometry pipeline.
- QA question shapes: 1,498 direct reads ("What is/was the ..."), 419 difference/derived questions.
- No official tolerance field; answers are approximate by construction ("approximately"), so any end-to-end evaluation must define its own tolerance convention.
- Naming quirk: `QA_data.json` references image names with per-question suffixes (e.g. `..._q1.jpg`) while `QA_images/` holds 1,154 base files; a base-name mapping is required during migration.

## Sample image audit (47 QA images through the Phase 4 CPU structure scan)

| Outcome | Images |
|---|---:|
| no_linear_left_y_axis (strict axis_localizer) | 27 |
| multiple_y_axes_or_panels_detected | 7 |
| axis recoverable, no usable value labels | 9 |
| axis recoverable, has value labels | 4 (viable_single_line 1, viable_multi_line 1, other 2 rejected for anchors/series) |

- Charts with OCR-visible data value labels: 4/47 (~9%). FinChart-Bench is effectively a natural no-label dataset.
- Detected line-series count on axis-recoverable images: 0 series 5, 1 series 4, 2 series 2, 3 series 1, 8 series 1 — multi-line exists but the strict single-linear-axis gate (27/47 rejections) is the dominant blocker, followed by multi-axis/panel layouts (7/47).

## Feasibility answers

1. **Line / multi-line quantity**: present but modest under the current strict axis gate; roughly a quarter of sampled images are axis-recoverable line charts, and only a minority of those are multi-line.
2. **Natural no-value-label charts**: yes, ~91% of sampled images lack OCR-visible value labels. Masked-value pseudo-Gold construction (the Phase 1-4 protocol) does NOT transfer to this dataset.
3. **QA suitability for end-to-end evaluation**: good. QA answers are numeric Gold and the reasoning field supplies intermediate values; the dataset directly supports `question -> geometry -> numeric answer` evaluation without any masking.
4. **Underlying values / verifiable numbers**: answers are approximate (no tolerance, "approximately" wording in reasoning). Verifiable with a self-defined tolerance (e.g. 5% of axis span or official-style relative tolerance), but not exact.

## Recommendation

Do not migrate FinChart-Bench for masked-value module evaluation. If a Phase 5 goes toward end-to-end QA, FinChart-Bench QA is a reasonable target with three preconditions: (a) download the full QA_images set (~1,154 files, GitHub reachable); (b) invest in relaxing `axis_localizer` (the 57% axis rejection rate would otherwise cap coverage); (c) define an approximate-answer tolerance convention before any accuracy claim. This phase (Phase 4) does not run FinChart-Bench.

## Artifacts

- `phase4_audit/finchart_bench/{QA,MC,TF}_data.json`, `README.md` (upstream metadata)
- `phase4_audit/finchart_bench/sample_images/` (47 images)
- `phase4_audit/finchart_bench/sample_scan.json` (per-image structure audit rows)
