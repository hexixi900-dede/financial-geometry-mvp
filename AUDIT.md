# Audit guide

This Git snapshot packages a completed CPU experiment without rerunning it. The source experiment outputs remain in place; repository preparation only adds publication metadata, makes runtime paths configurable, and copies a representative overlay subset.

## Frozen-result integrity

`audit/frozen_results.sha256` records SHA-256 digests captured before repository preparation for `MVP_REPORT.md` and every final CSV. The same files are checked again after preparation. The project-specific source code and README may differ from the earlier execution snapshot only to remove absolute runtime dependencies and document the publication boundary.

## Sample-level traceability

- `results/geometry_samples.csv` contains one row per accepted chart with y-axis tick values and pixel positions, bar bbox or line point, raw geometry value, pseudo-Gold, and absolute/relative/axis-normalized error.
- `results/calibrated_predictions.csv` adds the chart-level split, predicted error, corrected value, and Raw/Calibrated errors.
- `results/chart_split.csv` is the authoritative train/test assignment. The split key is `chart_id`, and overlap is zero.
- `config/calibrator_model.json` contains the fitted feature scaling and model coefficients.

## Representative overlays

The repository includes 40 of the 87 overlays. Selection is deterministic and favors audit coverage rather than visual appearance:

1. include every Line sample because Line evidence is scarce;
2. include every test-split Bar sample;
3. fill the remaining budget with train-split Bar samples spanning raw axis-normalized-error quantiles.

`examples/debug_overlays/selection_manifest.csv` records the reason and image SHA-256 for each selected sample. The selection process does not alter any metric or result table.

## Exclusions

The complete FinMME dataset, preprocessed source image cache, OCR/model weights, all masked images, the full overlay directory, raw JSONL scans, logs, virtual environments, archives, and local credentials are excluded by `.gitignore`. No upstream dataset or model artifact is vendored.

## Method provenance

See `audit/chartagent_method_notes.md`. The project independently implements the small CPU Geometry subset described by the ChartAgent appendix and does not reproduce the complete Agent.
