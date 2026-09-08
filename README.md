# Financial Geometry MVP

## Current implementation (2026-09-08)

**Start with [CURRENT_STATUS.md](CURRENT_STATUS.md)** for the current method, code map, execution instructions and experiment status. The latest implementation combines image-based VLM target planning, deterministic geometry/arithmetic, and an image-based VLM final answer. It covers single-choice, multiple-choice and numerical questions within the supported chart cohort.

**The new unified experiment has not produced results yet: it is waiting for GPU availability.** The reports and numerical results below describe historical phases, not the new pipeline's accuracy.


CPU-only minimum validation of calibrated geometry reasoning on existing FinMME charts. This repository is an auditable, lightweight snapshot of the completed experiment; the dataset, OCR models, masked-image corpus, and full debug-overlay corpus are intentionally excluded.

## Result snapshot

- 87 independent charts: 72 Bar and 15 Line.
- Chart-level split: 65 train / 22 test, with zero chart overlap.
- Test Raw Geometry MAE: 1.7602704404; mean axis-normalized error: 1.3941%.
- Test Calibrated Geometry MAE: 1.4215724658; mean axis-normalized error: 1.2387%.
- The full interpretation, bootstrap uncertainty, limitations, and separate Geometry/Calibration verdicts are in [`MVP_REPORT.md`](MVP_REPORT.md).

The checked-in CSV files and report are frozen experiment outputs. Repository preparation did not rerun the experiment or change those files.

## Phase 2: natural no-label Bar QA

Phase 2 is an independent held-out diagnostic on charts that naturally omit data-value labels. It tests the complete path `Question → semantic target → detected x label/bar → y-axis geometry → answer` against an existing Qwen2.5-VL-7B Direct baseline. It does not modify or retrain the Phase 1 geometry/calibration experiment.

The strict FinMME screen evaluated 99 candidates but retained only one clean question/chart, so the comparison is inconclusive and must not be presented as evidence that Geometry beats Raw VLM. All three systems are officially 0/1; post-prediction visual audit flags a likely mismatch between the only chart and its released Gold. The full rejection accounting is retained rather than padding the subset.

- `PHASE2_REPORT.md`: separate Geometry and Calibration verdicts, uncertainty, screening counts, and Material Passport.
- `src/phase2_natural_bar.py`: CPU OCR, all-bar detection, automatic semantic targeting, geometry, frozen calibration, and Python QA.
- `src/evaluate_phase2.py`: official tolerance/choice scoring, chart-cluster bootstrap, exact paired comparison, and report generation.
- `results/phase2_*.csv`: sample, screening, split, metric, paired-comparison, and failure-attribution tables.
- `examples/phase2_debug_overlays/`: 40 representative accepted/rejected overlays and their selection manifest.

The Phase 2 targeting and Geometry stages never receive Gold, tolerance, hidden value-label boxes, or manual target coordinates. Gold is joined only after predictions exist. Existing complete Raw VLM predictions are reused by SHA-256, so Phase 2 does not require a new GPU run.

## Phase 3: Line x-axis grounding and 2D geometry

Phase 3 is an isolated CPU experiment on strict, simple, single-series, single-y-axis Line charts. It masks only explicit data-value text while protecting the detected line pixels, then compares the historical value-label-center proxy (`Oracle-X`) with two no-leak automatic paths:

1. `Question → OCR x-axis anchor → target_x → narrow-column point → y-axis value`
2. `Question → OCR x-axis anchor/interpolation → target_x ± window → robust local line fit → y-axis value`

The automatic parser accepts only the masked chart and question. Pseudo-Gold, the masked label box, and its center x remain in a construction-only audit namespace. Rotated x-band OCR handles vertical years/months; unbracketed targets return `x_grounding_unrecoverable` instead of extrapolating.

The strict result contains 17 labels from 8 charts (13 direct anchors, 4 piecewise interpolations) and 51 full local overlays. The repository publishes 40 representative overlays while excluding the masked-image corpus and full overlay directory. See `PHASE3_LINE_2D_REPORT.md` for metrics, failure attribution, calibration/bias controls, uncertainty, and the multi-line decision.

- `src/phase3_line_core.py`: semantic x grounding, rotated OCR, dominant-series extraction, column read, and robust local fit.
- `src/run_phase3_line_2d.py`: resumable scan, construction, no-leak execution, paired metrics, and calibration/bias controls.
- `src/validate_phase3.py`: frozen Phase 1/2 checks, no-leak AST check, split/output integrity, and metric recomputation.
- `results/phase3_*.csv`: screening, sample audit, chart split, metrics, pairwise bootstrap, local-fit effects, calibration, and failures.
- `examples/phase3_debug_overlays/`: 40 representative Oracle/column/local overlays plus selection manifest.

## Fixed scope

- Simple, single-y-axis vertical Bar and Line charts only.
- Mask an explicit data-value label, retain it as pseudo-Gold, and recover the value from geometry.
- No complete QA, Agent, RL, LoRA, multi-axis, Pie, Area, Boxplot, or GPU/VLM work.

## Repository map

- `src/`: Geometry extraction, masking, reporting, and calibration code.
- `scripts/`: CPU pipeline and representative-overlay curation utilities.
- `config/`: frozen run settings and fitted calibrator parameters.
- `results/`: sample-level outputs, chart split, aggregate metrics, CV, and bootstrap tables.
- `examples/debug_overlays/`: 40 deterministically selected overlays plus a selection manifest.
- `audit/`: frozen output hashes and method-source notes.
- `tests/`: smoke and output-integrity checks.

## External inputs

FinMME data and the existing OCR environment are not included. Point the pipeline at a local FinMME workspace using environment variables; no source file depends on a server-specific absolute path.

```bash
cp .env.example .env
# Edit .env for the local machine, then:
set -a
source .env
set +a
bash scripts/run_cpu_pipeline.sh
```

For the Phase 2 CPU path, point `RAW_VLM_PATH` at an existing complete direct-baseline JSONL (or use the default under `FINMME_ROOT`) and run:

```bash
bash scripts/run_phase2_cpu.sh
```

Phase 3 uses explicit CLI arguments so it has no server-specific path dependency:

```bash
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=""
python src/run_phase3_line_2d.py \
  --project . \
  --manifest results/finmme_chart_manifest.csv \
  --candidate-scan audit/candidate_scan_v3.jsonl \
  --chart-decisions config/phase3_line_chart_decisions.csv \
  --model-dir /path/to/easyocr/models \
  --stage construct
python src/run_phase3_line_2d.py \
  --project . \
  --manifest results/finmme_chart_manifest.csv \
  --candidate-scan audit/candidate_scan_v3.jsonl \
  --chart-decisions config/phase3_line_chart_decisions.csv \
  --model-dir /path/to/easyocr/models \
  --stage analyze
python src/validate_phase3.py --project .
```

Required setting:

- `FINMME_ROOT`: an existing FinMME reproduction workspace.

Optional overrides are documented in `.env.example`. The CPU runner validates each external input before starting and sets `CUDA_VISIBLE_DEVICES=""`.

The stages are resumable JSONL pipelines. `scan_candidates.py` skips chart IDs already present in its output; `run_geometry_samples.py` skips charts already attempted. Reproduction is not needed to audit the frozen results.

## Gold-isolation invariant

Original OCR text and numeric pseudo-Gold are used only to construct and audit the mask. After masking, the Geometry path receives a target locator whose `text` is empty and whose `numeric_value` is `None`. Raw geometry is computed before pseudo-Gold enters error calculation. Every accepted row records `geometry_gold_access=False`.

## Data and publication policy

The repository excludes the complete FinMME dataset, source JPEG cache, OCR/model weights, all masked images, the full overlay corpora, logs, temporary scans, archives, and local environment files. Phase 1 and Phase 3 each publish 40 audit overlays; their selections are recorded in the corresponding `examples/*debug_overlays/selection_manifest.csv` files.

FinMME and ChartAgent remain governed by their own upstream terms. The Geometry implementation here is an independent CPU subset based on the ChartAgent paper appendix's behavioral descriptions; no author repository or function-body source was available or copied. See `audit/chartagent_method_notes.md`.
