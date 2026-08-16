# Financial Geometry MVP

CPU-only minimum validation of calibrated geometry reasoning on existing FinMME charts. This repository is an auditable, lightweight snapshot of the completed experiment; the dataset, OCR models, masked-image corpus, and full debug-overlay corpus are intentionally excluded.

## Result snapshot

- 87 independent charts: 72 Bar and 15 Line.
- Chart-level split: 65 train / 22 test, with zero chart overlap.
- Test Raw Geometry MAE: 1.7602704404; mean axis-normalized error: 1.3941%.
- Test Calibrated Geometry MAE: 1.4215724658; mean axis-normalized error: 1.2387%.
- The full interpretation, bootstrap uncertainty, limitations, and separate Geometry/Calibration verdicts are in [`MVP_REPORT.md`](MVP_REPORT.md).

The checked-in CSV files and report are frozen experiment outputs. Repository preparation did not rerun the experiment or change those files.

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

Required setting:

- `FINMME_ROOT`: an existing FinMME reproduction workspace.

Optional overrides are documented in `.env.example`. The CPU runner validates each external input before starting and sets `CUDA_VISIBLE_DEVICES=""`.

The stages are resumable JSONL pipelines. `scan_candidates.py` skips chart IDs already present in its output; `run_geometry_samples.py` skips charts already attempted. Reproduction is not needed to audit the frozen results.

## Gold-isolation invariant

Original OCR text and numeric pseudo-Gold are used only to construct and audit the mask. After masking, the Geometry path receives a target locator whose `text` is empty and whose `numeric_value` is `None`. Raw geometry is computed before pseudo-Gold enters error calculation. Every accepted row records `geometry_gold_access=False`.

## Data and publication policy

The repository excludes the complete FinMME dataset, source JPEG cache, OCR/model weights, all masked images, the full overlay corpus, logs, temporary scans, archives, and local environment files. The 40 published overlays are a stratified audit subset; their selection is recorded in `examples/debug_overlays/selection_manifest.csv`.

FinMME and ChartAgent remain governed by their own upstream terms. The Geometry implementation here is an independent CPU subset based on the ChartAgent paper appendix's behavioral descriptions; no author repository or function-body source was available or copied. See `audit/chartagent_method_notes.md`.
