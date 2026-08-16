#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd -- "$script_dir/.." && pwd)}"

: "${FINMME_ROOT:?Set FINMME_ROOT to an existing FinMME workspace.}"
python_bin="${PYTHON_BIN:-$FINMME_ROOT/phase6_chart_structure_transfer_rev2_screen_v1/.venv_ocr/bin/python}"
ocr_model_dir="${OCR_MODEL_DIR:-$FINMME_ROOT/phase6_chart_structure_transfer_rev2_screen_v1/.ocr_models}"
records_path="${FINMME_RECORDS_PATH:-$FINMME_ROOT/outputs/subsets/qwen25vl7b_official_protocol_full_records.jsonl}"

for required_path in "$python_bin" "$ocr_model_dir" "$records_path"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required input does not exist: $required_path" >&2
    exit 2
  fi
done

cd "$project_root"
mkdir -p audit debug_overlays logs masked_images results
export PYTHONPATH="$project_root/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export CUDA_VISIBLE_DEVICES=""

"$python_bin" src/prepare_manifest.py \
  --records "$records_path" \
  --project "$project_root" \
  > logs/prepare_manifest.log 2>&1

"$python_bin" src/scan_candidates.py \
  --manifest results/finmme_chart_manifest.csv \
  --output audit/candidate_scan_v3.jsonl \
  --model-dir "$ocr_model_dir" \
  --max-new-charts 1496 \
  --target-viable 240 \
  --progress-every 10 \
  >> logs/candidate_scan_v3_full.log 2>&1

"$python_bin" src/run_geometry_samples.py \
  --candidate-scan audit/candidate_scan_v3.jsonl \
  --project "$project_root" \
  --model-dir "$ocr_model_dir" \
  --max-samples 240 \
  --max-new-charts 400 \
  --progress-every 10 \
  >> logs/geometry_full.log 2>&1

"$python_bin" src/calibrate_geometry.py \
  --samples audit/geometry_samples.jsonl \
  --project "$project_root" \
  --seed 20260816 \
  --test-fraction 0.25 \
  > logs/calibration_full.log 2>&1

"$python_bin" src/generate_mvp_report.py \
  --project "$project_root" \
  > logs/report_generation.log 2>&1

"$python_bin" tests/smoke_test.py \
  > logs/smoke_test.log 2>&1

"$python_bin" tests/validate_outputs.py \
  --project "$project_root" \
  > logs/validation.log 2>&1
