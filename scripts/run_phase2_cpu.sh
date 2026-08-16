#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd -- "$script_dir/.." && pwd)}"

: "${FINMME_ROOT:?Set FINMME_ROOT to an existing FinMME reproduction workspace.}"
python_bin="${PYTHON_BIN:-$FINMME_ROOT/phase6_chart_structure_transfer_rev2_screen_v1/.venv_ocr/bin/python}"
ocr_model_dir="${OCR_MODEL_DIR:-$FINMME_ROOT/phase6_chart_structure_transfer_rev2_screen_v1/.ocr_models}"
records_path="${FINMME_RECORDS_PATH:-$FINMME_ROOT/outputs/subsets/qwen25vl7b_official_protocol_full_records.jsonl}"
candidate_scan_path="${CANDIDATE_SCAN_PATH:-$project_root/audit/candidate_scan_v3.jsonl}"
raw_vlm_path="${RAW_VLM_PATH:-$FINMME_ROOT/outputs/predictions/qwen25vl7b_direct_full.jsonl}"
calibrator_path="${CALIBRATOR_PATH:-$project_root/config/calibrator_model.json}"

for required_path in \
  "$python_bin" \
  "$ocr_model_dir" \
  "$records_path" \
  "$candidate_scan_path" \
  "$raw_vlm_path" \
  "$calibrator_path"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required input does not exist: $required_path" >&2
    exit 2
  fi
done

cd "$project_root"
mkdir -p logs phase2_audit phase2_debug_overlays results
export PYTHONPATH="$project_root/src"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"
export CUDA_VISIBLE_DEVICES=""

"$python_bin" src/phase2_natural_bar.py \
  --records "$records_path" \
  --candidate-scan "$candidate_scan_path" \
  --raw-vlm "$raw_vlm_path" \
  --calibrator "$calibrator_path" \
  --model-dir "$ocr_model_dir" \
  --project "$project_root" \
  --max-samples "${PHASE2_MAX_SAMPLES:-50}" \
  --max-new-charts "${PHASE2_MAX_NEW_CHARTS:-180}" \
  --progress-every "${PHASE2_PROGRESS_EVERY:-5}" \
  > logs/phase2_cpu.log 2>&1

"$python_bin" src/evaluate_phase2.py \
  --project "$project_root" \
  --seed "${PHASE2_SEED:-20260816}" \
  --bootstrap-iterations "${PHASE2_BOOTSTRAP_ITERATIONS:-10000}" \
  > logs/phase2_evaluation.log 2>&1
