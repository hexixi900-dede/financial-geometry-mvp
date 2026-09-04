#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${FINMME_ROOT:?Set FINMME_ROOT to the existing FinMME workspace}"
: "${FINCHART_ROOT:?Set FINCHART_ROOT to the FinChart-Bench checkout}"

python_bin="${PYTHON_BIN:-$FINMME_ROOT/phase6_chart_structure_transfer_rev2_screen_v1/.venv_ocr/bin/python}"
ocr_model_dir="${OCR_MODEL_DIR:-$FINMME_ROOT/phase6_chart_structure_transfer_rev2_screen_v1/.ocr_models}"
results_dir="$project_root/results"

export PYTHONPATH="$project_root/src"
export CUDA_VISIBLE_DEVICES=""
mkdir -p "$results_dir" "$project_root/logs"

"$python_bin" "$project_root/src/accept_phase4.py" --project "$project_root"
"$python_bin" "$project_root/src/phase5_natural_qa.py" prepare \
  --metadata "$FINCHART_ROOT/QA_data.json" \
  --image-dir "$FINCHART_ROOT/QA_images" \
  --output-dir "$results_dir" \
  --max-images "${PHASE5_MAX_IMAGES:-100}" \
  --max-questions "${PHASE5_MAX_QUESTIONS:-200}"
"$python_bin" "$project_root/src/phase5_natural_qa.py" run \
  --inputs "$results_dir/phase5_inputs.csv" \
  --model-dir "$ocr_model_dir" \
  --output "$results_dir/phase5_geometry_predictions.csv"
"$python_bin" "$project_root/src/phase5_natural_qa.py" evaluate \
  --predictions "$results_dir/phase5_geometry_predictions.csv" \
  --gold "$results_dir/phase5_gold.csv" \
  --output-dir "$results_dir"
