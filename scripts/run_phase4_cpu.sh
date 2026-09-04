#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd -- "$script_dir/.." && pwd)}"

: "${FINMME_ROOT:?Set FINMME_ROOT to an existing FinMME workspace.}"
python_bin="${PYTHON_BIN:-$FINMME_ROOT/phase6_chart_structure_transfer_rev2_screen_v1/.venv_ocr/bin/python}"
ocr_model_dir="${OCR_MODEL_DIR:-$FINMME_ROOT/phase6_chart_structure_transfer_rev2_screen_v1/.ocr_models}"

for required_path in "$python_bin" "$ocr_model_dir"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required input does not exist: $required_path" >&2
    exit 2
  fi
done

cd "$project_root"
mkdir -p logs phase4_audit phase4_masked_images phase4_debug_overlays results
export PYTHONPATH="$project_root/src"
export CUDA_VISIBLE_DEVICES=""

# Stage 1: full-corpus scan (sharded, resumable) + deterministic merge.
if [[ "${PHASE4_SKIP_SCAN:-0}" != "1" ]]; then
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$python_bin" src/phase4_scan_charts.py \
    --project "$project_root" \
    --manifest results/finmme_chart_manifest.csv \
    --model-dir "$ocr_model_dir" \
    --stage all \
    --workers "${PHASE4_SCAN_WORKERS:-8}" \
    > logs/phase4_scan.log 2>&1

  # Stage 2: supplementary series re-detection with the final detector.
  OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$python_bin" src/phase4_redetect_series.py \
    --project "$project_root" \
    > logs/phase4_redetect.log 2>&1
fi

# Stage 3: construction + experiments + analysis (deterministic).
OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 "$python_bin" src/run_phase4_multiline.py \
  --project "$project_root" \
  --model-dir "$ocr_model_dir" \
  --stage all \
  > logs/phase4_construct.log 2>&1

# Stage 4: unit tests + validation + reproducibility snapshot.
"$python_bin" tests/test_phase4.py > logs/phase4_tests.log 2>&1
"$python_bin" src/verify_phase4_reproducibility.py --project "$project_root" --mode snapshot \
  > logs/phase4_reproducibility_snapshot.log 2>&1
"$python_bin" src/validate_phase4.py --project "$project_root" > logs/phase4_validation.log 2>&1
