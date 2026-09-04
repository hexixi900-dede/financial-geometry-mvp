#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/phase5_finmme_natural"
PYTHON_BIN="${PYTHON_BIN:?Set PYTHON_BIN to the existing OCR environment Python}"
OCR_MODEL_DIR="${OCR_MODEL_DIR:?Set OCR_MODEL_DIR to the existing EasyOCR model directory}"
FINMME_RECORDS="${FINMME_RECORDS:?Set FINMME_RECORDS to the existing FinMME full-record JSONL}"
FINMME_RAW_VLM="${FINMME_RAW_VLM:?Set FINMME_RAW_VLM to the existing Raw VLM prediction JSONL}"
FINMME_RAW_VLM_METRICS="${FINMME_RAW_VLM_METRICS:?Set FINMME_RAW_VLM_METRICS to its per-sample metric JSONL}"
STAGE="${1:-all}"

mkdir -p "$OUTPUT_ROOT/audit" "$OUTPUT_ROOT/results" "$OUTPUT_ROOT/debug_overlays" "$OUTPUT_ROOT/logs"
export PYTHONPATH="$PROJECT_ROOT/src"
export CUDA_VISIBLE_DEVICES=""

prepare() {
  "$PYTHON_BIN" "$PROJECT_ROOT/src/phase5_finmme_natural_line.py" prepare \
    --chart-scan "$PROJECT_ROOT/phase4_audit/chart_scan_v2.jsonl" \
    --records "$FINMME_RECORDS" \
    --raw-vlm "$FINMME_RAW_VLM" \
    --raw-vlm-metrics "$FINMME_RAW_VLM_METRICS" \
    --output-dir "$OUTPUT_ROOT/audit" \
    --max-questions 50
}

run_geometry() {
  "$PYTHON_BIN" "$PROJECT_ROOT/src/phase5_finmme_natural_line.py" run \
    --inputs "$OUTPUT_ROOT/audit/inputs.csv" \
    --model-dir "$OCR_MODEL_DIR" \
    --calibrator "$PROJECT_ROOT/config/calibrator_model.json" \
    --output "$OUTPUT_ROOT/audit/geometry_predictions.jsonl" \
    --overlay-dir "$OUTPUT_ROOT/debug_overlays" \
    --progress-every 5
}

evaluate() {
  "$PYTHON_BIN" "$PROJECT_ROOT/src/phase5_finmme_natural_line.py" evaluate \
    --predictions "$OUTPUT_ROOT/audit/geometry_predictions.jsonl" \
    --gold "$OUTPUT_ROOT/audit/gold.csv" \
    --raw-vlm "$OUTPUT_ROOT/audit/raw_vlm.csv" \
    --output-dir "$OUTPUT_ROOT/results"
}

case "$STAGE" in
  prepare) prepare ;;
  run) run_geometry ;;
  evaluate) evaluate ;;
  all) prepare; run_geometry; evaluate ;;
  *) echo "usage: $0 {prepare|run|evaluate|all}" >&2; exit 2 ;;
esac
