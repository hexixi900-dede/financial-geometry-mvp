#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
: "${OCR_PYTHON:?Set OCR_PYTHON to the existing OCR Python}"
: "${OCR_MODEL_DIR:?Set OCR_MODEL_DIR to the existing OCR model directory}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH="$PROJECT_ROOT/src"
exec "$OCR_PYTHON" -u src/phase8_hybrid_qa.py run-geometry \
  --inputs phase8_hybrid/geometry_inputs.jsonl \
  --plans phase8_hybrid/plans_full.jsonl \
  --cached-charts phase8_hybrid/chart_geometry_inputs.jsonl \
  --model-dir "$OCR_MODEL_DIR" \
  --output phase8_hybrid/predictions_final.jsonl --repair-geometry --threads 4
