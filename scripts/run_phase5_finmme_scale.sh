#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/phase5_finmme_scale"
PYTHON_BIN="${PYTHON_BIN:-/data/liu_jun/finmme_reproduction/phase6_chart_structure_transfer_rev2_screen_v1/.venv_ocr/bin/python}"
MODEL_DIR="${MODEL_DIR:-/data/liu_jun/finmme_reproduction/phase6_chart_structure_transfer_rev2_screen_v1/.ocr_models}"
STAGE="${1:-prepare}"

mkdir -p "$OUTPUT_ROOT/manifests" "$OUTPUT_ROOT/audit" "$OUTPUT_ROOT/logs"
export PYTHONPATH="$PROJECT_ROOT/src"
export CUDA_VISIBLE_DEVICES=""

prepare() {
  "$PYTHON_BIN" "$PROJECT_ROOT/src/partition_remaining_manifest.py" \
    --manifest "$PROJECT_ROOT/results/finmme_chart_manifest.csv" \
    --processed "$PROJECT_ROOT/audit/candidate_scan_v3.jsonl" \
    --output-a "$OUTPUT_ROOT/manifests/remaining_a.csv" \
    --output-b "$OUTPUT_ROOT/manifests/remaining_b.csv"
}

scan() {
  for shard in a b; do
    nohup env OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 "$PYTHON_BIN" \
      "$PROJECT_ROOT/src/scan_candidates.py" \
      --manifest "$OUTPUT_ROOT/manifests/remaining_${shard}.csv" \
      --output "$OUTPUT_ROOT/audit/candidate_scan_${shard}.jsonl" \
      --model-dir "$MODEL_DIR" \
      --max-new-charts 1000 \
      --target-viable 1000 \
      --threads 3 \
      --progress-every 10 \
      > "$OUTPUT_ROOT/logs/scan_${shard}.log" 2>&1 < /dev/null &
    echo "$!" > "$OUTPUT_ROOT/logs/scan_${shard}.pid"
    echo "started shard ${shard}: PID $!"
  done
}

construct() {
  for shard in a b; do
    pid_file="$OUTPUT_ROOT/logs/scan_${shard}.pid"
    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      echo "scan shard ${shard} is still running" >&2
      exit 3
    fi
  done
  "$PYTHON_BIN" "$PROJECT_ROOT/src/merge_candidate_scans.py" \
    --output "$OUTPUT_ROOT/audit/candidate_scan.jsonl" \
    --backup "$OUTPUT_ROOT/audit/candidate_scan.previous.jsonl" \
    "$OUTPUT_ROOT/audit/candidate_scan_a.jsonl" \
    "$OUTPUT_ROOT/audit/candidate_scan_b.jsonl"
  "$PYTHON_BIN" "$PROJECT_ROOT/src/run_geometry_samples.py" \
    --candidate-scan "$OUTPUT_ROOT/audit/candidate_scan.jsonl" \
    --project "$OUTPUT_ROOT" \
    --model-dir "$MODEL_DIR" \
    --max-samples 100 \
    --max-new-charts 200 \
    --progress-every 10
}

evaluate() {
  "$PYTHON_BIN" "$PROJECT_ROOT/src/evaluate_phase5_finmme_scale.py" \
    --new-samples "$OUTPUT_ROOT/audit/geometry_samples.jsonl" \
    --old-samples "$PROJECT_ROOT/audit/geometry_samples.jsonl" \
    --old-split "$PROJECT_ROOT/results/chart_split.csv" \
    --model "$PROJECT_ROOT/config/calibrator_model.json" \
    --output-dir "$OUTPUT_ROOT/results"
}

case "$STAGE" in
  prepare) prepare ;;
  scan) scan ;;
  construct) construct ;;
  evaluate) evaluate ;;
  *) echo "usage: $0 {prepare|scan|construct|evaluate}" >&2; exit 2 ;;
esac
