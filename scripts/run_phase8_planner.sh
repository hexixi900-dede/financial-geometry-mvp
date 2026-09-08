#!/usr/bin/env bash
set -uo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1
: "${VLM_PYTHON:?Set VLM_PYTHON to the existing Qwen-VL Python}"
: "${VLM_MODEL:?Set VLM_MODEL to the existing local model directory}"
export CUDA_VISIBLE_DEVICES=0
# The Python runner waits for three 60-second idle checks and resumes JSONL.
# Exit 75 releases only our own model when another compute process appears.
while true; do
  "$VLM_PYTHON" -u src/vlm_semantic_planner.py \
    --inputs phase8_hybrid/planner_inputs.jsonl --model "$VLM_MODEL" \
    --output phase8_hybrid/plans_full.jsonl
  result=$?
  if [ "$result" -ne 75 ]; then exit "$result"; fi
  sleep 60
done
