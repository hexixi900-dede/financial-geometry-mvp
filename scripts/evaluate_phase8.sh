#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
: "${FINMME_RAW_METRICS:?Set FINMME_RAW_METRICS to the existing full baseline metrics JSONL}"
python3 src/phase8_hybrid_qa.py evaluate \
  --inputs phase8_hybrid/geometry_inputs.jsonl \
  --plans phase8_hybrid/plans_full.jsonl \
  --gold phase8_hybrid/gold.jsonl \
  --baseline phase8_hybrid/baseline.jsonl \
  --predictions phase8_hybrid/predictions_final.jsonl \
  --raw-metrics "$FINMME_RAW_METRICS" \
  --full-audit phase8_hybrid/full_question_audit.csv \
  --phase7-samples results/phase7_finmme_geometry_qa_samples.csv \
  --output-dir phase8_hybrid/results
