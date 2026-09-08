#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ocr_python=/data/liu_jun/finmme_reproduction/phase6_chart_structure_transfer_rev2_screen_v1/.venv_ocr/bin/python
vlm_python=/data/liu_jun/finmme_reproduction/.venv_phase2a/bin/python
model=/data/liu_jun/blind_cssa/models/Qwen2.5-VL-7B-Instruct
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p phase9_evidence/logs
printf 'geometry_running\n' > phase9_evidence/status.txt
CUDA_VISIBLE_DEVICES= "$ocr_python" -u src/phase8_hybrid_qa.py run-geometry \
 --inputs phase8_hybrid/geometry_inputs.jsonl --plans phase8_hybrid/plans_full.jsonl \
 --cached-charts phase9_evidence/charts.jsonl \
 --model-dir /data/liu_jun/finmme_reproduction/phase6_chart_structure_transfer_rev2_screen_v1/.ocr_models \
 --output phase9_evidence/measurements.jsonl --repair-geometry --evidence-only --threads 4
if [ ! -f phase9_evidence/prompts.jsonl ]; then
 python3 src/vlm_evidence_answerer.py prepare --inputs phase8_hybrid/geometry_inputs.jsonl \
  --measurements phase9_evidence/measurements.jsonl --output phase9_evidence/prompts.jsonl
fi
printf 'answering_or_waiting_for_gpu\n' > phase9_evidence/status.txt
while true; do
 set +e
 CUDA_VISIBLE_DEVICES=0 "$vlm_python" -u src/vlm_evidence_answerer.py run \
  --inputs phase9_evidence/prompts.jsonl --model "$model" --output phase9_evidence/replies.jsonl
 result=$?
 set -e
 if [ "$result" -eq 0 ]; then break; fi
 if [ "$result" -ne 75 ]; then printf 'failed\n' > phase9_evidence/status.txt; exit "$result"; fi
 sleep 60
done
python3 src/vlm_evidence_answerer.py evaluate --inputs phase8_hybrid/geometry_inputs.jsonl \
 --gold phase8_hybrid/gold.jsonl --baseline phase8_hybrid/baseline.jsonl \
 --replies phase9_evidence/replies.jsonl --prompts phase9_evidence/prompts.jsonl \
 --raw-metrics /data/liu_jun/finmme_reproduction/outputs/metrics/qwen25vl7b_direct_full_per_sample.jsonl \
 --output phase9_evidence/summary.json
printf 'complete\n' > phase9_evidence/status.txt
