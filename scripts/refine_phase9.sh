#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
vlm_python=/data/liu_jun/finmme_reproduction/.venv_phase2a/bin/python
ocr_python=/data/liu_jun/finmme_reproduction/phase6_chart_structure_transfer_rev2_screen_v1/.venv_ocr/bin/python
printf 'waiting_for_gpu_semantic_retry\n' > phase9_evidence/status.txt
while true; do
 set +e
 CUDA_VISIBLE_DEVICES=0 "$vlm_python" -u src/vlm_semantic_planner.py --inputs phase9_evidence/planner_retry_inputs.jsonl \
  --model /data/liu_jun/blind_cssa/models/Qwen2.5-VL-7B-Instruct --output phase9_evidence/plans_retry.jsonl
 rc=$?
 set -e
 if [ "$rc" -eq 0 ]; then break; fi
 if [ "$rc" -ne 75 ]; then exit "$rc"; fi
 sleep 60
done
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$ocr_python" -u src/phase8_hybrid_qa.py run-geometry \
 --inputs phase9_evidence/planner_retry_inputs.jsonl --plans phase9_evidence/plans_retry.jsonl \
 --cached-charts phase9_evidence/charts.jsonl \
 --model-dir /data/liu_jun/finmme_reproduction/phase6_chart_structure_transfer_rev2_screen_v1/.ocr_models \
 --output phase9_evidence/measurements_retry.jsonl --repair-geometry --evidence-only --threads 4
python3 - <<'PY'
import json,shutil
from pathlib import Path
p=Path('phase9_evidence')
def rd(path):return {r['sample_id']:r for r in map(json.loads,path.read_text().splitlines())}
original=p/'measurements.jsonl';backup=p/'measurements_before_semantic_retry.jsonl'
if not backup.exists():shutil.copy2(original,backup)
merged=rd(backup);merged.update(rd(p/'measurements_retry.jsonl'))
original.write_text(''.join(json.dumps(r)+'\n' for r in merged.values()))
if (p/'prompts.jsonl').exists():(p/'prompts.jsonl').rename(p/'prompts_before_semantic_retry.jsonl')
PY
exec bash scripts/run_phase9.sh
