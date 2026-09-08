#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
root=phase9_evidence/unified
vlm_python=/data/liu_jun/finmme_reproduction/.venv_phase2a/bin/python
while true; do
 set +e
 CUDA_VISIBLE_DEVICES=0 "$vlm_python" -u src/run_unified_resident.py
 rc=$?
 set -e
 if [ "$rc" -eq 0 ]; then break; fi
 if [ "$rc" -ne 75 ]; then
  printf 'failed (exit %s); see run.log\n' "$rc" > "$root/status.txt"
  exit "$rc"
 fi
 printf 'waiting_for_gpu; progress saved\n' > "$root/status.txt"
 sleep 60
done
