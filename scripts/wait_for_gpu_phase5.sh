#!/usr/bin/env bash
set -uo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 command [args ...]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
WAIT_LOG="$LOG_DIR/phase5_gpu_wait.log"
RUN_LOG="$LOG_DIR/phase5_vlm.log"
mkdir -p "$LOG_DIR"

idle_count=0
while true; do
  timestamp="$(date --iso-8601=seconds)"
  compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')"
  process_status=$?
  gpu_state="$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"
  gpu_status=$?
  idle=true
  if [ "$process_status" -ne 0 ] || [ "$gpu_status" -ne 0 ] || [ -n "$compute_pids" ] || [ -z "$gpu_state" ]; then
    idle=false
  fi
  while IFS=',' read -r memory_used utilization; do
    memory_used="${memory_used//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    if [ -z "$memory_used" ] || [ "$memory_used" -gt 1024 ] || [ "$utilization" -gt 5 ]; then
      idle=false
    fi
  done <<< "$gpu_state"

  if [ "$idle" = true ]; then
    idle_count=$((idle_count + 1))
  else
    idle_count=0
  fi
  echo "$timestamp idle=$idle consecutive=$idle_count pids=${compute_pids:-none} gpu=[$gpu_state]" >> "$WAIT_LOG"

  if [ "$idle_count" -ge 3 ]; then
    echo "$timestamp three consecutive idle checks passed; starting: $*" >> "$WAIT_LOG"
    exec "$@" >> "$RUN_LOG" 2>&1
  fi
  sleep 60
done
