#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/phase5_finmme_scale"
LOG_FILE="$OUTPUT_ROOT/logs/continue.log"

scan_is_running() {
  local shard="$1"
  local pid
  pid="$(cat "$OUTPUT_ROOT/logs/scan_${shard}.pid")"
  ps -p "$pid" -o args= 2>/dev/null | grep -q "scan_candidates.py.*remaining_${shard}.csv"
}

while scan_is_running a || scan_is_running b; do
  a_rows=$(wc -l < "$OUTPUT_ROOT/audit/candidate_scan_a.jsonl" 2>/dev/null || echo 0)
  b_rows=$(wc -l < "$OUTPUT_ROOT/audit/candidate_scan_b.jsonl" 2>/dev/null || echo 0)
  echo "$(date --iso-8601=seconds) scanning a=$a_rows/284 b=$b_rows/283" >> "$LOG_FILE"
  sleep 60
done

a_rows=$(wc -l < "$OUTPUT_ROOT/audit/candidate_scan_a.jsonl")
b_rows=$(wc -l < "$OUTPUT_ROOT/audit/candidate_scan_b.jsonl")
if [ "$a_rows" -ne 284 ] || [ "$b_rows" -ne 283 ]; then
  echo "$(date --iso-8601=seconds) incomplete scan a=$a_rows b=$b_rows; stopping" >> "$LOG_FILE"
  exit 3
fi

echo "$(date --iso-8601=seconds) scans complete; constructing" >> "$LOG_FILE"
"$PROJECT_ROOT/scripts/run_phase5_finmme_scale.sh" construct >> "$LOG_FILE" 2>&1
echo "$(date --iso-8601=seconds) construction complete; evaluating" >> "$LOG_FILE"
"$PROJECT_ROOT/scripts/run_phase5_finmme_scale.sh" evaluate >> "$LOG_FILE" 2>&1
echo "$(date --iso-8601=seconds) Phase 5 CPU pipeline complete" >> "$LOG_FILE"
