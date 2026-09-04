"""Phase 4 supplementary pass: re-detect line series with the final detector.

The full-corpus scan records OCR/axis/labels/anchors with the scan-time
series detector. This pass re-runs only the (cheap, OCR-free) final series
detector on every scanned chart that has a usable single linear axis, and
re-derives the single/multi-line viability classification. The original scan
file is never modified; the result is written as chart_scan_v2.jsonl and is
what construction consumes.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path
from typing import Any

import cv2

from phase4_series import detect_line_series_components, strip_masks


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()

    rows = read_jsonl(project / "phase4_audit" / "chart_scan.jsonl")
    output: list[dict[str, Any]] = []
    started = time.monotonic()
    reclassified = 0
    for index, row in enumerate(rows, 1):
        row = dict(row)
        axis = row.get("axis")
        usable_axis = bool(axis) and not axis.get("right_axis_detected") and int(axis.get("additional_y_axis_count", 0)) == 0
        if not usable_axis or row.get("status") in {"image_read_failed", "image_too_small", "scan_exception"}:
            output.append(row)
            continue
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            row["status"] = "image_read_failed"
            output.append(row)
            continue
        series = detect_line_series_components(image, axis)
        row["line_series"] = strip_masks(series)
        row["line_series_count"] = len(series)
        row["series_detector"] = "phase4_final_saturation_plus_neutral"
        labels = int(row.get("data_label_candidate_count", 0))
        anchors = int(row.get("x_anchor_count", 0))
        temporal = int(row.get("x_temporal_anchor_count", 0))
        old_status = str(row.get("status"))
        reasons: list[str] = []
        if labels == 0:
            reasons.append("no_data_value_label")
        if temporal < 2 and anchors < 2:
            reasons.append("insufficient_x_anchors")
        if not reasons and len(series) == 0:
            reasons.append("no_line_series")
        if reasons:
            row["status"] = "rejected"
            row["rejection_reasons"] = reasons
        else:
            row["status"] = "viable_single_line" if len(series) == 1 else "viable_multi_line"
            row["rejection_reasons"] = []
        if row["status"] != old_status:
            reclassified += 1
            row["reclassified_from"] = old_status
        output.append(row)
        if index % 250 == 0:
            print(
                json.dumps(
                    {"event": "phase4_redetect_progress", "done": index, "total": len(rows), "elapsed_seconds": round(time.monotonic() - started, 1)},
                    sort_keys=True,
                ),
                flush=True,
            )
    write_jsonl(project / "phase4_audit" / "chart_scan_v2.jsonl", output)
    counts: dict[str, int] = {}
    for row in output:
        counts[str(row.get("status"))] = counts.get(str(row.get("status")), 0) + 1
    print(
        json.dumps(
            {"event": "phase4_redetect_complete", "reclassified": reclassified, "status_counts": counts, "elapsed_seconds": round(time.monotonic() - started, 1)},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
