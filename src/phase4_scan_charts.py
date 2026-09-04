"""Phase 4 Part A: full-corpus FinMME chart scan for masked-value recovery.

Scans every unique chart in the existing FinMME manifest (not only
Numerical-associated charts) and records everything needed to decide whether
the chart can host a module-level masked-value geometry sample:

- linear single left y-axis (dual/multi-axis are out of scope and rejected);
- x-axis semantic anchors (direct labels / rotated OCR fallback);
- OCR data value-label candidates (pseudo-Gold source);
- line series components (single vs multi line structure);
- legend entries (series name -> swatch appearance);
- bar evidence from the frozen manifest prefilter (statistics only).

The scan never reads FinMME Gold answers. It is CPU-only, sharded
deterministically by chart_id, resumable per shard, and the merged output is
sorted by chart_id so the final artifact is deterministic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from collections import Counter
from multiprocessing import Process
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(type(value).__name__)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


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


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze_chart(reader: Any, manifest_row: dict[str, str]) -> dict[str, Any]:
    """Full per-chart structure analysis. Runs inside a worker process."""
    from geometry_core import axis_localizer, data_label_candidates, ocr_tokens
    from phase3_line_core import discover_x_axis_anchors
    from phase4_legend import detect_legend_entries
    from phase4_series import detect_line_series_components, strip_masks

    chart_id = str(manifest_row["chart_id"])
    result: dict[str, Any] = {
        "chart_id": chart_id,
        "representative_sample_id": manifest_row["representative_sample_id"],
        "image_path": manifest_row["image_path"],
        "caption": manifest_row["caption"],
        "has_numerical": int(manifest_row["has_numerical"]),
        "numerical_sample_ids": manifest_row["numerical_sample_ids"],
        "prefilter": {
            "bar_rectangles": int(manifest_row["bar_rectangles"]),
            "sloped_segments": int(manifest_row["sloped_segments"]),
            "color_fraction": float(manifest_row["color_fraction"]),
        },
    }
    started = time.monotonic()
    image = cv2.imread(str(manifest_row["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        result["status"] = "image_read_failed"
        return result
    height, width = image.shape[:2]
    result["image_size"] = [width, height]
    if width < 300 or height < 220:
        result["status"] = "image_too_small"
        return result
    raw = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        min_size=7,
        text_threshold=0.45,
        low_text=0.25,
        link_threshold=0.35,
        canvas_size=2560,
        mag_ratio=1.25,
    )
    tokens = ocr_tokens(raw)
    result["ocr_token_count"] = len(tokens)
    axis = axis_localizer(tokens, width, height)
    if axis is None:
        result["status"] = "no_linear_left_y_axis"
        result["elapsed_seconds"] = time.monotonic() - started
        return result
    if axis.get("right_axis_detected") or int(axis.get("additional_y_axis_count", 0)) > 0:
        result["status"] = "multiple_y_axes_or_panels_detected"
        result["axis"] = axis
        result["elapsed_seconds"] = time.monotonic() - started
        return result
    result["axis"] = axis

    anchors, anchor_audit = discover_x_axis_anchors(reader, image, tokens, axis)
    temporal_anchors = [anchor for anchor in anchors if anchor.get("scalar") is not None]
    result["x_axis_anchors"] = anchors
    result["x_axis_anchor_audit"] = anchor_audit
    result["x_anchor_count"] = len(anchors)
    result["x_temporal_anchor_count"] = len(temporal_anchors)

    labels = data_label_candidates(tokens, axis, width, height)
    result["data_label_candidates"] = [label.as_dict() for label in labels]
    result["data_label_candidate_count"] = len(labels)

    series = detect_line_series_components(image, axis)
    result["line_series"] = strip_masks(series)
    result["line_series_count"] = len(series)
    coverages = sorted((float(item["coverage_fraction"]) for item in series), reverse=True)

    legend = detect_legend_entries(image, tokens, axis)
    result["legend"] = legend
    result["legend_entry_count"] = len(legend["entries"])

    bar_rectangles = int(manifest_row["bar_rectangles"])
    result["bar_candidate"] = bool(bar_rectangles >= 2 and len(series) == 0)

    reasons: list[str] = []
    if len(labels) == 0:
        reasons.append("no_data_value_label")
    if len(temporal_anchors) < 2 and len(anchors) < 2:
        reasons.append("insufficient_x_anchors")
    if reasons:
        result["status"] = "rejected"
        result["rejection_reasons"] = reasons
        result["elapsed_seconds"] = time.monotonic() - started
        return result
    if len(series) == 0:
        result["status"] = "rejected"
        result["rejection_reasons"] = ["no_line_series"] + (["bar_candidate_noted"] if result["bar_candidate"] else [])
        result["elapsed_seconds"] = time.monotonic() - started
        return result
    if len(series) == 1:
        result["status"] = "viable_single_line"
    else:
        result["status"] = "viable_multi_line"
        result["series_coverage"] = coverages
    result["rejection_reasons"] = []
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def worker_main(args: argparse.Namespace, shard_index: int, manifest_rows: list[dict[str, str]]) -> None:
    import easyocr
    import torch

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=str(args.model_dir),
        download_enabled=False,
        verbose=False,
    )
    shard_path = args.project / "phase4_audit" / "scan_shards" / f"shard_{shard_index:02d}.jsonl"
    done = {str(row["chart_id"]) for row in read_jsonl(shard_path)}
    todo = [row for row in manifest_rows if str(row["chart_id"]) not in done]
    print(
        json.dumps({"event": "phase4_shard_start", "shard": shard_index, "todo": len(todo), "done": len(done)}, sort_keys=True),
        flush=True,
    )
    started = time.monotonic()
    status_counts: Counter[str] = Counter()
    for index, manifest_row in enumerate(todo, 1):
        try:
            result = analyze_chart(reader, manifest_row)
        except Exception as error:  # keep the shard alive; record and continue
            result = {
                "chart_id": str(manifest_row["chart_id"]),
                "image_path": manifest_row["image_path"],
                "status": "scan_exception",
                "exception": f"{type(error).__name__}: {error}",
            }
        append_jsonl(shard_path, result)
        status_counts[str(result.get("status"))] += 1
        if index % args.progress_every == 0 or str(result.get("status", "")).startswith("viable"):
            print(
                json.dumps(
                    {
                        "event": "phase4_scan_progress",
                        "shard": shard_index,
                        "processed": index,
                        "remaining": len(todo) - index,
                        "last_chart": str(manifest_row["chart_id"])[:12],
                        "last_status": result.get("status"),
                        "elapsed_seconds": round(time.monotonic() - started, 1),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "event": "phase4_shard_complete",
                "shard": shard_index,
                "processed": len(todo),
                "status_counts": dict(status_counts),
                "elapsed_seconds": round(time.monotonic() - started, 1),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run_scan(args: argparse.Namespace) -> None:
    manifest = load_csv(args.manifest)
    manifest.sort(key=lambda row: str(row["chart_id"]))
    if args.limit:
        manifest = manifest[: args.limit]
    workers = max(1, int(args.workers))
    shards = [manifest[index::workers] for index in range(workers)]
    if workers == 1:
        worker_main(args, 0, shards[0])
        return
    processes: list[Process] = []
    for shard_index, shard_rows in enumerate(shards):
        process = Process(target=worker_main, args=(args, shard_index, shard_rows))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failures = [process.exitcode for process in processes if process.exitcode != 0]
    print(json.dumps({"event": "phase4_scan_all_shards_complete", "worker_failures": failures}, sort_keys=True), flush=True)


def run_merge(args: argparse.Namespace) -> None:
    shard_dir = args.project / "phase4_audit" / "scan_shards"
    merged: dict[str, dict[str, Any]] = {}
    for shard_path in sorted(shard_dir.glob("shard_*.jsonl")):
        for row in read_jsonl(shard_path):
            chart_id = str(row["chart_id"])
            if chart_id not in merged:
                merged[chart_id] = row
    rows = [merged[chart_id] for chart_id in sorted(merged)]
    write_jsonl(args.project / "phase4_audit" / "chart_scan.jsonl", rows)
    status_counts = Counter(str(row.get("status")) for row in rows)
    reason_counts: Counter[str] = Counter()
    for row in rows:
        for reason in row.get("rejection_reasons", []):
            reason_counts[str(reason)] += 1
    summary = {
        "charts_scanned": len(rows),
        "status_counts": dict(status_counts),
        "rejection_reason_counts": dict(reason_counts),
        "viable_single_line_charts": status_counts.get("viable_single_line", 0),
        "viable_multi_line_charts": status_counts.get("viable_multi_line", 0),
        "multi_line_label_candidates": sum(
            int(row.get("data_label_candidate_count", 0)) for row in rows if row.get("status") == "viable_multi_line"
        ),
        "single_line_label_candidates": sum(
            int(row.get("data_label_candidate_count", 0)) for row in rows if row.get("status") == "viable_single_line"
        ),
    }
    print(json.dumps({"event": "phase4_scan_merge", **summary}, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["scan", "merge", "all"], default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"scan", "all"}:
        run_scan(args)
    if args.stage in {"merge", "all"}:
        run_merge(args)


if __name__ == "__main__":
    main()
