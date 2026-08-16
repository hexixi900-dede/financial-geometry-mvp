from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import easyocr
import torch

from geometry_core import (
    axis_localizer,
    choose_geometry_target,
    data_label_candidates,
    label_preserves_geometry,
    ocr_tokens,
)


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


def load_existing(path: Path) -> tuple[set[str], int]:
    processed: set[str] = set()
    viable = 0
    if not path.exists():
        return processed, viable
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            processed.add(str(row["chart_id"]))
            viable += int(row.get("status") == "viable")
    return processed, viable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--max-new-charts", type=int, default=900)
    parser.add_argument("--target-viable", type=int, default=320)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    cv2.setNumThreads(6)
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)

    with args.manifest.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    eligible = [
        row
        for row in manifest
        if int(row["has_numerical"]) == 1
        and int(row["keyword_exclude"]) == 0
        and (int(row["bar_rectangles"]) >= 2 or int(row["sloped_segments"]) >= 1)
        and int(row["image_width"]) >= 300
        and int(row["image_height"]) >= 220
    ]
    bar_pool = [
        row
        for row in eligible
        if 2 <= int(row["bar_rectangles"]) <= 38
        and int(row["sloped_segments"]) < 3
        and float(row["color_fraction"]) <= 0.48
    ]
    line_pool = [
        row
        for row in eligible
        if int(row["bar_rectangles"]) < 3
        and 2 <= int(row["sloped_segments"]) <= 70
        and float(row["color_fraction"]) <= 0.12
    ]
    bar_pool.sort(
        key=lambda row: (
            -int(row["keyword_include"]),
            abs(int(row["bar_rectangles"]) - 10),
            int(row["sloped_segments"]),
            -float(row["prefilter_priority"]),
            row["chart_id"],
        )
    )
    line_pool.sort(
        key=lambda row: (
            -int(row["keyword_include"]),
            abs(int(row["sloped_segments"]) - 14),
            int(row["bar_rectangles"]),
            -float(row["prefilter_priority"]),
            row["chart_id"],
        )
    )
    ordered: list[dict[str, str]] = []
    seen_order: set[str] = set()
    bar_index = 0
    line_index = 0
    while bar_index < len(bar_pool) or line_index < len(line_pool):
        for _ in range(2):
            if bar_index < len(bar_pool):
                row = bar_pool[bar_index]
                bar_index += 1
                if row["chart_id"] not in seen_order:
                    ordered.append(row)
                    seen_order.add(row["chart_id"])
        if line_index < len(line_pool):
            row = line_pool[line_index]
            line_index += 1
            if row["chart_id"] not in seen_order:
                ordered.append(row)
                seen_order.add(row["chart_id"])
    remaining = sorted(
        (row for row in eligible if row["chart_id"] not in seen_order),
        key=lambda row: (-float(row["prefilter_priority"]), row["chart_id"]),
    )
    eligible = ordered + remaining
    processed, viable_total = load_existing(args.output)
    reader_start = time.monotonic()
    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=str(args.model_dir),
        download_enabled=False,
        verbose=False,
    )
    print(
        json.dumps(
            {
                "event": "reader_ready",
                "reader_seconds": time.monotonic() - reader_start,
                "eligible": len(eligible),
                "already_processed": len(processed),
                "already_viable": viable_total,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    new_processed = 0
    status_counts: dict[str, int] = {}
    start = time.monotonic()
    for row in eligible:
        chart_id = str(row["chart_id"])
        if chart_id in processed:
            continue
        if new_processed >= args.max_new_charts or viable_total >= args.target_viable:
            break
        item_start = time.monotonic()
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        result: dict[str, Any] = {
            "chart_id": chart_id,
            "representative_sample_id": row["representative_sample_id"],
            "image_path": row["image_path"],
            "caption": row["caption"],
            "numerical_sample_ids": row["numerical_sample_ids"],
            "numerical_gold": row["numerical_gold"],
            "numerical_unit": row["numerical_unit"],
            "numerical_tolerance": row["numerical_tolerance"],
            "prefilter": {
                "priority": float(row["prefilter_priority"]),
                "bar_rectangles": int(row["bar_rectangles"]),
                "sloped_segments": int(row["sloped_segments"]),
                "color_fraction": float(row["color_fraction"]),
            },
        }
        if image is None:
            result["status"] = "image_read_failed"
        else:
            raw_ocr = reader.readtext(
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
            tokens = ocr_tokens(raw_ocr)
            height, width = image.shape[:2]
            result["image_size"] = [width, height]
            result["ocr_token_count"] = len(tokens)
            result["numeric_tokens"] = [
                token.as_dict() for token in tokens if token.numeric_value is not None
            ]
            axis = axis_localizer(tokens, width, height)
            if axis is None:
                result["status"] = "no_linear_left_y_axis"
            elif axis["right_axis_detected"] or int(axis.get("additional_y_axis_count", 0)) > 0:
                result["status"] = "multiple_y_axes_or_panels_detected"
                result["axis"] = axis
            else:
                result["axis"] = axis
                label_candidates = data_label_candidates(tokens, axis, width, height)
                viable_labels: list[dict[str, Any]] = []
                for label in label_candidates:
                    geometry = choose_geometry_target(image, label, axis)
                    if geometry is None or not label_preserves_geometry(label, geometry):
                        continue
                    if geometry["chart_type"] == "bar" and int(row["sloped_segments"]) >= 3:
                        continue
                    if geometry["chart_type"] == "line" and int(row["bar_rectangles"]) >= 3:
                        continue
                    viable_labels.append(
                        {
                            "label": label.as_dict(),
                            "geometry_on_original": geometry,
                            "construction_score": float(label.confidence)
                            + float(geometry["score"]),
                        }
                    )
                viable_labels.sort(
                    key=lambda item: (
                        -float(item["construction_score"]),
                        item["label"]["bbox"][1],
                        item["label"]["bbox"][0],
                    )
                )
                result["data_label_candidate_count"] = len(label_candidates)
                result["viable_labels"] = viable_labels
                result["status"] = "viable" if viable_labels else "no_geometry_linked_data_label"
        result["ocr_and_screen_seconds"] = time.monotonic() - item_start
        append_jsonl(args.output, result)
        new_processed += 1
        status = str(result["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "viable":
            viable_total += 1
        if new_processed % args.progress_every == 0 or status == "viable":
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "new_processed": new_processed,
                        "viable_total": viable_total,
                        "last_status": status,
                        "last_chart_id": chart_id,
                        "elapsed_seconds": time.monotonic() - start,
                        "status_counts_new": status_counts,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "event": "complete",
                "new_processed": new_processed,
                "viable_total": viable_total,
                "elapsed_seconds": time.monotonic() - start,
                "status_counts_new": status_counts,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
