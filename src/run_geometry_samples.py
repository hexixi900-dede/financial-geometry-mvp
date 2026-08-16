from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import easyocr
import numpy as np
import torch

from geometry_core import (
    axis_localizer,
    choose_geometry_target,
    draw_debug_overlay,
    geometry_value,
    label_preserves_geometry,
    mask_value_text,
    ocr_tokens,
    parse_numeric_label,
    token_from_dict,
)


ALLOWLIST = "0123456789.,-%()$€£¥₹kKmMbBtTxX"


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def crop_confirm_label(reader: easyocr.Reader, image: np.ndarray, label: Any) -> dict[str, Any] | None:
    height, width = image.shape[:2]
    pad_x = max(4, int(round(0.45 * (label.y2 - label.y1))))
    pad_y = max(4, int(round(0.25 * (label.y2 - label.y1))))
    x1 = max(0, int(math.floor(label.x1)) - pad_x)
    y1 = max(0, int(math.floor(label.y1)) - pad_y)
    x2 = min(width, int(math.ceil(label.x2)) + pad_x)
    y2 = min(height, int(math.ceil(label.y2)) + pad_y)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    scale = max(2.0, min(4.0, 96.0 / max(crop.shape[0], 1)))
    enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    label_width = max(1.0, label.x2 - label.x1)
    label_height = max(1.0, label.y2 - label.y1)
    if label_height >= 1.18 * label_width:
        orientations = [
            (90, cv2.rotate(enlarged, cv2.ROTATE_90_CLOCKWISE)),
            (270, cv2.rotate(enlarged, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ]
    else:
        orientations = [(0, enlarged)]
    target = float(label.numeric_value)
    agreements: list[dict[str, Any]] = []
    numeric_predictions: list[dict[str, Any]] = []
    for orientation, oriented in orientations:
        raw = reader.readtext(
            oriented,
            detail=1,
            paragraph=False,
            allowlist=ALLOWLIST,
            min_size=5,
            text_threshold=0.35,
            low_text=0.18,
            link_threshold=0.30,
            canvas_size=1024,
            mag_ratio=1.0,
        )
        for item in raw:
            text = str(item[1])
            value, suffix = parse_numeric_label(text)
            if value is None:
                continue
            prediction = {
                "text": text,
                "value": value,
                "suffix": suffix,
                "confidence": float(item[2]),
                "orientation": orientation,
            }
            numeric_predictions.append(prediction)
            absolute_tolerance = max(1e-6, abs(target) * 1e-6)
            if abs(value - target) <= absolute_tolerance:
                agreements.append(prediction)
    if not agreements:
        return None
    best = max(agreements, key=lambda item: item["confidence"])
    best_numeric_confidence = max(item["confidence"] for item in numeric_predictions)
    if best["confidence"] + 0.05 < best_numeric_confidence:
        return None
    return {
        **best,
        "crop_bbox": [x1, y1, x2, y2],
        "orientation_search": [item[0] for item in orientations],
        "numeric_candidates": numeric_predictions,
    }


def candidate_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    viable = [row for row in rows if row.get("status") == "viable"]
    line_rows: list[dict[str, Any]] = []
    bar_rows: list[dict[str, Any]] = []
    for row in viable:
        types = {item["geometry_on_original"]["chart_type"] for item in row["viable_labels"]}
        if "line" in types:
            line_rows.append(row)
        else:
            bar_rows.append(row)
    line_rows.sort(key=lambda row: (-max(item["construction_score"] for item in row["viable_labels"]), row["chart_id"]))
    bar_rows.sort(key=lambda row: (-max(item["construction_score"] for item in row["viable_labels"]), row["chart_id"]))
    ordered: list[dict[str, Any]] = []
    line_index = 0
    bar_index = 0
    while line_index < len(line_rows) or bar_index < len(bar_rows):
        for _ in range(2):
            if bar_index < len(bar_rows):
                ordered.append(bar_rows[bar_index])
                bar_index += 1
        if line_index < len(line_rows):
            ordered.append(line_rows[line_index])
            line_index += 1
    return ordered


def flat_sample_row(row: dict[str, Any]) -> dict[str, Any]:
    axis = row["axis"]
    geometry = row["geometry"]
    label = row["label"]
    return {
        "sample_id": row["sample_id"],
        "chart_id": row["chart_id"],
        "representative_sample_id": row["representative_sample_id"],
        "chart_type": geometry["chart_type"],
        "source_image_path": row["source_image_path"],
        "masked_image_path": row["masked_image_path"],
        "debug_overlay_path": row["debug_overlay_path"],
        "masked_label_text": label["text"],
        "pseudo_gold": row["pseudo_gold"],
        "label_ocr_confidence": label["confidence"],
        "crop_confirm_text": row["crop_confirmation"]["text"],
        "crop_confirm_confidence": row["crop_confirmation"]["confidence"],
        "label_bbox": json.dumps(label["bbox"], separators=(",", ":")),
        "mask_pixel_count": row["mask_pixel_count"],
        "tick_values": json.dumps([tick["value"] for tick in axis["ticks"]], separators=(",", ":")),
        "tick_pixel_y": json.dumps([tick["pixel_y"] for tick in axis["ticks"]], separators=(",", ":")),
        "tick_details": json.dumps(axis["ticks"], separators=(",", ":")),
        "axis_slope": axis["slope"],
        "axis_intercept": axis["intercept"],
        "axis_r_squared": axis["r_squared"],
        "axis_max_residual_pixels": axis["max_residual_pixels"],
        "axis_tick_step": axis["median_value_step"],
        "axis_span": axis["axis_span"],
        "plot_bbox": json.dumps(axis["plot_bbox"], separators=(",", ":")),
        "bar_bbox": json.dumps(geometry.get("bbox"), separators=(",", ":")),
        "line_point": json.dumps(geometry.get("point"), separators=(",", ":")),
        "target_x": geometry["target_x"],
        "target_y": geometry["target_y"],
        "geometry_score": geometry["score"],
        "raw_geometry_value": row["raw_geometry_value"],
        "absolute_error": row["absolute_error"],
        "relative_error": row["relative_error"],
        "axis_normalized_error": row["axis_normalized_error"],
        "finmme_numerical_sample_ids": row["finmme_numerical_sample_ids"],
        "finmme_numerical_gold": row["finmme_numerical_gold"],
        "finmme_numerical_unit": row["finmme_numerical_unit"],
        "finmme_numerical_tolerance": row["finmme_numerical_tolerance"],
        "geometry_gold_access": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-scan", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=240)
    parser.add_argument("--max-new-charts", type=int, default=400)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    cv2.setNumThreads(6)
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)
    sample_jsonl = args.project / "audit" / "geometry_samples.jsonl"
    attempts_jsonl = args.project / "audit" / "construction_attempts.jsonl"
    existing_samples = read_jsonl(sample_jsonl)
    attempted = {str(row["chart_id"]) for row in read_jsonl(attempts_jsonl)}
    completed_charts = {str(row["chart_id"]) for row in existing_samples}
    attempted |= completed_charts

    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=str(args.model_dir),
        download_enabled=False,
        verbose=False,
    )
    candidate_rows = candidate_order(read_jsonl(args.candidate_scan))
    new_attempts = 0
    start = time.monotonic()
    for candidate_row in candidate_rows:
        chart_id = str(candidate_row["chart_id"])
        if chart_id in attempted:
            continue
        if len(existing_samples) >= args.max_samples or new_attempts >= args.max_new_charts:
            break
        new_attempts += 1
        attempt_start = time.monotonic()
        attempt: dict[str, Any] = {
            "chart_id": chart_id,
            "representative_sample_id": candidate_row["representative_sample_id"],
            "status": "no_label_survived",
        }
        image = cv2.imread(candidate_row["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            attempt["status"] = "image_read_failed"
        else:
            labels = sorted(
                candidate_row["viable_labels"],
                key=lambda item: -float(item["construction_score"]),
            )
            for label_item in labels:
                label = token_from_dict(label_item["label"])
                confirmation = crop_confirm_label(reader, image, label)
                if confirmation is None:
                    continue
                original_geometry = label_item["geometry_on_original"]
                if (
                    original_geometry["chart_type"] == "bar"
                    and int(candidate_row["prefilter"]["sloped_segments"]) >= 3
                ):
                    continue
                if (
                    original_geometry["chart_type"] == "line"
                    and int(candidate_row["prefilter"]["bar_rectangles"]) >= 3
                ):
                    continue
                if (
                    original_geometry["chart_type"] == "line"
                    and float(candidate_row["prefilter"]["color_fraction"]) > 0.12
                ):
                    continue
                if not label_preserves_geometry(label, original_geometry, margin=3.0):
                    continue
                masked, pixel_mask = mask_value_text(image, label)
                raw_masked_ocr = reader.readtext(
                    masked,
                    detail=1,
                    paragraph=False,
                    min_size=7,
                    text_threshold=0.45,
                    low_text=0.25,
                    link_threshold=0.35,
                    canvas_size=2560,
                    mag_ratio=1.25,
                )
                masked_tokens = ocr_tokens(raw_masked_ocr)
                height, width = masked.shape[:2]
                masked_axis = axis_localizer(masked_tokens, width, height)
                if masked_axis is None:
                    continue
                if masked_axis["right_axis_detected"] or int(masked_axis.get("additional_y_axis_count", 0)) > 0:
                    continue
                # Geometry receives only target location. Text and pseudo-Gold are deliberately removed.
                target_locator = replace(label, text="", numeric_value=None, numeric_suffix="", confidence=0.0)
                masked_geometry = choose_geometry_target(masked, target_locator, masked_axis)
                if masked_geometry is None:
                    continue
                if masked_geometry["chart_type"] != original_geometry["chart_type"]:
                    continue
                if not label_preserves_geometry(label, masked_geometry, margin=2.0):
                    continue
                pseudo_gold = float(label.numeric_value)
                raw_value = geometry_value(masked_geometry, masked_axis)
                axis_span = max(float(masked_axis["axis_span"]), 1e-12)
                absolute_error = abs(raw_value - pseudo_gold)
                relative_error = absolute_error / max(abs(pseudo_gold), 1e-12)
                axis_normalized_error = absolute_error / axis_span
                sample_id = f"fgmvp_{len(existing_samples):04d}"
                masked_path = args.project / "masked_images" / f"{sample_id}.png"
                overlay_path = args.project / "debug_overlays" / f"{sample_id}.png"
                masked_path.parent.mkdir(parents=True, exist_ok=True)
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(masked_path), masked)
                overlay = draw_debug_overlay(
                    image,
                    label,
                    masked_axis,
                    masked_geometry,
                    raw_value,
                    pseudo_gold,
                )
                cv2.imwrite(str(overlay_path), overlay)
                sample = {
                    "sample_id": sample_id,
                    "chart_id": chart_id,
                    "representative_sample_id": candidate_row["representative_sample_id"],
                    "source_image_path": candidate_row["image_path"],
                    "masked_image_path": str(masked_path),
                    "debug_overlay_path": str(overlay_path),
                    "label": label.as_dict(),
                    "crop_confirmation": confirmation,
                    "mask_pixel_count": int(np.count_nonzero(pixel_mask)),
                    "axis": masked_axis,
                    "geometry": masked_geometry,
                    "pseudo_gold": pseudo_gold,
                    "raw_geometry_value": raw_value,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                    "axis_normalized_error": axis_normalized_error,
                    "finmme_numerical_sample_ids": candidate_row["numerical_sample_ids"],
                    "finmme_numerical_gold": candidate_row["numerical_gold"],
                    "finmme_numerical_unit": candidate_row["numerical_unit"],
                    "finmme_numerical_tolerance": candidate_row["numerical_tolerance"],
                    "geometry_gold_access": False,
                }
                append_jsonl(sample_jsonl, sample)
                existing_samples.append(sample)
                attempt["status"] = "accepted"
                attempt["sample_id"] = sample_id
                break
        attempt["seconds"] = time.monotonic() - attempt_start
        append_jsonl(attempts_jsonl, attempt)
        if new_attempts % args.progress_every == 0 or attempt["status"] == "accepted":
            type_counts: dict[str, int] = {}
            for sample in existing_samples:
                chart_type = sample["geometry"]["chart_type"]
                type_counts[chart_type] = type_counts.get(chart_type, 0) + 1
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "new_attempts": new_attempts,
                        "samples": len(existing_samples),
                        "type_counts": type_counts,
                        "last_status": attempt["status"],
                        "elapsed_seconds": time.monotonic() - start,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    flat_rows = [flat_sample_row(row) for row in existing_samples]
    write_csv(args.project / "results" / "geometry_samples.csv", flat_rows)
    print(
        json.dumps(
            {
                "event": "complete",
                "new_attempts": new_attempts,
                "samples": len(existing_samples),
                "elapsed_seconds": time.monotonic() - start,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
