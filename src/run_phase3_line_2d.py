from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
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

from calibrate_geometry import feature_vector
from geometry_core import (
    axis_localizer,
    choose_geometry_target,
    data_label_candidates,
    get_edgepoint,
    interpolate_pixel_to_value,
    label_preserves_geometry,
    mask_value_text,
    ocr_tokens,
    token_from_dict,
)
from phase3_line_core import (
    auto_geometry_pipeline,
    detect_dominant_line_series,
    discover_x_axis_anchors,
    draw_phase3_overlay,
    geometry_value,
    infer_construction_target,
)
from run_geometry_samples import crop_confirm_label


SEED = 20260816


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json_safe(value: Any) -> Any:
    if value is None or value == "":
        return None
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return None


def make_reader(model_dir: Path) -> easyocr.Reader:
    return easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=str(model_dir),
        download_enabled=False,
        verbose=False,
    )


def full_ocr(reader: easyocr.Reader, image: np.ndarray) -> list[Any]:
    return ocr_tokens(
        reader.readtext(
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
    )


def scan_additional_line_candidates(args: argparse.Namespace, reader: easyocr.Reader) -> None:
    output = args.project / "phase3_audit" / "additional_line_scan.jsonl"
    already = {str(row["chart_id"]) for row in read_jsonl(output)}
    existing_scan_ids = {str(row["chart_id"]) for row in read_jsonl(args.candidate_scan)}
    manifest = load_csv(args.manifest)
    pool = [
        row
        for row in manifest
        if str(row["chart_id"]) not in existing_scan_ids
        and int(row["has_numerical"]) == 1
        and int(row["keyword_exclude"]) == 0
        and int(row["image_width"]) >= 300
        and int(row["image_height"]) >= 220
        and int(row["bar_rectangles"]) < 3
        and 2 <= int(row["sloped_segments"]) <= 70
        and float(row["color_fraction"]) <= 0.12
    ]
    pool.sort(
        key=lambda row: (
            -int(row["keyword_include"]),
            abs(int(row["sloped_segments"]) - 14),
            int(row["bar_rectangles"]),
            -float(row["prefilter_priority"]),
            row["chart_id"],
        )
    )
    started = time.monotonic()
    new_count = 0
    status_counts: Counter[str] = Counter()
    for manifest_row in pool:
        chart_id = str(manifest_row["chart_id"])
        if chart_id in already:
            continue
        image = cv2.imread(manifest_row["image_path"], cv2.IMREAD_COLOR)
        result: dict[str, Any] = {
            "chart_id": chart_id,
            "representative_sample_id": manifest_row["representative_sample_id"],
            "image_path": manifest_row["image_path"],
            "caption": manifest_row["caption"],
            "numerical_sample_ids": manifest_row["numerical_sample_ids"],
            "numerical_gold": manifest_row["numerical_gold"],
            "numerical_unit": manifest_row["numerical_unit"],
            "numerical_tolerance": manifest_row["numerical_tolerance"],
            "prefilter": {
                "priority": float(manifest_row["prefilter_priority"]),
                "bar_rectangles": int(manifest_row["bar_rectangles"]),
                "sloped_segments": int(manifest_row["sloped_segments"]),
                "color_fraction": float(manifest_row["color_fraction"]),
            },
        }
        if image is None:
            result["status"] = "image_read_failed"
        else:
            tokens = full_ocr(reader, image)
            height, width = image.shape[:2]
            result["image_size"] = [width, height]
            result["ocr_tokens"] = [token.as_dict() for token in tokens]
            axis = axis_localizer(tokens, width, height)
            if axis is None:
                result["status"] = "no_linear_left_y_axis"
            elif axis.get("right_axis_detected") or int(axis.get("additional_y_axis_count", 0)) > 0:
                result["status"] = "multiple_y_axes_or_panels_detected"
                result["axis"] = axis
            else:
                result["axis"] = axis
                viable: list[dict[str, Any]] = []
                for label in data_label_candidates(tokens, axis, width, height):
                    geometry = choose_geometry_target(image, label, axis)
                    if geometry is None or geometry.get("chart_type") != "line":
                        continue
                    if not label_preserves_geometry(label, geometry):
                        continue
                    viable.append(
                        {
                            "label": label.as_dict(),
                            "geometry_on_original": geometry,
                            "construction_score": float(label.confidence) + float(geometry["score"]),
                        }
                    )
                viable.sort(key=lambda row: -float(row["construction_score"]))
                result["viable_labels"] = viable
                result["status"] = "viable_line" if viable else "no_geometry_linked_line_label"
        append_jsonl(output, result)
        new_count += 1
        status_counts[str(result["status"])] += 1
        if new_count % args.progress_every == 0 or result["status"] == "viable_line":
            print(
                json.dumps(
                    {
                        "event": "phase3_additional_scan_progress",
                        "new": new_count,
                        "total_pool": len(pool),
                        "last_chart": chart_id,
                        "last_status": result["status"],
                        "elapsed_seconds": time.monotonic() - started,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    print(
        json.dumps(
            {
                "event": "phase3_additional_scan_complete",
                "new": new_count,
                "pool": len(pool),
                "status_counts": dict(status_counts),
                "elapsed_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def line_candidate_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(args.candidate_scan):
        line_labels = [
            item
            for item in row.get("viable_labels", [])
            if (item.get("geometry_on_original") or {}).get("chart_type") == "line"
        ]
        if line_labels:
            copied = dict(row)
            copied["viable_labels"] = line_labels
            copied["candidate_source"] = "phase1_candidate_scan_v3"
            rows.append(copied)
    for row in read_jsonl(args.project / "phase3_audit" / "additional_line_scan.jsonl"):
        if row.get("status") == "viable_line" and row.get("viable_labels"):
            copied = dict(row)
            copied["candidate_source"] = "phase3_additional_line_scan"
            rows.append(copied)
    by_chart: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_chart[str(row["chart_id"])] = row
    return list(by_chart.values())


def load_decisions(path: Path) -> dict[str, dict[str, str]]:
    return {str(row["chart_id"]): row for row in load_csv(path)}


def construction_line_x(
    series_mask: np.ndarray | None,
    label: Any,
    pseudo_gold: float,
    axis: dict[str, Any],
) -> float:
    if series_mask is None:
        return float(label.cx)
    expected_y = (pseudo_gold - float(axis["intercept"])) / float(axis["slope"])
    ys, xs = np.nonzero(series_mask)
    height, width = series_mask.shape
    keep = (
        (np.abs(ys.astype(float) - expected_y) <= max(5.0, 0.014 * height))
        & (xs >= label.cx - 0.13 * width)
        & (xs <= label.cx + 0.13 * width)
    )
    if int(np.count_nonzero(keep)) == 0:
        return float(label.cx)
    candidate_x = xs[keep].astype(float)
    candidate_y = ys[keep].astype(float)
    cost = 2.5 * np.abs(candidate_y - expected_y) + 0.12 * np.abs(candidate_x - label.cx)
    best_cost = float(np.min(cost))
    near = cost <= best_cost + 1.5
    return float(np.median(candidate_x[near]))


def mask_value_text_preserve_line(
    image_bgr: np.ndarray,
    label: Any,
    series_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    polygon = np.asarray(label.polygon, dtype=np.float32)
    center = polygon.mean(axis=0, keepdims=True)
    expanded = center + 1.04 * (polygon - center)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, image_bgr.shape[1] - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, image_bgr.shape[0] - 1)
    cv2.fillConvexPoly(mask, np.round(expanded).astype(np.int32), 255)
    protected_pixels = 0
    if series_mask is not None:
        protected = cv2.dilate(series_mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        protected_pixels = int(np.count_nonzero((mask > 0) & protected))
        mask[protected] = 0
    masked = cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)
    return masked, mask, protected_pixels


def relative_project_path(project: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project))
    except ValueError:
        return str(path)


def construct_samples(args: argparse.Namespace, reader: easyocr.Reader) -> None:
    decisions = load_decisions(args.chart_decisions)
    candidates = {str(row["chart_id"]): row for row in line_candidate_rows(args)}
    screening_rows: list[dict[str, Any]] = []
    for chart_id, row in sorted(candidates.items()):
        decision = decisions.get(chart_id)
        screening_rows.append(
            {
                "chart_id": chart_id,
                "representative_sample_id": row.get("representative_sample_id"),
                "image_path": row.get("image_path"),
                "candidate_source": row.get("candidate_source"),
                "viable_line_label_count": len(row.get("viable_labels", [])),
                "include_simple_single_line": 0 if decision is None else int(decision.get("include", "0")),
                "screening_reason": "not_reviewed" if decision is None else decision.get("reason", ""),
                "review_basis": "not_reviewed" if decision is None else decision.get("review_basis", ""),
            }
        )
    write_csv(args.project / "results" / "phase3_chart_screening.csv", screening_rows)

    selected = [
        candidates[chart_id]
        for chart_id, decision in decisions.items()
        if int(decision.get("include", "0")) == 1 and chart_id in candidates
    ]
    selected.sort(key=lambda row: str(row["chart_id"]))
    samples: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    masked_dir = args.project / "phase3_masked_images"
    overlay_dir = args.project / "phase3_debug_overlays"
    masked_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for chart_index, row in enumerate(selected, 1):
        chart_id = str(row["chart_id"])
        image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            attempts.append({"chart_id": chart_id, "status": "image_read_failed"})
            continue
        original_tokens = full_ocr(reader, image)
        height, width = image.shape[:2]
        original_axis = axis_localizer(original_tokens, width, height)
        if original_axis is None:
            attempts.append({"chart_id": chart_id, "status": "original_axis_unrecoverable"})
            continue
        original_anchors, original_anchor_audit = discover_x_axis_anchors(reader, image, original_tokens, original_axis)
        construction_series, construction_mask = detect_dominant_line_series(image, original_axis)
        chart_accepts = 0
        labels = sorted(row.get("viable_labels", []), key=lambda item: -float(item["construction_score"]))
        for label_index, label_item in enumerate(labels):
            attempt: dict[str, Any] = {
                "chart_id": chart_id,
                "candidate_label_index": label_index,
                "label_text": label_item["label"]["text"],
                "status": "rejected",
            }
            label = token_from_dict(label_item["label"])
            confirmation = crop_confirm_label(reader, image, label)
            if confirmation is None:
                attempt["reason"] = "crop_ocr_did_not_confirm_value"
                attempts.append(attempt)
                continue
            pseudo_gold = float(label.numeric_value)
            construction_x = construction_line_x(construction_mask, label, pseudo_gold, original_axis)
            semantic = infer_construction_target(construction_x, original_anchors, width)
            if semantic is None:
                attempt["reason"] = "no_reliable_semantic_x_target"
                attempt["construction_x"] = construction_x
                attempts.append(attempt)
                continue
            question = f"What is the value for {semantic['target_semantic_label']}?"
            masked, pixel_mask, protected_line_pixels = mask_value_text_preserve_line(
                image,
                label,
                construction_mask,
            )
            sample_id = f"phase3_line_{len(samples) + 1:04d}"
            masked_path = masked_dir / f"{sample_id}.png"
            cv2.imwrite(str(masked_path), masked)

            # The automatic parser intentionally receives only the masked image and question.
            auto = auto_geometry_pipeline(reader, masked, question)
            target_locator = replace(label, text="", numeric_value=None, numeric_suffix="", confidence=0.0)
            oracle_axis = auto.get("axis")
            oracle_geometry = None if oracle_axis is None else get_edgepoint(masked, target_locator, oracle_axis)
            oracle_value = None if oracle_geometry is None or oracle_axis is None else interpolate_pixel_to_value(float(oracle_geometry["target_y"]), oracle_axis)
            auto_column = auto.get("auto_column_geometry")
            auto_local = auto.get("auto_local_geometry")
            auto_column_value = None if oracle_axis is None else geometry_value(auto_column, oracle_axis)
            auto_local_value = None if oracle_axis is None else geometry_value(auto_local, oracle_axis)
            sample = {
                "sample_id": sample_id,
                "chart_id": chart_id,
                "representative_sample_id": row.get("representative_sample_id"),
                "candidate_source": row.get("candidate_source"),
                "source_image_path": row["image_path"],
                "masked_image_path": relative_project_path(args.project, masked_path),
                "question": question,
                "semantic_target": semantic,
                "parser_runtime_contract": {
                    "inputs": ["masked_chart", "question"],
                    "forbidden_inputs": ["pseudo_gold", "masked_value_label_bbox", "masked_value_label_center_x"],
                },
                "auto_status": auto.get("status"),
                "x_axis_anchors": auto.get("x_axis_anchors", []),
                "x_axis_anchor_audit": auto.get("x_axis_anchor_audit"),
                "grounding": auto.get("grounding"),
                "local_search_window": auto.get("local_search_window"),
                "detected_series": auto.get("series"),
                "axis": oracle_axis,
                "oracle_x_geometry": oracle_geometry,
                "auto_x_column_geometry": auto_column,
                "auto_x_local_geometry": auto_local,
                "oracle_x_value": oracle_value,
                "auto_x_column_value": auto_column_value,
                "auto_x_local_value": auto_local_value,
                "pseudo_gold": pseudo_gold,
                "mask_pixel_count": int(np.count_nonzero(pixel_mask)),
                "protected_line_pixel_count": protected_line_pixels,
                "mask_overlaps_protected_series_pixels": 0,
                "construction_only": {
                    "masked_value_label": label.as_dict(),
                    "crop_confirmation": confirmation,
                    "original_x_axis_anchors": original_anchors,
                    "original_x_axis_anchor_audit": original_anchor_audit,
                    "construction_series": construction_series,
                    "construction_line_x": construction_x,
                    "old_phase1_value_label_center_x": label.cx,
                    "old_geometry_on_original": label_item["geometry_on_original"],
                },
            }
            samples.append(sample)
            chart_accepts += 1
            attempt["status"] = "accepted"
            attempt["sample_id"] = sample_id
            attempt["semantic_mode"] = semantic["construction_mode"]
            attempt["auto_status"] = auto.get("status")
            attempts.append(attempt)

            overlay_axis = oracle_axis or original_axis
            overlay_anchors = auto.get("x_axis_anchors", original_anchors)
            overlay_grounding = auto.get("grounding") or {
                "target_semantic_label": semantic["target_semantic_label"],
                "mode": "x_grounding_unrecoverable",
            }
            variants = [
                ("oracle_x", oracle_geometry, oracle_value),
                ("auto_column", auto_column, auto_column_value),
                ("auto_local", auto_local, auto_local_value),
            ]
            for method_name, geometry, value in variants:
                overlay = draw_phase3_overlay(
                    image,
                    sample_id,
                    method_name,
                    overlay_anchors,
                    overlay_grounding,
                    overlay_axis,
                    geometry,
                    auto.get("series"),
                    value,
                    pseudo_gold,
                    oracle_geometry,
                )
                cv2.imwrite(str(overlay_dir / f"{sample_id}_{method_name}.png"), overlay)
        if chart_accepts == 0:
            attempts.append({"chart_id": chart_id, "status": "no_label_accepted"})
        print(
            json.dumps(
                {
                    "event": "phase3_construct_chart",
                    "chart_index": chart_index,
                    "selected_charts": len(selected),
                    "chart_id": chart_id,
                    "accepted_on_chart": chart_accepts,
                    "samples_total": len(samples),
                    "elapsed_seconds": time.monotonic() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    write_jsonl(args.project / "phase3_audit" / "line_samples.jsonl", samples)
    write_jsonl(args.project / "phase3_audit" / "construction_attempts.jsonl", attempts)
    print(json.dumps({"event": "phase3_construct_complete", "samples": len(samples), "charts": len({row['chart_id'] for row in samples})}, sort_keys=True), flush=True)


def stable_hash(text: str) -> str:
    return hashlib.sha256(f"{SEED}:{text}".encode("utf-8")).hexdigest()


def chart_splits(rows: list[dict[str, Any]]) -> dict[str, str]:
    chart_ids = sorted({str(row["chart_id"]) for row in rows}, key=stable_hash)
    if len(chart_ids) <= 1:
        return {chart_id: "test" for chart_id in chart_ids}
    test_count = max(1, min(len(chart_ids) - 1, int(round(0.30 * len(chart_ids)))))
    return {chart_id: ("test" if index < test_count else "train") for index, chart_id in enumerate(chart_ids)}


def error_fields(value: float | None, gold: float, axis_span: float) -> dict[str, Any]:
    if value is None or not math.isfinite(value):
        return {"prediction": None, "absolute_error": None, "axis_normalized_error": None, "signed_error": None}
    return {
        "prediction": value,
        "absolute_error": abs(value - gold),
        "axis_normalized_error": abs(value - gold) / max(axis_span, 1e-12),
        "signed_error": value - gold,
    }


def summarize_method(rows: list[dict[str, Any]], method: str, field: str) -> dict[str, Any]:
    eligible = [row for row in rows if row.get(field) is not None and math.isfinite(float(row[field]))]
    errors = np.asarray([abs(float(row[field]) - float(row["pseudo_gold"])) for row in eligible], dtype=float)
    axis_errors = np.asarray([error / max(float(row["axis"]["axis_span"]), 1e-12) for error, row in zip(errors, eligible)], dtype=float)
    return {
        "method": method,
        "eligible_samples": len(eligible),
        "total_samples": len(rows),
        "coverage": len(eligible) / max(len(rows), 1),
        "mae": None if len(errors) == 0 else float(np.mean(errors)),
        "median_absolute_error": None if len(errors) == 0 else float(np.median(errors)),
        "mean_axis_normalized_error": None if len(errors) == 0 else float(np.mean(axis_errors)),
        "median_axis_normalized_error": None if len(errors) == 0 else float(np.median(axis_errors)),
        "large_error_count_axis_gt_0_02": int(np.count_nonzero(axis_errors > 0.02)),
    }


def paired_cluster_bootstrap(
    rows: list[dict[str, Any]],
    left_field: str,
    right_field: str,
    iterations: int = 3000,
) -> dict[str, Any]:
    paired = [row for row in rows if row.get(left_field) is not None and row.get(right_field) is not None]
    by_chart: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        by_chart[str(row["chart_id"])].append(row)
    charts = sorted(by_chart)
    if not charts:
        return {"paired_samples": 0, "paired_charts": 0}
    observed = float(
        np.mean([abs(float(row[left_field]) - float(row["pseudo_gold"])) - abs(float(row[right_field]) - float(row["pseudo_gold"])) for row in paired])
    )
    rng = np.random.default_rng(SEED)
    draws: list[float] = []
    for _ in range(iterations):
        sampled_charts = rng.choice(charts, size=len(charts), replace=True)
        sampled_rows = [row for chart_id in sampled_charts for row in by_chart[str(chart_id)]]
        draws.append(
            float(
                np.mean(
                    [
                        abs(float(row[left_field]) - float(row["pseudo_gold"]))
                        - abs(float(row[right_field]) - float(row["pseudo_gold"]))
                        for row in sampled_rows
                    ]
                )
            )
        )
    return {
        "paired_samples": len(paired),
        "paired_charts": len(charts),
        "left_mae_minus_right_mae": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_left_error_gt_right": float(np.mean(np.asarray(draws) > 0)),
        "iterations": iterations,
    }


def phase1_biases(project: Path) -> dict[str, float]:
    rows = read_jsonl(project / "audit" / "geometry_samples.jsonl")
    split = {row["chart_id"]: row["split"] for row in load_csv(project / "results" / "chart_split.csv")}
    train = [row for row in rows if split.get(str(row["chart_id"])) == "train"]
    line = [row for row in train if row.get("geometry", {}).get("chart_type") == "line"]
    return {
        "global_mean_bias": float(np.mean([float(row["pseudo_gold"]) - float(row["raw_geometry_value"]) for row in train])),
        "line_type_mean_bias": float(np.mean([float(row["pseudo_gold"]) - float(row["raw_geometry_value"]) for row in line])),
        "global_median_bias": float(np.median([float(row["pseudo_gold"]) - float(row["raw_geometry_value"]) for row in train])),
        "line_type_median_bias": float(np.median([float(row["pseudo_gold"]) - float(row["raw_geometry_value"]) for row in line])),
        "phase1_train_samples": float(len(train)),
        "phase1_train_line_samples": float(len(line)),
    }


def learned_calibrated_value(project: Path, row: dict[str, Any]) -> float | None:
    raw_value = row.get("auto_x_local_value")
    geometry = row.get("auto_x_local_geometry")
    axis = row.get("axis")
    if raw_value is None or geometry is None or axis is None:
        return None
    model = json.loads((project / "config" / "calibrator_model.json").read_text(encoding="utf-8"))
    feature_row = {"axis": axis, "geometry": geometry, "raw_geometry_value": raw_value}
    vector = feature_vector(feature_row)
    mean = np.asarray(model["feature_mean"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    normalized = (vector - mean) / np.where(scale < 1e-12, 1.0, scale)
    predicted_normalized_error = float(coefficients[0] + normalized @ coefficients[1:])
    return float(raw_value) + predicted_normalized_error * float(axis["axis_span"])


def failure_attribution(row: dict[str, Any]) -> str:
    if row.get("auto_status") not in {"ok", None}:
        status = str(row.get("auto_status"))
        if status.startswith("x_"):
            return "x_grounding"
        if "series" in status:
            return "series_matching"
        if "point" in status:
            return "line_fitting"
        if "axis" in status:
            return "y_axis_mapping"
        return "unrecoverable_other"
    grounding = row.get("grounding") or {}
    semantic = row.get("semantic_target") or {}
    predicted_x = grounding.get("predicted_target_x")
    construction_x = semantic.get("construction_target_x")
    anchors = row.get("x_axis_anchors") or []
    if predicted_x is None or construction_x is None:
        return "x_grounding"
    anchor_x = sorted(float(anchor["center_x"]) for anchor in anchors)
    gap = float(np.median(np.diff(anchor_x))) if len(anchor_x) >= 2 else 40.0
    if abs(float(predicted_x) - float(construction_x)) > max(8.0, 0.28 * gap):
        return "x_grounding"
    series = row.get("detected_series") or {}
    geometry = row.get("auto_x_local_geometry")
    axis = row.get("axis")
    if geometry is None or axis is None:
        return "line_fitting"
    raw_value = row.get("auto_x_local_value")
    if raw_value is not None:
        axis_error = abs(float(raw_value) - float(row["pseudo_gold"])) / max(float(axis["axis_span"]), 1e-12)
        if axis_error <= 0.01:
            return "within_expected_error"
    construction_series = (row.get("construction_only") or {}).get("construction_series") or {}
    detected_color = np.asarray(series.get("median_bgr", []), dtype=float)
    construction_color = np.asarray(construction_series.get("median_bgr", []), dtype=float)
    if detected_color.shape == (3,) and construction_color.shape == (3,):
        if float(np.linalg.norm(detected_color - construction_color)) > 75.0:
            return "series_matching"
    if float(axis.get("r_squared", 0.0)) < 0.995 or float(axis.get("max_residual_pixels", 99.0)) > 3.0:
        return "y_axis_mapping"
    gold_y = (float(row["pseudo_gold"]) - float(axis["intercept"])) / float(axis["slope"])
    plot_height = float(axis["plot_bbox"][3]) - float(axis["plot_bbox"][1])
    if abs(float(geometry["target_y"]) - gold_y) > max(4.0, 0.012 * plot_height):
        return "line_fitting"
    return "within_expected_error"


def analyze_samples(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.project / "phase3_audit" / "line_samples.jsonl")
    splits = chart_splits(rows)
    split_rows = [
        {
            "chart_id": chart_id,
            "split": split,
            "sample_count": sum(str(row["chart_id"]) == chart_id for row in rows),
            "leakage_check": "chart_disjoint",
        }
        for chart_id, split in sorted(splits.items())
    ]
    write_csv(args.project / "results" / "phase3_chart_split.csv", split_rows)
    phase1_split = {row["chart_id"]: row["split"] for row in load_csv(args.project / "results" / "chart_split.csv")}
    biases = phase1_biases(args.project)
    flat: list[dict[str, Any]] = []
    for row in rows:
        axis = row.get("axis")
        axis_span = math.nan if axis is None else float(axis["axis_span"])
        gold = float(row["pseudo_gold"])
        grounding = row.get("grounding") or {}
        semantic = row.get("semantic_target") or {}
        local_value = row.get("auto_x_local_value")
        global_bias = None if local_value is None else float(local_value) + biases["global_mean_bias"]
        type_bias = None if local_value is None else float(local_value) + biases["line_type_mean_bias"]
        global_median_bias = None if local_value is None else float(local_value) + biases["global_median_bias"]
        type_median_bias = None if local_value is None else float(local_value) + biases["line_type_median_bias"]
        learned = learned_calibrated_value(args.project, row)
        calibration_eligible = phase1_split.get(str(row["chart_id"])) != "train"
        attribution = failure_attribution(row)
        output = {
            "sample_id": row["sample_id"],
            "chart_id": row["chart_id"],
            "phase3_split": splits.get(str(row["chart_id"])),
            "question": row["question"],
            "target_semantic_label": semantic.get("target_semantic_label"),
            "construction_semantic_mode": semantic.get("construction_mode"),
            "detected_x_axis_label": grounding.get("matched_detected_label"),
            "x_grounding_mode": grounding.get("mode"),
            "predicted_target_x": grounding.get("predicted_target_x"),
            "construction_target_x": semantic.get("construction_target_x"),
            "old_value_label_center_x": (row.get("construction_only") or {}).get("old_phase1_value_label_center_x"),
            "x_grounding_error_pixels": None if grounding.get("predicted_target_x") is None else abs(float(grounding["predicted_target_x"]) - float(semantic["construction_target_x"])),
            "local_search_window": json.dumps(row.get("local_search_window"), separators=(",", ":")),
            "selected_series_id": (row.get("detected_series") or {}).get("series_id"),
            "selected_series_color_bgr": json.dumps((row.get("detected_series") or {}).get("median_bgr"), separators=(",", ":")),
            "series_candidate_count": (row.get("detected_series") or {}).get("series_candidate_count"),
            "comparable_distinct_series_count": (row.get("detected_series") or {}).get("comparable_distinct_series_count"),
            "series_color_distance_to_construction": None
            if not (row.get("detected_series") or {}).get("median_bgr") or not ((row.get("construction_only") or {}).get("construction_series") or {}).get("median_bgr")
            else float(
                np.linalg.norm(
                    np.asarray((row.get("detected_series") or {})["median_bgr"], dtype=float)
                    - np.asarray(((row.get("construction_only") or {})["construction_series"])["median_bgr"], dtype=float)
                )
            ),
            "fitted_local_line": json.dumps((row.get("auto_x_local_geometry") or {}).get("fitted_segments"), separators=(",", ":")),
            "final_auto_local_point": json.dumps((row.get("auto_x_local_geometry") or {}).get("point"), separators=(",", ":")),
            "y_axis_ticks": json.dumps([] if axis is None else axis.get("ticks", []), separators=(",", ":")),
            "y_axis_slope": None if axis is None else axis.get("slope"),
            "y_axis_intercept": None if axis is None else axis.get("intercept"),
            "axis_span": axis_span,
            "oracle_x_value": row.get("oracle_x_value"),
            "auto_x_column_value": row.get("auto_x_column_value"),
            "auto_x_local_value": local_value,
            "global_constant_bias_value": global_bias,
            "line_type_constant_bias_value": type_bias,
            "global_median_constant_bias_value": global_median_bias,
            "line_type_median_constant_bias_value": type_median_bias,
            "learned_calibrated_value": learned,
            "pseudo_gold": gold,
            "auto_status": row.get("auto_status"),
            "failure_attribution": attribution,
            "calibration_eval_eligible_unseen_to_phase1_train": int(calibration_eligible),
            "parser_used_hidden_bbox_or_gold": 0,
            "x_axis_anchors": json.dumps(row.get("x_axis_anchors", []), separators=(",", ":")),
            "masked_image_path": row["masked_image_path"],
            "source_image_path": row["source_image_path"],
        }
        for prefix, field in [
            ("oracle_x", "oracle_x_value"),
            ("auto_x_column", "auto_x_column_value"),
            ("auto_x_local", "auto_x_local_value"),
        ]:
            for key, value in error_fields(row.get(field), gold, axis_span).items():
                output[f"{prefix}_{key}"] = value
        flat.append(output)
    write_csv(args.project / "results" / "phase3_line_samples.csv", flat)
    metric_rows = [
        summarize_method(rows, "oracle_x_old_phase1_local_fit", "oracle_x_value"),
        summarize_method(rows, "auto_x_column", "auto_x_column_value"),
        summarize_method(rows, "auto_x_local_line_fit", "auto_x_local_value"),
    ]
    write_csv(args.project / "results" / "phase3_method_metrics.csv", metric_rows)
    pairwise = [
        {"comparison": "auto_x_column_minus_oracle_x", **paired_cluster_bootstrap(rows, "auto_x_column_value", "oracle_x_value")},
        {"comparison": "auto_x_local_minus_oracle_x", **paired_cluster_bootstrap(rows, "auto_x_local_value", "oracle_x_value")},
        {"comparison": "auto_x_column_minus_auto_x_local", **paired_cluster_bootstrap(rows, "auto_x_column_value", "auto_x_local_value")},
    ]
    write_csv(args.project / "results" / "phase3_pairwise_metrics.csv", pairwise)

    local_effects: list[dict[str, Any]] = []
    for row in flat:
        column = row.get("auto_x_column_value")
        local = row.get("auto_x_local_value")
        if column in {None, ""} and local not in {None, ""}:
            outcome = "recovered_missing_column_point"
            delta = None
        elif column not in {None, ""} and local not in {None, ""}:
            column_error = abs(float(column) - float(row["pseudo_gold"]))
            local_error = abs(float(local) - float(row["pseudo_gold"]))
            delta = column_error - local_error
            outcome = "improved" if delta > 1e-9 else "worsened" if delta < -1e-9 else "tied"
        else:
            outcome = "both_unrecoverable"
            delta = None
        local_effects.append(
            {
                "sample_id": row["sample_id"],
                "chart_id": row["chart_id"],
                "target_semantic_label": row["target_semantic_label"],
                "column_absolute_error": None if column in {None, ""} else abs(float(column) - float(row["pseudo_gold"])),
                "local_absolute_error": None if local in {None, ""} else abs(float(local) - float(row["pseudo_gold"])),
                "column_error_minus_local_error": delta,
                "local_fitting_outcome": outcome,
                "local_fit_residual_pixels": None
                if not row.get("fitted_local_line")
                else (read_json_safe(row.get("fitted_local_line")) or [{}])[0].get("median_residual"),
            }
        )
    write_csv(args.project / "results" / "phase3_local_fitting_effects.csv", local_effects)

    calibration_rows = [row for row in flat if int(row["calibration_eval_eligible_unseen_to_phase1_train"]) == 1]
    calibration_metrics: list[dict[str, Any]] = []
    for method, field in [
        ("raw_auto_x_local", "auto_x_local_value"),
        ("global_mean_constant_bias_phase1_train", "global_constant_bias_value"),
        ("line_type_mean_constant_bias_phase1_train", "line_type_constant_bias_value"),
        ("global_median_constant_bias_phase1_train", "global_median_constant_bias_value"),
        ("line_type_median_constant_bias_phase1_train", "line_type_median_constant_bias_value"),
        ("learned_phase1_calibrator", "learned_calibrated_value"),
    ]:
        eligible = [row for row in calibration_rows if row.get(field) not in {None, ""}]
        errors = np.asarray([abs(float(row[field]) - float(row["pseudo_gold"])) for row in eligible], dtype=float)
        axis_errors = np.asarray([error / max(float(row["axis_span"]), 1e-12) for error, row in zip(errors, eligible)], dtype=float)
        calibration_metrics.append(
            {
                "method": method,
                "evaluation_scope": "phase3 charts unseen to Phase1 calibrator train",
                "eligible_samples": len(eligible),
                "eligible_charts": len({row["chart_id"] for row in eligible}),
                "mae": None if len(errors) == 0 else float(np.mean(errors)),
                "median_absolute_error": None if len(errors) == 0 else float(np.median(errors)),
                "mean_axis_normalized_error": None if len(errors) == 0 else float(np.mean(axis_errors)),
            }
        )
    write_csv(args.project / "results" / "phase3_calibration_bias_metrics.csv", calibration_metrics)
    write_csv(
        args.project / "results" / "phase3_bias_constants.csv",
        [{"constant": key, "value": value, "training_source": "frozen Phase1 train charts"} for key, value in biases.items()],
    )
    attribution_counts = Counter(row["failure_attribution"] for row in flat)
    write_csv(
        args.project / "results" / "phase3_failure_attribution.csv",
        [
            {
                "failure_attribution": key,
                "sample_count": value,
                "fraction": value / max(len(flat), 1),
            }
            for key, value in sorted(attribution_counts.items())
        ],
    )
    summary = {
        "sample_count": len(rows),
        "chart_count": len({row["chart_id"] for row in rows}),
        "direct_grounding_samples": sum((row.get("grounding") or {}).get("mode") == "direct_anchor" for row in rows),
        "interpolated_grounding_samples": sum((row.get("grounding") or {}).get("mode") == "piecewise_linear_interpolation" for row in rows),
        "x_grounding_unrecoverable_samples": sum(str(row.get("auto_status", "")).startswith("x_") for row in rows),
        "overlay_count": len(list((args.project / "phase3_debug_overlays").glob("*.png"))),
        "train_test_chart_overlap": len(
            {chart for chart, split in splits.items() if split == "train"}
            & {chart for chart, split in splits.items() if split == "test"}
        ),
        "method_metrics": metric_rows,
        "pairwise": pairwise,
        "calibration_metrics": calibration_metrics,
        "failure_attribution": dict(attribution_counts),
        "local_fitting_effect_counts": dict(Counter(row["local_fitting_outcome"] for row in local_effects)),
    }
    (args.project / "results" / "phase3_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "phase3_analysis_complete", **summary}, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-scan", type=Path, required=True)
    parser.add_argument("--chart-decisions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["scan", "construct", "analyze", "all"], default="all")
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(6)
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)
    reader = None
    if args.stage in {"scan", "construct", "all"}:
        reader = make_reader(args.model_dir)
    if args.stage in {"scan", "all"}:
        scan_additional_line_candidates(args, reader)
    if args.stage in {"construct", "all"}:
        construct_samples(args, reader)
    if args.stage in {"analyze", "all"}:
        analyze_samples(args)


if __name__ == "__main__":
    main()
