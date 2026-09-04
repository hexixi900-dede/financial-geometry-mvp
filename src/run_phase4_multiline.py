"""Phase 4 runner: sample construction, no-leak experiments, and analysis.

Stages:
- construct: build masked-value samples from the Phase 4 full-corpus scan,
  run Auto-Series and (multi-line only) Oracle-Series configurations, and
  write per-sample overlays for both successes and rejections.
- analyze: chart-disjoint split, stratified metrics, calibration controls,
  rejection audit, Phase 3 overlap diagnostic.

No Phase 1/2/3 output file is read for anything except the frozen Phase 1
calibration/bias constants and the frozen Phase 3 overlap comparison; Phase 4
writes only phase4-prefixed artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
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
    interpolate_pixel_to_value,
    ocr_tokens,
    token_from_dict,
)
from phase3_line_core import discover_x_axis_anchors, infer_construction_target
from phase4_legend import detect_legend_entries
from phase4_multiline_core import auto_multiline_pipeline, draw_phase4_overlay
from phase4_series import color_distance, detect_line_series_components, match_series_by_color
from run_geometry_samples import crop_confirm_label
from run_phase3_line_2d import construction_line_x


SEED = 20260819
MAX_SAMPLES_PER_CHART = 6


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


def is_area_like(component: dict[str, Any], axis: dict[str, Any]) -> bool:
    plot_height = max(1.0, float(axis["plot_bbox"][3]) - float(axis["plot_bbox"][1]))
    return (
        float(component["fill_ratio"]) >= 0.55
        and float(component["bbox"][3]) >= 0.45 * plot_height
        and float(component["coverage_fraction"]) >= 0.80
    )


def coherent_construction_anchors(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only anchors that can generate a meaningful semantic x target.

    Temporal anchors (year/quarter/month) are always kept. Numeric anchors are
    kept only when they form a systematic axis: at least three, sorted by x,
    non-decreasing values, and mostly unique. This mirrors the temporal
    coherence pass inside anchor discovery and stops bottom-area data labels or
    fragmented OCR digits from becoming degenerate construction targets such
    as "value for 25.44?" or "value for 8?".
    """
    temporal = [anchor for anchor in anchors if anchor.get("scalar_family") in {"year", "quarter", "month"}]
    numeric = [anchor for anchor in anchors if anchor.get("scalar_family") == "numeric" and anchor.get("scalar") is not None]
    keep = list(temporal)
    if len(numeric) >= 3:
        ordered = sorted(numeric, key=lambda anchor: float(anchor["center_x"]))
        values = [float(anchor["scalar"]) for anchor in ordered]
        diffs = np.diff(values)
        unique_fraction = len(set(values)) / len(values)
        if bool(np.all(diffs >= -1e-9)) and unique_fraction >= 0.8 and float(np.ptp(values)) > 0:
            keep.extend(ordered)
    return sorted(keep, key=lambda anchor: float(anchor["center_x"]))


def mask_value_text_protect_all(
    image_bgr: np.ndarray,
    label: Any,
    series_masks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    polygon = np.asarray(label.polygon, dtype=np.float32)
    center = polygon.mean(axis=0, keepdims=True)
    expanded = center + 1.04 * (polygon - center)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, image_bgr.shape[1] - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, image_bgr.shape[0] - 1)
    cv2.fillConvexPoly(mask, np.round(expanded).astype(np.int32), 255)
    protected_pixels = 0
    if series_masks:
        union = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        for series_mask in series_masks:
            union |= series_mask.astype(np.uint8)
        protected = cv2.dilate(union, np.ones((5, 5), np.uint8)) > 0
        protected_pixels = int(np.count_nonzero((mask > 0) & protected))
        mask[protected] = 0
    masked = cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)
    return masked, mask, protected_pixels


def associate_label_to_series(
    label: Any,
    pseudo_gold: float,
    series: list[dict[str, Any]],
    axis: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    """Construction-time only: use the label's own value to identify its line."""
    expected_y = (pseudo_gold - float(axis["intercept"])) / float(axis["slope"])
    band = max(5.0, 0.014 * height)
    scored: list[dict[str, Any]] = []
    for component in series:
        ys, xs = np.nonzero(component["mask"])
        keep = np.abs(ys.astype(float) - expected_y) <= band
        if int(np.count_nonzero(keep)) < 3:
            continue
        min_dx = float(np.min(np.abs(xs[keep].astype(float) - float(label.cx))))
        scored.append(
            {
                "series_id": int(component["series_id"]),
                "min_dx": min_dx,
                "band_pixel_count": int(np.count_nonzero(keep)),
            }
        )
    if not scored:
        return {"status": "rejected", "reason": "no_series_pixels_near_gold_y"}
    scored.sort(key=lambda item: (item["min_dx"], item["series_id"]))
    best = scored[0]
    if best["min_dx"] > max(12.0, 0.08 * width):
        return {"status": "rejected", "reason": "label_too_far_from_any_series", "best_dx": best["min_dx"]}
    crossing_note = False
    if len(scored) > 1 and scored[1]["min_dx"] <= best["min_dx"] + 3.0:
        crossing_note = True
    return {
        "status": "ok",
        "series_id": best["series_id"],
        "association_min_dx": best["min_dx"],
        "association_candidates": scored,
        "crossing_at_label": crossing_note,
    }


def relative_project_path(project: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project))
    except ValueError:
        return str(path)


def construct_samples(args: argparse.Namespace, reader: easyocr.Reader) -> None:
    scan_path = args.project / "phase4_audit" / "chart_scan_v2.jsonl"
    if not scan_path.exists():
        scan_path = args.project / "phase4_audit" / "chart_scan.jsonl"
    scan_rows = {
        str(row["chart_id"]): row
        for row in read_jsonl(scan_path)
        if str(row.get("status", "")).startswith("viable")
    }
    chart_ids = sorted(scan_rows)
    samples: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    masked_dir = args.project / "phase4_masked_images"
    overlay_dir = args.project / "phase4_debug_overlays"
    masked_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for chart_index, chart_id in enumerate(chart_ids, 1):
        scan = scan_rows[chart_id]
        chart_kind = "multi_line" if scan.get("status") == "viable_multi_line" else "single_line"
        chart_attempt: dict[str, Any] = {"chart_id": chart_id, "chart_kind": chart_kind, "labels_seen": 0}
        image = cv2.imread(str(scan["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            chart_attempt["status"] = "image_read_failed"
            attempts.append(chart_attempt)
            continue
        height, width = image.shape[:2]
        tokens = full_ocr(reader, image)
        axis = axis_localizer(tokens, width, height)
        if axis is None:
            chart_attempt["status"] = "original_axis_unrecoverable"
            attempts.append(chart_attempt)
            continue
        anchors, anchor_audit = discover_x_axis_anchors(reader, image, tokens, axis)
        construction_anchors = coherent_construction_anchors(anchors)
        scalar_anchors = [anchor for anchor in construction_anchors if anchor.get("scalar") is not None]
        if len(scalar_anchors) < 2:
            chart_attempt["status"] = "no_coherent_x_anchor_sequence"
            attempts.append(chart_attempt)
            continue
        if int(scan["prefilter"]["bar_rectangles"]) >= 3:
            chart_attempt["status"] = "bar_line_combination_out_of_scope"
            attempts.append(chart_attempt)
            continue
        series = detect_line_series_components(image, axis)
        series = [component for component in series if not is_area_like(component, axis)]
        if not series:
            chart_attempt["status"] = "no_line_series_at_construction"
            attempts.append(chart_attempt)
            continue
        if chart_kind == "multi_line" and len(series) < 2:
            chart_kind = "single_line"
            chart_attempt["chart_kind"] = chart_kind
        elif chart_kind == "single_line" and len(series) >= 2:
            chart_kind = "multi_line"
            chart_attempt["chart_kind"] = chart_kind
        legend = detect_legend_entries(image, tokens, axis)
        color_match = match_series_by_color(legend.get("entries", []), series)
        series_to_legend: dict[int, dict[str, Any]] = {}
        for entry_match in color_match["entry_matches"]:
            if entry_match.get("matched_series_id") is not None:
                series_to_legend[int(entry_match["matched_series_id"])] = entry_match
        if chart_kind == "multi_line" and not series_to_legend:
            chart_attempt["status"] = "legend_unrecoverable_at_construction"
            attempts.append(chart_attempt)
            continue

        chart_accepts = 0
        chart_attempt["status"] = "no_label_accepted"
        labels = sorted(
            scan.get("data_label_candidates", []),
            key=lambda item: (-float(item["confidence"]), float(item["bbox"][1]), float(item["bbox"][0])),
        )[: MAX_SAMPLES_PER_CHART]
        for label_index, label_item in enumerate(labels):
            chart_attempt["labels_seen"] += 1
            attempt: dict[str, Any] = {
                "chart_id": chart_id,
                "chart_kind": chart_kind,
                "candidate_label_index": label_index,
                "label_text": label_item["text"],
                "status": "rejected",
            }
            label = token_from_dict(label_item)
            confirmation = crop_confirm_label(reader, image, label)
            if confirmation is None:
                attempt["reason"] = "crop_ocr_did_not_confirm_value"
                attempts.append(attempt)
                continue
            pseudo_gold = float(label.numeric_value)
            if chart_kind == "single_line":
                association = {"status": "ok", "series_id": int(series[0]["series_id"]), "crossing_at_label": False}
            else:
                association = associate_label_to_series(label, pseudo_gold, series, axis, width, height)
            if association.get("status") != "ok":
                attempt["reason"] = association.get("reason")
                attempts.append(attempt)
                continue
            target_component = next(item for item in series if int(item["series_id"]) == int(association["series_id"]))
            legend_entry = series_to_legend.get(int(association["series_id"]))
            if chart_kind == "multi_line" and legend_entry is None:
                attempt["reason"] = "target_series_not_in_legend"
                attempts.append(attempt)
                continue
            series_name = None if legend_entry is None else str(legend_entry["legend_name"])
            construction_x = construction_line_x(target_component["mask"], label, pseudo_gold, axis)
            semantic = infer_construction_target(construction_x, construction_anchors, width)
            if semantic is None:
                attempt["reason"] = "no_reliable_semantic_x_target"
                attempts.append(attempt)
                continue
            if chart_kind == "multi_line":
                question = f"What is the value of {series_name} in {semantic['target_semantic_label']}?"
            else:
                question = f"What is the value for {semantic['target_semantic_label']}?"
            masked, pixel_mask, protected_pixels = mask_value_text_protect_all(
                image, label, [component["mask"] for component in series]
            )
            sample_id = f"phase4_line_{len(samples) + 1:04d}"
            masked_path = masked_dir / f"{sample_id}.png"
            cv2.imwrite(str(masked_path), masked)

            # Auto-Series run: the pipeline receives only the masked chart and question.
            auto = auto_multiline_pipeline(reader, masked, question)
            oracle = None
            if chart_kind == "multi_line" and series_name is not None:
                # Oracle-Series diagnostic: correct series identity is supplied;
                # the value and its location are never supplied.
                oracle = auto_multiline_pipeline(reader, masked, question, oracle_series_name=series_name)

            sample = {
                "sample_id": sample_id,
                "chart_id": chart_id,
                "chart_kind": chart_kind,
                "representative_sample_id": scan.get("representative_sample_id"),
                "source_image_path": scan["image_path"],
                "masked_image_path": relative_project_path(args.project, masked_path),
                "question": question,
                "target_series_name": series_name,
                "semantic_target": semantic,
                "parser_runtime_contract": {
                    "inputs": ["masked_chart", "question"],
                    "forbidden_inputs": ["pseudo_gold", "masked_value_label_bbox", "masked_value_label_center_x"],
                    "oracle_series_diagnostic": "oracle runs additionally receive the correct series name only",
                },
                "auto_status": auto.get("status"),
                "auto_status_reason": auto.get("status_reason"),
                "auto_result": auto,
                "oracle_status": None if oracle is None else oracle.get("status"),
                "oracle_status_reason": None if oracle is None else oracle.get("status_reason"),
                "oracle_result": oracle,
                "pseudo_gold": pseudo_gold,
                "mask_pixel_count": int(np.count_nonzero(pixel_mask)),
                "protected_series_pixel_count": protected_pixels,
                "mask_overlaps_protected_series_pixels": 0,
                "construction_only": {
                    "masked_value_label": label.as_dict(),
                    "crop_confirmation": confirmation,
                    "original_x_axis_anchors": anchors,
                    "coherent_construction_anchors": construction_anchors,
                    "original_x_axis_anchor_audit": anchor_audit,
                    "construction_series_id": int(association["series_id"]),
                    "construction_series_color_bgr": target_component["median_bgr"],
                    "series_association": association,
                    "construction_line_x": construction_x,
                    "legend_entries_original": legend.get("entries", []),
                    "legend_color_match_original": color_match,
                    "detected_series_original": [
                        {key: value for key, value in component.items() if key != "mask"} for component in series
                    ],
                },
            }
            samples.append(sample)
            chart_accepts += 1
            attempt["status"] = "accepted"
            attempt["sample_id"] = sample_id
            attempt["semantic_mode"] = semantic["construction_mode"]
            attempt["auto_status"] = auto.get("status")
            attempts.append(attempt)

            variants = [("auto_column", auto, auto.get("auto_column_geometry"), auto.get("auto_column_value")),
                        ("auto_local", auto, auto.get("auto_local_geometry"), auto.get("auto_local_value"))]
            if oracle is not None:
                variants.extend(
                    [("oracle_column", oracle, oracle.get("auto_column_geometry"), oracle.get("auto_column_value")),
                     ("oracle_local", oracle, oracle.get("auto_local_geometry"), oracle.get("auto_local_value"))]
                )
            for method_name, pipeline_result, geometry, value in variants:
                overlay = draw_phase4_overlay(image, sample_id, method_name, pipeline_result, geometry, value, pseudo_gold)
                cv2.imwrite(str(overlay_dir / f"{sample_id}_{method_name}.png"), overlay)
        if chart_accepts:
            chart_attempt["status"] = "accepted"
            chart_attempt["accepted"] = chart_accepts
        attempts.append(chart_attempt)
        print(
            json.dumps(
                {
                    "event": "phase4_construct_chart",
                    "chart_index": chart_index,
                    "charts_total": len(chart_ids),
                    "chart_id": chart_id[:12],
                    "chart_kind": chart_kind,
                    "accepted_on_chart": chart_accepts,
                    "samples_total": len(samples),
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    write_jsonl(args.project / "phase4_audit" / "line_samples.jsonl", samples)
    write_jsonl(args.project / "phase4_audit" / "construction_attempts.jsonl", attempts)
    print(
        json.dumps(
            {
                "event": "phase4_construct_complete",
                "samples": len(samples),
                "charts": len({row["chart_id"] for row in samples}),
                "multi_line_samples": sum(row["chart_kind"] == "multi_line" for row in samples),
                "single_line_samples": sum(row["chart_kind"] == "single_line" for row in samples),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def stable_hash(text: str) -> str:
    return hashlib.sha256(f"{SEED}:{text}".encode("utf-8")).hexdigest()


def chart_splits(rows: list[dict[str, Any]]) -> dict[str, str]:
    chart_ids = sorted({str(row["chart_id"]) for row in rows}, key=stable_hash)
    if len(chart_ids) <= 1:
        return {chart_id: "test" for chart_id in chart_ids}
    test_count = max(1, min(len(chart_ids) - 1, int(round(0.30 * len(chart_ids)))))
    return {chart_id: ("test" if index < test_count else "train") for index, chart_id in enumerate(chart_ids)}


def method_metrics(rows: list[dict[str, Any]], method: str, value_getter: Any) -> dict[str, Any]:
    eligible: list[tuple[dict[str, Any], float]] = []
    for row in rows:
        value = value_getter(row)
        if value is not None and math.isfinite(float(value)):
            eligible.append((row, float(value)))
    errors = np.asarray([abs(value - float(row["pseudo_gold"])) for row, value in eligible], dtype=float)
    axis_errors = np.asarray(
        [
            error / max(float((row["auto_axis"] or {}).get("axis_span", math.nan)), 1e-12)
            for (row, _), error in zip(eligible, errors)
        ],
        dtype=float,
    )
    relative = np.asarray(
        [error / max(abs(float(row["pseudo_gold"])), 1e-12) for (row, _), error in zip(eligible, errors)],
        dtype=float,
    )
    total = len(rows)
    return {
        "method": method,
        "total_samples": total,
        "eligible_samples": len(eligible),
        "coverage": len(eligible) / max(total, 1),
        "mae": None if len(errors) == 0 else float(np.mean(errors)),
        "median_absolute_error": None if len(errors) == 0 else float(np.median(errors)),
        "mean_relative_error": None if len(relative) == 0 else float(np.mean(relative)),
        "mean_axis_normalized_error": None if len(axis_errors) == 0 else float(np.mean(axis_errors)),
        "median_axis_normalized_error": None if len(axis_errors) == 0 else float(np.median(axis_errors)),
        "within_1pct_axis_span": None if len(axis_errors) == 0 else float(np.mean(axis_errors <= 0.01)),
        "within_2pct_axis_span": None if len(axis_errors) == 0 else float(np.mean(axis_errors <= 0.02)),
        "within_5pct_axis_span": None if len(axis_errors) == 0 else float(np.mean(axis_errors <= 0.05)),
    }


def paired_cluster_bootstrap(
    rows: list[dict[str, Any]],
    left_getter: Any,
    right_getter: Any,
    iterations: int = 3000,
) -> dict[str, Any]:
    paired: list[tuple[dict[str, Any], float, float]] = []
    for row in rows:
        left = left_getter(row)
        right = right_getter(row)
        if left is not None and right is not None and math.isfinite(float(left)) and math.isfinite(float(right)):
            paired.append((row, float(left), float(right)))
    by_chart: dict[str, list[tuple[dict[str, Any], float, float]]] = defaultdict(list)
    for item in paired:
        by_chart[str(item[0]["chart_id"])].append(item)
    charts = sorted(by_chart)
    if not charts:
        return {"paired_samples": 0, "paired_charts": 0}
    observed = float(
        np.mean([abs(left - float(row["pseudo_gold"])) - abs(right - float(row["pseudo_gold"])) for row, left, right in paired])
    )
    rng = np.random.default_rng(SEED)
    draws: list[float] = []
    for _ in range(iterations):
        sampled = rng.choice(charts, size=len(charts), replace=True)
        items = [item for chart_id in sampled for item in by_chart[str(chart_id)]]
        draws.append(
            float(
                np.mean([abs(left - float(row["pseudo_gold"])) - abs(right - float(row["pseudo_gold"])) for row, left, right in items])
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
    }


def learned_calibrated_value(project: Path, axis: dict[str, Any], geometry: dict[str, Any] | None, raw_value: float | None) -> float | None:
    if raw_value is None or geometry is None or axis is None:
        return None
    model = json.loads((project / "config" / "calibrator_model.json").read_text(encoding="utf-8"))
    vector = feature_vector({"axis": axis, "geometry": geometry, "raw_geometry_value": raw_value})
    mean = np.asarray(model["feature_mean"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    normalized = (vector - mean) / np.where(scale < 1e-12, 1.0, scale)
    predicted_normalized_error = float(coefficients[0] + normalized @ coefficients[1:])
    return float(raw_value) + predicted_normalized_error * float(axis["axis_span"])


def _get(result: dict[str, Any] | None, key: str) -> Any:
    return None if result is None else result.get(key)


def analyze_samples(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.project / "phase4_audit" / "line_samples.jsonl")
    splits = chart_splits(rows)
    write_csv(
        args.project / "results" / "phase4_chart_split.csv",
        [
            {
                "chart_id": chart_id,
                "split": split,
                "chart_kind": next(row["chart_kind"] for row in rows if str(row["chart_id"]) == chart_id),
                "sample_count": sum(str(row["chart_id"]) == chart_id for row in rows),
                "leakage_check": "chart_disjoint",
            }
            for chart_id, split in sorted(splits.items())
        ],
    )
    phase1_split = {row["chart_id"]: row["split"] for row in load_csv(args.project / "results" / "chart_split.csv")}
    biases = phase1_biases(args.project)

    flat: list[dict[str, Any]] = []
    for row in rows:
        auto = row.get("auto_result") or {}
        oracle = row.get("oracle_result")
        axis = auto.get("axis") or (_get(oracle, "axis")) or {}
        row["auto_axis"] = axis
        axis_span = float(axis.get("axis_span", math.nan)) if axis else math.nan
        gold = float(row["pseudo_gold"])
        grounding = auto.get("grounding") or {}
        semantic = row.get("semantic_target") or {}
        resolution = auto.get("series_resolution") or {}
        local_value = auto.get("auto_local_value")
        calibration_eligible = phase1_split.get(str(row["chart_id"])) != "train"
        output = {
            "sample_id": row["sample_id"],
            "chart_id": row["chart_id"],
            "chart_kind": row["chart_kind"],
            "phase4_split": splits.get(str(row["chart_id"])),
            "question": row["question"],
            "target_series_name": row.get("target_series_name"),
            "target_semantic_label": semantic.get("target_semantic_label"),
            "construction_semantic_mode": semantic.get("construction_mode"),
            "x_grounding_mode": grounding.get("mode"),
            "predicted_target_x": grounding.get("predicted_target_x"),
            "construction_target_x": semantic.get("construction_target_x"),
            "x_grounding_error_pixels": None
            if grounding.get("predicted_target_x") is None or semantic.get("construction_target_x") is None
            else abs(float(grounding["predicted_target_x"]) - float(semantic["construction_target_x"])),
            "detected_series_count": len(auto.get("detected_series") or []),
            "legend_entry_count": len((auto.get("legend") or {}).get("entries", [])),
            "series_resolution_mode": resolution.get("series_mode"),
            "series_color_distance": resolution.get("color_distance_to_matched_line"),
            "series_color_margin": resolution.get("color_margin"),
            "line_crossing_near_target": (auto.get("series_local_overlap") or {}).get("line_crossing_near_target"),
            "local_overlap_fraction": (auto.get("series_local_overlap") or {}).get("local_overlap_fraction"),
            "series_column_coverage": (auto.get("series_continuity") or {}).get("column_coverage"),
            "axis_span": axis_span,
            "auto_status": row.get("auto_status"),
            "auto_status_reason": row.get("auto_status_reason"),
            "oracle_status": row.get("oracle_status"),
            "oracle_status_reason": row.get("oracle_status_reason"),
            "auto_column_value": auto.get("auto_column_value"),
            "auto_local_value": local_value,
            "oracle_column_value": _get(oracle, "auto_column_value"),
            "oracle_local_value": _get(oracle, "auto_local_value"),
            "pseudo_gold": gold,
            "calibration_eval_eligible_unseen_to_phase1_train": int(calibration_eligible),
            "parser_used_hidden_bbox_or_gold": 0,
            "masked_image_path": row["masked_image_path"],
            "source_image_path": row["source_image_path"],
        }
        for prefix, value in [
            ("auto_column", auto.get("auto_column_value")),
            ("auto_local", local_value),
            ("oracle_column", _get(oracle, "auto_column_value")),
            ("oracle_local", _get(oracle, "auto_local_value")),
        ]:
            if value is None or not math.isfinite(float(value)):
                output[f"{prefix}_absolute_error"] = None
                output[f"{prefix}_axis_normalized_error"] = None
            else:
                output[f"{prefix}_absolute_error"] = abs(float(value) - gold)
                output[f"{prefix}_axis_normalized_error"] = abs(float(value) - gold) / max(axis_span, 1e-12)
        flat.append(output)
    write_csv(args.project / "results" / "phase4_line_samples.csv", flat)

    single = [row for row in rows if row["chart_kind"] == "single_line"]
    multi = [row for row in rows if row["chart_kind"] == "multi_line"]
    getters = {
        "auto_column": lambda row: (row.get("auto_result") or {}).get("auto_column_value"),
        "auto_local": lambda row: (row.get("auto_result") or {}).get("auto_local_value"),
        "oracle_column": lambda row: _get(row.get("oracle_result"), "auto_column_value"),
        "oracle_local": lambda row: _get(row.get("oracle_result"), "auto_local_value"),
    }
    metric_rows: list[dict[str, Any]] = []
    for subset_name, subset in [("single_line", single), ("multi_line", multi), ("all", rows)]:
        for method, getter in getters.items():
            if subset_name == "single_line" and method.startswith("oracle"):
                continue
            metric_rows.append({"subset": subset_name, **method_metrics(subset, method, getter)})
    write_csv(args.project / "results" / "phase4_method_metrics.csv", metric_rows)

    # Stratified by x-grounding mode.
    stratified: list[dict[str, Any]] = []
    for subset_name, subset in [("single_line", single), ("multi_line", multi)]:
        for mode in ["direct_anchor", "piecewise_linear_interpolation"]:
            mode_rows = [
                row
                for row in subset
                if ((row.get("auto_result") or {}).get("grounding") or {}).get("mode") == mode
            ]
            if not mode_rows:
                continue
            for method in ["auto_column", "auto_local"]:
                stratified.append(
                    {"subset": subset_name, "x_grounding_mode": mode, **method_metrics(mode_rows, method, getters[method])}
                )
    write_csv(args.project / "results" / "phase4_xmode_metrics.csv", stratified)

    pairwise = [
        {
            "comparison": "single_auto_column_minus_auto_local",
            **paired_cluster_bootstrap(single, getters["auto_column"], getters["auto_local"]),
        },
        {
            "comparison": "multi_auto_series_local_minus_oracle_series_local",
            **paired_cluster_bootstrap(multi, getters["auto_local"], getters["oracle_local"]),
        },
        {
            "comparison": "multi_auto_column_minus_auto_local",
            **paired_cluster_bootstrap(multi, getters["auto_column"], getters["auto_local"]),
        },
    ]
    write_csv(args.project / "results" / "phase4_pairwise_metrics.csv", pairwise)

    # Calibration controls on charts unseen to the Phase 1 calibrator train split.
    calibration_rows = [row for row in rows if phase1_split.get(str(row["chart_id"])) != "train"]
    calibration_metrics: list[dict[str, Any]] = []
    corrections: dict[str, Any] = {
        "raw_auto_local": lambda row: (row.get("auto_result") or {}).get("auto_local_value"),
        "global_mean_bias": lambda row: None
        if (row.get("auto_result") or {}).get("auto_local_value") is None
        else float((row.get("auto_result") or {})["auto_local_value"]) + biases["global_mean_bias"],
        "global_median_bias": lambda row: None
        if (row.get("auto_result") or {}).get("auto_local_value") is None
        else float((row.get("auto_result") or {})["auto_local_value"]) + biases["global_median_bias"],
        "line_type_mean_bias": lambda row: None
        if (row.get("auto_result") or {}).get("auto_local_value") is None
        else float((row.get("auto_result") or {})["auto_local_value"]) + biases["line_type_mean_bias"],
        "line_type_median_bias": lambda row: None
        if (row.get("auto_result") or {}).get("auto_local_value") is None
        else float((row.get("auto_result") or {})["auto_local_value"]) + biases["line_type_median_bias"],
        "learned_phase1_calibrator": lambda row: learned_calibrated_value(
            args.project,
            row.get("auto_axis") or (row.get("auto_result") or {}).get("axis") or {},
            (row.get("auto_result") or {}).get("auto_local_geometry"),
            (row.get("auto_result") or {}).get("auto_local_value"),
        ),
    }
    for method, getter in corrections.items():
        calibration_metrics.append({"subset": "calibration_eligible", **method_metrics(calibration_rows, method, getter)})
    write_csv(args.project / "results" / "phase4_calibration_bias_metrics.csv", calibration_metrics)
    write_csv(
        args.project / "results" / "phase4_bias_constants.csv",
        [{"constant": key, "value": value, "training_source": "frozen Phase1 train charts"} for key, value in biases.items()],
    )

    # Rejection audit: runtime statuses + construction-stage rejections.
    runtime_rejections = [
        {
            "stage": "runtime",
            "chart_id": row["chart_id"],
            "chart_kind": row["chart_kind"],
            "sample_id": row["sample_id"],
            "status": row.get("auto_status"),
            "reason": row.get("auto_status_reason"),
        }
        for row in rows
        if row.get("auto_status") != "success"
    ]
    construction_rejections = [
        {
            "stage": "construction",
            "chart_id": row.get("chart_id"),
            "chart_kind": row.get("chart_kind"),
            "sample_id": row.get("sample_id"),
            "status": row.get("status"),
            "reason": row.get("reason"),
        }
        for row in read_jsonl(args.project / "phase4_audit" / "construction_attempts.jsonl")
        if row.get("status") not in {"accepted", None} or row.get("labels_seen")
    ]
    write_csv(args.project / "results" / "phase4_rejection_audit.csv", runtime_rejections + construction_rejections)

    rejection_summary = Counter(str(row.get("auto_status")) for row in rows if row.get("auto_status") != "success")
    rejection_reasons = Counter(
        f"{row.get('auto_status')}:{row.get('auto_status_reason')}" for row in rows if row.get("auto_status") != "success"
    )

    # Error slices for multi-line (only when bucket sizes allow).
    slices: list[dict[str, Any]] = []
    multi_flat = [row for row in flat if row["chart_kind"] == "multi_line"]
    slice_specs = [
        ("line_crossing_near_target", lambda row: str(row.get("line_crossing_near_target"))),
        ("series_count_bucket", lambda row: "2" if row.get("detected_series_count") == 2 else "3+"),
        (
            "color_margin_bucket",
            lambda row: "missing"
            if row.get("series_color_margin") is None
            else ("lt_14" if float(row["series_color_margin"]) < 14 else "ge_14"),
        ),
    ]
    if len(multi_flat) >= 10:
        for slice_name, key_fn in slice_specs:
            buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in multi_flat:
                buckets[key_fn(row)].append(row)
            for bucket, bucket_rows in sorted(buckets.items()):
                if len(bucket_rows) < 5:
                    continue
                slices.append(
                    {
                        "slice": slice_name,
                        "bucket": bucket,
                        **method_metrics(
                            [row for row in rows if row["sample_id"] in {item["sample_id"] for item in bucket_rows}],
                            "auto_local",
                            getters["auto_local"],
                        ),
                    }
                )
    write_csv(args.project / "results" / "phase4_error_slices.csv", slices)

    # Phase 3 overlap diagnostic (frozen Phase 3 file is read, never written).
    phase3_rows = load_csv(args.project / "results" / "phase3_line_samples.csv")
    phase3_by_label = {
        (str(row["chart_id"]), str(row["target_semantic_label"])): row for row in phase3_rows
    }
    overlap_rows: list[dict[str, Any]] = []
    for row in flat:
        key = (str(row["chart_id"]), str(row["target_semantic_label"]))
        frozen = phase3_by_label.get(key)
        if frozen is None:
            continue
        phase4_value = row.get("auto_local_value")
        phase3_value = None if frozen.get("auto_x_local_value") in {None, ""} else float(frozen["auto_x_local_value"])
        overlap_rows.append(
            {
                "chart_id": row["chart_id"],
                "target_semantic_label": row["target_semantic_label"],
                "phase3_auto_local_value": phase3_value,
                "phase4_auto_local_value": phase4_value,
                "pseudo_gold": row["pseudo_gold"],
                "phase3_abs_error": None if phase3_value is None else abs(phase3_value - float(row["pseudo_gold"])),
                "phase4_abs_error": None if phase4_value is None else abs(float(phase4_value) - float(row["pseudo_gold"])),
            }
        )
    write_csv(args.project / "results" / "phase4_phase3_overlap.csv", overlap_rows)

    summary = {
        "sample_count": len(rows),
        "chart_count": len({row["chart_id"] for row in rows}),
        "single_line_samples": len(single),
        "multi_line_samples": len(multi),
        "single_line_charts": len({row["chart_id"] for row in single}),
        "multi_line_charts": len({row["chart_id"] for row in multi}),
        "direct_grounding_samples": sum(((row.get("auto_result") or {}).get("grounding") or {}).get("mode") == "direct_anchor" for row in rows),
        "interpolated_grounding_samples": sum(
            ((row.get("auto_result") or {}).get("grounding") or {}).get("mode") == "piecewise_linear_interpolation" for row in rows
        ),
        "runtime_rejection_status_counts": dict(rejection_summary),
        "runtime_rejection_reason_counts": dict(rejection_reasons),
        "train_test_chart_overlap": len(
            {chart for chart, split in splits.items() if split == "train"}
            & {chart for chart, split in splits.items() if split == "test"}
        ),
        "method_metrics": metric_rows,
        "pairwise": pairwise,
        "phase3_overlap_samples": len(overlap_rows),
    }
    (args.project / "results" / "phase4_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "phase4_analysis_complete", **summary}, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["construct", "analyze", "all"], default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cv2.setNumThreads(6)
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)
    if args.stage in {"construct", "all"}:
        reader = make_reader(args.model_dir)
        construct_samples(args, reader)
    if args.stage in {"analyze", "all"}:
        analyze_samples(args)


if __name__ == "__main__":
    main()
