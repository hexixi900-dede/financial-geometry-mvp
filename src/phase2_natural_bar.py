from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import easyocr
import numpy as np
import torch

from calibrate_geometry import feature_vector
from geometry_core import (
    OCRToken,
    _quantized_components,
    axis_localizer,
    interpolate_pixel_to_value,
    ocr_tokens,
    parse_numeric_label,
)


SUPPORTED_QUESTION_TYPES = {"numerical", "single_choice"}
DISALLOWED_QUESTION_TERMS = re.compile(
    r"\b(average|mean|total|sum|all companies|all years|all periods|cumulative|"
    r"stacked|combined|both axes|right axis|area|line|trend|moving average)\b",
    re.IGNORECASE,
)
GROWTH_TERMS = re.compile(
    r"percentage (?:increase|decrease|change)|percent (?:increase|decrease|change)|growth rate",
    re.IGNORECASE,
)
DIFFERENCE_TERMS = re.compile(
    r"\b(?:difference between|difference in|difference of|how much (?:more|less|higher|lower)|"
    r"increase in|decrease in|increase from|decrease from|exceed(?:ed|s|ing)?)\b",
    re.IGNORECASE,
)
DIRECT_TERMS = re.compile(
    r"\b(?:what (?:is|was|were)|what value|how much|value of|amount of|percentage of|rate in|margin in)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = str(text).lower().replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"(?<=\d)\s+(?=fy\d)", "", text)
    return " ".join(re.findall(r"[a-z0-9]+", text))


def compact_text(text: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def sequence_windows(question: str, length: int) -> Iterable[str]:
    words = normalize_text(question).split()
    if not words:
        return []
    span = max(1, length)
    return (
        " ".join(words[start : start + size])
        for size in range(max(1, span - 1), min(len(words), span + 2) + 1)
        for start in range(0, len(words) - size + 1)
    )


def label_question_score(label: str, question: str) -> float:
    label_norm = normalize_text(label)
    question_norm = normalize_text(question)
    label_compact = compact_text(label)
    question_compact = compact_text(question)
    if len(label_compact) < 2:
        return 0.0
    if label_norm and re.search(rf"(?<![a-z0-9]){re.escape(label_norm)}(?![a-z0-9])", question_norm):
        return 1.0
    if label_compact in question_compact:
        return 0.96
    year_suffix = re.fullmatch(r"((?:19|20)\d{2})[ef]", label_compact)
    if year_suffix and year_suffix.group(1) in question_compact:
        return 0.94
    window_score = max(
        (SequenceMatcher(None, label_norm, window).ratio() for window in sequence_windows(question, len(label_norm.split()))),
        default=0.0,
    )
    return float(window_score)


def parse_operation(question: str) -> str | None:
    if DISALLOWED_QUESTION_TERMS.search(question):
        return None
    if GROWTH_TERMS.search(question):
        return "growth_rate"
    if DIFFERENCE_TERMS.search(question):
        return "difference"
    if DIRECT_TERMS.search(question):
        return "direct"
    return None


def component_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    ax1, ay1, aw, ah = [float(first[key]) for key in ("x", "y", "w", "h")]
    bx1, by1, bw, bh = [float(second[key]) for key in ("x", "y", "w", "h")]
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = aw * ah + bw * bh - overlap
    return overlap / max(union, 1.0)


def detect_all_bars(image: np.ndarray, axis: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    height, width = image.shape[:2]
    plot_left, plot_top, plot_right, plot_bottom = map(float, axis["plot_bbox"])
    plot_height = max(1.0, plot_bottom - plot_top)
    zero_pixel = -float(axis["intercept"]) / float(axis["slope"])
    expected_base = zero_pixel if plot_top <= zero_pixel <= plot_bottom + 0.06 * height else plot_bottom
    raw_components = _quantized_components(image, axis)
    candidates: list[dict[str, Any]] = []
    for component in raw_components:
        x, y, box_w, box_h = [int(component[key]) for key in ("x", "y", "w", "h")]
        bottom = y + box_h
        if x < plot_left - 0.01 * width or x + box_w > plot_right + 0.01 * width:
            continue
        if y < plot_top - 0.02 * height or bottom > plot_bottom + 0.08 * height:
            continue
        if box_w < max(5, 0.007 * width) or box_w > 0.16 * width:
            continue
        if box_h < max(10, 0.030 * height) or box_h > 0.94 * plot_height:
            continue
        if box_h < 0.58 * box_w or float(component["fill"]) < 0.50:
            continue
        # A truncated axis may place the true bar baseline at the lowest tick rather
        # than at the extrapolated plot bottom. Keep a wider, still local band here;
        # the shared baseline cohort is estimated below without using any values.
        base_distance = abs(bottom - expected_base)
        if base_distance > max(12.0, 0.13 * height):
            continue
        candidates.append({**component, "base_distance_pixels": float(base_distance)})

    candidates.sort(key=lambda item: (-int(item["area"]), -float(item["fill"])))
    deduplicated: list[dict[str, Any]] = []
    for candidate in candidates:
        center = float(candidate["x"]) + float(candidate["w"]) / 2
        duplicate = False
        for kept in deduplicated:
            kept_center = float(kept["x"]) + float(kept["w"]) / 2
            if component_iou(candidate, kept) >= 0.58 or (
                abs(center - kept_center) <= 0.22 * min(float(candidate["w"]), float(kept["w"]))
                and abs((candidate["y"] + candidate["h"]) - (kept["y"] + kept["h"])) <= 4
                and abs(candidate["y"] - kept["y"]) <= 5
            ):
                duplicate = True
                break
        if not duplicate:
            deduplicated.append(candidate)

    baseline_tolerance = max(7.0, 0.026 * height)
    baseline_cohorts: list[tuple[tuple[float, float], list[dict[str, Any]], float]] = []
    for anchor in deduplicated:
        anchor_bottom = float(anchor["y"] + anchor["h"])
        cohort = [
            item
            for item in deduplicated
            if abs(float(item["y"] + item["h"]) - anchor_bottom) <= baseline_tolerance
        ]
        median_base = float(np.median([item["y"] + item["h"] for item in cohort]))
        baseline_cohorts.append(
            (
                (float(len(cohort)), -abs(median_base - expected_base) / max(height, 1)),
                cohort,
                median_base,
            )
        )
    if baseline_cohorts:
        _, deduplicated, detected_base = max(baseline_cohorts, key=lambda item: item[0])
    else:
        detected_base = float(expected_base)

    if len(deduplicated) >= 2:
        median_width = float(np.median([item["w"] for item in deduplicated]))
        deduplicated = [
            item for item in deduplicated if 0.48 * median_width <= float(item["w"]) <= 1.90 * median_width
        ]
    deduplicated.sort(key=lambda item: float(item["x"]) + float(item["w"]) / 2)
    bars: list[dict[str, Any]] = []
    for index, component in enumerate(deduplicated):
        x, y, box_w, box_h = [int(component[key]) for key in ("x", "y", "w", "h")]
        bars.append(
            {
                "bar_id": f"bar_{index:02d}",
                "bbox": [x, y, box_w, box_h],
                "target_x": float(x + box_w / 2),
                "target_y": float(y),
                "score": float(2.0 * component["fill"] + min(box_h / plot_height, 1.0) - 3.0 * component["base_distance_pixels"] / max(height, 1)),
                "fill_ratio": float(component["fill"]),
                "base_distance_pixels": float(component["base_distance_pixels"]),
                "median_bgr": component["median_bgr"],
                "chart_type": "bar",
            }
        )
    diagnostics = {
        "raw_component_count": len(raw_components),
        "bar_candidate_count": len(candidates),
        "deduplicated_bar_count": len(bars),
        "expected_base_y": float(expected_base),
        "detected_shared_base_y": float(detected_base),
        "baseline_tolerance_pixels": float(baseline_tolerance),
    }
    return bars, diagnostics


def label_bars(
    tokens: list[OCRToken], bars: list[dict[str, Any]], axis: dict[str, Any], width: int, height: int
) -> list[dict[str, Any]]:
    if not bars:
        return bars
    axis_indices = {int(index) for index in axis["token_indices"]}
    base_y = float(np.median([bar["bbox"][1] + bar["bbox"][3] for bar in bars]))
    centers = [float(bar["target_x"]) for bar in bars]
    for index, bar in enumerate(bars):
        left_bound = float(axis["plot_bbox"][0]) if index == 0 else (centers[index - 1] + centers[index]) / 2
        right_bound = float(axis["plot_bbox"][2]) if index == len(bars) - 1 else (centers[index] + centers[index + 1]) / 2
        label_tokens = [
            token
            for token in tokens
            if token.index not in axis_indices
            and token.confidence >= 0.30
            and left_bound <= token.cx <= right_bound
            and base_y - 0.018 * height <= token.y1 <= base_y + 0.22 * height
            and token.y2 - token.y1 <= 0.10 * height
        ]
        label_tokens.sort(key=lambda token: (round(token.cy / max(7.0, 0.018 * height)), token.x1))
        text_parts: list[str] = []
        seen: set[str] = set()
        for token in label_tokens:
            normalized = normalize_text(token.text)
            if not normalized or normalized in seen:
                continue
            text_parts.append(token.text.strip())
            seen.add(normalized)
        bar["detected_x_label"] = " ".join(text_parts).strip()
        bar["x_label_tokens"] = [token.as_dict() for token in label_tokens]
        bar["x_label_bbox"] = (
            [
                float(min(token.x1 for token in label_tokens)),
                float(min(token.y1 for token in label_tokens)),
                float(max(token.x2 for token in label_tokens)),
                float(max(token.y2 for token in label_tokens)),
            ]
            if label_tokens
            else None
        )
    return bars


def numeric_interior_tokens(
    tokens: list[OCRToken], bars: list[dict[str, Any]], axis: dict[str, Any], width: int, height: int
) -> list[dict[str, Any]]:
    if not bars:
        return []
    axis_indices = {int(index) for index in axis["token_indices"]}
    base_y = float(np.median([bar["bbox"][1] + bar["bbox"][3] for bar in bars]))
    plot_left, plot_top, plot_right, _ = map(float, axis["plot_bbox"])
    return [
        token.as_dict()
        for token in tokens
        if token.index not in axis_indices
        and token.numeric_value is not None
        and token.confidence >= 0.50
        and plot_left + 0.005 * width <= token.cx <= plot_right - 0.005 * width
        and plot_top - 0.02 * height <= token.cy <= base_y - 0.012 * height
    ]


def analyze_chart(
    reader: easyocr.Reader, chart_id: str, image_path: Path, prefilter: dict[str, Any]
) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {
        "chart_id": chart_id,
        "image_path": str(image_path),
        "prefilter": prefilter,
        "gold_access": False,
    }
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {**result, "status": "image_read_failed"}
    height, width = image.shape[:2]
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
    axis = axis_localizer(tokens, width, height)
    result.update(
        {
            "image_size": [width, height],
            "ocr_tokens": [token.as_dict() for token in tokens],
            "ocr_seconds": time.monotonic() - started,
        }
    )
    if axis is None:
        return {**result, "status": "no_linear_left_y_axis"}
    result["axis"] = axis
    if axis.get("right_axis_detected") or int(axis.get("additional_y_axis_count", 0)) > 0:
        return {**result, "status": "multiple_y_axes_or_panels"}
    bars, diagnostics = detect_all_bars(image, axis)
    result["bar_detection"] = diagnostics
    bars = label_bars(tokens, bars, axis, width, height)
    result["bars"] = bars
    if not (2 <= len(bars) <= 20):
        return {**result, "status": "bar_count_out_of_scope"}
    if int(prefilter.get("sloped_segments", 0)) > 1:
        return {**result, "status": "possible_line_or_combination"}
    interior = numeric_interior_tokens(tokens, bars, axis, width, height)
    result["interior_numeric_tokens"] = interior
    if interior:
        return {**result, "status": "direct_data_value_or_numeric_annotation_present"}
    labelled = sum(bool(bar.get("detected_x_label")) for bar in bars)
    result["x_label_coverage"] = labelled / len(bars)
    if labelled / len(bars) < 0.70:
        return {**result, "status": "x_axis_labels_unreadable_or_grouped_bars"}
    widths = np.asarray([bar["bbox"][2] for bar in bars], dtype=float)
    if float(widths.max() / max(widths.min(), 1.0)) > 2.2:
        return {**result, "status": "inconsistent_bar_geometry"}
    return {**result, "status": "eligible_natural_no_label_bar"}


def semantic_plan(question: str, options: str, bars: list[dict[str, Any]]) -> dict[str, Any]:
    operation = parse_operation(question)
    result: dict[str, Any] = {
        "question": question,
        "operation": operation,
        "gold_access": False,
        "manual_target_coordinates_access": False,
        "hidden_value_label_bbox_access": False,
    }
    if operation is None:
        return {**result, "status": "unsupported_question_operation"}
    match_rows: list[dict[str, Any]] = []
    for bar in bars:
        full_label = bar.get("detected_x_label", "")
        phrases = [full_label] + [str(token.get("text", "")) for token in bar.get("x_label_tokens", [])]
        phrase_scores = [(phrase, label_question_score(phrase, question)) for phrase in phrases if normalize_text(phrase)]
        if not phrase_scores:
            continue
        best_phrase, best_score = max(phrase_scores, key=lambda item: (item[1], len(compact_text(item[0]))))
        match_rows.append(
            {
                "bar_id": bar["bar_id"],
                "label": best_phrase,
                "detected_full_label": full_label,
                "score": best_score,
                "question_position": compact_text(question).find(compact_text(best_phrase)),
            }
        )
    scored = sorted(
        match_rows,
        key=lambda item: (-float(item["score"]), int(item["question_position"]) if int(item["question_position"]) >= 0 else 10**9),
    )
    required = 2 if operation in {"difference", "growth_rate"} else 1
    strong = [item for item in scored if float(item["score"]) >= 0.78]
    if len(strong) < required:
        return {**result, "status": "semantic_target_not_found", "label_matches": scored[:5]}
    if required == 1 and len(strong) > 1 and float(strong[0]["score"]) - float(strong[1]["score"]) < 0.06:
        return {**result, "status": "semantic_target_ambiguous", "label_matches": strong[:5]}
    selected = strong[:required]
    if required == 2:
        selected.sort(key=lambda item: int(item["question_position"]) if int(item["question_position"]) >= 0 else 10**9)
        if selected[0]["bar_id"] == selected[1]["bar_id"]:
            return {**result, "status": "semantic_targets_not_distinct", "label_matches": selected}
    return {
        **result,
        "status": "success",
        "target_categories": [item["label"] for item in selected],
        "target_bar_ids": [item["bar_id"] for item in selected],
        "label_matches": selected,
        "semantic_target": " | ".join(item["label"] for item in selected),
    }


def load_calibrator(path: Path) -> dict[str, Any]:
    model = json.loads(path.read_text(encoding="utf-8"))
    if model.get("model") != "huber_ridge_error_calibrator":
        raise RuntimeError(f"Unsupported calibrator: {model.get('model')}")
    return model


def calibrate_value(raw_value: float, axis: dict[str, Any], bar: dict[str, Any], model: dict[str, Any]) -> tuple[float, float]:
    synthetic_row = {"axis": axis, "geometry": bar, "raw_geometry_value": raw_value}
    features = feature_vector(synthetic_row)
    mean = np.asarray(model["feature_mean"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    normalized = (features - mean) / np.where(scale < 1e-12, 1.0, scale)
    predicted_normalized_error = float(coefficients[0] + normalized @ coefficients[1:])
    predicted_error = predicted_normalized_error * float(axis["axis_span"])
    return float(raw_value + predicted_error), float(predicted_error)


def parse_option_rows(options: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in str(options).splitlines():
        match = re.match(r"\s*([A-Z])\s*[:.)-]\s*(.*?)\s*$", line)
        if match:
            rows.append((match.group(1), match.group(2)))
    return rows


def numeric_option_value(text: str) -> float | None:
    normalized = text.replace(",", "")
    matches = re.findall(r"(?<![A-Za-z0-9])[-+]?\d+(?:\.\d+)?\s*(?:%|[kKmMbBtT])?", normalized)
    values: list[float] = []
    for match in matches:
        value, _ = parse_numeric_label(match.strip())
        if value is not None:
            values.append(float(value))
    non_year = [value for value in values if not (1900 <= abs(value) <= 2100)]
    return non_year[0] if non_year else (values[0] if len(values) == 1 else None)


def execute_operation(operation: str, values: list[float], question: str) -> float:
    if operation == "direct" and len(values) == 1:
        return float(values[0])
    if operation == "difference" and len(values) == 2:
        return float(abs(values[0] - values[1]))
    if operation == "growth_rate" and len(values) == 2:
        if abs(values[0]) < 1e-12:
            raise ZeroDivisionError("Cannot compute a growth rate from a zero baseline")
        signed = 100.0 * (values[1] - values[0]) / abs(values[0])
        if re.search(r"percentage decrease", question, re.IGNORECASE):
            signed = -signed
        return float(signed)
    raise ValueError(f"Unsupported operation/value count: {operation}/{len(values)}")


def answer_from_geometry(
    question_type: str, options: str, operation: str, values: list[float], question: str
) -> tuple[str | float, float, dict[str, Any]]:
    numeric_answer = execute_operation(operation, values, question)
    reasoning = {"executor": "python", "formula": operation, "numeric_answer": numeric_answer}
    if question_type == "numerical":
        return numeric_answer, numeric_answer, reasoning
    option_values = [(letter, numeric_option_value(text), text) for letter, text in parse_option_rows(options)]
    usable = [(letter, value, text) for letter, value, text in option_values if value is not None]
    if len(usable) < 2:
        raise ValueError("Single-choice question does not provide at least two numeric options")
    selected = min(usable, key=lambda item: abs(float(item[1]) - numeric_answer))
    reasoning["option_values"] = option_values
    reasoning["selected_option"] = selected[0]
    return selected[0], numeric_answer, reasoning


def parse_reference_number(reference: Any) -> float | None:
    match = re.search(r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(reference))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def correctness(question_type: str, prediction: Any, reference: Any, tolerance: Any) -> bool:
    if question_type == "numerical":
        predicted = parse_reference_number(prediction)
        gold = parse_reference_number(reference)
        try:
            tol = float(tolerance)
        except (TypeError, ValueError):
            return False
        return predicted is not None and gold is not None and math.isfinite(tol) and abs(predicted - gold) <= tol
    return str(prediction).strip().upper() == str(reference).strip().upper()


def draw_overlay(
    image: np.ndarray,
    sample_id: str,
    question: str,
    analysis: dict[str, Any],
    plan: dict[str, Any],
    raw_values: list[float],
    calibrated_values: list[float],
) -> np.ndarray:
    canvas = image.copy()
    axis = analysis["axis"]
    selected_ids = set(plan["target_bar_ids"])
    left, top, right, bottom = [int(round(value)) for value in axis["plot_bbox"]]
    cv2.rectangle(canvas, (left, top), (right, bottom), (255, 160, 0), 2)
    for tick in axis["ticks"]:
        y = int(round(float(tick["pixel_y"])))
        cv2.circle(canvas, (left, y), 5, (255, 0, 255), -1)
        cv2.putText(canvas, f"{tick['value']:g}@{y}", (max(2, left + 7), max(14, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1, cv2.LINE_AA)
    selected_index = 0
    for bar in analysis["bars"]:
        x, y, width, height = map(int, bar["bbox"])
        chosen = bar["bar_id"] in selected_ids
        color = (0, 0, 255) if chosen else (140, 140, 140)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 2 if chosen else 1)
        cv2.putText(canvas, f"{bar['bar_id']}:{bar.get('detected_x_label','')}", (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        if chosen:
            cv2.circle(canvas, (int(round(bar["target_x"])), int(round(bar["target_y"]))), 6, (0, 0, 255), -1)
            cv2.putText(canvas, f"raw={raw_values[selected_index]:.3g} cal={calibrated_values[selected_index]:.3g} xy=({bar['target_x']:.0f},{bar['target_y']:.0f})", (x, min(canvas.shape[0] - 8, y + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1, cv2.LINE_AA)
            selected_index += 1
    panel_height = 72
    panel = np.full((panel_height, canvas.shape[1], 3), 255, dtype=np.uint8)
    lines = [
        f"{sample_id} op={plan['operation']} target={plan['semantic_target']}",
        question[:150],
        "RED=selected bar/top; GRAY=other detected bars; MAGENTA=y ticks; BLUE=plot bbox",
    ]
    for index, line in enumerate(lines):
        cv2.putText(panel, line, (8, 19 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([panel, canvas])


def chart_cache_key(chart_id: str, image_path: str) -> str:
    return hashlib.sha256(f"phase2_bar_detector_v2\0{chart_id}\0{image_path}".encode("utf-8")).hexdigest()


def repeated_time_categories(label: str) -> int:
    patterns = re.findall(
        r"(?:\b(?:19|20)\d{2}[ef]?\b|\bfy\s*\d{2,4}\b|\b\d{1,2}[-/]"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/]\d{2,4}\b)",
        str(label),
        flags=re.IGNORECASE,
    )
    return len(patterns)


def non_bar_marker_evidence(image: np.ndarray, analysis: dict[str, Any]) -> dict[str, int]:
    axis = analysis["axis"]
    bars = analysis["bars"]
    plot_left, plot_top, plot_right, _ = map(float, axis["plot_bbox"])
    base_y = float(np.median([bar["bbox"][1] + bar["bbox"][3] for bar in bars]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(18, int(0.035 * image.shape[1])),
        param1=90,
        param2=18,
        minRadius=max(3, int(0.004 * min(image.shape[:2]))),
        maxRadius=max(9, int(0.025 * min(image.shape[:2]))),
    )
    interior_circles = 0
    if circles is not None:
        for x, y, _ in np.asarray(circles[0]):
            if plot_left + 0.025 * image.shape[1] <= x <= plot_right and plot_top <= y <= base_y - 0.045 * image.shape[0]:
                interior_circles += 1

    dark = (gray < 75).astype(np.uint8)
    horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((1, max(9, int(0.014 * image.shape[1]))), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(horizontal, connectivity=8)
    bar_tops = [float(bar["target_y"]) for bar in bars]
    marker_segments = 0
    for x, y, width, height, area in stats[1:count]:
        cy = y + height / 2
        if not (plot_left <= x and x + width <= plot_right and plot_top <= cy <= base_y - 0.035 * image.shape[0]):
            continue
        if not (0.014 * image.shape[1] <= width <= 0.12 * image.shape[1]) or height > 0.025 * image.shape[0]:
            continue
        if any(abs(cy - top_y) <= max(5.0, 0.008 * image.shape[0]) for top_y in bar_tops):
            continue
        if area / max(width * height, 1) >= 0.45:
            marker_segments += 1
    return {"interior_circle_markers": interior_circles, "interior_horizontal_markers": marker_segments}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--candidate-scan", type=Path, required=True)
    parser.add_argument("--raw-vlm", type=Path, required=True)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-new-charts", type=int, default=180)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    cv2.setNumThreads(6)
    torch.set_num_threads(6)
    torch.set_num_interop_threads(2)

    records = read_jsonl(args.records)
    raw_rows = read_jsonl(args.raw_vlm)
    raw_by_sample = {str(row["sample_id"]): row for row in raw_rows}
    scan_rows = read_jsonl(args.candidate_scan)
    scan_by_chart = {str(row["chart_id"]): row for row in scan_rows}
    calibrator = load_calibrator(args.calibrator)

    # Targeting inputs deliberately exclude reference, tolerance, and every hidden/manual bbox.
    target_inputs: list[dict[str, Any]] = []
    evaluation_by_sample: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        chart_id = str(record["official_preprocessed_image_sha256"])
        scan = scan_by_chart.get(chart_id)
        raw = raw_by_sample.get(sample_id)
        if scan is None or raw is None:
            continue
        evaluation_by_sample[sample_id] = {
            "reference": record.get("reference"),
            "tolerance": record.get("tolerance"),
            "unit": record.get("unit"),
        }
        target_inputs.append(
            {
                "sample_id": sample_id,
                "chart_id": chart_id,
                "question": str(record.get("question", "")),
                "question_type": str(record.get("question_type", "")),
                "options": str(record.get("options", "")),
                "image_path": str(raw["image_path"]),
                "raw_vlm_prediction": raw.get("parsed_prediction"),
                "raw_vlm_raw_response": raw.get("raw_response"),
                "raw_vlm_parse_status": raw.get("parse_status"),
                "source_screen_status": scan.get("status"),
                "prefilter": scan.get("prefilter", {}),
            }
        )

    source_candidates = [
        row
        for row in target_inputs
        if row["source_screen_status"] == "no_geometry_linked_data_label"
        and row["question_type"] in SUPPORTED_QUESTION_TYPES
        and parse_operation(row["question"]) is not None
        and 2 <= int(row["prefilter"].get("bar_rectangles", 0)) <= 30
        and int(row["prefilter"].get("sloped_segments", 0)) <= 1
    ]
    operation_priority = {"direct": 0, "difference": 1, "growth_rate": 2}
    source_candidates.sort(
        key=lambda row: (
            operation_priority.get(parse_operation(row["question"]), 9),
            0 if row["question_type"] == "numerical" else 1,
            abs(int(row["prefilter"].get("bar_rectangles", 0)) - 7),
            int(row["sample_id"]),
        )
    )

    cache_path = args.project / "phase2_audit" / "chart_analysis_cache.jsonl"
    existing_cache = read_jsonl(cache_path) if cache_path.exists() else []
    cache = {str(row["cache_key"]): row for row in existing_cache}
    unique_chart_inputs: list[dict[str, Any]] = []
    seen_chart_keys: set[str] = set()
    for row in source_candidates:
        key = chart_cache_key(row["chart_id"], row["image_path"])
        if key not in seen_chart_keys:
            unique_chart_inputs.append(row)
            seen_chart_keys.add(key)
    reader = easyocr.Reader(
        ["en"], gpu=False, model_storage_directory=str(args.model_dir), download_enabled=False, verbose=False
    )
    print(json.dumps({"event": "reader_ready", "source_candidates": len(source_candidates), "unique_charts": len(unique_chart_inputs), "cached_charts": len(cache)}, sort_keys=True), flush=True)

    analyzed_new = 0
    for source in unique_chart_inputs:
        key = chart_cache_key(source["chart_id"], source["image_path"])
        if key in cache:
            continue
        if analyzed_new >= args.max_new_charts:
            break
        analysis = analyze_chart(reader, source["chart_id"], Path(source["image_path"]), source["prefilter"])
        analysis["cache_key"] = key
        append_jsonl(cache_path, analysis)
        cache[key] = analysis
        analyzed_new += 1
        if analyzed_new % args.progress_every == 0 or analysis["status"] == "eligible_natural_no_label_bar":
            print(json.dumps({"event": "chart_progress", "new": analyzed_new, "chart_id": source["chart_id"], "status": analysis["status"], "bars": len(analysis.get("bars", []))}, sort_keys=True), flush=True)

    screening_rows: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    accepted_charts: set[str] = set()
    phase1_rows = read_jsonl(args.project / "audit" / "geometry_samples.jsonl")
    phase1_charts = {str(row["chart_id"]) for row in phase1_rows}
    overlay_dir = args.project / "phase2_debug_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for source in source_candidates:
        key = chart_cache_key(source["chart_id"], source["image_path"])
        analysis = cache.get(key)
        if analysis is None:
            screening_rows.append({"sample_id": source["sample_id"], "chart_id": source["chart_id"], "question": source["question"], "decision": "reject", "reason": "chart_not_analyzed_within_budget"})
            continue
        if analysis["status"] != "eligible_natural_no_label_bar":
            screening_rows.append({"sample_id": source["sample_id"], "chart_id": source["chart_id"], "question": source["question"], "decision": "reject", "reason": analysis["status"]})
            continue
        image_for_structure = cv2.imread(source["image_path"], cv2.IMREAD_COLOR)
        marker_evidence = non_bar_marker_evidence(image_for_structure, analysis)
        if marker_evidence["interior_circle_markers"] >= 2 or marker_evidence["interior_horizontal_markers"] >= 2:
            screening_rows.append({"sample_id": source["sample_id"], "chart_id": source["chart_id"], "question": source["question"], "decision": "reject", "reason": "possible_marker_series_or_combination", "detail": json.dumps(marker_evidence, sort_keys=True)})
            continue
        if any(repeated_time_categories(bar.get("detected_x_label", "")) > 1 for bar in analysis["bars"]):
            screening_rows.append({"sample_id": source["sample_id"], "chart_id": source["chart_id"], "question": source["question"], "decision": "reject", "reason": "multiple_time_categories_assigned_to_one_bar"})
            continue
        if source["chart_id"] in accepted_charts:
            screening_rows.append({"sample_id": source["sample_id"], "chart_id": source["chart_id"], "question": source["question"], "decision": "reject", "reason": "one_question_per_chart_for_clean_subset"})
            continue
        plan = semantic_plan(source["question"], source["options"], analysis["bars"])
        if plan["status"] != "success":
            screening_rows.append({"sample_id": source["sample_id"], "chart_id": source["chart_id"], "question": source["question"], "decision": "reject", "reason": plan["status"], "semantic_plan": json.dumps(plan, ensure_ascii=False, sort_keys=True)})
            continue
        bar_by_id = {bar["bar_id"]: bar for bar in analysis["bars"]}
        selected_bars = [bar_by_id[bar_id] for bar_id in plan["target_bar_ids"]]
        raw_target_values = [interpolate_pixel_to_value(float(bar["target_y"]), analysis["axis"]) for bar in selected_bars]
        calibrated_pairs = [calibrate_value(value, analysis["axis"], bar, calibrator) for value, bar in zip(raw_target_values, selected_bars)]
        calibrated_target_values = [pair[0] for pair in calibrated_pairs]
        predicted_errors = [pair[1] for pair in calibrated_pairs]
        try:
            raw_prediction, raw_numeric_answer, raw_reasoning = answer_from_geometry(source["question_type"], source["options"], plan["operation"], raw_target_values, source["question"])
            calibrated_prediction, calibrated_numeric_answer, calibrated_reasoning = answer_from_geometry(source["question_type"], source["options"], plan["operation"], calibrated_target_values, source["question"])
        except (ValueError, ZeroDivisionError) as exc:
            screening_rows.append({"sample_id": source["sample_id"], "chart_id": source["chart_id"], "question": source["question"], "decision": "reject", "reason": "qa_executor_rejected", "detail": str(exc)})
            continue
        evaluation = evaluation_by_sample[source["sample_id"]]
        raw_geometry_correct = correctness(source["question_type"], raw_prediction, evaluation["reference"], evaluation["tolerance"])
        calibrated_correct = correctness(source["question_type"], calibrated_prediction, evaluation["reference"], evaluation["tolerance"])
        raw_vlm_correct = correctness(source["question_type"], source["raw_vlm_prediction"], evaluation["reference"], evaluation["tolerance"])
        gold_number = parse_reference_number(evaluation["reference"])
        raw_absolute_error = abs(raw_numeric_answer - gold_number) if source["question_type"] == "numerical" and gold_number is not None else None
        calibrated_absolute_error = abs(calibrated_numeric_answer - gold_number) if source["question_type"] == "numerical" and gold_number is not None else None
        row = {
            "sample_id": source["sample_id"],
            "chart_id": source["chart_id"],
            "image_path": source["image_path"],
            "question": source["question"],
            "question_type": source["question_type"],
            "options": source["options"],
            "operation": plan["operation"],
            "semantic_target": plan["semantic_target"],
            "target_categories": plan["target_categories"],
            "detected_x_axis_labels": [bar.get("detected_x_label", "") for bar in analysis["bars"]],
            "target_label_match_scores": [item["score"] for item in plan["label_matches"]],
            "selected_bar_ids": plan["target_bar_ids"],
            "selected_bar_bboxes": [bar["bbox"] for bar in selected_bars],
            "y_axis_ticks": analysis["axis"]["ticks"],
            "axis": analysis["axis"],
            "bar_tops": [[bar["target_x"], bar["target_y"]] for bar in selected_bars],
            "raw_target_values": raw_target_values,
            "calibrated_target_values": calibrated_target_values,
            "predicted_target_errors": predicted_errors,
            "raw_geometry_value": raw_numeric_answer,
            "calibrated_value": calibrated_numeric_answer,
            "raw_geometry_prediction": raw_prediction,
            "calibrated_prediction": calibrated_prediction,
            "raw_vlm_prediction": source["raw_vlm_prediction"],
            "raw_vlm_raw_response": source["raw_vlm_raw_response"],
            "gold": evaluation["reference"],
            "tolerance": evaluation["tolerance"],
            "unit": evaluation["unit"],
            "raw_vlm_correct": raw_vlm_correct,
            "raw_geometry_correct": raw_geometry_correct,
            "calibrated_geometry_correct": calibrated_correct,
            "raw_absolute_error": raw_absolute_error,
            "calibrated_absolute_error": calibrated_absolute_error,
            "raw_reasoning": raw_reasoning,
            "calibrated_reasoning": calibrated_reasoning,
            "target_audit": "automatic_exact_or_high_confidence_ocr_question_match",
            "geometry_audit": "correct" if raw_geometry_correct else ("incorrect_direct_read" if plan["operation"] == "direct" else "final_answer_incorrect_component_gold_unavailable"),
            "reasoning_audit": "supported_operation_executed_in_python",
            "failure_attribution": "none" if raw_geometry_correct else ("geometry" if plan["operation"] == "direct" else "geometry_or_semantic_decomposition"),
            "targeting_gold_access": False,
            "geometry_gold_access": False,
            "hidden_value_label_bbox_access": False,
            "phase1_chart_overlap": source["chart_id"] in phase1_charts,
        }
        overlay_path = overlay_dir / f"phase2_{int(source['sample_id']):05d}.png"
        image = cv2.imread(source["image_path"], cv2.IMREAD_COLOR)
        overlay = draw_overlay(image, source["sample_id"], source["question"], analysis, plan, raw_target_values, calibrated_target_values)
        cv2.imwrite(str(overlay_path), overlay)
        row["debug_overlay_path"] = str(overlay_path)
        accepted.append(row)
        accepted_charts.add(source["chart_id"])
        screening_rows.append({"sample_id": source["sample_id"], "chart_id": source["chart_id"], "question": source["question"], "decision": "accept", "reason": "clean_natural_no_label_bar_and_supported_target"})
        if len(accepted) >= args.max_samples:
            break

    serialized_rows: list[dict[str, Any]] = []
    for row in accepted:
        public_row = {
            key: value
            for key, value in row.items()
            if key != "image_path"
        }
        public_row["source_image_filename"] = Path(str(row["image_path"])).name
        serialized_rows.append({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in public_row.items()})
    write_csv(args.project / "results" / "phase2_sample_audit.csv", serialized_rows)
    write_csv(args.project / "results" / "phase2_screening.csv", screening_rows)
    with (args.project / "phase2_audit" / "accepted_samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in accepted:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "accepted_samples": len(accepted),
        "accepted_charts": len(accepted_charts),
        "phase1_chart_overlap": len(accepted_charts & phase1_charts),
        "operations": dict(Counter(row["operation"] for row in accepted)),
        "question_types": dict(Counter(row["question_type"] for row in accepted)),
        "raw_vlm_correct": sum(bool(row["raw_vlm_correct"]) for row in accepted),
        "raw_geometry_correct": sum(bool(row["raw_geometry_correct"]) for row in accepted),
        "calibrated_geometry_correct": sum(bool(row["calibrated_geometry_correct"]) for row in accepted),
        "screening_reasons": dict(Counter(row["reason"] for row in screening_rows)),
        "raw_vlm_sha256": sha256_file(args.raw_vlm),
        "calibrator_sha256": sha256_file(args.calibrator),
        "gpu_used": False,
    }
    write_json(args.project / "phase2_audit" / "run_summary.json", summary)
    print(json.dumps({"event": "complete", **summary}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
