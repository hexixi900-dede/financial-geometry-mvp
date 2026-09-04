from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import cv2
import numpy as np

from geometry_core import OCRToken, axis_localizer, interpolate_pixel_to_value, ocr_tokens


MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


@dataclass(frozen=True)
class AxisScalar:
    value: float
    family: str
    canonical: str


def normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _repair_digit_context(text: str) -> str:
    table = str.maketrans({"o": "0", "i": "1", "l": "1", "s": "5", "g": "9", "b": "8"})
    return text.lower().translate(table)


def _expand_year(value: int) -> int:
    if value >= 1000:
        return value
    return 2000 + value if value <= 69 else 1900 + value


def parse_axis_scalar(text: str) -> AxisScalar | None:
    raw = str(text).strip().replace("–", "-").replace("—", "-").replace("’", "'").strip("-/' ")
    compact = re.sub(r"\s+", "", raw.lower())
    repaired = _repair_digit_context(compact)
    repaired = re.sub(r"^8q(?=(?:fy)?[0-9])", "3q", repaired)
    repaired = re.sub(r"^([1-4])0fy", r"\1qfy", repaired)
    repaired = re.sub(r"^([1-4])qfx\?([0-9])", r"\1qfy2\2", repaired)

    match = re.fullmatch(r"(?:fy)?([0-9]{2,4})f?", repaired)
    if match and ("fy" in repaired or len(match.group(1)) == 4):
        year = _expand_year(int(match.group(1)))
        if not 1900 <= year <= 2100:
            return None
        suffix = "F" if repaired.endswith("f") else ""
        canonical = f"FY{year % 100:02d}{suffix}" if "fy" in repaired else f"{year}{suffix}"
        return AxisScalar(float(year), "year", canonical)

    quarter_patterns = [
        r"([1-4])q(?:fy)?([0-9]{2,4})f?",
        r"q([1-4])(?:fy)?([0-9]{2,4})f?",
    ]
    for pattern in quarter_patterns:
        match = re.fullmatch(pattern, repaired)
        if match:
            quarter = int(match.group(1))
            year = _expand_year(int(match.group(2)))
            suffix = "F" if repaired.endswith("f") else ""
            return AxisScalar(year + (quarter - 1) / 4.0, "quarter", f"{quarter}QFY{year % 100:02d}{suffix}")

    month_match = re.fullmatch(r"([a-z]{3,4})[-'/]?([0-9]{2,4})", compact)
    if month_match and month_match.group(1) in MONTHS:
        month = MONTHS[month_match.group(1)]
        year = _expand_year(int(month_match.group(2)))
        return AxisScalar(year + (month - 1) / 12.0, "month", f"{datetime(year, month, 1):%b-%y}")

    numeric = repaired.replace(",", "")
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", numeric):
        try:
            value = float(numeric)
        except ValueError:
            return None
        return AxisScalar(value, "numeric", f"{value:g}")
    return None


def scalar_to_label(value: float, family: str, examples: Iterable[str]) -> str | None:
    example_text = " ".join(str(item) for item in examples).lower()
    if family == "year":
        year = int(round(value))
        if abs(value - year) > 0.12:
            return None
        return f"FY{year % 100:02d}" if "fy" in example_text else str(year)
    if family == "quarter":
        quarter_value = round(value * 4.0) / 4.0
        if abs(value - quarter_value) > 0.08:
            return None
        year = int(math.floor(quarter_value + 1e-8))
        quarter = int(round((quarter_value - year) * 4.0)) + 1
        if quarter == 5:
            year += 1
            quarter = 1
        return f"{quarter}QFY{year % 100:02d}"
    if family == "month":
        year = int(math.floor(value + 1e-8))
        month = int(round((value - year) * 12.0)) + 1
        month = min(12, max(1, month))
        reconstructed = year + (month - 1) / 12.0
        if abs(value - reconstructed) > 0.05:
            return None
        return f"{datetime(year, month, 1):%b-%y}"
    if family == "numeric":
        rounded = round(value)
        return str(rounded) if abs(value - rounded) <= 0.05 else f"{value:g}"
    return None


def _token_to_anchor(token: OCRToken, source: str) -> dict[str, Any]:
    scalar = parse_axis_scalar(token.text)
    return {
        "label": token.text,
        "bbox": [token.x1, token.y1, token.x2, token.y2],
        "center_x": token.cx,
        "center_y": token.cy,
        "confidence": token.confidence,
        "source": source,
        "normalized": normalize_label(token.text),
        "scalar": None if scalar is None else scalar.value,
        "scalar_family": None if scalar is None else scalar.family,
        "canonical": None if scalar is None else scalar.canonical,
    }


def _map_rotated_polygon(
    polygon: Iterable[Iterable[float]],
    rotation: str,
    crop_x: int,
    crop_y: int,
    crop_width: int,
    crop_height: int,
) -> tuple[tuple[float, float], ...]:
    mapped: list[tuple[float, float]] = []
    for point in polygon:
        rx, ry = float(point[0]), float(point[1])
        if rotation == "cw":
            local_x = ry
            local_y = crop_height - 1 - rx
        elif rotation == "ccw":
            local_x = crop_width - 1 - ry
            local_y = rx
        else:
            raise ValueError(rotation)
        mapped.append((local_x + crop_x, local_y + crop_y))
    return tuple(mapped)


def _rotated_tokens(
    reader: Any,
    image_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> list[OCRToken]:
    x1, y1, x2, y2 = bbox
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return []
    crop_height, crop_width = crop.shape[:2]
    outputs: list[OCRToken] = []
    for rotation, rotated in (
        ("cw", cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)),
        ("ccw", cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ):
        raw = reader.readtext(
            rotated,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-/'",
            min_size=5,
            text_threshold=0.35,
            low_text=0.18,
            link_threshold=0.30,
            canvas_size=2560,
            mag_ratio=1.25,
        )
        for item in raw:
            if len(item) < 3:
                continue
            polygon = _map_rotated_polygon(item[0], rotation, x1, y1, crop_width, crop_height)
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            text = str(item[1])
            outputs.append(
                OCRToken(
                    index=100000 + len(outputs),
                    text=text,
                    confidence=float(item[2]),
                    polygon=polygon,
                    x1=min(xs),
                    y1=min(ys),
                    x2=max(xs),
                    y2=max(ys),
                    cx=(min(xs) + max(xs)) / 2.0,
                    cy=(min(ys) + max(ys)) / 2.0,
                    numeric_value=None,
                    numeric_suffix="",
                )
            )
    return outputs


def x_axis_band(axis: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int, float]:
    left, _, right, bottom = map(float, axis["plot_bbox"])
    zero_y = -float(axis["intercept"]) / float(axis["slope"])
    if 0 <= zero_y <= min(height - 1, bottom + 0.10 * height):
        baseline = zero_y
    else:
        baseline = bottom
    x1 = max(0, int(math.floor(left - 0.02 * width)))
    x2 = min(width, int(math.ceil(right + 0.01 * width)))
    y1 = max(0, int(math.floor(baseline - 0.035 * height)))
    y2 = min(height, int(math.ceil(max(bottom, baseline) + 0.20 * height)))
    return x1, y1, x2, y2, float(baseline)


def discover_x_axis_anchors(
    reader: Any,
    image_bgr: np.ndarray,
    tokens: list[OCRToken],
    axis: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2, baseline = x_axis_band(axis, width, height)
    axis_indices = set(int(index) for index in axis.get("token_indices", []))
    base_tokens = [
        token
        for token in tokens
        if token.index not in axis_indices
        and token.confidence >= 0.30
        and x1 <= token.cx <= x2
        and y1 <= token.cy <= y2
        and token.x2 - token.x1 <= 0.32 * width
        and token.y2 - token.y1 <= 0.24 * height
    ]
    temporal_base = [token for token in base_tokens if (parse_axis_scalar(token.text) or AxisScalar(0, "", "")).family in {"year", "quarter", "month"}]
    rotated = _rotated_tokens(reader, image_bgr, (x1, y1, x2, y2)) if len(temporal_base) < 3 else []
    rotated = [
        token
        for token in rotated
        if token.confidence >= 0.28
        and x1 <= token.cx <= x2
        and y1 <= token.cy <= y2
        and parse_axis_scalar(token.text) is not None
    ]
    candidates = [_token_to_anchor(token, "full_image_ocr") for token in base_tokens]
    candidates.extend(_token_to_anchor(token, "rotated_x_band_ocr") for token in rotated)
    candidates = [
        anchor
        for anchor in candidates
        if anchor["normalized"]
        and (
            anchor["scalar"] is not None
            or re.search(r"[A-Za-z]", str(anchor["label"])) is not None
        )
    ]
    candidates.sort(
        key=lambda anchor: (
            float(anchor["center_x"]),
            0 if anchor["source"] == "full_image_ocr" else 1,
            -float(anchor["confidence"]),
        )
    )
    deduplicated: list[dict[str, Any]] = []
    for anchor in candidates:
        duplicate_index = None
        for index, existing in enumerate(deduplicated):
            same_scalar = (
                anchor["scalar"] is not None
                and existing["scalar"] is not None
                and abs(float(anchor["scalar"]) - float(existing["scalar"])) <= 1e-6
            )
            same_text = anchor["normalized"] == existing["normalized"]
            if abs(float(anchor["center_x"]) - float(existing["center_x"])) <= max(7.0, 0.012 * width) and (same_scalar or same_text):
                duplicate_index = index
                break
        if duplicate_index is None:
            deduplicated.append(anchor)
        else:
            existing = deduplicated[duplicate_index]
            existing_score = float(existing["confidence"]) + (0.10 if existing["source"] == "full_image_ocr" else 0.0)
            anchor_score = float(anchor["confidence"]) + (0.10 if anchor["source"] == "full_image_ocr" else 0.0)
            if anchor_score > existing_score:
                deduplicated[duplicate_index] = anchor

    temporal = [anchor for anchor in deduplicated if anchor["scalar_family"] in {"year", "quarter", "month"}]
    if len(temporal) >= 3:
        ordered = sorted(temporal, key=lambda anchor: float(anchor["center_x"]))
        scalar_diffs = np.diff([float(anchor["scalar"]) for anchor in ordered])
        direction = 1.0 if float(np.median(scalar_diffs)) >= 0 else -1.0
        monotonic: list[dict[str, Any]] = []
        last = None
        for anchor in ordered:
            scalar = float(anchor["scalar"])
            if last is None or direction * (scalar - last) > -1e-6:
                monotonic.append(anchor)
                last = scalar
        temporal_ids = {id(anchor) for anchor in temporal}
        deduplicated = [anchor for anchor in deduplicated if id(anchor) not in temporal_ids] + monotonic
        # Single OCR digits from a failed vertical-date read are not useful once a
        # coherent temporal sequence has been recovered from the rotated band.
        deduplicated = [anchor for anchor in deduplicated if anchor.get("scalar_family") != "numeric"]
    deduplicated.sort(key=lambda anchor: float(anchor["center_x"]))
    audit = {
        "band_bbox": [x1, y1, x2, y2],
        "baseline_y": baseline,
        "full_band_token_count": len(base_tokens),
        "rotated_ocr_used": bool(rotated),
        "rotated_token_count": len(rotated),
        "anchor_count": len(deduplicated),
    }
    return deduplicated, audit


def _temporal_family(anchor: dict[str, Any]) -> str | None:
    family = anchor.get("scalar_family")
    return str(family) if family in {"year", "quarter", "month", "numeric"} else None


def infer_construction_target(
    construction_x: float,
    anchors: list[dict[str, Any]],
    image_width: int,
) -> dict[str, Any] | None:
    numeric = [anchor for anchor in anchors if _temporal_family(anchor) is not None]
    if len(numeric) < 2:
        return None
    numeric.sort(key=lambda anchor: float(anchor["center_x"]))
    gaps = np.diff([float(anchor["center_x"]) for anchor in numeric])
    median_gap = float(np.median(gaps[gaps > 2])) if np.any(gaps > 2) else 0.08 * image_width
    nearest = min(numeric, key=lambda anchor: abs(float(anchor["center_x"]) - construction_x))
    direct_threshold = max(8.0, min(0.30 * median_gap, 0.035 * image_width))
    if abs(float(nearest["center_x"]) - construction_x) <= direct_threshold:
        label = nearest.get("canonical") or nearest["label"]
        return {
            "target_semantic_label": str(label),
            "construction_mode": "direct_anchor",
            "construction_target_x": float(nearest["center_x"]),
            "source_anchor_labels": [nearest["label"]],
            "source_anchor_center_x": [float(nearest["center_x"])],
        }
    for left, right in zip(numeric, numeric[1:]):
        left_x = float(left["center_x"])
        right_x = float(right["center_x"])
        if not (left_x < construction_x < right_x):
            continue
        if left["scalar_family"] != right["scalar_family"]:
            continue
        ratio = (construction_x - left_x) / max(right_x - left_x, 1e-9)
        scalar = float(left["scalar"]) + ratio * (float(right["scalar"]) - float(left["scalar"]))
        target_label = scalar_to_label(scalar, str(left["scalar_family"]), [left["label"], right["label"]])
        if target_label is None:
            continue
        return {
            "target_semantic_label": target_label,
            "construction_mode": "interpolated_semantic_target",
            "construction_target_x": float(construction_x),
            "source_anchor_labels": [left["label"], right["label"]],
            "source_anchor_center_x": [left_x, right_x],
        }
    return None


def _extract_question_scalar(question: str) -> AxisScalar | None:
    patterns = [
        r"(?:[1-4]\s*q\s*(?:fy)?\s*[0-9]{2,4}f?)",
        r"(?:q\s*[1-4]\s*(?:fy)?\s*[0-9]{2,4}f?)",
        r"(?:fy\s*[0-9]{2,4}f?)",
        r"(?:[A-Za-z]{3,4}[-'/ ]+[0-9]{2,4})",
        r"(?:19[0-9]{2}|20[0-9]{2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            scalar = parse_axis_scalar(match.group(0))
            if scalar is not None:
                return scalar
    return None


def ground_question_to_x(question: str, anchors: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_question = normalize_label(question)
    direct: list[dict[str, Any]] = []
    for anchor in anchors:
        candidates = [anchor.get("normalized"), normalize_label(anchor.get("canonical") or "")]
        match_lengths = [len(value) for value in candidates if value and value in normalized_question]
        if match_lengths:
            direct.append({"anchor": anchor, "match_length": max(match_lengths)})
    if direct:
        best = max(direct, key=lambda item: (item["match_length"], float(item["anchor"]["confidence"])))
        anchor = best["anchor"]
        return {
            "status": "ok",
            "mode": "direct_anchor",
            "target_semantic_label": anchor.get("canonical") or anchor["label"],
            "matched_detected_label": anchor["label"],
            "predicted_target_x": float(anchor["center_x"]),
            "interpolation_anchors": [],
        }

    target = _extract_question_scalar(question)
    if target is None:
        return {"status": "x_grounding_unrecoverable", "reason": "question_target_not_parseable"}
    compatible = [
        anchor
        for anchor in anchors
        if anchor.get("scalar") is not None
        and (
            anchor.get("scalar_family") == target.family
            or {str(anchor.get("scalar_family")), target.family} <= {"year", "quarter", "month"}
        )
    ]
    compatible.sort(key=lambda anchor: float(anchor["scalar"]))
    for left, right in zip(compatible, compatible[1:]):
        left_value = float(left["scalar"])
        right_value = float(right["scalar"])
        if left_value <= target.value <= right_value and right_value > left_value:
            ratio = (target.value - left_value) / (right_value - left_value)
            target_x = float(left["center_x"]) + ratio * (float(right["center_x"]) - float(left["center_x"]))
            return {
                "status": "ok",
                "mode": "piecewise_linear_interpolation",
                "target_semantic_label": target.canonical,
                "matched_detected_label": None,
                "predicted_target_x": target_x,
                "interpolation_anchors": [left, right],
            }
    return {"status": "x_grounding_unrecoverable", "reason": "no_bracketing_semantic_anchors", "target_semantic_label": target.canonical}


def detect_dominant_line_series(
    image_bgr: np.ndarray,
    axis: dict[str, Any],
) -> tuple[dict[str, Any] | None, np.ndarray | None]:
    height, width = image_bgr.shape[:2]
    left, top, right, bottom = [int(round(value)) for value in axis["plot_bbox"]]
    left = max(0, min(width - 1, left))
    right = max(left + 1, min(width, right))
    top = max(0, min(height - 1, top))
    bottom = max(top + 1, min(height, bottom))
    crop = image_bgr[top:bottom, left:right]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturated = (hsv[:, :, 1] >= 38) & (hsv[:, :, 2] >= 28) & (hsv[:, :, 2] <= 252)
    if not saturated.any():
        return None, None
    quantized = (crop // 28).astype(np.uint8)
    packed = quantized[:, :, 0].astype(np.int32) + 10 * quantized[:, :, 1].astype(np.int32) + 100 * quantized[:, :, 2].astype(np.int32)
    values, counts = np.unique(packed[saturated], return_counts=True)
    candidates: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    for packed_value in values[np.argsort(counts)[::-1][:40]]:
        seed = (packed == packed_value) & saturated
        if int(seed.sum()) < 10:
            continue
        median_color = np.median(crop[seed], axis=0)
        lower = np.clip(median_color - 34, 0, 255).astype(np.uint8)
        upper = np.clip(median_color + 34, 0, 255).astype(np.uint8)
        mask = (cv2.inRange(crop, lower, upper) > 0) & saturated
        mask_u8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        for component_index in range(1, count):
            x, y, box_width, box_height, area = map(int, stats[component_index])
            coverage = box_width / max(right - left, 1)
            if coverage < 0.24 or area < 18 or box_height < 2:
                continue
            component = labels == component_index
            column_count = int(np.count_nonzero(np.any(component, axis=0)))
            fill = area / max(box_width * box_height, 1)
            score = 4.0 * coverage + 0.8 * min(column_count / max(box_width, 1), 1.0) + 0.15 * math.log1p(area) - 0.35 * min(fill, 0.8)
            full_mask = np.zeros((height, width), dtype=np.uint8)
            full_mask[top:bottom, left:right][component] = 1
            candidates.append(
                {
                    "series_id": len(candidates),
                    "median_bgr": [float(value) for value in median_color],
                    "bbox": [x + left, y + top, box_width, box_height],
                    "coverage_fraction": coverage,
                    "component_area": area,
                    "column_count": column_count,
                    "fill_ratio": fill,
                    "score": score,
                }
            )
            masks.append(full_mask)
    if not candidates:
        return None, None
    order = sorted(range(len(candidates)), key=lambda index: float(candidates[index]["score"]), reverse=True)
    best_index = order[0]
    best = dict(candidates[best_index])
    comparable: list[int] = []
    best_color = np.asarray(best["median_bgr"], dtype=float)
    for index in order[1:]:
        candidate = candidates[index]
        color_distance = float(np.linalg.norm(np.asarray(candidate["median_bgr"], dtype=float) - best_color))
        if float(candidate["score"]) >= float(best["score"]) - 0.45 and color_distance >= 28:
            comparable.append(int(candidate["series_id"]))
    best["series_candidate_count"] = len(candidates)
    best["comparable_distinct_series_count"] = 1 + len(comparable)
    best["comparable_series_ids"] = comparable
    return best, masks[best_index]


def column_point(
    series_mask: np.ndarray,
    target_x: float,
    axis: dict[str, Any],
    max_radius: int = 7,
) -> dict[str, Any] | None:
    left, top, right, bottom = map(float, axis["plot_bbox"])
    x_center = int(round(target_x))
    for radius in range(0, max_radius + 1):
        x1 = max(int(math.floor(left)), x_center - radius)
        x2 = min(int(math.ceil(right)), x_center + radius + 1)
        ys, xs = np.nonzero(series_mask[:, x1:x2])
        keep = (ys >= top) & (ys <= bottom)
        if int(np.count_nonzero(keep)) == 0:
            continue
        selected_y = ys[keep].astype(float)
        selected_x = xs[keep].astype(float) + x1
        distances = np.abs(selected_x - target_x)
        nearest = float(np.min(distances))
        near = distances <= nearest + 0.5
        return {
            "chart_type": "line",
            "method": "auto_x_column",
            "point": [float(target_x), float(np.median(selected_y[near]))],
            "target_x": float(target_x),
            "target_y": float(np.median(selected_y[near])),
            "search_radius_pixels": radius,
            "local_pixel_count": int(np.count_nonzero(near)),
            "score": 0.0,
            "both_sides": 0,
            "local_columns": int(len(np.unique(selected_x[near]))),
            "local_fit_residual_pixels": 0.0,
        }
    return None


def _robust_fit(x: np.ndarray, y: np.ndarray, target_x: float) -> dict[str, Any] | None:
    if len(x) < 4 or float(np.ptp(x)) < 2.0:
        return None
    design = np.column_stack([x - target_x, np.ones_like(x)])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    weights = np.ones(len(x), dtype=float)
    for _ in range(12):
        residual = y - design @ coefficients
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) + 0.35
        normalized = np.abs(residual) / (2.5 * scale)
        weights = np.where(normalized < 1.0, (1.0 - normalized**2) ** 2, 0.03)
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_y = y * np.sqrt(weights)
        updated, *_ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
        if float(np.max(np.abs(updated - coefficients))) < 1e-7:
            coefficients = updated
            break
        coefficients = updated
    residual = y - design @ coefficients
    return {
        "slope": float(coefficients[0]),
        "target_y": float(coefficients[1]),
        "median_residual": float(np.median(np.abs(residual))),
        "inlier_count": int(np.count_nonzero(weights >= 0.20)),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
    }


def robust_local_line_point(
    series_mask: np.ndarray,
    target_x: float,
    axis: dict[str, Any],
    window_half_width: float,
) -> dict[str, Any] | None:
    left, top, right, bottom = map(float, axis["plot_bbox"])
    x1 = max(int(math.floor(left)), int(math.floor(target_x - window_half_width)))
    x2 = min(int(math.ceil(right)), int(math.ceil(target_x + window_half_width)) + 1)
    ys, xs_local = np.nonzero(series_mask[:, x1:x2])
    xs = xs_local.astype(float) + x1
    keep = (ys >= top) & (ys <= bottom)
    xs = xs[keep]
    ys = ys[keep].astype(float)
    if len(xs) < 8:
        return None
    column_x: list[float] = []
    column_y: list[float] = []
    for unique_x in np.unique(xs):
        values = ys[xs == unique_x]
        column_x.append(float(unique_x))
        column_y.append(float(np.median(values)))
    x = np.asarray(column_x, dtype=float)
    y = np.asarray(column_y, dtype=float)
    if len(x) < 5:
        return None
    fits: list[dict[str, Any]] = []
    all_fit = _robust_fit(x, y, target_x)
    if all_fit is not None:
        all_fit["side"] = "all"
        fits.append(all_fit)
    left_fit = _robust_fit(x[x <= target_x], y[x <= target_x], target_x)
    right_fit = _robust_fit(x[x >= target_x], y[x >= target_x], target_x)
    if left_fit is not None and right_fit is not None:
        left_fit["side"] = "left"
        right_fit["side"] = "right"
        side_difference = abs(float(left_fit["target_y"]) - float(right_fit["target_y"]))
        if side_difference <= max(8.0, 0.025 * (bottom - top)):
            target_y = float(np.median([left_fit["target_y"], right_fit["target_y"]]))
            fits.extend([left_fit, right_fit])
            fit_mode = "two_sided_piecewise_robust"
        else:
            target_y = float(all_fit["target_y"]) if all_fit is not None else float(np.median(y))
            fit_mode = "global_robust_side_disagreement"
    else:
        target_y = float(all_fit["target_y"]) if all_fit is not None else float(np.median(y))
        fit_mode = "global_robust"
    residual = min(float(fit["median_residual"]) for fit in fits) if fits else math.nan
    return {
        "chart_type": "line",
        "method": "auto_x_local_line_fit",
        "point": [float(target_x), target_y],
        "target_x": float(target_x),
        "target_y": target_y,
        "search_window": [float(x1), float(x2 - 1)],
        "window_half_width": float(window_half_width),
        "fit_mode": fit_mode,
        "fitted_segments": fits,
        "score": 0.0 if math.isnan(residual) else 1.0 / (1.0 + residual),
        "both_sides": int(left_fit is not None and right_fit is not None),
        "local_columns": int(len(x)),
        "local_fit_residual_pixels": residual,
    }


def recommended_window(target_x: float, anchors: list[dict[str, Any]], image_width: int) -> float:
    distances = sorted(
        abs(float(anchor["center_x"]) - target_x)
        for anchor in anchors
        if abs(float(anchor["center_x"]) - target_x) > 3.0
    )
    nearest_spacing = distances[0] if distances else 0.09 * image_width
    return float(max(9.0, min(0.055 * image_width, 0.48 * nearest_spacing)))


def prepare_auto_geometry_chart(
    reader: Any,
    masked_bgr: np.ndarray,
    precomputed_tokens: list[OCRToken] | None = None,
    precomputed_axis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if precomputed_tokens is None:
        raw = reader.readtext(
            masked_bgr,
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
    else:
        tokens = precomputed_tokens
    height, width = masked_bgr.shape[:2]
    axis = precomputed_axis or axis_localizer(tokens, width, height)
    if axis is None:
        return {"status": "axis_unrecoverable"}
    if axis.get("right_axis_detected") or int(axis.get("additional_y_axis_count", 0)) > 0:
        return {"status": "multiple_y_axes_out_of_scope", "axis": axis}
    anchors, anchor_audit = discover_x_axis_anchors(reader, masked_bgr, tokens, axis)
    series, series_mask = detect_dominant_line_series(masked_bgr, axis)
    return {
        "status": "ok",
        "axis": axis,
        "x_axis_anchors": anchors,
        "x_axis_anchor_audit": anchor_audit,
        "series": series,
        "_series_mask": series_mask,
        "masked_ocr_token_count": len(tokens),
    }


def auto_geometry_pipeline(
    reader: Any,
    masked_bgr: np.ndarray,
    question: str,
    precomputed_tokens: list[OCRToken] | None = None,
    precomputed_axis: dict[str, Any] | None = None,
    prepared_chart: dict[str, Any] | None = None,
) -> dict[str, Any]:
    chart = prepared_chart or prepare_auto_geometry_chart(
        reader,
        masked_bgr,
        precomputed_tokens=precomputed_tokens,
        precomputed_axis=precomputed_axis,
    )
    if chart.get("status") != "ok":
        return chart
    height, width = masked_bgr.shape[:2]
    axis = chart["axis"]
    anchors = chart["x_axis_anchors"]
    grounding = ground_question_to_x(question, anchors)
    result: dict[str, Any] = {
        "status": grounding["status"],
        "axis": axis,
        "x_axis_anchors": anchors,
        "x_axis_anchor_audit": chart["x_axis_anchor_audit"],
        "grounding": grounding,
        "masked_ocr_token_count": chart["masked_ocr_token_count"],
    }
    if grounding["status"] != "ok":
        return result
    series, series_mask = chart["series"], chart["_series_mask"]
    result["series"] = series
    if series is None or series_mask is None:
        result["status"] = "series_unrecoverable"
        return result
    target_x = float(grounding["predicted_target_x"])
    window = recommended_window(target_x, anchors, width)
    result["local_search_window"] = [target_x - window, target_x + window]
    result["auto_column_geometry"] = column_point(series_mask, target_x, axis)
    result["auto_local_geometry"] = robust_local_line_point(series_mask, target_x, axis, window)
    if result["auto_column_geometry"] is None and result["auto_local_geometry"] is None:
        result["status"] = "line_point_unrecoverable"
    return result


def geometry_value(geometry: dict[str, Any] | None, axis: dict[str, Any]) -> float | None:
    if geometry is None:
        return None
    return interpolate_pixel_to_value(float(geometry["target_y"]), axis)


def draw_phase3_overlay(
    original_bgr: np.ndarray,
    sample_id: str,
    method_name: str,
    anchors: list[dict[str, Any]],
    grounding: dict[str, Any],
    axis: dict[str, Any],
    geometry: dict[str, Any] | None,
    series: dict[str, Any] | None,
    raw_value: float | None,
    pseudo_gold: float,
    oracle_geometry: dict[str, Any] | None,
) -> np.ndarray:
    overlay = original_bgr.copy()
    height, width = overlay.shape[:2]
    for tick in axis.get("ticks", []):
        y = int(round(float(tick["pixel_y"])))
        cv2.circle(overlay, (int(round(float(tick["bbox"][2]))), y), 4, (255, 255, 0), -1)
        cv2.putText(overlay, f"{float(tick['value']):g}", (3, max(13, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 120, 0), 1, cv2.LINE_AA)
    for anchor in anchors:
        x1, y1, x2, y2 = map(lambda value: int(round(float(value))), anchor["bbox"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 210, 0), 1)
        cv2.circle(overlay, (int(round(float(anchor["center_x"]))), int(round(float(anchor["center_y"])))), 3, (255, 210, 0), -1)
        cv2.putText(overlay, str(anchor.get("canonical") or anchor["label"]), (x1, max(12, y1 - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 110, 0), 1, cv2.LINE_AA)
    target_x = grounding.get("predicted_target_x")
    if target_x is not None:
        x = int(round(float(target_x)))
        cv2.line(overlay, (x, 0), (x, height - 1), (255, 0, 255), 2)
    if geometry is not None:
        window = geometry.get("search_window")
        if window:
            wx1, wx2 = map(lambda value: int(round(float(value))), window)
            cv2.rectangle(overlay, (wx1, int(axis["plot_bbox"][1])), (wx2, int(axis["plot_bbox"][3])), (0, 210, 255), 1)
        for segment in geometry.get("fitted_segments", []):
            sx1, sx2 = float(segment["x_min"]), float(segment["x_max"])
            sy1 = float(segment["target_y"]) + float(segment["slope"]) * (sx1 - float(geometry["target_x"]))
            sy2 = float(segment["target_y"]) + float(segment["slope"]) * (sx2 - float(geometry["target_x"]))
            cv2.line(overlay, (int(round(sx1)), int(round(sy1))), (int(round(sx2)), int(round(sy2))), (0, 200, 0), 2)
        px, py = map(lambda value: int(round(float(value))), geometry["point"])
        cv2.circle(overlay, (px, py), 7, (0, 0, 255), 2)
    if oracle_geometry is not None:
        ox, oy = map(lambda value: int(round(float(value))), oracle_geometry["point"])
        cv2.drawMarker(overlay, (ox, oy), (255, 0, 0), cv2.MARKER_CROSS, 14, 2)
    lines = [
        f"{sample_id} {method_name} x={grounding.get('target_semantic_label','?')} mode={grounding.get('mode','?')}",
        f"target_x={float(target_x):.2f}" if target_x is not None else "target_x=unrecoverable",
        f"raw={raw_value:.6g} gold={pseudo_gold:.6g} abs={abs(raw_value-pseudo_gold):.6g}" if raw_value is not None else f"raw=NA gold={pseudo_gold:.6g}",
    ]
    if series is not None:
        lines.append(f"series={series['series_id']} color={tuple(round(v) for v in series['median_bgr'])} candidates={series['comparable_distinct_series_count']}")
    panel_height = 21 * len(lines) + 7
    cv2.rectangle(overlay, (0, 0), (min(width - 1, 900), panel_height), (255, 255, 255), -1)
    for index, line in enumerate(lines):
        cv2.putText(overlay, line, (7, 18 + 21 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 1, cv2.LINE_AA)
    return overlay
