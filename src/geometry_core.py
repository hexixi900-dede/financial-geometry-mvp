from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any, Iterable

import cv2
import numpy as np


NUMERIC_PATTERN = re.compile(
    r"^\s*[$€£¥₹]?\s*(?P<open>\()?\s*(?P<sign>[+\-]?)\s*"
    r"(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*"
    r"(?P<suffix>%|[kKmMbBtT]|bps?|x)?\s*(?P<close>\))?\s*$"
)


@dataclass(frozen=True)
class OCRToken:
    index: int
    text: str
    confidence: float
    polygon: tuple[tuple[float, float], ...]
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    numeric_value: float | None
    numeric_suffix: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "confidence": self.confidence,
            "polygon": [list(point) for point in self.polygon],
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "center": [self.cx, self.cy],
            "numeric_value": self.numeric_value,
            "numeric_suffix": self.numeric_suffix,
        }


def token_from_dict(data: dict[str, Any]) -> OCRToken:
    bbox = [float(value) for value in data["bbox"]]
    center = [float(value) for value in data["center"]]
    numeric = data.get("numeric_value")
    return OCRToken(
        index=int(data["index"]),
        text=str(data["text"]),
        confidence=float(data["confidence"]),
        polygon=tuple((float(point[0]), float(point[1])) for point in data["polygon"]),
        x1=bbox[0],
        y1=bbox[1],
        x2=bbox[2],
        y2=bbox[3],
        cx=center[0],
        cy=center[1],
        numeric_value=None if numeric is None else float(numeric),
        numeric_suffix=str(data.get("numeric_suffix", "")),
    )


def parse_numeric_label(text: str) -> tuple[float | None, str]:
    cleaned = (
        str(text)
        .strip()
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("％", "%")
    )
    match = NUMERIC_PATTERN.fullmatch(cleaned)
    if not match:
        return None, ""
    try:
        value = float(match.group("number").replace(",", ""))
    except ValueError:
        return None, ""
    if match.group("sign") == "-" or (match.group("open") and match.group("close")):
        value = -value
    suffix = (match.group("suffix") or "").lower()
    multiplier = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}.get(suffix, 1.0)
    return value * multiplier, suffix


def ocr_tokens(raw_result: Iterable[Any]) -> list[OCRToken]:
    tokens: list[OCRToken] = []
    for index, item in enumerate(raw_result):
        if len(item) < 3:
            continue
        raw_polygon, text, confidence = item[:3]
        polygon = tuple((float(point[0]), float(point[1])) for point in raw_polygon)
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        numeric_value, suffix = parse_numeric_label(str(text))
        tokens.append(
            OCRToken(
                index=index,
                text=str(text),
                confidence=float(confidence),
                polygon=polygon,
                x1=min(xs),
                y1=min(ys),
                x2=max(xs),
                y2=max(ys),
                cx=(min(xs) + max(xs)) / 2,
                cy=(min(ys) + max(ys)) / 2,
                numeric_value=numeric_value,
                numeric_suffix=suffix,
            )
        )
    return tokens


def _repair_percent_axis_suffix(tokens: list[OCRToken], width: int) -> list[OCRToken]:
    """Repair EasyOCR's common tick error `15% -> 159` only with cohort evidence."""
    percent_tokens = [token for token in tokens if token.numeric_suffix == "%"]
    if not percent_tokens:
        return tokens
    suspicious: dict[int, float] = {}
    for token in tokens:
        if token.numeric_suffix or token.numeric_value is None:
            continue
        cleaned = token.text.strip().replace("−", "-")
        negative_parentheses = cleaned.startswith("(") and cleaned.endswith(")")
        unsigned = cleaned[1:-1] if negative_parentheses else cleaned
        sign = -1.0 if unsigned.startswith("-") or negative_parentheses else 1.0
        digits = unsigned.lstrip("+-")
        if len(digits) < 2 or not digits.isdigit() or not digits.endswith("9"):
            continue
        if min(abs(token.x2 - marker.x2) for marker in percent_tokens) > 0.055 * width:
            continue
        repaired_digits = digits[:-1]
        if not repaired_digits:
            continue
        suspicious[token.index] = sign * float(repaired_digits)
    if len(suspicious) < 2:
        return tokens
    return [
        replace(token, numeric_value=suspicious[token.index], numeric_suffix="%")
        if token.index in suspicious
        else token
        for token in tokens
    ]


def _fit_axis_side(
    tokens: list[OCRToken],
    width: int,
    height: int,
    side: str,
) -> dict[str, Any] | None:
    if side == "left":
        candidates = [
            token
            for token in tokens
            if token.numeric_value is not None
            and token.confidence >= 0.45
            and token.x2 <= 0.30 * width
            and 0.025 * height <= token.cy <= 0.94 * height
            and token.y2 - token.y1 <= 0.09 * height
        ]
        anchor = lambda token: token.x2
    elif side == "right":
        candidates = [
            token
            for token in tokens
            if token.numeric_value is not None
            and token.confidence >= 0.25
            and token.x1 >= 0.70 * width
            and 0.025 * height <= token.cy <= 0.94 * height
            and token.y2 - token.y1 <= 0.09 * height
        ]
        anchor = lambda token: token.x1
    else:
        raise ValueError(side)
    if len(candidates) < 3:
        return None
    candidates = _repair_percent_axis_suffix(candidates, width)

    best: tuple[tuple[float, ...], list[OCRToken], float, float] | None = None
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1 :]:
            delta_y = second.cy - first.cy
            delta_value = float(second.numeric_value) - float(first.numeric_value)
            if abs(delta_y) < 0.10 * height or abs(delta_value) < 1e-12:
                continue
            if abs(anchor(first) - anchor(second)) > 0.055 * width:
                continue
            slope = delta_value / delta_y
            if side == "left" and slope >= 0:
                continue
            intercept = float(first.numeric_value) - slope * first.cy
            anchor_mid = (anchor(first) + anchor(second)) / 2
            inliers: list[OCRToken] = []
            for token in candidates:
                if abs(anchor(token) - anchor_mid) > 0.045 * width:
                    continue
                residual_pixels = abs((float(token.numeric_value) - (slope * token.cy + intercept)) / slope)
                if residual_pixels <= max(4.0, 0.014 * height):
                    inliers.append(token)
            if len(inliers) < 3:
                continue
            y_span = max(token.cy for token in inliers) - min(token.cy for token in inliers)
            if y_span < 0.20 * height:
                continue
            x_spread = float(np.std([anchor(token) for token in inliers]))
            score = (
                float(len(inliers)),
                y_span / height,
                -x_spread / width,
                float(np.mean([token.confidence for token in inliers])),
            )
            if best is None or score > best[0]:
                best = (score, inliers, slope, intercept)
    if best is None:
        return None

    _, inliers, _, _ = best
    # Keep the best-confidence token when OCR produced duplicate readings at one y.
    deduplicated: list[OCRToken] = []
    for token in sorted(inliers, key=lambda item: (item.cy, -item.confidence)):
        if deduplicated and abs(token.cy - deduplicated[-1].cy) <= 3:
            if token.confidence > deduplicated[-1].confidence:
                deduplicated[-1] = token
        else:
            deduplicated.append(token)
    if len(deduplicated) < 3:
        return None
    ys = np.asarray([token.cy for token in deduplicated], dtype=float)
    values = np.asarray([float(token.numeric_value) for token in deduplicated], dtype=float)
    slope, intercept = np.polyfit(ys, values, 1)
    predicted = slope * ys + intercept
    total = float(np.sum((values - values.mean()) ** 2))
    residual = float(np.sum((values - predicted) ** 2))
    r_squared = 1.0 - residual / total if total > 1e-18 else 0.0
    residual_pixels = np.abs((values - predicted) / slope)
    if (
        (side == "left" and slope >= 0)
        or abs(slope) < 1e-12
        or r_squared < 0.985
        or float(np.max(residual_pixels)) > max(5.0, 0.018 * height)
    ):
        return None
    ordered = sorted(deduplicated, key=lambda token: token.cy)
    pixel_diffs = np.diff([token.cy for token in ordered])
    value_diffs = np.abs(np.diff([float(token.numeric_value) for token in ordered]))
    median_pixel_step = float(np.median(pixel_diffs)) if len(pixel_diffs) else 0.0
    median_value_step = float(np.median(value_diffs)) if len(value_diffs) else 0.0
    return {
        "side": side,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "max_residual_pixels": float(np.max(residual_pixels)),
        "median_pixel_step": median_pixel_step,
        "median_value_step": median_value_step,
        "token_indices": [token.index for token in ordered],
        "ticks": [
            {
                "value": float(token.numeric_value),
                "pixel_y": float(token.cy),
                "text": token.text,
                "confidence": token.confidence,
                "bbox": [token.x1, token.y1, token.x2, token.y2],
            }
            for token in ordered
        ],
    }


def axis_localizer(tokens: list[OCRToken], width: int, height: int) -> dict[str, Any] | None:
    left = _fit_axis_side(tokens, width, height, "left")
    if left is None:
        return None
    right = _fit_axis_side(tokens, width, height, "right")
    step = max(float(left["median_pixel_step"]), 8.0)
    tick_ys = [float(tick["pixel_y"]) for tick in left["ticks"]]
    tick_rights = [float(tick["bbox"][2]) for tick in left["ticks"]]
    plot_top = max(0.015 * height, min(tick_ys) - 0.55 * step)
    plot_bottom = min(0.95 * height, max(tick_ys) + 0.55 * step)
    plot_left = min(0.36 * width, max(tick_rights) + 0.012 * width)
    plot_right = 0.985 * width
    left["plot_bbox"] = [float(plot_left), float(plot_top), float(plot_right), float(plot_bottom)]
    left["right_axis_detected"] = right is not None
    left["right_axis"] = right
    left_anchor = float(np.median([tick["bbox"][2] for tick in left["ticks"]]))
    numeric_tokens = [
        token
        for token in tokens
        if token.numeric_value is not None
        and token.confidence >= 0.50
        and 0.02 * height <= token.cy <= 0.94 * height
    ]
    additional_groups: list[dict[str, Any]] = []
    for first_index, first in enumerate(numeric_tokens):
        for second in numeric_tokens[first_index + 1 :]:
            if abs(first.x2 - second.x2) > 0.04 * width or abs(first.cy - second.cy) < 0.12 * height:
                continue
            slope = (float(second.numeric_value) - float(first.numeric_value)) / (second.cy - first.cy)
            if slope >= 0 or abs(slope) < 1e-12:
                continue
            intercept = float(first.numeric_value) - slope * first.cy
            anchor_mid = (first.x2 + second.x2) / 2
            inliers = [
                token
                for token in numeric_tokens
                if abs(token.x2 - anchor_mid) <= 0.035 * width
                and abs((float(token.numeric_value) - (slope * token.cy + intercept)) / slope)
                <= max(4.0, 0.014 * height)
            ]
            if len(inliers) < 3:
                continue
            y_span = max(token.cy for token in inliers) - min(token.cy for token in inliers)
            if y_span < 0.20 * height:
                continue
            indices = set(token.index for token in inliers)
            anchor_value = float(np.median([token.x2 for token in inliers]))
            if abs(anchor_value - left_anchor) <= 0.09 * width:
                continue
            if any(
                len(indices & set(group["token_indices"]))
                / max(1, len(indices | set(group["token_indices"])))
                >= 0.50
                for group in additional_groups
            ):
                continue
            additional_groups.append(
                {
                    "anchor_x": anchor_value,
                    "token_indices": sorted(indices),
                    "tick_count": len(indices),
                    "y_span": y_span,
                }
            )
    left["additional_y_axes"] = additional_groups
    left["additional_y_axis_count"] = len(additional_groups)
    top_value = interpolate_pixel_to_value(plot_top, left)
    bottom_value = interpolate_pixel_to_value(plot_bottom, left)
    left["axis_span"] = abs(top_value - bottom_value)
    left["axis_min_display"] = min(top_value, bottom_value)
    left["axis_max_display"] = max(top_value, bottom_value)
    return left


def interpolate_pixel_to_value(pixel_y: float, axis: dict[str, Any]) -> float:
    return float(axis["slope"]) * float(pixel_y) + float(axis["intercept"])


def data_label_candidates(
    tokens: list[OCRToken],
    axis: dict[str, Any],
    width: int,
    height: int,
) -> list[OCRToken]:
    plot_left, plot_top, plot_right, plot_bottom = map(float, axis["plot_bbox"])
    axis_indices = set(int(index) for index in axis["token_indices"])
    tick_step = max(float(axis["median_value_step"]), 1e-12)
    display_min = float(axis["axis_min_display"])
    display_max = float(axis["axis_max_display"])
    candidates: list[OCRToken] = []
    numeric_tokens = [
        token
        for token in tokens
        if token.numeric_value is not None and token.confidence >= 0.45
    ]
    for token in tokens:
        value = token.numeric_value
        if token.index in axis_indices or value is None or not math.isfinite(value):
            continue
        if token.confidence < 0.68:
            continue
        if not (plot_left + 0.018 * width <= token.cx <= plot_right - 0.018 * width):
            continue
        if not (plot_top - 0.035 * height <= token.cy <= plot_bottom - 0.018 * height):
            continue
        if token.y2 - token.y1 > 0.075 * height or token.x2 - token.x1 > 0.16 * width:
            continue
        if value < display_min - 0.75 * tick_step or value > display_max + 0.75 * tick_step:
            continue
        digits_only = re.fullmatch(r"\d{4}", token.text.strip())
        if digits_only and 1900 <= value <= 2100 and not (1900 <= display_min <= 2100):
            continue
        horizontal_peers = [
            other
            for other in numeric_tokens
            if other.index != token.index
            and abs(other.cy - token.cy) <= 0.026 * height
            and abs(other.cx - token.cx) >= 0.035 * width
        ]
        if len(horizontal_peers) >= 2:
            # Numeric rows with three or more aligned items are overwhelmingly x-axis ticks.
            continue
        candidates.append(token)
    return sorted(candidates, key=lambda token: (-token.confidence, token.cy, token.cx))


def _quantized_components(image_bgr: np.ndarray, axis: dict[str, Any]) -> list[dict[str, Any]]:
    height, width = image_bgr.shape[:2]
    left, top, right, bottom = [int(round(value)) for value in axis["plot_bbox"]]
    left = max(0, min(width - 1, left))
    right = max(left + 1, min(width, right))
    top = max(0, min(height - 1, top))
    bottom = max(top + 1, min(height, bottom))
    crop = image_bgr[top:bottom, left:right]
    quantized = (crop // 24).astype(np.uint8)
    packed = (
        quantized[:, :, 0].astype(np.int32)
        + 11 * quantized[:, :, 1].astype(np.int32)
        + 121 * quantized[:, :, 2].astype(np.int32)
    )
    values, counts = np.unique(packed, return_counts=True)
    order = np.argsort(counts)[::-1]
    components: list[dict[str, Any]] = []
    crop_area = crop.shape[0] * crop.shape[1]
    for packed_value in values[order[:48]]:
        raw_mask = (packed == packed_value).astype(np.uint8)
        fraction = float(raw_mask.mean())
        if fraction < 0.00008 or fraction > 0.22:
            continue
        pixel_colors = crop[raw_mask.astype(bool)]
        median_color = np.median(pixel_colors, axis=0)
        color_range = float(np.max(median_color) - np.min(median_color))
        brightness = float(np.mean(median_color))
        if brightness > 242 and color_range < 15:
            continue
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for component_index in range(1, count):
            x, y, box_w, box_h, area = map(int, stats[component_index])
            if area < 0.00008 * crop_area:
                continue
            component_mask = labels == component_index
            components.append(
                {
                    "x": x + left,
                    "y": y + top,
                    "w": box_w,
                    "h": box_h,
                    "area": area,
                    "fill": area / max(1, box_w * box_h),
                    "median_bgr": median_color.tolist(),
                    "color_range": color_range,
                    "brightness": brightness,
                    "mask": component_mask,
                    "mask_offset": [left, top],
                }
            )
    return components


def get_bar(
    image_bgr: np.ndarray,
    label: OCRToken,
    axis: dict[str, Any],
) -> dict[str, Any] | None:
    height, width = image_bgr.shape[:2]
    plot_left, plot_top, plot_right, plot_bottom = map(float, axis["plot_bbox"])
    zero_pixel = -float(axis["intercept"]) / float(axis["slope"])
    expected_base = zero_pixel if plot_top <= zero_pixel <= plot_bottom + 0.08 * height else plot_bottom
    components = _quantized_components(image_bgr, axis)
    bar_like: list[dict[str, Any]] = []
    for component in components:
        x, y, box_w, box_h = (component[key] for key in ("x", "y", "w", "h"))
        if box_w < max(5, 0.008 * width) or box_h < max(9, 0.025 * height):
            continue
        if box_w > 0.18 * width or box_h > 0.92 * (plot_bottom - plot_top):
            continue
        if component["fill"] < 0.48 or box_h < 0.60 * box_w:
            continue
        bar_like.append(component)
    candidates: list[dict[str, Any]] = []
    for component in bar_like:
        x, y, box_w, box_h = (component[key] for key in ("x", "y", "w", "h"))
        if not (x - 0.30 * box_w <= label.cx <= x + 1.30 * box_w):
            continue
        if y > label.y2 + 0.20 * height or y + box_h < label.y1 - 0.02 * height:
            continue
        if label.y2 < y:
            label_top_gap = y - label.y2
            if label_top_gap > 0.075 * height:
                continue
        elif label.y1 > y:
            label_depth = (label.cy - y) / max(box_h, 1)
            if label_depth > 0.38:
                continue
        else:
            # The mask would cross the bar top, violating the masked-value protocol.
            continue
        base_distance = abs((y + box_h) - expected_base) / max(height, 1)
        if base_distance > 0.16:
            continue
        stacked_evidence = 0
        for other in components:
            if other is component:
                continue
            other_x, other_y, other_w, other_h = (other[key] for key in ("x", "y", "w", "h"))
            if (
                other["fill"] < 0.30
                or other_w < 0.50 * box_w
                or other_h < max(7, 0.018 * height)
                or other_w > 1.55 * box_w
            ):
                continue
            overlap = max(0, min(x + box_w, other_x + other_w) - max(x, other_x))
            overlap_ratio = overlap / max(1, min(box_w, other_w))
            if overlap_ratio < 0.55:
                continue
            touches_top = abs((other_y + other_h) - y) <= 0.045 * height
            touches_bottom = abs((y + box_h) - other_y) <= 0.045 * height
            if touches_top or touches_bottom:
                stacked_evidence += 1
        if stacked_evidence:
            continue
        top_to_label = min(abs(y - label.y1), abs(y - label.y2)) / max(height, 1)
        center_distance = abs((x + box_w / 2) - label.cx) / max(width, 1)
        score = (
            2.2 * float(component["fill"])
            + 1.0 * min(box_h / max(plot_bottom - plot_top, 1), 1.0)
            - 4.0 * base_distance
            - 2.5 * center_distance
            - 1.5 * min(top_to_label, 0.25)
        )
        candidates.append(
            {
                "chart_type": "bar",
                "bbox": [int(x), int(y), int(box_w), int(box_h)],
                "target_x": float(x + box_w / 2),
                "target_y": float(y),
                "score": float(score),
                "fill_ratio": float(component["fill"]),
                "base_distance_pixels": float(abs((y + box_h) - expected_base)),
                "median_bgr": component["median_bgr"],
                "stacked_evidence": stacked_evidence,
            }
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["score"])


def get_edgepoint(
    image_bgr: np.ndarray,
    label: OCRToken,
    axis: dict[str, Any],
) -> dict[str, Any] | None:
    height, width = image_bgr.shape[:2]
    plot_left, plot_top, plot_right, plot_bottom = map(float, axis["plot_bbox"])
    crop = image_bgr[int(plot_top) : int(plot_bottom), int(plot_left) : int(plot_right)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturated = (hsv[:, :, 1] >= 42) & (hsv[:, :, 2] >= 35) & (hsv[:, :, 2] <= 252)
    quantized = (crop // 28).astype(np.uint8)
    packed = (
        quantized[:, :, 0].astype(np.int32)
        + 10 * quantized[:, :, 1].astype(np.int32)
        + 100 * quantized[:, :, 2].astype(np.int32)
    )
    values, counts = np.unique(packed[saturated], return_counts=True) if saturated.any() else ([], [])
    if len(values) == 0:
        return None
    order = np.argsort(counts)[::-1]
    target_x_local = label.cx - plot_left
    candidates: list[dict[str, Any]] = []
    for packed_value in np.asarray(values)[order[:36]]:
        seed = (packed == packed_value) & saturated
        if int(seed.sum()) < 8:
            continue
        colors = crop[seed]
        median_color = np.median(colors, axis=0)
        lower = np.clip(median_color - 32, 0, 255).astype(np.uint8)
        upper = np.clip(median_color + 32, 0, 255).astype(np.uint8)
        color_mask = cv2.inRange(crop, lower, upper) > 0
        color_mask &= saturated
        mask_u8 = cv2.morphologyEx(color_mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        for component_index in range(1, count):
            x, y, box_w, box_h, area = map(int, stats[component_index])
            if box_w < 0.07 * (plot_right - plot_left) or area < 12:
                continue
            if not (x - 0.045 * width <= target_x_local <= x + box_w + 0.045 * width):
                continue
            component_mask = labels == component_index
            ys, xs = np.nonzero(component_mask)
            local_window = np.abs(xs - target_x_local) <= max(8.0, 0.045 * width)
            xs = xs[local_window]
            ys = ys[local_window]
            if len(xs) < 8 or len(np.unique(xs)) < 4:
                continue
            column_xs: list[float] = []
            column_ys: list[float] = []
            for unique_x in np.unique(xs):
                y_values = ys[xs == unique_x]
                column_xs.append(float(unique_x))
                column_ys.append(float(np.median(y_values)))
            column_x_array = np.asarray(column_xs)
            column_y_array = np.asarray(column_ys)
            distances = np.abs(column_x_array - target_x_local)
            weights = 1.0 / (1.0 + distances)
            design = np.column_stack([column_x_array - target_x_local, np.ones_like(column_x_array)])
            weighted_design = design * np.sqrt(weights)[:, None]
            weighted_target = column_y_array * np.sqrt(weights)
            coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
            predicted_y_local = float(coefficients[1])
            predicted_y = predicted_y_local + plot_top
            nearest_label_distance = min(abs(predicted_y - label.y1), abs(predicted_y - label.y2))
            if nearest_label_distance > 0.19 * height:
                continue
            both_sides = bool(np.any(column_x_array < target_x_local - 2) and np.any(column_x_array > target_x_local + 2))
            coverage = (float(column_x_array.max()) - float(column_x_array.min())) / max(width, 1)
            residual = float(np.median(np.abs(design @ coefficients - column_y_array)))
            score = (
                2.0 * min(box_w / max(plot_right - plot_left, 1), 1.0)
                + 0.7 * both_sides
                + 2.0 * min(coverage, 0.15)
                - 0.015 * nearest_label_distance
                - 0.03 * residual
            )
            candidates.append(
                {
                    "chart_type": "line",
                    "point": [float(label.cx), predicted_y],
                    "target_x": float(label.cx),
                    "target_y": predicted_y,
                    "score": float(score),
                    "both_sides": int(both_sides),
                    "local_columns": len(column_xs),
                    "local_fit_residual_pixels": residual,
                    "median_bgr": median_color.tolist(),
                }
            )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["score"])


def choose_geometry_target(
    image_bgr: np.ndarray,
    label: OCRToken,
    axis: dict[str, Any],
) -> dict[str, Any] | None:
    bar = get_bar(image_bgr, label, axis)
    line = get_edgepoint(image_bgr, label, axis)
    if bar is not None and float(bar["score"]) >= 0.80:
        # A saturated horizontal bar-top edge can look like a line component.
        # Strong rectangular/base evidence is therefore authoritative in the simple-chart MVP.
        return bar
    if line is not None and float(line["score"]) >= 0.25:
        # A line superposed on several bar rectangles is a combination chart, outside MVP scope.
        plot_height = max(1.0, float(axis["plot_bbox"][3]) - float(axis["plot_bbox"][1]))
        bar_like_count = sum(
            component["fill"] >= 0.48
            and component["w"] >= max(5, 0.008 * image_bgr.shape[1])
            and component["h"] >= max(9, 0.025 * image_bgr.shape[0])
            and component["h"] >= 0.60 * component["w"]
            and component["h"] <= 0.92 * plot_height
            for component in _quantized_components(image_bgr, axis)
        )
        if bar_like_count >= 3:
            return None
        line["bar_like_component_count"] = int(bar_like_count)
        return line
    return bar if bar is not None and float(bar["score"]) >= 0.55 else None


def label_preserves_geometry(label: OCRToken, geometry: dict[str, Any], margin: float = 2.0) -> bool:
    target_x = float(geometry["target_x"])
    target_y = float(geometry["target_y"])
    inside_x = label.x1 - margin <= target_x <= label.x2 + margin
    inside_y = label.y1 - margin <= target_y <= label.y2 + margin
    return not (inside_x and inside_y)


def mask_value_text(image_bgr: np.ndarray, label: OCRToken) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    polygon = np.asarray(label.polygon, dtype=np.float32)
    center = polygon.mean(axis=0, keepdims=True)
    expanded = center + 1.04 * (polygon - center)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, image_bgr.shape[1] - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, image_bgr.shape[0] - 1)
    cv2.fillConvexPoly(mask, np.round(expanded).astype(np.int32), 255)
    masked = cv2.inpaint(image_bgr, mask, 3, cv2.INPAINT_TELEA)
    return masked, mask


def geometry_value(geometry: dict[str, Any], axis: dict[str, Any]) -> float:
    return interpolate_pixel_to_value(float(geometry["target_y"]), axis)


def draw_debug_overlay(
    original_bgr: np.ndarray,
    label: OCRToken,
    axis: dict[str, Any],
    geometry: dict[str, Any],
    raw_value: float,
    pseudo_gold: float,
    corrected_value: float | None = None,
) -> np.ndarray:
    overlay = original_bgr.copy()
    height, width = overlay.shape[:2]
    for tick in axis["ticks"]:
        y = int(round(float(tick["pixel_y"])))
        cv2.line(overlay, (0, y), (min(width - 1, int(0.22 * width)), y), (255, 255, 0), 1)
        cv2.circle(overlay, (int(round(float(tick["bbox"][2]))), y), 4, (255, 255, 0), -1)
        cv2.putText(
            overlay,
            f"{float(tick['value']):g}@{y}",
            (max(2, int(round(float(tick["bbox"][2]))) + 5), max(14, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 120, 0),
            1,
            cv2.LINE_AA,
        )
    x1, y1, x2, y2 = map(lambda value: int(round(value)), (label.x1, label.y1, label.x2, label.y2))
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if geometry["chart_type"] == "bar":
        x, y, box_w, box_h = map(int, geometry["bbox"])
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 200, 0), 2)
        cv2.line(overlay, (x, y), (x + box_w, y), (0, 0, 255), 2)
    else:
        x, y = map(lambda value: int(round(value)), geometry["point"])
        cv2.circle(overlay, (x, y), 7, (0, 0, 255), 2)
        cv2.line(overlay, (x, y), (max(0, int(axis["plot_bbox"][0])), y), (0, 200, 0), 1)
    lines = [
        f"type={geometry['chart_type']} target=({geometry['target_x']:.1f},{geometry['target_y']:.1f})",
        f"raw={raw_value:.6g} gold={pseudo_gold:.6g} abs={abs(raw_value-pseudo_gold):.6g}",
    ]
    if corrected_value is not None:
        lines.append(f"corrected={corrected_value:.6g} abs={abs(corrected_value-pseudo_gold):.6g}")
    panel_height = 22 * len(lines) + 8
    cv2.rectangle(overlay, (0, 0), (min(width - 1, 760), panel_height), (255, 255, 255), -1)
    for index, text in enumerate(lines):
        cv2.putText(
            overlay,
            text,
            (8, 20 + 22 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return overlay
