"""Phase 4 multi-line series detection and series-to-legend color matching.

This module generalizes the Phase 3 single dominant-series detector to return
every line-like colored component inside the plot area, and adds deterministic
color-based matching between legend entries and detected line components.

No function here receives pseudo-Gold, masked value-label boxes, or label
centers. Everything operates on the (masked) chart image plus OCR-derived
structures only.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def _bgr_to_lab(bgr: Any) -> np.ndarray:
    pixel = np.asarray(bgr, dtype=np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2Lab).reshape(3).astype(float)


def _bgr_to_hsv(bgr: Any) -> np.ndarray:
    pixel = np.asarray(bgr, dtype=np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV).reshape(3).astype(float)


def _component_color(crop: np.ndarray, mask: np.ndarray, channel: str) -> np.ndarray:
    colors = crop[mask]
    if channel == "neutral" and len(colors) >= 8:
        # A neutral component mixes the dark stroke core with gray anti-aliasing;
        # the darker part is the stroke's identity color.
        gray = cv2.cvtColor(colors.reshape(1, -1, 3), cv2.COLOR_BGR2GRAY).reshape(-1).astype(float)
        cutoff = np.percentile(gray, 60.0)
        darker = colors[gray <= cutoff]
        if len(darker) >= 4:
            return np.median(darker, axis=0)
    return np.median(colors, axis=0)


def _seed_components(
    crop: np.ndarray,
    feature_mask: np.ndarray,
    left: int,
    top: int,
    plot_width: int,
    image_shape: tuple[int, int],
    max_seed_colors: int,
    min_coverage: float,
    channel: str,
) -> list[dict[str, Any]]:
    height, width = image_shape
    quantized = (crop // 28).astype(np.uint8)
    packed = (
        quantized[:, :, 0].astype(np.int32)
        + 10 * quantized[:, :, 1].astype(np.int32)
        + 100 * quantized[:, :, 2].astype(np.int32)
    )
    values, counts = np.unique(packed[feature_mask], return_counts=True)
    components: list[dict[str, Any]] = []
    for packed_value in values[np.argsort(counts)[::-1][:max_seed_colors]]:
        seed = (packed == packed_value) & feature_mask
        if int(seed.sum()) < 10:
            continue
        median_color = np.median(crop[seed], axis=0)
        lower = np.clip(median_color - 34, 0, 255).astype(np.uint8)
        upper = np.clip(median_color + 34, 0, 255).astype(np.uint8)
        mask = (cv2.inRange(crop, lower, upper) > 0) & feature_mask
        mask_u8 = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        for component_index in range(1, count):
            x, y, box_width, box_height, area = map(int, stats[component_index])
            coverage = box_width / plot_width
            if coverage < min_coverage or area < 18 or box_height < 2:
                continue
            component_mask = labels == component_index
            column_count = int(np.count_nonzero(np.any(component_mask, axis=0)))
            column_y_median: list[float] = []
            for column_x in np.nonzero(np.any(component_mask, axis=0))[0]:
                column_y_median.append(float(np.median(np.nonzero(component_mask[:, column_x])[0])))
            y_std = float(np.std(column_y_median)) if len(column_y_median) >= 4 else 0.0
            full_mask = np.zeros((height, width), dtype=np.uint8)
            full_mask[top : top + crop.shape[0], left : left + crop.shape[1]][component_mask] = 1
            components.append(
                {
                    "median_bgr": [float(value) for value in _component_color(crop, component_mask, channel)],
                    "bbox": [x + left, y + top, box_width, box_height],
                    "coverage_fraction": coverage,
                    "component_area": area,
                    "column_count": column_count,
                    "fill_ratio": area / max(box_width * box_height, 1),
                    "centerline_y_std": y_std,
                    "mask": full_mask,
                }
            )
    return components


def detect_line_series_components(
    image_bgr: np.ndarray,
    axis: dict[str, Any],
    max_seed_colors: int = 40,
    min_coverage: float = 0.24,
    merge_color_distance: float = 30.0,
) -> list[dict[str, Any]]:
    """Return every line-like colored component inside the plot area.

    Two pixel channels are used: a saturated channel for colored lines and a
    neutral channel (dark/gray, low saturation) for black/gray lines. Neutral
    components that mostly overlap a saturated component are dropped (they are
    anti-aliasing shadows), and perfectly flat full-width neutral strokes are
    dropped as gridlines/spines. Same-color touching components are merged so
    one anti-aliased stroke split across quantization bins becomes one series.
    """
    height, width = image_bgr.shape[:2]
    left, top, right, bottom = [int(round(value)) for value in axis["plot_bbox"]]
    left = max(0, min(width - 1, left))
    right = max(left + 1, min(width, right))
    top = max(0, min(height - 1, top))
    bottom = max(top + 1, min(height, bottom))
    plot_width = max(right - left, 1)
    plot_height = max(bottom - top, 1)
    crop = image_bgr[top:bottom, left:right]
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturated = (hsv[:, :, 1] >= 38) & (hsv[:, :, 2] >= 28) & (hsv[:, :, 2] <= 252)
    neutral = (hsv[:, :, 1] < 60) & (hsv[:, :, 2] >= 30) & (hsv[:, :, 2] <= 215)

    raw_components = _seed_components(crop, saturated, left, top, plot_width, (height, width), max_seed_colors, min_coverage, "saturated")
    neutral_components = _seed_components(crop, neutral, left, top, plot_width, (height, width), max_seed_colors, min_coverage, "neutral")

    kept_neutral: list[dict[str, Any]] = []
    saturated_masks = [cv2.dilate(item["mask"], np.ones((5, 5), np.uint8)) > 0 for item in raw_components]
    for item in neutral_components:
        # Gridline/spine guard: a perfectly flat full-width neutral stroke.
        if item["coverage_fraction"] >= 0.95 and item["centerline_y_std"] < 2.0:
            continue
        # Anti-aliasing shadow of a colored line.
        shadow = False
        item_pixels = int(np.count_nonzero(item["mask"]))
        for saturated_mask in saturated_masks:
            overlap = int(np.count_nonzero((cv2.dilate(item["mask"], np.ones((5, 5), np.uint8)) > 0) & saturated_mask))
            if overlap > 0.5 * item_pixels:
                shadow = True
                break
        if not shadow:
            kept_neutral.append(item)
    raw_components.extend(kept_neutral)
    if not raw_components:
        return []

    # Merge components that share nearly the same color and touch each other:
    # one anti-aliased stroke can otherwise appear as several series.
    parent = list(range(len(raw_components)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    dilated = [cv2.dilate(item["mask"], np.ones((5, 5), np.uint8)) > 0 for item in raw_components]
    for first in range(len(raw_components)):
        color_first = np.asarray(raw_components[first]["median_bgr"], dtype=float)
        for second in range(first + 1, len(raw_components)):
            color_second = np.asarray(raw_components[second]["median_bgr"], dtype=float)
            distance = float(np.linalg.norm(color_first - color_second))
            # Nearly identical colors are the same stroke split by occlusion even
            # when the fragments no longer touch; slightly different colors must
            # also touch to merge.
            touching = bool(np.logical_and(dilated[first], dilated[second]).any())
            if distance < 20.0 or (distance <= merge_color_distance and touching):
                union(first, second)

    groups: dict[int, list[int]] = {}
    for index in range(len(raw_components)):
        groups.setdefault(find(index), []).append(index)

    series: list[dict[str, Any]] = []
    for root in sorted(groups):
        members = groups[root]
        merged_mask = np.zeros((height, width), dtype=np.uint8)
        weighted_color = np.zeros(3, dtype=float)
        total_area = 0
        for member in members:
            merged_mask |= raw_components[member]["mask"]
            area = int(raw_components[member]["component_area"])
            weighted_color += area * np.asarray(raw_components[member]["median_bgr"], dtype=float)
            total_area += area
        ys, xs = np.nonzero(merged_mask)
        if len(xs) == 0 or total_area == 0:
            continue
        median_bgr = weighted_color / total_area
        box_x, box_y = int(xs.min()), int(ys.min())
        box_width = int(xs.max() - xs.min() + 1)
        box_height = int(ys.max() - ys.min() + 1)
        coverage = box_width / plot_width
        if coverage < min_coverage:
            continue
        column_count = int(np.count_nonzero(np.any(merged_mask, axis=0)))
        fill = total_area / max(box_width * box_height, 1)
        score = (
            4.0 * coverage
            + 0.8 * min(column_count / max(box_width, 1), 1.0)
            + 0.15 * math.log1p(total_area)
            - 0.35 * min(fill, 0.8)
        )
        series.append(
            {
                "series_id": len(series),
                "median_bgr": [float(value) for value in median_bgr],
                "median_hsv": [float(value) for value in _bgr_to_hsv(np.round(median_bgr))],
                "median_lab": [float(value) for value in _bgr_to_lab(np.round(median_bgr))],
                "bbox": [box_x, box_y, box_width, box_height],
                "coverage_fraction": coverage,
                "component_area": int(total_area),
                "column_count": column_count,
                "fill_ratio": fill,
                "style_hint": "dashed" if column_count / max(box_width, 1) < 0.55 else "solid",
                "score": score,
                "merged_component_count": len(members),
                "mask": merged_mask,
            }
        )
    series.sort(key=lambda item: (-float(item["score"]), item["bbox"][0], item["bbox"][1]))
    for index, item in enumerate(series):
        item["series_id"] = index
    return series


def color_distance(bgr_a: Any, bgr_b: Any) -> dict[str, float]:
    """Combined RGB/Lab distance between two BGR colors."""
    a = np.asarray(bgr_a, dtype=float)
    b = np.asarray(bgr_b, dtype=float)
    lab_a = _bgr_to_lab(np.clip(np.round(a), 0, 255))
    lab_b = _bgr_to_lab(np.clip(np.round(b), 0, 255))
    rgb = float(np.linalg.norm(a - b))
    lab = float(np.linalg.norm(lab_a - lab_b))
    return {"rgb": rgb, "lab": lab, "combined": 0.5 * (rgb + lab)}


def match_series_by_color(
    legend_entries: list[dict[str, Any]],
    series: list[dict[str, Any]],
) -> dict[str, Any]:
    """Greedy deterministic color assignment between legend entries and series.

    Returns per-entry matches plus margins so callers can apply explainable
    ambiguity rules (close colors, missing entries, unmatched series).
    """
    entry_matches: list[dict[str, Any]] = []
    for entry_index, entry in enumerate(legend_entries):
        swatch = entry.get("swatch") or {}
        swatch_bgr = swatch.get("median_bgr")
        scored: list[tuple[float, int]] = []
        if swatch_bgr is not None:
            for component in series:
                distance = color_distance(swatch_bgr, component["median_bgr"])["combined"]
                scored.append((distance, int(component["series_id"])))
        scored.sort()
        best_distance, best_id = scored[0] if scored else (None, None)
        second_distance = scored[1][0] if len(scored) > 1 else None
        entry_matches.append(
            {
                "legend_index": entry_index,
                "legend_name": entry.get("name"),
                "matched_series_id": best_id,
                "color_distance": best_distance,
                "second_best_color_distance": second_distance,
                "color_margin": None if best_distance is None or second_distance is None else second_distance - best_distance,
            }
        )
    matched_ids = {match["matched_series_id"] for match in entry_matches if match["matched_series_id"] is not None}
    unmatched_series_ids = [int(item["series_id"]) for item in series if int(item["series_id"]) not in matched_ids]
    return {
        "entry_matches": entry_matches,
        "unmatched_series_ids": unmatched_series_ids,
    }


def local_continuity(series_mask: np.ndarray, target_x: float, window: float) -> dict[str, Any]:
    """Column coverage and gap diagnostics of a series mask near target_x."""
    height, width = series_mask.shape
    x1 = max(0, int(math.floor(target_x - window)))
    x2 = min(width, int(math.ceil(target_x + window)) + 1)
    if x2 <= x1:
        return {"columns_present": 0, "columns_total": 0, "column_coverage": 0.0, "largest_gap": 0}
    window_mask = series_mask[:, x1:x2]
    columns = np.nonzero(np.any(window_mask, axis=0))[0]
    columns_total = x2 - x1
    if len(columns) == 0:
        return {"columns_present": 0, "columns_total": columns_total, "column_coverage": 0.0, "largest_gap": columns_total}
    gaps = np.diff(columns) - 1
    return {
        "columns_present": int(len(columns)),
        "columns_total": int(columns_total),
        "column_coverage": float(len(columns) / columns_total),
        "largest_gap": int(gaps.max()) if len(gaps) else 0,
    }


def strip_masks(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """JSON-serializable copy of detected series without pixel masks."""
    cleaned: list[dict[str, Any]] = []
    for item in series:
        row = {key: value for key, value in item.items() if key != "mask"}
        cleaned.append(row)
    return cleaned
