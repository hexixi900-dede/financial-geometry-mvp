"""Phase 4 automatic geometry pipeline for single-axis single/multi-line charts.

Pipeline contract (no-leak): the parser receives only the masked chart image
and the question text. It never receives pseudo-Gold, the masked value-label
bbox, the value-label center, or any construction-only structure. The only
exception is the explicit `oracle_series_name` parameter, used solely by the
Oracle-Series *diagnostic* configuration (correct series identity is given,
but never the value or its location); the primary Auto-Series configuration
always runs with `oracle_series_name=None`.

Statuses: success, ambiguous_series, x_grounding_unrecoverable,
line_localization_failed, axis_mapping_failed.
"""
from __future__ import annotations

import math
import re
from typing import Any

import cv2
import numpy as np

from geometry_core import axis_localizer, interpolate_pixel_to_value, ocr_tokens
from phase3_line_core import (
    column_point,
    discover_x_axis_anchors,
    ground_question_to_x,
    recommended_window,
    robust_local_line_point,
)
from phase4_legend import detect_legend_entries, match_series_query, strip_query_boilerplate
from phase4_series import (
    color_distance,
    detect_line_series_components,
    local_continuity,
    match_series_by_color,
    strip_masks,
)


MAX_MATCH_COLOR_DISTANCE = 60.0
MIN_COLOR_MARGIN = 14.0
MAX_LOCAL_OVERLAP_FRACTION = 0.35


def _line_thickness(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0
    per_column = [np.count_nonzero(mask[:, x]) for x in np.unique(xs)]
    return float(np.median(per_column)) if per_column else 0.0


def _local_overlap(
    target_mask: np.ndarray,
    other_masks: list[np.ndarray],
    target_x: float,
    window: float,
) -> dict[str, Any]:
    """Check whether other series pixels crowd the target series near target_x."""
    height, width = target_mask.shape
    x1 = max(0, int(math.floor(target_x - window)))
    x2 = min(width, int(math.ceil(target_x + window)) + 1)
    dilated_target = cv2.dilate(target_mask, np.ones((5, 5), np.uint8)) > 0
    overlapping_columns = 0
    present_columns = 0
    crossing = False
    for other in other_masks:
        other_near = cv2.dilate(other, np.ones((3, 3), np.uint8)) > 0
        overlap = dilated_target & other_near
        overlap[:, :x1] = False
        overlap[:, x2:] = False
        columns = np.nonzero(np.any(overlap, axis=0))[0]
        overlapping_columns += int(len(columns))
        if len(columns) > 0:
            # A crossing means the other series passes through the target's
            # stroke inside the local window (overlap on both x sides).
            left_side = bool(np.any(columns < target_x - 1))
            right_side = bool(np.any(columns > target_x + 1))
            crossing = crossing or (left_side and right_side)
    target_columns = np.nonzero(np.any(target_mask[:, x1:x2], axis=0))[0]
    present_columns = int(len(target_columns))
    fraction = overlapping_columns / max(present_columns, 1)
    return {
        "local_overlap_columns": int(overlapping_columns),
        "target_columns_in_window": present_columns,
        "local_overlap_fraction": float(fraction),
        "line_crossing_near_target": bool(crossing),
    }


def _resolve_target_series(
    series_query: str | None,
    legend: dict[str, Any],
    series: list[dict[str, Any]],
    series_mode: str,
) -> dict[str, Any]:
    """Map a series query (or implicit single-line question) to a component."""
    if not series:
        return {"status": "line_localization_failed", "reason": "no_line_component_detected"}
    if not series_query:
        if len(series) == 1:
            return {
                "status": "ok",
                "series_mode": "single_line_implicit",
                "target_series_id": int(series[0]["series_id"]),
                "legend_match": None,
                "ambiguity": {"level": "none", "reasons": []},
            }
        return {
            "status": "ambiguous_series",
            "reason": "question_has_no_series_but_chart_has_multiple_lines",
            "series_mode": series_mode,
        }

    legend_entries = legend.get("entries", [])
    name_match = match_series_query(series_query, legend_entries)
    if name_match.get("status") != "ok":
        return {
            "status": "ambiguous_series",
            "reason": "series_name_not_matched_to_legend",
            "series_mode": series_mode,
            "name_match": name_match,
        }
    color_match = match_series_by_color(legend_entries, series)
    entry = next(
        (item for item in color_match["entry_matches"] if int(item["legend_index"]) == int(name_match["match"]["legend_index"])),
        None,
    )
    ambiguity_reasons: list[str] = []
    if entry is None or entry.get("matched_series_id") is None:
        return {
            "status": "ambiguous_series",
            "reason": "legend_entry_has_no_color_matched_line",
            "series_mode": series_mode,
            "name_match": name_match,
        }
    distance = float(entry["color_distance"])
    margin = entry.get("color_margin")
    if distance > MAX_MATCH_COLOR_DISTANCE:
        ambiguity_reasons.append("legend_swatch_color_far_from_best_line")
    if margin is not None and float(margin) < MIN_COLOR_MARGIN:
        ambiguity_reasons.append("another_line_has_similar_color")
    second_distance = entry.get("second_best_color_distance")
    matched_id = int(entry["matched_series_id"])
    diagnostics = {
        "series_mode": series_mode,
        "name_match": name_match,
        "color_distance_to_matched_line": distance,
        "second_line_color_distance": second_distance,
        "color_margin": margin,
        "ambiguity_reasons": ambiguity_reasons,
    }
    if "legend_swatch_color_far_from_best_line" in ambiguity_reasons:
        return {"status": "ambiguous_series", "reason": "legend_color_mismatch", **diagnostics}
    if "another_line_has_similar_color" in ambiguity_reasons:
        return {"status": "ambiguous_series", "reason": "similar_series_colors", **diagnostics}
    return {
        "status": "ok",
        "target_series_id": matched_id,
        "legend_match": entry,
        **diagnostics,
    }


def auto_multiline_pipeline(
    reader: Any,
    masked_bgr: np.ndarray,
    question: str,
    oracle_series_name: str | None = None,
) -> dict[str, Any]:
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
    height, width = masked_bgr.shape[:2]
    axis = axis_localizer(tokens, width, height)
    if axis is None:
        return {"status": "axis_mapping_failed", "reason": "axis_unrecoverable"}
    if axis.get("right_axis_detected") or int(axis.get("additional_y_axis_count", 0)) > 0:
        return {"status": "axis_mapping_failed", "reason": "multiple_y_axes_out_of_scope", "axis": axis}

    anchors, anchor_audit = discover_x_axis_anchors(reader, masked_bgr, tokens, axis)
    grounding = ground_question_to_x(question, anchors)
    series = detect_line_series_components(masked_bgr, axis)
    legend = detect_legend_entries(masked_bgr, tokens, axis)
    # The series query is whatever remains of the question after removing the
    # exact grounded x reference and the QA boilerplate; on a single-line
    # style question nothing remains and the implicit single series is used.
    series_question = question
    if grounding.get("status") == "ok":
        for reference in {grounding.get("matched_detected_label"), grounding.get("target_semantic_label")}:
            if reference:
                pattern = r"(?<![A-Za-z0-9])" + re.escape(str(reference)) + r"(?![A-Za-z0-9])"
                series_question = re.sub(pattern, " ", series_question, count=1, flags=re.IGNORECASE)
    series_query = (
        oracle_series_name if oracle_series_name is not None else (strip_query_boilerplate(series_question) or None)
    )
    series_mode = "oracle_series" if oracle_series_name is not None else "auto_series"
    resolution = _resolve_target_series(series_query, legend, series, series_mode)

    result: dict[str, Any] = {
        "status": "success",
        "axis": axis,
        "x_axis_anchors": anchors,
        "x_axis_anchor_audit": anchor_audit,
        "grounding": grounding,
        "legend": legend,
        "detected_series": strip_masks(series),
        "series_resolution": resolution,
        "masked_ocr_token_count": len(tokens),
    }
    if grounding.get("status") != "ok":
        result["status"] = "x_grounding_unrecoverable"
        result["status_reason"] = grounding.get("reason")
        return result
    if resolution.get("status") != "ok":
        result["status"] = resolution["status"]
        result["status_reason"] = resolution.get("reason")
        return result

    target_id = int(resolution["target_series_id"])
    target = next(item for item in series if int(item["series_id"]) == target_id)
    others = [item for item in series if int(item["series_id"]) != target_id]
    target_x = float(grounding["predicted_target_x"])
    window = recommended_window(target_x, anchors, width)
    continuity = local_continuity(target["mask"], target_x, window)
    overlap = _local_overlap(target["mask"], [item["mask"] for item in others], target_x, window)
    result["local_search_window"] = [target_x - window, target_x + window]
    result["series_continuity"] = continuity
    result["series_local_overlap"] = overlap
    result["target_line_thickness_pixels"] = _line_thickness(target["mask"])

    if continuity["column_coverage"] <= 0.0:
        result["status"] = "line_localization_failed"
        result["status_reason"] = "target_series_missing_near_target_x"
        return result
    if overlap["local_overlap_fraction"] > MAX_LOCAL_OVERLAP_FRACTION and len(others) > 0:
        result["status"] = "ambiguous_series"
        result["status_reason"] = "local_overlap_with_other_series"
        return result

    column_geometry = column_point(target["mask"], target_x, axis)
    local_geometry = robust_local_line_point(target["mask"], target_x, axis, window)
    result["auto_column_geometry"] = column_geometry
    result["auto_local_geometry"] = local_geometry
    result["auto_column_value"] = (
        None if column_geometry is None else interpolate_pixel_to_value(float(column_geometry["target_y"]), axis)
    )
    result["auto_local_value"] = (
        None if local_geometry is None else interpolate_pixel_to_value(float(local_geometry["target_y"]), axis)
    )
    if column_geometry is None and local_geometry is None:
        result["status"] = "line_localization_failed"
        result["status_reason"] = "no_recoverable_point_at_target_x"
    return result


def draw_phase4_overlay(
    original_bgr: np.ndarray,
    sample_id: str,
    method_name: str,
    pipeline: dict[str, Any],
    geometry: dict[str, Any] | None,
    raw_value: float | None,
    pseudo_gold: float,
) -> np.ndarray:
    """Audit overlay on the ORIGINAL chart with the full Phase 4 evidence set."""
    overlay = original_bgr.copy()
    height, width = overlay.shape[:2]
    axis = pipeline.get("axis") or {}
    for tick in axis.get("ticks", []):
        y = int(round(float(tick["pixel_y"])))
        cv2.circle(overlay, (int(round(float(tick["bbox"][2]))), y), 4, (255, 255, 0), -1)
        cv2.putText(overlay, f"{float(tick['value']):g}", (3, max(13, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 120, 0), 1, cv2.LINE_AA)
    for anchor in pipeline.get("x_axis_anchors", []):
        x1, y1, x2, y2 = map(lambda value: int(round(float(value))), anchor["bbox"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 210, 0), 1)
        cv2.circle(overlay, (int(round(float(anchor["center_x"]))), int(round(float(anchor["center_y"])))), 3, (255, 210, 0), -1)
    legend = pipeline.get("legend") or {}
    for entry in legend.get("entries", []):
        x1, y1, x2, y2 = map(lambda value: int(round(float(value))), entry["name_bbox"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 120, 220), 1)
        swatch = entry.get("swatch") or {}
        if swatch.get("bbox"):
            sx1, sy1, sx2, sy2 = map(int, swatch["bbox"])
            cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), (220, 60, 220), 1)
    resolution = pipeline.get("series_resolution") or {}
    target_id = resolution.get("target_series_id")
    for component in pipeline.get("detected_series", []):
        x, y, w, h = map(int, component["bbox"])
        is_target = target_id is not None and int(component["series_id"]) == int(target_id)
        color = (0, 0, 255) if is_target else (170, 170, 170)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2 if is_target else 1)
    grounding = pipeline.get("grounding") or {}
    target_x = grounding.get("predicted_target_x")
    if target_x is not None:
        x = int(round(float(target_x)))
        cv2.line(overlay, (x, 0), (x, height - 1), (255, 0, 255), 2)
    if geometry is not None:
        window = geometry.get("search_window")
        if window and axis.get("plot_bbox"):
            wx1, wx2 = map(lambda value: int(round(float(value))), window)
            cv2.rectangle(overlay, (wx1, int(axis["plot_bbox"][1])), (wx2, int(axis["plot_bbox"][3])), (0, 210, 255), 1)
        for segment in geometry.get("fitted_segments", []):
            sx1, sx2 = float(segment["x_min"]), float(segment["x_max"])
            sy1 = float(segment["target_y"]) + float(segment["slope"]) * (sx1 - float(geometry["target_x"]))
            sy2 = float(segment["target_y"]) + float(segment["slope"]) * (sx2 - float(geometry["target_x"]))
            cv2.line(overlay, (int(round(sx1)), int(round(sy1))), (int(round(sx2)), int(round(sy2))), (0, 200, 0), 2)
        px, py = map(lambda value: int(round(float(value))), geometry["point"])
        cv2.circle(overlay, (px, py), 7, (0, 0, 255), 2)
    axis_span = float(axis.get("axis_span", 0.0)) or 1e-12
    lines = [
        f"{sample_id} {method_name} status={pipeline.get('status')}",
        f"target={grounding.get('target_semantic_label','?')} mode={grounding.get('mode','?')} series_mode={resolution.get('series_mode','?')}",
        f"raw={raw_value:.6g} gold={pseudo_gold:.6g} abs={abs(raw_value-pseudo_gold):.6g} axis_norm={abs(raw_value-pseudo_gold)/axis_span:.4%}"
        if raw_value is not None
        else f"raw=NA gold={pseudo_gold:.6g} reason={pipeline.get('status_reason','')}",
    ]
    if resolution.get("color_distance_to_matched_line") is not None:
        lines.append(
            f"color_dist={float(resolution['color_distance_to_matched_line']):.1f} margin={resolution.get('color_margin')}"
        )
    panel_height = 21 * len(lines) + 7
    cv2.rectangle(overlay, (0, 0), (min(width - 1, 980), panel_height), (255, 255, 255), -1)
    for index, line in enumerate(lines):
        cv2.putText(overlay, line, (7, 18 + 21 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (15, 15, 15), 1, cv2.LINE_AA)
    return overlay
