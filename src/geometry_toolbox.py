"""Question-selected geometry tools composed from the existing CPU readers.

This module does not decide whether a question needs geometry. Unsupported
capabilities are reported to the caller, never converted to a Raw route.
The line/bar separation is a conservative rectangle/color mask, not a general
chart segmenter; same-color occlusion, hatch patterns and nonlinear axes remain
limitations. Height means unsigned physical extent, including on negative bars.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from geometry_core import (
    _fit_axis_side, _quantized_components, axis_localizer,
    interpolate_pixel_to_value, ocr_tokens,
)
from phase3_line_core import discover_x_axis_anchors, x_axis_band
from phase4_legend import detect_legend_entries
from phase8_geometry_adapter import (
    _bar_series_color, detect_series, pixel_region, read_bar_height, read_line, scalar,
)
from phase9_plan_contract import calculate, expression


LIMITATIONS = [
    "Linear Cartesian axes only; detected extra panels/axes are capability failures.",
    "Color-based bar/line separation cannot recover a line hidden by a same-color bar.",
    "Bar height is an unsigned extent, not a signed change.",
    "Line pixel sequences are not discrete observations for time averages.",
]


def _failure(reason, status="capability_failure", **audit):
    return dict(status=status, capability_status="unsupported_or_unrecovered",
                status_reason=reason, **audit)


def _complete_axis(axis, side, plot_bbox):
    result = {k: v for k, v in axis.items()
              if k not in {"right_axis", "right_axis_detected", "additional_y_axes",
                           "additional_y_axis_count"}}
    result.update(axis_id=side, side=side, plot_bbox=list(plot_bbox))
    limits = [interpolate_pixel_to_value(plot_bbox[i], result) for i in (1, 3)]
    result.update(axis_min_display=min(limits), axis_max_display=max(limits),
                  axis_span=abs(limits[1] - limits[0]),
                  value_units="axis_tick_units",
                  unit_note="Tick k/m/b/t suffixes are expanded by the existing OCR numeric parser.")
    return result


def merge_date_anchors(tokens, axis, width, height, anchors):
    """Retain temporal text already read by OCR, including full dates.

    The legacy discovery parser only retains year/month/quarter formats. Use the
    same temporal parser as target grounding and the existing x-axis band; take
    the first coherent label row, not dates in titles or footnotes. No extra OCR.
    """
    x1, y1, x2, y2, _ = x_axis_band(axis, width, height)
    axis_indices = set(axis.get("token_indices", []))
    candidates = []
    for token in tokens:
        parsed = scalar(token.text)
        if (token.index in axis_indices or token.confidence < .30 or parsed is None
                or parsed[1] != "temporal" or not x1 <= token.cx <= x2
                or not y1 <= token.cy <= y2):
            continue
        candidates.append(dict(label=token.text, center_x=token.cx, center_y=token.cy,
            bbox=[token.x1, token.y1, token.x2, token.y2], confidence=token.confidence,
            source="shared_temporal_parser_existing_ocr"))
    if len(candidates) < 2:
        return anchors
    candidates.sort(key=lambda a:a["center_y"])
    first_y = candidates[0]["center_y"]
    coherent = [a for a in candidates if abs(a["center_y"]-first_y) < max(8, .025*height)]
    coherent.sort(key=lambda a:a["center_x"])
    values = [scalar(a["label"])[0] for a in coherent]
    if len(coherent) < 2 or any(b <= a for a,b in zip(values,values[1:])):
        return anchors
    # Keep existing precise/rotated anchors. Add only previously discarded labels,
    # so an OCR box in this row cannot overwrite a better accepted location.
    merged = list(anchors)
    for anchor in coherent:
        value = scalar(anchor["label"])[0]
        if any(scalar(a["label"]) is not None and scalar(a["label"])[1]=="temporal"
               and abs(scalar(a["label"])[0]-value)<1e-6 for a in merged):
            continue
        merged.append(anchor)
    return sorted(merged,key=lambda a:float(a["center_x"]))


def prepare_chart(reader, image):
    """OCR and calibrate the original image once, without chart-kind admission gates."""
    if image is None or not hasattr(image, "shape") or image.ndim != 3:
        return _failure("invalid_image", axes={}, plot_bbox=None,
                        x_axis_anchors=[], legend={"entries": []}, ocr_tokens=[])
    height, width = image.shape[:2]
    raw = reader.readtext(image, detail=1, paragraph=False, min_size=7,
                          text_threshold=.45, low_text=.25, link_threshold=.35,
                          canvas_size=2560, mag_ratio=1.25)
    tokens = ocr_tokens(raw)
    chart = dict(image_size=[width, height], axes={}, plot_bbox=None,
                 x_axis_anchors=[], legend={"entries": []},
                 ocr_tokens=[t.as_dict() for t in tokens], limitations=LIMITATIONS)
    left = axis_localizer(tokens, width, height)
    # Keep the same fitted coordinate model used by the legacy readers.
    right = left.get("right_axis") if left else _fit_axis_side(tokens, width, height, "right")
    if left is None:
        # A right-only chart is explicitly reported rather than silently borrowing
        # a left plot extent or treating missing calibration as no need for geometry.
        return {**chart, **_failure("linear_left_axis_unrecovered_or_nonlinear",
                                  detected_right_axis=right)}
    known = set(left.get("token_indices", []))
    if right:
        known.update(right.get("token_indices", []))
    extra = [g for g in left.get("additional_y_axes", [])
             if len(set(g["token_indices"]) - known) >= 3]
    if extra:
        return {**chart, **_failure("multiple_panels_or_additional_axes",
                                  extra_axis_groups=extra, axis_detection=left)}
    # Both scales must describe the same vertical plotting region.
    if right:
        left_range = [min(t["pixel_y"] for t in left["ticks"]),
                      max(t["pixel_y"] for t in left["ticks"])]
        right_range = [min(t["pixel_y"] for t in right["ticks"]),
                       max(t["pixel_y"] for t in right["ticks"])]
        overlap = max(0, min(left_range[1], right_range[1]) -
                      max(left_range[0], right_range[0]))
        if overlap / max(1, min(np.ptp(left_range), np.ptp(right_range))) < .65:
            return {**chart, **_failure("left_right_axes_do_not_share_plot",
                                      axis_detection=left)}
    plot_bbox = list(left["plot_bbox"])
    if right:
        # Tick text starts outside the right plot border. Do not include it in
        # the mark detector as the historical 98.5%-width crop did.
        plot_bbox[2] = min(plot_bbox[2],
                           min(t["bbox"][0] for t in right["ticks"]) - .012 * width)
    if plot_bbox[2] - plot_bbox[0] < .15 * width:
        return {**chart, **_failure("invalid_shared_plot_bounds", plot_bbox=plot_bbox)}
    axes = {"left": _complete_axis(left, "left", plot_bbox)}
    if right:
        axes["right"] = _complete_axis(right, "right", plot_bbox)
    chart.update(axes=axes, plot_bbox=plot_bbox)
    # Exclude both scales' tick labels from legend/x-axis token candidates.
    parsing_axis = {**axes["left"], "token_indices": sorted(known)}
    try:
        anchors, anchor_audit = discover_x_axis_anchors(reader, image, tokens, parsing_axis)
    except (ValueError, IndexError, cv2.error) as exc:
        anchors, anchor_audit = [], {"status": "anchor_discovery_failed", "error": str(exc)}
    anchors = merge_date_anchors(tokens, parsing_axis, width, height, anchors)
    chart.update(x_axis_anchors=anchors, x_axis_audit=anchor_audit,
                 legend=detect_legend_entries(image, tokens, parsing_axis),
                 status="success", capability_status="supported")
    return chart


def _rectangles(image, axis):
    """Actual high-fill components inside the shared plot, without semantic labels."""
    height, width = image.shape[:2]
    left, top, right, bottom = axis["plot_bbox"]
    output = []
    seen = set()
    for comp in _quantized_components(image, axis):
        x, y, w, h = [int(comp[k]) for k in ("x", "y", "w", "h")]
        color = np.asarray(comp["median_bgr"], dtype=float)
        if color.min() >= 245:
            continue
        if not (5 <= w <= .17 * width and h >= 2 and comp["fill"] >= .65
                and x >= left - 3 and x+w <= right+3
                and y >= top-3 and y+h <= bottom+3):
            continue
        key = (x, y, w, h)
        if key not in seen:
            output.append(comp)
            seen.add(key)
    return output


def _whole_stack_region(image, axis, target):
    """Only bypass a series color when one whole multi-color stack is enclosed."""
    region = pixel_region(target, image)
    if region is None:
        return False, {"whole_stack_union": False}
    rx1, ry1, rx2, ry2 = region
    comps = []
    for comp in _rectangles(image, axis):
        x, y, w, h = [int(comp[k]) for k in ("x", "y", "w", "h")]
        # Include the full detected column, not only pieces overlapping the box.
        if rx1 <= x+w/2 <= rx2:
            comps.append(comp)
    columns = []
    for comp in sorted(comps, key=lambda c: c["x"] + c["w"]/2):
        cx = comp["x"] + comp["w"]/2
        for col in columns:
            ref = col[0]
            if abs(cx-ref["x"]-ref["w"]/2) <= max(3, min(comp["w"], ref["w"])*.25):
                col.append(comp)
                break
        else:
            columns.append([comp])
    if len(columns) != 1:
        return False, {"whole_stack_union": False, "region_column_count": len(columns)}
    col = sorted(columns[0], key=lambda c: c["y"])
    if len(col) < 2:
        return False, {"whole_stack_union": False}
    colors = [np.asarray(c["median_bgr"], dtype=float) for c in col]
    distinct = any(np.linalg.norm(a-b) > 50 for a in colors for b in colors)
    connected = all(b["y"] <= a["y"]+a["h"]+2 for a, b in zip(col, col[1:]))
    enclosed = (ry1 <= min(c["y"] for c in col)+2
                and ry2 >= max(c["y"]+c["h"] for c in col)-2)
    union = bool(distinct and connected and enclosed)
    return union, {"whole_stack_union": union, "region_column_count": 1,
                   "stack_component_count": len(col),
                   "complete_column_enclosed": bool(enclosed)}


def _bar_scope_issue(image, axis, target, result, legend):
    """Do not label a measured subset as a whole, or a multi-color union as a segment."""
    if result.get("status") != "success":
        return None
    x, y, w, h = result["bbox"]
    same_column = []
    for comp in _rectangles(image, axis):
        if abs(comp["x"]+comp["w"]/2-x-w/2) <= max(3, min(w, comp["w"])*.25):
            same_column.append(comp)
    if target["bar_scope"] == "whole":
        # A color-selected segment touching another piece is not a whole column.
        # Preserve the measured endpoints for visual repair, but do not use its
        # value until an explicit region encloses the complete column.
        if any((c["y"] < y-2 and c["y"]+c["h"] >= y-2)
               or (c["y"]+c["h"] > y+h+2 and c["y"] <= y+h+2)
               for c in same_column):
            return "whole_stack_requires_enclosing_region"
    elif _bar_series_color(target, legend) is None:
        colors = [np.asarray(c["median_bgr"], dtype=float) for c in same_column
                  if c["y"] >= y-2 and c["y"]+c["h"] <= y+h+2]
        if any(np.linalg.norm(a-b) > 50 for a in colors for b in colors):
            return "segment_region_contains_multiple_colors"
    return None


def _line_series(image, axis):
    """Remove rectangle-colored pixels before the reused line detector.

    Different-colored line pixels crossing a bar are preserved. Same-colored
    overlaps become gaps; a missing target is reported instead of synthesized.
    """
    clean = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    height, width = image.shape[:2]
    rectangles = []
    for comp in _rectangles(image, axis):
        x, y, w, h = [int(comp[k]) for k in ("x", "y", "w", "h")]
        if w < max(7, .008*width) or h < max(12, .04*height) or h < .45*w:
            continue
        crop = image[y:y+h, x:x+w].astype(float)
        color_pixels = np.linalg.norm(crop-np.asarray(comp["median_bgr"]), axis=2) < 45
        mask[y:y+h, x:x+w][color_pixels] = 1
        rectangles.append([x, y, w, h])
    clean[mask > 0] = 255
    detected = detect_series(clean, axis)
    plot_h = axis["plot_bbox"][3] - axis["plot_bbox"][1]
    rejected = [s for s in detected if s.get("fill_ratio", 0) >= .70
                and s["bbox"][3] >= max(12, .10*plot_h)
                and s["bbox"][2] <= .30*width]
    rejected_ids = {s["series_id"] for s in rejected}
    series = [s for s in detected if s["series_id"] not in rejected_ids]
    for i, s in enumerate(series):
        s["series_id"] = i
    return series, {"bar_rectangles_masked": rectangles,
                    "bar_pixels_masked": int(mask.sum()),
                    "filled_series_rejected": len(rejected),
                    "line_series_retained": len(series),
                    "separation_method": "high_fill_rectangle_color_mask",
                    "limitation": "Same-color bar/line overlap may remain unrecoverable."}


def _target_problem(target, axes):
    if target.get("mark_type") not in {"bar", "line"}:
        return "unsupported_mark_type"
    if target.get("axis_id") not in {"left", "right"}:
        return "missing_or_invalid_axis_id"
    if target["axis_id"] not in axes:
        return "requested_axis_unrecovered"
    if target.get("measurement", "value") not in {"value", "height", "sequence"}:
        return "unsupported_measurement"
    if target["mark_type"] == "line" and target.get("measurement") == "height":
        return "line_height_unsupported"
    if target["mark_type"] == "bar":
        if target.get("bar_scope") not in {"whole", "segment"}:
            return "missing_or_invalid_bar_scope"
        if target.get("measurement") == "sequence":
            return "bar_sequence_unsupported"
        if target["bar_scope"] == "segment" and not (target.get("series") or target.get("region")):
            return "segment_requires_series_or_region"
    return None


def measure_targets(reader, image, chart, plan, cache=None):
    """Run real mark readers on a prepared chart; useful for isolated tool tests."""
    base = dict(semantic_plan=plan, status="measurement_failed", capability_status="supported",
                geometry_values=[], numeric_answer=None, reasoning=None,
                target_audit=[], pipeline_audit=[], chart_audit=chart,
                limitations=LIMITATIONS)
    if chart.get("status") != "success":
        return {**base, **_failure(chart.get("status_reason", "chart_preparation_failed"))}
    targets = plan.get("targets", [])
    if not targets:
        return {**base, **_failure("missing_targets", status="invalid_plan")}
    cache = cache if cache is not None else {}
    values = []
    all_success = True
    for i, original_target in enumerate(targets):
        target = dict(original_target)
        target.setdefault("x_label", "")
        target.setdefault("series", "")
        target.setdefault("position", "label")
        target.setdefault("measurement", "value")
        axis_id = target.get("axis_id")
        axis = chart["axes"].get(axis_id)
        audit = dict(target_index=i, semantic_target=original_target,
                     mark_type=target.get("mark_type"), axis_id=axis_id,
                     axis=axis, y_axis_ticks=axis.get("ticks", []) if axis else [])
        problem = _target_problem(target, chart["axes"])
        if problem:
            result = _failure(problem, status="target_capability_failure")
            pipe = dict(result)
            base["capability_status"] = "partial_or_unsupported"
        else:
            legacy_chart = {**chart, "axis": axis}
            try:
                if target["mark_type"] == "bar":
                    scope_audit = {}
                    if target["bar_scope"] == "whole":
                        union, scope_audit = _whole_stack_region(image, axis, target)
                        if union:
                            target["series"] = ""
                    color = _bar_series_color(target, chart.get("legend"))
                    if (target["bar_scope"] == "segment" and color is None
                            and not target.get("region")):
                        result = _failure("segment_series_not_matched_and_no_region",
                                          status="target_capability_failure")
                    else:
                        result = read_bar_height(reader, image, legacy_chart, target)
                        scope_issue = _bar_scope_issue(image, axis, target, result, chart.get("legend"))
                        if scope_issue:
                            result.update(status="bar_scope_unrecoverable", status_reason=scope_issue)
                    result.update(scope_audit)
                    result["semantic_target"] = original_target
                    if target["measurement"] == "height":
                        result["measurement_semantics"] = "unsigned_extent"
                    value = result.get("value")
                    pipe = dict(result)
                    if result.get("point"):
                        pipe["auto_local_geometry"] = {
                            "point": result["point"], "bottom_point": result.get("bottom_point")}
                    pipe["auto_local_value"] = value
                else:
                    # Pixel series detection is shared between left/right targets;
                    # only the selected value-axis calibration changes.
                    if "line_series" not in cache:
                        cache["line_series"], cache["line_separation"] = _line_series(image, axis)
                    result = read_line(image, legacy_chart, target, cache["line_series"])
                    result["line_separation"] = cache["line_separation"]
                    pipe = dict(result)
                    value = result.get("auto_local_value")
                    if result.get("auto_local_geometry"):
                        result["geometry"] = result["auto_local_geometry"]
                if result.get("status") == "success" and value is None:
                    result.update(status="measurement_value_missing")
            except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError, cv2.error) as exc:
                result = dict(status="target_measurement_failed", status_reason=str(exc))
                pipe = dict(result)
        audit.update(result)
        audit.update(target_index=i, axis_id=axis_id, mark_type=target.get("mark_type"),
                     axis=axis, y_axis_ticks=axis.get("ticks", []) if axis else [])
        pipe.update(target_index=i, axis_id=axis_id, mark_type=target.get("mark_type"),
                    axis=axis, x_axis_anchors=chart.get("x_axis_anchors", []))
        base["target_audit"].append(audit)
        base["pipeline_audit"].append(pipe)
        success = audit.get("status") == "success"
        all_success = all_success and success
        values.append((audit.get("value") if target.get("mark_type") == "bar"
                       else audit.get("auto_local_value")) if success else None)
    base["geometry_values"] = values
    base["measurement_success_count"] = sum(a["status"] == "success" for a in base["target_audit"])
    base["measurement_target_count"] = len(targets)
    base["y_axis_ticks_by_axis"] = {key: axis["ticks"] for key, axis in chart["axes"].items()}
    if not all_success:
        base["status_reason"] = "one_or_more_targets_unrecovered"
        return base
    try:
        named = plan.get("calculations") or []
        if named:
            calculations = [
                dict(name=c.get("name", "calculation"), expression=c["expression"],
                     value=expression(c["expression"], values)) for c in named]
            numeric = None
        else:
            calculations = []
            numeric = calculate(plan.get("operation", "evidence"), values)
        base.update(status="success", capability_status="supported", numeric_answer=numeric,
                    reasoning=dict(executor="python",
                                   formula="named_expressions" if named else plan.get("operation", "evidence"),
                                   numeric_answer=numeric, calculations=calculations,
                                   value_units="per_target_axis_tick_units",
                                   answer_status="awaiting_evidence_vlm"))
    except (KeyError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        base.update(status="arithmetic_failed", status_reason=str(exc))
    return base


def measure_plan(reader, source, plan, cache=None):
    """Measure source.image_path on the original image; never consult old chart caches."""
    path = Path(source["image_path"]).resolve()
    base = dict(sample_id=str(source.get("sample_id", "")),
                chart_id=source.get("chart_id"), semantic_plan=plan,
                geometry_values=[], numeric_answer=None, target_audit=[], pipeline_audit=[])
    try:
        stat = path.stat()
        cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return {**base, **_failure("original_image_read_failed", status="image_read_failed")}
    except (OSError, cv2.error) as exc:
        return {**base, **_failure(str(exc), status="image_read_failed")}
    cache = cache if cache is not None else {}
    charts = cache.setdefault("charts", {})
    series = cache.setdefault("series", {})
    if cache_key not in charts:
        # Bound resident masks when a worker processes many unrelated charts.
        while len(charts) >= 8:
            expired = next(iter(charts))
            charts.pop(expired, None)
            series.pop(expired, None)
        try:
            charts[cache_key] = prepare_chart(reader, image)
        except (KeyError, TypeError, ValueError, IndexError, cv2.error) as exc:
            charts[cache_key] = _failure("chart_preparation_failed", error=str(exc))
    result = measure_targets(reader, image, charts[cache_key], plan,
                             series.setdefault(cache_key, {}))
    return {**base, **result, "geometry_image_path": str(path),
            "geometry_image_size": [image.shape[1], image.shape[0]],
            "geometry_input_source": "original_image_no_legacy_cache"}
