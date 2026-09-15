"""Target planning contract independent of the question-level Geometry decision."""
from __future__ import annotations

import json
import math
from typing import Any

from phase9_plan_contract import OPERATIONS, PLACEHOLDER_LABELS, calculate, expression_error
from vlm_need_router import _chart_messages
from vlm_semantic_planner import extract_json

PLAN_FIELDS = {"chart_context", "chart_type", "y_axis_count", "targets", "operation",
               "calculations", "reason"}
TARGET_FIELDS = {"x_label", "series", "position", "measurement", "region",
                 "mark_type", "axis_id", "bar_scope"}


def messages(row: dict[str, Any], min_pixels: int, max_pixels: int) -> list[dict[str, Any]]:
    return _chart_messages(row, min_pixels, max_pixels,
        'Produce a measurement PLAN, not data values or an answer. Output one JSON object. '
        'measurement must be the string "value", "height" or "sequence", NEVER a number.',
        """The question has already been selected for Geometry. Specify what to measure.
Use this COMPLETE JSON example as the output shape, replacing its example dates,
series, mark type and operation with those needed by the ACTUAL question:
{
  "chart_context":"Relevant visible axis units and series identities",
  "chart_type":"line",
  "y_axis_count":1,
  "targets":[
    {"x_label":"Jan-22","series":"Revenue","mark_type":"line","axis_id":"left",
     "position":"label","measurement":"value"},
    {"x_label":"Jan-23","series":"Revenue","mark_type":"line","axis_id":"left",
     "position":"label","measurement":"value"}
  ],
  "operation":"growth_rate",
  "calculations":[],
  "reason":"Recover old and new values; Python computes percentage change"
}

Target rules:
1. x_label is the REAL requested date/category; never output the placeholder "Date".
   series is the actual series name, optionally its visible color. Do not estimate
   data values. The example years/series above are not targets for the actual image.
2. mark_type is exactly "bar" or "line". axis_id is exactly "left" or "right",
   never "x" or "y". Bind every target to its matching value axis and unit.
   A bar-plus-line question can mix mark types and axes in one targets array.
3. measurement is "value" for a line point or absolute bar endpoint, "height" for
   a floating bar/stack segment's two-boundary extent, or "sequence" for a line
   interval trend/extrema. NEVER put an estimated number in measurement.
4. Each BAR target also needs "bar_scope":"whole" for a full column/floating bar,
   or "bar_scope":"segment" with measurement="height" for one stack component.
   A LINE target has no bar_scope. Use actual legend identity for grouped bars.
5. Optional region=[left,top,right,bottom] is a focused box around the target in
   0..1000 original-image coordinates. Provide it if it clarifies the mark/segment.
   For a whole stack, enclose every piece; for a segment, only the requested piece.
6. position is "label", "first" or "last". first/last means the WHOLE series endpoint,
   not the start/end of one year within a longer chart. Keep the actual date.

Arithmetic rules:
- direct: one target. difference: first minus second, so NEW then OLD for increase.
- growth_rate: OLD then NEW; Python computes (new-old)/old*100. ratio: first/second.
- sum/mean/min/max: list all required scalar observations. A monthly year's mean
  needs twelve separate real months, not endpoints or a mean of pixel samples.
- evidence: several independent readings, option comparisons or line sequences.
  Cover every option. A plotted rate/average can be read directly without recomputing.
- Usually use calculations=[]. For multi-step arithmetic only, use named expressions:
  [{"name":"increase","expression":{"op":"difference","args":[{"target":1},{"target":0}]}}].
  Set operation="evidence" when calculations is nonempty. References are zero-based;
  args can be nested supported expressions. No number literals or executable code.
chart_type is bar|line|bar_line|other; y_axis_count is the actual count or "unknown".
Do not output needs_geometry, route_geometry, a final answer, or estimated target values.""")


def _text(value: Any, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError("invalid_" + field)
    return value.strip()


def _sequence_expression_error(expr: dict[str, Any], targets: list[dict[str, Any]]) -> None:
    if "target" in expr:
        return
    for arg in expr["args"]:
        if "target" in arg and targets[arg["target"]]["measurement"] == "sequence":
            if expr["op"] not in {"min", "max"}:
                raise ValueError("sequence_requires_extremum_calculation")
        _sequence_expression_error(arg, targets)


def normalize_plan(value: dict[str, Any], question: str = "") -> dict[str, Any]:
    """Validate executable parameters, never re-decide whether Geometry is needed.

    Returns plan_valid=True on success; malformed or incomplete plans raise ValueError
    for the orchestrator's plan-repair path. Descriptive chart type and axis count
    are not capability filters. The question argument preserves the caller interface;
    correctness labels and answers are never inputs to contract validation.
    """
    if not isinstance(value, dict):
        raise ValueError("plan_not_object")
    if set(value) - (PLAN_FIELDS | {"plan_valid"}):
        raise ValueError("unexpected_plan_keys")
    result = dict(value)
    # Recover the known accidental wrapper without accepting new executable fields.
    context = result.get("chart_context")
    if isinstance(context, dict):
        nested_keys = (PLAN_FIELDS - {"chart_context"}) & set(context)
        for key in nested_keys:
            if key in result and json.dumps(result[key], sort_keys=True) != json.dumps(context[key], sort_keys=True):
                raise ValueError("conflicting_nested_plan_field:" + key)
            result.setdefault(key, context[key])
        remaining = {key: item for key, item in context.items() if key not in nested_keys}
        result["chart_context"] = json.dumps(remaining, ensure_ascii=False, sort_keys=True) if remaining else ""
    for key, default in {"chart_context":"", "reason":"", "calculations":[], "chart_type":"other", "y_axis_count":"unknown"}.items():
        result.setdefault(key, default)
    if PLAN_FIELDS - set(result):
        raise ValueError("missing_plan_fields:" + ",".join(sorted(PLAN_FIELDS - set(result))))
    if "plan_valid" in result and type(result["plan_valid"]) is not bool:
        raise ValueError("invalid_plan_valid")
    chart_context = _text(result["chart_context"], "chart_context", allow_empty=True)
    chart_type = _text(result["chart_type"], "chart_type").lower()
    if chart_type not in {"bar", "line", "bar_line", "other"}:
        raise ValueError("invalid_chart_type")
    axis_count = result["y_axis_count"]
    if not ((type(axis_count) is int and axis_count > 0) or axis_count == "unknown"):
        raise ValueError("invalid_y_axis_count")
    operation = _text(result["operation"], "operation").lower()
    if operation not in OPERATIONS:
        raise ValueError("unsupported_operation")
    raw_targets = result["targets"]
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("missing_measurements")
    targets = []
    for raw in raw_targets:
        if not isinstance(raw, dict) or set(raw) - TARGET_FIELDS:
            raise ValueError("unexpected_target_keys")
        if not {"x_label", "series", "mark_type", "axis_id"} <= set(raw):
            raise ValueError("missing_target_fields")
        target = dict(raw)
        target["x_label"] = _text(raw["x_label"], "x_label", allow_empty=True)
        target["series"] = _text(raw["series"], "series", allow_empty=True)
        if target["x_label"].lower().strip("<>") in PLACEHOLDER_LABELS:
            raise ValueError("placeholder_target_label")
        target.setdefault("position", "label")
        target.setdefault("measurement", "value")
        for field in ("mark_type", "axis_id", "position", "measurement"):
            target[field] = _text(target[field], field)
        if "bar_scope" in target:
            target["bar_scope"] = _text(target["bar_scope"], "bar_scope")
        if target["mark_type"] not in {"bar", "line"}:
            raise ValueError("unsupported_mark_type")
        if target["axis_id"] not in {"left", "right"}:
            raise ValueError("missing_or_invalid_axis_binding")
        if target["position"] not in {"label", "first", "last"} or target["measurement"] not in {"value", "height", "sequence"}:
            raise ValueError("unsupported_target_locator")
        region = target.get("region")
        if "region" in target:
            if not isinstance(region, list) or len(region) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1000 for v in region):
                raise ValueError("invalid_visual_region")
            if region[0] >= region[2] or region[1] >= region[3]:
                raise ValueError("invalid_visual_region")
        if not target["x_label"] and region is None and target["position"] == "label":
            raise ValueError("missing_target_location")
        if target["mark_type"] == "bar":
            if target.get("bar_scope") not in {"whole", "segment"}:
                raise ValueError("missing_or_invalid_bar_scope")
            if target["measurement"] == "sequence":
                raise ValueError("bar_sequence_unsupported")
            if target["bar_scope"] == "segment" and target["measurement"] != "height":
                raise ValueError("stack_segment_requires_height")
        elif "bar_scope" in target or target["measurement"] == "height":
            raise ValueError("invalid_line_measurement")
        targets.append(target)
    raw_calculations = result["calculations"]
    if not isinstance(raw_calculations, list):
        raise ValueError("invalid_calculations")
    calculations = []
    for raw in raw_calculations:
        if not isinstance(raw, dict) or set(raw) != {"name", "expression"}:
            raise ValueError("invalid_calculation_fields")
        name = _text(raw["name"], "calculation_name")
        try:
            error = expression_error(raw["expression"], len(targets))
        except (TypeError, RecursionError) as exc:
            raise ValueError("invalid_calculation") from exc
        if error:
            raise ValueError(error)
        _sequence_expression_error(raw["expression"], targets)
        if "target" in raw["expression"] and targets[raw["expression"]["target"]]["measurement"] == "sequence":
            raise ValueError("sequence_requires_extremum_calculation")
        calculations.append({"name": name, "expression": raw["expression"]})
    if calculations:
        operation = "evidence"
    if operation == "direct" and len(targets) != 1 or operation in {"difference", "growth_rate", "ratio"} and len(targets) != 2:
        raise ValueError("operation_target_count_mismatch")
    if any(t["measurement"] == "sequence" for t in targets) and operation != "evidence":
        raise ValueError("sequence_requires_evidence")
    return {"chart_context": chart_context, "chart_type": chart_type,
            "y_axis_count": axis_count, "targets": targets, "operation": operation,
            "calculations": calculations, "reason": _text(result["reason"], "reason", allow_empty=True),
            "plan_valid": True}
