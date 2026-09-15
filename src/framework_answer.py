"""Final chart answer and one optional plan repair, with per-target axis evidence."""
from __future__ import annotations

import json
from pathlib import Path

from framework_plan import PLAN_FIELDS, TARGET_FIELDS
from vlm_evidence_answerer import compact_value, parse_reply
from vlm_need_router import _chart_messages


def evidence(source, measurement):
    """Keep successful local reads in partial failures and preserve their axis identity."""
    original_plan = measurement.get("semantic_plan") or {}
    plan = {key: item for key, item in original_plan.items()
            if key in PLAN_FIELDS or key == "plan_valid"}
    targets = plan.get("targets") or []
    plan["targets"] = [{key: item for key, item in target.items() if key in TARGET_FIELDS}
                       for target in targets]
    audits = measurement.get("target_audit") or []
    indexed = {audit.get("target_index", i): audit for i, audit in enumerate(audits)}
    values = measurement.get("geometry_values") or []
    readings = []
    for index, target in enumerate(plan["targets"]):
        audit = indexed.get(index, {})
        axis = audit.get("axis") or {}
        ticks = axis.get("ticks") or audit.get("y_axis_ticks") or []
        value = None
        if audit.get("status") == "success":
            value = audit.get("auto_local_value", audit.get("value"))
            if value is None and len(values) == len(targets):
                value = values[index]
        readings.append({
            "target_index": index,
            "target": target,
            "axis_id": audit.get("axis_id", target.get("axis_id")),
            "mark_type": audit.get("mark_type", target.get("mark_type")),
            "status": audit.get("status", "not_measured"),
            "reason": audit.get("status_reason") or audit.get("error"),
            "measured_value": compact_value(value) if value is not None else None,
            "axis_tick_texts_and_values": [
                {key: tick[key] for key in ("text", "value", "pixel_y") if key in tick}
                for tick in ticks],
            "axis_label_or_unit": axis.get("unit") or axis.get("label"),
            "units": "numeric scale of THIS target's selected axis ticks",
            "provenance": ("completed local pixel measurement; check target and axis identity"
                           if value is not None else "no successful measured value"),
        })
    named = (measurement.get("reasoning") or {}).get("calculations") or []
    named = [{key: item for key, item in calculation.items()
              if key in {"name", "expression", "value", "status", "error"}} for calculation in named]
    has_named = bool(plan.get("calculations"))
    result = {
        "measurement_status": measurement.get("status"),
        "failure_reason": measurement.get("status_reason") or measurement.get("planner_error") or measurement.get("error"),
        "semantic_plan": plan,
        "measurements": readings,
        "operation": "evidence" if has_named else plan.get("operation"),
        "python_result": (measurement.get("numeric_answer")
                          if not has_named and plan.get("operation") != "evidence"
                          and measurement.get("status") == "success" else None),
        "calculations": named,
        "unit_source": "Original chart axes/title, question and supplied answer unit.",
    }
    if measurement.get("planner_error") or not plan.get("plan_valid"):
        result["planner_error"] = measurement.get("planner_error") or "missing_or_invalid_plan"
        result["planner_raw_text"] = measurement.get("planner_raw_text") or measurement.get("raw_text") or ""
    return result


def messages(source, measurement, overlay_path, min_pixels, max_pixels, allow_revision=True):
    kind = source["question_type"]
    answer_rules = {
        "single_choice": ('Return exactly one provided option letter.', "A"),
        "multiple_choice": ('Evaluate EVERY option and return all correct letters as a JSON array.', ["A", "C"]),
        "numerical": ('Return a JSON number in the requested answer unit, without prose in answer.', 1.23),
    }
    if kind not in answer_rules:
        raise ValueError("unsupported_question_type")
    rule, example = answer_rules[kind]
    prompt = """Answer using the original chart, question, options and valid measured evidence.
Check that each solid measured mark is the requested series, category/date and correct axis.
Orange boxes show proposed targets; they are not measured evidence. Missing values are not zero.
Each target's readings use that target's selected axis and its listed ticks. Never apply left-axis
calibration to a right-axis series. For a stack segment use its two-boundary height; a segment's
absolute top is not the segment magnitude. Verify legend identity on grouped, stacked and mixed charts.

Python results are supplied only when they were computed. Named calculations are the arithmetic
results when operation=evidence; do not invent a second conflicting top-level calculation.
difference is first minus second (NEW minus OLD for an increase); growth_rate takes OLD then NEW
and already multiplies by 100. Interpret units from the chart, supplied unit and options: MYR is
a currency code, not a million suffix. Do not multiply the same scale twice.
A period average requires every actual observation at its true frequency, such as twelve separate
monthly targets for a monthly year. Pixel trace samples and two endpoints are not discrete-period
observations. Sequence gaps are missing measurements, not recovered or interpolated evidence.

Some targets may fail even when others succeed. Use the valid partial reads and the original image.
If you need a visual estimate, identify it as a visual estimate in evidence_used; do not call it a
Geometry measurement or Python result. Always produce your best final answer after the available
repair, even if some measurements remain missing. Do not abstain or request a cached Raw answer.
"""
    if allow_revision:
        prompt += """
If the plan, axis binding or target location is wrong or incomplete and repair would help, return
{"status":"revise_plan","reason":"specific correction","plan":{...COMPLETE corrected plan...}}.
The program will perform ONE revised measurement pass. Do not supply an answer with a revision.
The complete plan fields are chart_context, chart_type (bar|line|bar_line|other), y_axis_count
(actual positive integer or "unknown"), targets, operation, calculations, reason.
Each target has x_label, series, mark_type (bar|line), axis_id (left|right), position
(label|first|last), measurement (value|height|sequence), and optional original-image normalized
region=[left,top,right,bottom] in 0..1000 with nonzero width and height.
A bar also requires bar_scope (whole|segment); a segment requires measurement=height.
A line must not have bar_scope. Different targets may use different marks and axes.
Use actual dates/categories. first/last is the endpoint of the WHOLE plotted series, not a
subinterval. Provide a focused region when OCR or legend matching failed. Enumerate all periods
for an average. sequence is a line interval and requires operation=evidence.
Operators: direct, difference, growth_rate, ratio, sum, mean, min, max, evidence.
direct needs one target; difference/growth_rate/ratio need two in the stated order.
calculations is [] or named expression objects, for example
[{"name":"increase","expression":{"op":"difference","args":[{"target":1},{"target":0}]}}].
Use only zero-based target references and supported nested operators, no literals or executable code.
With named calculations use operation=evidence. Do not include routing or direct-label flags.
A repair changes measurements, not whether this question entered Geometry.
If repair is unnecessary or unlikely to help, answer directly.
"""
    else:
        prompt += "\nThis is the final pass. No further plan revision is available. Return status=answered.\n"
    prompt += "\nQuestion type: " + kind + "\n" + rule
    prompt += "\nFor an answer return one JSON object; this answer is a format example only:\n"
    prompt += json.dumps({"status": "answered", "answer": example,
                          "evidence_used": "Brief explanation distinguishing measurements from visual estimates."})
    prompt += "\nActual Geometry evidence:\n" + json.dumps(evidence(source, measurement), ensure_ascii=False)
    result = _chart_messages(source, min_pixels, max_pixels,
        "Answer from the original chart and correctly attributed evidence. Return one JSON object.", prompt)
    content = result[1]["content"]
    content.insert(0, {"type": "text", "text": "Image 1: original chart used for the Raw comparison."})
    content[2:2] = [
        {"type": "text", "text": "Image 2: derived measurement overlay. Orange dashed boxes are proposed targets; solid numbered marks show actual program locations. Purple lines mark axis ticks; yellow boxes are OCR anchors. Compare positions and series with Image 1."},
        {"type": "image", "image": Path(overlay_path).resolve().as_uri(),
         "min_pixels": min_pixels, "max_pixels": max_pixels},
    ]
    return result
