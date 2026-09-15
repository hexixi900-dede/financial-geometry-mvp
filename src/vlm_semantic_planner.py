#!/usr/bin/env python3
"""Qwen-VL semantic planner that is forbidden to return chart values."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


from phase9_plan_contract import OPERATIONS
ALLOWED_OPERATIONS = OPERATIONS | {"unsupported"}
ALLOWED_CHART_TYPES = {"bar", "line", "other"}


# Chart metadata and chart-type-conditioned examples are inspired by ChartAgent
# (Kaur et al., https://arxiv.org/html/2510.04514v3, Sections 3.3 and N).
# These are our synthetic examples, not official ChartAgent code or FinMME items.
SYNTHETIC_PLANNING_EXAMPLES = [
    {
        "synthetic_chart": "A black Closing Price line and green Forecast markers share one USD y axis. The x axis includes 30-Dec-2022; the title says As of 17-Jun-2023, matching the end of the black line. No values are printed beside the marks.",
        "question": "Which statements describe Closing Price from 30-Dec-2022 to 17-Jun-2023?",
        "options": "A: The increase exceeds USD 2. B: The percentage growth exceeds 5%. C: The ending price is below the starting price.",
        "plan": {
            "chart_context": "The black line is Closing Price; green markers are Forecast. The left axis is USD. The dated title identifies the final Closing Price observation as 17-Jun-2023.",
            "chart_type": "line", "y_axis_count": 1, "has_direct_value_labels": False,
            "operation": "evidence",
            "targets": [
                {"x_label": "30-Dec-2022", "series": "Closing Price", "position": "label", "measurement": "value"},
                {"x_label": "17-Jun-2023", "series": "Closing Price", "position": "last", "measurement": "value"},
            ],
            "calculations": [
                {"name": "increase: new minus old", "expression": {"op": "difference", "args": [{"target": 1}, {"target": 0}]}},
                {"name": "percentage growth: old then new", "expression": {"op": "growth_rate", "args": [{"target": 0}, {"target": 1}]}},
            ],
            "reason": "Read the same Closing Price series at 30-Dec-2022 and 17-Jun-2023. Both values and the two calculations cover all three options; do not select options here.",
        },
    },
    {
        "synthetic_chart": "A monthly Revenue line covers Jan-2020 through Dec-2022 on one y axis labelled USD million. Month labels are visible and there are no values printed beside points.",
        "question": "What is the mean monthly revenue during 2021?",
        "options": "",
        "plan": {
            "chart_context": "Revenue is a monthly line in USD million. The chart spans Jan-2020 to Dec-2022, while the requested interval is Jan-2021 to Dec-2021.",
            "chart_type": "line", "y_axis_count": 1, "has_direct_value_labels": False,
            "operation": "mean",
            "targets": [{"x_label": month + "-2021", "series": "Revenue", "position": "label", "measurement": "value"}
                        for month in ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")],
            "calculations": [],
            "reason": "Measure all twelve monthly observations in 2021. Jan-2021 and Dec-2021 are inside the chart, so neither is the first or last observation of the whole series.",
        },
    },
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gpu_is_idle(max_memory_mib: int = 1024, max_utilization: int = 5) -> bool:
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    state = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if processes.returncode or state.returncode:
        return False
    if any(line.strip() for line in processes.stdout.splitlines()):
        return False
    rows = [line.split(",") for line in state.stdout.splitlines() if line.strip()]
    return bool(rows) and all(int(memory.strip()) <= max_memory_mib and int(util.strip()) <= max_utilization for memory, util in rows)


def messages(row: dict[str, Any], min_pixels: int, max_pixels: int) -> list[dict[str, Any]]:
    examples = json.dumps(SYNTHETIC_PLANNING_EXAMPLES, ensure_ascii=False)
    prompt = f"""Inspect the attached chart, the question, and EVERY answer option. Your job is to identify
what the measuring program should read, without guessing the chart's data values or answering the question.

First describe the visible chart in chart_context: relevant axis labels/units, legend entries,
time range, and which visual mark represents the requested quantity. Keep this brief. Dates,
axis ticks and option thresholds may be mentioned; do not estimate target values in any field.
Use the original image to distinguish similarly named series, points, segments and whole columns.

Return ONE JSON object with chart_context, chart_type, y_axis_count, has_direct_value_labels,
operation, targets, calculations, and reason. Use chart_type bar, line or other. Use the actual
number of y axes, or "unknown" if unclear. has_direct_value_labels refers to the REQUESTED
values printed beside marks, not axis ticks. Set it true when those values can be read directly.

Each target has x_label and series, with optional position, measurement and region:
- x_label is a real date/category or a specific interval with its actual start and end.
  Never put a schema placeholder such as date, month, year, start_date or end_date here.
  Identify the requested series by its actual legend/name and, if useful, color or line style.
- position defaults to label. The program can match or interpolate real date/category labels
  using OCR; an approximate region is OPTIONAL. Do not invent a bounding box for a clear date.
- first/last means the beginning/end of the WHOLE selected plotted series. Use it only when
  the question asks for that endpoint, or a dated title clearly identifies it. Still retain
  the real date when known. The first/last month WITHIN a requested year is usually not the
  first/last point of a chart spanning several years: use the explicit date and position=label.
- measurement defaults to value: one line point or bar endpoint. height measures the two ends
  of a floating bar or a stacked segment. Distinguish a named segment from the entire column.
- sequence is for a line over a stated interval: trends, interval extrema, or always-above
  claims. Use operation=evidence and cover the whole required interval for every relevant
  series, not just an apparent winning point. Supply a region spanning that interval so the
  program can delimit the trace. Pixel samples are not monthly observations for taking means.
- If region helps disambiguate a mark, supply [left,top,right,bottom] normalized to 0..1000 in
  the ORIGINAL IMAGE. Enclose the plotted mark/segment rather than its text label. Even a point
  needs a small nonzero box; left < right and top < bottom. Geometry refines these coordinates.

Choose the measurements before deciding the arithmetic. Keep target indices stable from zero:
- direct: one target. difference: first MINUS second. For an increase, this is NEW minus OLD.
- growth_rate: OLD first, NEW second; Python computes (new-old)/old * 100.
- ratio: first divided by second. sum/mean/min/max: all relevant scalar target observations.
  A full-year monthly mean needs all twelve months of that year, not one point or two endpoints.
  Use the actual observation frequency (e.g. four quarters for a quarterly mean). If you cannot
  cover the requested periods, say so and use unsupported rather than a partial-year estimate.
- Reading a series that already plots a rate or average is direct. Do not compute growth again.
- evidence: multiple independent comparisons, sequences, or named multi-step calculations.
  calculations is a list of name/expression objects. An expression is a target reference such
  as {{"target":0}}, or {{"op":"difference","args":[{{"target":1}},{{"target":0}}]}}.
  Operators may be nested; use only target references, no guessed values or executable code.
- For MULTIPLE CHOICE, cover EVERY option's evidence and share repeated targets. Do not select
  options here and do not stop after measuring evidence for the first plausible answer.

The following TWO COMPLETE EXAMPLES describe invented charts. They illustrate the contract
and arithmetic roles only. Do not copy their series or dates into the actual chart's plan:
{examples}

Now return only the plan for the ATTACHED ACTUAL CHART. The synthetic charts above are unrelated.
Caption: {row.get('caption', '')}
Question type: {row.get('question_type', '')}
Requested answer unit: {row.get('unit', '')}
Question: {row['question']}
Options: {row.get('options', '')}"""
    return [
        {"role": "system", "content": "Plan semantic chart targets only. Never answer the chart question."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Path(row.get("vlm_image_path") or row["image_path"]).resolve().as_uri(),
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]


def extract_json(text: str) -> dict[str, Any]:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no_json_object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("json_not_object")
    return value


def normalize_plan(value: dict[str, Any], question: str) -> dict[str, Any]:
    allowed_keys = {
        "chart_type",
        "y_axis_count",
        "has_direct_value_labels",
        "operation",
        "targets",
        "reason",
        "calculations",
        "chart_context",
        # A correction may echo the normalized plan shown in the overlay prompt.
        # Accept these derived fields but recompute them below rather than trust them.
        "route_geometry",
        "route_reason",
    }
    if set(value) - allowed_keys:
        raise ValueError("unexpected_plan_keys")
    context = value.get("chart_context")
    if isinstance(context, dict):
        # Some model replies place the requested plan inside chart_context.
        # Lift only contract fields; descriptive metadata stays in the context.
        # Derived routing fields remain untrusted and are recomputed below.
        nested_keys = (allowed_keys - {"chart_context", "route_geometry", "route_reason"}) & set(context)
        if nested_keys:
            value = dict(value)
            for key in sorted(nested_keys):
                if key in value and json.dumps(value[key], sort_keys=True) != json.dumps(context[key], sort_keys=True):
                    raise ValueError(f"conflicting_nested_plan_field:{key}")
                value.setdefault(key, context[key])
            value["chart_context"] = {
                key: item for key, item in context.items() if key not in nested_keys
            }
    chart_type = str(value.get("chart_type", "other")).strip().lower()
    operation = str(value.get("operation", "unsupported")).strip().lower()
    if chart_type not in ALLOWED_CHART_TYPES:
        chart_type = "other"
    if operation not in ALLOWED_OPERATIONS:
        operation = "unsupported"
    axis_raw = value.get("y_axis_count", "unknown")
    try:
        y_axis_count: int | str = int(axis_raw)
    except (TypeError, ValueError):
        y_axis_count = "unknown"
    direct_labels = value.get("has_direct_value_labels")
    if not isinstance(direct_labels, bool):
        direct_labels = None
    targets: list[dict[str, Any]] = []
    raw_targets = value.get("targets", [])
    if not isinstance(raw_targets, list):
        raise ValueError("targets_not_array")
    for target in raw_targets:
        if not isinstance(target, dict):
            raise ValueError("target_not_object")
        if set(target) - {"x_label", "series", "position", "measurement", "region"}:
            raise ValueError("unexpected_target_keys")
        x_label = str(target.get("x_label") or "").strip()
        series = str(target.get("series") or "").strip()
        if not (x_label or target.get("region") or target.get("position") in {"first", "last"}):
            raise ValueError("target_missing_location; indices must be preserved")
        if x_label or target.get("region") or target.get("position") in {"first", "last"}:
            position = target.get("position", "label")
            measurement = target.get("measurement", "value")
            if position not in {"label", "first", "last"} or measurement not in {"value", "height", "sequence"}:
                raise ValueError("unsupported_target_locator")
            targets.append({"x_label": x_label, "series": series, "position": position, "measurement": measurement, **({"region": target["region"]} if "region" in target else {})})
    reason = str(value.get("reason", "")).strip()
    plan = {
        "chart_context": str(value.get("chart_context", "")).strip(),
        "chart_type": chart_type,
        "y_axis_count": y_axis_count,
        "has_direct_value_labels": direct_labels,
        "operation": operation,
        "targets": targets,
        "reason": reason,
        "route_geometry": False,
        "calculations": value.get("calculations", []),
    }
    from phase9_plan_contract import validate_plan
    plan, route_reason = validate_plan(plan, question)
    plan["route_geometry"] = route_reason == "ok"
    plan["route_reason"] = route_reason
    return plan


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["sample_id"]): row for row in read_jsonl(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=451584)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = read_jsonl(args.inputs)
    if args.limit is not None:
        rows = rows[: args.limit]
    completed = load_completed(args.output)
    if all(str(row["sample_id"]) in completed for row in rows):
        print("Semantic plans are complete; no GPU work needed", flush=True)
        return
    consecutive_idle = 0
    while consecutive_idle < 1:
        idle = gpu_is_idle()
        consecutive_idle = consecutive_idle + 1 if idle else 0
        print(json.dumps({"time": time.time(), "gpu_idle": idle, "consecutive_idle": consecutive_idle}), flush=True)
        if consecutive_idle < 1:
            time.sleep(60)
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    ).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            sample_id = str(row["sample_id"])
            if sample_id in completed:
                continue
            processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, capture_output=True)
            foreign = [pid.strip() for pid in processes.stdout.splitlines() if pid.strip() and pid.strip() != str(os.getpid())]
            if processes.returncode or foreign:
                print("Yielding our VLM process; another GPU process appeared. Resume from saved JSONL.", flush=True)
                raise SystemExit(75)
            prompt_messages = messages(row, args.min_pixels, args.max_pixels)
            prompt = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(prompt_messages)
            model_inputs: dict[str, Any] = {
                "text": [prompt],
                "images": image_inputs,
                "padding": True,
                "return_tensors": "pt",
            }
            if video_inputs:
                model_inputs["videos"] = video_inputs
            encoded = processor(**model_inputs).to(args.device)
            with torch.inference_mode():
                generated_ids = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            generated = generated_ids[:, encoded["input_ids"].shape[1] :]
            raw_text = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
            try:
                plan = normalize_plan(extract_json(raw_text), row["question"])
                status, error = "success", ""
            except (ValueError, json.JSONDecodeError) as exc:
                plan, status, error = {}, "plan_parse_failed", str(exc)
            record = {
                "sample_id": sample_id,
                "chart_id": row["chart_id"],
                "status": status,
                "plan": plan,
                "raw_text": raw_text,
                "messages": prompt_messages,
                "planner_version": "visual_regions_multitarget_v1",
                "error": error,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            completed[sample_id] = record
            print(json.dumps({"processed": index, "total": len(rows), "sample_id": sample_id, "status": status}), flush=True)


if __name__ == "__main__":
    main()
