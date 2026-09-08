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
    prompt = f"""You are the visual planner for a chart measurement system. Inspect the ORIGINAL IMAGE,
question AND every answer option. Specify the complete visual evidence needed to decide ALL options.
Do not answer the question or output guessed chart data values. Pixel regions are allowed.
Return one JSON object:
{{"chart_type":"bar|line|other","y_axis_count":1,"has_direct_value_labels":false,
"operation":"direct|difference|growth_rate|sum|mean|min|max|ratio|evidence|unsupported",
"targets":[{{"x_label":"category/date or interval description","series":"requested visual series/segment",
"position":"label|first|last","measurement":"value|height|sequence",
"region":[left,top,right,bottom]}}],
"calculations":[],"reason":"brief explanation of what to measure"}}

- region is a tight approximate bounding box in ORIGINAL IMAGE coordinates normalized to 0..1000.
  Enclose the requested plotted mark, not its text label. Geometry will snap to image pixels;
  your approximate coordinates are not the final measured values. Supply a region for each target.
- Read titles, dates, legend and visual encodings to identify the actual requested object.
  x_label is its category/date; series identifies the line or bar segment. first/last refers to
  the first/last plotted observation, even if the date is absent from axis ticks.
- value reads a line point or bar endpoint. height reads both ends of a bar/segment, including
  floating bars. Enclose only the requested segment, or the entire column if its total is requested.
- sequence reads the entire selected line within the region's horizontal interval. Use it for
  trends, extrema, rankings over time or claims such as always above. Cover the full required
  interval, not only the point you believe wins. Use operation=evidence for sequences.
- Return as many targets as needed, in a stable order. Multi-choice uses the SAME measurements:
  cover every option, not only options you believe correct. Share repeated measurements.
- direct requires one target; difference subtracts second from first; growth_rate uses old then
  new; ratio divides first by second. sum/mean/min/max apply to all scalar targets.
  Use evidence for multiple independent comparisons, sequences, or several calculations.
- For multi-step arithmetic, calculations may contain named expression trees, for example
  {{"name":"difference of totals","expression":{{"op":"difference","args":[
  {{"op":"sum","args":[{{"target":0}},{{"target":1}}]}},{{"target":2}}]}}}}.
  target indices start at zero. Allowed op names are the arithmetic operations above.
  Use only measured target references, no guessed numeric literals or executable code.
- Reading an already plotted rate or average is direct. Do not substitute total growth for
  annualized growth. If required evidence cannot be obtained, mark unsupported.
- has_direct_value_labels means requested values printed beside marks, NOT axis ticks.
  Such questions use Raw VLM. Do not copy those printed values into your plan.

Caption: {row.get('caption', '')}
Question type: {row.get('question_type', '')}
Question: {row['question']}
Options: {row.get('options', '')}"""
    return [
        {"role": "system", "content": "Plan semantic chart targets only. Never answer the chart question."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": Path(row["image_path"]).resolve().as_uri(),
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
    }
    if set(value) - allowed_keys:
        raise ValueError("unexpected_plan_keys")
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
    targets: list[dict[str, str]] = []
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
    if re.search(r"\d", reason):
        reason = ""
    plan = {
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
