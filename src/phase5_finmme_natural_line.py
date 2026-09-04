"""Natural no-value-label FinMME line-chart Geometry validation.

The run command reads only inputs.csv. Gold, tolerance, and the existing Raw
VLM predictions are joined later by evaluate, so Geometry cannot select an x
position or line from answer-side information.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import cv2
import easyocr

from phase2_natural_bar import calibrate_value, execute_operation, load_calibrator
from phase3_line_core import parse_axis_scalar
from phase4_multiline_core import auto_multiline_pipeline
from phase5_natural_qa import operation_for_question, simple_question, x_targets


BOUNDARY_OR_AGGREGATE = re.compile(
    r"\b(?:range|start of|end of|beginning of|early|late|throughout|during the year|year[- ]to[- ]date)\b",
    re.IGNORECASE,
)
SERIES_PATTERNS = (
    (re.compile(r"\bclosing price\b", re.IGNORECASE), "closing price"),
    (re.compile(r"\bstock price\b", re.IGNORECASE), "stock price"),
    (re.compile(r"\bshare price\b", re.IGNORECASE), "share price"),
    (re.compile(r"\bP/?E(?: ratio| value)?\b", re.IGNORECASE), "P/E"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def numeric(value: Any) -> float | None:
    match = re.search(r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(value))
    if not match:
        return None
    try:
        result = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def semantic_series(question: str) -> str:
    for pattern, name in SERIES_PATTERNS:
        if pattern.search(question):
            return name
    return ""


def structural_candidate(scan: dict[str, Any]) -> bool:
    reasons = set(scan.get("rejection_reasons") or [])
    return (
        scan.get("status") == "rejected"
        and "no_data_value_label" in reasons
        and "insufficient_x_anchors" not in reasons
        and int(scan.get("line_series_count", 0)) == 1
    )


def target_granularity_mismatch(targets: list[str], scan: dict[str, Any]) -> bool:
    parsed_targets = [parse_axis_scalar(target) for target in targets]
    target_families = {target.family for target in parsed_targets if target is not None}
    anchor_families = {
        anchor.get("scalar_family")
        for anchor in scan.get("x_axis_anchors", [])
        if anchor.get("scalar_family") in {"year", "quarter", "month"}
    }
    return target_families == {"year"} and "year" not in anchor_families and bool(anchor_families & {"quarter", "month"})


def prepare(args: argparse.Namespace) -> None:
    records = {str(row["sample_id"]): row for row in read_jsonl(args.records)}
    raw_vlm = {str(row["sample_id"]): row for row in read_jsonl(args.raw_vlm)}
    raw_metrics = {str(row["sample_id"]): row for row in read_jsonl(args.raw_vlm_metrics)}
    scans = [row for row in read_jsonl(args.chart_scan) if structural_candidate(row)]

    inputs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    screening: list[dict[str, Any]] = []
    accepted_charts: set[str] = set()

    for scan in sorted(scans, key=lambda row: str(row["chart_id"])):
        sample_ids = [
            item.strip()
            for item in re.split(r"[|,]", str(scan.get("numerical_sample_ids", "")))
            if item.strip()
        ]
        for sample_id in sample_ids:
            record = records.get(sample_id)
            caption = str(scan.get("caption", ""))
            decision = {
                "sample_id": sample_id,
                "chart_id": scan["chart_id"],
                "caption": caption,
                "question": "" if record is None else record.get("question", ""),
                "decision": "reject",
                "reason": "",
            }
            if record is None:
                decision["reason"] = "record_missing"
                screening.append(decision)
                continue
            question = str(record.get("question", ""))
            operation = operation_for_question(question)
            if record.get("question_type") != "numerical" or numeric(record.get("reference")) is None:
                decision["reason"] = "not_numeric_question"
            elif operation is None or not simple_question(question, operation):
                decision["reason"] = "question_not_simple_point_reading"
            elif BOUNDARY_OR_AGGREGATE.search(question):
                decision["reason"] = "target_is_interval_or_boundary_not_point"
            elif re.search(r"\bband\b", caption, re.IGNORECASE):
                decision["reason"] = "reference_band_not_simple_single_line"
            elif int((scan.get("prefilter") or {}).get("bar_rectangles", 0)) > 2:
                decision["reason"] = "possible_bar_or_bar_line_combination"
            elif int(scan.get("legend_entry_count", 0)) > 1 and re.search(
                r"comparison of| versus | vs\.? ", str(scan.get("caption", "")), re.IGNORECASE
            ):
                decision["reason"] = "multi_entity_line_chart"
            elif str(scan["chart_id"]) in accepted_charts:
                decision["reason"] = "one_question_per_chart"
            else:
                required = 2 if operation in {"difference", "growth_rate"} else 1
                targets = x_targets(question, 10)
                if len(targets) != required:
                    decision["reason"] = "temporal_target_count_mismatch"
                elif target_granularity_mismatch(targets, scan):
                    decision["reason"] = "target_granularity_coarser_than_axis"
                else:
                    annotated = int(scan.get("legend_entry_count", 0)) > 1 or re.search(
                        r"\b(?:target price|rating|recommendation)\b", caption, re.IGNORECASE
                    )
                    tier = "exploratory_annotated" if annotated else "primary_clean"
                    series = semantic_series(question)
                    inputs.append(
                        {
                            "sample_id": sample_id,
                            "chart_id": scan["chart_id"],
                            "image_path": scan["image_path"],
                            "question": question,
                            "operation": operation,
                            "semantic_targets": " | ".join(targets),
                            "series_semantic": series,
                            "evaluation_tier": tier,
                        }
                    )
                    gold.append(
                        {
                            "sample_id": sample_id,
                            "reference": record.get("reference"),
                            "tolerance": record.get("tolerance"),
                            "unit": record.get("unit", ""),
                            "question_type": record.get("question_type", ""),
                        }
                    )
                    raw = raw_vlm.get(sample_id, {})
                    metric = raw_metrics.get(sample_id, {})
                    baselines.append(
                        {
                            "sample_id": sample_id,
                            "raw_vlm_response": raw.get("raw_response"),
                            "raw_vlm_parsed_prediction": raw.get("parsed_prediction"),
                            "raw_vlm_parse_status": raw.get("parse_status"),
                            "existing_official_code_correct": metric.get("official_code_correct"),
                            "existing_paper_question_score": metric.get("paper_question_score"),
                        }
                    )
                    accepted_charts.add(str(scan["chart_id"]))
                    decision.update({"decision": "accept", "reason": tier, "operation": operation, "targets": " | ".join(targets)})
            screening.append(decision)
            if len(inputs) >= args.max_questions:
                break
        if len(inputs) >= args.max_questions:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "inputs.csv", inputs)
    write_csv(args.output_dir / "gold.csv", gold)
    write_csv(args.output_dir / "raw_vlm.csv", baselines)
    write_csv(args.output_dir / "screening.csv", screening)
    summary = {
        "structural_natural_single_line_charts": len(scans),
        "accepted_questions": len(inputs),
        "accepted_charts": len({row["chart_id"] for row in inputs}),
        "tiers": dict(Counter(row["evaluation_tier"] for row in inputs)),
        "operations": dict(Counter(row["operation"] for row in inputs)),
        "rejection_reasons": dict(Counter(row["reason"] for row in screening if row["decision"] == "reject")),
        "geometry_runtime_fields_exclude": ["reference", "tolerance", "raw_vlm_prediction", "hidden_bbox", "manual_target_x"],
    }
    write_json(args.output_dir / "prepare_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def make_reader(model_dir: Path) -> easyocr.Reader:
    return easyocr.Reader(
        ["en"], gpu=False, model_storage_directory=str(model_dir), download_enabled=False, verbose=False
    )


def geometry_question(target: str, series: str) -> str:
    display_target = re.sub(r"([A-Za-z]+)(\d{2,4})$", r"\1 \2", target)
    return f"What is the value of {series} at {display_target}?" if series else f"What is the value at {display_target}?"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")[:40] or "target"


def draw_overlay(
    image: Any,
    sample_id: str,
    target: str,
    question: str,
    pipeline: dict[str, Any],
    raw_value: float | None,
) -> Any:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    axis = pipeline.get("axis") or {}
    for tick in axis.get("ticks", []):
        y = int(round(float(tick["pixel_y"])))
        cv2.circle(canvas, (int(round(float(tick["bbox"][2]))), y), 4, (255, 255, 0), -1)
        cv2.putText(canvas, f"{float(tick['value']):g}", (3, max(14, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 100, 0), 1, cv2.LINE_AA)
    for anchor in pipeline.get("x_axis_anchors", []):
        x1, y1, x2, y2 = [int(round(float(value))) for value in anchor["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 210, 0), 1)
        cv2.circle(canvas, (int(round(float(anchor["center_x"]))), int(round(float(anchor["center_y"])))), 3, (255, 210, 0), -1)
    resolution = pipeline.get("series_resolution") or {}
    selected_id = resolution.get("target_series_id")
    for series in pipeline.get("detected_series", []):
        x, y, w, h = map(int, series["bbox"])
        selected = selected_id is not None and int(series["series_id"]) == int(selected_id)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 0, 255) if selected else (150, 150, 150), 2 if selected else 1)
    grounding = pipeline.get("grounding") or {}
    target_x = grounding.get("predicted_target_x")
    if target_x is not None:
        x = int(round(float(target_x)))
        cv2.line(canvas, (x, 0), (x, height - 1), (255, 0, 255), 2)
    geometry = pipeline.get("auto_local_geometry")
    if geometry:
        window = geometry.get("search_window")
        if window and axis.get("plot_bbox"):
            x1, x2 = [int(round(float(value))) for value in window]
            cv2.rectangle(canvas, (x1, int(axis["plot_bbox"][1])), (x2, int(axis["plot_bbox"][3])), (0, 210, 255), 1)
        for segment in geometry.get("fitted_segments", []):
            sx1, sx2 = float(segment["x_min"]), float(segment["x_max"])
            sy1 = float(segment["target_y"]) + float(segment["slope"]) * (sx1 - float(geometry["target_x"]))
            sy2 = float(segment["target_y"]) + float(segment["slope"]) * (sx2 - float(geometry["target_x"]))
            cv2.line(canvas, (round(sx1), round(sy1)), (round(sx2), round(sy2)), (0, 190, 0), 2)
        px, py = [int(round(float(value))) for value in geometry["point"]]
        cv2.circle(canvas, (px, py), 7, (0, 0, 255), 2)
    lines = [
        f"{sample_id} target={target} status={pipeline.get('status')}",
        f"x={target_x if target_x is not None else 'NA'} mode={grounding.get('mode','NA')} series={selected_id if selected_id is not None else 'NA'}",
        f"raw_geometry={raw_value:.6g}" if raw_value is not None else f"raw_geometry=NA reason={pipeline.get('status_reason','')}",
        question[:150],
    ]
    panel_height = 22 * len(lines) + 6
    panel = canvas[:panel_height, : min(width, 1100)]
    panel[:] = 255
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (7, 18 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (15, 15, 15), 1, cv2.LINE_AA)
    return canvas


def run(args: argparse.Namespace) -> None:
    rows = read_csv(args.inputs)
    calibrator = load_calibrator(args.calibrator)
    reader = make_reader(args.model_dir)
    existing = {str(row["sample_id"]) for row in read_jsonl(args.output)} if args.output.exists() else set()
    args.overlay_dir.mkdir(parents=True, exist_ok=True)
    completed = len(existing)
    for index, row in enumerate(rows, 1):
        if row["sample_id"] in existing:
            continue
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        base: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "chart_id": row["chart_id"],
            "question": row["question"],
            "operation": row["operation"],
            "semantic_targets": row["semantic_targets"].split(" | "),
            "series_semantic": row["series_semantic"],
            "evaluation_tier": row["evaluation_tier"],
        }
        if image is None:
            append_jsonl(args.output, {**base, "status": "image_read_failed", "target_results": []})
            continue
        target_results: list[dict[str, Any]] = []
        for target_index, target in enumerate(base["semantic_targets"], 1):
            pipeline = auto_multiline_pipeline(reader, image, geometry_question(target, row["series_semantic"]))
            local_value = pipeline.get("auto_local_value") if pipeline.get("status") == "success" else None
            column_value = pipeline.get("auto_column_value") if pipeline.get("status") == "success" else None
            calibrated_value = None
            predicted_error = None
            geometry = pipeline.get("auto_local_geometry")
            if local_value is not None and geometry is not None:
                calibrated_value, predicted_error = calibrate_value(float(local_value), pipeline["axis"], geometry, calibrator)
            target_results.append(
                {
                    "semantic_target": target,
                    "geometry_question": geometry_question(target, row["series_semantic"]),
                    "status": pipeline.get("status"),
                    "status_reason": pipeline.get("status_reason"),
                    "raw_column_value": column_value,
                    "raw_local_value": local_value,
                    "calibrated_local_value": calibrated_value,
                    "predicted_calibration_error": predicted_error,
                    "pipeline": pipeline,
                }
            )
            overlay = draw_overlay(image, row["sample_id"], target, row["question"], pipeline, local_value)
            cv2.imwrite(str(args.overlay_dir / f"{row['sample_id']}_{target_index}_{safe_name(target)}.png"), overlay)

        local_values = [item["raw_local_value"] for item in target_results]
        column_values = [item["raw_column_value"] for item in target_results]
        calibrated_values = [item["calibrated_local_value"] for item in target_results]
        status = "success" if all(value is not None for value in local_values) else "geometry_failed"
        result = {**base, "status": status, "target_results": target_results}
        if status == "success":
            result["geometry_local_prediction"] = execute_operation(row["operation"], [float(value) for value in local_values], row["question"])
            if all(value is not None for value in column_values):
                result["geometry_column_prediction"] = execute_operation(row["operation"], [float(value) for value in column_values], row["question"])
            if all(value is not None for value in calibrated_values):
                result["geometry_calibrated_prediction"] = execute_operation(row["operation"], [float(value) for value in calibrated_values], row["question"])
        append_jsonl(args.output, result)
        completed += 1
        if completed == 1 or completed % args.progress_every == 0:
            print(json.dumps({"completed": completed, "total": len(rows), "sample_id": row["sample_id"], "status": status}), flush=True)
    print(json.dumps({"rows": len(rows), "completed": completed, "output": str(args.output)}, sort_keys=True))


def evaluate(args: argparse.Namespace) -> None:
    predictions = {str(row["sample_id"]): row for row in read_jsonl(args.predictions)}
    gold = {row["sample_id"]: row for row in read_csv(args.gold)}
    raw_vlm = {row["sample_id"]: row for row in read_csv(args.raw_vlm)}
    samples: list[dict[str, Any]] = []
    methods = {
        "raw_vlm": lambda row, baseline: numeric(baseline.get("raw_vlm_response")),
        "geometry_column": lambda row, baseline: numeric(row.get("geometry_column_prediction")),
        "geometry_local": lambda row, baseline: numeric(row.get("geometry_local_prediction")),
        "geometry_calibrated": lambda row, baseline: numeric(row.get("geometry_calibrated_prediction")),
    }
    for sample_id, target in gold.items():
        row = predictions.get(sample_id, {"sample_id": sample_id, "status": "missing_prediction", "target_results": []})
        baseline = raw_vlm.get(sample_id, {})
        reference = numeric(target.get("reference"))
        tolerance = numeric(target.get("tolerance"))
        output: dict[str, Any] = {
            "sample_id": sample_id,
            "chart_id": row.get("chart_id"),
            "evaluation_tier": row.get("evaluation_tier"),
            "operation": row.get("operation"),
            "question": row.get("question"),
            "semantic_targets": " | ".join(row.get("semantic_targets", [])),
            "series_semantic": row.get("series_semantic"),
            "geometry_status": row.get("status"),
            "reference": target.get("reference"),
            "tolerance": target.get("tolerance"),
            "unit": target.get("unit"),
            "target_statuses": " | ".join(str(item.get("status")) for item in row.get("target_results", [])),
            "detected_x_labels": " | ".join(str((item.get("pipeline", {}).get("grounding") or {}).get("matched_detected_label") or "") for item in row.get("target_results", [])),
            "predicted_target_x": " | ".join(str((item.get("pipeline", {}).get("grounding") or {}).get("predicted_target_x") or "") for item in row.get("target_results", [])),
            "selected_series_ids": " | ".join(str((item.get("pipeline", {}).get("series_resolution") or {}).get("target_series_id") or "") for item in row.get("target_results", [])),
            "target_points": json.dumps([item.get("pipeline", {}).get("auto_local_geometry", {}).get("point") for item in row.get("target_results", [])]),
            "raw_geometry_values": " | ".join(str(item.get("raw_local_value") or "") for item in row.get("target_results", [])),
            "calibrated_geometry_values": " | ".join(str(item.get("calibrated_local_value") or "") for item in row.get("target_results", [])),
            "y_axis_ticks": json.dumps([(item.get("pipeline", {}).get("axis") or {}).get("ticks", []) for item in row.get("target_results", [])]),
            "x_axis_anchors": json.dumps([item.get("pipeline", {}).get("x_axis_anchors", []) for item in row.get("target_results", [])]),
            "raw_vlm_response": baseline.get("raw_vlm_response"),
            "raw_vlm_existing_official_code_correct": baseline.get("existing_official_code_correct"),
            "raw_vlm_existing_paper_question_score": baseline.get("existing_paper_question_score"),
        }
        for method, getter in methods.items():
            value = getter(row, baseline)
            error = None if value is None or reference is None else abs(value - reference)
            output[f"{method}_prediction"] = value
            output[f"{method}_absolute_error"] = error
            output[f"{method}_tolerance_correct"] = int(error is not None and tolerance is not None and error <= tolerance)
        samples.append(output)

    metric_rows: list[dict[str, Any]] = []
    scopes = {
        "all": samples,
        "primary_clean": [row for row in samples if row["evaluation_tier"] == "primary_clean"],
        "exploratory_annotated": [row for row in samples if row["evaluation_tier"] == "exploratory_annotated"],
    }
    for scope, rows in scopes.items():
        for method in methods:
            covered = [row for row in rows if numeric(row.get(f"{method}_prediction")) is not None]
            errors = [float(row[f"{method}_absolute_error"]) for row in covered]
            metric_rows.append(
                {
                    "scope": scope,
                    "method": method,
                    "samples": len(rows),
                    "covered": len(covered),
                    "coverage": len(covered) / len(rows) if rows else 0.0,
                    "accuracy": sum(int(row[f"{method}_tolerance_correct"]) for row in rows) / len(rows) if rows else 0.0,
                    "conditional_accuracy": sum(int(row[f"{method}_tolerance_correct"]) for row in covered) / len(covered) if covered else 0.0,
                    "mae": mean(errors) if errors else "",
                    "median_absolute_error": median(errors) if errors else "",
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "samples.csv", samples)
    write_csv(args.output_dir / "metrics.csv", metric_rows)
    summary = {
        "samples": len(samples),
        "geometry_success": sum(row["geometry_status"] == "success" for row in samples),
        "target_failure_statuses": dict(Counter(status for row in samples for status in row["target_statuses"].split(" | ") if status and status != "success")),
    }
    write_json(args.output_dir / "evaluation_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--chart-scan", type=Path, required=True)
    prep.add_argument("--records", type=Path, required=True)
    prep.add_argument("--raw-vlm", type=Path, required=True)
    prep.add_argument("--raw-vlm-metrics", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--max-questions", type=int, default=50)
    run_command = commands.add_parser("run")
    run_command.add_argument("--inputs", type=Path, required=True)
    run_command.add_argument("--model-dir", type=Path, required=True)
    run_command.add_argument("--calibrator", type=Path, required=True)
    run_command.add_argument("--output", type=Path, required=True)
    run_command.add_argument("--overlay-dir", type=Path, required=True)
    run_command.add_argument("--progress-every", type=int, default=5)
    score = commands.add_parser("evaluate")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--gold", type=Path, required=True)
    score.add_argument("--raw-vlm", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
