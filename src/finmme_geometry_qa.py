#!/usr/bin/env python3
"""End-to-end FinMME QA with the existing Bar/Line Geometry modules."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

BOUNDARY_RE = re.compile(
    r"\b(?:range|start of|end of|beginning of|early|late|throughout|during the year|year[- ]to[- ]date)\b",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def natural_chart_kind(scan: dict[str, Any]) -> str | None:
    reasons = set(scan.get("rejection_reasons") or [])
    prefilter = scan.get("prefilter") or {}
    if (
        scan.get("bar_candidate")
        and int(scan.get("line_series_count", 0)) == 0
        and int(prefilter.get("sloped_segments", 0)) <= 1
        and int(scan.get("data_label_candidate_count", 0)) == 0
        and scan.get("axis") is not None
    ):
        return "bar"
    if (
        scan.get("status") == "rejected"
        and "no_data_value_label" in reasons
        and "insufficient_x_anchors" not in reasons
        and int(scan.get("line_series_count", 0)) == 1
    ):
        return "line"
    return None


def options_are_numeric(options: str) -> bool:
    from phase2_natural_bar import numeric_option_value, parse_option_rows

    values = [numeric_option_value(text) for _, text in parse_option_rows(options)]
    return len(values) >= 2 and sum(value is not None for value in values) == len(values)


def prepare(args: argparse.Namespace) -> None:
    from phase5_natural_qa import COMPLEX_QUESTION_RE, operation_for_question, simple_question, x_targets

    records = read_jsonl(args.records)
    scans = {str(row["chart_id"]): row for row in read_jsonl(args.chart_scan)}
    raw_metrics = {str(row["sample_id"]): row for row in read_jsonl(args.raw_metrics)}
    inputs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    rejected = Counter()

    for record in records:
        scan = scans.get(str(record["official_preprocessed_image_sha256"]))
        kind = None if scan is None else natural_chart_kind(scan)
        if kind is None:
            continue
        question_type = str(record.get("question_type", ""))
        question = str(record.get("question", ""))
        operation = operation_for_question(question)
        if question_type not in {"numerical", "single_choice"}:
            rejected["unsupported_question_type"] += 1
            continue
        if operation is None or COMPLEX_QUESTION_RE.search(question) or BOUNDARY_RE.search(question):
            rejected["unsupported_operation_or_aggregate"] += 1
            continue
        if question_type == "single_choice" and not options_are_numeric(str(record.get("options", ""))):
            rejected["non_numeric_choice_options"] += 1
            continue
        targets = x_targets(question, 10)
        required = 2 if operation in {"difference", "growth_rate"} else 1
        if kind == "line" and (not simple_question(question, operation) or len(targets) != required):
            rejected["line_target_not_simple"] += 1
            continue

        sample_id = str(record["sample_id"])
        metric = raw_metrics.get(sample_id, {})
        inputs.append(
            {
                "sample_id": sample_id,
                "chart_id": scan["chart_id"],
                "chart_kind": kind,
                "image_path": scan["image_path"],
                "question": question,
                "question_type": question_type,
                "options": record.get("options", ""),
                "operation": operation,
                "line_targets": targets,
                "prefilter": scan.get("prefilter", {}),
            }
        )
        gold.append(
            {
                "sample_id": sample_id,
                "reference": record.get("reference"),
                "tolerance": record.get("tolerance"),
                "unit": record.get("unit", ""),
            }
        )
        baselines.append(
            {
                "sample_id": sample_id,
                "raw_vlm_correct": bool(metric.get("official_code_correct")),
                "raw_vlm_score": float(metric.get("paper_question_score", 0.0) or 0.0),
            }
        )

    write_jsonl(args.output_dir / "inputs.jsonl", inputs)
    write_jsonl(args.output_dir / "gold.jsonl", gold)
    write_jsonl(args.output_dir / "baseline.jsonl", baselines)
    summary = {
        "candidate_questions": len(inputs),
        "candidate_charts": len({row["chart_id"] for row in inputs}),
        "chart_types": dict(Counter(row["chart_kind"] for row in inputs)),
        "question_types": dict(Counter(row["question_type"] for row in inputs)),
        "operations": dict(Counter(row["operation"] for row in inputs)),
        "rejected": dict(rejected),
    }
    (args.output_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


def finish_answer(row: dict[str, Any], values: list[float]) -> dict[str, Any]:
    from phase2_natural_bar import answer_from_geometry

    prediction, numeric_answer, reasoning = answer_from_geometry(
        row["question_type"], row["options"], row["operation"], values, row["question"]
    )
    return {
        "status": "success",
        "prediction": prediction,
        "numeric_answer": numeric_answer,
        "geometry_values": values,
        "reasoning": reasoning,
    }


def run_bar(reader: Any, row: dict[str, Any], image: Any) -> dict[str, Any]:
    from phase2_natural_bar import analyze_chart, semantic_plan

    analysis = analyze_chart(reader, row["chart_id"], Path(row["image_path"]), row["prefilter"])
    if analysis.get("status") != "eligible_natural_no_label_bar":
        return {"status": str(analysis.get("status")), "stage": "bar_chart_analysis"}
    plan = semantic_plan(row["question"], row["options"], analysis["bars"])
    if plan.get("status") != "success":
        return {"status": str(plan.get("status")), "stage": "semantic_targeting"}
    bars = {bar["bar_id"]: bar for bar in analysis["bars"]}
    selected = [bars[bar_id] for bar_id in plan["target_bar_ids"]]
    values = [float(analysis["axis"]["slope"]) * float(bar["target_y"]) + float(analysis["axis"]["intercept"]) for bar in selected]
    return {
        **finish_answer(row, values),
        "semantic_targets": plan["target_categories"],
        "selected_targets": [{"bar_id": bar["bar_id"], "bbox": bar["bbox"]} for bar in selected],
        "y_axis_ticks": analysis["axis"]["ticks"],
    }


def run_line(reader: Any, row: dict[str, Any], image: Any) -> dict[str, Any]:
    from phase4_multiline_core import auto_multiline_pipeline
    from phase5_finmme_natural_line import geometry_question, semantic_series

    series = semantic_series(row["question"])
    pipelines = [
        auto_multiline_pipeline(reader, image, geometry_question(target, series))
        for target in row["line_targets"]
    ]
    if any(item.get("status") != "success" for item in pipelines):
        return {
            "status": " | ".join(str(item.get("status")) for item in pipelines),
            "stage": "x_grounding_or_line_fit",
        }
    values = [item.get("auto_local_value") for item in pipelines]
    if any(value is None for value in values):
        return {"status": "line_point_unrecoverable", "stage": "line_fitting"}
    result = finish_answer(row, [float(value) for value in values if value is not None])
    result.update(
        {
            "semantic_targets": row["line_targets"],
            "selected_targets": [item.get("auto_local_geometry") for item in pipelines],
            "detected_x_labels": [item.get("grounding", {}).get("matched_detected_label") for item in pipelines],
            "y_axis_ticks": pipelines[0]["axis"]["ticks"],
        }
    )
    return result


def run(args: argparse.Namespace) -> None:
    import cv2
    import easyocr
    import torch

    cv2.setNumThreads(args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    rows = read_jsonl(args.inputs)
    completed = {str(row["sample_id"]): row for row in read_jsonl(args.output)} if args.output.exists() else {}
    reader = easyocr.Reader(
        ["en"], gpu=False, model_storage_directory=str(args.model_dir), download_enabled=False, verbose=False
    )
    with args.output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            if row["sample_id"] in completed:
                continue
            image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
            base = {key: row[key] for key in ("sample_id", "chart_id", "chart_kind", "question_type", "operation")}
            if image is None:
                result = {**base, "status": "image_read_failed", "stage": "image"}
            else:
                try:
                    detail = run_bar(reader, row, image) if row["chart_kind"] == "bar" else run_line(reader, row, image)
                    result = {**base, **detail}
                except (ValueError, ZeroDivisionError) as exc:
                    result = {**base, "status": "reasoning_failed", "stage": "reasoning", "error": str(exc)}
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(f"{index}/{len(rows)} {row['sample_id']} {result['status']}", flush=True)


def aggregate(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    covered = [row for row in rows if row["status"] == "success"]
    return {
        "scope": scope,
        "questions": len(rows),
        "coverage": len(covered) / len(rows) if rows else 0.0,
        "raw_vlm_accuracy": sum(row["raw_vlm_correct"] for row in rows) / len(rows) if rows else 0.0,
        "geometry_accuracy_all": sum(row["geometry_correct"] for row in rows) / len(rows) if rows else 0.0,
        "geometry_accuracy_covered": sum(row["geometry_correct"] for row in covered) / len(covered) if covered else 0.0,
        "fallback_accuracy": sum(row["fallback_correct"] for row in rows) / len(rows) if rows else 0.0,
    }


def reference_number(value: Any) -> float | None:
    match = re.search(r"[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def correctness(question_type: str, prediction: Any, reference: Any, tolerance: Any) -> bool:
    """Apply the FinMME numeric tolerance or exact choice-letter match."""
    if question_type == "numerical":
        predicted = reference_number(prediction)
        gold = reference_number(reference)
        try:
            tol = float(tolerance)
        except (TypeError, ValueError):
            return False
        return predicted is not None and gold is not None and math.isfinite(tol) and abs(predicted - gold) <= tol
    return str(prediction).strip().upper() == str(reference).strip().upper()


def evaluate(args: argparse.Namespace) -> None:
    inputs = {str(row["sample_id"]): row for row in read_jsonl(args.inputs)}
    gold = {str(row["sample_id"]): row for row in read_jsonl(args.gold)}
    baseline = {str(row["sample_id"]): row for row in read_jsonl(args.baseline)}
    predictions = {str(row["sample_id"]): row for row in read_jsonl(args.predictions)}
    raw_all = read_jsonl(args.raw_metrics)
    output: list[dict[str, Any]] = []

    for sample_id, source in inputs.items():
        prediction = predictions.get(sample_id, {"status": "missing_prediction", "stage": "runtime"})
        answer = gold[sample_id]
        raw = baseline[sample_id]
        correct = prediction.get("status") == "success" and correctness(
            source["question_type"], prediction.get("prediction"), answer.get("reference"), answer.get("tolerance")
        )
        fallback_correct = correct if prediction.get("status") == "success" else bool(raw["raw_vlm_correct"])
        output.append(
            {
                "sample_id": sample_id,
                "chart_id": source["chart_id"],
                "chart_kind": source["chart_kind"],
                "question_type": source["question_type"],
                "operation": source["operation"],
                "question": source["question"],
                "status": prediction.get("status"),
                "failure_stage": prediction.get("stage", ""),
                "semantic_targets": json.dumps(prediction.get("semantic_targets", []), ensure_ascii=False),
                "detected_x_labels": json.dumps(prediction.get("detected_x_labels", []), ensure_ascii=False),
                "selected_targets": json.dumps(prediction.get("selected_targets", []), ensure_ascii=False),
                "y_axis_ticks": json.dumps(prediction.get("y_axis_ticks", []), ensure_ascii=False),
                "geometry_values": json.dumps(prediction.get("geometry_values", [])),
                "numeric_answer": prediction.get("numeric_answer", ""),
                "reasoning": json.dumps(prediction.get("reasoning", {}), ensure_ascii=False),
                "geometry_prediction": prediction.get("prediction", ""),
                "gold": answer.get("reference"),
                "tolerance": answer.get("tolerance"),
                "geometry_correct": int(correct),
                "raw_vlm_correct": int(bool(raw["raw_vlm_correct"])),
                "fallback_correct": int(fallback_correct),
            }
        )

    metrics = [aggregate(output, "all")]
    for field in ("chart_kind", "question_type", "operation"):
        for value in sorted({str(row[field]) for row in output}):
            metrics.append(aggregate([row for row in output if str(row[field]) == value], f"{field}:{value}"))

    official_total = sum(bool(row.get("official_code_correct")) for row in raw_all)
    paper_total = sum(float(row.get("paper_question_score", 0.0) or 0.0) for row in raw_all)
    official_hard = official_total
    official_fallback = official_total
    paper_hard = paper_total
    paper_fallback = paper_total
    official_by_id = {str(row["sample_id"]): int(bool(row.get("official_code_correct"))) for row in raw_all}
    paper_by_id = {
        str(row["sample_id"]): float(row.get("paper_question_score", 0.0) or 0.0) for row in raw_all
    }
    for row in output:
        old_official = official_by_id[row["sample_id"]]
        old_paper = paper_by_id[row["sample_id"]]
        official_hard += row["geometry_correct"] - old_official
        official_fallback += row["fallback_correct"] - old_official
        paper_hard += row["geometry_correct"] - old_paper
        paper_fallback += row["fallback_correct"] - old_paper
    metrics.extend(
        [
            {"scope": "full_finmme_raw_vlm_official", "questions": len(raw_all), "accuracy": official_total / len(raw_all)},
            {"scope": "full_finmme_geometry_hard_route_official", "questions": len(raw_all), "accuracy": official_hard / len(raw_all)},
            {"scope": "full_finmme_geometry_fallback_official", "questions": len(raw_all), "accuracy": official_fallback / len(raw_all)},
            {"scope": "full_finmme_raw_vlm_paper_protocol", "questions": len(raw_all), "accuracy": paper_total / len(raw_all)},
            {"scope": "full_finmme_geometry_hard_route_paper_protocol", "questions": len(raw_all), "accuracy": paper_hard / len(raw_all)},
            {"scope": "full_finmme_geometry_fallback_paper_protocol", "questions": len(raw_all), "accuracy": paper_fallback / len(raw_all)},
        ]
    )
    write_csv(args.output_dir / "finmme_geometry_qa_samples.csv", output)
    write_csv(args.output_dir / "finmme_geometry_qa_metrics.csv", metrics)
    failures = Counter(row["failure_stage"] or "answered_but_wrong" for row in output if not row["geometry_correct"])
    summary = {"routed_questions": len(output), "runtime_success": sum(row["status"] == "success" for row in output), "failure_stages": dict(failures)}
    (args.output_dir / "finmme_geometry_qa_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--records", type=Path, required=True)
    prep.add_argument("--chart-scan", type=Path, required=True)
    prep.add_argument("--raw-metrics", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--inputs", type=Path, required=True)
    run_parser.add_argument("--model-dir", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--threads", type=int, default=4)
    score = commands.add_parser("evaluate")
    score.add_argument("--inputs", type=Path, required=True)
    score.add_argument("--gold", type=Path, required=True)
    score.add_argument("--baseline", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--raw-metrics", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    {"prepare": prepare, "run": run, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
