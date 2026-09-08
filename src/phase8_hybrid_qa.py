#!/usr/bin/env python3
"""FinMME hybrid QA: VLM semantics, Geometry values, Python arithmetic."""

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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def single_y_axis(scan: dict[str, Any]) -> bool:
    axis = scan.get("axis") or {}
    return bool(axis) and not axis.get("right_axis_detected") and int(axis.get("additional_y_axis_count", 0)) == 0


def natural_chart_kind(scan: dict[str, Any]) -> str | None:
    if not single_y_axis(scan) or int(scan.get("data_label_candidate_count", 0)) != 0:
        return None
    if (
        scan.get("bar_candidate")
        and int(scan.get("line_series_count", 0)) == 0
        and int((scan.get("prefilter") or {}).get("sloped_segments", 0)) <= 1
    ):
        return "bar"
    reasons = set(scan.get("rejection_reasons") or [])
    if "no_data_value_label" in reasons and 1 <= int(scan.get("line_series_count", 0)) <= 4:
        return "line"
    return None


def numeric_options(options: str) -> bool:
    rows = [line for line in str(options).splitlines() if re.match(r"\s*[A-Z]\s*[:.)-]", line)]
    return len(rows) >= 2 and all(re.search(r"[-+]?\d", row) for row in rows)


def prepare(args: argparse.Namespace) -> None:
    records = read_jsonl(args.records)
    scans = {str(row["chart_id"]): row for row in read_jsonl(args.chart_scan)}
    raw = {str(row["sample_id"]): row for row in read_jsonl(args.raw_metrics)}
    planner_inputs: list[dict[str, Any]] = []
    geometry_inputs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    baseline: list[dict[str, Any]] = []
    question_audit = Counter()

    for record in records:
        sample_id = str(record["sample_id"])
        chart_id = str(record["official_preprocessed_image_sha256"])
        scan = scans.get(chart_id)
        if scan is None:
            question_audit["scan_missing"] += 1
            continue
        label_tier = "unknown_not_scanned" if "data_label_candidate_count" not in scan else "visible_value_label_candidate" if int(scan["data_label_candidate_count"]) else "no_value_label_candidate"
        question_audit[f"label_tier:{label_tier}"] += 1
        kind = natural_chart_kind(scan)
        if kind is None:
            question_audit["not_natural_single_axis_bar_line"] += 1
            continue
        question_type = str(record.get("question_type", ""))
        if question_type not in {"numerical", "single_choice", "multiple_choice"}:
            question_audit["unsupported_question_type"] += 1
            continue
        shared = {
            "sample_id": sample_id,
            "chart_id": chart_id,
            "chart_kind_hint": kind,
            "line_series_count_hint": int(scan.get("line_series_count", 0)),
            "image_path": scan["image_path"],
            "caption": str(record.get("verified_caption", "")),
            "question": str(record.get("question", "")),
            "question_type": question_type,
            "prefilter": scan.get("prefilter", {}),
            "label_tier": label_tier,
        }
        planner_inputs.append({**shared, "options": str(record.get("options", ""))})
        geometry_inputs.append({**shared, "options": str(record.get("options", ""))})
        gold.append(
            {
                "sample_id": sample_id,
                "reference": record.get("reference"),
                "tolerance": record.get("tolerance"),
                "unit": record.get("unit", ""),
            }
        )
        metric = raw.get(sample_id, {})
        baseline.append(
            {
                "sample_id": sample_id,
                "raw_vlm_correct": int(bool(metric.get("official_code_correct"))),
                "raw_vlm_paper_score": float(metric.get("paper_question_score", 0.0) or 0.0),
            }
        )
        question_audit["broad_candidate"] += 1

    write_jsonl(args.output_dir / "planner_inputs.jsonl", planner_inputs)
    write_jsonl(args.output_dir / "geometry_inputs.jsonl", geometry_inputs)
    write_jsonl(args.output_dir / "gold.jsonl", gold)
    write_jsonl(args.output_dir / "baseline.jsonl", baseline)
    selected_charts = {row["chart_id"] for row in planner_inputs}
    # Whitelist only image-derived axis/OCR/legend information. Never copy data-label boxes.
    write_jsonl(args.output_dir / "chart_geometry_inputs.jsonl", [
        {"chart_id": chart_id, "axis": scan["axis"], "x_axis_anchors": scan.get("x_axis_anchors", []), "legend": scan.get("legend") or {"entries": []}}
        for chart_id, scan in scans.items() if chart_id in selected_charts
    ])
    chart_audit = Counter()
    for scan in scans.values():
        label = "unknown_not_scanned" if "data_label_candidate_count" not in scan else "visible_value_label_candidate" if int(scan["data_label_candidate_count"]) else "no_value_label_candidate"
        chart_audit[f"label_tier:{label}"] += 1
        kind = natural_chart_kind(scan)
        if kind:
            chart_audit[f"natural_kind:{kind}"] += 1
    summary = {
        "finmme_questions": len(records),
        "finmme_charts": len(scans),
        "broad_candidates": len(planner_inputs),
        "candidate_charts": len({row["chart_id"] for row in planner_inputs}),
        "candidate_chart_kinds": dict(Counter(row["chart_kind_hint"] for row in planner_inputs)),
        "candidate_question_types": dict(Counter(row["question_type"] for row in planner_inputs)),
        "chart_audit": dict(chart_audit),
        "question_audit": dict(question_audit),
    }
    (args.output_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    chart_rows = []
    for chart_id, scan in scans.items():
        axis = scan.get("axis") or {}
        line_count = scan.get("line_series_count")
        type_hint = "unknown"
        if line_count is not None:
            type_hint = "bar_or_combo" if scan.get("bar_candidate") and line_count else "bar" if scan.get("bar_candidate") else "single_line" if line_count == 1 else "multi_line" if line_count > 1 else "unclassified"
        label_count = scan.get("data_label_candidate_count")
        chart_rows.append({"chart_id": chart_id, "scan_status": scan.get("status"), "type_hint": type_hint, "line_series_count": line_count,
            "axis_tier": "multiple_axes_or_panels" if scan.get("status") == "multiple_y_axes_or_panels_detected" or axis.get("right_axis_detected") or axis.get("additional_y_axis_count",0) else "single_y_axis_candidate" if axis else "unknown",
            "value_label_tier": "unknown_not_scanned" if label_count is None else "detected" if label_count else "not_detected", "value_label_candidate_count": label_count,
            "in_phase8_cohort": int(chart_id in selected_charts)})
    write_csv(args.output_dir / "full_chart_audit.csv", chart_rows)
    chart_by_id = {row["chart_id"]:row for row in chart_rows}
    candidate_ids = {row["sample_id"] for row in planner_inputs}
    write_csv(args.output_dir / "full_question_audit.csv", [
        {"sample_id":str(record["sample_id"]), "chart_id":record["official_preprocessed_image_sha256"], "question_type":record["question_type"],
         "in_phase8_cohort":int(str(record["sample_id"]) in candidate_ids),
         **{key:chart_by_id.get(record["official_preprocessed_image_sha256"],{}).get(key,"unknown") for key in ("type_hint","axis_tier","value_label_tier")}}
        for record in records])
    print(json.dumps(summary, sort_keys=True))


def finish_measurement(row, plan, values):
    from phase2_natural_bar import execute_operation, answer_from_geometry
    if not row.get("evidence_only"):
        return answer_from_geometry(row["question_type"], row["options"], plan["operation"], values, row["question"])
    from phase9_plan_contract import calculate, expression
    value = calculate(plan["operation"], values)
    calculations = [{"name": c.get("name", "calculation"), "expression": c["expression"],
                     "value": expression(c["expression"], values)} for c in plan.get("calculations", [])]
    return None, value, {"executor": "python", "formula": plan["operation"], "numeric_answer": value,
                         "calculations": calculations, "answer_status": "awaiting_evidence_vlm", "value_units": "axis_tick_units"}


def synthetic_bar_question(x_label: str) -> str:
    return f"What is the value for {x_label}?"


def run_bar(reader: Any, row: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    from phase2_natural_bar import analyze_chart
    from phase8_geometry_adapter import match_bar_target

    analysis = analyze_chart(reader, row["chart_id"], Path(row["image_path"]), row["prefilter"])
    from phase4_legend import detect_legend_entries
    from geometry_core import token_from_dict
    import cv2
    bar_image = cv2.imread(row["image_path"])
    legend = detect_legend_entries(bar_image, [token_from_dict(t) for t in analysis.get("ocr_tokens", [])], analysis.get("axis")) if analysis.get("axis") else {"entries": []}
    analysis["legend"] = legend
    if any(t.get("measurement") == "height" or t.get("region") for t in plan["targets"]):
        from phase8_geometry_adapter import read_bar_height
        if not analysis.get("axis") or not single_y_axis(analysis):
            return {"status": "unsupported_axis", "stage": "bar_geometry", "bar_audit": analysis}
        targets = [read_bar_height(reader, bar_image, analysis, t) for t in plan["targets"]]
        if any(t["status"] != "success" for t in targets):
            return {"status":"bar_target_unrecoverable", "stage":"bar_geometry", "target_audit":targets, "bar_audit":analysis}
        values = [t["value"] for t in targets]
        prediction,numeric_answer,reasoning = finish_measurement(row,plan,values)
        return dict(status="success", stage="answer", geometry_values=values, numeric_answer=numeric_answer,
                    prediction=prediction, reasoning=reasoning, target_audit=targets, bar_audit=analysis,
                    y_axis_ticks=analysis["axis"]["ticks"])
    if analysis.get("status") != "eligible_natural_no_label_bar":
        return {"status": str(analysis.get("status")), "stage": "bar_geometry", "bar_audit": analysis}
    bars = {str(bar["bar_id"]): bar for bar in analysis["bars"]}
    selected: list[dict[str, Any]] = []
    target_audit: list[dict[str, Any]] = []
    for target in plan["targets"]:
        match = match_bar_target(target, analysis["bars"], legend)
        if match.get("status") != "success":
            return {
                "status": str(match.get("status")),
                "stage": "target_grounding",
                "failed_target": target,
                "target_match": match,
            }
        bar = bars[str(match["target_bar_ids"][0])]
        selected.append(bar)
        target_audit.append({"semantic_target": target, "match": match, "bar_id": bar["bar_id"], "bbox": bar["bbox"], "point": [bar["target_x"], bar["target_y"]]})
    if len({str(bar["bar_id"]) for bar in selected}) != len(selected):
        return {"status": "targets_not_distinct", "stage": "target_grounding", "target_audit": target_audit}
    values = [float(analysis["axis"]["slope"]) * float(bar["target_y"]) + float(analysis["axis"]["intercept"]) for bar in selected]
    prediction, numeric_answer, reasoning = finish_measurement(row, plan, values)
    return {
        "status": "success",
        "stage": "answer",
        "geometry_values": values,
        "numeric_answer": numeric_answer,
        "prediction": prediction,
        "reasoning": reasoning,
        "target_audit": target_audit,
        "y_axis_ticks": analysis["axis"]["ticks"],
        "bar_audit": analysis,
    }


def run_line(
    reader: Any,
    image: Any,
    row: dict[str, Any],
    plan: dict[str, Any],
    cache: dict[tuple[str, str, str], dict[str, Any]],
    chart: dict[str, Any] | None = None,
    series_cache: dict[str, Any] | None = None,
    repair_geometry: bool = False,
) -> dict[str, Any]:
    from phase2_natural_bar import answer_from_geometry
    from phase4_multiline_core import auto_multiline_pipeline
    from phase5_finmme_natural_line import geometry_question

    pipelines: list[dict[str, Any]] = []
    for target in plan["targets"]:
        key = (row["chart_id"], json.dumps(target, sort_keys=True))
        if key not in cache:
            if chart is not None:
                from phase8_geometry_adapter import read_line
                from phase4_series import detect_line_series_components
                if series_cache.get("chart_id") != row["chart_id"]:
                    series_cache.clear()
                    from phase8_geometry_adapter import detect_series
                    detector = detect_series if repair_geometry else detect_line_series_components
                    series_cache.update(chart_id=row["chart_id"], series=detector(image, chart["axis"]))
                cache[key] = read_line(image, {**chart, "single_series_hint": row.get("line_series_count_hint") == 1, "caption": row.get("caption", "")}, target, series_cache["series"])
            else:
                cache[key] = auto_multiline_pipeline(reader, image, geometry_question(target["x_label"], target["series"]))
        pipelines.append(cache[key])
    failed = [item for item in pipelines if item.get("status") != "success"]
    if failed:
        return {
            "status": " | ".join(str(item.get("status")) for item in pipelines),
            "stage": "line_geometry",
            "pipeline_audit": pipelines,
        }
    values = [item.get("auto_local_value") for item in pipelines]
    if any(value is None for value in values):
        return {"status": "line_value_missing", "stage": "line_geometry", "pipeline_audit": pipelines}
    numeric_values = [value if isinstance(value, dict) else float(value) for value in values]
    prediction, numeric_answer, reasoning = finish_measurement(row, plan, numeric_values)
    return {
        "status": "success",
        "stage": "answer",
        "geometry_values": numeric_values,
        "numeric_answer": numeric_answer,
        "prediction": prediction,
        "reasoning": reasoning,
        "target_audit": [
            {
                "semantic_target": target,
                "grounding": pipeline.get("grounding"),
                "series_resolution": pipeline.get("series_resolution"),
                "local_search_window": pipeline.get("local_search_window"),
                "geometry": pipeline.get("auto_local_geometry"),
            }
            for target, pipeline in zip(plan["targets"], pipelines)
        ],
        "y_axis_ticks": pipelines[0]["axis"]["ticks"],
        "pipeline_audit": pipelines,
    }


def run_geometry(args: argparse.Namespace) -> None:
    import cv2
    import easyocr
    import torch

    cv2.setNumThreads(args.threads)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(2)
    inputs = read_jsonl(args.inputs)
    plans = {str(row["sample_id"]): row for row in read_jsonl(args.plans)}
    charts = {row["chart_id"]: row for row in read_jsonl(args.cached_charts)} if args.cached_charts else {}
    completed = {str(row["sample_id"]): row for row in read_jsonl(args.output)} if args.output.exists() else {}
    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=str(args.model_dir),
        download_enabled=False,
        verbose=False,
    )
    line_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    series_cache: dict[str, Any] = {}
    repair_path = args.output.with_name("chart_ocr_repairs.jsonl")
    repaired = {row["chart_id"]: row for row in read_jsonl(repair_path)} if args.repair_geometry and repair_path.exists() else {}
    if args.repair_geometry:
        charts.update(repaired)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(inputs, 1):
            sample_id = str(row["sample_id"])
            row["evidence_only"] = args.evidence_only
            if sample_id in completed:
                continue
            planner_record = plans.get(sample_id)
            base = {
                "sample_id": sample_id,
                "chart_id": row["chart_id"],
                "chart_kind_hint": row["chart_kind_hint"],
            }
            if planner_record is None or planner_record.get("status") != "success":
                result = {**base, "status": "semantic_plan_missing_or_invalid", "stage": "semantic_planner"}
            else:
                plan = planner_record.get("plan") or {}
                from phase8_geometry_adapter import validate_plan
                plan, reason = validate_plan(plan, row["question"])
                if reason != "ok":
                    result = {**base, "status": "semantic_plan_rejected", "stage": "semantic_planner", "semantic_plan": plan, "status_reason": reason}
                else:
                    image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
                    if image is None:
                        result = {**base, "status": "image_read_failed", "stage": "image", "semantic_plan": plan}
                    else:
                        try:
                            chart = charts.get(row["chart_id"])
                            if args.repair_geometry and plan["chart_type"] == "line" and chart is not None and row["chart_id"] not in repaired:
                                from phase8_geometry_adapter import scalar, refresh_anchors
                                n_temporal = sum(scalar(a["label"]) is not None and scalar(a["label"])[1] == "temporal" for a in chart["x_axis_anchors"])
                                from phase8_geometry_adapter import ground_label
                                needs_anchor = any(ground_label(t["x_label"], chart["x_axis_anchors"])["status"] != "ok" for t in plan["targets"])
                                if n_temporal < 2 or needs_anchor:
                                    chart = refresh_anchors(reader, image, chart)
                                charts[row["chart_id"]] = repaired[row["chart_id"]] = chart
                                with repair_path.open("a", encoding="utf-8") as repair_log:
                                    repair_log.write(json.dumps(chart, ensure_ascii=False) + "\n")
                            detail = (
                                run_bar(reader, row, plan)
                                if plan["chart_type"] == "bar"
                                else run_line(reader, image, row, plan, line_cache, chart, series_cache, args.repair_geometry)
                            )
                            result = {**base, "semantic_plan": plan, **detail}
                        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
                            result = {
                                **base,
                                "status": "hybrid_execution_failed",
                                "stage": "reasoning",
                                "semantic_plan": plan,
                                "error": str(exc),
                            }
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            completed[sample_id] = result
            print(json.dumps({"processed": index, "total": len(inputs), "sample_id": sample_id, "status": result["status"]}), flush=True)


def aggregate(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    completed = [row for row in rows if row["status"] == "success"]
    return {
        "scope": scope,
        "questions": len(rows),
        "geometry_coverage": len(completed) / len(rows) if rows else 0.0,
        "raw_vlm_accuracy": sum(row["raw_vlm_correct"] for row in rows) / len(rows) if rows else 0.0,
        "hybrid_accuracy_all": sum(row["hybrid_correct"] for row in rows) / len(rows) if rows else 0.0,
        "hybrid_accuracy_completed": sum(row["hybrid_correct"] for row in completed) / len(completed) if completed else 0.0,
        "selective_accuracy": sum(row["selective_correct"] for row in rows) / len(rows) if rows else 0.0,
    }


def evaluate(args: argparse.Namespace) -> None:
    from finmme_geometry_qa import correctness

    inputs = {str(row["sample_id"]): row for row in read_jsonl(args.inputs)}
    gold = {str(row["sample_id"]): row for row in read_jsonl(args.gold)}
    baseline = {str(row["sample_id"]): row for row in read_jsonl(args.baseline)}
    predictions = {str(row["sample_id"]): row for row in read_jsonl(args.predictions)}
    plans = {str(row["sample_id"]): row for row in read_jsonl(args.plans)}
    raw_all = read_jsonl(args.raw_metrics)
    old_phase7: dict[str, dict[str, str]] = {}
    if args.phase7_samples and args.phase7_samples.exists():
        with args.phase7_samples.open(encoding="utf-8", newline="") as handle:
            old_phase7 = {str(row["sample_id"]): row for row in csv.DictReader(handle)}
    output: list[dict[str, Any]] = []

    for sample_id, source in inputs.items():
        prediction = predictions.get(sample_id, {"status": "missing_prediction", "stage": "runtime"})
        plan_record = plans.get(sample_id, {})
        plan = plan_record.get("plan") or {}
        answer = gold[sample_id]
        raw = baseline[sample_id]
        hybrid_correct = prediction.get("status") == "success" and correctness(
            source["question_type"], prediction.get("prediction"), answer.get("reference"), answer.get("tolerance")
        )
        selective_correct = int(hybrid_correct) if prediction.get("status") == "success" else int(raw["raw_vlm_correct"])
        old = old_phase7.get(sample_id)
        output.append(
            {
                "sample_id": sample_id,
                "chart_id": source["chart_id"],
                "chart_kind_hint": source["chart_kind_hint"],
                "series_tier_hint": "bar" if source["chart_kind_hint"]=="bar" else "single_line" if source["line_series_count_hint"]==1 else "multi_line",
                "question_type": source["question_type"],
                "question": source["question"],
                "planner_status": plan_record.get("status", "missing"),
                "vlm_chart_type": plan.get("chart_type", ""),
                "vlm_operation": plan.get("operation", ""),
                "vlm_direct_value_labels": plan.get("has_direct_value_labels", ""),
                "vlm_targets": json.dumps(plan.get("targets", []), ensure_ascii=False),
                "vlm_route_geometry": int(prediction.get("stage") not in {"semantic_planner", "runtime"} and bool(prediction.get("semantic_plan"))),
                "status_reason": prediction.get("status_reason", ""),
                "status": prediction.get("status"),
                "failure_stage": "answered_wrong_unattributed" if prediction.get("status") == "success" and not hybrid_correct else prediction.get("stage", ""),
                "target_audit": json.dumps(prediction.get("target_audit", []), ensure_ascii=False),
                "y_axis_ticks": json.dumps(prediction.get("y_axis_ticks", []), ensure_ascii=False),
                "geometry_values": json.dumps(prediction.get("geometry_values", [])),
                "numeric_answer": prediction.get("numeric_answer", ""),
                "prediction": prediction.get("prediction", ""),
                "gold": answer.get("reference"),
                "tolerance": answer.get("tolerance"),
                "hybrid_correct": int(hybrid_correct),
                "raw_vlm_correct": int(raw["raw_vlm_correct"]),
                "selective_correct": selective_correct,
                "phase7_rule_correct": "" if old is None else old.get("geometry_correct", ""),
                "phase7_rule_status": "" if old is None else old.get("status", ""),
            }
        )

    metrics = [aggregate(output, "broad_candidates")]
    planner_accepted = [row for row in output if row["vlm_route_geometry"] and row["vlm_direct_value_labels"] is not True]
    metrics.append(aggregate(planner_accepted, "vlm_planner_accepted"))
    frozen = [row for row in output if row["sample_id"] in old_phase7]
    if frozen:
        frozen_metric = aggregate(frozen, "frozen_phase7_167_vlm_semantics")
        frozen_metric["phase7_rule_accuracy_all"] = sum(int(row["phase7_rule_correct"] or 0) for row in frozen) / len(frozen)
        metrics.append(frozen_metric)
    for field in ("chart_kind_hint", "series_tier_hint", "question_type", "vlm_operation"):
        for value in sorted({str(row[field]) for row in output if str(row[field])}):
            metrics.append(aggregate([row for row in output if str(row[field]) == value], f"{field}:{value}"))

    official_total = sum(int(bool(row.get("official_code_correct"))) for row in raw_all)
    paper_total = sum(float(row.get("paper_question_score", 0.0) or 0.0) for row in raw_all)
    official_hard, official_selective = official_total, official_total
    paper_hard, paper_selective = paper_total, paper_total
    raw_official = {str(row["sample_id"]): int(bool(row.get("official_code_correct"))) for row in raw_all}
    raw_paper = {str(row["sample_id"]): float(row.get("paper_question_score", 0.0) or 0.0) for row in raw_all}
    for row in output:
        old_official = raw_official[row["sample_id"]]
        old_paper = raw_paper[row["sample_id"]]
        official_hard += row["hybrid_correct"] - old_official
        official_selective += row["selective_correct"] - old_official
        paper_hard += row["hybrid_correct"] - old_paper
        if row["status"] == "success":
            paper_selective += row["hybrid_correct"] - old_paper
    total = len(raw_all)
    metrics.extend(
        [
            {"scope": "full_finmme_raw_vlm_official", "questions": total, "accuracy": official_total / total},
            {"scope": "full_finmme_hybrid_hard_official", "questions": total, "accuracy": official_hard / total},
            {"scope": "full_finmme_hybrid_selective_official", "questions": total, "accuracy": official_selective / total},
            {"scope": "full_finmme_raw_vlm_paper_protocol", "questions": total, "accuracy": paper_total / total},
            {"scope": "full_finmme_hybrid_hard_paper_protocol", "questions": total, "accuracy": paper_hard / total},
            {"scope": "full_finmme_hybrid_selective_paper_protocol", "questions": total, "accuracy": paper_selective / total},
        ]
    )
    write_csv(args.output_dir / "phase8_hybrid_samples.csv", output)
    write_csv(args.output_dir / "phase8_hybrid_metrics.csv", metrics)
    by_id = {row["sample_id"]: row for row in output}
    full_routing = []
    for raw in raw_all:
        sample_id = str(raw["sample_id"])
        candidate = by_id.get(sample_id)
        use_geometry = candidate is not None and candidate["status"] == "success"
        raw_correct = int(bool(raw.get("official_code_correct")))
        full_routing.append({"sample_id":sample_id, "in_phase8_cohort":int(candidate is not None),
            "route":"geometry" if use_geometry else "raw_vlm", "geometry_status":"not_routed" if candidate is None else candidate["status"],
            "raw_vlm_correct":raw_correct, "system_correct":candidate["hybrid_correct"] if use_geometry else raw_correct})
    write_csv(args.output_dir / "phase8_full_dataset_routing.csv", full_routing)
    if args.full_audit:
        with args.full_audit.open(newline="", encoding="utf-8") as handle:
            audit_by_id={row["sample_id"]:row for row in csv.DictReader(handle)}
        strata=[]
        for field in ("value_label_tier", "axis_tier", "type_hint", "question_type"):
            groups=sorted({row.get(field,"unknown") for row in audit_by_id.values()})
            for value in groups:
                subset=[row for row in full_routing if audit_by_id.get(row["sample_id"],{}).get(field)==value]
                if subset:
                    strata.append({"field":field,"stratum":value,"questions":len(subset),"geometry_answers":sum(row["route"]=="geometry" for row in subset),
                        "raw_vlm_accuracy":sum(row["raw_vlm_correct"] for row in subset)/len(subset),"selective_accuracy":sum(row["system_correct"] for row in subset)/len(subset)})
        write_csv(args.output_dir/"phase8_full_dataset_strata.csv",strata)
    summary = {
        "broad_candidates": len(output),
        "planner_parse_success": sum(row["planner_status"] == "success" for row in output),
        "planner_route_geometry": len(planner_accepted),
        "geometry_success": sum(row["status"] == "success" for row in output),
        "hybrid_correct": sum(row["hybrid_correct"] for row in output),
        "failure_stages": dict(Counter(row["failure_stage"] or "answered_wrong" for row in output if not row["hybrid_correct"])),
    }
    (args.output_dir / "phase8_hybrid_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--records", type=Path, required=True)
    prep.add_argument("--chart-scan", type=Path, required=True)
    prep.add_argument("--raw-metrics", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    run = commands.add_parser("run-geometry")
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--plans", type=Path, required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--cached-charts", type=Path)
    run.add_argument("--repair-geometry", action="store_true")
    run.add_argument("--evidence-only", action="store_true")
    score = commands.add_parser("evaluate")
    score.add_argument("--inputs", type=Path, required=True)
    score.add_argument("--plans", type=Path, required=True)
    score.add_argument("--gold", type=Path, required=True)
    score.add_argument("--baseline", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--raw-metrics", type=Path, required=True)
    score.add_argument("--phase7-samples", type=Path)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--full-audit", type=Path)
    return root


def main() -> None:
    args = parser().parse_args()
    {"prepare": prepare, "run-geometry": run_geometry, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
