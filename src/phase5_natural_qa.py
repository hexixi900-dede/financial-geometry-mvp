"""FinChart-Bench natural no-label Bar/Line pilot.

The runtime command reads only phase5_inputs.csv (image + question). Gold is
kept in a separate file and is joined only by the evaluate command.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2
import easyocr

from geometry_core import axis_localizer, interpolate_pixel_to_value, token_from_dict
from phase2_natural_bar import (
    analyze_chart,
    answer_from_geometry,
    parse_operation,
    semantic_plan,
)
from phase3_line_core import (
    auto_geometry_pipeline,
    detect_dominant_line_series,
    geometry_value,
    prepare_auto_geometry_chart,
)


TARGET_RE = re.compile(
    r"\b(?:[1-4]\s*[Qq](?:\s*FY)?\s*\d{2,4}|[Qq]\s*[1-4]\s*(?:FY\s*)?\d{2,4}|"
    r"FY\s*\d{2,4}|(?:19|20)\d{2}[EeFf]?|"
    r"\d{1,2}/\d{1,2}/\d{2,4}|[Qq][1-9]|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*[- '/]\d{2,4})\b",
    re.IGNORECASE,
)
COMPLEX_QUESTION_RE = re.compile(
    r"\b(?:highest|lowest|maximum|minimum|peak|average|mean|total|sum|among all|how many subjects|"
    r"combined|cumulative)\b",
    re.IGNORECASE,
)


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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def base_image_name(question_image: str) -> str:
    stem, suffix = Path(question_image).stem, Path(question_image).suffix
    return stem.rsplit("_", 1)[0] + suffix


def operation_for_question(question: str) -> str | None:
    lowered = question.lower()
    if "percentage point" not in lowered and re.search(
        r"\b(?:by what percentage|what percentage did|percentage (?:increase|decrease|change))\b",
        lowered,
    ):
        return "growth_rate"
    return parse_operation(question)


def simple_question(question: str, operation: str) -> bool:
    if COMPLEX_QUESTION_RE.search(question):
        return False
    required = 2 if operation in {"difference", "growth_rate"} else 1
    targets = x_targets(question, 10)
    if len(targets) != required:
        return False
    residual = TARGET_RE.sub(" ", question)
    return re.search(r"\d", residual) is None


def prepare(args: argparse.Namespace) -> None:
    records = json.loads(args.metadata.read_text(encoding="utf-8"))
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        operation = operation_for_question(str(record.get("question", "")))
        try:
            float(record.get("answer"))
        except (TypeError, ValueError):
            continue
        if operation and simple_question(str(record.get("question", "")), operation):
            by_image[base_image_name(record["image"])].append({**record, "operation": operation})

    available = [name for name in sorted(by_image) if (args.image_dir / name).exists()]
    if len(available) > args.max_images:
        available = [
            available[round(index * (len(available) - 1) / max(args.max_images - 1, 1))]
            for index in range(args.max_images)
        ]
    inputs: list[dict[str, Any]] = []
    gold: list[dict[str, Any]] = []
    for image_name in available:
        image_path = args.image_dir / image_name
        for record in by_image[image_name]:
            sample_id = f"finchart_qa_{len(inputs) + 1:05d}"
            inputs.append(
                {
                    "sample_id": sample_id,
                    "image_id": Path(image_name).stem,
                    "image_path": str(image_path),
                    "question": record["question"],
                    "operation": record["operation"],
                }
            )
            gold.append(
                {
                    "sample_id": sample_id,
                    "gold": float(record["answer"]),
                    "source_question_image": record["image"],
                }
            )
            if len(inputs) >= args.max_questions:
                break
        if len(inputs) >= args.max_questions:
            break

    write_csv(args.output_dir / "phase5_inputs.csv", inputs)
    write_csv(args.output_dir / "phase5_gold.csv", gold)
    summary = {"images": len({row["image_id"] for row in inputs}), "questions": len(inputs), "operations": dict(Counter(r["operation"] for r in inputs))}
    (args.output_dir / "phase5_prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def make_reader(model_dir: Path) -> easyocr.Reader:
    return easyocr.Reader(["en"], gpu=False, model_storage_directory=str(model_dir), download_enabled=False)


def x_targets(question: str, required: int) -> list[str]:
    targets: list[str] = []
    for match in TARGET_RE.finditer(question):
        value = re.sub(r"\s+", "", match.group(0))
        if value.lower() not in {item.lower() for item in targets}:
            targets.append(value)
    return targets[:required]


def chart_has_plot_numbers(analysis: dict[str, Any], image: Any) -> tuple[bool, dict[str, Any] | None]:
    tokens = [token_from_dict(item) for item in analysis.get("ocr_tokens", [])]
    height, width = image.shape[:2]
    axis = analysis.get("axis") or axis_localizer(tokens, width, height)
    if axis is None:
        return True, None
    axis_indices = {int(value) for value in axis["token_indices"]}
    left, top, right, bottom = map(float, axis["plot_bbox"])
    interior = [
        token
        for token in tokens
        if token.index not in axis_indices
        and token.numeric_value is not None
        and token.confidence >= 0.50
        and left <= token.cx <= right
        and top <= token.cy <= bottom
    ]
    return bool(interior), axis


def run_geometry(args: argparse.Namespace) -> None:
    reader = make_reader(args.model_dir)
    rows = read_csv(args.inputs)
    outputs: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    line_cache: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, 1):
        if index == 1 or index % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "starting": index,
                        "total": len(rows),
                        "success_so_far": sum(item["status"] == "success" for item in outputs),
                    }
                ),
                flush=True,
            )
        image_path = Path(row["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        base = {"sample_id": row["sample_id"], "image_id": row["image_id"], "question": row["question"], "operation": row["operation"]}
        if image is None:
            outputs.append({**base, "status": "image_read_failed"})
            continue

        if row["image_id"] not in cache:
            cache[row["image_id"]] = analyze_chart(reader, row["image_id"], image_path, {"sloped_segments": 0})
        bar = cache[row["image_id"]]
        if bar.get("status") == "eligible_natural_no_label_bar":
            line_component, _ = detect_dominant_line_series(image, bar["axis"])
            if line_component is not None:
                outputs.append({**base, "chart_kind": "bar", "status": "bar_line_combination_out_of_scope"})
                continue
            plan = semantic_plan(row["question"], "", bar["bars"])
            if plan.get("status") != "success":
                outputs.append({**base, "chart_kind": "bar", "status": plan["status"]})
                continue
            chosen = {item["bar_id"]: item for item in bar["bars"]}
            values = [interpolate_pixel_to_value(float(chosen[item]["target_y"]), bar["axis"]) for item in plan["target_bar_ids"]]
            try:
                answer, _, reasoning = answer_from_geometry("numerical", "", plan["operation"], values, row["question"])
            except (ValueError, ZeroDivisionError) as exc:
                outputs.append({**base, "chart_kind": "bar", "status": "reasoning_failed", "reason": str(exc)})
                continue
            outputs.append(
                {
                    **base,
                    "chart_kind": "bar",
                    "status": "success",
                    "semantic_targets": " | ".join(plan["target_categories"]),
                    "target_ids": " | ".join(plan["target_bar_ids"]),
                    "geometry_values": " | ".join(f"{value:.12g}" for value in values),
                    "prediction": answer,
                    "axis_span": bar["axis"]["axis_span"],
                    "reasoning": reasoning["formula"],
                }
            )
            continue

        has_numbers, axis = chart_has_plot_numbers(bar, image)
        if has_numbers:
            outputs.append({**base, "chart_kind": "line", "status": "not_natural_no_label_or_axis_unrecoverable"})
            continue
        required = 2 if row["operation"] in {"difference", "growth_rate"} else 1
        targets = x_targets(row["question"], required)
        if len(targets) != required:
            outputs.append({**base, "chart_kind": "line", "status": "x_targets_not_parseable"})
            continue
        cached_tokens = [token_from_dict(item) for item in bar.get("ocr_tokens", [])]
        if row["image_id"] not in line_cache:
            line_cache[row["image_id"]] = prepare_auto_geometry_chart(
                reader,
                image,
                precomputed_tokens=cached_tokens,
                precomputed_axis=axis,
            )
        prepared_line = line_cache[row["image_id"]]
        pipelines = [
            auto_geometry_pipeline(
                reader,
                image,
                f"What is the value for {target}?",
                precomputed_tokens=cached_tokens,
                precomputed_axis=axis,
                prepared_chart=prepared_line,
            )
            for target in targets
        ]
        if any(item.get("status") != "ok" for item in pipelines):
            reason = " | ".join(str(item.get("status")) for item in pipelines)
            outputs.append({**base, "chart_kind": "line", "status": "line_geometry_failed", "reason": reason})
            continue
        if any(int((item.get("series") or {}).get("comparable_distinct_series_count", 1)) != 1 for item in pipelines):
            outputs.append({**base, "chart_kind": "line", "status": "multi_line_out_of_primary_scope"})
            continue
        values = [geometry_value(item.get("auto_local_geometry"), item["axis"]) for item in pipelines]
        if any(value is None for value in values):
            outputs.append({**base, "chart_kind": "line", "status": "line_point_unrecoverable"})
            continue
        numeric_values = [float(value) for value in values if value is not None]
        try:
            answer, _, reasoning = answer_from_geometry("numerical", "", row["operation"], numeric_values, row["question"])
        except (ValueError, ZeroDivisionError) as exc:
            outputs.append({**base, "chart_kind": "line", "status": "reasoning_failed", "reason": str(exc)})
            continue
        outputs.append(
            {
                **base,
                "chart_kind": "line",
                "status": "success",
                "semantic_targets": " | ".join(targets),
                "target_ids": " | ".join(str(item["grounding"].get("matched_detected_label") or item["grounding"].get("target_semantic_label")) for item in pipelines),
                "geometry_values": " | ".join(f"{value:.12g}" for value in numeric_values),
                "prediction": answer,
                "axis_span": max(float(item["axis"]["axis_span"]) for item in pipelines),
                "reasoning": reasoning["formula"],
            }
        )
    write_csv(args.output, outputs)
    print(json.dumps({"rows": len(outputs), "success": sum(row["status"] == "success" for row in outputs), "statuses": dict(Counter(row["status"] for row in outputs))}, sort_keys=True))


def parse_float(value: str | None) -> float | None:
    try:
        number = float(value) if value not in {None, ""} else None
    except ValueError:
        return None
    return number if number is not None and math.isfinite(number) else None


def evaluate(args: argparse.Namespace) -> None:
    predictions = {row["sample_id"]: row for row in read_csv(args.predictions)}
    gold = {row["sample_id"]: row for row in read_csv(args.gold)}
    tolerance_reference = (
        {row["sample_id"]: row for row in read_csv(args.tolerance_reference)}
        if args.tolerance_reference
        else {}
    )
    joined: list[dict[str, Any]] = []
    for sample_id, target in gold.items():
        row = predictions.get(sample_id, {"sample_id": sample_id, "status": "missing_prediction"})
        prediction = parse_float(row.get("prediction"))
        reference = float(target["gold"])
        axis_span = parse_float(row.get("axis_span")) or 0.0
        frozen_tolerance = parse_float(tolerance_reference.get(sample_id, {}).get("tolerance"))
        tolerance = frozen_tolerance or max(0.05 * abs(reference), 0.02 * axis_span, 1e-6)
        error = None if prediction is None else abs(prediction - reference)
        relative_error = None if error is None or abs(reference) < 1e-12 else error / abs(reference)
        axis_normalized_error = None if error is None or axis_span <= 0 else error / axis_span
        joined.append(
            {
                **row,
                "gold": reference,
                "tolerance": tolerance,
                "absolute_error": error,
                "relative_error": relative_error,
                "axis_normalized_error": axis_normalized_error,
                "correct": int(error is not None and error <= tolerance),
            }
        )
    write_csv(args.output_dir / "phase5_samples.csv", joined)

    metrics: list[dict[str, Any]] = []
    for kind in ["all", "bar", "line"]:
        rows = joined if kind == "all" else [row for row in joined if row.get("chart_kind") == kind]
        covered = [row for row in rows if parse_float(row.get("prediction")) is not None]
        errors = [float(row["absolute_error"]) for row in covered]
        relative_errors = [float(row["relative_error"]) for row in covered if parse_float(row.get("relative_error")) is not None]
        normalized_errors = [float(row["axis_normalized_error"]) for row in covered if parse_float(row.get("axis_normalized_error")) is not None]
        metrics.append(
            {
                "chart_kind": kind,
                "samples": len(rows),
                "covered": len(covered),
                "coverage": len(covered) / len(rows) if rows else 0.0,
                "accuracy": sum(int(row["correct"]) for row in rows) / len(rows) if rows else 0.0,
                "conditional_accuracy": sum(int(row["correct"]) for row in covered) / len(covered) if covered else 0.0,
                "mae": mean(errors) if errors else "",
                "median_absolute_error": median(errors) if errors else "",
                "mean_relative_error": mean(relative_errors) if relative_errors else "",
                "mean_axis_normalized_error": mean(normalized_errors) if normalized_errors else "",
            }
        )
    write_csv(args.output_dir / "phase5_metrics.csv", metrics)
    print(json.dumps(metrics, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--metadata", type=Path, required=True)
    prep.add_argument("--image-dir", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--max-images", type=int, default=100)
    prep.add_argument("--max-questions", type=int, default=200)
    run = commands.add_parser("run")
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--model-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--progress-every", type=int, default=10)
    score = commands.add_parser("evaluate")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--gold", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--tolerance-reference", type=Path)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run_geometry(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
