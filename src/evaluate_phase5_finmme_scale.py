"""Evaluate new FinMME geometry samples as an external Phase 1 holdout."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from calibrate_geometry import feature_vector


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def training_biases(old_rows: list[dict[str, Any]], split_path: Path) -> dict[str, float]:
    split = {row["chart_id"]: row["split"] for row in read_csv(split_path)}
    normalized: dict[str, list[float]] = defaultdict(list)
    for row in old_rows:
        if split.get(str(row["chart_id"])) != "train":
            continue
        span = max(float(row["axis"]["axis_span"]), 1e-12)
        error = (float(row["pseudo_gold"]) - float(row["raw_geometry_value"])) / span
        normalized["all"].append(error)
        normalized[str(row["geometry"]["chart_type"])].append(error)
    result: dict[str, float] = {}
    for kind, values in normalized.items():
        result[f"{kind}_mean"] = mean(values)
        result[f"{kind}_median"] = median(values)
    return result


def learned_prediction(row: dict[str, Any], model: dict[str, Any]) -> float:
    features = feature_vector(row)
    center = np.asarray(model["feature_mean"], dtype=float)
    scale = np.asarray(model["feature_scale"], dtype=float)
    coefficients = np.asarray(model["coefficients"], dtype=float)
    normalized = (features - center) / np.where(scale < 1e-12, 1.0, scale)
    predicted_normalized_error = float(coefficients[0] + normalized @ coefficients[1:])
    return float(row["raw_geometry_value"]) + predicted_normalized_error * float(row["axis"]["axis_span"])


def metric(rows: list[dict[str, Any]], method: str, chart_type: str) -> dict[str, Any]:
    selected = rows if chart_type == "all" else [row for row in rows if row["chart_type"] == chart_type]
    errors = [float(row[f"{method}_absolute_error"]) for row in selected]
    normalized = [float(row[f"{method}_axis_normalized_error"]) for row in selected]
    return {
        "scope": "new_external_holdout",
        "chart_type": chart_type,
        "method": method,
        "samples": len(selected),
        "charts": len({row["chart_id"] for row in selected}),
        "mae": mean(errors) if errors else "",
        "median_absolute_error": median(errors) if errors else "",
        "mean_axis_normalized_error": mean(normalized) if normalized else "",
        "median_axis_normalized_error": median(normalized) if normalized else "",
        "within_2pct_axis": mean(value <= 0.02 for value in normalized) if normalized else "",
        "within_5pct_axis": mean(value <= 0.05 for value in normalized) if normalized else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-samples", type=Path, required=True)
    parser.add_argument("--old-samples", type=Path, required=True)
    parser.add_argument("--old-split", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    new_rows = read_jsonl(args.new_samples)
    old_rows = read_jsonl(args.old_samples)
    old_ids = {str(row["chart_id"]) for row in old_rows}
    overlap = old_ids & {str(row["chart_id"]) for row in new_rows}
    if overlap:
        raise RuntimeError(f"new samples overlap Phase 1 charts: {len(overlap)}")
    model = json.loads(args.model.read_text(encoding="utf-8"))
    biases = training_biases(old_rows, args.old_split)

    methods = ["raw", "learned", "global_mean", "global_median", "type_mean", "type_median"]
    predictions: list[dict[str, Any]] = []
    for row in new_rows:
        chart_type = str(row["geometry"]["chart_type"])
        raw = float(row["raw_geometry_value"])
        gold = float(row["pseudo_gold"])
        span = max(float(row["axis"]["axis_span"]), 1e-12)
        values = {
            "raw": raw,
            "learned": learned_prediction(row, model),
            "global_mean": raw + biases["all_mean"] * span,
            "global_median": raw + biases["all_median"] * span,
            "type_mean": raw + biases.get(f"{chart_type}_mean", biases["all_mean"]) * span,
            "type_median": raw + biases.get(f"{chart_type}_median", biases["all_median"]) * span,
        }
        output: dict[str, Any] = {
            "sample_id": row["sample_id"],
            "chart_id": row["chart_id"],
            "chart_type": chart_type,
            "pseudo_gold": gold,
            "axis_span": span,
        }
        for method, value in values.items():
            error = abs(value - gold)
            output[f"{method}_value"] = value
            output[f"{method}_absolute_error"] = error
            output[f"{method}_axis_normalized_error"] = error / span
        predictions.append(output)

    metric_rows = [
        metric(predictions, method, chart_type)
        for chart_type in ["all", "bar", "line"]
        for method in methods
    ]
    combined = old_rows + new_rows
    combined_metrics = []
    for chart_type in ["all", "bar", "line"]:
        selected = combined if chart_type == "all" else [row for row in combined if row["geometry"]["chart_type"] == chart_type]
        errors = [float(row["absolute_error"]) for row in selected]
        normalized = [float(row["axis_normalized_error"]) for row in selected]
        combined_metrics.append(
            {
                "scope": "combined_descriptive_only",
                "chart_type": chart_type,
                "method": "raw",
                "samples": len(selected),
                "charts": len({str(row["chart_id"]) for row in selected}),
                "mae": mean(errors) if errors else "",
                "median_absolute_error": median(errors) if errors else "",
                "mean_axis_normalized_error": mean(normalized) if normalized else "",
                "median_axis_normalized_error": median(normalized) if normalized else "",
            }
        )

    write_csv(args.output_dir / "phase5_holdout_predictions.csv", predictions)
    write_csv(args.output_dir / "phase5_holdout_metrics.csv", metric_rows)
    write_csv(args.output_dir / "phase5_combined_descriptive_metrics.csv", combined_metrics)
    (args.output_dir / "phase5_frozen_biases.json").write_text(
        json.dumps(biases, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"new_samples": len(new_rows), "phase1_overlap": 0, "biases": biases}, sort_keys=True))


if __name__ == "__main__":
    main()
