#!/usr/bin/env python3
"""Summarize existing Geometry results as thresholded accuracy.

This script only reads saved sample-level CSV files. It does not rerun OCR,
geometry recovery, calibration, or any GPU model.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median


THRESHOLDS = (0.01, 0.02, 0.05)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def add_summary(
    output: list[dict[str, object]],
    *,
    dataset: str,
    scope: str,
    chart_type: str,
    method: str,
    rows: list[dict[str, str]],
    absolute_error_column: str,
    normalized_error_column: str,
) -> None:
    errors = [
        (number(row, absolute_error_column), number(row, normalized_error_column))
        for row in rows
    ]
    covered = [(absolute, normalized) for absolute, normalized in errors if absolute is not None and normalized is not None]

    for threshold in THRESHOLDS:
        correct = sum(normalized <= threshold for _, normalized in covered)
        total = len(rows)
        covered_count = len(covered)
        output.append(
            {
                "dataset": dataset,
                "scope": scope,
                "chart_type": chart_type,
                "method": method,
                "total_samples": total,
                "covered_samples": covered_count,
                "coverage": covered_count / total if total else 0.0,
                "mae": mean(absolute for absolute, _ in covered) if covered else "",
                "median_absolute_error": median(absolute for absolute, _ in covered) if covered else "",
                "axis_error_threshold": threshold,
                "correct_samples": correct,
                "accuracy_all": correct / total if total else 0.0,
                "accuracy_covered": correct / covered_count if covered_count else 0.0,
            }
        )


def phase1_summary(path: Path, output: list[dict[str, object]]) -> None:
    rows = read_rows(path)
    methods = (
        ("raw_geometry", "raw_absolute_error", "raw_axis_normalized_error"),
        ("calibrated_geometry", "calibrated_absolute_error", "calibrated_axis_normalized_error"),
    )
    for scope in ("all", "test"):
        scoped = rows if scope == "all" else [row for row in rows if row.get("split") == scope]
        for chart_type in ("all", "bar", "line"):
            selected = scoped if chart_type == "all" else [row for row in scoped if row.get("chart_type") == chart_type]
            for method, absolute_column, normalized_column in methods:
                add_summary(
                    output,
                    dataset="phase1_masked_value",
                    scope=scope,
                    chart_type=chart_type,
                    method=method,
                    rows=selected,
                    absolute_error_column=absolute_column,
                    normalized_error_column=normalized_column,
                )


def phase3_summary(path: Path, output: list[dict[str, object]]) -> None:
    rows = read_rows(path)
    methods = (
        ("oracle_x_geometry", "oracle_x_absolute_error", "oracle_x_axis_normalized_error"),
        ("auto_x_column", "auto_x_column_absolute_error", "auto_x_column_axis_normalized_error"),
        ("auto_x_local_line_fit", "auto_x_local_absolute_error", "auto_x_local_axis_normalized_error"),
    )
    for method, absolute_column, normalized_column in methods:
        add_summary(
            output,
            dataset="phase3_line_2d",
            scope="all",
            chart_type="line",
            method=method,
            rows=rows,
            absolute_error_column=absolute_column,
            normalized_error_column=normalized_column,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1", type=Path, default=Path("results/calibrated_predictions.csv"))
    parser.add_argument("--phase3", type=Path, default=Path("results/phase3_line_samples.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/geometry_accuracy_summary.csv"))
    args = parser.parse_args()

    output: list[dict[str, object]] = []
    phase1_summary(args.phase1, output)
    phase3_summary(args.phase3, output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
