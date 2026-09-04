#!/usr/bin/env python3
"""Report Geometry accuracy with construction failures counted as incorrect."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()

    attempts = [
        row
        for row in read_jsonl(args.project / "audit" / "label_attempts.jsonl")
        if row["status"] != "not_attempted_due_limit"
    ]
    samples = read_jsonl(args.project / "audit" / "geometry_samples.jsonl")
    sample_by_id = {row["sample_id"]: row for row in samples}
    output: list[dict] = []

    for chart_type in ("all", "bar", "line"):
        selected = attempts if chart_type == "all" else [row for row in attempts if row["chart_type"] == chart_type]
        recovered = [sample_by_id[row["sample_id"]] for row in selected if row["status"] == "accepted"]
        absolute_errors = [float(row["absolute_error"]) for row in recovered]
        normalized_errors = [float(row["axis_normalized_error"]) for row in recovered]
        for threshold in (0.01, 0.02, 0.05):
            correct = sum(error <= threshold for error in normalized_errors)
            output.append(
                {
                    "chart_type": chart_type,
                    "candidate_labels": len(selected),
                    "covered_labels": len(recovered),
                    "coverage": len(recovered) / len(selected) if selected else 0.0,
                    "charts": len({row["chart_id"] for row in selected}),
                    "covered_charts": len({row["chart_id"] for row in recovered}),
                    "mae_covered": mean(absolute_errors) if absolute_errors else "",
                    "median_absolute_error_covered": median(absolute_errors) if absolute_errors else "",
                    "axis_error_threshold": threshold,
                    "correct_labels": correct,
                    "accuracy_all": correct / len(selected) if selected else 0.0,
                    "accuracy_covered": correct / len(recovered) if recovered else 0.0,
                }
            )

    output_path = args.project / "results" / "geometry_scale_accuracy.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(json.dumps({"candidate_labels": len(attempts), "recovered": len(samples), "output": str(output_path)}))


if __name__ == "__main__":
    main()
