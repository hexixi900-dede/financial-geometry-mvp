"""Lightweight acceptance check for the uncommitted Phase 4 prototype."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean, median


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows: list[dict[str, str]], field: str) -> dict[str, float | int]:
    eligible = [row for row in rows if row.get(field)]
    errors = [abs(float(row[field]) - float(row["pseudo_gold"])) for row in eligible]
    return {
        "total": len(rows),
        "eligible": len(eligible),
        "coverage": len(eligible) / len(rows) if rows else 0.0,
        "mae": mean(errors) if errors else float("nan"),
        "median_absolute_error": median(errors) if errors else float("nan"),
    }


def choose(rows: list[dict[str, str]], kind: str, success: bool, count: int) -> list[dict[str, str]]:
    expected = "success" if success else None
    selected = [
        row
        for row in rows
        if row["chart_kind"] == kind
        and ((row["auto_status"] == expected) if success else (row["auto_status"] != "success"))
    ]
    return selected[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()

    rows = read_csv(project / "results" / "phase4_line_samples.csv")
    splits = read_csv(project / "results" / "phase4_chart_split.csv")
    train = {row["chart_id"] for row in splits if row["split"] == "train"}
    test = {row["chart_id"] for row in splits if row["split"] == "test"}
    hidden_violations = sum(int(row["parser_used_hidden_bbox_or_gold"]) != 0 for row in rows)

    selected = (
        choose(rows, "multi_line", True, 16)
        + choose(rows, "single_line", True, 4)
        + choose(rows, "multi_line", False, 8)
        + choose(rows, "single_line", False, 2)
    )
    overlay_source = project / "phase4_debug_overlays"
    overlay_target = project / "examples" / "phase4_debug_overlays"
    overlay_target.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    missing_overlays = 0
    for row in selected:
        source = overlay_source / f"{row['sample_id']}_auto_local.png"
        target = overlay_target / source.name
        if source.exists():
            shutil.copy2(source, target)
        else:
            missing_overlays += 1
        manifest.append(
            {
                "sample_id": row["sample_id"],
                "chart_id": row["chart_id"],
                "chart_kind": row["chart_kind"],
                "auto_status": row["auto_status"],
                "status_reason": row["auto_status_reason"],
                "overlay_path": str(target.relative_to(project)),
                "manual_target_ok": "",
                "manual_geometry_ok": "",
                "notes": "",
            }
        )

    write_csv(project / "results" / "phase4_acceptance_samples.csv", manifest)
    multi = [row for row in rows if row["chart_kind"] == "multi_line"]
    summary = {
        "status": "automated_checks_passed_manual_overlay_review_pending"
        if not (train & test) and hidden_violations == 0 and missing_overlays == 0
        else "failed",
        "samples": len(rows),
        "charts": len({row["chart_id"] for row in rows}),
        "chart_kinds": dict(Counter(row["chart_kind"] for row in rows)),
        "runtime_statuses": dict(Counter(row["auto_status"] for row in rows)),
        "train_test_chart_overlap": len(train & test),
        "runtime_hidden_input_violations": hidden_violations,
        "selected_overlays": len(selected),
        "missing_selected_overlays": missing_overlays,
        "multi_line_auto_column": metrics(multi, "auto_column_value"),
        "multi_line_auto_local": metrics(multi, "auto_local_value"),
    }
    output = project / "results" / "phase4_acceptance.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

