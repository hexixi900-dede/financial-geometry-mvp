from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "question",
    "semantic_target",
    "detected_x_axis_labels",
    "selected_bar_ids",
    "selected_bar_bboxes",
    "y_axis_ticks",
    "bar_tops",
    "raw_geometry_value",
    "calibrated_value",
    "raw_geometry_prediction",
    "calibrated_prediction",
    "raw_vlm_prediction",
    "gold",
    "tolerance",
    "raw_vlm_correct",
    "raw_geometry_correct",
    "calibrated_geometry_correct",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_phase1(project: Path) -> int:
    manifest = project / "audit" / "frozen_results.sha256"
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = project / relative.strip().lstrip("*")
        assert path.is_file(), f"Frozen Phase 1 artifact missing: {path}"
        assert sha256_file(path) == expected, f"Frozen Phase 1 artifact changed: {path}"
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()

    frozen_checked = validate_frozen_phase1(project)
    rows = read_jsonl(project / "phase2_audit" / "accepted_samples.jsonl")
    assert rows, "No accepted Phase 2 samples"
    assert len(rows) == len({str(row["sample_id"]) for row in rows})
    assert len(rows) == len({str(row["chart_id"]) for row in rows}), "More than one accepted question per chart"
    for row in rows:
        missing = REQUIRED_FIELDS - set(row)
        assert not missing, f"{row.get('sample_id')} missing {sorted(missing)}"
        assert row["targeting_gold_access"] is False
        assert row["geometry_gold_access"] is False
        assert row["hidden_value_label_bbox_access"] is False
        assert row["phase1_chart_overlap"] is False
        assert len(row["selected_bar_ids"]) == len(row["selected_bar_bboxes"])
        assert len(row["selected_bar_ids"]) == len(row["bar_tops"])
        assert len(row["selected_bar_ids"]) == len(row["raw_target_values"])
        assert len(row["selected_bar_ids"]) == len(row["calibrated_target_values"])
        assert len(row["y_axis_ticks"]) >= 3

    with (project / "results" / "phase2_sample_audit.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(rows)
    assert "/data/" not in (project / "results" / "phase2_sample_audit.csv").read_text(encoding="utf-8")
    for required in [
        "phase2_metrics.csv",
        "phase2_paired_comparisons.csv",
        "phase2_failure_audit.csv",
        "phase2_chart_split.csv",
        "phase2_screening.csv",
    ]:
        assert (project / "results" / required).is_file(), required
    assert (project / "PHASE2_REPORT.md").is_file()

    review_dir = project / "phase2_debug_overlays" / "representative"
    overlays = list(review_dir.glob("*.png"))
    assert len(overlays) >= 30, f"Need at least 30 representative overlays, found {len(overlays)}"
    with (review_dir / "selection_manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    assert len(manifest_rows) == len(overlays)

    suspicious = re.compile(r"(?i)(api[_-]?key|access[_-]?token|password|passwd)\s*=\s*['\"][^'\"]+")
    for path in list((project / "src").glob("phase2*.py")) + list((project / "scripts").glob("*phase2*.sh")):
        text = path.read_text(encoding="utf-8")
        assert "/data/" not in text, f"Absolute server path in source: {path}"
        assert not suspicious.search(text), f"Possible secret in source: {path}"

    print(
        json.dumps(
            {
                "status": "ok",
                "phase1_frozen_files_checked": frozen_checked,
                "phase2_samples": len(rows),
                "representative_overlays": len(overlays),
                "unique_phase2_charts": len({str(row['chart_id']) for row in rows}),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
