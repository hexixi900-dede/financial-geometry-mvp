from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from calibrate_geometry import FEATURE_NAMES


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    samples = load_csv(project / "results" / "geometry_samples.csv")
    sample_json = load_jsonl(project / "audit" / "geometry_samples.jsonl")
    candidates = {
        str(row["chart_id"]): row
        for row in load_jsonl(project / "audit" / "candidate_scan_v3.jsonl")
    }
    predictions = load_csv(project / "results" / "calibrated_predictions.csv")
    split_rows = load_csv(project / "results" / "chart_split.csv")

    assert samples, "no samples"
    assert len(samples) == len(sample_json), "flat/full sample count mismatch"
    assert not any(
        forbidden in feature.lower()
        for feature in FEATURE_NAMES
        for forbidden in ("gold", "label", "crop", "mask")
    ), "construction/Gold metadata leaked into calibrator features"
    assert len(samples) == len(predictions), "prediction/sample count mismatch"
    chart_ids = [row["chart_id"] for row in samples]
    assert len(chart_ids) == len(set(chart_ids)), "more than one sample per chart"
    train = {row["chart_id"] for row in split_rows if row["split"] == "train"}
    test = {row["chart_id"] for row in split_rows if row["split"] == "test"}
    assert train and test and not (train & test), "chart-level split leakage or empty split"
    assert set(chart_ids) == train | test, "split does not cover all sample charts"

    required = {
        "tick_values",
        "tick_pixel_y",
        "raw_geometry_value",
        "pseudo_gold",
        "absolute_error",
        "relative_error",
        "axis_normalized_error",
        "geometry_gold_access",
    }
    assert not (required - set(samples[0])), f"missing columns: {required - set(samples[0])}"

    for row in samples:
        assert row["geometry_gold_access"].strip().lower() == "false"
        ticks = json.loads(row["tick_values"])
        pixel_y = json.loads(row["tick_pixel_y"])
        assert len(ticks) == len(pixel_y) and len(ticks) >= 3
        bar_bbox = json.loads(row["bar_bbox"])
        line_point = json.loads(row["line_point"])
        if row["chart_type"] == "bar":
            assert isinstance(bar_bbox, list) and len(bar_bbox) == 4 and line_point is None
        elif row["chart_type"] == "line":
            assert isinstance(line_point, list) and len(line_point) == 2 and bar_bbox is None
        else:
            raise AssertionError(f"unexpected chart type: {row['chart_type']}")
        raw = float(row["raw_geometry_value"])
        gold = float(row["pseudo_gold"])
        axis_span = float(row["axis_span"])
        assert math.isclose(float(row["absolute_error"]), abs(raw - gold), rel_tol=1e-9, abs_tol=1e-10)
        assert math.isclose(
            float(row["axis_normalized_error"]), abs(raw - gold) / axis_span, rel_tol=1e-9, abs_tol=1e-10
        )
        for key in ("source_image_path", "masked_image_path", "debug_overlay_path"):
            path = Path(row[key])
            if not path.is_absolute():
                path = project / path
            assert path.is_file() and path.stat().st_size > 0, f"missing artifact: {path}"

    for row in sample_json:
        assert not row["axis"].get("right_axis_detected", False)
        assert int(row["axis"].get("additional_y_axis_count", 0)) == 0
        assert int(row["geometry"].get("stacked_evidence", 0)) == 0
        if row["geometry"]["chart_type"] == "bar":
            assert int(candidates[str(row["chart_id"])]["prefilter"]["sloped_segments"]) < 3
        else:
            assert float(candidates[str(row["chart_id"])]["prefilter"]["color_fraction"]) <= 0.12

    sample_ids = {row["sample_id"] for row in samples}
    assert sample_ids == {row["sample_id"] for row in predictions}
    assert len(list((project / "masked_images").glob("fgmvp_*.png"))) == len(samples)
    assert len(list((project / "debug_overlays").glob("fgmvp_*.png"))) == len(samples)
    assert (project / "MVP_REPORT.md").stat().st_size > 0
    print(
        json.dumps(
            {
                "status": "PASS",
                "samples": len(samples),
                "train_charts": len(train),
                "test_charts": len(test),
                "chart_overlap": len(train & test),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
