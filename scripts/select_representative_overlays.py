#!/usr/bin/env python3
"""Copy a deterministic audit subset of existing debug overlays.

This utility does not run Geometry or Calibration and never edits result tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_spaced_rows(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0 or frame.empty:
        return frame.iloc[0:0]
    ordered = frame.sort_values(["axis_normalized_error", "sample_id"])
    if count >= len(ordered):
        return ordered
    positions = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)] if count > 1 else [len(ordered) // 2]
    return ordered.iloc[positions]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()

    project = args.project.resolve()
    geometry_path = project / "results" / "geometry_samples.csv"
    predictions_path = project / "results" / "calibrated_predictions.csv"
    source_dir = project / "debug_overlays"
    output_dir = project / "examples" / "debug_overlays"

    geometry = pd.read_csv(geometry_path)
    predictions = pd.read_csv(predictions_path)[
        ["sample_id", "split", "calibrated_absolute_error", "calibrated_axis_normalized_error"]
    ]
    rows = geometry.merge(predictions, on="sample_id", how="left", validate="one_to_one")

    required = rows[(rows["chart_type"] == "line") | ((rows["chart_type"] == "bar") & (rows["split"] == "test"))].copy()
    if len(required) > args.count:
        raise SystemExit(f"The required audit strata contain {len(required)} rows, exceeding --count={args.count}.")

    selected_ids = set(required["sample_id"])
    remaining = rows[(rows["chart_type"] == "bar") & (rows["split"] == "train") & ~rows["sample_id"].isin(selected_ids)]
    fillers = evenly_spaced_rows(remaining, args.count - len(required)).copy()
    selected = pd.concat([required, fillers], ignore_index=True).sort_values("sample_id")

    reasons: dict[str, str] = {}
    for row in selected.itertuples():
        if row.chart_type == "line":
            reasons[row.sample_id] = "all_line_samples"
        elif row.split == "test":
            reasons[row.sample_id] = "all_test_bar_samples"
        else:
            reasons[row.sample_id] = "train_bar_error_quantile"

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_dir.glob("*.png")}
    expected = {f"{sample_id}.png" for sample_id in selected["sample_id"]}
    unexpected = existing - expected
    if unexpected:
        raise SystemExit(f"Refusing to remove unexpected existing overlays: {sorted(unexpected)}")

    manifest_rows: list[dict[str, object]] = []
    for row in selected.itertuples():
        source = source_dir / f"{row.sample_id}.png"
        destination = output_dir / source.name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        manifest_rows.append(
            {
                "sample_id": row.sample_id,
                "chart_id": row.chart_id,
                "representative_sample_id": row.representative_sample_id,
                "chart_type": row.chart_type,
                "split": row.split,
                "selection_reason": reasons[row.sample_id],
                "pseudo_gold": row.pseudo_gold,
                "raw_geometry_value": row.raw_geometry_value,
                "raw_absolute_error": row.absolute_error,
                "raw_axis_normalized_error": row.axis_normalized_error,
                "calibrated_absolute_error": row.calibrated_absolute_error,
                "calibrated_axis_normalized_error": row.calibrated_axis_normalized_error,
                "overlay_file": f"examples/debug_overlays/{destination.name}",
                "overlay_sha256": sha256(destination),
            }
        )

    manifest_path = output_dir / "selection_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"selected={len(manifest_rows)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
