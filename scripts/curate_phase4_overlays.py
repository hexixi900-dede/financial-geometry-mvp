#!/usr/bin/env python3
"""Select a deterministic, audit-covering subset of Phase 4 debug overlays.

The selection favors audit coverage: every runtime-rejection class and the
multi-line configuration are represented before error-quantile fillers are
added. This utility never edits result tables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def evenly_spaced(rows: list[dict[str, str]], count: int, key: str) -> list[dict[str, str]]:
    if count <= 0 or not rows:
        return []
    ordered = sorted(rows, key=lambda row: (float(row.get(key) or "1e9"), row["sample_id"]))
    if count >= len(ordered):
        return ordered
    positions = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)] if count > 1 else [len(ordered) // 2]
    return [ordered[position] for position in positions]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=60)
    args = parser.parse_args()
    project = args.project.resolve()

    samples = read_csv(project / "results" / "phase4_line_samples.csv")
    source_dir = project / "phase4_debug_overlays"
    output_dir = project / "examples" / "phase4_debug_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected: dict[str, str] = {}  # filename -> reason

    # 1. Every runtime rejection class: up to 3 samples per class (auto_local overlay).
    rejected = [row for row in samples if row.get("auto_status") != "success"]
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rejected:
        by_class.setdefault(f"{row['auto_status']}:{row.get('auto_status_reason') or ''}", []).append(row)
    for class_name in sorted(by_class):
        for row in sorted(by_class[class_name], key=lambda item: item["sample_id"])[:3]:
            selected[f"{row['sample_id']}_auto_local.png"] = f"rejected:{class_name}"

    # 2. Multi-line successes: both auto and oracle views of the same samples.
    multi_ok = [row for row in samples if row["chart_kind"] == "multi_line" and row.get("auto_status") == "success"]
    for row in evenly_spaced(multi_ok, 12, "auto_local_axis_normalized_error"):
        selected[f"{row['sample_id']}_auto_local.png"] = "multi_line_success_quantile"
        selected[f"{row['sample_id']}_oracle_local.png"] = "multi_line_oracle_pair"

    # 3. Single-line successes: error quantiles on the local-fit view.
    single_ok = [row for row in samples if row["chart_kind"] == "single_line" and row.get("auto_status") == "success"]
    remaining = max(0, args.budget - len(selected))
    for row in evenly_spaced(single_ok, remaining, "auto_local_axis_normalized_error"):
        selected.setdefault(f"{row['sample_id']}_auto_local.png", "single_line_success_quantile")

    manifest_rows: list[dict[str, object]] = []
    for filename in sorted(selected):
        source = source_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_dir / filename
        shutil.copy2(source, destination)
        manifest_rows.append(
            {
                "overlay_file": f"examples/phase4_debug_overlays/{filename}",
                "selection_reason": selected[filename],
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
