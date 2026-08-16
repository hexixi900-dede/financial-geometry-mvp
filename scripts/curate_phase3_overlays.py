from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()
    if not 30 <= args.count <= 50:
        raise ValueError("published Phase3 overlay count must be between 30 and 50")
    rows = read_csv(args.project / "results" / "phase3_line_samples.csv")
    source = args.project / "phase3_debug_overlays"
    destination = args.project / "examples" / "phase3_debug_overlays"
    destination.mkdir(parents=True, exist_ok=True)

    selected: list[tuple[str, str, str]] = []
    for method in ["auto_local", "auto_column"]:
        for row in rows:
            selected.append((row["sample_id"], method, f"complete_{method}_audit"))
    remaining = max(0, args.count - len(selected))
    worst_oracle = sorted(
        rows,
        key=lambda row: float(row["oracle_x_absolute_error"]),
        reverse=True,
    )[:remaining]
    for row in worst_oracle:
        selected.append((row["sample_id"], "oracle_x", "largest_old_label_center_proxy_error"))
    selected = selected[: args.count]

    manifest: list[dict[str, str]] = []
    for sample_id, method, reason in selected:
        filename = f"{sample_id}_{method}.png"
        source_path = source / filename
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        shutil.copy2(source_path, destination / filename)
        manifest.append(
            {
                "sample_id": sample_id,
                "method": method,
                "filename": filename,
                "selection_reason": reason,
            }
        )
    with (destination / "selection_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    print({"published_phase3_overlays": len(manifest), "destination": str(destination)})


if __name__ == "__main__":
    main()
