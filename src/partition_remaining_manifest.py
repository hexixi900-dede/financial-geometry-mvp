from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def processed_ids(path: Path) -> set[str]:
    result: set[str] = set()
    if not path.exists():
        return result
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                result.add(str(json.loads(line)["chart_id"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return result


def eligible(row: dict[str, str]) -> bool:
    return (
        int(row["has_numerical"]) == 1
        and int(row["keyword_exclude"]) == 0
        and (int(row["bar_rectangles"]) >= 2 or int(row["sloped_segments"]) >= 1)
        and int(row["image_width"]) >= 300
        and int(row["image_height"]) >= 220
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--output-a", type=Path, required=True)
    parser.add_argument("--output-b", type=Path, required=True)
    args = parser.parse_args()

    done = processed_ids(args.processed)
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        remaining = [row for row in reader if eligible(row) and row["chart_id"] not in done]
    remaining.sort(key=lambda row: hashlib.sha256(row["chart_id"].encode()).hexdigest())
    partitions = [remaining[::2], remaining[1::2]]
    for path, rows in zip((args.output_a, args.output_b), partitions):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"processed": len(done), "remaining": len(remaining), "a": len(partitions[0]), "b": len(partitions[1])}))


if __name__ == "__main__":
    main()
