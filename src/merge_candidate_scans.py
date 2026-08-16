from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    rows: list[dict] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for path in args.inputs:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"malformed JSONL: {path}:{line_number}: {exc}") from exc
                chart_id = str(row["chart_id"])
                if chart_id in seen:
                    duplicates.append(chart_id)
                    continue
                seen.add(chart_id)
                rows.append(row)
    if duplicates:
        raise RuntimeError(f"duplicate chart ids across inputs: {len(duplicates)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        shutil.copy2(args.output, args.backup)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output)
    print(json.dumps({"status": "complete", "rows": len(rows), "unique_chart_ids": len(seen), "backup": str(args.backup)}))


if __name__ == "__main__":
    main()
