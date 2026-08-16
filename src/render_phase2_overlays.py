from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from phase2_natural_bar import chart_cache_key


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def select_representative(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    accepted = [row for row in rows if row["decision"] == "accept"]
    rejected_by_reason: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in rows:
        if row["decision"] != "accept":
            rejected_by_reason[row["reason"]].append(row)
    selected = accepted[:count]
    reasons = sorted(rejected_by_reason)
    while len(selected) < count and reasons:
        remaining: list[str] = []
        for reason in reasons:
            queue = rejected_by_reason[reason]
            if queue and len(selected) < count:
                selected.append(queue.popleft())
            if queue:
                remaining.append(reason)
        reasons = remaining
    return selected


def render(
    image: np.ndarray,
    screen: dict[str, str],
    analysis: dict[str, Any],
    accepted: dict[str, Any] | None,
) -> np.ndarray:
    canvas = image.copy()
    axis = analysis.get("axis")
    selected_ids = set(accepted.get("selected_bar_ids", [])) if accepted else set()
    if axis:
        left, top, right, bottom = [int(round(value)) for value in axis["plot_bbox"]]
        cv2.rectangle(canvas, (left, top), (right, bottom), (255, 160, 0), 2)
        for tick in axis["ticks"]:
            y = int(round(float(tick["pixel_y"])))
            cv2.circle(canvas, (left, y), 5, (255, 0, 255), -1)
            cv2.putText(canvas, f"{tick['value']:g}@{y}", (left + 7, max(14, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1, cv2.LINE_AA)
    for bar in analysis.get("bars", []):
        x, y, width, height = map(int, bar["bbox"])
        chosen = bar["bar_id"] in selected_ids
        color = (0, 0, 255) if chosen else (130, 130, 130)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), color, 2 if chosen else 1)
        cv2.putText(canvas, f"{bar['bar_id']}:{bar.get('detected_x_label','')}", (x, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        if chosen:
            cv2.circle(canvas, (int(round(bar["target_x"])), int(round(bar["target_y"]))), 6, color, -1)
    for token in analysis.get("interior_numeric_tokens", []):
        x1, y1, x2, y2 = map(lambda value: int(round(float(value))), token["bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 190, 255), 2)
    panel_height = 92
    panel = np.full((panel_height, canvas.shape[1], 3), 255, dtype=np.uint8)
    lines = [
        f"sample={screen['sample_id']} decision={screen['decision']} reason={screen['reason']}",
        f"analysis={analysis.get('status')} target={(accepted or {}).get('semantic_target','NA')}",
        screen["question"][:170],
        "RED target; GRAY detected bar; MAGENTA tick; YELLOW rejected numeric annotation; BLUE plot",
    ]
    for index, line in enumerate(lines):
        cv2.putText(panel, line, (8, 18 + index * 21), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([panel, canvas])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--raw-vlm", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()

    raw_by_sample = {str(row["sample_id"]): row for row in read_jsonl(args.raw_vlm)}
    accepted_by_sample = {
        str(row["sample_id"]): row
        for row in read_jsonl(args.project / "phase2_audit" / "accepted_samples.jsonl")
    }
    cache = {
        str(row["cache_key"]): row
        for row in read_jsonl(args.project / "phase2_audit" / "chart_analysis_cache.jsonl")
    }
    screening = read_csv(args.project / "results" / "phase2_screening.csv")
    usable: list[dict[str, Any]] = []
    for row in screening:
        raw = raw_by_sample.get(str(row["sample_id"]))
        if raw is None:
            continue
        key = chart_cache_key(str(row["chart_id"]), str(raw["image_path"]))
        if key not in cache:
            continue
        usable.append({**row, "image_path": str(raw["image_path"]), "cache_key": key})
    selected = select_representative(usable, args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for index, screen in enumerate(selected):
        image = cv2.imread(screen["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            continue
        analysis = cache[screen["cache_key"]]
        accepted = accepted_by_sample.get(str(screen["sample_id"]))
        overlay = render(image, screen, analysis, accepted)
        output_path = args.output_dir / f"phase2_review_{index:02d}_s{int(screen['sample_id']):05d}.png"
        cv2.imwrite(str(output_path), overlay)
        manifest.append(
            {
                "overlay": output_path.name,
                "sample_id": screen["sample_id"],
                "chart_id": screen["chart_id"],
                "decision": screen["decision"],
                "reason": screen["reason"],
                "analysis_status": analysis.get("status"),
            }
        )
    write_csv(args.output_dir / "selection_manifest.csv", manifest)
    print(json.dumps({"rendered": len(manifest), "accepted_in_review": sum(row["decision"] == "accept" for row in manifest), "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
