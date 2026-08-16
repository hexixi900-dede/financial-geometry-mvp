from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


INCLUDE_TERMS = re.compile(
    r"\b(bar|bars|column|columns|line|trend|sales|revenue|income|earnings|ebit|"
    r"margin|growth|price|rate|yield|volume|production|cost|expense|index|ratio|"
    r"loan|deposit|export|import|market share|cash flow|profit)\b",
    re.IGNORECASE,
)
EXCLUDE_TERMS = re.compile(
    r"\b(pie|donut|doughnut|area chart|scatter|bubble|candlestick|waterfall|"
    r"stacked|treemap|heatmap|radar|spider|box ?plot|histogram|dual[- ]axis|"
    r"two y[- ]axes|secondary axis|right y[- ]axis|combination chart|combo chart)\b",
    re.IGNORECASE,
)


def jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def image_structure_features(path: str) -> dict[str, float | int]:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return {
            "image_width": 0,
            "image_height": 0,
            "bar_rectangles": 0,
            "sloped_segments": 0,
            "color_fraction": 0.0,
        }
    height, width = image.shape[:2]
    scale = min(1.0, 640.0 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_h, small_w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    color_mask = ((saturation >= 35) & (value >= 35) & (value <= 250)).astype(np.uint8)
    color_fraction = float(color_mask.mean())

    bar_rectangles = 0
    quantized = (image // 32).astype(np.uint8)
    packed = (
        quantized[:, :, 0].astype(np.int32)
        + 8 * quantized[:, :, 1].astype(np.int32)
        + 64 * quantized[:, :, 2].astype(np.int32)
    )
    values_, counts = np.unique(packed, return_counts=True)
    order = np.argsort(counts)[::-1]
    image_area = small_h * small_w
    for packed_value in values_[order[:24]]:
        mask = (packed == packed_value).astype(np.uint8)
        fraction = float(mask.mean())
        if fraction < 0.00015 or fraction > 0.18:
            continue
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for stat in stats[1:count]:
            x, y, box_w, box_h, area = map(int, stat)
            if box_w < max(4, int(0.008 * small_w)):
                continue
            if box_h < max(8, int(0.025 * small_h)):
                continue
            if box_w > 0.18 * small_w or box_h > 0.85 * small_h:
                continue
            fill = area / max(1, box_w * box_h)
            if fill >= 0.55 and box_h >= 0.65 * box_w and area >= 0.0002 * image_area:
                bar_rectangles += 1

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 160)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(20, int(0.04 * small_w)),
        minLineLength=max(20, int(0.06 * small_w)),
        maxLineGap=max(3, int(0.012 * small_w)),
    )
    sloped_segments = 0
    if lines is not None:
        for segment in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = map(int, segment)
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            if dx >= 0.06 * small_w and 0.08 <= dy / max(dx, 1) <= 5.0:
                sloped_segments += 1

    return {
        "image_width": width,
        "image_height": height,
        "bar_rectangles": bar_rectangles,
        "sloped_segments": sloped_segments,
        "color_fraction": color_fraction,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with args.records.open(encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))

    question_type_counts = Counter(str(row["question_type"]) for row in rows)
    field_rows: list[dict[str, Any]] = []
    for field in [
        "sample_id",
        "image_path",
        "question",
        "question_type",
        "reference",
        "unit",
        "tolerance",
        "verified_caption",
        "related_sentences",
    ]:
        values = [row.get(field) for row in rows]
        missing = sum(value is None or (isinstance(value, str) and not value.strip()) for value in values)
        field_rows.append(
            {
                "field": field,
                "rows": len(values),
                "missing_or_empty": missing,
                "usable": missing < len(values),
                "notes": {
                    "reference": "FinMME Gold; numerical rows contain a numeric string",
                    "question_type": "numerical is the released Calculation category",
                    "tolerance": "absolute tolerance; only meaningful for numerical rows",
                    "image_path": "existing preprocessed JPEG cache; no re-download required",
                }.get(field, ""),
            }
        )

    tolerance_counter: Counter[str] = Counter()
    numeric_rows = [row for row in rows if row["question_type"] == "numerical"]
    for row in numeric_rows:
        value = jsonable(row.get("tolerance"))
        tolerance_counter["NaN_or_missing" if value is None else str(value)] += 1

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["official_preprocessed_image_sha256"])].append(row)

    chart_rows: list[dict[str, Any]] = []
    for index, (chart_id, chart_questions) in enumerate(sorted(grouped.items())):
        representative = min(chart_questions, key=lambda item: int(item["sample_id"]))
        numerical = [item for item in chart_questions if item["question_type"] == "numerical"]
        text = " ".join(
            str(item.get(key, ""))
            for item in chart_questions
            for key in ("verified_caption", "question", "related_sentences")
        )
        features = image_structure_features(str(representative["image_path"]))
        has_include = bool(INCLUDE_TERMS.search(text))
        has_exclude = bool(EXCLUDE_TERMS.search(text))
        priority = (
            4.0 * bool(numerical)
            + 1.5 * has_include
            - 4.0 * has_exclude
            + min(int(features["bar_rectangles"]), 12) / 4.0
            + min(int(features["sloped_segments"]), 12) / 8.0
            + min(float(features["color_fraction"]), 0.25) * 2.0
        )
        chart_rows.append(
            {
                "chart_index": index,
                "chart_id": chart_id,
                "representative_sample_id": representative["sample_id"],
                "image_path": representative["image_path"],
                "original_image_size": "x".join(map(str, representative["original_image_size"])),
                "processed_image_size": "x".join(map(str, representative["official_preprocessed_image_size"])),
                "question_count": len(chart_questions),
                "has_numerical": int(bool(numerical)),
                "numerical_count": len(numerical),
                "numerical_sample_ids": "|".join(str(item["sample_id"]) for item in numerical),
                "numerical_gold": "|".join(str(item["reference"]) for item in numerical),
                "numerical_unit": "|".join(str(item["unit"]) for item in numerical),
                "numerical_tolerance": "|".join(str(jsonable(item.get("tolerance"))) for item in numerical),
                "caption": representative.get("verified_caption", ""),
                "keyword_include": int(has_include),
                "keyword_exclude": int(has_exclude),
                "prefilter_priority": priority,
                **features,
            }
        )

    chart_rows.sort(key=lambda item: (-float(item["prefilter_priority"]), str(item["chart_id"])))
    write_csv(args.project / "results" / "finmme_field_audit.csv", field_rows)
    write_csv(
        args.project / "results" / "finmme_question_type_counts.csv",
        [{"question_type": key, "count": value} for key, value in sorted(question_type_counts.items())],
    )
    write_csv(
        args.project / "results" / "finmme_tolerance_counts.csv",
        [{"tolerance": key, "count": value} for key, value in tolerance_counter.most_common()],
    )
    write_csv(args.project / "results" / "finmme_chart_manifest.csv", chart_rows)
    summary = {
        "records_path": str(args.records),
        "row_count": len(rows),
        "unique_chart_count": len(chart_rows),
        "question_type_counts": dict(question_type_counts),
        "numerical_rows": len(numeric_rows),
        "numerical_tolerance_nan_or_missing": tolerance_counter["NaN_or_missing"],
        "existing_image_paths_present": sum(Path(str(row["image_path"])).is_file() for row in rows),
        "charts_with_numerical": sum(int(row["has_numerical"]) for row in chart_rows),
    }
    (args.project / "audit" / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
