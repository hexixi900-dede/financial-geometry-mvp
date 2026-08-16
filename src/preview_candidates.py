from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from geometry_core import draw_debug_overlay, geometry_value, token_from_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        if row.get("status") != "viable":
            continue
        selected = row["viable_labels"][0]
        label = token_from_dict(selected["label"])
        axis = row["axis"]
        geometry = selected["geometry_on_original"]
        image = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        raw = geometry_value(geometry, axis)
        overlay = draw_debug_overlay(
            image,
            label,
            axis,
            geometry,
            raw_value=raw,
            pseudo_gold=float(label.numeric_value),
        )
        output = args.output_dir / f"{int(row['representative_sample_id']):05d}_{geometry['chart_type']}.png"
        cv2.imwrite(str(output), overlay)
        print(output)


if __name__ == "__main__":
    main()
