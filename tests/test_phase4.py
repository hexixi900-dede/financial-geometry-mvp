"""Phase 4 unit tests: legend parsing, series matching, ambiguity rules."""
from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np

from phase4_legend import match_series_query, strip_query_boilerplate
from phase4_multiline_core import _resolve_target_series, auto_multiline_pipeline
from phase4_series import color_distance, detect_line_series_components, local_continuity, match_series_by_color


def _axis() -> dict:
    return {
        "slope": -1.0,
        "intercept": 100.0,
        "axis_span": 100.0,
        "plot_bbox": [40.0, 20.0, 460.0, 220.0],
        "ticks": [],
        "token_indices": [],
    }


def _synthetic_multiline() -> tuple[np.ndarray, dict]:
    image = np.full((240, 480, 3), 255, dtype=np.uint8)
    # Blue-ish descending line and orange-ish ascending line across the plot.
    blue = (200, 120, 40)
    orange = (40, 130, 230)
    for x in range(50, 450):
        y_blue = int(round(60 + 0.25 * (x - 50)))
        y_orange = int(round(180 - 0.30 * (x - 50)))
        cv2.circle(image, (x, y_blue), 2, blue, -1)
        cv2.circle(image, (x, y_orange), 2, orange, -1)
    axis = _axis()
    return image, axis


def main() -> None:
    # Boilerplate stripping keeps the series phrase only.
    assert strip_query_boilerplate("What is the value of EBITDA Margin in FY23?") == "EBITDA Margin"
    assert strip_query_boilerplate("What is the value for FY23?") == ""
    assert strip_query_boilerplate("What was the revenue in 3QFY24?") == "revenue"

    # Series query matching: containment and fuzzy.
    entries = [
        {"legend_index": 0, "name": "Revenue", "normalized": "revenue"},
        {"legend_index": 1, "name": "EBITDA Margin", "normalized": "ebitdamargin"},
    ]
    match = match_series_query("EBITDA Margin", entries)
    assert match["status"] == "ok" and match["match"]["legend_index"] == 1
    match = match_series_query("ebitda margn", entries)  # OCR typo -> fuzzy
    assert match["status"] == "ok" and match["match"]["legend_index"] == 1
    assert match_series_query("nonexistent series", entries)["status"] == "no_match"
    assert match_series_query("Revenue", [])["status"] == "no_match"

    # Synthetic two-line image yields two distinct colored components.
    image, axis = _synthetic_multiline()
    series = detect_line_series_components(image, axis)
    assert len(series) == 2, f"expected 2 series, got {len(series)}"
    colors = sorted(tuple(round(v) for v in item["median_bgr"]) for item in series)
    assert any(abs(c[0] - 200) < 40 for c in colors), colors
    assert any(abs(c[2] - 230) < 40 for c in colors), colors

    # Color matching assigns each legend entry to the right component.
    legend_entries = [
        {"legend_index": 0, "name": "A", "normalized": "a", "swatch": {"median_bgr": [200.0, 120.0, 40.0]}},
        {"legend_index": 1, "name": "B", "normalized": "b", "swatch": {"median_bgr": [40.0, 130.0, 230.0]}},
    ]
    match = match_series_by_color(legend_entries, series)
    assert len(match["entry_matches"]) == 2
    assert match["entry_matches"][0]["matched_series_id"] != match["entry_matches"][1]["matched_series_id"]
    assert all(item["color_margin"] and item["color_margin"] > 20 for item in match["entry_matches"])

    # Color distance sanity.
    distance = color_distance([200, 120, 40], [40, 130, 230])
    assert distance["rgb"] > 200 and distance["combined"] > 100
    assert color_distance([200, 120, 40], [201, 121, 41])["combined"] < 5

    # Continuity on a solid line is full; empty mask is zero.
    solid = np.zeros((240, 480), dtype=np.uint8)
    solid[100, 200:260] = 1
    continuity = local_continuity(solid, 230, 30)
    assert continuity["column_coverage"] > 0.98 and continuity["largest_gap"] == 0
    empty = np.zeros((240, 480), dtype=np.uint8)
    assert local_continuity(empty, 230, 30)["column_coverage"] == 0.0

    # Ambiguity resolution rules.
    legend_two = {
        "entries": [
            {"legend_index": 0, "name": "Revenue", "normalized": "revenue", "swatch": {"median_bgr": [200.0, 120.0, 40.0]}},
            {"legend_index": 1, "name": "Margin", "normalized": "margin", "swatch": {"median_bgr": [40.0, 130.0, 230.0]}},
        ]
    }
    series_dicts = [
        {"series_id": 0, "median_bgr": [200.0, 120.0, 40.0]},
        {"series_id": 1, "median_bgr": [40.0, 130.0, 230.0]},
    ]
    resolved = _resolve_target_series("Revenue", legend_two, series_dicts, "auto_series")
    assert resolved["status"] == "ok" and resolved["target_series_id"] == 0
    resolved = _resolve_target_series(None, legend_two, series_dicts, "auto_series")
    assert resolved["status"] == "ambiguous_series"
    resolved = _resolve_target_series(None, legend_two, [series_dicts[0]], "auto_series")
    assert resolved["status"] == "ok" and resolved["series_mode"] == "single_line_implicit"
    resolved = _resolve_target_series("Unknown", legend_two, series_dicts, "auto_series")
    assert resolved["status"] == "ambiguous_series"
    similar = [
        {"series_id": 0, "median_bgr": [200.0, 120.0, 40.0]},
        {"series_id": 1, "median_bgr": [203.0, 124.0, 44.0]},
    ]
    resolved = _resolve_target_series("Revenue", legend_two, similar, "auto_series")
    assert resolved["status"] == "ambiguous_series" and resolved["reason"] == "similar_series_colors"

    # Construction anchor coherence: polluted numeric anchors are dropped.
    from run_phase4_multiline import coherent_construction_anchors

    def _anchor(label, family, scalar, x):
        return {"label": label, "scalar_family": family, "scalar": scalar, "center_x": x}

    polluted = [
        _anchor("2", "numeric", 2.0, 10.0),
        _anchor("7", "numeric", 7.0, 50.0),
        _anchor("1", "numeric", 1.0, 90.0),
        _anchor("25.44", "numeric", 25.44, 130.0),
        _anchor("8", "numeric", 8.0, 170.0),
    ]
    assert coherent_construction_anchors(polluted) == []
    temporal = [_anchor("FY22", "year", 2022.0, 10.0), _anchor("FY23", "year", 2023.0, 90.0)]
    assert len(coherent_construction_anchors(temporal + polluted)) == 2
    systematic = [_anchor("10", "numeric", 10.0, 10.0), _anchor("20", "numeric", 20.0, 90.0), _anchor("30", "numeric", 30.0, 170.0)]
    assert len(coherent_construction_anchors(systematic)) == 3

    # No-leak AST contract on the automatic pipeline.
    source = Path(__file__).resolve().parent.parent / "src" / "phase4_multiline_core.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "auto_multiline_pipeline"]
    assert len(functions) == 1
    arguments = [argument.arg for argument in functions[0].args.args]
    assert arguments == ["reader", "masked_bgr", "question", "oracle_series_name"], arguments
    body = ast.get_source_segment(source.read_text(encoding="utf-8"), functions[0]) or ""
    forbidden = ["pseudo_gold", "masked_value_label", "label.center", "label.cx", "construction"]
    assert not any(term in body for term in forbidden), "auto pipeline references a forbidden construction-only input"

    print("phase4 unit tests passed")


if __name__ == "__main__":
    main()
