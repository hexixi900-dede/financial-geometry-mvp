from __future__ import annotations

from dataclasses import replace

import numpy as np

from calibrate_geometry import assign_chart_splits
from geometry_core import OCRToken, axis_localizer, interpolate_pixel_to_value, parse_numeric_label


def main() -> None:
    assert parse_numeric_label("31%")[0] == 31.0
    assert parse_numeric_label("(1,200.5)")[0] == -1200.5
    assert parse_numeric_label("FY2024")[0] is None
    axis = {"slope": -0.5, "intercept": 50.0}
    assert interpolate_pixel_to_value(20, axis) == 40.0

    label = OCRToken(
        index=1,
        text="12.3",
        confidence=0.99,
        polygon=((1.0, 1.0), (3.0, 1.0), (3.0, 2.0), (1.0, 2.0)),
        x1=1.0,
        y1=1.0,
        x2=3.0,
        y2=2.0,
        cx=2.0,
        cy=1.5,
        numeric_value=12.3,
        numeric_suffix="",
    )
    locator = replace(label, text="", confidence=0.0, numeric_value=None, numeric_suffix="")
    assert locator.numeric_value is None and locator.text == ""

    percent_ticks = []
    for index, (text, value, y, suffix) in enumerate(
        [
            ("259", 259.0, 30.0, ""),
            ("159", 159.0, 110.0, ""),
            ("109", 109.0, 150.0, ""),
            ("59", 59.0, 190.0, ""),
            ("0%", 0.0, 230.0, "%"),
        ]
    ):
        percent_ticks.append(
            replace(
                label,
                index=10 + index,
                text=text,
                numeric_value=value,
                numeric_suffix=suffix,
                x1=10.0,
                x2=45.0,
                cx=27.5,
                y1=y - 8.0,
                y2=y + 8.0,
                cy=y,
                polygon=((10.0, y - 8.0), (45.0, y - 8.0), (45.0, y + 8.0), (10.0, y + 8.0)),
            )
        )
    repaired_axis = axis_localizer(percent_ticks, width=500, height=330)
    assert repaired_axis is not None
    assert [tick["value"] for tick in repaired_axis["ticks"]] == [25.0, 15.0, 10.0, 5.0, 0.0]

    inverted_right_ticks = []
    for index, (value, y) in enumerate([(10.0, 40.0), (12.0, 120.0), (14.0, 200.0)]):
        inverted_right_ticks.append(
            replace(
                label,
                index=30 + index,
                text=f"{value:g}",
                numeric_value=value,
                x1=450.0,
                x2=482.0,
                cx=466.0,
                y1=y - 8.0,
                y2=y + 8.0,
                cy=y,
                polygon=((450.0, y - 8.0), (482.0, y - 8.0), (482.0, y + 8.0), (450.0, y + 8.0)),
            )
        )
    dual_axis = axis_localizer(percent_ticks + inverted_right_ticks, width=500, height=330)
    assert dual_axis is not None and dual_axis["right_axis_detected"]
    assert dual_axis["right_axis"]["slope"] > 0

    rows = [
        {"chart_id": "chart_a", "geometry": {"chart_type": "bar"}},
        {"chart_id": "chart_a", "geometry": {"chart_type": "bar"}},
        {"chart_id": "chart_b", "geometry": {"chart_type": "bar"}},
        {"chart_id": "chart_c", "geometry": {"chart_type": "line"}},
        {"chart_id": "chart_d", "geometry": {"chart_type": "line"}},
    ]
    split = assign_chart_splits(rows, seed=7, test_fraction=0.25)
    train = {chart_id for chart_id, value in split.items() if value == "train"}
    test = {chart_id for chart_id, value in split.items() if value == "test"}
    assert train.isdisjoint(test)
    assert np.isfinite([parse_numeric_label("0.5")[0]]).all()
    print("smoke_test: PASS")


if __name__ == "__main__":
    main()
