from __future__ import annotations

from phase2_natural_bar import (
    answer_from_geometry,
    correctness,
    execute_operation,
    semantic_plan,
)


def main() -> None:
    bars = [
        {"bar_id": "bar_00", "detected_x_label": "FY21"},
        {"bar_id": "bar_01", "detected_x_label": "FY22"},
        {"bar_id": "bar_02", "detected_x_label": "FY23"},
    ]
    plan = semantic_plan(
        "What is the difference between FY21 and FY23?",
        "",
        bars,
    )
    assert plan["status"] == "success"
    assert plan["operation"] == "difference"
    assert plan["target_bar_ids"] == ["bar_00", "bar_02"]
    assert plan["gold_access"] is False
    assert plan["hidden_value_label_bbox_access"] is False
    assert execute_operation("direct", [12.5], "") == 12.5
    assert execute_operation("difference", [10.0, 16.0], "") == 6.0
    assert execute_operation("growth_rate", [10.0, 15.0], "percentage increase") == 50.0
    assert execute_operation("growth_rate", [10.0, 7.0], "percentage decrease") == 30.0
    choice, numeric, _ = answer_from_geometry(
        "single_choice",
        "A: 10%\nB: 15%\nC: 20%",
        "direct",
        [14.7],
        "What was the value in FY23?",
    )
    assert choice == "B"
    assert numeric == 14.7
    assert correctness("numerical", 12.1, "10", 2.0) is False
    assert correctness("numerical", 11.9, "10", 2.0) is True
    assert correctness("numerical", "2.5e1", "25", 0.0) is True
    assert correctness("single_choice", "B", "B", None) is True
    print("phase2 unit tests passed")


if __name__ == "__main__":
    main()
