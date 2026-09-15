"""CPU visual tool tests, not OCR or end-to-end benchmark accuracy.

Rectangles and lines are actually drawn and measured by the existing pixel
readers. Most tests supply manual axis/legend/category metadata to isolate
tool composition. The calibration test supplies manual OCR boxes; it tests
axis fitting and coordinate binding, not an OCR engine's recognition.
"""
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from geometry_toolbox import measure_targets, prepare_chart


BLUE = [160, 60, 20]
GREEN = [30, 165, 45]
RED = [30, 35, 220]


def target(mark="bar", axis="left", **kwargs):
    result = dict(mark_type=mark, axis_id=axis, bar_scope="whole",
                  x_label="2024", series="", position="label", measurement="value")
    result.update(kwargs)
    return result


def chart():
    bbox = [40, 20, 360, 260]
    ticks_left = [dict(pixel_y=y, value=26-.1*y, text=str(26-.1*y),
                       bbox=[10, y-5, 30, y+5]) for y in (60, 160, 260)]
    ticks_right = [dict(pixel_y=y, value=2600-10*y, text=str(2600-10*y),
                        bbox=[370, y-5, 395, y+5]) for y in (60, 160, 260)]
    return dict(status="success", capability_status="supported", plot_bbox=bbox,
                axes={
                    "left": dict(axis_id="left", plot_bbox=bbox, slope=-.1,
                                 intercept=26, ticks=ticks_left),
                    "right": dict(axis_id="right", plot_bbox=bbox, slope=-10,
                                  intercept=2600, ticks=ticks_right)},
                x_axis_anchors=[dict(label="2024", center_x=200, confidence=1)],
                ocr_tokens=[dict(text="2024", center=[100, 275])],
                legend={"entries": [
                    dict(legend_index=i, name=name, normalized=name,
                         swatch={"median_bgr": color})
                    for i, (name, color) in enumerate((("blue", BLUE), ("green", GREEN), ("red", RED)))
                ]})


class ToolboxComposition(unittest.TestCase):
    def setUp(self):
        self.image = np.full((300, 400, 3), 255, np.uint8)
        self.chart = chart()

    def run_targets(self, targets, operation="evidence", calculations=None):
        plan = dict(targets=targets, operation=operation, calculations=calculations or [])
        return measure_targets(None, self.image, self.chart, plan)

    def test_grouped_bar_series_selects_distinct_columns(self):
        self.image[150:260, 70:90] = BLUE
        self.image[80:260, 110:130] = GREEN
        targets = [target(series=s) for s in ("blue", "green")]
        out = self.run_targets(targets)
        self.assertEqual(out["status"], "success")
        self.assertAlmostEqual(out["geometry_values"][0], 11)
        self.assertAlmostEqual(out["geometry_values"][1], 18)
        self.assertNotEqual(out["target_audit"][0]["bbox"], out["target_audit"][1]["bbox"])

    def test_stack_segment_and_whole_region_do_not_share_color_filter(self):
        self.image[150:260, 80:110] = BLUE
        self.image[100:150, 80:110] = GREEN
        base = dict(region=[190, 320, 290, 900], measurement="height", series="green")
        segment = target(**base)
        segment["bar_scope"] = "segment"
        whole = target(**base)
        out = self.run_targets([segment, whole])
        self.assertEqual(out["status"], "success")
        self.assertAlmostEqual(out["geometry_values"][0], 4.9)
        self.assertAlmostEqual(out["geometry_values"][1], 15.9)
        self.assertTrue(out["target_audit"][1]["whole_stack_union"])
        self.assertEqual(out["target_audit"][0]["measurement_semantics"], "unsigned_extent")

    def test_dual_axis_same_pixel_has_distinct_values_and_evidence(self):
        self.image[100:260, 80:110] = BLUE
        targets = [target(axis=a, series="blue", region=[190, 320, 290, 900])
                   for a in ("left", "right")]
        out = self.run_targets(targets)
        self.assertEqual(out["status"], "success")
        self.assertEqual(out["geometry_values"], [16, 1600])
        self.assertEqual([p["axis_id"] for p in out["pipeline_audit"]], ["left", "right"])
        self.assertNotEqual(out["target_audit"][0]["y_axis_ticks"],
                            out["target_audit"][1]["y_axis_ticks"])

    def test_mixed_bar_and_line_use_actual_detector_and_selected_axes(self):
        self.image[120:260, 180:210] = BLUE
        cv2.line(self.image, (60, 80), (340, 220), tuple(RED), 3)
        targets = [target(series="blue", region=[440, 380, 540, 900]),
                   target(mark="line", axis="right", series="red")]
        out = self.run_targets(targets)
        self.assertEqual(out["status"], "success", out.get("status_reason"))
        self.assertAlmostEqual(out["geometry_values"][0], 14, delta=.11)
        self.assertAlmostEqual(out["geometry_values"][1], 1100, delta=15)
        self.assertGreater(out["pipeline_audit"][1]["line_separation"]["bar_pixels_masked"], 0)
        self.assertEqual(out["pipeline_audit"][1]["line_separation"]["line_series_retained"], 1)
        self.assertEqual(out["target_audit"][1]["axis_id"], "right")

    def test_named_calculation_does_not_execute_conflicting_top_level_operation(self):
        self.image[150:260, 70:90] = BLUE
        self.image[80:260, 110:130] = GREEN
        targets = [target(series=s) for s in ("blue", "green")]
        # direct would reject two targets upstream; the explicit expression is
        # authoritative here and must not be replaced by the first target.
        out = self.run_targets(targets, operation="direct", calculations=[
            {"name": "increase", "expression": {"op": "difference",
                "args": [{"target": 1}, {"target": 0}]}}])
        self.assertEqual(out["status"], "success")
        self.assertIsNone(out["numeric_answer"])
        self.assertEqual(out["reasoning"]["formula"], "named_expressions")
        self.assertAlmostEqual(out["reasoning"]["calculations"][0]["value"], 7)

    def test_partial_measurements_are_preserved_without_numeric_result(self):
        self.image[150:260, 70:90] = BLUE
        self.chart["axes"].pop("right")
        out = self.run_targets([target(series="blue", region=[150, 200, 250, 900]),
                                target(axis="right")], operation="difference")
        self.assertEqual(out["status"], "measurement_failed")
        self.assertEqual(out["measurement_success_count"], 1)
        self.assertEqual(out["capability_status"], "partial_or_unsupported")
        self.assertIsNone(out["numeric_answer"])
        self.assertEqual(out["geometry_values"], [11, None])
        self.assertEqual(out["target_audit"][1]["status_reason"], "requested_axis_unrecovered")

    def test_pixel_sequence_cannot_be_averaged_as_period_observations(self):
        cv2.line(self.image, (60, 80), (340, 220), tuple(RED), 3)
        t = target(mark="line", series="red", measurement="sequence",
                   region=[150, 240, 850, 800])
        out = self.run_targets([t], operation="mean")
        self.assertEqual(out["status"], "arithmetic_failed")
        self.assertEqual(out["status_reason"], "arithmetic_requires_scalar_measurements")
        self.assertIsNone(out["numeric_answer"])

    def test_negative_bar_height_is_explicit_unsigned_extent(self):
        self.chart["axes"]["left"].update(intercept=15)
        self.image[150:220, 80:110] = BLUE
        out = self.run_targets([target(series="blue", region=[190, 490, 290, 750],
                                       measurement="height")], operation="direct")
        self.assertEqual(out["status"], "success")
        self.assertAlmostEqual(out["numeric_answer"], 6.9)
        self.assertEqual(out["target_audit"][0]["measurement_semantics"], "unsigned_extent")



    def test_named_segment_without_identity_does_not_become_whole_stack(self):
        self.image[150:260, 80:110] = BLUE
        self.image[100:150, 80:110] = GREEN
        self.chart["legend"] = {"entries": []}
        t = target(series="unknown", measurement="height")
        t["bar_scope"] = "segment"
        out = self.run_targets([t])
        self.assertEqual(out["status"], "measurement_failed")
        self.assertEqual(out["geometry_values"], [None])
        self.assertEqual(out["target_audit"][0]["status_reason"],
                         "segment_series_not_matched_and_no_region")

    def test_whole_stack_without_enclosing_region_does_not_return_segment_height(self):
        self.image[150:260, 80:110] = BLUE
        self.image[100:150, 80:110] = GREEN
        out = self.run_targets([target(series="green", measurement="height")])
        self.assertEqual(out["status"], "measurement_failed")
        self.assertEqual(out["geometry_values"], [None])
        self.assertEqual(out["target_audit"][0]["status_reason"],
                         "whole_stack_requires_enclosing_region")

    def test_broad_segment_region_without_legend_does_not_return_total(self):
        self.image[150:260, 80:110] = BLUE
        self.image[100:150, 80:110] = GREEN
        self.chart["legend"] = {"entries": []}
        t = target(series="", measurement="height", region=[190, 320, 290, 900])
        t["bar_scope"] = "segment"
        out = self.run_targets([t])
        self.assertEqual(out["status"], "measurement_failed")
        self.assertEqual(out["geometry_values"], [None])
        self.assertEqual(out["target_audit"][0]["status_reason"],
                         "segment_region_contains_multiple_colors")


class CalibrationComposition(unittest.TestCase):
    def test_manual_ocr_ticks_fit_both_axes_and_exclude_right_text(self):
        image = np.full((300, 400, 3), 255, np.uint8)
        manual_ocr = []
        for x1, x2, values in ((10, 30, (20, 10, 0)), (370, 395, (1000, 500, 0))):
            for y, value in zip((50, 150, 250), values):
                box = [[x1, y-5], [x2, y-5], [x2, y+5], [x1, y+5]]
                manual_ocr.append([box, str(value), .99])
        class ManualBoxes:
            count = 0
            def readtext(self, *args, **kwargs):
                self.count += 1
                return manual_ocr if self.count == 1 else []
        out = prepare_chart(ManualBoxes(), image)
        self.assertEqual(out["status"], "success", out)
        self.assertEqual(set(out["axes"]), {"left", "right"})
        self.assertLess(out["plot_bbox"][2], 370)
        self.assertEqual(out["axes"]["left"]["plot_bbox"], out["axes"]["right"]["plot_bbox"])
        self.assertAlmostEqual(out["axes"]["left"]["slope"], -.1)
        self.assertAlmostEqual(out["axes"]["right"]["slope"], -5)

    def test_detected_extra_panel_is_capability_failure_not_router_decision(self):
        axes = chart()["axes"]
        detected = {**axes["left"], "token_indices": [0, 1, 2],
                    "right_axis": None,
                    "additional_y_axes": [{"token_indices": [6, 7, 8], "tick_count": 3}]}
        class EmptyOCR:
            def readtext(self, *args, **kwargs):
                return []
        with patch("geometry_toolbox.axis_localizer", return_value=detected):
            out = prepare_chart(EmptyOCR(), np.full((300, 400, 3), 255, np.uint8))
        self.assertEqual(out["status"], "capability_failure")
        self.assertEqual(out["status_reason"], "multiple_panels_or_additional_axes")
        self.assertNotIn("need_geometry", out)


if __name__ == "__main__":
    unittest.main()
