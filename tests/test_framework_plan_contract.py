"""CPU contracts for independent routing and mixed chart target planning."""
import copy
import unittest

import framework_plan as planner
import vlm_need_router as router
from phase9_plan_contract import expression


def mixed_plan():
    return {
        "chart_context": "Revenue bars use the left USD axis; Margin line uses right percent axis.",
        "chart_type": "bar_line", "y_axis_count": 2, "operation": "difference",
        "targets": [
            {"x_label": "2021", "series": "Revenue", "mark_type": "bar", "axis_id": "left",
             "bar_scope": "whole", "measurement": "value"},
            {"x_label": "2022", "series": "Revenue", "mark_type": "bar", "axis_id": "left",
             "bar_scope": "whole", "measurement": "value"},
            {"x_label": "2022", "series": "Margin", "mark_type": "line", "axis_id": "right"},
        ],
        "calculations": [{"name": "Revenue increase", "expression": {
            "op": "difference", "args": [{"target": 1}, {"target": 0}]}}],
        "reason": "Compare revenue change with the separate margin reading.",
    }


class IndependentContractTest(unittest.TestCase):
    def test_input_boundary_and_original_resolution(self):
        base = {"question": "What was revenue?", "options": "A: 10; B: 20",
                "unit": "USD million", "image_path": "/tmp/geometry_thumbnail.jpg",
                "vlm_image_path": "/tmp/original_raw.png"}
        enriched = {**base, "caption": "FORBIDDEN_DATASET_CAPTION", "gold": "SECRET_GOLD",
                    "raw_prediction": "SECRET_RAW", "chart_type": "other",
                    "y_axis_count": 2, "has_direct_value_labels": True,
                    "route_geometry": False, "structural_candidate": False}
        for module in (router, planner):
            actual = module.messages(enriched, 200704, 451584)
            self.assertEqual(actual, module.messages(base, 200704, 451584))
            image = actual[1]["content"][0]
            self.assertEqual(image["image"], "file:///tmp/original_raw.png")
            self.assertEqual((image["min_pixels"], image["max_pixels"]), (200704, 451584))
            changed = dict(base, question="Which title is shown?")
            self.assertNotEqual(actual, module.messages(changed, 200704, 451584))

    def test_router_classifies_evidence_not_arithmetic(self):
        for invalid in ({}, {"needs_geometry":False}, {"evidence_source":None},
                        {"evidence_source":"unknown"}, {"evidence_source":True}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                router.normalize_router(invalid)
        self.assertTrue(router.normalize_router({"evidence_source":"coordinate_reading"})["needs_geometry"])
        # A stray contradictory boolean cannot override the actual source decision.
        printed=router.normalize_router({"evidence_source":"printed_values","needs_geometry":True,
                                         "reason":"Subtract the two printed values."})
        self.assertFalse(printed["needs_geometry"])
        self.assertFalse(router.normalize_router({"evidence_source":"qualitative"})["needs_geometry"])

    def test_mixed_marks_dual_axis_and_single_arithmetic_result(self):
        normalized = planner.normalize_plan(mixed_plan())
        self.assertTrue(normalized["plan_valid"])
        self.assertNotIn("route_geometry", normalized)
        self.assertNotIn("needs_geometry", normalized)
        self.assertEqual([t["axis_id"] for t in normalized["targets"]], ["left", "left", "right"])
        self.assertEqual(normalized["operation"], "evidence")
        self.assertEqual(expression(normalized["calculations"][0]["expression"], [100, 125, 17]), 25)
        self.assertIsNone(planner.calculate(normalized["operation"], [100, 125, 17]))
        self.assertEqual(planner.normalize_plan(normalized), normalized)
        missing_axis = mixed_plan()
        del missing_axis["targets"][2]["axis_id"]
        with self.assertRaises(ValueError):
            planner.normalize_plan(missing_axis)
        malformed_enum = mixed_plan()
        malformed_enum["targets"][2]["axis_id"] = ["right"]
        with self.assertRaises(ValueError):
            planner.normalize_plan(malformed_enum)
        malformed_expression = mixed_plan()
        malformed_expression["calculations"][0]["expression"]["op"] = ["difference"]
        with self.assertRaises(ValueError):
            planner.normalize_plan(malformed_expression)

    def test_nested_recovery_preserves_indices_and_rejects_conflict(self):
        flat = mixed_plan()
        nested = {"chart_context": {"description": flat["chart_context"],
                                    **{k: v for k, v in flat.items() if k != "chart_context"}}}
        recovered = planner.normalize_plan(nested)
        self.assertEqual(recovered["targets"], planner.normalize_plan(flat)["targets"])
        self.assertEqual(recovered["calculations"], flat["calculations"])
        with self.assertRaisesRegex(ValueError, "conflicting_nested_plan_field:targets"):
            planner.normalize_plan({**nested, "targets": []})

    def test_segment_and_discrete_mean_integrity(self):
        plan = mixed_plan()
        plan.update(chart_type="bar", operation="direct", calculations=[],
                    targets=[{**plan["targets"][0], "series": "Export", "bar_scope": "segment",
                              "measurement": "height", "region": [100, 100, 200, 250]}])
        self.assertTrue(planner.normalize_plan(plan)["plan_valid"])
        plan["targets"][0]["measurement"] = "value"
        with self.assertRaisesRegex(ValueError, "stack_segment_requires_height"):
            planner.normalize_plan(plan)
        plan.update(chart_type="line", operation="mean",
                    targets=[{"x_label": "Jan-Dec 2022", "series": "Revenue", "mark_type": "line",
                              "axis_id": "left", "measurement": "sequence"}])
        with self.assertRaisesRegex(ValueError, "sequence_requires_evidence"):
            planner.normalize_plan(plan)
        plan["calculations"] = [{"name": "Wrong pixel mean", "expression":
                                 {"op": "mean", "args": [{"target": 0}]}}]
        with self.assertRaisesRegex(ValueError, "sequence_requires_extremum_calculation"):
            planner.normalize_plan(plan)
        plan["targets"] = [{"x_label": month + " 2022", "series": "Revenue", "mark_type": "line",
                            "axis_id": "left"} for month in ("Jan", "Feb", "Mar", "Apr", "May",
                            "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")]
        plan["calculations"] = []
        normalized = planner.normalize_plan(plan)
        self.assertEqual(planner.calculate(normalized["operation"], list(range(1, 13))), 6.5)
        malformed = copy.deepcopy(plan)
        malformed["targets"][0]["guessed_value"] = 99
        with self.assertRaisesRegex(ValueError, "unexpected_target_keys"):
            planner.normalize_plan(malformed)


if __name__ == "__main__":
    unittest.main()
