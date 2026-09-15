"""CPU checks for the actual chart_context nesting failure."""
import copy
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vlm_semantic_planner import normalize_plan
legacy_path = ROOT.parent / "20260913_visual_feedback_v3/src/vlm_semantic_planner.py"
spec = importlib.util.spec_from_file_location("v3_planner_reference", legacy_path)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)

def base():
    return dict(chart_context="Revenue, monthly observations, USD million.",
                chart_type="line", y_axis_count=1, has_direct_value_labels=False,
                operation="direct", targets=[dict(x_label="Jan-2021", series="Revenue")],
                calculations=[], reason="Read the requested observation.")

class NestedPlanContractTests(unittest.TestCase):
    def test_top_level_plan_keeps_original_meaning(self):
        original = base()
        original["chart_context"] = {"y_axis_unit": "USD million", "unknown_metadata": ["retained"]}
        untouched = copy.deepcopy(original)
        self.assertEqual(normalize_plan(original, ""), legacy.normalize_plan(original, ""))
        self.assertEqual(original, untouched)

    def test_nested_plan_recovers_targets_and_preserves_metadata(self):
        semantic = base()
        semantic.pop("chart_context")
        context = dict(semantic, x_axis_labels=["Jan-2021"], y_axis_unit="USD million",
                       custom_chart_note={"scale": "linear"})
        supplied = {"chart_context": context}
        untouched = copy.deepcopy(supplied)
        result = normalize_plan(supplied, "")
        self.assertTrue(result["route_geometry"])
        self.assertEqual(result["targets"][0]["x_label"], "Jan-2021")
        self.assertIn("custom_chart_note", result["chart_context"])
        self.assertIn("USD million", result["chart_context"])
        self.assertNotIn("targets", result["chart_context"])
        self.assertEqual(supplied, untouched)

    def test_conflicting_explicit_fields_are_not_silently_selected(self):
        for key, conflicting in [("operation", "mean"), ("has_direct_value_labels", True),
                                 ("targets", [{"x_label": "Feb-2021", "series": "Revenue"}])]:
            with self.subTest(key=key):
                original = base()
                original["chart_context"] = {key: conflicting}
                with self.assertRaisesRegex(ValueError, "conflicting_nested_plan_field:" + key):
                    normalize_plan(original, "")

    def test_unwrap_does_not_correct_semantics_or_relax_contract(self):
        for key, supplied, expected in [
            ("has_direct_value_labels", True, "direct_value_labels_or_unknown"),
            ("operation", "sequence", "missing_measurements"),
            ("targets", [{"x_label": "date", "series": "Revenue"}], "placeholder_target_label"),
        ]:
            with self.subTest(key=key):
                semantic = base()
                semantic.pop("chart_context")
                semantic[key] = supplied
                result = normalize_plan({"chart_context": semantic}, "")
                self.assertFalse(result["route_geometry"])
                self.assertEqual(result["route_reason"], expected)

if __name__ == "__main__":
    unittest.main()
