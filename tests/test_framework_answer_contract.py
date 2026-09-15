import copy
import json
import unittest

import framework_answer as answer


def source():
    return {"question_type": "single_choice", "question": "Which claim matches the bars and line?",
            "options": "A: Revenue rose\nB: Margin doubled", "unit": "",
            "image_path": "/tmp/original.png", "vlm_image_path": "/tmp/original.png"}


def partial():
    targets = [
        {"x_label": "2021", "series": "Revenue", "mark_type": "bar", "axis_id": "left", "bar_scope": "whole"},
        {"x_label": "2022", "series": "Revenue", "mark_type": "bar", "axis_id": "left", "bar_scope": "whole"},
        {"x_label": "2022", "series": "Margin", "mark_type": "line", "axis_id": "right"},
    ]
    left = {"ticks": [{"text": "0", "value": 0}, {"text": "100", "value": 100}], "unit": "USD million"}
    right = {"ticks": [{"text": "0%", "value": 0}, {"text": "20%", "value": 20}], "unit": "%"}
    return {"status": "partial", "status_reason": "one target missing", "geometry_values": [80, None, 12],
            "semantic_plan": {"chart_context": "Left revenue; right margin", "chart_type": "bar_line",
              "y_axis_count": 2, "targets": targets, "operation": "difference",
              "calculations": [{"name": "increase", "expression":
                {"op": "difference", "args": [{"target": 1}, {"target": 0}]}}],
              "reason": "Both series matter", "plan_valid": True},
            "y_axis_ticks": [{"text": "WRONG GLOBAL TICK", "value": 999}],
            "numeric_answer": -100, "reasoning": {"calculations": []},
            "target_audit": [
                {"target_index": 0, "axis_id": "left", "mark_type": "bar", "status": "success",
                 "value": 80, "axis": left},
                {"target_index": 1, "axis_id": "left", "mark_type": "bar", "status": "failed",
                 "value": 9999, "axis": left, "status_reason": "no component matched"},
                {"target_index": 2, "axis_id": "right", "mark_type": "line", "status": "success",
                 "auto_local_value": 12, "axis": right},
            ]}


class FrameworkAnswerContractTest(unittest.TestCase):
    def test_dual_axis_partial_keeps_target_indices_and_axis_provenance(self):
        measured = partial()
        # Out-of-order audit serialization must not reassign a value to another target.
        measured["target_audit"].reverse()
        evidence = answer.evidence(source(), measured)
        reads = evidence["measurements"]
        self.assertEqual([r["target_index"] for r in reads], [0, 1, 2])
        self.assertEqual([r["measured_value"] for r in reads], [80, None, 12])
        self.assertEqual([r["axis_id"] for r in reads], ["left", "left", "right"])
        self.assertEqual(reads[0]["axis_tick_texts_and_values"][-1]["value"], 100)
        self.assertEqual(reads[2]["axis_tick_texts_and_values"][-1]["value"], 20)
        self.assertIsNone(evidence["python_result"])
        self.assertEqual(evidence["operation"], "evidence")
        self.assertNotIn("WRONG GLOBAL TICK", json.dumps(evidence))

    def test_input_boundary_original_images_and_shared_answer_parser(self):
        base = source()
        contaminated = {**base, "gold": "SECRET_GOLD", "caption": "SECRET_CAPTION",
                        "raw_prediction": "SECRET_CACHED_ANSWER"}
        actual = answer.messages(contaminated, partial(), "/tmp/overlay.png", 200704, 451584)
        self.assertEqual(actual, answer.messages(base, partial(), "/tmp/overlay.png", 200704, 451584))
        images = [item for item in actual[1]["content"] if item["type"] == "image"]
        self.assertEqual([i["image"] for i in images], ["file:///tmp/original.png", "file:///tmp/overlay.png"])
        self.assertEqual([(i["min_pixels"], i["max_pixels"]) for i in images],
                         [(200704, 451584), (200704, 451584)])
        prediction, status, _ = answer.parse_reply('{"status":"answered","answer":"A"}', base)
        self.assertEqual((prediction, status), ("A", "success"))
        plan = partial()["semantic_plan"]
        prediction, status, parsed = answer.parse_reply(json.dumps(
            {"status": "revise_plan", "reason": "repair right axis", "plan": plan}), base)
        self.assertEqual(status, "revision_requested")
        self.assertEqual(parsed["plan"]["targets"][2]["axis_id"], "right")

    def test_invalid_plan_preserves_failure_for_repair_without_fake_readings(self):
        failure = {"status": "plan_parse_failed", "planner_error": "missing_target_fields",
                   "planner_raw_text": '{"targets":[]}', "semantic_plan": {}}
        data = answer.evidence(source(), failure)
        self.assertEqual(data["planner_error"], "missing_target_fields")
        self.assertEqual(data["planner_raw_text"], '{"targets":[]}')
        self.assertEqual(data["measurements"], [])
        self.assertIsNone(data["python_result"])


if __name__ == "__main__":
    unittest.main()
