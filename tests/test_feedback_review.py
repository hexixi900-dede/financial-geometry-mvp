"""Regression checks for actual feedback marks and unambiguous answer formats."""
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from geometry_visual_feedback import render_measurement_overlay
from vlm_evidence_answerer import build_prompt, parse_reply


class FeedbackReview(unittest.TestCase):
    def source(self, kind='multiple_choice'):
        return dict(question_type=kind, question='Which statements hold?',
                    options='A: first\nB: second\nC: third', unit='million')

    def test_explicit_choice_formats_keep_the_same_decision(self):
        for kind, answer, expected in (
            ('single_choice', ['C'], 'C'),
            ('multiple_choice', 'AC', 'AC'),
            ('multiple_choice', 'C, A', 'AC'),
            ('multiple_choice', ['A', 'C'], 'AC'),
        ):
            with self.subTest(kind=kind, answer=answer):
                raw = json.dumps(dict(status='answered', answer=answer))
                self.assertEqual(parse_reply(raw, self.source(kind))[:2], (expected, 'success'))
        for kind, answer in (
            ('single_choice', ['A', 'C']), ('single_choice', []),
            ('multiple_choice', 'A and C'), ('multiple_choice', 'A,A'),
            ('multiple_choice', 'A,Z'), ('multiple_choice', 'Because C is true'),
            ('numerical', '70 MYR bn'),
        ):
            with self.subTest(kind=kind, answer=answer):
                raw = json.dumps(dict(status='answered', answer=answer))
                self.assertEqual(parse_reply(raw, self.source(kind))[1], 'answer_parse_failed')
        raw = '{"status":"insufficient_evidence","answer":"AC"}'
        self.assertEqual(parse_reply(raw, self.source())[:2], (None, 'insufficient_evidence'))

    def test_overlay_keeps_unmeasured_gap_and_isolated_sample_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            original = path / 'original.png'
            Image.new('RGB', (200, 100), 'white').save(original)
            source = {'image_path': str(original)}
            def render(trace, name):
                measurement = dict(status='success', pipeline_audit=[{
                    'auto_local_geometry': {'trace_points': trace}}])
                output = render_measurement_overlay(source, measurement, path / name)
                return Image.open(output).convert('RGB')
            measured_color = (225, 29, 72)
            with render([[10, 40], [11, 40], [80, 40], [81, 40]], 'gap.png') as overlay:
                self.assertEqual(overlay.getpixel((10, 40)), measured_color)
                self.assertEqual(overlay.getpixel((45, 40)), (255, 255, 255))
                self.assertEqual(overlay.getpixel((80, 40)), measured_color)
            with render([[20, 20]], 'one.png') as overlay:
                self.assertEqual(overlay.getpixel((20, 20)), measured_color)

    def test_prompt_distinguishes_wrong_evidence_and_ordinary_approximation(self):
        success = build_prompt(self.source(), {'status': 'success'}, allow_revision=True)
        self.assertIn('Do not copy a measurement or Python result', success)
        self.assertIn('normal\npixel approximation alone is not a reason', success)
        self.assertIn('Requested answer unit: million', success)
        failed = build_prompt(self.source(), {'status': 'line_localization_failed'}, allow_revision=True)
        self.assertIn('request the complete corrected plan', failed)
        final = build_prompt(self.source(), {'status': 'line_localization_failed'}, allow_revision=False)
        self.assertIn('No further revision is available', final)
        self.assertNotIn('"status":"revise_plan"', final)
        self.assertIn('Return insufficient_evidence.', final)


if __name__ == '__main__':
    unittest.main()
