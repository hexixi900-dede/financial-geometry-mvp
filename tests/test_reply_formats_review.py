"""Accept observed bare choice arrays without weakening answer validation."""
import unittest

from vlm_evidence_answerer import parse_reply


class ReplyFormatReview(unittest.TestCase):
    def source(self, kind):
        return dict(question_type=kind, options='A: one\nB: two\nC: three\nD: four')

    def test_observed_bare_choice_arrays(self):
        # These are the six observed parse-failure payloads, in source order.
        for raw, kind, expected in (
            ('["C"]', 'single_choice', 'C'),
            ('["B"]', 'single_choice', 'B'),
            ('["D"]', 'single_choice', 'D'),
            ('["A"]', 'single_choice', 'A'),
            ('["A", "C"]', 'multiple_choice', 'AC'),
            ('["D"]', 'single_choice', 'D'),
        ):
            with self.subTest(raw=raw, kind=kind):
                self.assertEqual(parse_reply(raw, self.source(kind))[:2], (expected, 'success'))

    def test_arrays_retain_cardinality_and_option_validation(self):
        for raw, kind in (
            ('[]', 'single_choice'), ('["A","C"]', 'single_choice'),
            ('[]', 'multiple_choice'), ('["A","A"]', 'multiple_choice'),
            ('["Z"]', 'single_choice'), ('["A","Z"]', 'multiple_choice'),
            ('[1.23]', 'numerical'), ('["A"', 'single_choice'),
        ):
            with self.subTest(raw=raw, kind=kind):
                self.assertEqual(parse_reply(raw, self.source(kind))[1], 'answer_parse_failed')

    def test_explicit_insufficient_evidence_is_not_promoted(self):
        raw = '{"status":"insufficient_evidence","answer":"A"}'
        self.assertEqual(parse_reply(raw, self.source('single_choice'))[:2],
                         (None, 'insufficient_evidence'))

    def test_existing_object_answer_still_uses_same_validation(self):
        raw = '{"status":"answered","answer":["C","A"]}'
        self.assertEqual(parse_reply(raw, self.source('multiple_choice'))[:2], ('AC', 'success'))


if __name__ == '__main__':
    unittest.main()
