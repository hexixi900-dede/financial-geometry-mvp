"""Verify no-Raw-fallback scoring without loading a VLM or consulting Gold during inference."""
import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from vlm_evidence_answerer import evaluate


class NoRawFallbackEvaluation(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.metrics = [
            dict(sample_id='0', question_type='single_choice', prediction='B', reference='A',
                 official_code_correct=False),
            dict(sample_id='1', question_type='multiple_choice', prediction='AC', reference='AC',
                 official_code_correct=True),
            dict(sample_id='2', question_type='numerical', prediction=10, reference=10,
                 official_code_correct=True),
            dict(sample_id='3', question_type='single_choice', prediction='C', reference='C',
                 official_code_correct=True),
            dict(sample_id='4', question_type='multiple_choice', prediction='AC', reference='AC',
                 official_code_correct=True),
            dict(sample_id='5', question_type='single_choice', prediction='A', reference='A',
                 official_code_correct=True),
            dict(sample_id='6', question_type='single_choice', prediction='A', reference='A',
                 official_code_correct=True),
        ]
        self.args = Namespace(**{name: self.root / (name + '.jsonl')
                                for name in ('inputs', 'gold', 'baseline', 'replies',
                                             'prompts', 'raw_metrics', 'measurements')},
                              output=self.root / 'summary.json', raw_predictions=None,
                              no_raw_fallback=True)
        candidates = self.metrics[:6]
        self.write('inputs', [dict(sample_id=r['sample_id'], question_type=r['question_type'])
                              for r in candidates])
        self.write('gold', [dict(sample_id=r['sample_id'], reference=r['reference'], tolerance=0.2)
                            for r in candidates])
        self.write('baseline', [dict(sample_id=r['sample_id'],
                                     raw_vlm_correct=int(r['official_code_correct']))
                                for r in candidates])
        self.write('raw_metrics', self.metrics)
        self.write('prompts', [dict(sample_id=str(i)) for i in (0, 1, 2, 4)])
        self.measurements = [
            dict(sample_id='0', status='success', geometry_entered=True),
            dict(sample_id='1', status='measurement_failed', geometry_entered=True),
            dict(sample_id='2', status='measurement_failed', geometry_entered=True),
            dict(sample_id='3', status='raw_route', geometry_entered=False),
            dict(sample_id='4', status='revision_failed', geometry_entered=True),
        ]
        self.write('measurements', self.measurements)
        self.replies = [
            dict(sample_id='0', status='success', prediction='A'),
            dict(sample_id='1', status='success', prediction='AB'),
            dict(sample_id='2', status='invalid_answer', prediction=None),
            dict(sample_id='4', status='model_error', prediction=None),
        ]
        self.write('replies', self.replies)

    def write(self, name, records):
        getattr(self.args, name).write_text(''.join(json.dumps(r) + '\n' for r in records))

    def score(self):
        with contextlib.redirect_stdout(io.StringIO()):
            summary = evaluate(self.args)
        records = [json.loads(s) for s in
                   (self.root / 'full_predictions.jsonl').read_text().splitlines()]
        return summary, records

    def test_final_failures_are_wrong_and_raw_remains_only_as_comparison(self):
        summary, records = self.score()
        self.assertEqual([r['prediction'] for r in records], ['A', 'AB', None, 'C', None, 'A', 'A'])
        self.assertEqual([r['route'] for r in records],
                         ['geometry_enhanced', 'geometry_enhanced', 'geometry_failed', 'raw',
                          'geometry_failed', 'raw', 'raw'])
        self.assertEqual(records[2]['raw_prediction'], 10)
        self.assertEqual(records[2]['raw_correct'], 1)
        self.assertEqual(records[2]['selective_correct'], 0)
        self.assertFalse(records[2]['prediction_available'])
        self.assertTrue(records[2]['raw_prediction_available'])
        # The VLM's final choice is still scored even after a failed measurement.
        self.assertEqual(records[1]['prediction'], 'AB')
        self.assertEqual(summary['answered'], 2)
        self.assertEqual(summary['geometry_failed'], 2)
        self.assertEqual(summary['geometry_failure_counts'], {'invalid_answer': 1, 'model_error': 1})
        self.assertEqual(summary['full_raw_correct'], 6)
        self.assertEqual(summary['full_selective_correct'], 4)
        self.assertEqual(summary['main_table'][1]['Calculation'], 0)
        self.assertEqual(summary['raw_fallback_policy'], 'disabled_after_geometry_entry')
        self.assertEqual(summary['evaluation_status'], 'partial')
        self.assertTrue(summary['score_provisional'])
        self.assertTrue(records[5]['score_provisional'])
        self.assertEqual(summary['pending'], 1)

    def test_pending_entered_geometry_is_not_counted_as_terminal_failure(self):
        self.write('replies', self.replies[:-1])
        summary, records = self.score()
        self.assertEqual(records[4]['route'], 'geometry_pending')
        self.assertIsNone(records[4]['prediction'])
        self.assertTrue(records[4]['score_provisional'])
        self.assertEqual(summary['geometry_pending'], 1)
        self.assertEqual(summary['geometry_failed'], 1)
        self.assertEqual(summary['pending'], 2)
        self.assertEqual(summary['pending_answers'], 1)

    def test_completed_initial_raw_routes_preserve_raw_and_clear_provisional_status(self):
        self.write('measurements', self.measurements +
                   [dict(sample_id='5', status='raw_route', geometry_entered=False)])
        summary, records = self.score()
        self.assertEqual(summary['evaluation_status'], 'complete')
        self.assertFalse(summary['score_provisional'])
        self.assertEqual(summary['raw_routed'], 2)
        self.assertEqual(records[3]['prediction'], 'C')
        self.assertEqual(records[6]['route'], 'raw')

    def test_old_default_keeps_cached_raw_fallback(self):
        del self.args.no_raw_fallback
        summary, records = self.score()
        self.assertEqual(records[2]['prediction'], 10)
        self.assertEqual(records[4]['prediction'], 'AC')
        self.assertEqual(records[2]['route'], 'raw')
        self.assertEqual(summary['geometry_failed'], 0)
        self.assertEqual(summary['raw_fallback_policy'], 'cached_raw_on_unusable_evidence')

    def test_missing_entry_decision_fails_instead_of_silently_restoring_raw(self):
        self.measurements[2].pop('geometry_entered')
        self.write('measurements', self.measurements)
        with self.assertRaisesRegex(ValueError, 'boolean geometry_entered'):
            self.score()


if __name__ == '__main__':
    unittest.main()
