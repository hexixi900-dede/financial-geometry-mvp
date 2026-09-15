"""Only exercise full-population replacement, fallback, and completion reporting."""
import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from vlm_evidence_answerer import evaluate


class FullEvaluation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.metrics = [dict(sample_id=str(i), question_type='single_choice', prediction='A',
                             reference='A', official_code_correct=True) for i in range(11099)]
        updates = [dict(prediction='B', official_code_correct=False),
                   dict(question_type='multiple_choice', prediction='AC', reference='AC'),
                   dict(question_type='numerical', prediction=9, reference=10, official_code_correct=False),
                   dict(prediction='C', reference='C'),
                   dict(question_type='multiple_choice', prediction='CA', reference='AC')]
        for i, update in enumerate(updates):
            self.metrics[i].update(update)
        self.inputs = [dict(sample_id=r['sample_id'], question_type=r['question_type']) for r in self.metrics[:5]]
        self.args = Namespace(**{name: self.root / (name + '.jsonl')
                                for name in ('inputs', 'gold', 'baseline', 'replies', 'prompts', 'raw_metrics', 'measurements')},
                              output=self.root / 'summary.json', raw_predictions=None)
        self.write(self.args.inputs, self.inputs)
        self.write(self.args.gold, [dict(sample_id=r['sample_id'], reference=r['reference'], tolerance=0.2)
                                   for r in self.metrics[:5]])
        self.write(self.args.baseline, [dict(sample_id=r['sample_id'], raw_vlm_correct=int(r['official_code_correct']))
                                       for r in self.metrics[:5]])
        self.write(self.args.raw_metrics, self.metrics)
        self.write(self.args.prompts, [dict(sample_id=str(i)) for i in (0, 1, 2, 4)])
        self.write(self.args.measurements, [dict(sample_id=str(i), status='success',
                                               semantic_plan={'has_direct_value_labels': False}) for i in (0, 1, 2, 4)])
        self.write(self.args.replies, [dict(sample_id='0', status='success', prediction='A'),
                                      dict(sample_id='1', status='success', prediction='AB'),
                                      dict(sample_id='2', status='insufficient_evidence', prediction=None),
                                      dict(sample_id='4', status='success', prediction='CA'),
                                      dict(sample_id='5', status='success', prediction='B')])

    def write(self, path, records):
        path.write_text(''.join(json.dumps(r) + '\n' for r in records))

    def score(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return evaluate(self.args)

    def test_full_population_replaces_only_real_successes_and_preserves_raw(self):
        summary = self.score()
        records = [json.loads(line) for line in (self.root / 'full_predictions.jsonl').read_text().splitlines()]
        self.assertEqual(len(records), 11099)
        self.assertEqual([r['prediction'] for r in records[:6]], ['A', 'AB', 9, 'C', 'CA', 'A'])
        self.assertEqual([r['route'] for r in records[:6]],
                         ['geometry_enhanced', 'geometry_enhanced', 'raw', 'raw', 'geometry_enhanced', 'raw'])
        self.assertEqual(summary['rescued'], 1)
        self.assertEqual(summary['harmed'], 1)
        self.assertEqual(summary['full_raw_correct'], 11097)
        self.assertEqual(summary['full_selective_correct'], 11097)
        self.assertEqual(summary['full_selective_accuracy'], 11097 / 11099)
        self.assertEqual(summary['full_by_question_type']['multiple_choice']['selective_accuracy'], 0.5)
        self.assertEqual(summary['main_table'][1]['Overall'], 11097 / 11099)
        self.assertEqual(summary['questions'], 5)
        self.assertEqual(summary['evaluation_status'], 'partial')
        self.assertEqual(summary['pending_measurements'], 1)

    def test_explicit_raw_log_and_completed_raw_route(self):
        self.args.raw_predictions = self.root / 'raw_predictions.jsonl'
        raw_log = [dict(r, metadata={'tolerance': 0.2}) for r in self.metrics]
        self.write(self.args.raw_predictions, raw_log)
        with self.args.measurements.open('a') as handle:
            handle.write(json.dumps(dict(sample_id='3', status='raw_route')) + '\n')
        self.assertEqual(self.score()['evaluation_status'], 'complete')
        # Same correctness alone cannot establish that a prediction cache matches.
        raw_log[0]['prediction'] = 'D'
        self.write(self.args.raw_predictions, raw_log)
        with self.assertRaisesRegex(ValueError, 'Raw prediction/metric mismatch'):
            self.score()

    def test_score_only_cache_does_not_export_invented_answers(self):
        self.score()  # An older real export must not survive a score-only reevaluation.
        for record in self.metrics:
            record.pop('prediction')
        self.write(self.args.raw_metrics, self.metrics)
        self.args.measurements.unlink()
        self.args.measurements = None
        summary = self.score()
        self.assertFalse(summary['full_prediction_exported'])
        self.assertFalse((self.root / 'full_predictions.jsonl').exists())
        self.assertEqual(summary['evaluation_status'], 'completion_unverified')
        self.assertEqual(summary['full_questions'], 11099)


if __name__ == '__main__':
    unittest.main()
