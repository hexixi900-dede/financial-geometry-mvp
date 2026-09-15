"""Terminal VLM answers survive failed measurements without cache substitution."""
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import run_visual_feedback as runner

class NoFallbackRunner(unittest.TestCase):
    def run_case(self, initial_reply, final_reply, direct=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        root=Path(temp.name);out=root/'full';out.mkdir()
        source={'sample_id':'1','chart_id':'chart','question':'Read the chart',
                'question_type':'single_choice','options':'A: first\nB: second'}
        plan={'chart_type':'line','y_axis_count':1,'has_direct_value_labels':direct,
              'operation':'direct','targets':[{'x_label':'2023','series':''}]}
        runner.write_rows(out/'geometry_inputs.jsonl',[source])
        runner.write_rows(out/'plans.jsonl',[dict(sample_id='1',chart_id='chart',status='success',plan=plan)])
        measurement=dict(sample_id='1',chart_id='chart',status='line_localization_failed',semantic_plan=plan)
        args=Namespace(root=root,batch_size=64)
        replies=[]
        def review(runtime,args,src,m,out,attempt):
            replies.append((attempt,m['status']))
            return dict(sample_id='1',chart_id='chart',prompt='original image plus available evidence',
                        messages=[],overlay_path='overlay.png',attempt=attempt,
                        **(initial_reply if attempt==0 else final_reply))
        with patch.object(runner,'measure',return_value={'1':measurement}), \
             patch.object(runner,'review',side_effect=review), \
             patch.object(runner,'normalize_plan',return_value=plan), \
             patch.object(runner,'evaluate'):
            runner.run_cohort(args,'full',object())
            count=len(replies)
            runner.run_cohort(args,'full',object())
            self.assertEqual(len(replies),count,'Completed records must not trigger more model calls')
        return runner.load_log(out/'measurements.jsonl')['1'],runner.load_log(out/'replies.jsonl').get('1'),replies

    def test_failed_revision_still_gets_final_vlm_answer(self):
        m,r,calls=self.run_case(
            dict(status='revision_requested',prediction=None,parsed_reply={'plan':{}},raw_reply='revision'),
            dict(status='success',prediction='B',parsed_reply={'status':'answered','answer':'B'}))
        self.assertEqual(calls,[(0,'line_localization_failed'),(1,'line_localization_failed')])
        self.assertTrue(m['geometry_entered']);self.assertTrue(r['plan_revised'])
        self.assertEqual(r['status'],'success');self.assertEqual(r['prediction'],'B')

    def test_invalid_final_answer_stays_failed_for_scorer(self):
        m,r,calls=self.run_case(dict(status='insufficient_evidence',prediction=None),
                                dict(status='answer_parse_failed',prediction=None))
        self.assertEqual(len(calls),2);self.assertTrue(m['geometry_entered'])
        self.assertEqual(r['status'],'answer_parse_failed');self.assertIsNone(r['prediction'])

    def test_initial_direct_label_route_does_not_enter_geometry_feedback(self):
        m,r,calls=self.run_case({}, {}, direct=True)
        self.assertEqual(calls,[]);self.assertIsNone(r)
        self.assertFalse(m['geometry_entered']);self.assertEqual(m['feedback_status'],'raw_route')

if __name__=='__main__':unittest.main()
