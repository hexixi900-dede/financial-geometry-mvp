"""Verify routing and scoring policy without loading either model."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_question_router as run
from evaluate_question_router import evaluate
from run_unified_resident import write_rows, load_log


class RoutingFlow(unittest.TestCase):
    def test_negative_skips_tools_positive_failure_answers_and_errors_never_use_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); out=root/'pilot'; out.mkdir()
            sources=[dict(sample_id=str(i),chart_id=str(i),image_path='/tmp/original.png',
                          question='Question',question_type='single_choice',options='A: yes\nB: no') for i in range(3)]
            write_rows(out/'geometry_inputs.jsonl',sources)
            raw=[dict(sample_id=str(i),question_type='single_choice',prediction='A',reference='A') for i in range(3)]
            metrics=[dict(r,official_code_correct=True) for r in raw]
            write_rows(root/'raw.jsonl',raw);write_rows(root/'metrics.jsonl',metrics)
            args=SimpleNamespace(root=root,batch_size=3,raw_predictions=root/'raw.jsonl',raw_metrics=root/'metrics.jsonl')
            decisions=iter([False,True,None])
            def route(runtime,args,source):
                need=next(decisions)
                return {**run.audit(source),'needs_geometry':need,'status':'success' if need is not None else 'router_error'}
            def measure(args,out,sources,plan_name,name):
                self.assertEqual([s['sample_id'] for s in sources],['1'])
                return {'1':{**run.audit(sources[0]),'status':'capability_failure',
                             'semantic_plan':{},'geometry_values':[],'capability_status':'unsupported'}}
            def review(runtime,args,source,m,out,attempt):
                self.assertEqual(source['sample_id'],'1')
                self.assertEqual(m['status'],'capability_failure')
                return {**run.audit(source),'status':'success','prediction':'B','prompt':'evidence',
                        'messages':[],'overlay_path':'/tmp/overlay.png'}
            with patch.object(run,'route_question',side_effect=route) as route_mock, \
                 patch.object(run,'plan_targets',return_value={**run.audit(sources[1]),'status':'success','plan':{'plan_valid':True}}) as plan_mock, \
                 patch.object(run,'measure',side_effect=measure),patch.object(run,'review',side_effect=review) as review_mock:
                run.run_cohort(args,'pilot',object())
                self.assertEqual(plan_mock.call_count,1)
                self.assertEqual(review_mock.call_count,1)
                run.run_cohort(args,'pilot',object())
                self.assertEqual(route_mock.call_count,3)  # Terminal resume performs no more inference.
                self.assertEqual(review_mock.call_count,1)
            outcomes=load_log(out/'outcomes.jsonl')
            self.assertEqual([outcomes[str(i)]['route'] for i in range(3)],['raw','framework','router_error'])
            scored=load_log(out/'predictions_final.jsonl')
            self.assertEqual([scored[str(i)]['hybrid_correct'] for i in range(3)],[1,0,0])
            self.assertEqual(scored['1']['prediction'],'B')
            self.assertIsNone(scored['2']['prediction'])
            self.assertEqual(json.loads((out/'summary.json').read_text())['measurement_none'],1)
            # An invalid enhanced answer is still an error even if Raw was correct.
            outcomes['1'].update(status='answer_parse_failed',prediction=None)
            write_rows(out/'outcomes.jsonl',outcomes.values())
            evaluate(out,root/'metrics.jsonl',root/'raw.jsonl')
            self.assertIsNone(load_log(out/'predictions_final.jsonl')['1']['prediction'])

    def test_invalid_router_boolean_is_retried_once_then_explicit_error(self):
        args=SimpleNamespace(min_pixels=200704,max_pixels=802816)
        source=dict(sample_id='0',chart_id='x',question='Unlabelled increase?',image_path='/tmp/chart.png')
        class BrokenOutput:
            calls=0
            def generate(self,msg,max_tokens):
                self.calls+=1
                return '{"needs_geometry":"false","reason":"not sure"}'
        runtime=BrokenOutput()
        decision=run.route_question(runtime,args,source)
        self.assertEqual(runtime.calls,2)
        self.assertIsNone(decision['needs_geometry'])
        self.assertEqual(decision['status'],'router_error')


if __name__=='__main__':unittest.main()
