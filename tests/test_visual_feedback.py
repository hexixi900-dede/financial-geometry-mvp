import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vlm_semantic_planner import normalize_plan, SYNTHETIC_PLANNING_EXAMPLES
from vlm_evidence_answerer import parse_reply, build_prompt
from run_visual_feedback import run_cohort, feedback_eligible, final_reply, JOURNALS
from run_unified_resident import write_rows, append, load_log


def plan(label='2021'):
    return dict(chart_context='Revenue is a monthly line in MYR bn.', chart_type='line',
                y_axis_count=1, has_direct_value_labels=False, operation='direct',
                targets=[dict(x_label=label,series='Revenue')], calculations=[])

class VisualFeedback(unittest.TestCase):
    def test_real_dates_and_complete_synthetic_examples(self):
        for example in SYNTHETIC_PLANNING_EXAMPLES:
            p=normalize_plan(example['plan'],example['question'])
            self.assertTrue(p['route_geometry'])
            self.assertEqual(p['chart_context'],example['plan']['chart_context'])
        p=normalize_plan(plan('date'),'Revenue in 2021?')
        self.assertEqual(p['route_reason'],'placeholder_target_label')
        self.assertTrue(feedback_eligible({'plan':p},{'semantic_plan':p}))
        p['has_direct_value_labels']=True
        self.assertFalse(feedback_eligible({'plan':p},{}))

    def test_feedback_schema_units_images_and_no_gold(self):
        from PIL import Image
        from geometry_visual_feedback import build_feedback_messages, render_measurement_overlay
        with tempfile.TemporaryDirectory() as d:
            image=Path(d)/'original.png';Image.new('RGB',(200,100),'white').save(image)
            source=dict(image_path=str(image),vlm_image_path=str(image),question='Revenue?',
                        question_type='numerical',options='',unit='MYR bn',reference='SECRET_GOLD')
            m=dict(status='line_localization_failed',semantic_plan=normalize_plan(plan(),'Revenue?'))
            output=render_measurement_overlay(source,m,Path(d)/'overlay.png')
            msg=build_feedback_messages(source,m,output,200704,802816)
            self.assertNotIn('SECRET_GOLD',json.dumps(msg))
            images=[p for p in msg[1]['content'] if p['type']=='image']
            self.assertEqual(len(images),2)
            self.assertEqual(images[0]['max_pixels'],802816)
            self.assertIn('revise_plan',build_prompt(source,m,allow_revision=True))
            raw=json.dumps(dict(status='revise_plan',plan=plan(),reason='wrong point'))
            pred,status,obj=parse_reply(raw,source)
            self.assertEqual(status,'revision_requested');self.assertIsNone(pred)
            terminal=final_reply(dict(status=status,prediction=None),dict(status='success'),True)
            self.assertEqual(terminal['status'],'revision_limit_reached')
            terminal=final_reply(dict(status='success',prediction=1),m)
            self.assertIsNone(terminal['prediction'])

    def test_repair_then_answer_and_restart_reuses_saved_feedback(self):
        from PIL import Image
        import run_visual_feedback as runner
        class FakeRuntime:
            calls=0
            def generate(self,msg,tokens):
                self.calls+=1
                if msg[0]['content'].startswith('Plan semantic'):
                    return json.dumps(plan('date'))
                if self.calls==2:
                    return json.dumps(dict(status='revise_plan',reason='use real year',plan=plan('2021')))
                return '{"status":"answered","answer":["A","C"]}'
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);out=root/'pilot';out.mkdir()
            image=root/'image.png';Image.new('RGB',(200,100),'white').save(image)
            src=dict(sample_id='1',chart_id='c',image_path=str(image),question='Revenue in 2021?',
                     question_type='multiple_choice',options='A: one\nB: two\nC: three')
            write_rows(out/'geometry_inputs.jsonl',[src])
            args=SimpleNamespace(root=root,batch_size=64,min_pixels=200704,max_pixels=802816,model='fake')
            runtime=FakeRuntime();measure_calls=[];interrupt=[True]
            def cpu(args,out,batch,plans_path,name):
                measure_calls.append(name)
                if name=='measurements_revision' and interrupt[0]:
                    interrupt[0]=False;raise RuntimeError('simulated interruption after feedback checkpoint')
                completed=load_log(out/(name+'.jsonl'))
                for src in batch:
                    if src['sample_id'] in completed:continue
                    p=load_log(plans_path)[src['sample_id']]['plan']
                    status='success' if p['route_geometry'] else 'semantic_plan_rejected'
                    record=dict(sample_id=src['sample_id'],status=status,semantic_plan=p,
                                geometry_values=[70] if status=='success' else [],numeric_answer=70)
                    append(out/(name+'.jsonl'),record)
                return load_log(out/(name+'.jsonl'))
            with patch.object(runner,'measure',side_effect=cpu),patch.object(runner,'evaluate'):
                with self.assertRaises(RuntimeError):run_cohort(args,'pilot',runtime)
                self.assertEqual(runtime.calls,2)
                run_cohort(args,'pilot',runtime)
                self.assertEqual(runtime.calls,3)
                answer=load_log(out/'replies.jsonl')['1']
                self.assertEqual(answer['prediction'],'AC');self.assertTrue(answer['plan_revised'])
                self.assertEqual(load_log(out/'measurements.jsonl')['1']['semantic_plan']['targets'][0]['x_label'],'2021')
                self.assertEqual(len(load_log(out/'feedback_initial.jsonl')),1)
                run_cohort(args,'pilot',runtime)
                self.assertEqual(runtime.calls,3)
                self.assertTrue(all((out/(n+'.jsonl')).exists() for n in JOURNALS))

if __name__=='__main__':unittest.main()
