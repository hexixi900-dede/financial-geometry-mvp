import json
import unittest
import numpy as np
from phase9_plan_contract import validate_plan, expression
from vlm_semantic_planner import messages, normalize_plan
from vlm_evidence_answerer import parse_reply, build_prompt
from phase8_geometry_adapter import read_line, read_bar_height
from phase8_hybrid_qa import finish_measurement

class UnifiedMeasurements(unittest.TestCase):
    def test_options_reach_visual_planner_and_multiple_choices_parse(self):
        row=dict(image_path='/tmp/chart.png',question='Which claims hold?',question_type='multiple_choice',options='A: first claim\nB: second claim\nC: third claim')
        msg=messages(row,200704,451584)
        self.assertEqual(msg[1]['content'][0]['type'],'image')
        self.assertIn(row['options'],msg[1]['content'][1]['text'])
        self.assertEqual(parse_reply('{"status":"answered","answer":["C","A"]}',row)[:2],('AC','success'))
        self.assertEqual(parse_reply('{"status":"answered","answer":["Z"]}',row)[1],'answer_parse_failed')
        self.assertIn('EVERY',build_prompt(row,{}))

    def test_multi_target_and_arithmetic(self):
        p=normalize_plan(dict(chart_type='bar',y_axis_count=1,has_direct_value_labels=False,operation='evidence',
            targets=[dict(x_label=str(i),series='',region=[i*100,100,i*100+50,400]) for i in range(4)]), 'Which categories are highest?')
        self.assertTrue(p['route_geometry'])
        p['calculations']=[dict(name='difference of totals',expression={'op':'difference','args':[
            {'op':'sum','args':[{'target':0},{'target':1}]},{'op':'sum','args':[{'target':2},{'target':3}]}]})]
        _, value, reasoning=finish_measurement({'evidence_only':True},p,[10,20,3,7])
        self.assertIsNone(value)
        self.assertEqual(reasoning['calculations'][0]['value'],20)
        self.assertEqual(expression({'op':'growth_rate','args':[{'target':0},{'target':1}]},[50,75]),50)
        p['has_direct_value_labels']=True
        self.assertEqual(validate_plan(p)[1],'direct_value_labels_or_unknown')

    def test_pixels_endpoint_and_interval(self):
        image=np.full((300,400,3),255,np.uint8);mask=np.zeros((300,400),np.uint8)
        for x in range(60,341):mask[240-x//2,x]=1
        axis=dict(plot_bbox=[40,20,360,260],slope=-.1,intercept=26,ticks=[])
        chart=dict(axis=axis,x_axis_anchors=[],legend={'entries':[]},single_series_hint=True)
        series=[dict(series_id=0,mask=mask,bbox=[60,70,281,141],median_bgr=[0,0,0],median_hsv=[0,0,0])]
        t=dict(x_label='latest',series='',position='last',measurement='value',region=[800,200,880,350])
        out=read_line(image,chart,t,series)
        self.assertEqual(out['status'],'success')
        self.assertAlmostEqual(out['auto_local_value'],19)
        t.update(measurement='sequence',position='label',region=[150,100,850,900])
        out=read_line(image,chart,t,series)
        self.assertEqual(len(out['auto_local_value']['points']),281)
        self.assertAlmostEqual(out['auto_local_value']['min'],5)
        self.assertAlmostEqual(out['auto_local_value']['max'],19)

    def test_floating_and_stacked_regions(self):
        image=np.full((300,400,3),255,np.uint8)
        image[150:250,80:110]=[150,50,20]
        image[100:150,80:110]=[30,160,40]
        image[80:100,180:210]=[180,20,120]
        axis=dict(plot_bbox=[40,20,360,260],slope=-.1,intercept=26,ticks=[{'pixel_y':260}])
        class NoOCR:
            def readtext(self,*a,**kw):raise AssertionError('Visual regions must not require label OCR')
        def read(region):
            return read_bar_height(NoOCR(),image,{'axis':axis},dict(x_label='target',series='',measurement='height',region=region))
        total=read([190,320,290,850]);segment=read([190,320,290,510]);floating=read([440,250,540,350])
        self.assertEqual(total['status'],'success')
        self.assertAlmostEqual(total['value'],14.9,places=1)
        self.assertAlmostEqual(segment['value'],4.9,places=1)
        self.assertAlmostEqual(floating['value'],1.9,places=1)

    def test_tight_bar_endpoint_and_negative_bar(self):
        image=np.full((300,400,3),255,np.uint8)
        image[60:240,80:110]=[140,30,20]
        image[150:230,180:210]=[20,140,30]
        axis=dict(plot_bbox=[40,20,360,260],slope=-.1,intercept=15,ticks=[{'pixel_y':260}])
        pos=read_bar_height(None,image,{'axis':axis},dict(x_label='',measurement='value',region=[195,180,280,230]))
        neg=read_bar_height(None,image,{'axis':axis},dict(x_label='',measurement='value',region=[445,730,530,780]))
        self.assertEqual(pos['status'],'success')
        self.assertAlmostEqual(pos['value'],9)
        self.assertAlmostEqual(neg['value'],-7.9)

    def test_resume_never_overwrites_full_and_recovers_partial_tail(self):
        import tempfile
        from pathlib import Path
        from run_unified_resident import append,merge_progress,load_log
        with tempfile.TemporaryDirectory() as d:
            pilot=Path(d)/'pilot.jsonl';full=Path(d)/'full.jsonl'
            append(pilot,{'sample_id':'1','value':'old'})
            append(full,{'sample_id':'1','value':'new'})
            append(full,{'sample_id':'2','value':'full only'})
            with full.open('ab') as f:f.write(b'{"sample_id":')
            merge_progress(pilot,full)
            self.assertEqual(load_log(full)['1']['value'],'new')
            self.assertEqual(len(load_log(full)),2)

    def test_resident_pipeline_continues_full_then_resumes_without_reinference(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        from run_unified_resident import run_cohort,merge_progress,write_rows,rows,append
        class FakeRuntime:
            calls=0
            def generate(self,msg,tokens):
                self.calls+=1
                if tokens==1536:
                    return json.dumps(dict(chart_type='bar',y_axis_count=1,has_direct_value_labels=False,
                        operation='direct',targets=[dict(x_label='a',series='',region=[100,100,200,200])]))
                return '{"status":"answered","answer":["A","C"]}'
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            src=lambda sid:dict(sample_id=sid,chart_id=sid,chart_kind_hint='bar',prefilter={},image_path='/tmp/chart.png',
                                question='Which claims?',question_type='multiple_choice',options='A: one\nB: two\nC: three',caption='')
            for cohort,items in [('pilot',[src('1')]),('full',[src('1'),src('2')])]:
                (root/cohort).mkdir();write_rows(root/cohort/'geometry_inputs.jsonl',items)
            args=SimpleNamespace(root=root,batch_size=128,ocr_python='fake',ocr_models='fake',raw_metrics='fake',model='fake')
            def cpu_step(cmd,**kw):
                if 'run-geometry' in cmd:
                    for r in rows(Path(cmd[cmd.index('--inputs')+1])):
                        plan=rows(Path(cmd[cmd.index('--plans')+1]))
                        plan=next(p['plan'] for p in plan if p['sample_id']==r['sample_id'])
                        append(Path(cmd[cmd.index('--output')+1]),dict(sample_id=r['sample_id'],status='success',semantic_plan=plan,geometry_values=[10],numeric_answer=10))
                return SimpleNamespace(returncode=0)
            runtime=FakeRuntime()
            with patch('run_unified_resident.subprocess.run',side_effect=cpu_step):
                run_cohort(args,'pilot',runtime)
                for name in ('plans','measurements','replies'):merge_progress(root/'pilot'/(name+'.jsonl'),root/'full'/(name+'.jsonl'))
                run_cohort(args,'full',runtime)
                self.assertEqual(runtime.calls,4)
                self.assertEqual(len(rows(root/'full'/'replies.jsonl')),2)
                run_cohort(args,'pilot',runtime);run_cohort(args,'full',runtime)
                self.assertEqual(runtime.calls,4)

if __name__=='__main__':unittest.main()
