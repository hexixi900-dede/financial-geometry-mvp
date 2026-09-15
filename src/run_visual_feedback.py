"""ChartAgent-inspired visual evidence review, with at most one plan correction.

Initial and revised artifacts are separate. measurements/replies contain only the
terminal choice for each question, so the full-population scorer can preserve Raw only for initial direct routes.
"""
import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path

from run_unified_resident import Runtime, append, invoke, load_log, merge_progress
from run_unified_resident import write_rows
from vlm_semantic_planner import messages, normalize_plan, extract_json
from vlm_evidence_answerer import rows, parse_reply, evidence
from geometry_visual_feedback import render_measurement_overlay, build_feedback_messages

METHOD_VERSION='visual_feedback_v4'
JOURNALS=('plans','measurements_initial','feedback_initial','revision_plans',
          'measurements_revision','feedback_final','prompts','replies','measurements')


def feedback_eligible(plan_record, measurement):
    plan=measurement.get('semantic_plan') or plan_record.get('plan') or {}
    # Do not override the intended direct-label or unsupported-chart Raw routes.
    if plan.get('has_direct_value_labels') is True:return False
    if plan and (plan.get('chart_type') not in {'bar','line'} or plan.get('y_axis_count') not in {1,'unknown',None}):return False
    if plan.get('operation')=='unsupported':return False
    return True


def measure(args, out, batch, plan_path, output_name):
    completed=load_log(out/(output_name+'.jsonl'))
    todo=[r for r in batch if str(r['sample_id']) not in completed]
    if todo:
        inputs=out/(output_name+'_inputs.jsonl');write_rows(inputs,todo)
        subprocess.run([args.ocr_python,'-u','src/phase8_hybrid_qa.py','run-geometry',
            '--inputs',str(inputs),'--plans',str(plan_path),
            '--cached-charts',str(args.root/'full/chart_geometry_inputs.jsonl'),
            '--model-dir',args.ocr_models,'--output',str(out/(output_name+'.jsonl')),
            '--repair-geometry','--evidence-only','--threads','4'],
            env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
    return load_log(out/(output_name+'.jsonl'))


def review(runtime,args,source,measurement,out,attempt):
    overlay=out/'overlays'/(str(source['sample_id'])+f'_{attempt}.png')
    render_measurement_overlay(source,measurement,overlay)
    msg=build_feedback_messages(source,measurement,str(overlay),args.min_pixels,args.max_pixels,
                                allow_revision=attempt==0,no_raw_fallback=True)
    tokens=3072 if attempt==0 else 512
    raw,error=invoke(runtime,msg,tokens)
    answer,status,parsed=(None,'inference_error',{}) if error else parse_reply(raw,source)
    text='\n'.join(part['text'] for m in msg if isinstance(m.get('content'),list)
                    for part in m['content'] if part.get('type')=='text')
    return dict(sample_id=str(source['sample_id']),chart_id=source['chart_id'],status=status,
        prediction=answer,raw_reply=raw,parsed_reply=parsed,prompt=text,messages=msg,
        evidence=evidence(source,measurement,no_raw_fallback=True),overlay_path=str(overlay),attempt=attempt,
        model=str(args.model),generation=dict(do_sample=False,max_new_tokens=tokens,
        min_pixels=args.min_pixels,max_pixels=args.max_pixels),method_version=METHOD_VERSION,
        input_protocol=source.get('input_protocol'),vlm_image_sha256=source.get('vlm_image_sha256'),error=error)


def final_reply(reply, measurement, revised=False, no_raw_fallback=False):
    result=dict(reply)
    if result.get('status')=='revision_requested':
        result.update(status='revision_limit_reached',prediction=None)
    if not no_raw_fallback and result.get('status')=='success' and measurement.get('status')!='success':
        result.update(status='insufficient_evidence',prediction=None,
                      error='No successful measurements support this answer')
    result['plan_revised']=revised
    return result


def run_cohort(args,cohort,runtime):
    out=args.root/cohort
    sources=rows(out/'geometry_inputs.jsonl')
    logs={name:load_log(out/(name+'.jsonl')) for name in JOURNALS}
    pending=[src for src in sources if str(src['sample_id']) not in logs['measurements']]
    def save(name,record):
        sid=str(record['sample_id'])
        if sid not in logs[name]:
            append(out/(name+'.jsonl'),record);logs[name][sid]=record
    for start in range(0,len(pending),args.batch_size):
        batch=pending[start:start+args.batch_size]
        (args.root/'status.txt').write_text(f'{cohort}: semantic planning batch {start//args.batch_size+1}\n')
        for src in batch:
            sid=str(src['sample_id'])
            if sid in logs['plans']:continue
            msg=messages(src,args.min_pixels,args.max_pixels)
            raw,error=invoke(runtime,msg,3072)
            plan={};status='inference_error' if error else 'success'
            if not error:
                try:plan=normalize_plan(extract_json(raw),src['question'])
                except (ValueError,TypeError,AttributeError,RecursionError) as exc:
                    status='plan_parse_failed';error=str(exc)
            save('plans',dict(sample_id=sid,chart_id=src['chart_id'],status=status,plan=plan,
                error=error,raw_text=raw,messages=msg,planner_version=METHOD_VERSION,
                input_protocol=src.get('input_protocol'),vlm_image_sha256=src.get('vlm_image_sha256')))
            print(json.dumps(dict(stage='plan',cohort=cohort,sample_id=sid,status=status)),flush=True)
        (args.root/'status.txt').write_text(f'{cohort}: initial CPU geometry\n')
        logs['measurements_initial']=measure(args,out,batch,out/'plans.jsonl','measurements_initial')
        (args.root/'status.txt').write_text(f'{cohort}: review image and measured evidence\n')
        revision_batch=[]
        for src in batch:
            sid=str(src['sample_id']);m=logs['measurements_initial'][sid];p=logs['plans'][sid]
            if not feedback_eligible(p,m):continue
            if sid not in logs['feedback_initial']:
                audited={**m,'semantic_plan':m.get('semantic_plan') or p.get('plan',{}),
                         'planner_error':p.get('error')}
                if p.get('status')!='success':audited['planner_raw_text']=p.get('raw_text','')
                save('feedback_initial',review(runtime,args,src,audited,out,0))
            first=logs['feedback_initial'][sid]
            if first['status']=='revision_requested':
                if sid not in logs['revision_plans']:
                    error=None;plan={};status='success'
                    try:plan=normalize_plan(first['parsed_reply']['plan'],src['question'])
                    except (KeyError,ValueError,TypeError,AttributeError,RecursionError) as exc:
                        status='plan_parse_failed';error=str(exc)
                    save('revision_plans',dict(sample_id=sid,chart_id=src['chart_id'],plan=plan,
                        status=status,error=error,raw_text=first['raw_reply'],planner_version=METHOD_VERSION,
                        revision_source='visual_feedback'))
                revision_batch.append(src)
        if revision_batch:
            (args.root/'status.txt').write_text(f'{cohort}: one corrected CPU measurement pass; model retained\n')
            logs['measurements_revision']=measure(args,out,revision_batch,out/'revision_plans.jsonl','measurements_revision')
        (args.root/'status.txt').write_text(f'{cohort}: final evidence answers\n')
        for src in batch:
            sid=str(src['sample_id']);m=logs['measurements_initial'][sid]
            first=logs['feedback_initial'].get(sid);reply=None;revised=False
            if first:
                if first['status']=='revision_requested':
                    revised=True;m=logs['measurements_revision'][sid]
                # Once entered, always finish with this workflow's VLM answer.
                # Failed/partial measurements remain visible; they never select
                # a cached Raw answer. One revision is the measurement limit.
                if revised or first['status']!='success':
                    if sid not in logs['feedback_final']:
                        save('feedback_final',review(runtime,args,src,m,out,1))
                    reply=final_reply(logs['feedback_final'][sid],m,revised,no_raw_fallback=True)
                else:reply=final_reply(first,m,no_raw_fallback=True)
                reply['geometry_entered']=True
            if reply:
                save('prompts',{**src,'prompt':reply['prompt'],'messages':reply['messages'],
                                'overlay_path':reply['overlay_path'],'attempt':reply['attempt']})
                save('replies',reply)
                print(json.dumps(dict(stage='answer',cohort=cohort,sample_id=sid,
                                     status=reply['status'],revised=revised)),flush=True)
            # Commit terminal measurement last, after the model logs. A restart can
            # replay bookkeeping without another model call or losing the raw route.
            save('measurements',{**m,'method_version':METHOD_VERSION,'plan_revised':revised,
                                 'feedback_status':reply['status'] if reply else 'raw_route',
                                 'geometry_entered':first is not None})
        evaluate(args,out)
    for name in JOURNALS:(out/(name+'.jsonl')).touch(exist_ok=True)
    evaluate(args,out)


def evaluate(args,out):
    for name in ('measurements','prompts','replies'):(out/(name+'.jsonl')).touch(exist_ok=True)
    cmd=['python3','src/vlm_evidence_answerer.py','evaluate',
         '--inputs',str(out/'geometry_inputs.jsonl'),'--gold',str(out/'gold.jsonl'),
         '--baseline',str(out/'baseline.jsonl'),'--replies',str(out/'replies.jsonl'),
         '--prompts',str(out/'prompts.jsonl'),'--measurements',str(out/'measurements.jsonl'),
         '--raw-metrics',args.raw_metrics,'--raw-predictions',args.raw_predictions,
         '--output',str(out/'summary.json'),'--no-raw-fallback']
    # The summary file is the UI; avoid appending a full 11k-score report after each batch.
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model',default='/data/liu_jun/blind_cssa/models/Qwen2.5-VL-7B-Instruct')
    p.add_argument('--ocr-python',default='/data/liu_jun/finmme_reproduction/phase6_chart_structure_transfer_rev2_screen_v1/.venv_ocr/bin/python')
    p.add_argument('--ocr-models',default='/data/liu_jun/finmme_reproduction/phase6_chart_structure_transfer_rev2_screen_v1/.ocr_models')
    p.add_argument('--raw-metrics',default='/data/liu_jun/finmme_reproduction/outputs/metrics/qwen25vl7b_direct_full_per_sample.jsonl')
    p.add_argument('--raw-predictions',default='/data/liu_jun/finmme_reproduction/outputs/predictions/qwen25vl7b_direct_full.jsonl')
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--min-pixels',type=int,default=200704)
    p.add_argument('--max-pixels',type=int,default=802816)
    p.add_argument('--gpu-poll-seconds',type=float,default=10)
    args=p.parse_args()
    protocol=json.loads((args.root/'input_protocol.json').read_text())
    if protocol.get('post_geometry_raw_fallback') is not False:
        raise ValueError('This version requires the no-post-Geometry-fallback protocol')
    if protocol.get('method_version')!=METHOD_VERSION:
        raise ValueError('Prepare a separate visual-feedback run; never mix old and new predictions')
    if (args.min_pixels,args.max_pixels)!=(protocol['min_pixels'],protocol['max_pixels']):
        raise ValueError('VLM image settings must match the frozen Raw protocol')
    if args.batch_size<1 or args.gpu_poll_seconds<=0:raise ValueError('Invalid runtime limits')
    with (args.root/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        runtime=Runtime(args.model,args.root/'status.txt',args.min_pixels,args.max_pixels,args.gpu_poll_seconds)
        run_cohort(args,'pilot',runtime)
        for name in JOURNALS:merge_progress(args.root/'pilot'/(name+'.jsonl'),args.root/'full'/(name+'.jsonl'))
        run_cohort(args,'full',runtime)
        (args.root/'status.txt').write_text('complete\n')


if __name__=='__main__':main()
