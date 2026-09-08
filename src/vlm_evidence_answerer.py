"""Geometry evidence -> Qwen answer. CPU prepare/evaluate never load the model."""
from __future__ import annotations
import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path


def rows(path):
    if not Path(path).exists(): return []
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def compact_value(value):
    if not isinstance(value, dict) or value.get('kind') != 'sequence': return value
    return {**{k:v for k,v in value.items() if k not in {'points','x_axis_anchors'}},
            'samples_x_pixel_value': [[p['x_pixel'], round(p['value'],6)] for p in value['points']],
            'x_axis_anchors': [{'label':a['label'],'center_x':a['center_x']} for a in value['x_axis_anchors']]}


def evidence(source, measurement):
    plan = measurement.get('semantic_plan', {})
    targets = plan.get('targets', [])
    values = measurement.get('geometry_values', [])
    ticks = measurement.get('y_axis_ticks', [])
    return {
        'measurements': [dict(target=t, value=compact_value(v), units='same numeric scale as the supplied y-axis ticks')
                         for t, v in zip(targets, values)],
        'axis_tick_texts_and_values': [{'text': t['text'], 'value': t['value']} for t in ticks],
        'operation': plan.get('operation'),
        'python_result': measurement.get('numeric_answer'),
        'calculations': measurement.get('reasoning', {}).get('calculations', []),
        'unit_source': 'Use the axis title, caption and question; never interpret a currency code prefix as a scale suffix.',
    }


def build_prompt(source, measurement):
    kind = source['question_type']
    answer_rule = {
        'single_choice': 'This is SINGLE CHOICE. Return answer as exactly one provided letter, e.g. "A". Do not return an array.',
        'multiple_choice': 'This is MULTIPLE CHOICE. Evaluate EVERY option and return all correct letters as a JSON array, e.g. ["A","C"]. Do not stop at one answer.',
        'numerical': 'This is a NUMERICAL question. Return answer as a JSON number in the requested units, without a sentence or option letters.',
    }[kind]
    example = {'single_choice':'A','multiple_choice':['A','C'],'numerical':1.23}[kind]
    response_schema = json.dumps(dict(status='answered|insufficient_evidence', answer=example, evidence_used='brief measurement and unit explanation'))
    return '''Answer this financial chart question using the supplied geometry measurements.
The numbers below were measured from chart pixels and axis ticks. They are estimates, not ground truth.
Use them as the numerical evidence; do not replace them with a fresh visual guess.
Python has performed the listed arithmetic when a result is present. operation=evidence supplies multiple measurements or sequences for your comparison. Interpret the chart's unit/title and the answer options semantically.
For direct/difference, the result is in axis tick units. For growth_rate, the result is percentage change, already multiplied by 100.
Do not multiply units twice. MYR is a currency code, not M (million). An axis labelled in billions with ticks 60,70,80 has a reading near 70 in billions.
If the requested answer cannot follow from these measurements, return status=insufficient_evidence.
Return exactly one JSON object using this format (the answer below is only an example): ''' + response_schema + '''.
For sequences, use their complete pixel samples and x-axis anchors to evaluate intervals/trends; inspect coverage before making claims about the entire interval. Pixel samples are not uniformly spaced observations in time.

Question type: ''' + kind + '\n' + answer_rule + '\nCaption: ' + source.get('caption', '') + '\nQuestion: ' + source['question'] + '\nOptions:\n' + source.get('options', '') + '\nGeometry evidence:\n' + json.dumps(evidence(source, measurement), ensure_ascii=False)


def parse_reply(raw, source):
    try:
        obj = json.loads(raw[raw.index('{'):raw.rindex('}')+1])
        if obj.get('status') != 'answered': return None, 'insufficient_evidence', obj
        ans = obj['answer']
        if source['question_type'] in {'single_choice', 'multiple_choice'}:
            allowed = re.findall(r'^\s*([A-Z])\s*[:.)-]', source.get('options',''), re.M)
            if source['question_type'] == 'multiple_choice':
                if not isinstance(ans, list) or not ans: raise ValueError('expected_choice_array')
                letters = [str(v).strip().upper() for v in ans]
                if any(v not in allowed for v in letters) or len(set(letters)) != len(letters):
                    raise ValueError('invalid_choices')
                ans = ''.join(sorted(letters))
            else:
                ans = str(ans).strip().upper()
                if ans not in allowed: raise ValueError('invalid_choice')
        else:
            if isinstance(ans, bool): raise ValueError('invalid_number')
            ans = float(str(ans).strip().replace(',', ''))
            if not math.isfinite(ans): raise ValueError('nonfinite_number')
        return ans, 'success', obj
    except (ValueError, KeyError, TypeError, AttributeError):
        return None, 'answer_parse_failed', {}


def prepare(args):
    sources = {r['sample_id']:r for r in rows(args.inputs)}
    measurements = {r['sample_id']:r for r in rows(args.measurements)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    with args.output.open('x') as f:
        for sid, source in sources.items():
            m = measurements.get(sid, {})
            if m.get('status') != 'success' or m.get('semantic_plan',{}).get('has_direct_value_labels') is not False:
                continue
            rec = {k:source[k] for k in ['sample_id','chart_id','image_path','question','question_type','options','caption']}
            rec.update(prompt=build_prompt(source,m), evidence=evidence(source,m))
            f.write(json.dumps(rec,ensure_ascii=False)+'\n'); accepted += 1
    print(json.dumps({'prepared':accepted,'total_questions':len(sources)}),flush=True)


def foreign_gpu_processes():
    r = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if r.returncode: raise RuntimeError('nvidia-smi failed')
    return [v.strip() for v in r.stdout.splitlines() if v.strip() and v.strip()!=str(os.getpid())]


def run(args):
    inputs=rows(args.inputs); done={r['sample_id'] for r in rows(args.output)}
    pending=[r for r in inputs if r['sample_id'] not in done]
    if args.limit: pending=pending[:args.limit]
    priority={'43':0,'18':1,'124':2,'11095':3,'7738':4}
    pending.sort(key=lambda r:priority.get(r['sample_id'],100))
    if not pending:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.touch(exist_ok=True)
        print('All evidence answers already complete.',flush=True);return
    if foreign_gpu_processes():
        print('GPU occupied; no model loaded. Run this command again when idle.',flush=True)
        raise SystemExit(75)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True,min_pixels=200704,max_pixels=451584)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('a') as f:
        for index,row in enumerate(pending,1):
            if foreign_gpu_processes():
                print('Another GPU process appeared; releasing our model.',flush=True);raise SystemExit(75)
            messages=[{'role':'system','content':'Use the measured evidence to answer. Output JSON only.'},
                      {'role':'user','content':[{'type':'image','image':Path(row['image_path']).resolve().as_uri(),'min_pixels':200704,'max_pixels':451584},{'type':'text','text':row['prompt']}]}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            images,_=process_vision_info(messages)
            encoded=processor(text=[text],images=images,padding=True,return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                ids=model.generate(**encoded,max_new_tokens=256,do_sample=False)
            raw=processor.batch_decode(ids[:,encoded['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
            answer,status,parsed=parse_reply(raw,row)
            record=dict(sample_id=row['sample_id'],chart_id=row['chart_id'],status=status,prediction=answer,
                        raw_reply=raw,parsed_reply=parsed,prompt=row['prompt'],messages=messages,model=str(args.model),
                        generation={'do_sample':False,'max_new_tokens':256},evidence=row['evidence'])
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
            print(json.dumps({'processed':index,'pending':len(pending),'sample_id':row['sample_id'],'status':status}),flush=True)


def evaluate(args):
    from finmme_geometry_qa import correctness
    inputs=rows(args.inputs); gold={r['sample_id']:r for r in rows(args.gold)}
    raw={r['sample_id']:r for r in rows(args.baseline)}; replies={r['sample_id']:r for r in rows(args.replies)}
    expected={r['sample_id'] for r in rows(args.prompts)}
    records=[]
    for src in inputs:
        sid=src['sample_id'];r=replies.get(sid,{})
        success=r.get('status')=='success'
        ok=int(success and correctness(src['question_type'],r.get('prediction'),gold[sid]['reference'],gold[sid]['tolerance']))
        base=int(raw[sid]['raw_vlm_correct'])
        records.append(dict(sample_id=sid,question_type=src['question_type'],status=r.get('status','pending' if sid in expected else 'raw_route'),raw_correct=base,hybrid_correct=ok,selective_correct=ok if success else base))
    n=len(records);base=sum(r['raw_correct'] for r in records);hard=sum(r['hybrid_correct'] for r in records);sel=sum(r['selective_correct'] for r in records)
    all_raw=rows(args.raw_metrics);total=len(all_raw);total_correct=sum(bool(r['official_code_correct']) for r in all_raw)
    raw_by_id={r['sample_id']:r for r in all_raw}
    if any(int(bool(raw_by_id[r['sample_id']]['official_code_correct'])) != raw[r['sample_id']]['raw_vlm_correct'] for r in inputs):
        raise ValueError('Full Raw cache does not match the frozen cohort baseline; refusing a mixed-baseline score')
    pending=len(expected-set(replies))
    summary=dict(evaluation_status='partial' if pending else 'complete',questions=n,prompts=len(expected),replies=len(replies),pending=pending,
                 raw_correct=base,hard_correct=hard,selective_correct=sel,raw_accuracy=base/n,hard_accuracy=hard/n,selective_accuracy=sel/n,
                 full_questions=total,full_raw_accuracy=total_correct/total,full_selective_accuracy=(total_correct-base+sel)/total,
                 rescued=sum(not r['raw_correct'] and r['selective_correct'] for r in records),harmed=sum(r['raw_correct'] and not r['selective_correct'] for r in records))
    summary['by_question_type'] = {}
    for kind in sorted({r['question_type'] for r in records}):
        subset = [r for r in records if r['question_type'] == kind]
        summary['by_question_type'][kind] = dict(questions=len(subset),
            raw_correct=sum(r['raw_correct'] for r in subset),
            selective_correct=sum(r['selective_correct'] for r in subset),
            answered=sum(r['status']=='success' for r in subset))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_name('per_sample.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--inputs',type=Path,required=True);prep.add_argument('--measurements',type=Path,required=True);prep.add_argument('--output',type=Path,required=True)
    runner=sub.add_parser('run');runner.add_argument('--inputs',type=Path,required=True);runner.add_argument('--model',type=Path,required=True);runner.add_argument('--output',type=Path,required=True);runner.add_argument('--limit',type=int)
    ev=sub.add_parser('evaluate')
    for flag in ['inputs','gold','baseline','replies','prompts','raw-metrics','output']:ev.add_argument('--'+flag,type=Path,required=True)
    args=p.parse_args();{'prepare':prepare,'run':run,'evaluate':evaluate}[args.command](args)

if __name__=='__main__':main()
